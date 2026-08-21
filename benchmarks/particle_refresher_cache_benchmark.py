"""Benchmark distributed particle-refresher construction versus cached lookup.

For a CPU-only run that still exercises particle collectives, use for example:

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 \
        python benchmarks/particle_refresher_cache_benchmark.py
"""

import argparse
import re
import time
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding

from PyPIC3D.boundary_conditions.ghost_cells import MESH_AXES
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_PERIODIC
from PyPIC3D.particles import particle_tile_communication as particle_comm
from PyPIC3D.particles.particle_class import TiledParticles
from PyPIC3D.utilities.grids import build_yee_grid


jax.config.update("jax_enable_x64", True)


class _BenchmarkStaticParameters(NamedTuple):
    tile_shape: tuple
    particle_boundary_conditions: tuple
    field_mesh: object


def _configuration():
    devices = jax.devices()
    if len(devices) < 2:
        raise RuntimeError(
            "This benchmark needs two JAX devices to inspect the particle migration collective; "
            "force two CPU devices with XLA_FLAGS=--xla_force_host_platform_device_count=2."
        )

    mesh_shape = (2, 1, 1)
    tile_shape = (2, 1, 1)
    mesh = Mesh(np.asarray(devices[:2]).reshape(mesh_shape), MESH_AXES)
    static_parameters = _BenchmarkStaticParameters(
        tile_shape=tile_shape,
        particle_boundary_conditions=(BC_PERIODIC, BC_PERIODIC, BC_PERIODIC),
        field_mesh=mesh,
    )

    dynamic_parameters = SimpleNamespace(
        dx=jnp.asarray(1.0),
        dy=jnp.asarray(1.0),
        dz=jnp.asarray(1.0),
        Nx=jnp.asarray(4),
        Ny=jnp.asarray(1),
        Nz=jnp.asarray(1),
        x_wind=jnp.asarray(4.0),
        y_wind=jnp.asarray(1.0),
        z_wind=jnp.asarray(1.0),
    )
    grid_parameters = SimpleNamespace(
        **vars(dynamic_parameters),
        x_min=jnp.asarray(-2.0),
        y_min=jnp.asarray(-0.5),
        z_min=jnp.asarray(-0.5),
    )
    center_grid, vertex_grid = build_yee_grid(grid_parameters)
    dynamic_parameters.grids = SimpleNamespace(center=center_grid, vertex=vertex_grid)

    x = jnp.zeros(mesh_shape + (1, 2, 3), dtype=jnp.float64)
    u = jnp.zeros_like(x)
    active = jnp.zeros(mesh_shape + (1, 2), dtype=bool)
    x = x.at[0, 0, 0, 0, 0].set(jnp.asarray((0.25, 0.0, 0.0)))
    u = u.at[0, 0, 0, 0, 0].set(jnp.asarray((1.0, 0.0, 0.0)))
    active = active.at[0, 0, 0, 0, 0].set(True)
    particles = TiledParticles(
        x=jax.device_put(x, NamedSharding(mesh, particle_comm.PARTICLE_STATE_TILE_SPEC)),
        u=jax.device_put(u, NamedSharding(mesh, particle_comm.PARTICLE_STATE_TILE_SPEC)),
        active=jax.device_put(active, NamedSharding(mesh, particle_comm.PARTICLE_ACTIVE_TILE_SPEC)),
    )
    return static_parameters, dynamic_parameters, particles


def _lowered_hlo(refresher, particles, dynamic_parameters):
    lowered = jax.jit(lambda current: refresher(current, dynamic_parameters)).lower(particles)
    return str(lowered.compiler_ir(dialect="stablehlo"))


def _communication_signature(hlo):
    return {
        "collective_permute": hlo.count("stablehlo.collective_permute"),
        "all_reduce": hlo.count("stablehlo.all_reduce"),
        "packet_shapes": sorted(set(re.findall(r"tensor<[^>]*x7xf(?:32|64)>", hlo))),
    }


def _outputs_equal(first, second):
    return all(
        np.array_equal(np.asarray(first_leaf), np.asarray(second_leaf))
        for first_leaf, second_leaf in zip(
            jax.tree_util.tree_leaves(first),
            jax.tree_util.tree_leaves(second),
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    iterations = max(1000, args.iterations)
    static_parameters, dynamic_parameters, particles = _configuration()

    uncached_refresher = particle_comm._build_distributed_particle_refresher(static_parameters)
    uncached_hlo = _lowered_hlo(uncached_refresher, particles, dynamic_parameters)

    particle_comm._cached_distributed_particle_refresher.cache_clear()
    cached_refresher = particle_comm.make_distributed_particle_refresher(static_parameters)
    cached_hlo = _lowered_hlo(cached_refresher, particles, dynamic_parameters)

    uncached_output = jax.block_until_ready(uncached_refresher(particles, dynamic_parameters))
    cached_output = jax.block_until_ready(cached_refresher(particles, dynamic_parameters))

    start = time.perf_counter()
    for _ in range(iterations):
        particle_comm._build_distributed_particle_refresher(static_parameters)
    uncached_seconds = time.perf_counter() - start

    same_callable_reused = True
    start = time.perf_counter()
    for _ in range(iterations):
        current = particle_comm.make_distributed_particle_refresher(static_parameters)
        same_callable_reused = same_callable_reused and current is cached_refresher
    cached_seconds = time.perf_counter() - start

    uncached_per_lookup = uncached_seconds / iterations
    cached_per_lookup = cached_seconds / iterations
    print(f"iterations: {iterations}")
    print(f"uncached total: {uncached_seconds:.6f} s")
    print(f"uncached per construction: {uncached_per_lookup * 1.0e6:.3f} us")
    print(f"cached total: {cached_seconds:.6f} s")
    print(f"cached per lookup: {cached_per_lookup * 1.0e6:.3f} us")
    print(f"speedup: {uncached_seconds / cached_seconds:.2f}x")
    print(f"same callable reused: {same_callable_reused}")
    print(f"uncached HLO communication signature: {_communication_signature(uncached_hlo)}")
    print(f"cached HLO communication signature: {_communication_signature(cached_hlo)}")
    print(f"HLO communication unchanged: {_communication_signature(uncached_hlo) == _communication_signature(cached_hlo)}")
    print(f"steady-state numerical output unchanged: {_outputs_equal(uncached_output, cached_output)}")


if __name__ == "__main__":
    main()
