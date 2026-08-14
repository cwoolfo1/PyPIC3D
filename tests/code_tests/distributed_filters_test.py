import unittest
from types import SimpleNamespace

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding

from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_PERIODIC
from PyPIC3D.utilities.filters import (
    bilinear_filter,
    digital_filter,
    tiled_bilinear_filter,
    tiled_bilinear_filter_vector,
    tiled_digital_filter,
    tiled_digital_filter_vector,
)


jax.config.update("jax_enable_x64", True)


def _mesh(mesh_shape):
    n_devices = int(np.prod(mesh_shape))
    devices = jax.devices()
    if len(devices) < n_devices:
        raise unittest.SkipTest(f"Need {n_devices} JAX devices, got {len(devices)}")
    return Mesh(np.asarray(devices[:n_devices]).reshape(mesh_shape), ghost_cells.MESH_AXES)


def _static_parameters(mesh_shape, tile_shape, g=2):
    periodic = (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
    return SimpleNamespace(
        tile_shape=tuple(int(width) for width in tile_shape),
        guard_cells=int(g),
        boundary_conditions=periodic,
        particle_boundary_conditions=periodic,
        field_mesh=_mesh(mesh_shape),
    )


def _tile_interior(interior, mesh_shape, tile_shape, g):
    ntx, nty, ntz = mesh_shape
    tile_nx, tile_ny, tile_nz = tile_shape

    interior_tiles = interior.reshape(
        ntx,
        tile_nx,
        nty,
        tile_ny,
        ntz,
        tile_nz,
    ).transpose(0, 2, 4, 1, 3, 5)

    tiles = jnp.zeros(
        (
            ntx,
            nty,
            ntz,
            tile_nx + 2 * g,
            tile_ny + 2 * g,
            tile_nz + 2 * g,
        ),
        dtype=interior.dtype,
    )
    return tiles.at[:, :, :, g:-g, g:-g, g:-g].set(interior_tiles)


def _assemble_interior(field_tiles, g):
    ntx, nty, ntz, local_nx, local_ny, local_nz = field_tiles.shape
    tile_nx = local_nx - 2 * g
    tile_ny = local_ny - 2 * g
    tile_nz = local_nz - 2 * g

    interior_tiles = field_tiles[:, :, :, g:-g, g:-g, g:-g]
    return interior_tiles.transpose(0, 3, 1, 4, 2, 5).reshape(
        ntx * tile_nx,
        nty * tile_ny,
        ntz * tile_nz,
    )


def _periodic_field(interior, g):
    return jnp.pad(interior, ((g, g), (g, g), (g, g)), mode="wrap")


class TestDistributedFilters(unittest.TestCase):
    def assert_allclose(self, actual, expected):
        self.assertTrue(
            jnp.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12),
            msg=f"\n{actual}\n!=\n{expected}",
        )

    def _scalar_filter_case(self, tiled_filter, local_filter, alpha=None):
        mesh_shape = (2, 2, 2)
        tile_shape = (3, 3, 3)
        g = 2
        static_parameters = _static_parameters(mesh_shape, tile_shape, g)

        global_shape = tuple(mesh_size * tile_size for mesh_size, tile_size in zip(mesh_shape, tile_shape))
        values = jnp.arange(np.prod(global_shape), dtype=jnp.float64).reshape(global_shape)
        interior = jnp.sin(values / 11.0) + 0.01 * values

        tiles = _tile_interior(interior, mesh_shape, tile_shape, g)
        sharding = NamedSharding(static_parameters.field_mesh, ghost_cells.SCALAR_TILE_SPEC)
        sharded_tiles = jax.device_put(tiles, sharding)

        if alpha is None:
            actual = tiled_filter(sharded_tiles, static_parameters)
            expected = local_filter(_periodic_field(interior, g), num_guard_cells=g)
        else:
            actual = tiled_filter(sharded_tiles, alpha, static_parameters)
            expected = local_filter(_periodic_field(interior, g), alpha, num_guard_cells=g)

        actual.block_until_ready()
        self.assertEqual(actual.sharding, sharding)
        self.assertEqual(
            {tuple(shard.data.shape) for shard in actual.addressable_shards},
            {(1, 1, 1, 7, 7, 7)},
        )
        self.assert_allclose(_assemble_interior(actual, g), expected[g:-g, g:-g, g:-g])

        # The post-filter refresh must populate the tile interface from the
        # filtered owner interior rather than leaving the input guards stale.
        self.assert_allclose(
            actual[0, 0, 0, -g, g:-g, g:-g],
            actual[1, 0, 0, g, g:-g, g:-g],
        )

    def test_tiled_digital_filter_stays_distributed_and_matches_global_stencil(self):
        self._scalar_filter_case(tiled_digital_filter, digital_filter, alpha=0.6)

    def test_tiled_bilinear_filter_stays_distributed_and_matches_global_stencil(self):
        self._scalar_filter_case(tiled_bilinear_filter, bilinear_filter)

    def test_local_filters_read_guards_without_reading_another_tile_axis(self):
        g = 2
        tiles = jnp.zeros((2, 1, 1, 7, 7, 7), dtype=jnp.float64)
        tiles = tiles.at[0, 0, 0, -g - 1, 3, 3].set(2.0)
        tiles = tiles.at[0, 0, 0, -g, 3, 3].set(12.0)
        changed_remote_interior = tiles.at[1, 0, 0, g, 3, 3].set(1000.0)

        digital = digital_filter(tiles, 0.6, num_guard_cells=g)
        changed_digital = digital_filter(changed_remote_interior, 0.6, num_guard_cells=g)
        bilinear = bilinear_filter(tiles, num_guard_cells=g)
        changed_bilinear = bilinear_filter(changed_remote_interior, num_guard_cells=g)

        self.assertEqual(digital[0, 0, 0, -g - 1, 3, 3], changed_digital[0, 0, 0, -g - 1, 3, 3])
        self.assertEqual(bilinear[0, 0, 0, -g - 1, 3, 3], changed_bilinear[0, 0, 0, -g - 1, 3, 3])
        self.assertNotEqual(digital[0, 0, 0, -g - 1, 3, 3], digital_filter(tiles.at[0, 0, 0, -g, 3, 3].set(0.0), 0.6, num_guard_cells=g)[0, 0, 0, -g - 1, 3, 3])
        self.assertNotEqual(bilinear[0, 0, 0, -g - 1, 3, 3], bilinear_filter(tiles.at[0, 0, 0, -g, 3, 3].set(0.0), num_guard_cells=g)[0, 0, 0, -g - 1, 3, 3])

    def test_tiled_vector_filters_preserve_stacked_and_tuple_layouts(self):
        mesh_shape = (2, 2, 2)
        tile_shape = (3, 3, 3)
        g = 2
        static_parameters = _static_parameters(mesh_shape, tile_shape, g)

        interior = jnp.arange(6 * 6 * 6, dtype=jnp.float64).reshape((6, 6, 6)) / 19.0
        tiles = _tile_interior(interior, mesh_shape, tile_shape, g)
        stacked = jnp.stack((tiles, -2.0 * tiles, tiles + 3.0), axis=0)
        sharding = NamedSharding(static_parameters.field_mesh, ghost_cells.VECTOR_TILE_SPEC)
        sharded_stacked = jax.device_put(stacked, sharding)

        stacked_digital = tiled_digital_filter_vector(sharded_stacked, 0.55, static_parameters)
        tuple_digital = tiled_digital_filter_vector(tuple(sharded_stacked[i] for i in range(3)), 0.55, static_parameters)
        stacked_bilinear = tiled_bilinear_filter_vector(sharded_stacked, static_parameters)
        tuple_bilinear = tiled_bilinear_filter_vector(tuple(sharded_stacked[i] for i in range(3)), static_parameters)

        self.assertEqual(stacked_digital.sharding, sharding)
        self.assertEqual(stacked_bilinear.sharding, sharding)
        self.assertIsInstance(tuple_digital, tuple)
        self.assertIsInstance(tuple_bilinear, tuple)
        for component in range(3):
            self.assert_allclose(tuple_digital[component], stacked_digital[component])
            self.assert_allclose(tuple_bilinear[component], stacked_bilinear[component])

    def test_tiled_filters_preserve_reduced_axis_behavior(self):
        mesh_shape = (2, 1, 1)
        tile_shape = (3, 1, 1)
        g = 2
        static_parameters = _static_parameters(mesh_shape, tile_shape, g)

        interior = jnp.arange(6, dtype=jnp.float64).reshape((6, 1, 1))
        tiles = _tile_interior(interior, mesh_shape, tile_shape, g)
        sharding = NamedSharding(static_parameters.field_mesh, ghost_cells.SCALAR_TILE_SPEC)
        sharded_tiles = jax.device_put(tiles, sharding)

        digital = tiled_digital_filter(sharded_tiles, 0.6, static_parameters)
        bilinear = tiled_bilinear_filter(sharded_tiles, static_parameters)
        expected_digital = digital_filter(_periodic_field(interior, g), 0.6, num_guard_cells=g)
        expected_bilinear = bilinear_filter(_periodic_field(interior, g), num_guard_cells=g)

        self.assert_allclose(_assemble_interior(digital, g), expected_digital[g:-g, g:-g, g:-g])
        self.assert_allclose(_assemble_interior(bilinear, g), expected_bilinear[g:-g, g:-g, g:-g])


if __name__ == "__main__":
    unittest.main()
