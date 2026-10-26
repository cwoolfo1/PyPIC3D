"""Single-puncture specialization of arXiv:1311.6483v2, Eqs. (2.5)--(2.13).

Gaussian units: G=c=1, rho_EM=(E^2+B^2)/(8*pi), flux(D)=4*pi*Q.
The regular correction u is exactly one; no elliptic solve is needed.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.absorbing_sponge import make_reference_sponge
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.relativity.bssn_parameters import build_bssn_parameters
from PyPIC3D.relativity.core import D_FIELD_LOCATIONS
from PyPIC3D.relativity.dynamical_relativity import build_bssn_metric, initialize_dynamical_relativity
from PyPIC3D.utilities.grids import build_yee_grid, build_tiled_yee_grids
from PyPIC3D.utilities.parameters import build_dynamic_parameters, build_static_parameters


MASS = 1.0
CHARGE = 0.5
NUM_POINTS = 96
HALF_WIDTH = 12.0
CFL = 0.2
FINAL_TIME = 200.0
KO_DISSIPATION = 0.02
MOMENTUM_DAMPING = 0.002


def native_coordinates(dynamic, location):
    """Broadcast the three native tiled axes, including exterior samples."""
    axes = [dynamic.grids.tiled_center_grid[i] if loc == "C"
            else dynamic.grids.tiled_vertex_grid[i] for i, loc in enumerate(location)]
    return jnp.broadcast_arrays(axes[0][..., :, None, None],
                                axes[1][..., None, :, None], axes[2][..., None, None, :])


def build_initial_data(n=NUM_POINTS, *, mass=MASS, charge=CHARGE, half_width=HALF_WIDTH,
                       cfl=CFL, nu=KO_DISSIPATION, kappa=MOMENTUM_DAMPING,
                       output_dir=".", dt=None):
    """Return static, dynamic, empty particles/species, and seeded field state."""
    if n < 12 or n % 2:
        raise ValueError("Use an even grid size >=12 to avoid the puncture at C/D/B nodes")
    if mass <= 0 or abs(charge) >= mass:
        raise ValueError("A subextremal puncture requires M>0 and |Q|<M")
    if half_width <= 3.0 or cfl <= 0:
        raise ValueError("The domain must contain the 3M sponge and dt must be positive")
    if not jax.config.x64_enabled:
        raise ValueError("Charged puncture initialization requires JAX float64")

    dx = 2 * half_width / n
    # PyPIC3D x_min denotes the first C node, not the cell's left face.
    origin = -half_width + dx / 2
    config = dict(Nx=n, Ny=n, Nz=n, dx=dx, dy=dx, dz=dx, dt=cfl*dx if dt is None else dt,
                  x_wind=2*half_width, y_wind=2*half_width, z_wind=2*half_width,
                  x_min=origin, y_min=origin, z_min=origin, C=1., eps=1., mu=1., kb=1.)
    static = build_static_parameters(dict(
        **config, name="charged_vacuum_puncture", output_dir=str(output_dir),
        solver="dynamic_metric", metric="numerical", shape_factor=2, guard_cells=3,
        tile_shape=(n, n, n), particle_batch_size=1, particle_pusher="hybrid_boris_geodesic",
        current_deposition="esirkepov", current_filter="none",
        boundary_conditions=(3, 3, 3), particle_boundary_conditions=(0, 0, 0)))
    center, vertex = build_yee_grid(SimpleNamespace(**config))
    tiled_center, tiled_vertex = build_tiled_yee_grids(
        static, SimpleNamespace(**config, grids=SimpleNamespace(center=center, vertex=vertex)))
    config["grids"] = dict(center=center, vertex=vertex, tiled_center_grid=tiled_center,
                           tiled_vertex_grid=tiled_vertex)
    dynamic = build_dynamic_parameters(config)
    bssn = build_bssn_parameters(static, dynamic, gravity_G=1., evolution_parameters=dict(
        gauge=1, zero_shift=0, eta=2., g=.75, kappa=kappa, nu=nu,
        xl_bc=1, xr_bc=1, yl_bc=1, yr_bc=1, zl_bc=1, zr_bc=1))

    x, y, z = jnp.meshgrid(*(axis[1:-1] for axis in center), indexing="ij")
    r = jnp.sqrt(x*x + y*y + z*z)
    psi2 = (1 + mass/(2*r))**2 - (charge/(2*r))**2
    state = bssn.state._replace(conformal_factor=1/psi2, lapse=1/psi2)
    dynamic = dynamic._replace(bssn_parameters=bssn._replace(state=state))
    metric = build_bssn_metric(state, static, dynamic)

    density = []
    for i, location in enumerate(D_FIELD_LOCATIONS):
        xyz = native_coordinates(dynamic, location)
        r2 = sum(q*q for q in xyz)
        density.append(charge * xyz[i] / r2**1.5)
    density = tuple(density)
    zero = tuple(jnp.zeros_like(q) for q in density)
    absorber = make_reference_sponge(density, zero, static, dynamic,
                                    width=3., order=4, target_attenuation=1.e-3)
    # Initialization densitizes using the interpolated Yee geometry. Dividing
    # by that SAME volume gives exactly the desired native Coulomb density.
    physical = tuple(q / sample.sqrt_gamma for q, sample in zip(density, metric.D))
    x = jnp.zeros((1, 1, 1, 0, 0, 3), dtype=jnp.float64)
    particles = TiledParticles(x, x, jnp.zeros(x.shape[:-1], dtype=bool))
    v = jnp.zeros((0,), dtype=jnp.float64)
    species = SpeciesConfig(v, v, v, jnp.zeros((0, 3), dtype=bool))
    fields = (physical, zero, zero, zero[0], zero[0], (zero, zero), metric,
              (physical, zero), jnp.asarray(False))
    particles, fields, dynamic = initialize_dynamical_relativity(
        particles, species, fields, static, dynamic, absorber=absorber)
    return static, dynamic, particles, species, fields
