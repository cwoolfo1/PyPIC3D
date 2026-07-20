import jax
import jax.numpy as jnp

from PyPIC3D.deposition.GR_direct_deposition import GR_direct_deposition
from PyPIC3D.evolve import time_loop_static_metric
from PyPIC3D.initialization import (
    _encode_current_calculation,
    _validate_tiled_yee_configuration,
    validate_field_solver,
)
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.pusher.hybrid_boris_geodesic import hybrid_boris_geodesic_push
from PyPIC3D.relativity.flat import initialize_flat_cartesian_metric
from tests.kernel_fixtures import active_interior, empty_tiled_vector, kernel_parameters


def _single_particle_state(static_parameters, dynamic_parameters, u):
    x = jnp.zeros((1, 1, 1, 1, 1, 3))
    u = jnp.asarray(u, dtype=float).reshape((1, 1, 1, 1, 1, 3))
    active = jnp.ones((1, 1, 1, 1, 1), dtype=bool)
    particles = TiledParticles(x=x, u=u, active=active)
    species = SpeciesConfig(
        charge=jnp.asarray([1.0]),
        mass=jnp.asarray([1.0]),
        weight=jnp.asarray([1.0]),
        update_x=jnp.asarray([[True, True, True]]),
        update_u=jnp.asarray([[True, True, True]]),
    )
    return particles, species


def test_flat_cartesian_metric_matches_center_grid_shape():
    static_parameters, dynamic_parameters = kernel_parameters(Nx=4, Ny=3, Nz=2)

    metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
    g = int(static_parameters.guard_cells)
    shape = (1, 1, 1, 4 + 2 * g, 3 + 2 * g, 2 + 2 * g)

    assert metric.center.lapse.shape == shape
    assert metric.center.gamma_inv.shape == shape + (3, 3)
    assert jnp.allclose(metric.center.lapse, 1.0)
    assert jnp.allclose(metric.center.gamma_inv[..., 0, 0], 1.0)
    assert jnp.allclose(metric.center.sqrt_gamma, 1.0)


def test_hybrid_boris_geodesic_push_advances_flat_neutral_particle_with_u_over_gamma():
    static_parameters, dynamic_parameters = kernel_parameters(
        Nx=4,
        Ny=4,
        Nz=4,
        x_wind=4.0,
        y_wind=4.0,
        z_wind=4.0,
        dt=0.2,
    )
    metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
    particles, species = _single_particle_state(static_parameters, dynamic_parameters, (0.3, 0.4, 0.0))
    species = species._replace(charge=jnp.asarray([0.0]))
    zeros = empty_tiled_vector(static_parameters, dynamic_parameters)

    pushed, centered = hybrid_boris_geodesic_push(
        particles,
        species,
        zeros,
        zeros,
        metric,
        static_parameters,
        dynamic_parameters,
    )

    gamma = jnp.sqrt(1.0 + 0.3**2 + 0.4**2)
    expected_x = jnp.asarray((0.3, 0.4, 0.0)) * dynamic_parameters.dt / gamma
    assert jnp.allclose(pushed.x[0, 0, 0, 0, 0], expected_x)
    assert jnp.allclose(centered.x[0, 0, 0, 0, 0], 0.5 * expected_x)
    assert jnp.allclose(pushed.u[0, 0, 0, 0, 0], jnp.asarray((0.3, 0.4, 0.0)))


def test_GR_direct_deposition_uses_lapse_scaled_contravariant_three_velocity():
    static_parameters, dynamic_parameters = kernel_parameters(
        Nx=1,
        Ny=1,
        Nz=1,
        x_wind=1.0,
        y_wind=1.0,
        z_wind=1.0,
        dt=0.1,
        tile_shape=(1, 1, 1),
        shape_factor=1,
        current_filter="none",
    )
    metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
    metric = metric._replace(
        center=metric.center._replace(lapse=jnp.full_like(metric.center.lapse, 2.0))
    )
    particles, species = _single_particle_state(static_parameters, dynamic_parameters, (0.5, 0.0, 0.0))
    J = empty_tiled_vector(static_parameters, dynamic_parameters)

    J = GR_direct_deposition(
        particles,
        species,
        J,
        metric,
        static_parameters,
        dynamic_parameters,
    )

    interior = active_interior(static_parameters, dynamic_parameters)
    gamma = jnp.sqrt(1.0 + 0.5**2)
    expected = 2.0 * 0.5 / gamma
    assert jnp.allclose(J[0][interior], expected)
    assert jnp.allclose(J[1][interior], 0.0)
    assert jnp.allclose(J[2][interior], 0.0)


def test_static_metric_time_loop_keeps_metric_state_tail():
    static_parameters, dynamic_parameters = kernel_parameters(
        Nx=4,
        Ny=4,
        Nz=4,
        x_wind=4.0,
        y_wind=4.0,
        z_wind=4.0,
        dt=0.05,
        solver="static_metric",
        current_deposition="GR_direct",
        particle_pusher="hybrid_boris_geodesic",
    )
    metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
    particles, species = _single_particle_state(static_parameters, dynamic_parameters, (0.2, 0.0, 0.0))
    D = empty_tiled_vector(static_parameters, dynamic_parameters)
    B = empty_tiled_vector(static_parameters, dynamic_parameters)
    J = empty_tiled_vector(static_parameters, dynamic_parameters)
    rho = jnp.zeros_like(J[0])
    phi = jnp.zeros_like(J[0])
    external_fields = (D, B)
    static_metric_state = (B, B, D, D)
    fields = (D, B, J, rho, phi, external_fields, metric, static_metric_state, jnp.asarray(False))

    particles, fields = time_loop_static_metric(
        particles,
        species,
        fields,
        static_parameters,
        dynamic_parameters,
    )

    assert len(fields) == 9
    assert fields[6] is metric
    assert len(fields[7]) == 4
    assert bool(jax.device_get(fields[-1])) is False
    assert jnp.all(jnp.isfinite(particles.x))
    assert jnp.all(jnp.isfinite(particles.u))


def test_static_metric_dispatch_contract_accepts_hybrid_gr_direct_path():
    static_parameters, dynamic_parameters = kernel_parameters(
        solver="static_metric",
        current_deposition="GR_direct",
        particle_pusher="hybrid_boris_geodesic",
    )
    static_config = {
        "solver": "static_metric",
        "current_calculation": "GR_direct_deposition",
        "particle_pusher": "hybrid_boris_geodesic",
        "filter_j": "none",
        "particle_tile_nx": static_parameters.tile_shape[0],
        "particle_tile_ny": static_parameters.tile_shape[1],
        "particle_tile_nz": static_parameters.tile_shape[2],
    }
    dynamic_config = {
        "Nx": int(dynamic_parameters.Nx),
        "Ny": int(dynamic_parameters.Ny),
        "Nz": int(dynamic_parameters.Nz),
    }

    validate_field_solver("static_metric")

    assert _encode_current_calculation("GR_direct_deposition") == "GR_direct"
    _validate_tiled_yee_configuration(static_config, dynamic_config)
