import unittest

import jax
import jax.numpy as jnp
import numpy as np

from PyPIC3D.deposition.GR_direct_deposition import GR_direct_deposition
from PyPIC3D.deposition.J_from_rhov import J_from_rhov
from PyPIC3D.solvers.gr_static.time_loop import time_loop_static_metric
from PyPIC3D.initialization import (
    _encode_current_calculation,
    _validate_tiled_yee_configuration,
    validate_field_solver,
)
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.particles.particle_tile_communication import shard_tiled_particles
import PyPIC3D.pusher.hybrid_boris_geodesic as hybrid_pusher
from PyPIC3D.pusher.hybrid_boris_geodesic import (
    GR_position_update,
    _magnetic_boris_rotation,
    geodesic_velocity,
    hybrid_boris_geodesic_push,
)
from PyPIC3D.relativity.core import (
    B_FIELD_LOCATIONS,
    D_FIELD_LOCATIONS,
    Metric,
    metric_for_location,
)
from PyPIC3D.relativity.flat import (
    initialize_flat_cartesian_metric,
    initialize_flat_cylindrical_metric,
    initialize_flat_spherical_metric,
)
from PyPIC3D.relativity.kerr_schild import (
    initialize_kerr_schild_cartesian_metric,
    initialize_kerr_schild_spherical_metric,
)
from PyPIC3D.solvers.gr_static.static_metric import (
    compute_covariant_E,
    compute_covariant_H,
    update_D_relativity,
)
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
    )
    return particles, species


def _constant_tiled_vector(static_parameters, dynamic_parameters, values):
    vector = empty_tiled_vector(static_parameters, dynamic_parameters)
    return tuple(vector[i].at[:, :, :, :, :, :].set(values[i]) for i in range(3))


def _replace_lapse_shift(metric, lapse, shift):
    def replace_one(metric_at_location):
        shift_array = jnp.zeros_like(metric_at_location.shift)
        for i, value in enumerate(shift):
            shift_array = shift_array.at[..., i].set(value)
        return metric_at_location._replace(
            lapse=jnp.full_like(metric_at_location.lapse, lapse),
            shift=shift_array,
        )

    return metric._replace(
        D=tuple(replace_one(metric_at_location) for metric_at_location in metric.D),
        B=tuple(replace_one(metric_at_location) for metric_at_location in metric.B),
        center=replace_one(metric.center),
        vertex=replace_one(metric.vertex),
    )


def _metric_locations_with_grids(metric, dynamic_parameters):
    center_grid = dynamic_parameters.grids.tiled_center_grid
    vertex_grid = dynamic_parameters.grids.tiled_vertex_grid
    metric_locations = (
        tuple(zip(metric.D, D_FIELD_LOCATIONS))
        + tuple(zip(metric.B, B_FIELD_LOCATIONS))
        + ((metric.center, ("C", "C", "C")),)
        + ((metric.vertex, ("V", "V", "V")),)
    )

    return tuple(
        (
            metric_at_location,
            metric_for_location(center_grid, vertex_grid, location),
        )
        for metric_at_location, location in metric_locations
    )


def test_flat_cartesian_metric_matches_center_grid_shape():
    static_parameters, dynamic_parameters = kernel_parameters(Nx=4, Ny=3, Nz=2)

    metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
    g = int(static_parameters.guard_cells)
    shape = (1, 1, 1, 4 + 2 * g, 3 + 2 * g, 2 + 2 * g)

    assert metric.center.lapse.shape == shape
    assert metric.center.gamma_inv.shape == shape + (3, 3)
    assert metric.center_grad_gamma_inv.shape == shape + (3, 3, 3)
    assert jnp.allclose(metric.center.lapse, 1.0)
    assert jnp.allclose(metric.center.gamma_inv[..., 0, 0], 1.0)
    assert jnp.all(jnp.isfinite(metric.center_grad_gamma_inv))
    assert jnp.allclose(metric.center.sqrt_gamma, 1.0)


def test_kerr_schild_metric_initializers_build_finite_derivatives():
    static_parameters, dynamic_parameters = kernel_parameters(
        Nx=4,
        Ny=4,
        Nz=4,
        x_wind=1.0,
        y_wind=0.8,
        z_wind=1.0,
        x_min=2.0,
        y_min=0.17,
        z_min=0.5,
        tile_shape=(4, 4, 4),
        solver="static_metric",
        current_deposition="GR_direct",
        particle_pusher="hybrid_boris_geodesic",
    )

    cartesian = initialize_kerr_schild_cartesian_metric(
        static_parameters,
        dynamic_parameters,
        mass=0.1,
        spin=0.2,
    )
    spherical = initialize_kerr_schild_spherical_metric(
        static_parameters,
        dynamic_parameters,
        mass=0.1,
        spin=0.2,
    )

    assert jnp.all(jnp.isfinite(cartesian.center.christoffel))
    assert jnp.all(jnp.isfinite(cartesian.center.grad_lapse))
    assert jnp.all(jnp.isfinite(cartesian.center.grad_shift))
    assert cartesian.center_grad_gamma_inv.shape == cartesian.center.lapse.shape + (3, 3, 3)
    assert jnp.all(jnp.isfinite(cartesian.center_grad_gamma_inv))
    assert jnp.all(jnp.isfinite(spherical.center.christoffel))
    assert jnp.all(jnp.isfinite(spherical.center.grad_lapse))
    assert jnp.all(jnp.isfinite(spherical.center.grad_shift))
    assert spherical.center_grad_gamma_inv.shape == spherical.center.lapse.shape + (3, 3, 3)
    assert jnp.all(jnp.isfinite(spherical.center_grad_gamma_inv))


def test_flat_cylindrical_metric_fills_nonzero_christoffels():
    static_parameters, dynamic_parameters = kernel_parameters(
        Nx=8,
        Ny=2,
        Nz=1,
        x_wind=8.0,
        y_wind=1.0,
        z_wind=1.0,
        x_min=2.5,
        y_min=0.2,
        z_min=0.2,
        tile_shape=(8, 2, 1),
        solver="static_metric",
        current_deposition="GR_direct",
        particle_pusher="hybrid_boris_geodesic",
    )
    metric = initialize_flat_cylindrical_metric(static_parameters, dynamic_parameters)

    g = int(static_parameters.guard_cells)
    active = (slice(None), slice(None), slice(None), slice(g, -g), slice(g, -g), slice(g, -g))
    r = dynamic_parameters.grids.tiled_center_grid[0][:, :, :, g:-g]
    r = r[:, :, :, :, jnp.newaxis, jnp.newaxis]

    gamma_r_phiphi = metric.center.christoffel[active + (0, 1, 1)]
    gamma_phi_rphi = metric.center.christoffel[active + (1, 0, 1)]
    expected_r_phiphi = -jnp.broadcast_to(r, gamma_r_phiphi.shape)
    expected_phi_rphi = 1.0 / jnp.broadcast_to(r, gamma_phi_rphi.shape)

    mask = jnp.abs(expected_r_phiphi) > float(dynamic_parameters.dx)
    assert jnp.allclose(gamma_r_phiphi[mask], expected_r_phiphi[mask], rtol=0.0, atol=0.25)
    assert jnp.allclose(gamma_phi_rphi[mask], expected_phi_rphi[mask], rtol=0.0, atol=0.25)


def test_static_metric_constitutive_fields_include_lapse_and_shift_terms():
    static_parameters, dynamic_parameters = kernel_parameters(
        Nx=4,
        Ny=4,
        Nz=4,
        x_wind=4.0,
        y_wind=4.0,
        z_wind=4.0,
        tile_shape=(4, 4, 4),
        solver="static_metric",
        current_deposition="GR_direct",
        particle_pusher="hybrid_boris_geodesic",
    )
    metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
    metric = _replace_lapse_shift(metric, lapse=0.7, shift=(0.2, -0.1, 0.15))
    D = _constant_tiled_vector(static_parameters, dynamic_parameters, (1.1, -0.7, 0.3))
    B = _constant_tiled_vector(static_parameters, dynamic_parameters, (0.4, 0.9, -1.2))

    E = compute_covariant_E(D, B, metric)
    H = compute_covariant_H(D, B, metric)

    g = int(static_parameters.guard_cells)
    active = (slice(None), slice(None), slice(None), slice(g, -g), slice(g, -g), slice(g, -g))
    beta = jnp.asarray((0.2, -0.1, 0.15))
    D0 = jnp.asarray((1.1, -0.7, 0.3))
    B0 = jnp.asarray((0.4, 0.9, -1.2))
    expected_E = 0.7 * D0 + jnp.cross(beta, B0)
    expected_H = 0.7 * B0 - jnp.cross(beta, D0)

    for i in range(3):
        assert jnp.allclose(E[i][active], expected_E[i])
        assert jnp.allclose(H[i][active], expected_H[i])


def test_magnetic_boris_rotation_raises_covariant_momentum_in_cross_product():
    gamma = jnp.asarray(
        (
            (1.0, 0.2, 0.0),
            (0.2, 2.0, 0.1),
            (0.0, 0.1, 3.0),
        )
    )
    gamma_inv = jnp.linalg.inv(gamma)
    metric = Metric(
        lapse=jnp.asarray(0.8),
        shift=jnp.asarray((0.0, 0.0, 0.0)),
        gamma=gamma,
        gamma_inv=gamma_inv,
        sqrt_gamma=jnp.sqrt(jnp.linalg.det(gamma)),
        christoffel=jnp.zeros((3, 3, 3)),
        grad_lapse=jnp.zeros(3),
        grad_shift=jnp.zeros((3, 3)),
    )
    u_minus = jnp.asarray((0.3, -0.4, 0.2))
    B_con = jnp.asarray((0.1, 0.5, -0.2))
    q_over_m = 1.7
    dt = 0.31

    u_plus = _magnetic_boris_rotation(u_minus, B_con, metric, q_over_m, dt)

    Gamma_minus = jnp.sqrt(1.0 + jnp.einsum("i,ij,j->", u_minus, gamma_inv, u_minus))
    u0_bar = Gamma_minus / metric.lapse
    t_con = q_over_m * B_con * 0.5 * dt / u0_bar
    t_cov = gamma @ t_con
    t_norm = jnp.dot(t_con, t_cov)
    u_minus_con = gamma_inv @ u_minus
    u_prime = u_minus + metric.sqrt_gamma * jnp.cross(u_minus_con, t_con)
    s_con = 2.0 * t_con / (1.0 + t_norm)
    u_prime_con = gamma_inv @ u_prime
    expected = u_minus + metric.sqrt_gamma * jnp.cross(u_prime_con, s_con)

    np.testing.assert_allclose(np.asarray(u_plus), np.asarray(expected), rtol=0.0, atol=1.0e-12)


def test_geodesic_rhs_is_removed_from_hybrid_pusher_module():
    assert not hasattr(hybrid_pusher, "geodesic_rhs")


def test_GR_position_update_uses_lapse_scaled_contravariant_velocity_minus_shift():
    gamma = jnp.asarray(
        (
            (1.0, 0.2, 0.0),
            (0.2, 1.5, 0.1),
            (0.0, 0.1, 2.0),
        )
    )
    gamma_inv = jnp.linalg.inv(gamma)
    metric = Metric(
        lapse=jnp.asarray(0.7),
        shift=jnp.asarray((0.2, -0.1, 0.15)),
        gamma=gamma,
        gamma_inv=gamma_inv,
        sqrt_gamma=jnp.sqrt(jnp.linalg.det(gamma)),
        christoffel=jnp.zeros((3, 3, 3)),
        grad_lapse=jnp.zeros(3),
        grad_shift=jnp.zeros((3, 3)),
    )
    u_cov = jnp.asarray((0.4, -0.2, 0.1))

    dx_dt = GR_position_update(jnp.asarray((0.0, 0.0, 0.0)), u_cov, metric)

    Gamma = jnp.sqrt(1.0 + jnp.einsum("i,ij,j->", u_cov, gamma_inv, u_cov))
    expected = metric.lapse * (gamma_inv @ u_cov) / Gamma - metric.shift
    np.testing.assert_allclose(np.asarray(dx_dt), np.asarray(expected), rtol=0.0, atol=1.0e-12)


def test_geodesic_velocity_returns_zero_for_flat_constant_metric():
    metric = Metric(
        lapse=jnp.asarray(1.0),
        shift=jnp.asarray((0.0, 0.0, 0.0)),
        gamma=jnp.eye(3),
        gamma_inv=jnp.eye(3),
        sqrt_gamma=jnp.asarray(1.0),
        christoffel=jnp.zeros((3, 3, 3)),
        grad_lapse=jnp.zeros(3),
        grad_shift=jnp.zeros((3, 3)),
    )
    u_cov = jnp.asarray((0.4, -0.2, 0.1))
    grad_gamma_inv = jnp.zeros((3, 3, 3))

    du_dt = geodesic_velocity(
        jnp.asarray((0.0, 0.0, 0.0)),
        u_cov,
        metric,
        grad_gamma_inv,
    )

    assert du_dt.shape == u_cov.shape
    np.testing.assert_allclose(np.asarray(du_dt), np.zeros(3), rtol=0.0, atol=1.0e-12)


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


def test_hybrid_boris_geodesic_push_uses_current_position_for_both_electric_half_steps():
    static_parameters, dynamic_parameters = kernel_parameters(
        Nx=8,
        Ny=2,
        Nz=2,
        x_wind=8.0,
        y_wind=2.0,
        z_wind=2.0,
        dt=0.2,
        tile_shape=(8, 2, 2),
        solver="static_metric",
        current_deposition="GR_direct",
        particle_pusher="hybrid_boris_geodesic",
    )
    metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
    D = empty_tiled_vector(static_parameters, dynamic_parameters)
    B = empty_tiled_vector(static_parameters, dynamic_parameters)

    D_grid_x = dynamic_parameters.grids.tiled_center_grid[0][:, :, :, :, jnp.newaxis, jnp.newaxis]
    D = (D[0].at[:, :, :, :, :, :].set(D_grid_x), D[1], D[2])

    x_n = jnp.asarray((2.0, 0.0, 0.0))
    u_n_minushalf = jnp.asarray((0.3, 0.0, 0.0))
    particles = TiledParticles(
        x=x_n.reshape((1, 1, 1, 1, 1, 3)),
        u=u_n_minushalf.reshape((1, 1, 1, 1, 1, 3)),
        active=jnp.ones((1, 1, 1, 1, 1), dtype=bool),
    )
    species = SpeciesConfig(
        charge=jnp.asarray([1.0]),
        mass=jnp.asarray([1.0]),
        weight=jnp.asarray([1.0]),
        update_x=jnp.asarray([[True, True, True]]),
    )

    pushed, centered = hybrid_boris_geodesic_push(
        particles,
        species,
        D,
        B,
        metric,
        static_parameters,
        dynamic_parameters,
    )

    D_grid = hybrid_pusher._metric_component_grid(
        hybrid_pusher.D_FIELD_LOCATIONS[0],
        dynamic_parameters,
        0,
        0,
        0,
    )
    D_at_x_n = hybrid_pusher._sample_scalar(
        D[0][0, 0, 0],
        x_n[:1],
        x_n[1:2],
        x_n[2:3],
        D_grid,
        static_parameters.shape_factor,
        (True, True, True),
        (static_parameters.guard_cells,) * 3,
    )[0]
    expected_u = u_n_minushalf + dynamic_parameters.dt * jnp.asarray((D_at_x_n, 0.0, 0.0))
    expected_dx_dt = expected_u / jnp.sqrt(1.0 + jnp.dot(expected_u, expected_u))

    np.testing.assert_allclose(
        np.asarray(pushed.u[0, 0, 0, 0, 0]),
        np.asarray(expected_u),
        rtol=0.0,
        atol=1.0e-6,
    )
    assert jnp.allclose(centered.u[0, 0, 0, 0, 0], pushed.u[0, 0, 0, 0, 0])
    np.testing.assert_allclose(
        np.asarray(centered.x[0, 0, 0, 0, 0]),
        np.asarray(x_n + 0.5 * dynamic_parameters.dt * expected_dx_dt),
        rtol=0.0,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(pushed.x[0, 0, 0, 0, 0]),
        np.asarray(x_n + dynamic_parameters.dt * expected_dx_dt),
        rtol=0.0,
        atol=1.0e-6,
    )


def test_hybrid_boris_geodesic_push_accepts_multiple_species_in_one_tile():
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
    zeros = empty_tiled_vector(static_parameters, dynamic_parameters)
    x = jnp.zeros((1, 1, 1, 2, 3, 3))
    u = jnp.asarray(
        (
            ((0.1, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.0, 0.0)),
            ((-0.1, 0.0, 0.0), (-0.2, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ),
        dtype=float,
    ).reshape((1, 1, 1, 2, 3, 3))
    active = jnp.asarray(
        (
            (True, True, False),
            (True, True, False),
        ),
        dtype=bool,
    ).reshape((1, 1, 1, 2, 3))
    particles = TiledParticles(x=x, u=u, active=active)
    species = SpeciesConfig(
        charge=jnp.asarray([0.0, 0.0]),
        mass=jnp.asarray([1.0, 2.0]),
        weight=jnp.asarray([1.0, 1.0]),
        update_x=jnp.asarray([[True, True, True], [True, True, True]]),
    )

    pushed, centered = hybrid_boris_geodesic_push(
        particles,
        species,
        zeros,
        zeros,
        metric,
        static_parameters,
        dynamic_parameters,
    )

    assert pushed.x.shape == x.shape
    assert centered.x.shape == x.shape
    assert jnp.all(jnp.isfinite(pushed.x))


def test_hybrid_boris_geodesic_push_masks_position_and_velocity_by_species_direction():
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
    D = _constant_tiled_vector(static_parameters, dynamic_parameters, (0.2, 0.2, 0.2))
    B = empty_tiled_vector(static_parameters, dynamic_parameters)
    particles = TiledParticles(
        x=jnp.zeros((1, 1, 1, 2, 1, 3)),
        u=jnp.zeros((1, 1, 1, 2, 1, 3)),
        active=jnp.ones((1, 1, 1, 2, 1), dtype=bool),
    )
    species = SpeciesConfig(
        charge=jnp.asarray([1.0, 1.0]),
        mass=jnp.asarray([1.0, 1.0]),
        weight=jnp.asarray([1.0, 1.0]),
        update_x=jnp.asarray(
            (
                (False, True, False),
                (True, False, True),
            )
        ),
    )

    pushed, centered = hybrid_boris_geodesic_push(
        particles,
        species,
        D,
        B,
        metric,
        static_parameters,
        dynamic_parameters,
    )

    update_mask = species.update_x.reshape((1, 1, 1, 2, 1, 3))
    assert jnp.allclose(jnp.where(update_mask, 0.0, pushed.u), 0.0)
    assert jnp.allclose(jnp.where(update_mask, 0.0, pushed.x), 0.0)
    assert jnp.allclose(jnp.where(update_mask, 0.0, centered.x), 0.0)
    assert jnp.all(jnp.abs(pushed.u[update_mask]) > 0.0)
    assert jnp.all(jnp.abs(pushed.x[update_mask]) > 0.0)
    assert jnp.all(jnp.abs(centered.x[update_mask]) > 0.0)


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


def test_GR_direct_deposition_returns_fpic_shifted_source_current():
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
    metric = _replace_lapse_shift(metric, lapse=0.7, shift=(0.2, -0.1, 0.15))
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
    expected = jnp.asarray((0.7 * 0.5 / gamma, 0.0, 0.0)) - jnp.asarray((0.2, -0.1, 0.15))
    assert jnp.allclose(J[0][interior], expected[0])
    assert jnp.allclose(J[1][interior], expected[1])
    assert jnp.allclose(J[2][interior], expected[2])


def test_GR_direct_deposition_masks_complete_shifted_current_by_direction():
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
    metric = _replace_lapse_shift(metric, lapse=0.7, shift=(0.2, -0.1, 0.15))
    particles, species = _single_particle_state(static_parameters, dynamic_parameters, (0.5, 0.25, -0.125))
    J_template = empty_tiled_vector(static_parameters, dynamic_parameters)

    full_current = GR_direct_deposition(
        particles,
        species,
        J_template,
        metric,
        static_parameters,
        dynamic_parameters,
    )
    masked_current = GR_direct_deposition(
        particles,
        species._replace(update_x=jnp.asarray([[False, True, False]])),
        J_template,
        metric,
        static_parameters,
        dynamic_parameters,
    )
    disabled_current = GR_direct_deposition(
        particles,
        species._replace(update_x=jnp.zeros((1, 3), dtype=bool)),
        J_template,
        metric,
        static_parameters,
        dynamic_parameters,
    )

    assert jnp.allclose(masked_current[0], 0.0)
    assert jnp.allclose(masked_current[1], full_current[1])
    assert jnp.allclose(masked_current[2], 0.0)
    for component in disabled_current:
        assert jnp.allclose(component, 0.0)


def test_GR_direct_deposition_is_adjoint_to_staggered_field_gather():
    positions = jnp.asarray(
        (
            ((0.1, 0.2, 0.3), (3.9, 1.7, 2.2)),
            ((4.1, 2.8, 1.4), (7.9, 3.8, 3.7)),
        )
    ).reshape((2, 1, 1, 1, 2, 3))
    u_cov = jnp.asarray(
        (
            ((0.31, -0.17, 0.09), (-0.22, 0.28, -0.13)),
            ((0.19, 0.11, -0.24), (-0.27, -0.16, 0.21)),
        )
    ).reshape((2, 1, 1, 1, 2, 3))
    active = jnp.ones((2, 1, 1, 1, 2), dtype=bool)
    species = SpeciesConfig(
        charge=jnp.asarray([-0.7]),
        mass=jnp.asarray([1.0]),
        weight=jnp.asarray([1.3]),
        update_x=jnp.asarray([[True, True, True]]),
    )

    for shape_factor in (1, 2):
        static_parameters, dynamic_parameters = kernel_parameters(
            Nx=8,
            Ny=4,
            Nz=4,
            x_wind=8.0,
            y_wind=4.0,
            z_wind=4.0,
            x_min=0.0,
            y_min=0.0,
            z_min=0.0,
            dt=0.1,
            tile_shape=(4, 4, 4),
            shape_factor=shape_factor,
            current_filter="none",
            solver="static_metric",
            current_deposition="GR_direct",
            particle_pusher="hybrid_boris_geodesic",
        )
        metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
        metric = _replace_lapse_shift(metric, lapse=0.73, shift=(0.12, -0.08, 0.05))
        particles = TiledParticles(x=positions, u=u_cov, active=active)

        center_grid = dynamic_parameters.grids.tiled_center_grid
        vertex_grid = dynamic_parameters.grids.tiled_vertex_grid
        D_grids = tuple(
            metric_for_location(center_grid, vertex_grid, location)
            for location in D_FIELD_LOCATIONS
        )
        D = []
        for component, grid in enumerate(D_grids):
            x_grid, y_grid, z_grid = grid
            field = (
                jnp.sin(2.0 * jnp.pi * x_grid[..., :, jnp.newaxis, jnp.newaxis] / 8.0)
                + (0.3 + 0.1 * component)
                * jnp.cos(2.0 * jnp.pi * y_grid[..., jnp.newaxis, :, jnp.newaxis] / 4.0)
                + (0.2 - 0.05 * component)
                * jnp.sin(2.0 * jnp.pi * z_grid[..., jnp.newaxis, jnp.newaxis, :] / 4.0)
            )
            D.append(field)
        D = tuple(D)

        J = GR_direct_deposition(
            particles,
            species,
            empty_tiled_vector(static_parameters, dynamic_parameters),
            metric,
            static_parameters,
            dynamic_parameters,
        )

        g = int(static_parameters.guard_cells)
        interior = (
            slice(None),
            slice(None),
            slice(None),
            slice(g, -g),
            slice(g, -g),
            slice(g, -g),
        )
        grid_work = sum(
            jnp.sum(
                metric.D[i].sqrt_gamma[interior]
                * D[i][interior]
                * J[i][interior]
            )
            for i in range(3)
        )
        grid_work *= dynamic_parameters.dx * dynamic_parameters.dy * dynamic_parameters.dz

        gathered_D = []
        for tx in range(2):
            tile_grids = tuple(
                tuple(axis[tx, 0, 0] for axis in component_grid)
                for component_grid in D_grids
            )
            gathered_D.append(
                hybrid_pusher._sample_vector(
                    tuple(D[i][tx, 0, 0] for i in range(3)),
                    particles.x[tx, 0, 0, ..., 0],
                    particles.x[tx, 0, 0, ..., 1],
                    particles.x[tx, 0, 0, ..., 2],
                    tile_grids,
                    shape_factor,
                    (True, True, True),
                    (g, g, g),
                )
            )
        gathered_D = jnp.stack(gathered_D, axis=0).reshape(particles.x.shape)

        Gamma = jnp.sqrt(1.0 + jnp.sum(particles.u**2, axis=-1))
        source_velocity = 0.73 * particles.u / Gamma[..., jnp.newaxis]
        source_velocity -= jnp.asarray((0.12, -0.08, 0.05))
        weighted_charge = species.charge[0] * species.weight[0]
        particle_work = weighted_charge * jnp.sum(
            particles.active[..., jnp.newaxis]
            * source_velocity
            * gathered_D
        )

        scale = jnp.maximum(jnp.abs(grid_work), jnp.abs(particle_work))
        relative_residual = jnp.abs(grid_work - particle_work) / scale
        assert relative_residual < 1.0e-12


def test_flat_GR_direct_deposition_matches_standard_stencil_on_reduced_axes():
    positions = jnp.asarray(
        ((0.1, 0.0, 0.0), (0.9, 0.0, 0.0), (7.1, 0.0, 0.0), (7.9, 0.0, 0.0))
    ).reshape((1, 1, 1, 1, 4, 3))
    u_cov = jnp.asarray(
        ((0.31, -0.17, 0.09), (-0.22, 0.28, -0.13), (0.19, 0.11, -0.24), (-0.27, -0.16, 0.21))
    ).reshape((1, 1, 1, 1, 4, 3))
    active = jnp.ones((1, 1, 1, 1, 4), dtype=bool)
    species = SpeciesConfig(
        charge=jnp.asarray([-0.7]),
        mass=jnp.asarray([1.0]),
        weight=jnp.asarray([1.3]),
        update_x=jnp.asarray([[True, True, True]]),
    )

    for shape_factor in (1, 2):
        static_parameters, dynamic_parameters = kernel_parameters(
            Nx=8,
            Ny=1,
            Nz=1,
            x_wind=8.0,
            y_wind=1.0,
            z_wind=1.0,
            x_min=0.0,
            y_min=-0.5,
            z_min=-0.5,
            tile_shape=(8, 1, 1),
            shape_factor=shape_factor,
            current_filter="none",
            solver="static_metric",
            current_deposition="GR_direct",
            particle_pusher="hybrid_boris_geodesic",
        )
        metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
        GR_particles = TiledParticles(x=positions, u=u_cov, active=active)
        Gamma = jnp.sqrt(1.0 + jnp.sum(u_cov**2, axis=-1))
        standard_particles = GR_particles._replace(u=u_cov / Gamma[..., jnp.newaxis])
        J_template = empty_tiled_vector(static_parameters, dynamic_parameters)

        GR_J = GR_direct_deposition(
            GR_particles,
            species,
            J_template,
            metric,
            static_parameters,
            dynamic_parameters,
        )
        standard_J = J_from_rhov(
            standard_particles,
            species,
            J_template,
            static_parameters,
            dynamic_parameters,
        )

        for GR_component, standard_component in zip(GR_J, standard_J):
            np.testing.assert_allclose(
                np.asarray(GR_component),
                np.asarray(standard_component),
                rtol=0.0,
                atol=1.0e-12,
            )


def test_GR_direct_deposition_returns_physical_spherical_current():
    static_parameters, dynamic_parameters = kernel_parameters(
        Nx=8,
        Ny=1,
        Nz=1,
        x_wind=8.0,
        y_wind=1.0,
        z_wind=1.0,
        x_min=2.0,
        y_min=0.4,
        z_min=0.2,
        dt=0.1,
        tile_shape=(8, 1, 1),
        shape_factor=1,
        current_filter="none",
        solver="static_metric",
        current_deposition="GR_direct",
        particle_pusher="hybrid_boris_geodesic",
        metric="flat_spherical",
    )
    metric = initialize_flat_spherical_metric(static_parameters, dynamic_parameters)
    J_template = empty_tiled_vector(static_parameters, dynamic_parameters)
    species = SpeciesConfig(
        charge=jnp.asarray([1.0]),
        mass=jnp.asarray([1.0]),
        weight=jnp.asarray([1.0]),
        update_x=jnp.asarray([[True, True, True]]),
    )

    def deposit_radial_particle(radius):
        particles = TiledParticles(
            x=jnp.asarray((radius, 0.4, 0.2)).reshape((1, 1, 1, 1, 1, 3)),
            u=jnp.asarray((0.5, 0.0, 0.0)).reshape((1, 1, 1, 1, 1, 3)),
            active=jnp.ones((1, 1, 1, 1, 1), dtype=bool),
        )
        J = GR_direct_deposition(
            particles,
            species,
            J_template,
            metric,
            static_parameters,
            dynamic_parameters,
        )

        g = int(static_parameters.guard_cells)
        active = (
            slice(None),
            slice(None),
            slice(None),
            slice(g, -g),
            slice(g, -g),
            slice(g, -g),
        )
        physical_current_sum = jnp.sum(J[0][active])
        conformal_flux = jnp.sum(
            metric.D[0].sqrt_gamma[active]
            * J[0][active]
            * dynamic_parameters.dx
            * dynamic_parameters.dy
            * dynamic_parameters.dz
        )
        return physical_current_sum, conformal_flux

    inner_current, inner_flux = deposit_radial_particle(2.5)
    outer_current, outer_flux = deposit_radial_particle(6.5)
    expected_flux = 0.5 / jnp.sqrt(1.0 + 0.5**2)

    assert jnp.allclose(inner_flux, expected_flux, rtol=1.0e-5, atol=1.0e-6)
    assert jnp.allclose(outer_flux, expected_flux, rtol=1.0e-5, atol=1.0e-6)
    assert inner_current > outer_current


def test_update_D_relativity_consumes_physical_current_without_metric_rescaling():
    static_parameters, dynamic_parameters = kernel_parameters(
        Nx=4,
        Ny=4,
        Nz=1,
        x_wind=1.0,
        y_wind=0.8,
        z_wind=1.0,
        x_min=2.0,
        y_min=0.17,
        z_min=0.2,
        dt=0.1,
        tile_shape=(4, 4, 1),
        solver="static_metric",
        current_deposition="GR_direct",
        particle_pusher="hybrid_boris_geodesic",
        metric="flat_spherical",
    )
    metric = initialize_flat_spherical_metric(static_parameters, dynamic_parameters)
    D = empty_tiled_vector(static_parameters, dynamic_parameters)
    H = empty_tiled_vector(static_parameters, dynamic_parameters)
    J = _constant_tiled_vector(static_parameters, dynamic_parameters, (1.0, 0.0, 0.0))

    D_next = update_D_relativity(
        D,
        H,
        J,
        metric,
        static_parameters,
        dynamic_parameters,
        dynamic_parameters.dt,
    )

    g = int(static_parameters.guard_cells)
    active = (
        slice(None),
        slice(None),
        slice(None),
        slice(g, -g),
        slice(g, -g),
        slice(g, -g),
    )
    assert jnp.allclose(
        D_next[0][active],
        -4.0 * jnp.pi * dynamic_parameters.dt,
    )


def test_static_metric_time_loop_retiles_midpoint_and_fullstep_particles():
    static_parameters, dynamic_parameters = kernel_parameters(
        Nx=8,
        Ny=1,
        Nz=1,
        x_wind=8.0,
        y_wind=1.0,
        z_wind=1.0,
        dt=0.2,
        tile_shape=(4, 1, 1),
        solver="static_metric",
        current_deposition="GR_direct",
        particle_pusher="hybrid_boris_geodesic",
    )
    metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
    x = jnp.zeros((2, 1, 1, 1, 2, 3))
    u = jnp.zeros_like(x)
    active = jnp.zeros((2, 1, 1, 1, 2), dtype=bool)
    x = x.at[0, 0, 0, 0, 0].set(jnp.asarray((-0.02, 0.0, 0.0)))
    u = u.at[0, 0, 0, 0, 0].set(jnp.asarray((1.0, 0.0, 0.0)))
    active = active.at[0, 0, 0, 0, 0].set(True)
    particles = shard_tiled_particles(
        TiledParticles(x=x, u=u, active=active),
        static_parameters,
    )
    species = SpeciesConfig(
        charge=jnp.asarray([1.0]),
        mass=jnp.asarray([1.0]),
        weight=jnp.asarray([1.0]),
        update_x=jnp.asarray([[True, True, True]]),
    )
    D = empty_tiled_vector(static_parameters, dynamic_parameters)
    B = empty_tiled_vector(static_parameters, dynamic_parameters)
    J = empty_tiled_vector(static_parameters, dynamic_parameters)
    rho = jnp.zeros_like(J[0])
    phi = jnp.zeros_like(J[0])
    fields = (D, B, J, rho, phi, (D, B), metric, (D, B), jnp.asarray(False))

    particles, fields = time_loop_static_metric(
        particles,
        species,
        fields,
        static_parameters,
        dynamic_parameters,
    )

    assert int(jnp.sum(particles.active[0, 0, 0])) == 0
    assert int(jnp.sum(particles.active[1, 0, 0])) == 1
    assert particles.x[1, 0, 0, 0, 0, 0] > 0.0
    assert jnp.any(jnp.abs(fields[2][0][1, 0, 0]) > 0.0)
    assert bool(fields[-1]) is False


def test_static_metric_time_loop_reports_particle_refresh_overflow():
    static_parameters, dynamic_parameters = kernel_parameters(
        Nx=8,
        Ny=1,
        Nz=1,
        x_wind=8.0,
        y_wind=1.0,
        z_wind=1.0,
        dt=0.2,
        tile_shape=(4, 1, 1),
        solver="static_metric",
        current_deposition="GR_direct",
        particle_pusher="hybrid_boris_geodesic",
    )
    metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
    x = jnp.zeros((2, 1, 1, 1, 1, 3))
    u = jnp.zeros_like(x)
    active = jnp.ones((2, 1, 1, 1, 1), dtype=bool)
    x = x.at[0, 0, 0, 0, 0].set(jnp.asarray((-0.02, 0.0, 0.0)))
    x = x.at[1, 0, 0, 0, 0].set(jnp.asarray((1.0, 0.0, 0.0)))
    u = u.at[0, 0, 0, 0, 0].set(jnp.asarray((1.0, 0.0, 0.0)))
    particles = shard_tiled_particles(
        TiledParticles(x=x, u=u, active=active),
        static_parameters,
    )
    species = SpeciesConfig(
        charge=jnp.asarray([0.0]),
        mass=jnp.asarray([1.0]),
        weight=jnp.asarray([1.0]),
        update_x=jnp.asarray([[True, True, True]]),
    )
    D = empty_tiled_vector(static_parameters, dynamic_parameters)
    B = empty_tiled_vector(static_parameters, dynamic_parameters)
    J = empty_tiled_vector(static_parameters, dynamic_parameters)
    rho = jnp.zeros_like(J[0])
    phi = jnp.zeros_like(J[0])
    fields = (D, B, J, rho, phi, (D, B), metric, (D, B), jnp.asarray(False))

    particles, fields = time_loop_static_metric(
        particles,
        species,
        fields,
        static_parameters,
        dynamic_parameters,
    )

    assert bool(fields[-1]) is True
    assert int(jnp.sum(particles.active)) == 1


def test_flat_cylindrical_metric_stores_signed_sqrt_gamma_at_all_yee_locations():
    static_parameters, dynamic_parameters = kernel_parameters(
        Nx=8,
        Ny=2,
        Nz=1,
        x_wind=8.0,
        y_wind=1.0,
        z_wind=1.0,
        x_min=-3.75,
        y_min=0.2,
        z_min=0.2,
        tile_shape=(8, 2, 1),
        solver="static_metric",
        current_deposition="GR_direct",
        particle_pusher="hybrid_boris_geodesic",
        metric="flat_cylindrical",
    )
    metric = initialize_flat_cylindrical_metric(static_parameters, dynamic_parameters)

    for metric_at_location, grid in _metric_locations_with_grids(metric, dynamic_parameters):
        R = grid[0][..., :, jnp.newaxis, jnp.newaxis]
        expected = jnp.broadcast_to(R, metric_at_location.sqrt_gamma.shape)

        assert jnp.allclose(metric_at_location.sqrt_gamma, expected)
        assert jnp.any(metric_at_location.sqrt_gamma < 0.0)
        assert jnp.any(metric_at_location.sqrt_gamma > 0.0)


def test_spherical_metrics_store_signed_sqrt_gamma_at_all_yee_locations():
    ntheta = 8
    dtheta = 2.0 * np.pi / ntheta
    static_parameters, dynamic_parameters = kernel_parameters(
        Nx=4,
        Ny=ntheta,
        Nz=1,
        x_wind=1.0,
        y_wind=2.0 * np.pi,
        z_wind=1.0,
        x_min=2.0,
        y_min=0.25 * dtheta,
        z_min=0.2,
        tile_shape=(4, ntheta, 1),
        solver="static_metric",
        current_deposition="GR_direct",
        particle_pusher="hybrid_boris_geodesic",
    )
    mass = 0.1
    spin = 0.2
    flat_metric = initialize_flat_spherical_metric(static_parameters, dynamic_parameters)
    kerr_metric = initialize_kerr_schild_spherical_metric(
        static_parameters,
        dynamic_parameters,
        mass=mass,
        spin=spin,
    )

    for metric_at_location, grid in _metric_locations_with_grids(flat_metric, dynamic_parameters):
        R = grid[0][..., :, jnp.newaxis, jnp.newaxis]
        theta = grid[1][..., jnp.newaxis, :, jnp.newaxis]
        expected = jnp.broadcast_to(
            R**2 * jnp.sin(theta),
            metric_at_location.sqrt_gamma.shape,
        )

        assert jnp.allclose(metric_at_location.sqrt_gamma, expected)
        assert jnp.any(metric_at_location.sqrt_gamma < 0.0)
        assert jnp.any(metric_at_location.sqrt_gamma > 0.0)

    for metric_at_location, grid in _metric_locations_with_grids(kerr_metric, dynamic_parameters):
        R = grid[0][..., :, jnp.newaxis, jnp.newaxis]
        theta = grid[1][..., jnp.newaxis, :, jnp.newaxis]
        rho_squared = R**2 + spin**2 * jnp.cos(theta) ** 2
        xi = 1.0 + 2.0 * mass * R / rho_squared
        expected = jnp.broadcast_to(
            rho_squared * jnp.sqrt(xi) * jnp.sin(theta),
            metric_at_location.sqrt_gamma.shape,
        )

        assert jnp.allclose(metric_at_location.sqrt_gamma, expected)
        assert jnp.any(metric_at_location.sqrt_gamma < 0.0)
        assert jnp.any(metric_at_location.sqrt_gamma > 0.0)


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
    static_metric_state = (D, B)
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
    assert len(fields[7]) == 2
    assert bool(jax.device_get(fields[-1])) is False
    assert jnp.all(jnp.isfinite(particles.x))
    assert jnp.all(jnp.isfinite(particles.u))


def test_static_metric_time_loop_accepts_empty_particle_storage():
    static_parameters, dynamic_parameters = kernel_parameters(
        Nx=4,
        Ny=4,
        Nz=1,
        x_wind=4.0,
        y_wind=4.0,
        z_wind=1.0,
        dt=0.05,
        tile_shape=(4, 4, 1),
        solver="static_metric",
        current_deposition="GR_direct",
        particle_pusher="hybrid_boris_geodesic",
    )
    metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
    particles = TiledParticles(
        x=jnp.zeros((1, 1, 1, 0, 0, 3)),
        u=jnp.zeros((1, 1, 1, 0, 0, 3)),
        active=jnp.zeros((1, 1, 1, 0, 0), dtype=bool),
    )
    species = SpeciesConfig(
        charge=jnp.zeros((0,)),
        mass=jnp.zeros((0,)),
        weight=jnp.zeros((0,)),
        update_x=jnp.zeros((0, 3), dtype=bool),
    )
    D = empty_tiled_vector(static_parameters, dynamic_parameters)
    B = empty_tiled_vector(static_parameters, dynamic_parameters)
    J = empty_tiled_vector(static_parameters, dynamic_parameters)
    rho = jnp.zeros_like(J[0])
    phi = jnp.zeros_like(J[0])
    external_fields = (D, B)
    static_metric_state = (D, B)
    fields = (D, B, J, rho, phi, external_fields, metric, static_metric_state, jnp.asarray(False))

    particles, fields = time_loop_static_metric(
        particles,
        species,
        fields,
        static_parameters,
        dynamic_parameters,
    )

    D_next, B_next, J_next = fields[:3]
    assert particles.x.shape == (1, 1, 1, 0, 0, 3)
    assert len(fields) == 9
    assert bool(jax.device_get(fields[-1])) is False
    for vector in (D_next, B_next, J_next):
        for component in vector:
            assert jnp.all(jnp.isfinite(component))


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
