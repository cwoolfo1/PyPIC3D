from functools import partial

import jax
import jax.numpy as jnp

from PyPIC3D.particles.particle_class import TiledParticles
from PyPIC3D.pusher.boris import interpolate_field_to_particles
from PyPIC3D.relativity.core import (
    B_FIELD_LOCATIONS,
    D_FIELD_LOCATIONS,
    Metric,
    contravariant_three_velocity,
    covariant_lorentz_factor,
    lower_vector,
)


def _metric_component_grid(location, dynamic_parameters, tx, ty, tz):
    center_grid = dynamic_parameters.grids.tiled_center_grid
    vertex_grid = dynamic_parameters.grids.tiled_vertex_grid
    return tuple(
        (center_grid[axis] if location[axis] == "C" else vertex_grid[axis])[tx, ty, tz]
        for axis in range(3)
    )


def _sample_scalar(field, x, y, z, grid, shape_factor):
    particle_shape = x.shape
    return interpolate_field_to_particles(
        field,
        x.reshape(-1),
        y.reshape(-1),
        z.reshape(-1),
        grid,
        shape_factor,
        ghost_cells=True,
    ).reshape(particle_shape)


def _sample_vector(field, x, y, z, grids, shape_factor):
    return jnp.stack(
        tuple(_sample_scalar(field[i], x, y, z, grids[i], shape_factor) for i in range(3)),
        axis=-1,
    )


def _sample_tensor(field, x, y, z, grid, shape_factor):
    rows = []
    for i in range(3):
        columns = []
        for j in range(3):
            columns.append(_sample_scalar(field[..., i, j], x, y, z, grid, shape_factor))
        rows.append(jnp.stack(tuple(columns), axis=-1))
    return jnp.stack(tuple(rows), axis=-2)


def _sample_rank3(field, x, y, z, grid, shape_factor):
    planes = []
    for i in range(3):
        rows = []
        for j in range(3):
            columns = []
            for k in range(3):
                columns.append(_sample_scalar(field[..., i, j, k], x, y, z, grid, shape_factor))
            rows.append(jnp.stack(tuple(columns), axis=-1))
        planes.append(jnp.stack(tuple(rows), axis=-2))
    return jnp.stack(tuple(planes), axis=-3)


def _sample_metric(metric, x, y, z, grid, shape_factor):
    return Metric(
        lapse=_sample_scalar(metric.lapse, x, y, z, grid, shape_factor),
        shift=_sample_vector(tuple(metric.shift[..., i] for i in range(3)), x, y, z, (grid, grid, grid), shape_factor),
        gamma=_sample_tensor(metric.gamma, x, y, z, grid, shape_factor),
        gamma_inv=_sample_tensor(metric.gamma_inv, x, y, z, grid, shape_factor),
        sqrt_gamma=_sample_scalar(metric.sqrt_gamma, x, y, z, grid, shape_factor),
        christoffel=_sample_rank3(metric.christoffel, x, y, z, grid, shape_factor),
        grad_lapse=_sample_vector(tuple(metric.grad_lapse[..., i] for i in range(3)), x, y, z, (grid, grid, grid), shape_factor),
        grad_shift=_sample_tensor(metric.grad_shift, x, y, z, grid, shape_factor),
    )


def _metric_tile(metric, tx, ty, tz):
    return Metric(
        lapse=metric.lapse[tx, ty, tz],
        shift=metric.shift[tx, ty, tz],
        gamma=metric.gamma[tx, ty, tz],
        gamma_inv=metric.gamma_inv[tx, ty, tz],
        sqrt_gamma=metric.sqrt_gamma[tx, ty, tz],
        christoffel=metric.christoffel[tx, ty, tz],
        grad_lapse=metric.grad_lapse[tx, ty, tz],
        grad_shift=metric.grad_shift[tx, ty, tz],
    )


def _grad_gamma_inv_from_christoffel(gamma_inv, christoffel):
    grad_gamma_inv = jnp.zeros(gamma_inv.shape[:-2] + (3, 3, 3), dtype=gamma_inv.dtype)
    for i in range(3):
        for l in range(3):
            for m in range(3):
                value = 0.0
                for n in range(3):
                    value = value - (
                        christoffel[..., l, i, n] * gamma_inv[..., n, m]
                        + christoffel[..., m, i, n] * gamma_inv[..., l, n]
                    )
                grad_gamma_inv = grad_gamma_inv.at[..., i, l, m].set(value)
    return grad_gamma_inv


def GR_position_update(position, u_cov, metric):
    """
    Coordinate velocity dx^i/dt from covariant spatial momentum u_i.
    """

    del position
    Gamma = covariant_lorentz_factor(u_cov, metric.gamma_inv)
    u_con = jnp.einsum("...ij,...j->...i", metric.gamma_inv, u_cov)
    return metric.lapse[..., jnp.newaxis] * u_con / Gamma[..., jnp.newaxis] - metric.shift


def geodesic_velocity(position, u_cov, metric):
    """
    Geodesic source term du_i/dt for covariant spatial momentum.
    """

    del position
    Gamma = covariant_lorentz_factor(u_cov, metric.gamma_inv)
    grad_gamma_inv = _grad_gamma_inv_from_christoffel(metric.gamma_inv, metric.christoffel)
    grad_beta_term = jnp.einsum("...j,...ji->...i", u_cov, metric.grad_shift)
    metric_force = (-0.5 * metric.lapse / Gamma)[..., jnp.newaxis] * jnp.einsum(
        "...l,...m,...ilm->...i",
        u_cov,
        u_cov,
        grad_gamma_inv,
    )

    return -Gamma[..., jnp.newaxis] * metric.grad_lapse + grad_beta_term + metric_force


def _magnetic_boris_rotation(u_minus, B_con, metric, q_over_m, dt):
    Gamma_minus = covariant_lorentz_factor(u_minus, metric.gamma_inv)
    u0_bar = Gamma_minus / metric.lapse
    t_con = (q_over_m * dt / (2.0 * u0_bar))[..., jnp.newaxis] * B_con
    t_cov = lower_vector(t_con, metric.gamma)
    t_norm = jnp.einsum("...i,...i->...", t_con, t_cov)

    u_minus_con = jnp.einsum("...ij,...j->...i", metric.gamma_inv, u_minus)
    u_prime = u_minus + metric.sqrt_gamma[..., jnp.newaxis] * jnp.cross(u_minus_con, t_con)
    s_con = 2.0 * t_con / (1.0 + t_norm)[..., jnp.newaxis]
    u_prime_con = jnp.einsum("...ij,...j->...i", metric.gamma_inv, u_prime)
    return u_minus + metric.sqrt_gamma[..., jnp.newaxis] * jnp.cross(u_prime_con, s_con)


def _electromagnetic_boris_step(position, u_cov, q_over_m, D_tiles, B_tiles, metric_tiles, static_parameters, dynamic_parameters, tx, ty, tz, dt):
    shape_factor = static_parameters.shape_factor
    x = position[..., 0]
    y = position[..., 1]
    z = position[..., 2]

    D_grids = tuple(
        _metric_component_grid(D_FIELD_LOCATIONS[i], dynamic_parameters, tx, ty, tz)
        for i in range(3)
    )
    B_grids = tuple(
        _metric_component_grid(B_FIELD_LOCATIONS[i], dynamic_parameters, tx, ty, tz)
        for i in range(3)
    )
    center_grid = _metric_component_grid(("C", "C", "C"), dynamic_parameters, tx, ty, tz)

    D_con = _sample_vector(
        tuple(D_tiles[i][tx, ty, tz] for i in range(3)),
        x,
        y,
        z,
        D_grids,
        shape_factor,
    )
    B_con = _sample_vector(
        tuple(B_tiles[i][tx, ty, tz] for i in range(3)),
        x,
        y,
        z,
        B_grids,
        shape_factor,
    )
    metric = _sample_metric(_metric_tile(metric_tiles.center, tx, ty, tz), x, y, z, center_grid, shape_factor)
    E_cov = lower_vector(D_con, metric.gamma)

    u_minus = u_cov + (q_over_m * dt / 2.0)[..., jnp.newaxis] * metric.lapse[..., jnp.newaxis] * E_cov
    u_plus = _magnetic_boris_rotation(u_minus, B_con, metric, q_over_m, dt)
    u_new = u_plus + (q_over_m * dt / 2.0)[..., jnp.newaxis] * metric.lapse[..., jnp.newaxis] * E_cov

    return u_new


def _sample_center_metric_at_position(position, metric_tiles, static_parameters, dynamic_parameters, tx, ty, tz):
    shape_factor = static_parameters.shape_factor
    center_grid = _metric_component_grid(("C", "C", "C"), dynamic_parameters, tx, ty, tz)

    return _sample_metric(
        _metric_tile(metric_tiles.center, tx, ty, tz),
        position[..., 0],
        position[..., 1],
        position[..., 2],
        center_grid,
        shape_factor,
    )


@partial(jax.jit, static_argnames="static_parameters")
def hybrid_boris_geodesic_push(
    particles,
    species_config,
    D_tiles,
    B_tiles,
    metric,
    static_parameters,
    dynamic_parameters,
):
    """
    Strang-split second-order 3+1 particle push.

    Particle positions are contravariant coordinates.  ``particles.u`` stores
    covariant spatial velocity components ``u_i``.
    """

    dt = dynamic_parameters.dt
    ntx, nty, ntz = particles.active.shape[:3]
    q_over_m = species_config.charge / species_config.mass
    q_over_m = q_over_m.reshape((species_config.charge.shape[0], 1))
    update_x = species_config.update_x.reshape((species_config.update_x.shape[0], 1, 3))

    def push_one_tile(x_tile, u_tile, active_tile, tx, ty, tz):
        active = active_tile[..., jnp.newaxis]
        qom_tile = jnp.broadcast_to(q_over_m, active_tile.shape)
        metric_n = _sample_center_metric_at_position(
            x_tile,
            metric,
            static_parameters,
            dynamic_parameters,
            tx,
            ty,
            tz,
        )

        u_after_first_em = _electromagnetic_boris_step(
            x_tile,
            u_tile,
            qom_tile,
            D_tiles,
            B_tiles,
            metric,
            static_parameters,
            dynamic_parameters,
            tx,
            ty,
            tz,
            dt / 2.0,
        )
        u_after_first_em = jnp.where(active & update_x, u_after_first_em, u_tile)
        # a disabled direction freezes both its covariant velocity and coordinate

        du_dt_n = geodesic_velocity(x_tile, u_after_first_em, metric_n)
        u_geo_mid = u_after_first_em + 0.5 * dt * du_dt_n
        du_dt_mid = geodesic_velocity(x_tile, u_geo_mid, metric_n)
        u_after_geodesic = u_after_first_em + dt * du_dt_mid
        u_after_geodesic = jnp.where(active & update_x, u_after_geodesic, u_tile)
        # midpoint geodesic velocity source at x^n; positions remain staggered until the velocity update is complete.

        u_new = _electromagnetic_boris_step(
            x_tile,
            u_after_geodesic,
            qom_tile,
            D_tiles,
            B_tiles,
            metric,
            static_parameters,
            dynamic_parameters,
            tx,
            ty,
            tz,
            dt / 2.0,
        )
        u_new = jnp.where(active & update_x, u_new, u_tile)
        # second half of the electromagnetic Boris step, reinterpolated at the same x^n position.

        dx_dt_n = GR_position_update(
            x_tile,
            u_new,
            metric_n,
        )
        x_half = x_tile + 0.5 * dt * dx_dt_n
        x_half = jnp.where(active & update_x, x_half, x_tile)

        metric_half = _sample_center_metric_at_position(
            x_half,
            metric,
            static_parameters,
            dynamic_parameters,
            tx,
            ty,
            tz,
        )
        dx_dt_half = GR_position_update(
            x_half,
            u_new,
            metric_half,
        )
        x_new = x_tile + dt * dx_dt_half
        x_new = jnp.where(active & update_x, x_new, x_tile)
        # centered particles use x^{n+1/2}; x^{n+1} uses a metric/RHS sampled at that midpoint.

        return x_new, u_new, x_half

    tx, ty, tz = jnp.meshgrid(
        jnp.arange(ntx),
        jnp.arange(nty),
        jnp.arange(ntz),
        indexing="ij",
    )

    push_tiles = push_one_tile
    push_tiles = jax.vmap(push_tiles, in_axes=(0, 0, 0, 0, 0, 0), out_axes=0)
    push_tiles = jax.vmap(push_tiles, in_axes=(0, 0, 0, 0, 0, 0), out_axes=0)
    push_tiles = jax.vmap(push_tiles, in_axes=(0, 0, 0, 0, 0, 0), out_axes=0)

    x_new, u_new, x_half = push_tiles(
        particles.x,
        particles.u,
        particles.active,
        tx,
        ty,
        tz,
    )

    particles = TiledParticles(x=x_new, u=u_new, active=particles.active)
    # pack the tiled particles into a single TiledParticles object.
    particles_n_plushalf = TiledParticles(x=x_half, u=u_new, active=particles.active)
    # pack the intermediate particles into a single TiledParticles object for centered current deposition.

    return particles, particles_n_plushalf
