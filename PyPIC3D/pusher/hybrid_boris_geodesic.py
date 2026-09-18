from functools import partial

import jax
import jax.numpy as jnp

from PyPIC3D.particles.particle_class import TiledParticles
from PyPIC3D.particles.particle_batching import (
    prepare_particle_batches, particle_batch_indices, number_of_particle_batches,
)
from PyPIC3D.relativity.particle_metric import (
    sample_particle_metric, safe_inactive_positions, check_particle_samples,
    particle_metric_valid,
)
from PyPIC3D.pusher.boris import interpolate_field_to_particles
from PyPIC3D.relativity.cartesian_particle_metric import (
    sample_regularized_metric, spherical_to_cartesian, cartesian_to_spherical,
    covariant_to_cartesian, cartesian_to_covariant, vector_to_cartesian,
)
from PyPIC3D.relativity.core import (
    B_FIELD_LOCATIONS,
    D_FIELD_LOCATIONS,
    Metric,
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


def _sample_scalar(
    field,
    x,
    y,
    z,
    grid,
    shape_factor,
    active_axes,
    inactive_axis_indices,
):
    particle_shape = x.shape
    component_shape = field.shape[3:]
    sampled_field = interpolate_field_to_particles(
        field,
        x.reshape(-1),
        y.reshape(-1),
        z.reshape(-1),
        grid,
        shape_factor,
        ghost_cells=True,
        active_axes=active_axes,
        inactive_axis_indices=inactive_axis_indices,
    )
    return sampled_field.reshape(particle_shape + component_shape)


def _sample_vector(
    field,
    x,
    y,
    z,
    grids,
    shape_factor,
    active_axes,
    inactive_axis_indices,
):
    fields = jnp.stack(field, axis=0)
    x_grids = jnp.stack((grids[0][0], grids[1][0], grids[2][0]), axis=0)
    y_grids = jnp.stack((grids[0][1], grids[1][1], grids[2][1]), axis=0)
    z_grids = jnp.stack((grids[0][2], grids[1][2], grids[2][2]), axis=0)

    def sample_component(component_field, x_grid, y_grid, z_grid):
        return _sample_scalar(
            component_field,
            x,
            y,
            z,
            (x_grid, y_grid, z_grid),
            shape_factor,
            active_axes,
            inactive_axis_indices,
        )

    sampled_components = jax.vmap(sample_component)(
        fields,
        x_grids,
        y_grids,
        z_grids,
    )
    return jnp.moveaxis(sampled_components, 0, -1)


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


def _runtime_particle_metric(metric, position, grid, static_parameters,
                             active_axes, inactive_axis_indices, *, derivatives=True, check=True):
    if static_parameters.particle_coordinates == 'cartesian':
        return sample_regularized_metric(
            metric, position, grid, static_parameters.metric, active_axes, inactive_axis_indices,
            cartesian=True, derivatives=derivatives, check=check)
    return sample_particle_metric(
        metric, position, grid, static_parameters.shape_factor, static_parameters.metric,
        active_axes, inactive_axis_indices, derivatives=derivatives, check=check)


def GR_position_update(position, u_cov, metric):
    """
    Coordinate velocity dx^i/dt from covariant spatial momentum u_i.
    """

    del position
    Gamma = covariant_lorentz_factor(u_cov, metric.gamma_inv)
    u_con = jnp.einsum("...ij,...j->...i", metric.gamma_inv, u_cov)
    return metric.lapse[..., jnp.newaxis] * u_con / Gamma[..., jnp.newaxis] - metric.shift


def geodesic_velocity(position, u_cov, metric, grad_gamma_inv):
    """
    Geodesic source term du_i/dt for covariant spatial momentum.
    """

    del position
    Gamma = covariant_lorentz_factor(u_cov, metric.gamma_inv)
    grad_beta_term = jnp.einsum("...j,...ji->...i", u_cov, metric.grad_shift)
    metric_force = (-0.5 * metric.lapse / Gamma)[..., jnp.newaxis] * jnp.einsum(
        "...l,...m,...ilm->...i",
        u_cov,
        u_cov,
        grad_gamma_inv,
    )

    return -Gamma[..., jnp.newaxis] * metric.grad_lapse + grad_beta_term + metric_force


def _implicit_velocity_newton(old, guess, metric, grad_inverse, dt, iterations, active,
                              dynamic_iterations=False):
    """Damped Newton fallback for the same fixed-position midpoint equation.

    Picard iteration need not contract near a coordinate pole. This changes
    only the nonlinear solution method, not the equation or its acceptance
    tolerance. The caller still checks the final residual.
    """
    def residual(value):
        return value-old-dt*geodesic_velocity(None, .5*(old+value), metric, grad_inverse)

    def iteration(_, value):
        midpoint = .5*(old+value)
        gamma = covariant_lorentz_factor(midpoint, metric.gamma_inv)
        contravariant = jnp.einsum('...ij,...j->...i', metric.gamma_inv, midpoint)
        quadratic = jnp.einsum('...i,...kij,...j->...k', midpoint, grad_inverse, midpoint)
        derivative = (
            -metric.grad_lapse[..., :, None]*contravariant[..., None, :]/gamma[..., None, None]
            + jnp.swapaxes(metric.grad_shift, -1, -2)
            - metric.lapse[..., None, None]/gamma[..., None, None]
              * jnp.einsum('...ijk,...k->...ij', grad_inverse, midpoint)
            + (.5*metric.lapse/gamma**3)[..., None, None]
              * quadratic[..., :, None]*contravariant[..., None, :])
        jacobian = jnp.eye(3)-.5*dt*derivative
        initial_residual = residual(value)
        delta = jnp.linalg.solve(jacobian, initial_residual[..., None])[..., 0]
        initial_norm = jnp.max(jnp.abs(initial_residual), axis=-1)
        unresolved = active & (initial_norm > 1e-8*jnp.maximum(1., jnp.max(jnp.abs(value), axis=-1)))
        # Backtrack independently per particle and retain the best finite step.
        def trial(k, state):
            best, norm = state
            candidate = value-(.5**k)*delta
            candidate_norm = jnp.max(jnp.abs(residual(candidate)), axis=-1)
            better = unresolved & jnp.isfinite(candidate_norm) & (candidate_norm < norm)
            return jnp.where(better[..., None], candidate, best), jnp.where(better, candidate_norm, norm)
        return jax.lax.fori_loop(0, 8, trial, (value, initial_norm))[0]

    if dynamic_iterations:
        def unfinished(state):
            count, value = state
            norm = jnp.max(jnp.abs(residual(value)), axis=-1)
            scale = jnp.maximum(1., jnp.max(jnp.abs(value), axis=-1))
            return (count < iterations) & jnp.any(active & (norm > 1e-8*scale))
        return jax.lax.while_loop(
            unfinished, lambda state: (state[0]+1, iteration(state[0], state[1])), (0, guess))[1]
    # A static loop is needed with checkify outside the multiple-tile vmap.
    return jax.lax.fori_loop(0, iterations, iteration, guess)


def _implicit_position_newton(old, guess, momentum, sample, dt, iterations,
                              active, update_x, spacing, dynamic_iterations=False,
                              cyclic_coordinate=None):
    """Solve the unchanged coordinate midpoint equation with a line search.

    Trial metrics are validated before accepting a step; rejected line-search
    points are not physical particle states. The caller checks the accepted
    midpoint metric and final residual through the normal diagnostic path.
    A coordinate absent from the metric interpolant can be eliminated exactly:
    its midpoint equation gives that coordinate directly once the others are
    known. This prevents its large near-axis sensitivity from dominating the
    line search. The caller must establish independence of that coordinate.
    """
    def project(value):
        if cyclic_coordinate is None:
            return value
        midpoint = .5*(old+value)
        metric, _, _ = sample(midpoint, False)
        rhs = old+dt*GR_position_update(midpoint, momentum, metric)
        k = cyclic_coordinate
        component = jnp.where(update_x[..., k], rhs[..., k], old[..., k])
        return value.at[..., k].set(jnp.where(active, component, value[..., k]))

    def residual(value):
        midpoint = .5*(old+value)
        metric, _, valid = sample(midpoint, False)
        rhs = old+dt*GR_position_update(midpoint, momentum, metric)
        error = value-jnp.where(update_x, rhs, old)
        norm = jnp.max(jnp.abs(error)/spacing, axis=-1)
        return error, jnp.where(valid & jnp.isfinite(norm), norm, jnp.inf)

    def linearization(value):
        midpoint = .5*(old+value)
        metric, grad_inverse, _ = sample(midpoint, True)
        gamma = covariant_lorentz_factor(momentum, metric.gamma_inv)
        contravariant = jnp.einsum('...ij,...j->...i', metric.gamma_inv, momentum)
        quadratic = jnp.einsum('...jkl,...k,...l->...j', grad_inverse, momentum, momentum)
        derivative = (
            contravariant[..., :, None]*metric.grad_lapse[..., None, :]/gamma[..., None, None]
            + (metric.lapse/gamma)[..., None, None]
              * jnp.einsum('...jik,...k->...ij', grad_inverse, momentum)
            - (.5*metric.lapse/gamma**3)[..., None, None]
              * contravariant[..., :, None]*quadratic[..., None, :]
            - metric.grad_shift)
        jacobian = jnp.eye(3)-.5*dt*jnp.where(update_x[..., :, None], derivative, 0.)
        error, initial_norm = residual(value)
        delta = jnp.linalg.solve(jacobian, error[..., None])[..., 0]
        return delta, initial_norm

    def needs_correction(delta, norm):
        # Eliminating phi makes its residual identically zero, but small
        # errors in theta may still imply a large azimuth error. Require the
        # Newton correction in every coordinate to resolve that conditioning.
        estimate = (jnp.max(jnp.abs(delta)/spacing, axis=-1)
                    if cyclic_coordinate is not None else jnp.zeros_like(norm))
        return active & ((norm > 1e-8) | (estimate > 1e-8))

    def iteration(_, value):
        delta, initial_norm = linearization(value)
        unresolved = needs_correction(delta, initial_norm)
        def trial(k, state):
            best, norm = state
            candidate = project(value-(.5**k)*delta)
            _, candidate_norm = residual(candidate)
            better = unresolved & (candidate_norm < norm)
            return jnp.where(better[..., None], candidate, best), jnp.where(better, candidate_norm, norm)
        return jax.lax.fori_loop(0, 8, trial, (value, initial_norm))[0]

    guess = project(guess)
    if dynamic_iterations:
        def unfinished(state):
            count, value = state
            if cyclic_coordinate is not None:
                delta, norm = linearization(value)
                return (count < iterations) & jnp.any(needs_correction(delta, norm))
            _, norm = residual(value)
            return (count < iterations) & jnp.any(active & (norm > 1e-8))
        return jax.lax.while_loop(
            unfinished, lambda state: (state[0]+1, iteration(state[0], state[1])), (0, guess))[1]
    return jax.lax.fori_loop(0, iterations, iteration, guess)


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


def _electromagnetic_boris_step(
    position,
    u_cov,
    q_over_m,
    D_tiles,
    B_tiles,
    metric_tiles,
    static_parameters,
    dynamic_parameters,
    tx,
    ty,
    tz,
    dt,
    active_axes,
    inactive_axis_indices,
):
    shape_factor = static_parameters.shape_factor
    gather_position = (cartesian_to_spherical(position)
                       if static_parameters.particle_coordinates == 'cartesian' else position)
    x = gather_position[..., 0]
    y = gather_position[..., 1]
    z = gather_position[..., 2]

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
        active_axes,
        inactive_axis_indices,
    )
    B_con = _sample_vector(
        tuple(B_tiles[i][tx, ty, tz] for i in range(3)),
        x,
        y,
        z,
        B_grids,
        shape_factor,
        active_axes,
        inactive_axis_indices,
    )
    metric, _ = _runtime_particle_metric(
        _metric_tile(metric_tiles.center, tx, ty, tz), position, center_grid,
        static_parameters, active_axes, inactive_axis_indices, derivatives=False)
    if static_parameters.particle_coordinates == 'cartesian':
        D_con = vector_to_cartesian(gather_position, D_con)
        B_con = vector_to_cartesian(gather_position, B_con)
    E_cov = lower_vector(D_con, metric.gamma)

    u_minus = u_cov + (q_over_m * dt / 2.0)[..., jnp.newaxis] * metric.lapse[..., jnp.newaxis] * E_cov
    u_plus = _magnetic_boris_rotation(u_minus, B_con, metric, q_over_m, dt)
    u_new = u_plus + (q_over_m * dt / 2.0)[..., jnp.newaxis] * metric.lapse[..., jnp.newaxis] * E_cov

    return u_new


def _sample_center_metric_at_position(
    position,
    metric_tiles,
    static_parameters,
    dynamic_parameters,
    tx,
    ty,
    tz,
    active_axes,
    inactive_axis_indices,
    *,
    derivatives=True,
):
    shape_factor = static_parameters.shape_factor
    center_grid = _metric_component_grid(("C", "C", "C"), dynamic_parameters, tx, ty, tz)

    return _runtime_particle_metric(
        _metric_tile(metric_tiles.center, tx, ty, tz),
        position,
        center_grid,
        static_parameters,
        active_axes,
        inactive_axis_indices,
        derivatives=derivatives,
    )[0]


def _sample_center_grad_gamma_inv_at_position(
    position,
    metric_tiles,
    static_parameters,
    dynamic_parameters,
    tx,
    ty,
    tz,
    active_axes,
    inactive_axis_indices,
):
    shape_factor = static_parameters.shape_factor
    center_grid = _metric_component_grid(("C", "C", "C"), dynamic_parameters, tx, ty, tz)

    return _runtime_particle_metric(
        _metric_tile(metric_tiles.center, tx, ty, tz), position, center_grid,
        static_parameters, active_axes, inactive_axis_indices)[1]


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

    The optional Cartesian particle chart integrates the same equations in
    X=r*e_r, with a regularized supplied-grid metric reconstruction. Storage
    remains spherical: each staggered Cartesian covector is expressed in the
    spherical basis at its associated stored position. Thus full and centered
    particle momenta use different bases in that mode. Native checkpoints are
    not interchangeable with this representation.

    This is a staggered leapfrog.  The incoming ``particles.u`` is
    ``u^{n-1/2}``; the velocity operator ``EM(dt/2) . geodesic(dt) . EM(dt/2)``
    is applied with every field and metric quantity sampled at ``x^n``, giving
    ``u^{n+1/2}``, and the position is only advanced afterwards.  A run must
    therefore start from ``u^{-1/2}``, which
    :func:`PyPIC3D.pusher.particle_push.seed_leapfrog_velocity` provides;
    starting from the physical ``u(0)`` costs a full order of accuracy.

    Returns the full-step particles ``(x^{n+1}, u^{n+1/2})`` and the centred
    particles ``(x^{n+1/2}, u^{n+1/2})`` used for the current deposition.
    """

    tile_nx, tile_ny, tile_nz = tuple(
        int(width) for width in static_parameters.tile_shape
    )
    g = int(static_parameters.guard_cells)
    if g < 3:
        raise ValueError("Hybrid Hermite particle metrics require guard_cells >= 3")
    dt = dynamic_parameters.dt
    ntx, nty, ntz = particles.active.shape[:3]
    single_tile = (ntx, nty, ntz) == (1, 1, 1)
    active_axes = (
        int(ntx) * tile_nx > 1,
        int(nty) * tile_ny > 1,
        int(ntz) * tile_nz > 1,
    )
    if static_parameters.particle_coordinates == 'cartesian' and active_axes[2]:
        raise ValueError('Cartesian particle chart currently requires an axisymmetric field grid')
    # A width-one local tile remains physical when other tiles extend the axis.
    inactive_axis_indices = (g, g, g)
    q_over_m = species_config.charge / species_config.mass

    def push_active_batch(x_tile, u_tile, active_tile, qom_tile, update_x, tx, ty, tz):
        active = active_tile[..., jnp.newaxis]
        original_x, original_u = x_tile, u_tile
        center_grid = _metric_component_grid(("C", "C", "C"), dynamic_parameters, tx, ty, tz)
        x_tile = safe_inactive_positions(x_tile, active_tile, center_grid, active_axes, g)
        u_tile = jnp.where(active, u_tile, 0.)
        old_coordinate_position = x_tile
        if static_parameters.particle_coordinates == 'cartesian':
            check_particle_samples(~active_tile | jnp.all(update_x, axis=-1), x_tile,
                                   'Cartesian particle chart requires all coordinate updates', jnp.array([tx,ty,tz]))
            u_tile = covariant_to_cartesian(x_tile, u_tile)
            x_tile = spherical_to_cartesian(x_tile)
        metric_n = _sample_center_metric_at_position(
            x_tile,
            metric,
            static_parameters,
            dynamic_parameters,
            tx,
            ty,
            tz,
            active_axes,
            inactive_axis_indices,
        )
        grad_gamma_inv_n = _sample_center_grad_gamma_inv_at_position(
            x_tile,
            metric,
            static_parameters,
            dynamic_parameters,
            tx,
            ty,
            tz,
            active_axes,
            inactive_axis_indices,
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
            active_axes,
            inactive_axis_indices,
        )
        u_after_first_em = jnp.where(active & update_x, u_after_first_em, u_tile)
        # a disabled direction freezes both its covariant velocity and coordinate

        du_dt_n = geodesic_velocity(
            x_tile,
            u_after_first_em,
            metric_n,
            grad_gamma_inv_n,
        )
        u_geo_mid = u_after_first_em + 0.5 * dt * du_dt_n
        du_dt_mid = geodesic_velocity(
            x_tile,
            u_geo_mid,
            metric_n,
            grad_gamma_inv_n,
        )
        u_after_geodesic = u_after_first_em + dt * du_dt_mid
        if static_parameters.geodesic_iterations:
            # Entity II implicit midpoint at fixed x^n, iterated from u_old.
            def velocity_iteration(_, guess):
                midpoint = .5*(u_after_first_em+guess)
                return u_after_first_em + dt*geodesic_velocity(
                    x_tile, midpoint, metric_n, grad_gamma_inv_n)
            u_after_geodesic = jax.lax.fori_loop(
                0, static_parameters.geodesic_iterations, velocity_iteration, u_after_first_em)
            picard_residual = u_after_geodesic-velocity_iteration(0, u_after_geodesic)
            picard_scale = jnp.maximum(1., jnp.max(jnp.abs(u_after_geodesic), axis=-1))
            needs_newton = active_tile & (jnp.max(jnp.abs(picard_residual), axis=-1) > 1e-8*picard_scale)
            u_after_geodesic = jax.lax.cond(
                jnp.any(needs_newton),
                lambda value: _implicit_velocity_newton(
                    u_after_first_em, value, metric_n, grad_gamma_inv_n, dt,
                    static_parameters.geodesic_iterations, needs_newton, single_tile),
                lambda value: value, u_after_geodesic)
            u_geo_mid = .5*(u_after_first_em+u_after_geodesic)
            residual = u_after_geodesic-velocity_iteration(0, u_after_geodesic)
            scale = jnp.maximum(1., jnp.max(jnp.abs(u_after_geodesic), axis=-1))
            check_particle_samples(~active_tile | (jnp.max(jnp.abs(residual), axis=-1) <= 1e-8*scale),
                                   x_tile, 'implicit velocity convergence', jnp.array([tx,ty,tz]))
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
            active_axes,
            inactive_axis_indices,
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

        if static_parameters.geodesic_iterations:
            def position_iteration(_, guess):
                midpoint = .5*(x_tile+guess)
                midpoint_metric = _sample_center_metric_at_position(
                    midpoint, metric, static_parameters, dynamic_parameters,
                    tx, ty, tz, active_axes, inactive_axis_indices, derivatives=False)
                candidate = x_tile + dt*GR_position_update(midpoint, u_new, midpoint_metric)
                return jnp.where(active & update_x, candidate, x_tile)
            spacing = jnp.array([dynamic_parameters.dx,dynamic_parameters.dy,dynamic_parameters.dz])
            if static_parameters.particle_coordinates == 'cartesian':
                spacing = jnp.full((3,), dynamic_parameters.dx)
            x_new = jax.lax.fori_loop(
                0, static_parameters.geodesic_iterations, position_iteration, x_tile)
            residual = x_new-position_iteration(0, x_new)
            needs_newton = active_tile & (jnp.max(jnp.abs(residual)/spacing, axis=-1) > 1e-8)
            def position_sample(position, derivatives):
                sampled, gradient = _runtime_particle_metric(
                    _metric_tile(metric.center, tx, ty, tz), position, center_grid,
                    static_parameters,
                    active_axes, inactive_axis_indices, derivatives=derivatives, check=False)
                name = ('flat_cartesian' if static_parameters.particle_coordinates == 'cartesian'
                        else static_parameters.metric)
                valid = particle_metric_valid(sampled, position, name)
                return sampled, gradient, valid
            x_new = jax.lax.cond(
                jnp.any(needs_newton),
                lambda value: _implicit_position_newton(
                    x_tile, value, u_new, position_sample, dt,
                    static_parameters.geodesic_iterations, needs_newton, update_x, spacing, single_tile),
                lambda value: value, x_new)
            residual = x_new-position_iteration(0, x_new)
            if (static_parameters.particle_coordinates == 'native' and not active_axes[2]
                    and static_parameters.metric in ('flat_spherical', 'kerr_schild_spherical')):
                # An unconverged Picard/Newton sequence can land beyond the
                # polar singularity. Retry from the old position in its chart,
                # eliminating cyclic phi from the line search exactly. Bounds
                # constrain trial midpoints, not accepted particle endpoints;
                # endpoints may still cross a pole and undergo normal reflection.
                retry = active_tile & (jnp.max(jnp.abs(residual)/spacing, axis=-1) > 1e-8)
                def chart_sample(position, derivatives):
                    sampled, gradient, valid = position_sample(position, derivatives)
                    valid &= (position[..., 1] > 0.) & (position[..., 1] < jnp.pi)
                    return sampled, gradient, valid
                x_new = jax.lax.cond(
                    jnp.any(retry),
                    lambda value: _implicit_position_newton(
                        x_tile, jnp.where(retry[..., None], x_tile, value), u_new,
                        chart_sample, dt, static_parameters.geodesic_iterations,
                        retry, update_x, spacing, single_tile, cyclic_coordinate=2),
                    lambda value: value, x_new)
                residual = x_new-position_iteration(0, x_new)
            check_particle_samples(~active_tile | (jnp.max(jnp.abs(residual)/spacing, axis=-1) <= 1e-8),
                                   x_tile, 'implicit position convergence', jnp.array([tx,ty,tz]))
            x_half = .5*(x_tile+x_new)
        else:
            metric_half = _sample_center_metric_at_position(
                x_half, metric, static_parameters, dynamic_parameters,
                tx, ty, tz, active_axes, inactive_axis_indices, derivatives=False)
            x_new = x_tile + dt*GR_position_update(x_half, u_new, metric_half)
        x_new = jnp.where(active & update_x, x_new, x_tile)
        for stage, value in (("first magnetic half-step", u_after_first_em),
                             ("geodesic midpoint", u_geo_mid),
                             ("geodesic full-step", u_after_geodesic),
                             ("second magnetic half-step", u_new),
                             ("position midpoint", x_half), ("position full-step", x_new)):
            check_particle_samples(~active_tile | jnp.all(jnp.isfinite(value),axis=-1),
                                   x_tile, stage, jnp.array([tx,ty,tz]))
        if static_parameters.particle_coordinates == 'cartesian':
            x_new = cartesian_to_spherical(x_new, old_coordinate_position[..., 2])
            x_half = cartesian_to_spherical(x_half, old_coordinate_position[..., 2])
            u_new = cartesian_to_covariant(x_new, u_new)
        displacement = jnp.abs(x_new-old_coordinate_position)/jnp.array([dynamic_parameters.dx,dynamic_parameters.dy,dynamic_parameters.dz])
        supported = jnp.all(jnp.where(jnp.array(active_axes), displacement <= 1, True),axis=-1)
        check_particle_samples(~active_tile | supported, x_tile,
                               "one-cell displacement bound", jnp.array([tx,ty,tz]))
        # centered particles use x^{n+1/2}; x^{n+1} uses a metric/RHS sampled at that midpoint.

        return (jnp.where(active, x_new, original_x),
                jnp.where(active, u_new, original_u),
                jnp.where(active, x_half, original_x))

    capacity = particles.active.shape[-2] * particles.active.shape[-1]
    if capacity == 0:
        return particles, particles
    batch_size = min(int(static_parameters.particle_batch_size), capacity)
    slots = particles.active.shape[-1]
    tile_shape = particles.active.shape[:3]
    x_flat = particles.x.reshape(tile_shape + (capacity, 3))
    u_flat = particles.u.reshape(tile_shape + (capacity, 3))

    def map_tiles(function):
        if tile_shape == (1, 1, 1):
            # vmap turns a particle-batch conditional into selection and runs
            # both branches. A local single tile can retain the actual Newton
            # fallback conditional instead of solving already converged batches.
            def one_tile(*args):
                result = function(*(value[0, 0, 0] for value in args))
                return jax.tree.map(lambda value: value[None, None, None], result)
            return one_tile
        for _ in range(3):
            function = jax.vmap(function)
        return function

    def prepare(active):
        _, _, indices, count = prepare_particle_batches(
            active, static_parameters.particle_batch_size
        )
        return indices, count

    indices, counts = map_tiles(prepare)(particles.active)
    n_batches = jnp.max(number_of_particle_batches(counts, batch_size))
    tx, ty, tz = jnp.meshgrid(jnp.arange(ntx), jnp.arange(nty), jnp.arange(ntz), indexing="ij")

    def push_selected(index_array, count, x, u, tx, ty, tz, batch_index):
        selected, valid = particle_batch_indices(index_array, count, batch_index, batch_size)
        species = selected // slots
        new_x, new_u, mid_x = push_active_batch(
            x[selected], u[selected], valid, q_over_m[species],
            species_config.update_x[species], tx, ty, tz,
        )
        return jnp.where(valid, selected, capacity), new_x, new_u, mid_x

    def write_tile(destination, values, original):
        # Padded lanes must not overwrite their repeated live source index.
        return original.at[destination].set(values, mode="drop")

    def advance_batch(state):
        batch_index, full_x, full_u, half_x = state
        destination, new_x, new_u, mid_x = map_tiles(push_selected)(
            indices, counts, x_flat, u_flat, tx, ty, tz,
            jnp.broadcast_to(batch_index, tile_shape),
        )
        return (batch_index + 1,
                map_tiles(write_tile)(destination, new_x, full_x),
                map_tiles(write_tile)(destination, new_u, full_u),
                map_tiles(write_tile)(destination, mid_x, half_x))

    # Keep the dynamic loop outside vmap: checkify rejects vmap-of-while.
    # Tiles share the maximum active batch count; padded lanes use safe metrics.
    _, x_new, u_new, x_half = jax.lax.while_loop(
        lambda state: state[0] < n_batches,
        advance_batch, (jnp.asarray(0), x_flat, u_flat, x_flat),
    )
    x_new = x_new.reshape(particles.x.shape)
    u_new = u_new.reshape(particles.u.shape)
    x_half = x_half.reshape(particles.x.shape)

    particles = TiledParticles(x=x_new, u=u_new, active=particles.active)
    # pack the tiled particles into a single TiledParticles object.
    half_u = u_new
    if static_parameters.particle_coordinates == 'cartesian':
        # Each stored momentum is expressed in the basis at its own stored
        # position; the staggered Cartesian covector is common to both states.
        half_u = jnp.where(particles.active[..., None], cartesian_to_covariant(
            x_half, covariant_to_cartesian(x_new, u_new)), u_new)
    particles_n_plushalf = TiledParticles(x=x_half, u=half_u, active=particles.active)
    # pack the intermediate particles into a single TiledParticles object for centered current deposition.

    return particles, particles_n_plushalf
