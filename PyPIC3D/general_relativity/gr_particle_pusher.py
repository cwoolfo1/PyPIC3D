import jax
from jax import jit
import jax.numpy as jnp

from PyPIC3D.boris import interpolate_field_to_particles

GR_PUSHER_NITER = 8
GR_EPS = 1e-30


def _interp_vec_field_to_particles(vec_field, x, y, z, grid, shape_factor):
    return jnp.stack(
        [
            interpolate_field_to_particles(vec_field[0], x, y, z, grid, shape_factor),
            interpolate_field_to_particles(vec_field[1], x, y, z, grid, shape_factor),
            interpolate_field_to_particles(vec_field[2], x, y, z, grid, shape_factor),
        ],
        axis=1,
    )


def _interp_mat_field_to_particles(mat_field, x, y, z, grid, shape_factor):
    return jnp.stack(
        [
            jnp.stack(
                [
                    interpolate_field_to_particles(mat_field[i, j], x, y, z, grid, shape_factor)
                    for j in range(3)
                ],
                axis=1,
            )
            for i in range(3)
        ],
        axis=1,
    )


def _interp_rank3_field_to_particles(rank3_field, x, y, z, grid, shape_factor):
    return jnp.stack(
        [
            jnp.stack(
                [
                    jnp.stack(
                        [
                            interpolate_field_to_particles(
                                rank3_field[i, j, k], x, y, z, grid, shape_factor
                            )
                            for k in range(3)
                        ],
                        axis=1,
                    )
                    for j in range(3)
                ],
                axis=1,
            )
            for i in range(3)
        ],
        axis=1,
    )


@jit
def relativistic_metric_single_particle(
    vx,
    vy,
    vz,
    Dxfield,
    Dyfield,
    Dzfield,
    Bxfield,
    Byfield,
    Bzfield,
    h_interp,
    alpha_interp,
    beta_interp,
    grad_alpha_interp,
    grad_beta_interp,
    grad_h_interp,
    q,
    m,
    dt,
    C,
):
    D_ = jnp.array([Dxfield, Dyfield, Dzfield])
    B_ = jnp.array([Bxfield, Byfield, Bzfield])

    # Guard against superluminal numerical noise in the input velocity.
    v2 = vx * vx + vy * vy + vz * vz
    v2 = jnp.minimum(v2, (1.0 - 1e-12) * (C * C))
    gamma = 1.0 / jnp.sqrt(1.0 - v2 / (C * C))
    u_init = gamma * jnp.array([vx, vy, vz])

    det_h = jnp.linalg.det(h_interp)
    sqrt_det_h = jnp.sqrt(jnp.maximum(det_h, GR_EPS))
    alpha_safe = jnp.maximum(alpha_interp, GR_EPS)
    E_cov = h_interp @ D_

    def em_half_push(u_cov):
        # Metric-aware Boris-like half push (Strang split half-step).
        u_minus = u_cov + (q * dt / (4.0 * m)) * alpha_safe * E_cov
        gamma_minus = jnp.sqrt(1.0 + (u_minus @ (h_interp @ u_minus)) / (C * C))
        B_phys = (alpha_safe / sqrt_det_h) * B_
        t = (q * dt / (4.0 * m * gamma_minus)) * B_phys
        s = 2.0 * t / (1.0 + jnp.dot(t, t))
        u_prime = u_minus + jnp.cross(u_minus, t)
        u_plus = u_minus + jnp.cross(u_prime, s)
        return u_plus + (q * dt / (4.0 * m)) * alpha_safe * E_cov

    # EM half push.
    u_em = em_half_push(u_init)

    # Iterative midpoint geodesic push (fixed-point/Newton-like update).
    u_geo = u_em
    for _ in range(GR_PUSHER_NITER):
        u_mid = 0.5 * (u_em + u_geo)
        gamma_mid = jnp.sqrt(1.0 + (u_mid @ (h_interp @ u_mid)) / (C * C))
        u0_mid = gamma_mid / alpha_safe
        geo_force = -C * alpha_safe * u0_mid * grad_alpha_interp
        geo_force = geo_force + C * jnp.einsum("j,ij->i", u_mid, grad_beta_interp)
        geo_force = geo_force - C * (alpha_safe / (2.0 * u0_mid)) * jnp.einsum(
            "j,k,ijk->i", u_mid, u_mid, grad_h_interp
        )
        u_geo = u_em + dt * geo_force

    # Final EM half push.
    newu = em_half_push(u_geo)

    new_gamma = jnp.sqrt(1.0 + (newu @ (h_interp @ newu)) / (C * C))
    newv = newu / new_gamma
    return newv[0], newv[1], newv[2]


@jit
def particle_push_relativistic_metric(particles, D, B, world, constants):
    q = particles.get_charge()
    m = particles.get_mass()
    x, y, z = particles.get_forward_position()
    particles.set_previous_forward_position(x, y, z)
    vx, vy, vz = particles.get_velocity()
    shape_factor = particles.get_shape()

    # Match SR Yee staggering: components live on face-centered grids.
    grid = world["grids"]["center"]
    staggered_grid = world["grids"]["vertex"]
    dt = world["dt"]
    dx, dy, dz = world["dx"], world["dy"], world["dz"]
    metric = world["metric"]

    Ex_grid = (staggered_grid[0], grid[1], grid[2])
    Ey_grid = (grid[0], staggered_grid[1], grid[2])
    Ez_grid = (grid[0], grid[1], staggered_grid[2])
    Bx_grid = (grid[0], staggered_grid[1], staggered_grid[2])
    By_grid = (staggered_grid[0], grid[1], staggered_grid[2])
    Bz_grid = (staggered_grid[0], staggered_grid[1], grid[2])

    Dx, Dy, Dz = D
    Bx, By, Bz = B

    Dxfield_atp = interpolate_field_to_particles(Dx, x, y, z, Ex_grid, shape_factor)
    Dyfield_atp = interpolate_field_to_particles(Dy, x, y, z, Ey_grid, shape_factor)
    Dzfield_atp = interpolate_field_to_particles(Dz, x, y, z, Ez_grid, shape_factor)
    Bxfield_atp = interpolate_field_to_particles(Bx, x, y, z, Bx_grid, shape_factor)
    Byfield_atp = interpolate_field_to_particles(By, x, y, z, By_grid, shape_factor)
    Bzfield_atp = interpolate_field_to_particles(Bz, x, y, z, Bz_grid, shape_factor)

    g00 = metric[0, 0]
    beta_cov = metric[0, 1:4]
    h_ = metric[1:4, 1:4]

    h_t = h_.transpose((2, 3, 4, 0, 1))
    h_inv_t = jnp.linalg.inv(h_t)
    h_inv = h_inv_t.transpose((3, 4, 0, 1, 2))
    beta = jnp.einsum("ij...,j...->i...", h_inv, beta_cov)
    beta_sq = jnp.einsum("i...,i...->...", beta_cov, beta)
    alpha = jnp.sqrt(jnp.maximum(beta_sq - g00, GR_EPS)) / constants["C"]

    dalpha_dx = (jnp.roll(alpha, shift=-1, axis=0) - jnp.roll(alpha, shift=1, axis=0)) / (
        2.0 * dx
    )
    dalpha_dy = (jnp.roll(alpha, shift=-1, axis=1) - jnp.roll(alpha, shift=1, axis=1)) / (
        2.0 * dy
    )
    dalpha_dz = (jnp.roll(alpha, shift=-1, axis=2) - jnp.roll(alpha, shift=1, axis=2)) / (
        2.0 * dz
    )
    grad_alpha = jnp.stack([dalpha_dx, dalpha_dy, dalpha_dz], axis=0)

    grad_beta = jnp.stack(
        [
            (jnp.roll(beta, shift=-1, axis=1) - jnp.roll(beta, shift=1, axis=1)) / (2.0 * dx),
            (jnp.roll(beta, shift=-1, axis=2) - jnp.roll(beta, shift=1, axis=2)) / (2.0 * dy),
            (jnp.roll(beta, shift=-1, axis=3) - jnp.roll(beta, shift=1, axis=3)) / (2.0 * dz),
        ],
        axis=0,
    )

    grad_h = jnp.stack(
        [
            (jnp.roll(h_, shift=-1, axis=2) - jnp.roll(h_, shift=1, axis=2)) / (2.0 * dx),
            (jnp.roll(h_, shift=-1, axis=3) - jnp.roll(h_, shift=1, axis=3)) / (2.0 * dy),
            (jnp.roll(h_, shift=-1, axis=4) - jnp.roll(h_, shift=1, axis=4)) / (2.0 * dz),
        ],
        axis=2,
    )

    h_atp = _interp_mat_field_to_particles(h_, x, y, z, grid, shape_factor)
    alpha_atp = interpolate_field_to_particles(alpha, x, y, z, grid, shape_factor)
    beta_atp = _interp_vec_field_to_particles(beta, x, y, z, grid, shape_factor)
    grad_alpha_atp = _interp_vec_field_to_particles(grad_alpha, x, y, z, grid, shape_factor)
    grad_beta_atp = _interp_mat_field_to_particles(grad_beta, x, y, z, grid, shape_factor)
    grad_h_atp = _interp_rank3_field_to_particles(grad_h, x, y, z, grid, shape_factor)

    C = constants["C"]
    metric_push_vmap = jax.vmap(
        relativistic_metric_single_particle,
        in_axes=(
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            None,
            None,
            None,
            None,
        ),
    )
    newvx, newvy, newvz = metric_push_vmap(
        vx,
        vy,
        vz,
        Dxfield_atp,
        Dyfield_atp,
        Dzfield_atp,
        Bxfield_atp,
        Byfield_atp,
        Bzfield_atp,
        h_atp,
        alpha_atp,
        beta_atp,
        grad_alpha_atp,
        grad_beta_atp,
        grad_h_atp,
        q,
        m,
        dt,
        C,
    )

    particles.set_velocity(newvx, newvy, newvz)

    # Coordinate push in 3+1 split (flat space gives x += dt * v).
    dx_dt = C * (alpha_atp * newvx - beta_atp[:, 0])
    dy_dt = C * (alpha_atp * newvy - beta_atp[:, 1])
    dz_dt = C * (alpha_atp * newvz - beta_atp[:, 2])

    particles.set_position(x + dt * dx_dt, y + dt * dy_dt, z + dt * dz_dt)
    return particles
