import jax
from jax import jit
import jax.numpy as jnp

from PyPIC3D.boris import interpolate_field_to_particles


@jit
def relativistic_metric_single_particle(vx, vy, vz, x, y, z, Dxfield, Dyfield, Dzfield, Bxfield, Byfield, Bzfield, q, m, dt, constants, metric_interp, grad_alpha_interp, grad_beta_interp, grad_h_interp):
    
    D_ = jnp.array([Dxfield, Dyfield, Dzfield])
    B_ = jnp.array([Bxfield, Byfield, Bzfield])
    # pack the interpolated fields into vectors for the particle push
    C = constants["C"]
    gamma = 1.0 / jnp.sqrt(1.0 - (vx**2 + vy**2 + vz**2) / C**2)
    # compute the Lorentz factor based on the current velocity of the particle

    u = gamma * jnp.array([vx, vy, vz])
    # compute the relativistic velocity u = gamma * v

    # vminus = v + q*dt/(2*m)*jnp.array([efield_atx, efield_aty, efield_atz])
    vminus = u + (q * (dt/2) / m ) * jnp.einsum("ij, j->i", metric_interp, D_)
    # initial half-acceleration step using the interpolated metric and electric field for the first lorentz push
    vminus = vminus - C * gamma * grad_alpha_interp * (dt/2)
    # apply the geodesic term from the lapse gradient to the half-velocity
    vminus = vminus + C * dt/2 * jnp.einsum("j, ij -> i", u, grad_beta_interp)
    # apply the geodesic term from the shift gradient to the half-velocity
    vminus = vminus - C * dt/2 * metric_interp[0, 0, ...] / (2*gamma) * jnp.einsum("j...,k...,ijk", u, u, grad_h_interp)
    # apply the geodesic term from the spatial metric gradient to the half-velocity

    det_h = jnp.linalg.det(metric_interp[1:, 1:, ...])
    sqrt_det_h = jnp.sqrt(det_h)
    
    levi_civita_tensor = jnp.zeros((3, 3, 3))
    levi_civita_tensor = levi_civita_tensor.at[0, 1, 2].set(1)
    levi_civita_tensor = levi_civita_tensor.at[1, 2, 0].set(1)
    levi_civita_tensor = levi_civita_tensor.at[2, 0, 1].set(1)
    levi_civita_tensor = levi_civita_tensor.at[0, 2, 1].set(-1)
    levi_civita_tensor = levi_civita_tensor.at[2, 1, 0].set(-1)
    levi_civita_tensor = levi_civita_tensor.at[1, 0, 2].set(-1)
    # construct the Levi-Civita symbol for the magnetic force calculation (3D antisymmetric tensor)
    
    B_term = q / m / sqrt_det_h / gamma * metric_interp[0, 0, ...] * jnp.einsum("ijk, jl..., l..., k... -> i", levi_civita_tensor, metric_interp[1:, 1:, ...], vminus, B_)
    # compute the magnetic part of the Lorentz force using the metric and the interpolated magnetic
    vplus = vminus + B_term

    newu = vplus + (q * (dt/2) / m ) * jnp.einsum("ij, j->i", metric_interp, D_)
    # final half-acceleration step to get the updated relativistic velocity u = gamma * v after the full Boris push
    newu = newu - C * gamma * grad_alpha_interp * (dt/2)
    # apply the geodesic term from the lapse gradient to the final velocity
    newu = newu + C * dt/2 * jnp.einsum("j, ij -> i", vplus, grad_beta_interp)
    # apply the geodesic term from the shift gradient to the final velocity
    newu = newu - C * dt/2 * metric_interp[0, 0, ...] / (2*gamma) * jnp.einsum("j...,k...,ijk", vplus, vplus, grad_h_interp)
    # apply the geodesic term from the spatial metric gradient to the final velocity

    new_gamma = jnp.sqrt(1.0 + (newu @ (metric_interp[1:, 1:, ...] @ newu)) / C**2)
    newu = newu / new_gamma
    # compute the new Lorentz factor based on the updated relativistic velocity, and then convert back to coordinate velocity by dividing by the new gamma factor to ensure the final velocity is subluminal

    return newu[0], newu[1], newu[2]

@jit
def particle_push_relativistic_metric(particles, D, B, world, constants):
    """
    Relativistic particle pusher that augments the EOM with geodesic terms
    from the supplied metric tensor.
    """
    q = particles.get_charge()
    m = particles.get_mass()
    x, y, z = particles.get_forward_position()
    vx, vy, vz = particles.get_velocity()
    shape_factor = particles.get_shape()

    grid = world["grids"]["vertex"]
    staggered_grid = world["grids"]["center"]
    # Define the grid locations for each field component based on the Yee cell staggering.
    dt = world["dt"]
    dx, dy, dz = world["dx"], world["dy"], world["dz"]
    metric = world["metric"]
    # get the time resolution and metric tensor

    Ex_grid = staggered_grid[0], grid[1], grid[2]
    Ey_grid = grid[0], staggered_grid[1], grid[2]
    Ez_grid = grid[0], grid[1], staggered_grid[2]
    Bx_grid = grid[0], staggered_grid[1], staggered_grid[2]
    By_grid = staggered_grid[0], grid[1], staggered_grid[2]
    Bz_grid = staggered_grid[0], staggered_grid[1], grid[2]

    Dx, Dy, Dz = D
    Bx, By, Bz = B

    Dxfield_atx = interpolate_field_to_particles(Dx, x, y, z, Ex_grid, shape_factor)
    Dyfield_aty = interpolate_field_to_particles(Dy, x, y, z, Ey_grid, shape_factor)
    Dzfield_atz = interpolate_field_to_particles(Dz, x, y, z, Ez_grid, shape_factor)
    Bxfield_atx = interpolate_field_to_particles(Bx, x, y, z, Bx_grid, shape_factor)
    Byfield_aty = interpolate_field_to_particles(By, x, y, z, By_grid, shape_factor)
    Bzfield_atz = interpolate_field_to_particles(Bz, x, y, z, Bz_grid, shape_factor)


    # Calculate gradients of lapse, shift, and spatial metric
    alpha = metric[0, 0, ...]
    beta = metric[0, 1:, ...]
    h_   = metric[1:, 1:, ...]

    dalpha_dx = (jnp.roll(alpha, shift=-1, axis=0) - jnp.roll(alpha, shift=1, axis=0)) / (2 * dx)
    dalpha_dy = (jnp.roll(alpha, shift=-1, axis=1) - jnp.roll(alpha, shift=1, axis=1)) / (2 * dy)
    dalpha_dz = (jnp.roll(alpha, shift=-1, axis=2) - jnp.roll(alpha, shift=1, axis=2)) / (2 * dz)
    # centered finite difference gradients of the lapse function

    dbetax_dx = (jnp.roll(beta[0,...], shift=-1, axis=0) - jnp.roll(beta[0,...], shift=1, axis=0)) / (2 * dx)
    dbetax_dy = (jnp.roll(beta[0,...], shift=-1, axis=1) - jnp.roll(beta[0,...], shift=1, axis=1)) / (2 * dy)
    dbetax_dz = (jnp.roll(beta[0,...], shift=-1, axis=2) - jnp.roll(beta[0,...], shift=1, axis=2)) / (2 * dz)

    dbetay_dx = (jnp.roll(beta[1,...], shift=-1, axis=0) - jnp.roll(beta[1,...], shift=1, axis=0)) / (2 * dx)
    dbetay_dy = (jnp.roll(beta[1,...], shift=-1, axis=1) - jnp.roll(beta[1,...], shift=1, axis=1)) / (2 * dy)
    dbetay_dz = (jnp.roll(beta[1,...], shift=-1, axis=2) - jnp.roll(beta[1,...], shift=1, axis=2)) / (2 * dz)

    dbetaz_dx = (jnp.roll(beta[2,...], shift=-1, axis=0) - jnp.roll(beta[2,...], shift=1, axis=0)) / (2 * dx)
    dbetaz_dy = (jnp.roll(beta[2,...], shift=-1, axis=1) - jnp.roll(beta[2,...], shift=1, axis=1)) / (2 * dy)
    dbetaz_dz = (jnp.roll(beta[2,...], shift=-1, axis=2) - jnp.roll(beta[2,...], shift=1, axis=2)) / (2 * dz)

    grad_alpha = jnp.array([dalpha_dx, dalpha_dy, dalpha_dz])
    # centered finite difference gradient of the lapse function as a 3-component vector field
    grad_betax = jnp.array([dbetax_dx, dbetax_dy, dbetax_dz])
    grad_betay = jnp.array([dbetay_dx, dbetay_dy, dbetay_dz])
    grad_betaz = jnp.array([dbetaz_dx, dbetaz_dy, dbetaz_dz])
    # centered finite difference gradients of the shift vector components
    dh_dx = (jnp.roll(h_, shift=-1, axis=2) - jnp.roll(h_, shift=1, axis=2)) / (2 * dx)
    dh_dy = (jnp.roll(h_, shift=-1, axis=3) - jnp.roll(h_, shift=1, axis=3)) / (2 * dy)
    dh_dz = (jnp.roll(h_, shift=-1, axis=4) - jnp.roll(h_, shift=1, axis=4)) / (2 * dz)
    # centered finite difference gradients of the spatial metric components (3,3,nx,ny,nz)

    dalpha_dx_at_particle = interpolate_field_to_particles(dalpha_dx, x, y, z, grid, shape_factor)
    dalpha_dy_at_particle = interpolate_field_to_particles(dalpha_dy, x, y, z, grid, shape_factor)
    dalpha_dz_at_particle = interpolate_field_to_particles(dalpha_dz, x, y, z, grid, shape_factor)

    dbetax_dx_at_particle = interpolate_field_to_particles(dbetax_dx, x, y, z, grid, shape_factor)
    dbetax_dy_at_particle = interpolate_field_to_particles(dbetax_dy, x, y, z, grid, shape_factor)
    dbetax_dz_at_particle = interpolate_field_to_particles(dbetax_dz, x, y, z, grid, shape_factor)

    dbetay_dx_at_particle = interpolate_field_to_particles(dbetay_dx, x, y, z, grid, shape_factor)
    dbetay_dy_at_particle = interpolate_field_to_particles(dbetay_dy, x, y, z, grid, shape_factor)
    dbetay_dz_at_particle = interpolate_field_to_particles(dbetay_dz, x, y, z, grid, shape_factor)

    dbetaz_dx_at_particle = interpolate_field_to_particles(dbetaz_dx, x, y, z, grid, shape_factor)
    dbetaz_dy_at_particle = interpolate_field_to_particles(dbetaz_dy, x, y, z, grid, shape_factor)
    dbetaz_dz_at_particle = interpolate_field_to_particles(dbetaz_dz, x, y, z, grid, shape_factor)

    dh_dx_at_particle = interpolate_field_to_particles(dh_dx, x, y, z, grid, shape_factor)
    dh_dy_at_particle = interpolate_field_to_particles(dh_dy, x, y, z, grid, shape_factor)
    dh_dz_at_particle = interpolate_field_to_particles(dh_dz, x, y, z, grid, shape_factor)

    interp_grad_alpha = jnp.array([dalpha_dx_at_particle, dalpha_dy_at_particle, dalpha_dz_at_particle])
    interp_grad_beta  = jnp.array([dbetax_dx_at_particle, dbetax_dy_at_particle, dbetax_dz_at_particle,
                                  dbetay_dx_at_particle, dbetay_dy_at_particle, dbetay_dz_at_particle,
                                  dbetaz_dx_at_particle, dbetaz_dy_at_particle, dbetaz_dz_at_particle]).reshape(3, 3, -1)
    interp_dh = jnp.array([dh_dx_at_particle, dh_dy_at_particle, dh_dz_at_particle]).reshape(3, 3, 3, -1)
    # interpolate the metric gradients to the particle positions

    metric_push_vmap = jax.vmap(
        relativistic_metric_single_particle,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None, None),
    )
    newvx, newvy, newvz = metric_push_vmap(
        vx, vy, vz, x, y, z,
        Dxfield_atx, Dyfield_aty, Dzfield_atz,
        Bxfield_atx, Byfield_aty, Bzfield_atz,
        q, m, dt, constants, metric
    )

    particles.set_velocity(newvx, newvy, newvz)
    # set the new velocities of the particles

    ############################# POSITION UPDATE WITH METRIC PUSH ##################################################

    alpha = metric[0, 0, ...]
    beta = metric[0, 1:, ...]
    h_   = metric[1:, 1:, ...]
    hinv_ = jnp.linalg.inv(h_)
    # Update positions using the new velocities and the metric's shift and lapse.

    betax_at_particle = interpolate_field_to_particles(beta[0,...], x, y, z, grid, shape_factor)
    betay_at_particle = interpolate_field_to_particles(beta[1,...], x, y, z, grid, shape_factor)
    betaz_at_particle = interpolate_field_to_particles(beta[2,...], x, y, z, grid, shape_factor)
    alpha_at_particle = interpolate_field_to_particles(alpha, x, y, z, grid, shape_factor)
    # interpolate the shift and lapse to the particle positions

    C = constants["C"]

    dx_dt = C * ( alpha_at_particle * jnp.einsum("j...,j...->...", hinv_[0, ...], newvx) - betax_at_particle )
    dy_dt = C * ( alpha_at_particle * jnp.einsum("j...,j...->...", hinv_[1, ...], newvy) - betay_at_particle )
    dz_dt = C * ( alpha_at_particle * jnp.einsum("j...,j...->...", hinv_[2, ...], newvz) - betaz_at_particle)
    # compute the coordinate velocity time derivatives using the new velocities and the metric terms

    x_ = x + dt * dx_dt
    y_ = y + dt * dy_dt
    z_ = z + dt * dz_dt
    # compute the updated coordinate positions after applying the metric push

    particles.set_position(x_, y_, z_)
    # set the new positions of the particles

    ############################### END OF POSITION UPDATE #########################################################

    return particles
