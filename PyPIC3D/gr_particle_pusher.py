import jax
from jax import jit
import jax.numpy as jnp

from PyPIC3D.boris import interpolate_field_to_particles
from PyPIC3D.metric import metric_terms_at_position, relativistic_metric_rhs


@jit
def relativistic_metric_single_particle(vx, vy, vz, x, y, z, efield_atx, efield_aty, efield_atz, bfield_atx, bfield_aty, bfield_atz, q, m, dt, constants, metric):
    """
    RK2 update for relativistic EOM in a static metric:
      dv/dt = (q/m/gamma)(E + v x B) - Gamma(v,v)
    """
    v = jnp.array([vx, vy, vz])
    xvec = jnp.array([x, y, z])
    efield = jnp.array([efield_atx, efield_aty, efield_atz])
    bfield = jnp.array([bfield_atx, bfield_aty, bfield_atz])

    k1 = relativistic_metric_rhs(v, xvec, efield, bfield, q, m, constants, metric)
    v_half = v + 0.5 * dt * k1
    x_half = xvec + 0.5 * dt * v
    k2 = relativistic_metric_rhs(v_half, x_half, efield, bfield, q, m, constants, metric)
    newv = v + dt * k2

    # Enforce a strict subluminal speed bound in the local metric.
    c = constants["C"]
    g_cov, _, _, _, _, _ = metric_terms_at_position(x_half[0], x_half[1], x_half[2], metric)
    v2 = newv @ (g_cov @ newv)
    vmax2 = (0.999999 * c) ** 2
    scale = jnp.sqrt(vmax2 / jnp.maximum(v2, vmax2))
    newv = newv * scale

    return newv[0], newv[1], newv[2]


@jit
def particle_push_relativistic_metric(particles, E, B, grid, staggered_grid, dt, constants, metric):
    """
    Relativistic particle pusher that augments the EOM with geodesic terms
    from the supplied metric tensor.
    """
    q = particles.get_charge()
    m = particles.get_mass()
    x, y, z = particles.get_forward_position()
    vx, vy, vz = particles.get_velocity()
    shape_factor = particles.get_shape()

    Ex_grid = staggered_grid[0], grid[1], grid[2]
    Ey_grid = grid[0], staggered_grid[1], grid[2]
    Ez_grid = grid[0], grid[1], staggered_grid[2]
    Bx_grid = grid[0], staggered_grid[1], staggered_grid[2]
    By_grid = staggered_grid[0], grid[1], staggered_grid[2]
    Bz_grid = staggered_grid[0], staggered_grid[1], grid[2]

    Ex, Ey, Ez = E
    Bx, By, Bz = B

    efield_atx = interpolate_field_to_particles(Ex, x, y, z, Ex_grid, shape_factor)
    efield_aty = interpolate_field_to_particles(Ey, x, y, z, Ey_grid, shape_factor)
    efield_atz = interpolate_field_to_particles(Ez, x, y, z, Ez_grid, shape_factor)
    bfield_atx = interpolate_field_to_particles(Bx, x, y, z, Bx_grid, shape_factor)
    bfield_aty = interpolate_field_to_particles(By, x, y, z, By_grid, shape_factor)
    bfield_atz = interpolate_field_to_particles(Bz, x, y, z, Bz_grid, shape_factor)

    metric_push_vmap = jax.vmap(
        relativistic_metric_single_particle,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None, None),
    )
    newvx, newvy, newvz = metric_push_vmap(
        vx, vy, vz, x, y, z,
        efield_atx, efield_aty, efield_atz,
        bfield_atx, bfield_aty, bfield_atz,
        q, m, dt, constants, metric
    )

    particles.set_velocity(newvx, newvy, newvz)
    return particles
