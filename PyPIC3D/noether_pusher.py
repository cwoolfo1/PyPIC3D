import jax
from jax import jit
import jax.numpy as jnp

from PyPIC3D.utils import (
    wrap_around
)


def magnetic_rotation_single_particle(vx, vy, vz, bfield_atx, bfield_aty, bfield_atz, q, m, dt, constants):

    v = jnp.array([vx, vy, vz])
    # convert v into an array

    t = q*dt/(2*m)*jnp.array([bfield_atx, bfield_aty, bfield_atz])
    # calculate the t vector

    vprime = v + jnp.cross(v, t)
    # calculate the v prime vector

    s = 2*t / (1 + t[0]**2 + t[1]**2 + t[2]**2)
    # calculate the s vector

    vplus = v + jnp.cross(vprime, s)
    # calculate the v plus vector

    return vplus[0], vplus[1], vplus[2]


def E_integration_step(E, x, y, z, vx, vy, vz, grid, direction, q, m, dt, dx, dy, dz):
    # integrating particle over 1 timestep along a single direction using Noether integrator

    shift = grid[direction][0]
    # get the grid shift in the direction of integration

    xi = jnp.floor( (x - grid[0][0]) / dx).astype(int)
    yi = jnp.floor( (y - grid[1][0]) / dy).astype(int)
    zi = jnp.floor( (z - grid[2][0]) / dz).astype(int)
    # calculate the initial grid indices of the particle

    x_indicies = jnp.array([xi, yi, zi])
    # convert to vector of index points

    ds = jnp.array([dx, dy, dz])
    # convert the resolution vector
    dx_ = ds[direction]
    # take the resolution along the direction of integration

    x = jnp.array([x, y, z])
    # convert to position vector
    x_ = x[direction]
    # take the 1D component of the position along the direction of integration

    v = jnp.array([vx, vy, vz])
    # convert to velocity vector
    v0 = v[direction]
    # take the 1D component of the velocity being integrated over

    dt_ = dt # total time still need to integrate


    def time_particle_substep(x_indicies, x_, v0, dt_):
        xi_ = x_indicies[direction]
        # get the grid index along the direction of integration

        x_indexing = jnp.mod(x_indicies[0], E.shape[0])
        y_indexing = jnp.mod(x_indicies[1], E.shape[1])
        z_indexing = jnp.mod(x_indicies[2], E.shape[2])
        # mod the indices to ensure they are within bounds

        E_ = E[x_indexing, y_indexing, z_indexing]
        # get the electric field in the current cell

        a = q / m * E_
        # electrostatic acceleration

        x_L = (xi_ - 0.5) * dx_ + shift
        x_R = (xi_ + 0.5) * dx_ + shift
        # left and right edges of the cell

        deltax_L = x_ - x_L
        deltax_R = x_R - x_
        # compute the distances to the left and right edges of the cell

        # deltax + v0*t + 0.5*a*t^2 = 0
        A = 0.5 * a
        B = v0
        C_L = deltax_L
        C_R = deltax_R
        # coefficients for quadratic equation to left and right boundaries

        D_R = B*B + 4 * A * C_R
        D_L = B*B - 4 * A * C_L
        # discriminant of quadratic equation

        t1_L = jax.lax.cond(
            jnp.abs(a) < 1e-15,
            lambda _: jax.lax.cond(
                jnp.abs(v0) < 1e-15,
                lambda _: jnp.inf,
                lambda _: -C_L / v0,
                operand=None,
            ),
            # if no acceleration, use linear formula
            # handle case where velocity is zero to avoid division by zero

            lambda _: jax.lax.cond(
                D_L < -1e-15,
                lambda _: jnp.inf,
                lambda _: (-B + jnp.sqrt( jnp.maximum(D_L, 0.0) ) ) / (2*A),
                operand=None,
            ),
            # compute time using quadratic formula if acceleration is non-zero
            operand=None,
        )

        t2_L = jax.lax.cond(
            jnp.abs(a) < 1e-15,
            lambda _: jax.lax.cond(
                jnp.abs(v0) < 1e-15,
                lambda _: jnp.inf,
                lambda _: -C_L / v0,
                operand=None,
            ),
            # if no acceleration, use linear formula
            # handle case where velocity is zero to avoid division by zero

            lambda _: jax.lax.cond(
                D_L < -1e-15,
                lambda _: jnp.inf,
                lambda _: (-B - jnp.sqrt( jnp.maximum(D_L, 0.0) ) ) / (2*A),
                operand=None,
            ),
            # compute time using quadratic formula if acceleration is non-zero

            operand=None,
        )

        t1_R = jax.lax.cond(
            jnp.abs(a) < 1e-15,
            lambda _: jax.lax.cond(
                jnp.abs(v0) < 1e-15,
                lambda _: jnp.inf,
                lambda _: C_R / v0,
                operand=None,
            ),
            # if no acceleration, use linear formula
            # handle case where velocity is zero to avoid division by zero

            lambda _: jax.lax.cond(
                D_R < -1e-15,
                lambda _: jnp.inf,
                lambda _: (-B + jnp.sqrt( jnp.maximum(D_R, 0.0) ) ) / (2*A),
                operand=None,
            ),
            # compute time using quadratic formula if acceleration is non-zero

            operand=None,
        )

        t2_R = jax.lax.cond(
            jnp.abs(a) < 1e-15,
            lambda _: jax.lax.cond(
                jnp.abs(v0) < 1e-15,
                lambda _: jnp.inf,
                lambda _: C_R / v0,
                operand=None,
            ),
            # if no acceleration, use linear formula
            # handle case where velocity is zero to avoid division by zero
            lambda _: jax.lax.cond(
                D_R < -1e-15,
                lambda _: jnp.inf,
                lambda _: (-B - jnp.sqrt( jnp.maximum(D_R, 0.0) ) ) / (2*A),
                operand=None,
            ),
            # compute time using quadratic formula if acceleration is non-zero
            operand=None,
        )

        t_L = jnp.where( jnp.array([t1_L, t2_L]) > 0, jnp.array([t1_L, t2_L]), jnp.inf )
        t_R = jnp.where( jnp.array([t1_R, t2_R]) > 0, jnp.array([t1_R, t2_R]), jnp.inf )
        # filter out negative times

        t_L = jnp.min( t_L )
        t_R = jnp.min( t_R )
        # select the minimum time to cross the cell

        t_ = jnp.min( jnp.array([t_L, t_R]) )

        return t_, a
        # method computes time needed to cross the nearest cell

    def body_fn(carry):
        x_indicies, x_, v0, dt_ = carry
        t_, a = time_particle_substep(x_indicies, x_, v0, dt_)
        # get time to cross cell and acceleration

        new_v0 = jax.lax.cond(
            t_ > dt_,
            lambda _: v0 + a*dt_,
            lambda _: v0 + a*t_,
            operand=None,
        ) # update the velocity based on whether crossing occurs

        new_x = jax.lax.cond(
            t_ > dt_,
            lambda _: x_ + v0* dt_ + 0.5*a*dt_*dt_,
            lambda _: x_ + v0*t_   + 0.5*a*t_*t_,
            operand=None,
        ) # update the position based on whether crossing occurs

        particle_direction = jax.lax.cond(
            new_x - x_ > 0,
            lambda _: 1,
            lambda _: -1,
            operand = None,
        ) # what direction is the particle moving? (to the left or right)

        x_indicies = x_indicies.at[direction].add( particle_direction * ( t_ <= dt_ ).astype(int) )
        # update the grid index if crossing occurs

        dt_set = jnp.minimum(dt_, t_)
        # determine the time step used in this substep

        dt_ = dt_ - dt_set
        # update the remaining time to integrate over

        carry = x_indicies, new_x, new_v0, dt_
        # pack carry variables

        return carry
    
    def cond_fn(carry):
        x_indicies, x_, v0, dt_ = carry
        return dt_ > 1e-15 * dt
    
    x_indicies, no_boundary_crossing_x, new_v0, dt_ = jax.lax.while_loop(cond_fn, body_fn, (x_indicies, x_, v0, dt_))
    # loop until the full dt has been integrated over

    return no_boundary_crossing_x, new_v0


@jit
def noether_particle_push(particles, E, B, grid, staggered_grid, dt, constants, periodic=True, relativistic=True):

    q = particles.get_charge()
    m = particles.get_mass()
    x, y, z = particles.get_forward_position()
    vx, vy, vz = particles.get_velocity()
    # get the charge, mass, position, and velocity of the particles

    particles.set_old_position(x, y, z)
    # store the old position before updating

    shape_factor = particles.get_shape()
    # get the shape factor of the particles

    dx = grid[0][1] - grid[0][0]
    dy = grid[1][1] - grid[1][0]
    dz = grid[2][1] - grid[2][0]
    # calculate the grid spacing in each direction

    ################## INTERPOLATION GRIDS ##########################
    Bx_grid = grid[0], staggered_grid[1], staggered_grid[2]
    By_grid = staggered_grid[0], grid[1], staggered_grid[2]
    Bz_grid = staggered_grid[0], staggered_grid[1], grid[2]
    # create the staggered grids for the magnetic field components

    Bx, By, Bz = B
    # calculate the magnetic field at the particle positions using their specific staggered grids
    bfield_atx = jax.lax.cond(
        shape_factor == 1,
        lambda _: create_trilinear_interpolator(Bx, Bx_grid, periodic)(x, y, z),
        lambda _: create_quadratic_interpolator(Bx, Bx_grid, periodic)(x, y, z),
        operand=None
    )
    bfield_aty = jax.lax.cond(
        shape_factor == 1,
        lambda _: create_trilinear_interpolator(By, By_grid, periodic)(x, y, z),
        lambda _: create_quadratic_interpolator(By, By_grid, periodic)(x, y, z),
        operand=None
    )
    bfield_atz = jax.lax.cond(
        shape_factor == 1,
        lambda _: create_trilinear_interpolator(Bz, Bz_grid, periodic)(x, y, z),
        lambda _: create_quadratic_interpolator(Bz, Bz_grid, periodic)(x, y, z),
        operand=None
    )
    # calculate the magnetic field at the particle positions

    # magnetic_rotation_vmap  = jax.vmap(magnetic_rotation_single_particle, in_axes=( 0, 0, 0, 0, 0, 0, None, None, None, None))
    # first perform the magnetic rotation step of the Boris algorithm

    # vx, vy, vz = magnetic_rotation_vmap( vx, vy, vz, bfield_atx, bfield_aty, bfield_atz, q, m, dt, constants )
    # perform magnetic rotation

    electric_field_update_vmap = jax.vmap(E_integration_step, in_axes=(None, 0, 0, 0, 0, 0, 0, None, None, None, None, None, None, None, None) )
    # the update the particles using energy conserving E integrator

    Ex, Ey, Ez = E
    # unpack the electric field components

    x, vx = electric_field_update_vmap(Ex, x, y, z, vx, vy, vz, grid, 0, q, m, dt, dx, dy, dz)
    # first update along x
    y, vy = electric_field_update_vmap(Ey, x, y, z, vx, vy, vz, grid, 1, q, m, dt, dx, dy, dz)
    # then update along y
    z, vz = electric_field_update_vmap(Ez, x, y, z, vx, vy, vz, grid, 2, q, m, dt, dx, dy, dz)
    # then update along z

    particles.set_velocity(vx, vy, vz)
    particles.set_position(x, y, z)
    # update the particle velocities and positions
    particles.boundary_conditions()
    # apply boundary conditions to the updated positions

    return particles



def create_trilinear_interpolator(field, grid, periodic=True):
    """
    Create a trilinear interpolation function for a given 3D field and grid.

    Handles cases where any dimension (nx, ny, nz) has only 1 grid point.
    In such cases, no interpolation is performed along that dimension.

    Args:
        field (ndarray): The 3D field to interpolate.
        grid (tuple): A tuple of three arrays representing the grid points in the x, y, and z directions.
        periodic (tuple): A tuple of three booleans indicating whether each dimension is periodic.
                         Default is (True, True, True) for fully periodic domains.

    Returns:
        function: A function that takes (x, y, z) coordinates and returns the interpolated values.
    """
    x_grid, y_grid, z_grid = grid

    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    dz = z_grid[1] - z_grid[0]
    # calculate the grid spacing in each direction

    x_min, x_max = x_grid[0], x_grid[-1]
    y_min, y_max = y_grid[0], y_grid[-1]
    z_min, z_max = z_grid[0], z_grid[-1]
    # get grid bounds

    x_wind = x_max - x_min
    y_wind = y_max - y_min
    z_wind = z_max - z_min
    # get spatial widths of the grid

    Nx = len(x_grid)
    Ny = len(y_grid)
    Nz = len(z_grid)
    # get the number of grid points in each direction

    @jit
    def interpolator(x, y, z):

        # Handle periodic boundaries by wrapping coordinates
        if periodic:
            x = x_min + jnp.mod(x - x_min, x_wind)
            y = y_min + jnp.mod(y - y_min, y_wind)
            z = z_min + jnp.mod(z - z_min, z_wind)

        # Convert coordinates to grid indices (fractional)
        # Handle single-point dimensions specially
        xi = jnp.where(Nx == 1, 0.0, (x - x_min) / dx)
        yi = jnp.where(Ny == 1, 0.0, (y - y_min) / dy)
        zi = jnp.where(Nz == 1, 0.0, (z - z_min) / dz)

        # Find the lower-left-bottom corner indices
        x0 = jnp.where(Nx == 1, 0, jnp.floor(xi).astype(int))
        y0 = jnp.where(Ny == 1, 0, jnp.floor(yi).astype(int))
        z0 = jnp.where(Nz == 1, 0, jnp.floor(zi).astype(int))

        # Handle boundary conditions
        if periodic:
            x0 = jnp.where(Nx == 1, 0, jnp.mod(x0, Nx))
            y0 = jnp.where(Ny == 1, 0, jnp.mod(y0, Ny))
            z0 = jnp.where(Nz == 1, 0, jnp.mod(z0, Nz))
            x1 = jnp.where(Nx == 1, 0, jnp.mod(x0 + 1, Nx))
            y1 = jnp.where(Ny == 1, 0, jnp.mod(y0 + 1, Ny))
            z1 = jnp.where(Nz == 1, 0, jnp.mod(z0 + 1, Nz))
        else:
            x0 = jnp.where(Nx == 1, 0, jnp.clip(x0, 0, Nx - 2))
            y0 = jnp.where(Ny == 1, 0, jnp.clip(y0, 0, Ny - 2))
            z0 = jnp.where(Nz == 1, 0, jnp.clip(z0, 0, Nz - 2))
            x1 = jnp.where(Nx == 1, 0, jnp.clip(x0 + 1, 0, Nx - 1))
            y1 = jnp.where(Ny == 1, 0, jnp.clip(y0 + 1, 0, Ny - 1))
            z1 = jnp.where(Nz == 1, 0, jnp.clip(z0 + 1, 0, Nz - 1))

        # Calculate the fractional parts (weights)
        # For single-point dimensions, weight is meaningless so set to 0
        wx = jnp.where(Nx == 1, 0.0, xi - jnp.floor(xi))
        wy = jnp.where(Ny == 1, 0.0, yi - jnp.floor(yi))
        wz = jnp.where(Nz == 1, 0.0, zi - jnp.floor(zi))

        # Trilinear interpolation
        c000 = field[x0, y0, z0]
        c001 = field[x0, y0, z1]
        c010 = field[x0, y1, z0]
        c011 = field[x0, y1, z1]
        c100 = field[x1, y0, z0]
        c101 = field[x1, y0, z1]
        c110 = field[x1, y1, z0]
        c111 = field[x1, y1, z1]

        # Interpolate along x-axis first
        c00 = c000 * (1 - wx) + c100 * wx
        c01 = c001 * (1 - wx) + c101 * wx
        c10 = c010 * (1 - wx) + c110 * wx
        c11 = c011 * (1 - wx) + c111 * wx

        # Then interpolate along y-axis
        c0 = c00 * (1 - wy) + c10 * wy
        c1 = c01 * (1 - wy) + c11 * wy

        # Finally interpolate along z-axis
        value = c0 * (1 - wz) + c1 * wz

        return value

    vmap_interpolator = jax.vmap(interpolator, in_axes=(0, 0, 0), out_axes=0)
    # vectorize the interpolator for batch processing

    return vmap_interpolator

def create_quadratic_interpolator(field, grid, periodic=True):
    """
    Create a quadratic interpolation function for a given 3D field and grid.

    Args:
        field (ndarray): The 3D field to interpolate.
        grid (tuple): A tuple of three arrays representing the grid points in the x, y, and z directions.
        periodic (tuple): A tuple of three booleans indicating whether each dimension is periodic.
                         Default is (True, True, True) for fully periodic domains.

    Returns:
        function: A function that takes (x, y, z) coordinates and returns the interpolated values.
    """
    x_grid, y_grid, z_grid = grid

    # Calculate grid spacing and bounds
    dx = x_grid[1] - x_grid[0] if len(x_grid) > 1 else 1.0
    dy = y_grid[1] - y_grid[0] if len(y_grid) > 1 else 1.0
    dz = z_grid[1] - z_grid[0] if len(z_grid) > 1 else 1.0

    x_min, x_max = x_grid[0], x_grid[-1]
    y_min, y_max = y_grid[0], y_grid[-1]
    z_min, z_max = z_grid[0], z_grid[-1]

    # Calculate domain widths
    x_width = x_max - x_min
    y_width = y_max - y_min
    z_width = z_max - z_min

    Nx = len(x_grid)
    Ny = len(y_grid)
    Nz = len(z_grid)

    @jit
    def interpolator(x, y, z):
        # Handle periodic boundaries by wrapping coordinates
        if periodic:
            x = x_min + jnp.mod(x - x_min, x_width)
            y = y_min + jnp.mod(y - y_min, y_width)
            z = z_min + jnp.mod(z - z_min, z_width)

        # Handle each dimension separately to account for different grid sizes
        def get_indices_and_points(coord, grid_1d, coord_min, d_spacing):
            grid_len = len(grid_1d)

            # Convert coordinate to fractional grid index
            if grid_len == 1:
                frac_idx = 0.0
            else:
                frac_idx = (coord - coord_min) / d_spacing

            # Use JAX-compatible conditionals
            idx_base = jnp.floor(frac_idx).astype(int) if grid_len > 1 else 0

            # For grids with 3+ points, use quadratic interpolation
            idx_quad = jnp.clip(idx_base, 1, grid_len - 2)

            # For grids with 2 points, use linear interpolation
            idx_lin = jnp.clip(idx_base, 0, 0)

            # For single-point grids, use index 0
            idx_single = 0

            # Select appropriate index based on grid size
            idx = jnp.where(grid_len >= 3, idx_quad,
                   jnp.where(grid_len == 2, idx_lin, idx_single))

            # Handle periodic boundary conditions for indices
            if periodic and grid_len > 1:
                # For periodic boundaries, wrap the indices
                idx_m1 = jnp.mod(idx - 1, grid_len)
                idx_0 = jnp.mod(idx, grid_len)
                idx_p1 = jnp.mod(idx + 1, grid_len)

                p0 = jnp.where(grid_len >= 3, grid_1d[idx_m1],
                      jnp.where(grid_len == 2, grid_1d[0], grid_1d[0]))
                p1 = jnp.where(grid_len >= 2, grid_1d[idx_0], grid_1d[0])
                p2 = jnp.where(grid_len >= 3, grid_1d[idx_p1],
                      jnp.where(grid_len == 2, grid_1d[1], grid_1d[0]))
            else:
                # For non-periodic boundaries, use clipping
                # Get the three points for interpolation
                # For 3+ points: normal stencil
                # For 2 points: duplicate last point
                # For 1 point: duplicate the single point
                p0 = jnp.where(grid_len >= 3, grid_1d[jnp.clip(idx - 1, 0, grid_len - 1)],
                      jnp.where(grid_len == 2, grid_1d[0], grid_1d[0]))

                p1 = jnp.where(grid_len >= 2, grid_1d[jnp.clip(idx, 0, grid_len - 1)], grid_1d[0])

                p2 = jnp.where(grid_len >= 3, grid_1d[jnp.clip(idx + 1, 0, grid_len - 1)],
                      jnp.where(grid_len == 2, grid_1d[1], grid_1d[0]))

            return idx, p0, p1, p2

        x_idx, x0, x1, x2 = get_indices_and_points(x, x_grid, x_min, dx)
        y_idx, y0, y1, y2 = get_indices_and_points(y, y_grid, y_min, dy)
        z_idx, z0, z1, z2 = get_indices_and_points(z, z_grid, z_min, dz)

        def quadratic_weights(t, t0, t1, t2):
            # Handle degenerate cases (when points are equal)
            eps = 1e-12

            # Check for degenerate cases using JAX-compatible logic
            all_same = (jnp.abs(t0 - t1) < eps) & (jnp.abs(t1 - t2) < eps)
            t0_eq_t1 = (jnp.abs(t0 - t1) < eps) & (jnp.abs(t1 - t2) >= eps)
            t1_eq_t2 = (jnp.abs(t0 - t1) >= eps) & (jnp.abs(t1 - t2) < eps)

            # Standard quadratic weights
            denom0 = (t0 - t1) * (t0 - t2)
            denom1 = (t1 - t0) * (t1 - t2)
            denom2 = (t2 - t0) * (t2 - t1)

            # Avoid division by zero by adding small epsilon where needed
            denom0 = jnp.where(jnp.abs(denom0) < eps, eps, denom0)
            denom1 = jnp.where(jnp.abs(denom1) < eps, eps, denom1)
            denom2 = jnp.where(jnp.abs(denom2) < eps, eps, denom2)

            w0_standard = (t - t1) * (t - t2) / denom0
            w1_standard = (t - t0) * (t - t2) / denom1
            w2_standard = (t - t0) * (t - t1) / denom2

            # Linear interpolation weights for degenerate cases
            w_lin_01 = jnp.where(jnp.abs(t1 - t0) < eps, 0.0, (t - t0) / (t1 - t0))
            w_lin_12 = jnp.where(jnp.abs(t2 - t1) < eps, 0.0, (t - t1) / (t2 - t1))

            # Select weights based on degenerate cases
            w0 = jnp.where(all_same, 0.0,
                  jnp.where(t0_eq_t1, 0.0,
                   jnp.where(t1_eq_t2, 1.0 - w_lin_01, w0_standard)))

            w1 = jnp.where(all_same, 1.0,
                  jnp.where(t0_eq_t1, 1.0 - w_lin_12,
                   jnp.where(t1_eq_t2, w_lin_01, w1_standard)))

            w2 = jnp.where(all_same, 0.0,
                  jnp.where(t0_eq_t1, w_lin_12,
                   jnp.where(t1_eq_t2, 0.0, w2_standard)))

            return w0, w1, w2

        wx0, wx1, wx2 = quadratic_weights(x, x0, x1, x2)
        wy0, wy1, wy2 = quadratic_weights(y, y0, y1, y2)
        wz0, wz1, wz2 = quadratic_weights(z, z0, z1, z2)

        interpolated_value = 0.0
        for i, wx in enumerate([wx0, wx1, wx2]):
            for j, wy in enumerate([wy0, wy1, wy2]):
                for k, wz in enumerate([wz0, wz1, wz2]):
                    # Calculate array indices with proper bounds checking
                    # Use JAX-compatible conditionals

                    # X index calculation
                    if periodic and Nx > 1:
                        xi_quad = jnp.mod(x_idx - 1 + i, Nx)  # For 3+ points with periodic
                        xi_lin = jnp.mod(x_idx + i - 1, Nx)   # For 2 points with periodic
                    else:
                        xi_quad = x_idx - 1 + i  # For 3+ points
                        xi_lin = jnp.clip(x_idx + i - 1, 0, max(1, Nx - 1))  # For 2 points
                    xi_single = 0  # For 1 point

                    xi = jnp.where(Nx >= 3, xi_quad,
                          jnp.where(Nx == 2, xi_lin, xi_single))

                    # Y index calculation
                    if periodic and Ny > 1:
                        yi_quad = jnp.mod(y_idx - 1 + j, Ny)
                        yi_lin = jnp.mod(y_idx + j - 1, Ny)
                    else:
                        yi_quad = y_idx - 1 + j
                        yi_lin = jnp.clip(y_idx + j - 1, 0, max(1, Ny - 1))
                    yi_single = 0

                    yi = jnp.where(Ny >= 3, yi_quad,
                          jnp.where(Ny == 2, yi_lin, yi_single))

                    # Z index calculation
                    if periodic and Nz > 1:
                        zi_quad = jnp.mod(z_idx - 1 + k, Nz)
                        zi_lin = jnp.mod(z_idx + k - 1, Nz)
                    else:
                        zi_quad = z_idx - 1 + k
                        zi_lin = jnp.clip(z_idx + k - 1, 0, max(1, Nz - 1))
                    zi_single = 0

                    zi = jnp.where(Nz >= 3, zi_quad,
                          jnp.where(Nz == 2, zi_lin, zi_single))

                    # Final bounds check (only needed for non-periodic)
                    if not periodic:
                        xi = jnp.clip(xi, 0, field.shape[0] - 1)
                        yi = jnp.clip(yi, 0, field.shape[1] - 1)
                        zi = jnp.clip(zi, 0, field.shape[2] - 1)

                    interpolated_value += wx * wy * wz * field[xi, yi, zi]

        return interpolated_value

    vmap_interpolator = jax.vmap(interpolator, in_axes=(0, 0, 0), out_axes=0)
    return vmap_interpolator