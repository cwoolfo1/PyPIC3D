import jax.numpy as jnp

from PyPIC3D.relativity.core import (
    B_FIELD_LOCATIONS,
    D_FIELD_LOCATIONS,
    Metric,
    YeeMetric,
    fill_metric_derivatives,
)


def _coordinate_mesh(grid):
    x_grid, y_grid, z_grid = grid
    X = x_grid[..., :, jnp.newaxis, jnp.newaxis]
    Y = y_grid[..., jnp.newaxis, :, jnp.newaxis]
    Z = z_grid[..., jnp.newaxis, jnp.newaxis, :]
    return jnp.broadcast_arrays(X, Y, Z)


def _location_grid(location, dynamic_parameters):
    center_grid = dynamic_parameters.grids.tiled_center_grid
    vertex_grid = dynamic_parameters.grids.tiled_vertex_grid
    return tuple(
        center_grid[axis] if location[axis] == "C" else vertex_grid[axis]
        for axis in range(3)
    )


def _empty_derivative_arrays(shape, dtype):
    return (
        jnp.zeros(shape + (3, 3, 3), dtype=dtype),
        jnp.zeros(shape + (3,), dtype=dtype),
        jnp.zeros(shape + (3, 3), dtype=dtype),
    )


def _kerr_schild_cartesian_on_grid(grid, dynamic_parameters, mass=1.0, spin=0.0):
    x, y, z = _coordinate_mesh(grid)
    dtype = x.dtype
    shape = x.shape
    eye = jnp.eye(3, dtype=dtype)

    a = spin
    rho_squared = x**2 + y**2 + z**2
    r_squared = 0.5 * (
        rho_squared
        - a**2
        + jnp.sqrt((rho_squared - a**2) ** 2 + 4.0 * a**2 * z**2)
    )
    r = jnp.sqrt(jnp.maximum(r_squared, 1.0e-30))

    denominator = jnp.where(r_squared + a**2 != 0.0, r_squared + a**2, 1.0)
    ell = jnp.stack(
        (
            (r * x + a * y) / denominator,
            (r * y - a * x) / denominator,
            z / r,
        ),
        axis=-1,
    )

    H = mass * r**3 / jnp.where(r**4 + a**2 * z**2 != 0.0, r**4 + a**2 * z**2, 1.0)
    two_H = 2.0 * H
    lapse = 1.0 / jnp.sqrt(1.0 + two_H)
    shift = (two_H / (1.0 + two_H))[..., jnp.newaxis] * ell

    inverse_factor = two_H / (1.0 + two_H)
    gamma = eye + two_H[..., jnp.newaxis, jnp.newaxis] * ell[..., :, jnp.newaxis] * ell[..., jnp.newaxis, :]
    gamma_inv = eye - inverse_factor[..., jnp.newaxis, jnp.newaxis] * ell[..., :, jnp.newaxis] * ell[..., jnp.newaxis, :]
    sqrt_gamma = jnp.sqrt(1.0 + two_H)
    christoffel, grad_lapse, grad_shift = _empty_derivative_arrays(shape, dtype)

    metric = Metric(
        lapse=lapse,
        shift=shift,
        gamma=gamma,
        gamma_inv=gamma_inv,
        sqrt_gamma=sqrt_gamma,
        christoffel=christoffel,
        grad_lapse=grad_lapse,
        grad_shift=grad_shift,
    )
    return fill_metric_derivatives(metric, dynamic_parameters.dx, dynamic_parameters.dy, dynamic_parameters.dz)


def _kerr_schild_spherical_on_grid(grid, dynamic_parameters, mass=1.0, spin=0.0):
    r, theta, _ = _coordinate_mesh(grid)
    dtype = r.dtype
    shape = r.shape

    a = spin
    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)
    safe_sin_theta = jnp.where(jnp.abs(sin_theta) > 0.0, sin_theta, 1.0)
    rho_squared = r**2 + a**2 * cos_theta**2
    safe_rho_squared = jnp.where(rho_squared != 0.0, rho_squared, 1.0)
    xi = 1.0 + 2.0 * mass * r / safe_rho_squared

    lapse = xi**-0.5
    shift = jnp.zeros(shape + (3,), dtype=dtype)
    shift = shift.at[..., 0].set((xi - 1.0) / xi)

    gamma = jnp.zeros(shape + (3, 3), dtype=dtype)
    gamma = gamma.at[..., 0, 0].set(xi)
    gamma = gamma.at[..., 1, 1].set(rho_squared)
    gamma = gamma.at[..., 2, 2].set(
        safe_sin_theta**2 * (rho_squared + a**2 * xi * safe_sin_theta**2)
    )
    gamma = gamma.at[..., 0, 2].set(-a * xi * safe_sin_theta**2)
    gamma = gamma.at[..., 2, 0].set(gamma[..., 0, 2])

    gamma_inv = jnp.zeros(shape + (3, 3), dtype=dtype)
    gamma_inv = gamma_inv.at[..., 0, 0].set(1.0 / xi + a**2 * safe_sin_theta**2 / safe_rho_squared)
    gamma_inv = gamma_inv.at[..., 1, 1].set(1.0 / safe_rho_squared)
    gamma_inv = gamma_inv.at[..., 2, 2].set(1.0 / (safe_rho_squared * safe_sin_theta**2))
    gamma_inv = gamma_inv.at[..., 0, 2].set(a / safe_rho_squared)
    gamma_inv = gamma_inv.at[..., 2, 0].set(gamma_inv[..., 0, 2])

    sqrt_gamma = rho_squared * jnp.sqrt(xi) * sin_theta
    christoffel, grad_lapse, grad_shift = _empty_derivative_arrays(shape, dtype)

    metric = Metric(
        lapse=lapse,
        shift=shift,
        gamma=gamma,
        gamma_inv=gamma_inv,
        sqrt_gamma=sqrt_gamma,
        christoffel=christoffel,
        grad_lapse=grad_lapse,
        grad_shift=grad_shift,
    )
    return fill_metric_derivatives(metric, dynamic_parameters.dx, dynamic_parameters.dy, dynamic_parameters.dz)


def _build_yee_metric(static_parameters, dynamic_parameters, metric_on_grid, mass=1.0, spin=0.0):
    del static_parameters
    D = tuple(
        metric_on_grid(_location_grid(location, dynamic_parameters), dynamic_parameters, mass=mass, spin=spin)
        for location in D_FIELD_LOCATIONS
    )
    B = tuple(
        metric_on_grid(_location_grid(location, dynamic_parameters), dynamic_parameters, mass=mass, spin=spin)
        for location in B_FIELD_LOCATIONS
    )
    center = metric_on_grid(dynamic_parameters.grids.tiled_center_grid, dynamic_parameters, mass=mass, spin=spin)
    vertex = metric_on_grid(dynamic_parameters.grids.tiled_vertex_grid, dynamic_parameters, mass=mass, spin=spin)
    return YeeMetric(D=D, B=B, center=center, vertex=vertex)


def initialize_kerr_schild_cartesian_metric(static_parameters, dynamic_parameters, mass=1.0, spin=0.0):
    """
    Build the ingoing Cartesian Kerr-Schild 3+1 metric on the tiled Yee grid.
    """

    return _build_yee_metric(
        static_parameters,
        dynamic_parameters,
        _kerr_schild_cartesian_on_grid,
        mass=mass,
        spin=spin,
    )


def initialize_kerr_schild_spherical_metric(static_parameters, dynamic_parameters, mass=1.0, spin=0.0):
    """
    Build the spherical Kerr-Schild 3+1 metric on the tiled Yee grid.
    """

    return _build_yee_metric(
        static_parameters,
        dynamic_parameters,
        _kerr_schild_spherical_on_grid,
        mass=mass,
        spin=spin,
    )
