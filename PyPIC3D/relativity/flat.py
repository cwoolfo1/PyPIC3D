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


def _zeros_like_metric_derivatives(shape, dtype):
    return (
        jnp.zeros(shape + (3, 3, 3), dtype=dtype),
        jnp.zeros(shape + (3,), dtype=dtype),
        jnp.zeros(shape + (3, 3), dtype=dtype),
    )


def _metric_from_diagonal(gamma_diag, sqrt_gamma):
    shape = gamma_diag.shape[:-1]
    dtype = gamma_diag.dtype
    eye = jnp.eye(3, dtype=dtype)

    gamma = gamma_diag[..., :, jnp.newaxis] * eye
    gamma_inv_diag = 1.0 / gamma_diag
    gamma_inv = gamma_inv_diag[..., :, jnp.newaxis] * eye

    lapse = jnp.ones(shape, dtype=dtype)
    shift = jnp.zeros(shape + (3,), dtype=dtype)
    christoffel, grad_lapse, grad_shift = _zeros_like_metric_derivatives(shape, dtype)

    return Metric(
        lapse=lapse,
        shift=shift,
        gamma=gamma,
        gamma_inv=gamma_inv,
        sqrt_gamma=sqrt_gamma,
        christoffel=christoffel,
        grad_lapse=grad_lapse,
        grad_shift=grad_shift,
    )


def _flat_cartesian_metric_on_grid(grid):
    X, _, _ = _coordinate_mesh(grid)
    gamma_diag = jnp.ones(X.shape + (3,), dtype=X.dtype)
    sqrt_gamma = jnp.ones_like(X)
    return _metric_from_diagonal(gamma_diag, sqrt_gamma)


def _flat_cylindrical_metric_on_grid(grid):
    R, _, _ = _coordinate_mesh(grid)
    gamma_diag = jnp.stack(
        (
            jnp.ones_like(R),
            R**2,
            jnp.ones_like(R),
        ),
        axis=-1,
    )
    sqrt_gamma = R
    return _metric_from_diagonal(gamma_diag, sqrt_gamma)


def _flat_spherical_metric_on_grid(grid):
    R, theta, _ = _coordinate_mesh(grid)
    sin_theta = jnp.sin(theta)
    gamma_diag = jnp.stack(
        (
            jnp.ones_like(R),
            R**2,
            R**2 * sin_theta**2,
        ),
        axis=-1,
    )
    sqrt_gamma = R**2 * sin_theta
    return _metric_from_diagonal(gamma_diag, sqrt_gamma)


def _maybe_fill_derivatives(metric, dynamic_parameters, fill_derivatives):
    if not fill_derivatives:
        return metric
    return fill_metric_derivatives(
        metric,
        dynamic_parameters.dx,
        dynamic_parameters.dy,
        dynamic_parameters.dz,
    )


def _build_yee_metric(dynamic_parameters, metric_on_grid, fill_derivatives=False):
    D = tuple(
        _maybe_fill_derivatives(
            metric_on_grid(_location_grid(location, dynamic_parameters)),
            dynamic_parameters,
            fill_derivatives,
        )
        for location in D_FIELD_LOCATIONS
    )
    B = tuple(
        _maybe_fill_derivatives(
            metric_on_grid(_location_grid(location, dynamic_parameters)),
            dynamic_parameters,
            fill_derivatives,
        )
        for location in B_FIELD_LOCATIONS
    )
    center = _maybe_fill_derivatives(
        metric_on_grid(dynamic_parameters.grids.tiled_center_grid),
        dynamic_parameters,
        fill_derivatives,
    )
    vertex = _maybe_fill_derivatives(
        metric_on_grid(dynamic_parameters.grids.tiled_vertex_grid),
        dynamic_parameters,
        fill_derivatives,
    )
    return YeeMetric(D=D, B=B, center=center, vertex=vertex)


def initialize_flat_cartesian_metric(static_parameters, dynamic_parameters):
    """
    Build the flat Cartesian 3+1 metric on the tiled Yee grid.
    """

    del static_parameters
    return _build_yee_metric(dynamic_parameters, _flat_cartesian_metric_on_grid)


def initialize_flat_cylindrical_metric(static_parameters, dynamic_parameters):
    """
    Build the flat cylindrical metric, ds^2 = dr^2 + r^2 dphi^2 + dz^2.
    """

    del static_parameters
    return _build_yee_metric(
        dynamic_parameters,
        _flat_cylindrical_metric_on_grid,
        fill_derivatives=True,
    )


def initialize_flat_spherical_metric(static_parameters, dynamic_parameters):
    """
    Build the flat spherical metric, ds^2 = dr^2 + r^2 dtheta^2 + r^2 sin^2(theta)dphi^2.
    """

    del static_parameters
    return _build_yee_metric(
        dynamic_parameters,
        _flat_spherical_metric_on_grid,
        fill_derivatives=True,
    )
