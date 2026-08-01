import jax.numpy as jnp

from PyPIC3D.relativity.core import (
    B_FIELD_LOCATIONS,
    D_FIELD_LOCATIONS,
    YeeMetric,
    analytic_metric_on_grid,
)


def _location_grid(location, dynamic_parameters):
    center_grid = dynamic_parameters.grids.tiled_center_grid
    vertex_grid = dynamic_parameters.grids.tiled_vertex_grid
    return tuple(
        center_grid[axis] if location[axis] == "C" else vertex_grid[axis]
        for axis in range(3)
    )


def _metric_terms_from_diagonal(gamma_diag, sqrt_gamma):
    dtype = gamma_diag.dtype
    eye = jnp.eye(3, dtype=dtype)

    gamma = gamma_diag[:, jnp.newaxis] * eye
    gamma_inv_diag = 1.0 / gamma_diag
    gamma_inv = gamma_inv_diag[:, jnp.newaxis] * eye

    return (
        jnp.asarray(1.0, dtype=dtype),
        jnp.zeros(3, dtype=dtype),
        gamma,
        gamma_inv,
        sqrt_gamma,
    )


def _flat_cartesian_metric_at_position(position):
    gamma_diag = jnp.ones(3, dtype=position.dtype)
    sqrt_gamma = jnp.asarray(1.0, dtype=position.dtype)
    return _metric_terms_from_diagonal(gamma_diag, sqrt_gamma)


def _flat_cylindrical_metric_at_position(position):
    R = position[0]
    gamma_diag = jnp.asarray(
        (
            1.0,
            R**2,
            1.0,
        ),
        dtype=position.dtype,
    )
    return _metric_terms_from_diagonal(gamma_diag, R)


def _flat_spherical_metric_at_position(position):
    R, theta, _ = position
    sin_theta = jnp.sin(theta)
    gamma_diag = jnp.asarray(
        (
            1.0,
            R**2,
            R**2 * sin_theta**2,
        ),
        dtype=position.dtype,
    )
    sqrt_gamma = R**2 * sin_theta
    return _metric_terms_from_diagonal(gamma_diag, sqrt_gamma)


def _build_yee_metric(dynamic_parameters, metric_at_position):
    D = tuple(
        analytic_metric_on_grid(
            _location_grid(location, dynamic_parameters),
            metric_at_position,
        )[0]
        for location in D_FIELD_LOCATIONS
    )
    B = tuple(
        analytic_metric_on_grid(
            _location_grid(location, dynamic_parameters),
            metric_at_position,
        )[0]
        for location in B_FIELD_LOCATIONS
    )
    center, center_grad_gamma_inv = analytic_metric_on_grid(
        dynamic_parameters.grids.tiled_center_grid,
        metric_at_position,
    )
    vertex, _ = analytic_metric_on_grid(
        dynamic_parameters.grids.tiled_vertex_grid,
        metric_at_position,
    )

    return YeeMetric(
        D=D,
        B=B,
        center=center,
        vertex=vertex,
        center_grad_gamma_inv=center_grad_gamma_inv,
    )


def initialize_flat_cartesian_metric(static_parameters, dynamic_parameters):
    """
    Build the flat Cartesian 3+1 metric on the tiled Yee grid.
    """

    del static_parameters
    return _build_yee_metric(dynamic_parameters, _flat_cartesian_metric_at_position)


def initialize_flat_cylindrical_metric(static_parameters, dynamic_parameters):
    """
    Build the flat cylindrical metric, ds^2 = dr^2 + r^2 dphi^2 + dz^2.
    """

    del static_parameters
    return _build_yee_metric(
        dynamic_parameters,
        _flat_cylindrical_metric_at_position,
    )


def initialize_flat_spherical_metric(static_parameters, dynamic_parameters):
    """
    Build the flat spherical metric, ds^2 = dr^2 + r^2 dtheta^2 + r^2 sin^2(theta)dphi^2.
    """

    del static_parameters
    return _build_yee_metric(
        dynamic_parameters,
        _flat_spherical_metric_at_position,
    )
