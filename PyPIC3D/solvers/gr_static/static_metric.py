import jax.numpy as jnp

from PyPIC3D.boundary_conditions.supergaussian import apply_tiled_supergaussian_absorber
from PyPIC3D.relativity.core import B_FIELD_LOCATIONS, D_FIELD_LOCATIONS
from PyPIC3D.boundary_conditions.staggered import refresh_fields
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING
from PyPIC3D.relativity.field_interpolation import (
    metric_weighted_interpolate as _metric_weighted_interpolate,
)


def _shift_cross_component(beta, vector_components, component):
    beta_x = beta[..., 0]
    beta_y = beta[..., 1]
    beta_z = beta[..., 2]
    vector_x, vector_y, vector_z = vector_components

    if component == 0:
        return beta_y * vector_z - beta_z * vector_y
    if component == 1:
        return beta_z * vector_x - beta_x * vector_z
    return beta_x * vector_y - beta_y * vector_x


def compute_covariant_E(D_tiles, B_tiles, metric):
    """
    Compute covariant E_i on the D component locations using FPIC Eq. (10).

    Uses standard metric-density-weighted transfers on a regular coordinate
    domain. All source and target metric samples, including halos, must have
    finite positive sqrt_gamma. Polar finite-volume geometry does not select
    a different constitutive operator.
    """

    E_cov = []
    for i, target_location in enumerate(D_FIELD_LOCATIONS):
        D_on_target = []
        B_on_target = []
        for j, source_location in enumerate(D_FIELD_LOCATIONS):
            D_on_target.append(
                _metric_weighted_interpolate(
                    D_tiles[j],
                    metric.D[j],
                    metric.D[i],
                    source_location,
                    target_location,
                )
            )
        for j, source_location in enumerate(B_FIELD_LOCATIONS):
            B_on_target.append(
                _metric_weighted_interpolate(
                    B_tiles[j],
                    metric.B[j],
                    metric.D[i],
                    source_location,
                    target_location,
                )
            )

        D_lower_i = 0.0
        for j in range(3):
            D_lower_i = D_lower_i + metric.D[i].gamma[..., i, j] * D_on_target[j]

        shift_cross = _shift_cross_component(metric.D[i].shift, tuple(B_on_target), i)
        E_cov.append(
            metric.D[i].lapse * D_lower_i
            + metric.D[i].sqrt_gamma * shift_cross
        )
    return tuple(E_cov)


def compute_covariant_H(D_tiles, B_tiles, metric):
    """
    Compute covariant H_i on the B component locations using FPIC Eq. (9).

    Uses the same standard transfers and regular-domain requirements as
    ``compute_covariant_E``.
    """

    H_cov = []
    for i, target_location in enumerate(B_FIELD_LOCATIONS):
        B_on_target = []
        D_on_target = []
        for j, source_location in enumerate(B_FIELD_LOCATIONS):
            B_on_target.append(
                _metric_weighted_interpolate(
                    B_tiles[j],
                    metric.B[j],
                    metric.B[i],
                    source_location,
                    target_location,
                )
            )
        for j, source_location in enumerate(D_FIELD_LOCATIONS):
            D_on_target.append(
                _metric_weighted_interpolate(
                    D_tiles[j],
                    metric.D[j],
                    metric.B[i],
                    source_location,
                    target_location,
                )
            )

        B_lower_i = 0.0
        for j in range(3):
            B_lower_i = B_lower_i + metric.B[i].gamma[..., i, j] * B_on_target[j]

        shift_cross = _shift_cross_component(metric.B[i].shift, tuple(D_on_target), i)
        H_cov.append(
            metric.B[i].lapse * B_lower_i
            - metric.B[i].sqrt_gamma * shift_cross
        )
    return tuple(H_cov)


def _update_field(fields, auxiliary, current, metric, static, dynamic, dt, *, magnetic):
    """Yee curl on owned nodes, including physical upper C endpoints."""
    from PyPIC3D.boundary_conditions.pec import owned_nodes
    locations = B_FIELD_LOCATIONS if magnetic else D_FIELD_LOCATIONS
    samples = metric.B if magnetic else metric.D
    if BC_CONDUCTING in static.boundary_conditions:
        auxiliary = refresh_fields(auxiliary, static,
                                   D_FIELD_LOCATIONS if magnetic else B_FIELD_LOCATIONS)
    spacing = (dynamic.dx, dynamic.dy, dynamic.dz)
    result = []
    for i, location in enumerate(locations):
        j, k = (i+1) % 3, (i+2) % 3
        def derivative(value, axis):
            if magnetic:
                return (jnp.roll(value, -1, axis+3)-value)/spacing[axis]
            return (value-jnp.roll(value, 1, axis+3))/spacing[axis]
        curl = derivative(auxiliary[k], j)-derivative(auxiliary[j], k)
        rate = -curl/samples[i].sqrt_gamma if magnetic else (
            curl/samples[i].sqrt_gamma - 4*jnp.pi*current[i])
        result.append(jnp.where(owned_nodes(fields[i].shape, location, static),
                                fields[i]+dt*rate, fields[i]))
    result = apply_tiled_supergaussian_absorber(
        tuple(result), static, dynamic, dt, locations=locations)
    return refresh_fields(result, static, locations, 'B' if magnetic else 'D', metric)


def update_D_relativity(D_tiles, H_tiles, J_tiles, metric, static_parameters, dynamic_parameters, dt):
    """Advance contravariant D and enforce its FIDO surface projection."""
    if metric.geometry is not None:
        from PyPIC3D.boundary_conditions.polar import update_fields
        return update_fields(D_tiles,H_tiles,J_tiles,metric,static_parameters,dynamic_parameters,dt)
    return _update_field(D_tiles, H_tiles, J_tiles, metric, static_parameters,
                         dynamic_parameters, dt, magnetic=False)


def update_B_relativity(E_tiles, B_tiles, metric, static_parameters, dynamic_parameters, dt):
    """Advance contravariant B and remove its normal surface component."""
    if metric.geometry is not None:
        from PyPIC3D.boundary_conditions.polar import update_fields
        return update_fields(B_tiles,E_tiles,None,metric,static_parameters,dynamic_parameters,dt,magnetic=True)
    return _update_field(B_tiles, E_tiles, None, metric, static_parameters,
                         dynamic_parameters, dt, magnetic=True)
