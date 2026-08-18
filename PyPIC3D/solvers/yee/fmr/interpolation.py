"""Configured trilinear or triquadratic prolongation P on staggered Yee E grids."""

from itertools import product

import jax.numpy as jnp

from .grids import _component_coordinate_axes, _coordinate_tolerance
from .types import (
    E_FIELD_LOCATIONS,
    FMR_DEFAULT_INTERPOLATION_ORDER,
    FMR_SUPPORTED_INTERPOLATION_ORDERS,
    FMRInterpolationMap,
)


def _interface_target_indices(fine_axes, bounds, tolerance):
    """Locate fine-grid points on the closed boundary of the refined patch."""

    coordinate_mesh = jnp.meshgrid(*fine_axes, indexing="ij")
    in_closed_patch = jnp.ones(coordinate_mesh[0].shape, dtype=bool)
    on_interface = jnp.zeros(coordinate_mesh[0].shape, dtype=bool)
    for coordinate, (lower, upper) in zip(coordinate_mesh, bounds):
        in_closed_patch &= (coordinate >= lower - tolerance) & (coordinate <= upper + tolerance)
        on_interface |= jnp.isclose(coordinate, lower, rtol=0.0, atol=tolerance)
        on_interface |= jnp.isclose(coordinate, upper, rtol=0.0, atol=tolerance)

    return jnp.argwhere(in_closed_patch & on_interface).astype(jnp.int32)


def _linear_axis_stencil(parent_axis, target_coordinates, tolerance):
    """Return the two parent indices and degree-one weights along one axis."""

    insertion_index = jnp.searchsorted(parent_axis, target_coordinates, side="left")
    left_candidate = jnp.clip(insertion_index - 1, 0, parent_axis.size - 1)
    right_candidate = jnp.clip(insertion_index, 0, parent_axis.size - 1)

    coincident_left = jnp.isclose(
        parent_axis[left_candidate],
        target_coordinates,
        rtol=0.0,
        atol=tolerance,
    )
    coincident_right = jnp.isclose(
        parent_axis[right_candidate],
        target_coordinates,
        rtol=0.0,
        atol=tolerance,
    )
    coincident = coincident_left | coincident_right
    coincident_index = jnp.where(coincident_left, left_candidate, right_candidate)

    left_index = insertion_index - 1
    right_index = insertion_index
    left_index = jnp.where(coincident, coincident_index, left_index)
    right_index = jnp.where(coincident, coincident_index, right_index)

    if bool(jnp.any((left_index < 0) | (right_index >= parent_axis.size))):
        raise ValueError("The FMR patch does not have enough parent cells for interface interpolation.")

    parent_width = parent_axis[right_index] - parent_axis[left_index]
    safe_parent_width = jnp.where(coincident, 1.0, parent_width)
    right_weight = jnp.where(
        coincident,
        0.0,
        (target_coordinates - parent_axis[left_index]) / safe_parent_width,
    )

    source_indices = jnp.stack((left_index, right_index), axis=1).astype(jnp.int32)
    weights = jnp.stack((1.0 - right_weight, right_weight), axis=1)
    return source_indices, weights


def _quadratic_axis_stencil(parent_axis, target_coordinates, tolerance):
    """Return three local parent indices and degree-two Lagrange weights."""

    del tolerance
    if parent_axis.size < 3:
        raise ValueError(
            "The FMR patch does not have enough parent cells for interface interpolation."
        )

    distances = jnp.abs(
        target_coordinates[:, jnp.newaxis] - parent_axis[jnp.newaxis, :]
    )
    source_indices = jnp.argsort(distances, axis=1, stable=True)[:, :3]
    source_indices = jnp.sort(source_indices, axis=1).astype(jnp.int32)

    x = target_coordinates
    x0 = parent_axis[source_indices[:, 0]]
    x1 = parent_axis[source_indices[:, 1]]
    x2 = parent_axis[source_indices[:, 2]]

    w0 = (x - x1) * (x - x2) / ((x0 - x1) * (x0 - x2))
    w1 = (x - x0) * (x - x2) / ((x1 - x0) * (x1 - x2))
    w2 = (x - x0) * (x - x1) / ((x2 - x0) * (x2 - x1))

    return source_indices, jnp.stack((w0, w1, w2), axis=1)


def _axis_stencil(parent_axis, target_coordinates, tolerance, interpolation_order):
    if isinstance(interpolation_order, bool):
        raise ValueError(
            "FMR interpolation_order must be 1 (linear) or 2 (quadratic)."
        )
    if interpolation_order == 1:
        return _linear_axis_stencil(parent_axis, target_coordinates, tolerance)
    if interpolation_order == 2:
        return _quadratic_axis_stencil(parent_axis, target_coordinates, tolerance)
    raise ValueError(
        "FMR interpolation_order must be 1 (linear) or 2 (quadratic)."
    )


def _tensor_product_stencil(axis_source_indices, axis_weights):
    """Combine three equal-width axis stencils into tensor-product donors."""

    stencil_width = axis_source_indices[0].shape[1]
    donor_offsets = jnp.asarray(
        tuple(product(range(stencil_width), repeat=3)),
        dtype=jnp.int32,
    )
    source_indices = jnp.stack(
        tuple(
            indices[:, donor_offsets[:, axis]]
            for axis, indices in enumerate(axis_source_indices)
        ),
        axis=2,
    )
    donor_weights = jnp.stack(
        tuple(
            weights[:, donor_offsets[:, axis]]
            for axis, weights in enumerate(axis_weights)
        ),
        axis=2,
    )
    interpolation_weights = jnp.prod(donor_weights, axis=2)

    # Coincident axes have fewer nonzero donors. Pack all nonzero tensor-product
    # donors first, matching the existing compact map layout.
    donor_order = jnp.argsort(interpolation_weights == 0.0, axis=1, stable=True)
    source_indices = jnp.take_along_axis(
        source_indices,
        donor_order[:, :, jnp.newaxis],
        axis=1,
    )
    interpolation_weights = jnp.take_along_axis(
        interpolation_weights,
        donor_order,
        axis=1,
    )
    source_indices = jnp.where(
        interpolation_weights[:, :, jnp.newaxis] != 0.0,
        source_indices,
        source_indices[:, :1, :],
    )
    return source_indices, interpolation_weights


def _validate_interpolation_stencil(source_indices, weights, parent_shape, guard_cells):
    """Check that every active donor lies on the parent grid and weights sum to one."""

    g = int(guard_cells)
    upper_bound = g + jnp.asarray(parent_shape, dtype=jnp.int32)
    active_donors = weights != 0.0
    donors_in_parent = jnp.all(
        (source_indices >= g) & (source_indices < upper_bound),
        axis=2,
    )
    if not bool(jnp.all(~active_donors | donors_in_parent)):
        raise ValueError(
            "The FMR patch does not have enough parent cells around it "
            "for all Yee interpolation stencils."
        )
    if not bool(jnp.allclose(jnp.sum(weights, axis=1), 1.0)):
        raise ValueError("FMR interpolation weights must sum to one for every target.")


def _build_component_interpolation_map(
    coarse_axes,
    fine_axes,
    fine_level,
    parent_shape,
    guard_cells,
    interpolation_order,
):
    bounds = (
        (fine_level.x_min, fine_level.x_max),
        (fine_level.y_min, fine_level.y_max),
        (fine_level.z_min, fine_level.z_max),
    )
    tolerance = _coordinate_tolerance(*coarse_axes, *fine_axes)
    target_indices = _interface_target_indices(fine_axes, bounds, tolerance)
    target_coordinates = tuple(
        fine_axes[axis][target_indices[:, axis]]
        for axis in range(3)
    )

    axis_stencils = tuple(
        _axis_stencil(
            coarse_axes[axis],
            target_coordinates[axis],
            tolerance,
            interpolation_order,
        )
        for axis in range(3)
    )
    axis_source_indices = tuple(stencil[0] for stencil in axis_stencils)
    axis_weights = tuple(stencil[1] for stencil in axis_stencils)

    source_indices, weights = _tensor_product_stencil(
        axis_source_indices,
        axis_weights,
    )
    _validate_interpolation_stencil(source_indices, weights, parent_shape, guard_cells)

    weight_dtype = jnp.result_type(*(axis.dtype for axis in coarse_axes), jnp.float32)
    return FMRInterpolationMap(
        target_indices=target_indices,
        source_indices=source_indices,
        weights=weights.astype(weight_dtype),
    )


def build_e_interface_maps(
    parent_level,
    fine_level,
    parent_grids,
    fine_grids,
    guard_cells,
    interpolation_order=FMR_DEFAULT_INTERPOLATION_ORDER,
):
    """Build configured tensor-product prolongation maps for Ex, Ey, and Ez."""

    if (
        isinstance(interpolation_order, bool)
        or interpolation_order not in FMR_SUPPORTED_INTERPOLATION_ORDERS
    ):
        raise ValueError(
            "FMR interpolation_order must be 1 (linear) or 2 (quadratic)."
        )

    parent_shape = (parent_level.Nx, parent_level.Ny, parent_level.Nz)
    return tuple(
        _build_component_interpolation_map(
            _component_coordinate_axes(parent_grids, locations),
            _component_coordinate_axes(fine_grids, locations),
            fine_level,
            parent_shape,
            guard_cells,
            interpolation_order,
        )
        for locations in E_FIELD_LOCATIONS
    )


def _interpolate_component(parent_component, interpolation_map):
    source = interpolation_map.source_indices
    source_values = parent_component[
        0,
        0,
        0,
        source[:, :, 0],
        source[:, :, 1],
        source[:, :, 2],
    ]
    return jnp.sum(interpolation_map.weights * source_values, axis=1)


def prolong_e_to_fine_interface(parent_E, fine_E, e_interface_maps):
    """Overwrite only coarse-controlled fine E interface locations."""

    prolonged = []
    for parent_component, fine_component, interpolation_map in zip(parent_E, fine_E, e_interface_maps):
        target = interpolation_map.target_indices
        values = _interpolate_component(parent_component, interpolation_map)
        fine_component = fine_component.at[
            0,
            0,
            0,
            target[:, 0],
            target[:, 1],
            target[:, 2],
        ].set(values, unique_indices=True)
        prolonged.append(fine_component)
    return tuple(prolonged)
