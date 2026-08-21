"""Fixed stagger-aware transfers for one 2:1 rectangular Yee refinement patch."""

from itertools import product

import jax.numpy as jnp

from .grids import _component_coordinate_axes, _coordinate_tolerance
from .types import B_FIELD_LOCATIONS, E_FIELD_LOCATIONS, FMRInterpolationMap


def _closed_interface_indices(axes, bounds, tolerance):
    coordinates = jnp.meshgrid(*axes, indexing="ij")
    in_patch = jnp.ones(coordinates[0].shape, dtype=bool)
    on_interface = jnp.zeros(coordinates[0].shape, dtype=bool)
    for coordinate, (lower, upper) in zip(coordinates, bounds):
        in_patch &= (coordinate >= lower - tolerance) & (coordinate <= upper + tolerance)
        on_interface |= jnp.isclose(coordinate, lower, rtol=0.0, atol=tolerance)
        on_interface |= jnp.isclose(coordinate, upper, rtol=0.0, atol=tolerance)
    return jnp.argwhere(in_patch & on_interface).astype(jnp.int32)


def _strict_interior_indices(axes, bounds, tolerance):
    coordinates = jnp.meshgrid(*axes, indexing="ij")
    inside = jnp.ones(coordinates[0].shape, dtype=bool)
    for coordinate, (lower, upper) in zip(coordinates, bounds):
        inside &= (coordinate > lower + tolerance) & (coordinate < upper - tolerance)
    return jnp.argwhere(inside).astype(jnp.int32)


def _nearest_indices(source_axis, source_indices, targets, width):
    source_coordinates = source_axis[source_indices]
    distances = jnp.abs(targets[:, jnp.newaxis] - source_coordinates[jnp.newaxis, :])
    local_indices = jnp.argsort(distances, axis=1, stable=True)[:, :width]
    local_indices = jnp.sort(local_indices, axis=1)
    return source_indices[local_indices].astype(jnp.int32)


def _fourth_order_lagrange_axis_stencil(source_axis, source_indices, targets):
    """Build the four-point Lagrange stencil with O(h^4) point error."""

    indices = _nearest_indices(source_axis, source_indices, targets, 4)
    x = targets
    x0 = source_axis[indices[:, 0]]
    x1 = source_axis[indices[:, 1]]
    x2 = source_axis[indices[:, 2]]
    x3 = source_axis[indices[:, 3]]
    w0 = (x-x1)*(x-x2)*(x-x3) / ((x0-x1)*(x0-x2)*(x0-x3))
    w1 = (x-x0)*(x-x2)*(x-x3) / ((x1-x0)*(x1-x2)*(x1-x3))
    w2 = (x-x0)*(x-x1)*(x-x3) / ((x2-x0)*(x2-x1)*(x2-x3))
    w3 = (x-x0)*(x-x1)*(x-x2) / ((x3-x0)*(x3-x1)*(x3-x2))
    return indices, jnp.stack((w0, w1, w2, w3), axis=1)


def _tensor_product_stencil(axis_indices, axis_weights):
    width = axis_indices[0].shape[1]
    donor_offsets = jnp.asarray(tuple(product(range(width), repeat=3)), dtype=jnp.int32)
    source_indices = jnp.stack(
        tuple(indices[:, donor_offsets[:, axis]] for axis, indices in enumerate(axis_indices)),
        axis=2,
    )
    donor_weights = jnp.stack(
        tuple(weights[:, donor_offsets[:, axis]] for axis, weights in enumerate(axis_weights)),
        axis=2,
    )
    return source_indices, jnp.prod(donor_weights, axis=2)


def _build_transfer_map(source_axes, target_axes, source_indices, target_indices, axis_stencil):
    target_coordinates = tuple(
        target_axes[axis][target_indices[:, axis]]
        for axis in range(3)
    )
    source_axis_indices = tuple(jnp.unique(source_indices[:, axis]) for axis in range(3))
    axis_stencils = tuple(
        axis_stencil(source_axes[axis], source_axis_indices[axis], target_coordinates[axis])
        for axis in range(3)
    )
    donor_indices, weights = _tensor_product_stencil(
        tuple(stencil[0] for stencil in axis_stencils),
        tuple(stencil[1] for stencil in axis_stencils),
    )
    return FMRInterpolationMap(target_indices, donor_indices, weights)


def _build_component_maps(fine_level, parent_grids, fine_grids, field_locations):
    bounds = (
        (fine_level.x_min, fine_level.x_max),
        (fine_level.y_min, fine_level.y_max),
        (fine_level.z_min, fine_level.z_max),
    )
    interface_maps = []
    restriction_maps = []

    for locations in field_locations:
        parent_axes = _component_coordinate_axes(parent_grids, locations)
        fine_axes = _component_coordinate_axes(fine_grids, locations)
        tolerance = _coordinate_tolerance(*parent_axes, *fine_axes)
        fine_interface = _closed_interface_indices(fine_axes, bounds, tolerance)
        parent_interior = _strict_interior_indices(parent_axes, bounds, tolerance)
        fine_interior = _strict_interior_indices(fine_axes, bounds, tolerance)
        parent_all = jnp.stack(
            jnp.meshgrid(
                *(jnp.arange(axis.size, dtype=jnp.int32) for axis in parent_axes),
                indexing="ij",
            ),
            axis=-1,
        ).reshape((-1, 3))

        # Four-point Lagrange values leave O(h^4) transfer error before either
        # Yee curl. Both transfer directions therefore remain at least
        # second-order consistent at the interface after differentiation.
        interface_maps.append(
            _build_transfer_map(
                parent_axes,
                fine_axes,
                parent_all,
                fine_interface,
                _fourth_order_lagrange_axis_stencil,
            )
        )
        restriction_maps.append(
            _build_transfer_map(
                fine_axes,
                parent_axes,
                fine_interior,
                parent_interior,
                _fourth_order_lagrange_axis_stencil,
            )
        )

    return tuple(interface_maps), tuple(restriction_maps)


def build_e_transfer_maps(parent_level, fine_level, parent_grids, fine_grids, guard_cells):
    del parent_level, guard_cells
    return _build_component_maps(
        fine_level,
        parent_grids,
        fine_grids,
        E_FIELD_LOCATIONS,
    )


def build_b_transfer_maps(parent_level, fine_level, parent_grids, fine_grids, guard_cells):
    del parent_level, guard_cells
    return _build_component_maps(
        fine_level,
        parent_grids,
        fine_grids,
        B_FIELD_LOCATIONS,
    )


def _apply_component_map(source_component, target_component, transfer_map):
    source = transfer_map.source_indices
    source_values = source_component[
        0, 0, 0, source[:, :, 0], source[:, :, 1], source[:, :, 2]
    ]
    values = jnp.sum(transfer_map.weights * source_values, axis=1)
    target = transfer_map.target_indices
    return target_component.at[
        0, 0, 0, target[:, 0], target[:, 1], target[:, 2]
    ].set(values, unique_indices=True)


def prolong_e_to_fine_interface(parent_E, fine_E, e_interface_maps):
    """Set coarse-controlled fine Ex(VCC), Ey(CVC), and Ez(CCV) values."""

    parent_Ex, parent_Ey, parent_Ez = parent_E
    fine_Ex, fine_Ey, fine_Ez = fine_E
    Ex_map, Ey_map, Ez_map = e_interface_maps
    return (
        _apply_component_map(parent_Ex, fine_Ex, Ex_map),
        _apply_component_map(parent_Ey, fine_Ey, Ey_map),
        _apply_component_map(parent_Ez, fine_Ez, Ez_map),
    )


def prolong_b_to_fine_interface(parent_B, fine_B, b_interface_maps):
    """Set coarse-controlled fine Bx(CVV), By(VCV), and Bz(VVC) values."""

    parent_Bx, parent_By, parent_Bz = parent_B
    fine_Bx, fine_By, fine_Bz = fine_B
    Bx_map, By_map, Bz_map = b_interface_maps
    return (
        _apply_component_map(parent_Bx, fine_Bx, Bx_map),
        _apply_component_map(parent_By, fine_By, By_map),
        _apply_component_map(parent_Bz, fine_Bz, Bz_map),
    )


def restrict_e_to_coarse_shadow(fine_E, parent_E, e_restriction_maps):
    """Reconstruct current Ex(VCC), Ey(CVC), and Ez(CCV) in the coarse shadow."""

    fine_Ex, fine_Ey, fine_Ez = fine_E
    parent_Ex, parent_Ey, parent_Ez = parent_E
    Ex_map, Ey_map, Ez_map = e_restriction_maps
    return (
        _apply_component_map(fine_Ex, parent_Ex, Ex_map),
        _apply_component_map(fine_Ey, parent_Ey, Ey_map),
        _apply_component_map(fine_Ez, parent_Ez, Ez_map),
    )


def restrict_b_to_coarse_shadow(fine_B, parent_B, b_restriction_maps):
    """Reconstruct current Bx(CVV), By(VCV), and Bz(VVC) in the coarse shadow."""

    fine_Bx, fine_By, fine_Bz = fine_B
    parent_Bx, parent_By, parent_Bz = parent_B
    Bx_map, By_map, Bz_map = b_restriction_maps
    return (
        _apply_component_map(fine_Bx, parent_Bx, Bx_map),
        _apply_component_map(fine_By, parent_By, By_map),
        _apply_component_map(fine_Bz, parent_Bz, Bz_map),
    )
