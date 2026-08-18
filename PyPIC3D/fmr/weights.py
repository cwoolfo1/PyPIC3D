"""Active composite-grid DOFs and diagonal inner-product weights M_E and M_B."""

import jax.numpy as jnp

from .grids import _component_coordinate_axes, _coordinate_tolerance
from .types import B_FIELD_LOCATIONS


def _component_inside_mask(grids, locations, level, refined_bounds, guard_cells):
    g = int(guard_cells)
    axes = _component_coordinate_axes(grids, locations)
    active_axes = tuple(axis[g:g + cells] for axis, cells in zip(axes, (level.Nx, level.Ny, level.Nz)))
    x, y, z = jnp.meshgrid(*active_axes, indexing="ij")

    tolerance = _coordinate_tolerance(*active_axes)
    inside = jnp.ones(x.shape, dtype=bool)
    for coordinate, (lower, upper) in zip((x, y, z), refined_bounds):
        inside &= (coordinate > lower + tolerance) & (coordinate < upper - tolerance)

    return inside[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :, :]


def build_b_active_masks(parent_level, fine_level, parent_grids, fine_grids, guard_cells):
    """Build stagger-aware coarse-active B masks for the two-level composite grid."""

    refined_bounds = (
        (fine_level.x_min, fine_level.x_max),
        (fine_level.y_min, fine_level.y_max),
        (fine_level.z_min, fine_level.z_max),
    )
    parent_masks = []
    fine_masks = []
    for locations in B_FIELD_LOCATIONS:
        parent_inside = _component_inside_mask(
            parent_grids,
            locations,
            parent_level,
            refined_bounds,
            guard_cells,
        )
        fine_inside = _component_inside_mask(
            fine_grids,
            locations,
            fine_level,
            refined_bounds,
            guard_cells,
        )
        parent_masks.append(~parent_inside)
        fine_masks.append(fine_inside)

    return tuple(parent_masks), tuple(fine_masks)


def build_fmr_metric_weights(
    parent_level,
    fine_level,
    e_interface_maps,
    parent_b_masks,
    fine_b_masks,
    guard_cells,
):
    """Build active-grid Cartesian volume weights for the FMR fields."""

    parent_volume = jnp.prod(jnp.asarray(parent_level.spacing))
    fine_volume = jnp.prod(jnp.asarray(fine_level.spacing))

    parent_e_weights = tuple(
        jnp.full(mask.shape, parent_volume)
        for mask in parent_b_masks
    )

    g = int(guard_cells)
    fine_shape = jnp.asarray((fine_level.Nx, fine_level.Ny, fine_level.Nz), dtype=jnp.int32)
    fine_e_weights = []
    for interpolation_map, mask in zip(e_interface_maps, fine_b_masks):
        weight = jnp.full(mask.shape, fine_volume)
        target = interpolation_map.target_indices - g
        physical = jnp.all((target >= 0) & (target < fine_shape), axis=1)
        target = target[physical]
        weight = weight.at[
            0,
            0,
            0,
            target[:, 0],
            target[:, 1],
            target[:, 2],
        ].set(0.0, unique_indices=True)
        fine_e_weights.append(weight)

    parent_b_weights = tuple(parent_volume * mask for mask in parent_b_masks)
    fine_b_weights = tuple(fine_volume * mask for mask in fine_b_masks)

    return (
        parent_e_weights,
        parent_b_weights,
        tuple(fine_e_weights),
        fine_b_weights,
    )


def _apply_weights(values, weights):
    return tuple(value * weight for value, weight in zip(values, weights))


def _apply_inverse_weights(values, weights):
    weighted_values = []
    for value, weight in zip(values, weights):
        safe_weight = jnp.where(weight != 0.0, weight, 1.0)
        weighted_values.append(jnp.where(weight != 0.0, value / safe_weight, 0.0))
    return tuple(weighted_values)
