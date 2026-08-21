"""Active composite-grid DOFs and diagonal inner-product weights M_E and M_B."""

import jax.numpy as jnp

from .grids import _component_coordinate_axes, _coordinate_tolerance
from .types import B_FIELD_LOCATIONS, E_FIELD_LOCATIONS


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


def build_field_active_masks(
    parent_level,
    fine_level,
    parent_grids,
    fine_grids,
    field_locations,
    guard_cells,
):
    """Build component-specific ownership masks for either Yee vector."""

    refined_bounds = (
        (fine_level.x_min, fine_level.x_max),
        (fine_level.y_min, fine_level.y_max),
        (fine_level.z_min, fine_level.z_max),
    )
    parent_masks = []
    fine_masks = []
    for locations in field_locations:
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


def _fine_e_active_masks(fine_level, e_coarse_to_fine_maps, guard_cells):
    """Build E ownership without borrowing the component-wise B shapes."""

    g = int(guard_cells)
    fine_shape = (fine_level.Nx, fine_level.Ny, fine_level.Nz)
    weight_shape = (1, 1, 1, *fine_shape)
    fine_shape_array = jnp.asarray(fine_shape, dtype=jnp.int32)

    masks = []
    for interpolation_map in e_coarse_to_fine_maps:
        mask = jnp.ones(weight_shape, dtype=bool)
        target = interpolation_map.target_indices - g
        physical = jnp.all((target >= 0) & (target < fine_shape_array), axis=1)
        target = target[physical]
        mask = mask.at[
            0,
            0,
            0,
            target[:, 0],
            target[:, 1],
            target[:, 2],
        ].set(False, unique_indices=True)
        masks.append(mask)

    return tuple(masks)


def _component_parent_composite_weights(
    parent_grids,
    fine_grids,
    locations,
    parent_level,
    fine_level,
    guard_cells,
):
    """Return coarse dual volumes not covered by active fine dual volumes."""

    g = int(guard_cells)
    parent_spacing = tuple(jnp.asarray(value) for value in parent_level.spacing)
    fine_spacing = tuple(jnp.asarray(value) for value in fine_level.spacing)
    parent_volume = jnp.prod(jnp.asarray(parent_level.spacing))
    refined_bounds = (
        (fine_level.x_min, fine_level.x_max),
        (fine_level.y_min, fine_level.y_max),
        (fine_level.z_min, fine_level.z_max),
    )

    parent_axes = _component_coordinate_axes(parent_grids, locations)
    parent_axes = tuple(
        axis[g:g + cells]
        for axis, cells in zip(
            parent_axes,
            (parent_level.Nx, parent_level.Ny, parent_level.Nz),
        )
    )
    fine_axes = _component_coordinate_axes(fine_grids, locations)
    fine_axes = tuple(
        axis[g:g + cells]
        for axis, cells in zip(
            fine_axes,
            (fine_level.Nx, fine_level.Ny, fine_level.Nz),
        )
    )

    # Constrained fine interface points own no volume.  Along each axis the
    # remaining fine dual intervals form one contiguous interval; its Cartesian
    # product is the region actually replacing coarse control volume.
    tolerance = _coordinate_tolerance(*parent_axes, *fine_axes)
    fine_covered_bounds = []
    for axis, spacing, (lower, upper) in zip(
        fine_axes,
        fine_spacing,
        refined_bounds,
    ):
        active = (axis > lower + tolerance) & (axis < upper - tolerance)
        covered_lower = jnp.min(jnp.where(active, axis - 0.5 * spacing, jnp.inf))
        covered_upper = jnp.max(jnp.where(active, axis + 0.5 * spacing, -jnp.inf))
        fine_covered_bounds.append((covered_lower, covered_upper))

    coordinates = jnp.meshgrid(*parent_axes, indexing="ij")
    overlap_volume = jnp.ones(coordinates[0].shape, dtype=parent_volume.dtype)
    for coordinate, spacing, (covered_lower, covered_upper) in zip(
        coordinates,
        parent_spacing,
        fine_covered_bounds,
    ):
        dual_lower = coordinate - 0.5 * spacing
        dual_upper = coordinate + 0.5 * spacing
        overlap_width = jnp.maximum(
            jnp.minimum(dual_upper, covered_upper)
            - jnp.maximum(dual_lower, covered_lower),
            0.0,
        )
        overlap_volume *= overlap_width

    weight = jnp.maximum(parent_volume - overlap_volume, 0.0)
    zero_tolerance = 32.0 * jnp.finfo(weight.dtype).eps * parent_volume
    weight = jnp.where(weight > zero_tolerance, weight, 0.0)
    return weight[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :, :]


def build_fmr_metric_weights(
    parent_level,
    fine_level,
    parent_grids,
    fine_grids,
    e_coarse_to_fine_maps,
    parent_b_masks,
    fine_b_masks,
    guard_cells,
):
    """Build geometric two-level composite weights for E and B."""

    fine_volume = jnp.prod(jnp.asarray(fine_level.spacing))

    parent_e_weights = tuple(
        _component_parent_composite_weights(
            parent_grids,
            fine_grids,
            locations,
            parent_level,
            fine_level,
            guard_cells,
        )
        for locations in E_FIELD_LOCATIONS
    )

    fine_e_masks = _fine_e_active_masks(
        fine_level,
        e_coarse_to_fine_maps,
        guard_cells,
    )
    fine_e_weights = tuple(fine_volume * mask for mask in fine_e_masks)

    parent_b_weights = tuple(
        _component_parent_composite_weights(
            parent_grids,
            fine_grids,
            locations,
            parent_level,
            fine_level,
            guard_cells,
        ) * mask
        for locations, mask in zip(B_FIELD_LOCATIONS, parent_b_masks)
    )
    fine_b_weights = tuple(fine_volume * mask for mask in fine_b_masks)

    return (
        parent_e_weights,
        parent_b_weights,
        tuple(fine_e_weights),
        fine_b_weights,
    )
