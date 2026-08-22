"""Fixed stagger-aware transfers for one 2:1 Yee refinement interface."""

from itertools import product

import jax.numpy as jnp

from .grids import SINGLE_TILE_INDEX, _coordinate_tolerance, component_coordinate_axes
from .types import B_FIELD_LOCATIONS, E_FIELD_LOCATIONS, FMRTransferMap


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


def _physical_indices(level, guard_cells):
    g = int(guard_cells)
    shape = level.shape
    return jnp.stack(
        jnp.meshgrid(
            *(jnp.arange(g, g + cells, dtype=jnp.int32) for cells in shape),
            indexing="ij",
        ),
        axis=-1,
    ).reshape((-1, 3))


def _indices_strictly_inside(indices, axes, bounds, tolerance):
    inside = jnp.ones(indices.shape[0], dtype=bool)
    for axis, (lower, upper) in enumerate(bounds):
        coordinate = axes[axis][indices[:, axis]]
        inside &= (coordinate > lower + tolerance) & (coordinate < upper - tolerance)
    return inside


def _unique_indices(*index_sets):
    nonempty = tuple(indices for indices in index_sets if indices.shape[0] != 0)
    if not nonempty:
        return jnp.zeros((0, 3), dtype=jnp.int32)
    return jnp.unique(jnp.concatenate(nonempty, axis=0), axis=0).astype(jnp.int32)


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
    return FMRTransferMap(target_indices, donor_indices, weights)


_CURL_COMPONENT_READS = (
    ((2, 1), (1, 2)),
    ((0, 2), (2, 0)),
    ((1, 0), (0, 1)),
)


def _curl_read_indices(output_active_indices, input_component, offset):
    """Return one field component's indices read by an active Yee curl."""

    reads = []
    for output_component, component_reads in enumerate(_CURL_COMPONENT_READS):
        output_indices = output_active_indices[output_component]
        for component, axis in component_reads:
            if component != input_component:
                continue
            shifted = output_indices.at[:, axis].add(offset)
            reads.extend((output_indices, shifted))
    return _unique_indices(*reads)


def _active_component_indices(level, grids, field_locations, bounds, guard_cells, fine):
    physical = _physical_indices(level, guard_cells)
    result = []
    for locations in field_locations:
        axes = component_coordinate_axes(grids, locations)
        tolerance = _coordinate_tolerance(*axes)
        inside = _indices_strictly_inside(physical, axes, bounds, tolerance)
        result.append(physical[inside] if fine else physical[~inside])
    return tuple(result)


def _build_component_maps(
    parent_level,
    fine_level,
    parent_grids,
    fine_grids,
    field_locations,
    curl_output_locations,
    curl_offset,
    guard_cells,
):
    bounds = tuple(zip(fine_level.lower, fine_level.upper))
    parent_output_active = _active_component_indices(
        parent_level,
        parent_grids,
        curl_output_locations,
        bounds,
        guard_cells,
        fine=False,
    )
    fine_output_active = _active_component_indices(
        fine_level,
        fine_grids,
        curl_output_locations,
        bounds,
        guard_cells,
        fine=True,
    )
    coarse_to_fine_maps = []
    fine_to_coarse_maps = []
    for locations in field_locations:
        parent_axes = component_coordinate_axes(parent_grids, locations)
        fine_axes = component_coordinate_axes(fine_grids, locations)
        tolerance = _coordinate_tolerance(*parent_axes, *fine_axes)
        fine_interface = _closed_interface_indices(fine_axes, bounds, tolerance)
        fine_interior = _strict_interior_indices(fine_axes, bounds, tolerance)
        parent_all = jnp.stack(
            jnp.meshgrid(
                *(jnp.arange(axis.size, dtype=jnp.int32) for axis in parent_axes),
                indexing="ij",
            ),
            axis=-1,
        ).reshape((-1, 3))

        component = len(coarse_to_fine_maps)
        fine_reads = _curl_read_indices(
            fine_output_active,
            component,
            curl_offset,
        )
        fine_read_is_owned = _indices_strictly_inside(
            fine_reads,
            fine_axes,
            bounds,
            tolerance,
        )
        fine_ghost = _unique_indices(fine_interface, fine_reads[~fine_read_is_owned])

        # Build the coarse-to-fine stencil first.  Its covered donors, together
        # with covered values read by the active coarse curl, define the narrow
        # coarse ghost region that must be refreshed from the current fine
        # solution.
        coarse_to_fine_map = _build_transfer_map(
            parent_axes,
            fine_axes,
            parent_all,
            fine_ghost,
            _fourth_order_lagrange_axis_stencil,
        )
        fine_ghost_donors = jnp.unique(
            coarse_to_fine_map.source_indices.reshape((-1, 3)),
            axis=0,
        )
        covered_fine_ghost_donors = fine_ghost_donors[
            _indices_strictly_inside(
                fine_ghost_donors,
                parent_axes,
                bounds,
                tolerance,
            )
        ]

        parent_reads = _curl_read_indices(
            parent_output_active,
            component,
            curl_offset,
        )
        covered_parent_reads = parent_reads[
            _indices_strictly_inside(
                parent_reads,
                parent_axes,
                bounds,
                tolerance,
            )
        ]
        coarse_ghost = _unique_indices(
            covered_fine_ghost_donors,
            covered_parent_reads,
        )

        # Four-point Lagrange values leave O(h^4) transfer error before either
        # Yee curl.  The fine-to-coarse map reads only fine-owned values; the
        # coarse-to-fine map reads only coarse-owned or refreshed ghost values.
        fine_to_coarse_map = _build_transfer_map(
            fine_axes,
            parent_axes,
            fine_interior,
            coarse_ghost,
            _fourth_order_lagrange_axis_stencil,
        )

        coarse_to_fine_maps.append(coarse_to_fine_map)
        fine_to_coarse_maps.append(fine_to_coarse_map)
    return tuple(coarse_to_fine_maps), tuple(fine_to_coarse_maps)


def build_e_transfer_maps(parent_level, fine_level, parent_grids, fine_grids, guard_cells):
    return _build_component_maps(
        parent_level,
        fine_level,
        parent_grids,
        fine_grids,
        E_FIELD_LOCATIONS,
        B_FIELD_LOCATIONS,
        1,
        guard_cells,
    )


def build_b_transfer_maps(parent_level, fine_level, parent_grids, fine_grids, guard_cells):
    return _build_component_maps(
        parent_level,
        fine_level,
        parent_grids,
        fine_grids,
        B_FIELD_LOCATIONS,
        E_FIELD_LOCATIONS,
        -1,
        guard_cells,
    )


def _apply_component_map(source_component, target_component, transfer_map):
    source = transfer_map.source_indices
    source_values = source_component[
        *SINGLE_TILE_INDEX,
        source[:, :, 0],
        source[:, :, 1],
        source[:, :, 2],
    ]
    values = jnp.sum(transfer_map.weights * source_values, axis=1)
    target = transfer_map.target_indices
    return target_component.at[
        *SINGLE_TILE_INDEX,
        target[:, 0],
        target[:, 1],
        target[:, 2],
    ].set(values, unique_indices=True)


def interpolate_coarse_to_fine(coarse_fields, fine_fields, interpolation_maps):
    """Interpolate a staggered field into fine refinement ghost cells."""

    return tuple(
        _apply_component_map(coarse, fine, interpolation_map)
        for coarse, fine, interpolation_map in zip(
            coarse_fields,
            fine_fields,
            interpolation_maps,
        )
    )


def interpolate_fine_to_coarse(fine_fields, coarse_fields, interpolation_maps):
    """Interpolate a staggered field into covered coarse ghost cells."""

    return tuple(
        _apply_component_map(fine, coarse, interpolation_map)
        for fine, coarse, interpolation_map in zip(
            fine_fields,
            coarse_fields,
            interpolation_maps,
        )
    )
