"""Construction of FMR runtime metadata and level-major field arrays."""

import jax.numpy as jnp

from .grids import _build_level_grids
from .interpolation import (
    build_b_transfer_maps,
    build_e_transfer_maps,
    interpolate_coarse_to_fine,
    interpolate_fine_to_coarse,
)
from .types import B_FIELD_LOCATIONS, E_FIELD_LOCATIONS, FMRLevelData, FMRParameters
from .weights import build_field_active_masks, build_fmr_metric_weights


def build_fmr_parameters(static_parameters, dynamic_parameters):
    """Build the static FMR interpolation, activity, and metric data once."""

    if not static_parameters.fmr_enabled:
        return None
    if len(static_parameters.fmr_levels) != 2:
        raise ValueError("The first FMR implementation requires root and one fine level.")
    if int(static_parameters.guard_cells) < 2:
        raise ValueError("FMR mesh-adapted Yee differencing requires at least two guard cells.")

    parent_level, fine_level = static_parameters.fmr_levels
    fine_grids = _build_level_grids(fine_level, static_parameters.guard_cells)
    (
        e_coarse_to_fine_maps,
        e_fine_to_coarse_maps,
        e_deep_shadow_indices,
    ) = build_e_transfer_maps(
        parent_level,
        fine_level,
        dynamic_parameters.grids,
        fine_grids,
        static_parameters.guard_cells,
    )
    (
        b_coarse_to_fine_maps,
        b_fine_to_coarse_maps,
        b_deep_shadow_indices,
    ) = build_b_transfer_maps(
        parent_level,
        fine_level,
        dynamic_parameters.grids,
        fine_grids,
        static_parameters.guard_cells,
    )
    parent_e_masks, fine_e_masks = build_field_active_masks(
        parent_level,
        fine_level,
        dynamic_parameters.grids,
        fine_grids,
        E_FIELD_LOCATIONS,
        static_parameters.guard_cells,
    )
    parent_b_masks, fine_b_masks = build_field_active_masks(
        parent_level,
        fine_level,
        dynamic_parameters.grids,
        fine_grids,
        B_FIELD_LOCATIONS,
        static_parameters.guard_cells,
    )
    (
        parent_e_weights,
        parent_b_weights,
        fine_e_weights,
        fine_b_weights,
    ) = build_fmr_metric_weights(
        parent_level,
        fine_level,
        dynamic_parameters.grids,
        fine_grids,
        e_coarse_to_fine_maps,
        parent_b_masks,
        fine_b_masks,
        static_parameters.guard_cells,
    )

    parent_data = FMRLevelData(
        grids=dynamic_parameters.grids,
        e_coarse_to_fine_maps=(),
        b_coarse_to_fine_maps=(),
        e_fine_to_coarse_maps=(),
        b_fine_to_coarse_maps=(),
        e_deep_shadow_indices=(),
        b_deep_shadow_indices=(),
        e_active_masks=parent_e_masks,
        b_active_masks=parent_b_masks,
        e_weights=parent_e_weights,
        b_weights=parent_b_weights,
    )
    fine_data = FMRLevelData(
        grids=fine_grids,
        e_coarse_to_fine_maps=e_coarse_to_fine_maps,
        b_coarse_to_fine_maps=b_coarse_to_fine_maps,
        e_fine_to_coarse_maps=e_fine_to_coarse_maps,
        b_fine_to_coarse_maps=b_fine_to_coarse_maps,
        e_deep_shadow_indices=e_deep_shadow_indices,
        b_deep_shadow_indices=b_deep_shadow_indices,
        e_active_masks=fine_e_masks,
        b_active_masks=fine_b_masks,
        e_weights=fine_e_weights,
        b_weights=fine_b_weights,
    )
    return FMRParameters(levels=(parent_data, fine_data))


def _fine_vector(level, guard_cells, templates):
    g = int(guard_cells)
    shape = (1, 1, 1, level.Nx + 2 * g, level.Ny + 2 * g, level.Nz + 2 * g)
    return tuple(jnp.zeros(shape, dtype=template.dtype) for template in templates)


def build_fmr_fields(E0, B0, J0, static_parameters, dynamic_parameters):
    """Allocate the one-patch fine fields and package level-major tuples."""

    fine_level = static_parameters.fmr_levels[1]
    E1 = _fine_vector(fine_level, static_parameters.guard_cells, E0)
    B1 = _fine_vector(fine_level, static_parameters.guard_cells, B0)
    J1 = _fine_vector(fine_level, static_parameters.guard_cells, J0)

    # Initialize the constrained fine E interface and curl-reachable ghost
    # cells.  The fine-owned interior remains zero until the caller populates
    # or evolves it.
    E1 = interpolate_coarse_to_fine(
        E0,
        E1,
        dynamic_parameters.fmr.levels[1].e_coarse_to_fine_maps,
    )
    return (E0, E1), (B0, B1), (J0, J1)


def synchronize_e_levels(E_levels, dynamic_parameters):
    """Fill the coarse and fine E ghost cells without touching deep shadow."""

    E0, E1 = E_levels
    fine_data = dynamic_parameters.fmr.levels[1]
    E0 = interpolate_fine_to_coarse(E1, E0, fine_data.e_fine_to_coarse_maps)
    E1 = interpolate_coarse_to_fine(E0, E1, fine_data.e_coarse_to_fine_maps)
    return E0, E1


def synchronize_b_levels(B_levels, dynamic_parameters):
    """Fill the coarse and fine B ghost cells without touching deep shadow."""

    B0, B1 = B_levels
    fine_data = dynamic_parameters.fmr.levels[1]
    B0 = interpolate_fine_to_coarse(B1, B0, fine_data.b_fine_to_coarse_maps)
    B1 = interpolate_coarse_to_fine(B0, B1, fine_data.b_coarse_to_fine_maps)
    return B0, B1
