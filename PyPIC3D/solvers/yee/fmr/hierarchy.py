"""Construction of the FMR hierarchy and level-major field arrays."""

import jax.numpy as jnp

from .grids import _build_level_grids
from .quadrature import build_field_active_masks
from .transfers import (
    build_b_transfer_maps,
    build_e_transfer_maps,
    interpolate_coarse_to_fine,
    interpolate_fine_to_coarse,
)
from .types import (
    B_FIELD_LOCATIONS,
    E_FIELD_LOCATIONS,
    FMRHierarchy,
    FMRInterfaceData,
    FMRLevelRuntime,
)


def build_fmr_hierarchy(static_parameters, dynamic_parameters):
    """Build FMR grids, ownership masks, and interface transfers once."""

    if not static_parameters.fmr_enabled:
        return None
    if len(static_parameters.fmr_levels) != 2:
        raise ValueError("The first FMR implementation requires root and one fine level.")
    if int(static_parameters.guard_cells) < 2:
        raise ValueError("FMR mesh-adapted Yee differencing requires at least two guard cells.")
    if any(level.tile_shape != level.shape for level in static_parameters.fmr_levels):
        raise NotImplementedError(
            "FMR evolution currently requires one logical tile on every level."
        )

    parent_level, fine_level = static_parameters.fmr_levels
    fine_grids = _build_level_grids(fine_level, static_parameters.guard_cells)
    e_coarse_to_fine_maps, e_fine_to_coarse_maps = build_e_transfer_maps(
        parent_level,
        fine_level,
        dynamic_parameters.grids,
        fine_grids,
        static_parameters.guard_cells,
    )
    b_coarse_to_fine_maps, b_fine_to_coarse_maps = build_b_transfer_maps(
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
    parent_runtime = FMRLevelRuntime(
        grids=dynamic_parameters.grids,
        e_active_masks=parent_e_masks,
        b_active_masks=parent_b_masks,
    )
    fine_runtime = FMRLevelRuntime(
        grids=fine_grids,
        e_active_masks=fine_e_masks,
        b_active_masks=fine_b_masks,
    )
    interface = FMRInterfaceData(
        e_coarse_to_fine_maps=e_coarse_to_fine_maps,
        b_coarse_to_fine_maps=b_coarse_to_fine_maps,
        e_fine_to_coarse_maps=e_fine_to_coarse_maps,
        b_fine_to_coarse_maps=b_fine_to_coarse_maps,
    )
    return FMRHierarchy(levels=(parent_runtime, fine_runtime), interface=interface)


def _fine_vector(level, guard_cells, templates):
    g = int(guard_cells)
    shape = (1, 1, 1, *(cells + 2 * g for cells in level.shape))
    return tuple(jnp.zeros(shape, dtype=template.dtype) for template in templates)


def initialize_fmr_field_levels(E0, B0, J0, static_parameters, dynamic_parameters):
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
        dynamic_parameters.fmr.interface.e_coarse_to_fine_maps,
    )
    return (E0, E1), (B0, B1), (J0, J1)


def synchronize_e_levels(E_levels, dynamic_parameters):
    """Fill the coarse and fine E ghost cells without touching deep shadow."""

    E0, E1 = E_levels
    interface = dynamic_parameters.fmr.interface
    E0 = interpolate_fine_to_coarse(E1, E0, interface.e_fine_to_coarse_maps)
    E1 = interpolate_coarse_to_fine(E0, E1, interface.e_coarse_to_fine_maps)
    return E0, E1


def synchronize_b_levels(B_levels, dynamic_parameters):
    """Fill the coarse and fine B ghost cells without touching deep shadow."""

    B0, B1 = B_levels
    interface = dynamic_parameters.fmr.interface
    B0 = interpolate_fine_to_coarse(B1, B0, interface.b_fine_to_coarse_maps)
    B1 = interpolate_coarse_to_fine(B0, B1, interface.b_coarse_to_fine_maps)
    return B0, B1
