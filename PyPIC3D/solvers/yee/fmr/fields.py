"""Construction of FMR runtime metadata and level-major field arrays."""

import jax.numpy as jnp

from .grids import _build_level_grids
from .interpolation import build_e_interface_maps, prolong_e_to_fine_interface
from .types import FMRLevelData, FMRParameters
from .weights import build_b_active_masks, build_fmr_metric_weights


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
    e_interface_maps = build_e_interface_maps(
        parent_level,
        fine_level,
        dynamic_parameters.grids,
        fine_grids,
        static_parameters.guard_cells,
        static_parameters.fmr_interpolation_order,
    )
    parent_b_masks, fine_b_masks = build_b_active_masks(
        parent_level,
        fine_level,
        dynamic_parameters.grids,
        fine_grids,
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
        e_interface_maps,
        parent_b_masks,
        fine_b_masks,
        static_parameters.guard_cells,
    )

    parent_data = FMRLevelData(
        grids=dynamic_parameters.grids,
        e_interface_maps=(),
        b_active_masks=parent_b_masks,
        e_weights=parent_e_weights,
        b_weights=parent_b_weights,
    )
    fine_data = FMRLevelData(
        grids=fine_grids,
        e_interface_maps=e_interface_maps,
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

    E1 = prolong_e_to_fine_interface(
        E0,
        E1,
        dynamic_parameters.fmr.levels[1].e_interface_maps,
    )
    return (E0, E1), (B0, B1), (J0, J1)
