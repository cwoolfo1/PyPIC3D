"""Composite forward curl C and its weighted discrete adjoint.

The reverse operator is

    C† = M_E^-1 C.T M_B.

The transpose is obtained with ``jax.linear_transpose``. Because C contains
the coarse-to-fine prolongation P, the transposed operator naturally contains
P^T, so no separately implemented restriction stencil is needed.
"""

import jax

from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONSTANT
from PyPIC3D.solvers.yee.first_order_yee import (
    assemble_yee_curl,
    yee_derivatives_e_to_b_refreshed,
)

from .interpolation import prolong_e_to_fine_interface
from .weights import _apply_inverse_weights, _apply_weights


def _active_vector(field_tiles, guard_cells):
    g = int(guard_cells)
    active = slice(g, -g)
    return tuple(component[:, :, :, active, active, active] for component in field_tiles)


def _fine_static_view(static_parameters):
    fine_level = static_parameters.fmr_levels[1]
    return static_parameters._replace(
        tile_shape=fine_level.tile_shape,
        boundary_conditions=(BC_CONSTANT, BC_CONSTANT, BC_CONSTANT),
        fmr_enabled=False,
        fmr_levels=(),
    )


def fmr_curl_e_to_b(E_levels, static_parameters, dynamic_parameters):
    """Apply the one canonical two-level FMR Maxwell spatial operator."""

    E0, E1 = E_levels
    g = int(static_parameters.guard_cells)
    fine_level = static_parameters.fmr_levels[1]
    parent_data, fine_data = dynamic_parameters.fmr.levels

    E0_work = ghost_cells.update_tiled_vector_ghost_cells(E0, static_parameters, g)
    E1_work = ghost_cells.update_tiled_vector_ghost_cells(E1, _fine_static_view(static_parameters), g)
    E1_work = prolong_e_to_fine_interface(E0_work, E1_work, fine_data.e_interface_maps)

    derivatives0 = yee_derivatives_e_to_b_refreshed(
        E0_work,
        (dynamic_parameters.dx, dynamic_parameters.dy, dynamic_parameters.dz),
        g,
    )
    derivatives1 = yee_derivatives_e_to_b_refreshed(E1_work, fine_level.spacing, g)
    curl0 = assemble_yee_curl(derivatives0)
    curl1 = assemble_yee_curl(derivatives1)

    curl0 = tuple(mask * component for mask, component in zip(parent_data.b_active_masks, curl0))
    curl1 = tuple(mask * component for mask, component in zip(fine_data.b_active_masks, curl1))
    return curl0, curl1


def fmr_curl_b_to_e(B_levels, E_template, static_parameters, dynamic_parameters):
    """Apply the metric-weighted FMR adjoint M_E^-1 C.T M_B.

    ``C.T`` supplies the fine-to-coarse contribution by transposing the
    configured coarse-to-fine prolongation used in ``fmr_curl_e_to_b``. There
    is deliberately no separately implemented restriction stencil.
    """

    g = int(static_parameters.guard_cells)
    B_active_levels = tuple(_active_vector(B_level, g) for B_level in B_levels)
    B_weighted_levels = tuple(
        _apply_weights(B_level, level_data.b_weights)
        for B_level, level_data in zip(B_active_levels, dynamic_parameters.fmr.levels)
    )
    transpose = jax.linear_transpose(
        lambda E: fmr_curl_e_to_b(E, static_parameters, dynamic_parameters),
        E_template,
    )
    transposed_E, = transpose(B_weighted_levels)
    transposed_E_active = tuple(_active_vector(E_level, g) for E_level in transposed_E)
    return tuple(
        _apply_inverse_weights(E_level, level_data.e_weights)
        for E_level, level_data in zip(transposed_E_active, dynamic_parameters.fmr.levels)
    )
