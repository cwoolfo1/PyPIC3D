"""Explicit two-level Yee curl used by ``dB/dt = -curl(E)``."""

from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONSTANT
from PyPIC3D.solvers.yee.first_order_yee import _forward_difference

from .fields import synchronize_e_levels
from .interpolation import prolong_e_to_fine_interface


def _fine_static_view(static_parameters):
    fine_level = static_parameters.fmr_levels[1]
    return static_parameters._replace(
        tile_shape=fine_level.tile_shape,
        boundary_conditions=(BC_CONSTANT, BC_CONSTANT, BC_CONSTANT),
        fmr_enabled=False,
        fmr_levels=(),
    )


def _curl_e_to_b(E, spacing, guard_cells):
    """Apply the six visible forward Yee differences for Bx, By, and Bz."""

    Ex, Ey, Ez = E
    dx, dy, dz = spacing
    g = int(guard_cells)

    # Bx(C,V,V) = dEz(C,C,V)/dy - dEy(C,V,C)/dz
    dEz_dy = _forward_difference(Ez, 1, dy, g)
    dEy_dz = _forward_difference(Ey, 2, dz, g)
    curl_x = dEz_dy - dEy_dz

    # By(V,C,V) = dEx(V,C,C)/dz - dEz(C,C,V)/dx
    dEx_dz = _forward_difference(Ex, 2, dz, g)
    dEz_dx = _forward_difference(Ez, 0, dx, g)
    curl_y = dEx_dz - dEz_dx

    # Bz(V,V,C) = dEy(C,V,C)/dx - dEx(V,C,C)/dy
    dEy_dx = _forward_difference(Ey, 0, dx, g)
    dEx_dy = _forward_difference(Ex, 1, dy, g)
    curl_z = dEy_dx - dEx_dy
    return curl_x, curl_y, curl_z


def fmr_curl_e_to_b(E_levels, static_parameters, dynamic_parameters):
    """Apply the explicit composite E-to-B curl on the root and fine patch."""

    g = int(static_parameters.guard_cells)
    parent_level, fine_level = static_parameters.fmr_levels
    parent_data, fine_data = dynamic_parameters.fmr.levels

    # Fine-owned E is first restricted into the coarse shadow. The same current
    # coarse state then supplies the fourth-order fine-interface values.
    E0_work, E1_work = synchronize_e_levels(E_levels, dynamic_parameters)
    E0_work = ghost_cells.update_tiled_vector_ghost_cells(
        E0_work,
        static_parameters,
        g,
    )
    E1_work = ghost_cells.update_tiled_vector_ghost_cells(
        E1_work,
        _fine_static_view(static_parameters),
        g,
    )
    # Some Yee interface coordinates live in the first fine ghost layer (for
    # example a C coordinate at the upper patch face). Set them after the
    # constant halo refresh so the physical interface value is not extrapolated.
    E1_work = prolong_e_to_fine_interface(
        E0_work,
        E1_work,
        fine_data.e_interface_maps,
    )

    curl0 = _curl_e_to_b(E0_work, parent_level.spacing, g)
    curl1 = _curl_e_to_b(E1_work, fine_level.spacing, g)

    curl0 = tuple(mask * component for mask, component in zip(parent_data.b_active_masks, curl0))
    curl1 = tuple(mask * component for mask, component in zip(fine_data.b_active_masks, curl1))
    return curl0, curl1
