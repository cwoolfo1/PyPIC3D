"""Explicit two-level Yee curls for the field-only FMR solver."""

from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.solvers.yee.first_order_yee import _forward_difference

from .transfers import interpolate_coarse_to_fine, interpolate_fine_to_coarse


def _backward_difference(field, axis, spacing, guard_cells):
    """Take one backward staggered Yee difference from B to E."""

    g = int(guard_cells)
    active = slice(g, -g)
    previous = slice(g - 1, -g - 1)

    current_slices = [active, active, active]
    previous_slices = [active, active, active]
    previous_slices[int(axis)] = previous
    current_slices = (slice(None), slice(None), slice(None), *current_slices)
    previous_slices = (slice(None), slice(None), slice(None), *previous_slices)
    return (field[current_slices] - field[previous_slices]) / spacing


def _curl_e_to_b(E, spacing, guard_cells):
    """Apply the six forward Yee differences for Bx, By, and Bz."""

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


def _curl_b_to_e(B, spacing, guard_cells):
    """Apply the six backward Yee differences for Ex, Ey, and Ez."""

    Bx, By, Bz = B
    dx, dy, dz = spacing
    g = int(guard_cells)

    # Ex(V,C,C) = dBz(V,V,C)/dy - dBy(V,C,V)/dz
    dBz_dy = _backward_difference(Bz, 1, dy, g)
    dBy_dz = _backward_difference(By, 2, dz, g)
    curl_x = dBz_dy - dBy_dz

    # Ey(C,V,C) = dBx(C,V,V)/dz - dBz(V,V,C)/dx
    dBx_dz = _backward_difference(Bx, 2, dz, g)
    dBz_dx = _backward_difference(Bz, 0, dx, g)
    curl_y = dBx_dz - dBz_dx

    # Ez(C,C,V) = dBy(V,C,V)/dx - dBx(C,V,V)/dy
    dBy_dx = _backward_difference(By, 0, dx, g)
    dBx_dy = _backward_difference(Bx, 1, dy, g)
    curl_z = dBy_dx - dBx_dy
    return curl_x, curl_y, curl_z


def _synchronize_curl_inputs(
    field_levels,
    fine_to_coarse_maps,
    coarse_to_fine_maps,
    static_parameters,
):
    """Refresh every coarse/fine value read by an active composite curl."""

    g = int(static_parameters.guard_cells)
    coarse, fine = field_levels
    coarse = interpolate_fine_to_coarse(fine, coarse, fine_to_coarse_maps)
    coarse = ghost_cells.update_tiled_vector_ghost_cells(
        coarse,
        static_parameters,
        g,
    )
    fine = interpolate_coarse_to_fine(coarse, fine, coarse_to_fine_maps)
    return coarse, fine


def fmr_curl_e_to_b(E_levels, static_parameters, dynamic_parameters):
    """Apply the explicit composite E-to-B curl on the root and fine patch."""

    g = int(static_parameters.guard_cells)
    parent_level, fine_level = static_parameters.fmr_levels
    parent_runtime, fine_runtime = dynamic_parameters.fmr.levels
    interface = dynamic_parameters.fmr.interface

    E0_work, E1_work = _synchronize_curl_inputs(
        E_levels,
        interface.e_fine_to_coarse_maps,
        interface.e_coarse_to_fine_maps,
        static_parameters,
    )

    curl0 = _curl_e_to_b(E0_work, parent_level.spacing, g)
    curl1 = _curl_e_to_b(E1_work, fine_level.spacing, g)

    curl0 = tuple(
        mask * component
        for mask, component in zip(parent_runtime.b_active_masks, curl0)
    )
    curl1 = tuple(
        mask * component
        for mask, component in zip(fine_runtime.b_active_masks, curl1)
    )
    return curl0, curl1


def fmr_curl_b_to_e(B_levels, static_parameters, dynamic_parameters):
    """Apply the explicit composite B-to-E curl on the root and fine patch."""

    g = int(static_parameters.guard_cells)
    parent_level, fine_level = static_parameters.fmr_levels
    parent_runtime, fine_runtime = dynamic_parameters.fmr.levels
    interface = dynamic_parameters.fmr.interface

    B0_work, B1_work = _synchronize_curl_inputs(
        B_levels,
        interface.b_fine_to_coarse_maps,
        interface.b_coarse_to_fine_maps,
        static_parameters,
    )

    curl0 = _curl_b_to_e(B0_work, parent_level.spacing, g)
    curl1 = _curl_b_to_e(B1_work, fine_level.spacing, g)

    curl0 = tuple(
        mask * component
        for mask, component in zip(parent_runtime.e_active_masks, curl0)
    )
    curl1 = tuple(
        mask * component
        for mask, component in zip(fine_runtime.e_active_masks, curl1)
    )
    return curl0, curl1
