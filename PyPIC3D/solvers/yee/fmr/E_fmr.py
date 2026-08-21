"""Explicit two-level Yee curl used by ``dE/dt = C**2 curl(B) - J/eps``."""

from PyPIC3D.boundary_conditions import ghost_cells

from .interpolation import fill_b_coarse_halo, fill_b_fine_halo


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


def _curl_b_to_e(B, spacing, guard_cells):
    """Apply the six visible backward Yee differences for Ex, Ey, and Ez."""

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


def fmr_curl_b_to_e(B_levels, E_template, static_parameters, dynamic_parameters):
    """Apply the explicit composite B-to-E curl on the root and fine patch."""

    del E_template
    g = int(static_parameters.guard_cells)
    parent_level, fine_level = static_parameters.fmr_levels
    parent_data, fine_data = dynamic_parameters.fmr.levels

    # Refresh only the covered coarse values needed by an active curl or by the
    # fine-halo interpolation.  Root tile and physical ghosts remain a separate
    # operation, after which the current coarse field supplies every fine value
    # read across the refinement boundary.
    B0_work, B1_work = B_levels
    B0_work = fill_b_coarse_halo(
        B1_work,
        B0_work,
        fine_data.b_coarse_halo_maps,
    )
    B0_work = ghost_cells.update_tiled_vector_ghost_cells(
        B0_work,
        static_parameters,
        g,
    )
    B1_work = fill_b_fine_halo(
        B0_work,
        B1_work,
        fine_data.b_fine_halo_maps,
    )

    curl0 = _curl_b_to_e(B0_work, parent_level.spacing, g)
    curl1 = _curl_b_to_e(B1_work, fine_level.spacing, g)

    curl0 = tuple(mask * component for mask, component in zip(parent_data.e_active_masks, curl0))
    curl1 = tuple(mask * component for mask, component in zip(fine_data.e_active_masks, curl1))
    return curl0, curl1
