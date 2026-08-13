import jax

from PyPIC3D.boundary_conditions.PML import (
    apply_tiled_pml_to_b_curl,
    apply_tiled_pml_to_e_curl,
)
from PyPIC3D.boundary_conditions.supergaussian import apply_tiled_supergaussian_absorber
from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING
from PyPIC3D.utilities.filters import digital_filter_vector


def _active_vector(field_tiles, g):
    active = slice(g, -g)
    return tuple(component[:, :, :, active, active, active] for component in field_tiles)


def yee_curl_e_to_b(E_tiles, static_parameters, dynamic_parameters):
    """
    Apply the linear ordinary Yee curl from centered E to staggered B.

    Halo refresh is part of this spatial operator.  The current field boundary
    maps are linear: they copy neighboring/periodic values, copy the adjacent
    interior for constant boundaries, or insert homogeneous zeros at conducting
    boundaries.  No nonzero prescribed value is injected into this path.
    """

    g = int(static_parameters.guard_cells)
    active = slice(g, -g)
    forward = slice(g + 1, None if g == 1 else -g + 1)

    Ex, Ey, Ez = ghost_cells.update_tiled_vector_ghost_cells(E_tiles, static_parameters, g)
    dx, dy, dz = dynamic_parameters.dx, dynamic_parameters.dy, dynamic_parameters.dz

    dEz_dy = (Ez[:, :, :, active, forward, active] - Ez[:, :, :, active, active, active]) / dy
    dEy_dz = (Ey[:, :, :, active, active, forward] - Ey[:, :, :, active, active, active]) / dz
    dEx_dz = (Ex[:, :, :, active, active, forward] - Ex[:, :, :, active, active, active]) / dz
    dEx_dy = (Ex[:, :, :, active, forward, active] - Ex[:, :, :, active, active, active]) / dy
    dEz_dx = (Ez[:, :, :, forward, active, active] - Ez[:, :, :, active, active, active]) / dx
    dEy_dx = (Ey[:, :, :, forward, active, active] - Ey[:, :, :, active, active, active]) / dx

    return (
        dEz_dy - dEy_dz,
        dEx_dz - dEz_dx,
        dEy_dx - dEx_dy,
    )


def yee_curl_b_to_e(B_tiles, E_template, static_parameters, dynamic_parameters):
    """
    Apply the algebraic transpose of ``yee_curl_e_to_b`` to active B.

    Transposing the constant-copy halo map changes the two exterior wall
    planes relative to the legacy independently refreshed backward stencil.
    """

    g = int(static_parameters.guard_cells)
    transpose_curl = jax.linear_transpose(
        lambda E: yee_curl_e_to_b(E, static_parameters, dynamic_parameters),
        E_template,
    )
    curl_B, = transpose_curl(_active_vector(B_tiles, g))
    return _active_vector(curl_B, g)


def update_E(E_tiles, B_tiles, J_tiles, static_parameters, dynamic_parameters, pml_state=None):
    """
    Update compact tiled electric fields without assembling a global field.

    The ordinary B-to-E curl is the algebraic transpose of the centered-E to
    staggered-B spatial operator, including its halo communication.
    """

    Ex, Ey, Ez = E_tiles
    g = int(static_parameters.guard_cells)
    active = slice(g, -g)
    # build interior slice for active axes
    Jx, Jy, Jz = J_tiles

    dt = dynamic_parameters.dt
    C = dynamic_parameters.C
    eps = dynamic_parameters.eps

    if pml_state is None:
        curl_x, curl_y, curl_z = yee_curl_b_to_e(
            B_tiles,
            E_tiles,
            static_parameters,
            dynamic_parameters,
        )
    else:
        backward = slice(g - 1, -g - 1)
        Bx, By, Bz = ghost_cells.update_tiled_vector_ghost_cells(B_tiles, static_parameters, g)
        dx, dy, dz = dynamic_parameters.dx, dynamic_parameters.dy, dynamic_parameters.dz

        # PML evolves six split directional terms and cannot use the static
        # transpose operator used by the ordinary Yee update.
        dBz_dy = (Bz[:, :, :, active, active, active] - Bz[:, :, :, active, backward, active]) / dy
        dBy_dz = (By[:, :, :, active, active, active] - By[:, :, :, active, active, backward]) / dz
        dBx_dz = (Bx[:, :, :, active, active, active] - Bx[:, :, :, active, active, backward]) / dz
        dBx_dy = (Bx[:, :, :, active, active, active] - Bx[:, :, :, active, backward, active]) / dy
        dBz_dx = (Bz[:, :, :, active, active, active] - Bz[:, :, :, backward, active, active]) / dx
        dBy_dx = (By[:, :, :, active, active, active] - By[:, :, :, backward, active, active]) / dx

        (curl_x, curl_y, curl_z), pml_state = apply_tiled_pml_to_e_curl(
            (dBz_dy, dBy_dz, dBx_dz, dBz_dx, dBy_dx, dBx_dy),
            static_parameters,
            dynamic_parameters,
            pml_state,
        )

    Ex = Ex.at[:, :, :, active, active, active].set(
        Ex[:, :, :, active, active, active]
        + (C**2 * curl_x - Jx[:, :, :, active, active, active] / eps) * dt
    )
    Ey = Ey.at[:, :, :, active, active, active].set(
        Ey[:, :, :, active, active, active]
        + (C**2 * curl_y - Jy[:, :, :, active, active, active] / eps) * dt
    )
    Ez = Ez.at[:, :, :, active, active, active].set(
        Ez[:, :, :, active, active, active]
        + (C**2 * curl_z - Jz[:, :, :, active, active, active] / eps) * dt
    )

    Ex, Ey, Ez = ghost_cells.update_tiled_vector_ghost_cells((Ex, Ey, Ez), static_parameters, g)
    # refresh tile halos before the digital field filter, matching the global
    # ghost-cell order in the standard Yee solver.

    Ex, Ey, Ez = digital_filter_vector((Ex, Ey, Ez), dynamic_parameters.alpha, num_guard_cells=g)

    bc_x, bc_y, bc_z = static_parameters.boundary_conditions
    if int(bc_x) == BC_CONDUCTING:
        Ey = ghost_cells.apply_tiled_zero_boundary(Ey, static_parameters, axis=0, num_guard_cells=g)
        Ez = ghost_cells.apply_tiled_zero_boundary(Ez, static_parameters, axis=0, num_guard_cells=g)
    if int(bc_y) == BC_CONDUCTING:
        Ex = ghost_cells.apply_tiled_zero_boundary(Ex, static_parameters, axis=1, num_guard_cells=g)
        Ez = ghost_cells.apply_tiled_zero_boundary(Ez, static_parameters, axis=1, num_guard_cells=g)
    if int(bc_z) == BC_CONDUCTING:
        Ex = ghost_cells.apply_tiled_zero_boundary(Ex, static_parameters, axis=2, num_guard_cells=g)
        Ey = ghost_cells.apply_tiled_zero_boundary(Ey, static_parameters, axis=2, num_guard_cells=g)
    # conducting walls zero tangential E components on the physical boundary
    # planes; the shared scalar helper refreshes halos through ppermute.

    E_tiles = apply_tiled_supergaussian_absorber(
        (Ex, Ey, Ez),
        static_parameters,
        dynamic_parameters,
        dynamic_parameters.dt,
    )
    # A supergaussian layer is a field-only sponge: it damps the evolved fields
    # after the Maxwell update without changing the deposited current.

    return ghost_cells.update_tiled_vector_ghost_cells(E_tiles, static_parameters, g), pml_state


def update_B(E_tiles, B_tiles, static_parameters, dynamic_parameters, pml_state=None, do_filter=False):
    """
    Update compact tiled magnetic fields without assembling a global field.

    The Yee curl is evaluated on each tile's physical interior after E halos
    have been refreshed from neighbor tiles or field boundary conditions.
    """

    Bx, By, Bz = B_tiles
    g = int(static_parameters.guard_cells)
    active = slice(g, -g)
    # build interior slice for active axes
    dt = dynamic_parameters.dt / 2  # half timestep for B update

    if pml_state is None:
        curl_x, curl_y, curl_z = yee_curl_e_to_b(
            E_tiles,
            static_parameters,
            dynamic_parameters,
        )
    else:
        forward = slice(g + 1, None if g == 1 else -g + 1)
        Ex, Ey, Ez = ghost_cells.update_tiled_vector_ghost_cells(E_tiles, static_parameters, g)
        dx, dy, dz = dynamic_parameters.dx, dynamic_parameters.dy, dynamic_parameters.dz

        # PML evolves six split directional terms and keeps the explicit
        # forward derivatives needed by its auxiliary magnetic state.
        dEz_dy = (Ez[:, :, :, active, forward, active] - Ez[:, :, :, active, active, active]) / dy
        dEy_dz = (Ey[:, :, :, active, active, forward] - Ey[:, :, :, active, active, active]) / dz
        dEx_dz = (Ex[:, :, :, active, active, forward] - Ex[:, :, :, active, active, active]) / dz
        dEx_dy = (Ex[:, :, :, active, forward, active] - Ex[:, :, :, active, active, active]) / dy
        dEz_dx = (Ez[:, :, :, forward, active, active] - Ez[:, :, :, active, active, active]) / dx
        dEy_dx = (Ey[:, :, :, forward, active, active] - Ey[:, :, :, active, active, active]) / dx

        (curl_x, curl_y, curl_z), pml_state = apply_tiled_pml_to_b_curl(
            (dEz_dy, dEy_dz, dEx_dz, dEz_dx, dEy_dx, dEx_dy),
            static_parameters,
            dynamic_parameters,
            pml_state,
        )

    Bx = Bx.at[:, :, :, active, active, active].set(Bx[:, :, :, active, active, active] - dt * curl_x)
    By = By.at[:, :, :, active, active, active].set(By[:, :, :, active, active, active] - dt * curl_y)
    Bz = Bz.at[:, :, :, active, active, active].set(Bz[:, :, :, active, active, active] - dt * curl_z)

    Bx, By, Bz = apply_tiled_supergaussian_absorber(
        (Bx, By, Bz),
        static_parameters,
        dynamic_parameters,
        dt,
    )
    # The B update is split into half steps, so the sponge uses the same half
    # timestep as Faraday's-law update here.

    def apply_filter(Bx, By, Bz):
        Bx, By, Bz = ghost_cells.update_tiled_vector_ghost_cells((Bx, By, Bz), static_parameters, g)
        # refresh tile halos before the digital field filter, matching the global
        # ghost-cell order in the standard Yee solver.
        Bx, By, Bz = digital_filter_vector((Bx, By, Bz), dynamic_parameters.alpha, num_guard_cells=g)
        # apply the digital filter to the updated B fields
        return (Bx, By, Bz) 


    Bx, By, Bz = jax.lax.cond(
        do_filter,
        lambda _: apply_filter(Bx, By, Bz),
        lambda _: (Bx, By, Bz),
        operand=None,
    )
    # if requested, apply the digital filter to the updated B fields, matching the global ghost-cell order in the standard Yee solver.

    return ghost_cells.update_tiled_vector_ghost_cells((Bx, By, Bz), static_parameters, g), pml_state
