import jax

from PyPIC3D.boundary_conditions.PML import (
    stretch_tiled_pml_b_derivatives,
    stretch_tiled_pml_e_derivatives,
)
from PyPIC3D.boundary_conditions.supergaussian import apply_tiled_supergaussian_absorber
from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING
from PyPIC3D.utilities.filters import digital_filter_vector


def _active_vector(field_tiles, g):
    active = slice(g, -g)
    return tuple(component[:, :, :, active, active, active] for component in field_tiles)


def _forward_difference_from_refreshed(field, axis, spacing, guard_cells):
    """Take one forward Yee difference after the vector halo refresh."""

    g = int(guard_cells)
    active = slice(g, -g)
    forward = slice(g + 1, None if g == 1 else -g + 1)

    current_slices = [active, active, active]
    forward_slices = [active, active, active]
    forward_slices[int(axis)] = forward

    current_slices = (slice(None), slice(None), slice(None), *current_slices)
    forward_slices = (slice(None), slice(None), slice(None), *forward_slices)
    return (field[forward_slices] - field[current_slices]) / spacing


def _yee_derivatives_from_refreshed_channels(channels, static_parameters, dynamic_parameters):
    """Apply the six canonical forward differences to refreshed field channels."""

    Ez_for_dy, Ey_for_dz, Ex_for_dz, Ez_for_dx, Ey_for_dx, Ex_for_dy = channels
    g = int(static_parameters.guard_cells)

    return (
        _forward_difference_from_refreshed(Ez_for_dy, 1, dynamic_parameters.dy, g),
        _forward_difference_from_refreshed(Ey_for_dz, 2, dynamic_parameters.dz, g),
        _forward_difference_from_refreshed(Ex_for_dz, 2, dynamic_parameters.dz, g),
        _forward_difference_from_refreshed(Ez_for_dx, 0, dynamic_parameters.dx, g),
        _forward_difference_from_refreshed(Ey_for_dx, 0, dynamic_parameters.dx, g),
        _forward_difference_from_refreshed(Ex_for_dy, 1, dynamic_parameters.dy, g),
    )


def yee_derivatives_e_to_b(E_tiles, static_parameters, dynamic_parameters):
    """
    Return the six forward Yee derivatives from centered E to staggered B.

    Halo refresh is part of this spatial operator.  The current field boundary
    maps are linear and do not inject nonzero prescribed values.

    Output order is
    ``dEz_dy, dEy_dz, dEx_dz, dEz_dx, dEy_dx, dEx_dy``.
    """

    g = int(static_parameters.guard_cells)
    Ex, Ey, Ez = ghost_cells.update_tiled_vector_ghost_cells(E_tiles, static_parameters, g)
    channels = (Ez, Ey, Ex, Ez, Ey, Ex)
    return _yee_derivatives_from_refreshed_channels(channels, static_parameters, dynamic_parameters)


def _independent_yee_derivative_channels(channels, static_parameters, dynamic_parameters):
    """
    Apply the forward geometry to six independent channels for transposition.

    Keeping the derivative channels independent lets the transpose return each
    backward derivative separately.  The channels share one vector halo
    exchange, so the reverse operator has one distributed transpose/fold phase.
    """

    g = int(static_parameters.guard_cells)
    channels = ghost_cells.update_tiled_vector_ghost_cells(channels, static_parameters, g)
    return _yee_derivatives_from_refreshed_channels(channels, static_parameters, dynamic_parameters)


def yee_derivatives_b_to_e(B_tiles, E_template, static_parameters, dynamic_parameters):
    """
    Return the six backward Yee derivatives from staggered B to centered E.

    The reverse geometry is ``D_backward = -D_forward.T``.  A single batched
    transpose keeps the six derivative channels separate for PML stretching
    while sharing the transpose of the distributed halo operation.

    Output order is
    ``dBz_dy, dBy_dz, dBx_dz, dBz_dx, dBy_dx, dBx_dy``.
    """

    g = int(static_parameters.guard_cells)
    Ex, Ey, Ez = E_template
    Bx, By, Bz = _active_vector(B_tiles, g)

    channel_templates = (Ez, Ey, Ex, Ez, Ey, Ex)
    derivative_cotangents = (Bx, Bx, By, By, Bz, Bz)
    transpose_derivatives = jax.linear_transpose(
        lambda channels: _independent_yee_derivative_channels(
            channels,
            static_parameters,
            dynamic_parameters,
        ),
        channel_templates,
    )
    transposed_channels, = transpose_derivatives(derivative_cotangents)
    backward_channels = tuple(-component for component in _active_vector(transposed_channels, g))

    dBx_dy, dBx_dz, dBy_dz, dBy_dx, dBz_dx, dBz_dy = backward_channels
    return dBz_dy, dBy_dz, dBx_dz, dBz_dx, dBy_dx, dBx_dy


def assemble_yee_curl(derivatives):
    """Assemble a Yee curl from its six directional derivatives."""

    d1, d2, d3, d4, d5, d6 = derivatives
    return d1 - d2, d3 - d4, d5 - d6


def yee_curl_e_to_b(E_tiles, static_parameters, dynamic_parameters):
    """Apply the canonical forward Yee curl from centered E to staggered B."""

    derivatives = yee_derivatives_e_to_b(E_tiles, static_parameters, dynamic_parameters)
    return assemble_yee_curl(derivatives)


def yee_curl_b_to_e(B_tiles, E_template, static_parameters, dynamic_parameters):
    """Apply the transpose-derived Yee curl from staggered B to centered E."""

    derivatives = yee_derivatives_b_to_e(
        B_tiles,
        E_template,
        static_parameters,
        dynamic_parameters,
    )
    return assemble_yee_curl(derivatives)


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

    derivatives = yee_derivatives_b_to_e(
        B_tiles,
        E_tiles,
        static_parameters,
        dynamic_parameters,
    )
    if pml_state is not None:
        derivatives, pml_state = stretch_tiled_pml_e_derivatives(
            derivatives,
            static_parameters,
            dynamic_parameters,
            pml_state,
        )
    curl_x, curl_y, curl_z = assemble_yee_curl(derivatives)

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

    derivatives = yee_derivatives_e_to_b(
        E_tiles,
        static_parameters,
        dynamic_parameters,
    )
    if pml_state is not None:
        derivatives, pml_state = stretch_tiled_pml_b_derivatives(
            derivatives,
            static_parameters,
            dynamic_parameters,
            pml_state,
        )
    curl_x, curl_y, curl_z = assemble_yee_curl(derivatives)

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
