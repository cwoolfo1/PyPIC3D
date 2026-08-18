"""Field updates and global B-E-B timestep orchestration for FMR."""

from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING

from .curls import fmr_curl_b_to_e, fmr_curl_e_to_b
from .interpolation import prolong_e_to_fine_interface


def update_B_fmr(E_levels, B_levels, static_parameters, dynamic_parameters):
    """Advance every active FMR B level by the leapfrog half step."""

    g = int(static_parameters.guard_cells)
    active = slice(g, -g)
    curl_E_levels = fmr_curl_e_to_b(E_levels, static_parameters, dynamic_parameters)
    dt = dynamic_parameters.dt / 2

    updated_levels = []
    for B_level, curl_level in zip(B_levels, curl_E_levels):
        updated_components = []
        for component, curl_component in zip(B_level, curl_level):
            component = component.at[:, :, :, active, active, active].add(-dt * curl_component)
            updated_components.append(component)
        updated_levels.append(tuple(updated_components))
    return tuple(updated_levels)


def _apply_root_conducting_boundaries(E0, static_parameters):
    Ex, Ey, Ez = E0
    g = int(static_parameters.guard_cells)
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
    return Ex, Ey, Ez


def update_E_fmr(E_levels, B_levels, J_levels, static_parameters, dynamic_parameters):
    """Advance active FMR E levels with the transpose-derived reverse curl."""

    g = int(static_parameters.guard_cells)
    active = slice(g, -g)
    curl_B_levels = fmr_curl_b_to_e(
        B_levels,
        E_levels,
        static_parameters,
        dynamic_parameters,
    )
    dt = dynamic_parameters.dt
    C = dynamic_parameters.C
    eps = dynamic_parameters.eps

    updated_levels = []
    for E_level, J_level, curl_level in zip(E_levels, J_levels, curl_B_levels):
        updated_components = []
        for component, current, curl_component in zip(E_level, J_level, curl_level):
            component = component.at[:, :, :, active, active, active].add(
                dt * (C**2 * curl_component - current[:, :, :, active, active, active] / eps)
            )
            updated_components.append(component)
        updated_levels.append(tuple(updated_components))

    E0 = _apply_root_conducting_boundaries(updated_levels[0], static_parameters)
    E1 = prolong_e_to_fine_interface(
        E0,
        updated_levels[1],
        dynamic_parameters.fmr.levels[1].e_interface_maps,
    )
    return E0, E1


def time_loop_electrodynamic_fmr_fields(
    particles,
    species_config,
    fields,
    static_parameters,
    dynamic_parameters,
):
    """Advance the field-only FMR state with one global B-E-B timestep."""

    del species_config
    E, B, J, rho, phi, external_fields, pml_state, overflow = fields

    B = update_B_fmr(E, B, static_parameters, dynamic_parameters)
    E = update_E_fmr(E, B, J, static_parameters, dynamic_parameters)
    B = update_B_fmr(E, B, static_parameters, dynamic_parameters)

    fields = (E, B, J, rho, phi, external_fields, pml_state, overflow)
    return particles, fields
