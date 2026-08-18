import math
import time
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import (
    BC_CONSTANT,
    BC_CONDUCTING,
    BC_PERIODIC,
)
from PyPIC3D.initialization import initialize_fields
from PyPIC3D.solvers.yee.fmr import (
    B_FIELD_LOCATIONS,
    E_FIELD_LOCATIONS,
    build_fmr_fields,
    build_fmr_parameters,
    load_fmr_from_toml,
    load_fmr_interpolation_order,
    prolong_e_to_fine_interface,
    time_loop_electrodynamic_fmr_fields,
)
from PyPIC3D.solvers.yee.fmr.grids import _component_coordinate_axes
from PyPIC3D.solvers.yee.first_order_yee import (
    _forward_difference as _yee_forward_difference,
    _mesh_adapted_difference,
)
from PyPIC3D.utilities.simulation_helpers import courant_condition
from tests.kernel_fixtures import kernel_parameters


jax.config.update("jax_enable_x64", True)


REGIONS = ("coarse", "fine", "interface")
# N is the number of physical intervals. Periodic storage has N points; the
# PEC cavity has N + 1 points so both conducting wall planes lie on the grid.
ROOT_RESOLUTIONS = (12, 24, 48)
DOMAIN_LENGTHS = (1.0, 1.0, 1.0)
PATCH_BOUNDS = ((0.25, 0.75),) * 3
REFINEMENT_RATIO = 2
INTERPOLATION_ORDER = 2
GUARD_CELLS = 2
CFL = 0.8


def _component_coordinates(grids, locations):
    x_axis, y_axis, z_axis = _component_coordinate_axes(grids, locations)
    return (
        x_axis[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, jnp.newaxis, jnp.newaxis],
        y_axis[jnp.newaxis, jnp.newaxis, jnp.newaxis, jnp.newaxis, :, jnp.newaxis],
        z_axis[jnp.newaxis, jnp.newaxis, jnp.newaxis, jnp.newaxis, jnp.newaxis, :],
    )


def _tm111_fields(grids, time_value, wave_speed):
    """Evaluate the source-free TM111 mode on all six Yee component grids."""

    lengths = jnp.asarray(DOMAIN_LENGTHS)
    kx, ky, kz = jnp.pi / lengths
    kc_squared = kx**2 + ky**2
    k_squared = kc_squared + kz**2
    omega = wave_speed * jnp.sqrt(k_squared)
    time_cosine = jnp.cos(omega * time_value)
    time_sine = jnp.sin(omega * time_value)

    E = []
    for component, locations in enumerate(E_FIELD_LOCATIONS):
        x, y, z = _component_coordinates(grids, locations)
        if component == 0:
            value = (
                -kz * kx / kc_squared
                * jnp.cos(kx * x)
                * jnp.sin(ky * y)
                * jnp.sin(kz * z)
            )
        elif component == 1:
            value = (
                -kz * ky / kc_squared
                * jnp.sin(kx * x)
                * jnp.cos(ky * y)
                * jnp.sin(kz * z)
            )
        else:
            value = (
                jnp.sin(kx * x)
                * jnp.sin(ky * y)
                * jnp.cos(kz * z)
            )
        E.append(value * time_cosine)

    # These signs give dB/dt = -curl(E) and dE/dt = c^2 curl(B).
    B = []
    for component, locations in enumerate(B_FIELD_LOCATIONS):
        x, y, z = _component_coordinates(grids, locations)
        if component == 0:
            value = (
                -ky * k_squared / (omega * kc_squared)
                * jnp.sin(kx * x)
                * jnp.cos(ky * y)
                * jnp.cos(kz * z)
            )
        elif component == 1:
            value = (
                kx * k_squared / (omega * kc_squared)
                * jnp.cos(kx * x)
                * jnp.sin(ky * y)
                * jnp.cos(kz * z)
            )
        else:
            value = jnp.zeros(jnp.broadcast_shapes(x.shape, y.shape, z.shape))
        B.append(value * time_sine)

    return tuple(E), tuple(B)


def _periodic_fields(grids, time_value, wave_speed):
    """Evaluate a transverse (1, 1, 1) periodic plane wave on the Yee grid."""

    lengths = jnp.asarray(DOMAIN_LENGTHS)
    wave_vector = 2.0 * jnp.pi / lengths
    wave_number = jnp.linalg.norm(wave_vector)
    omega = wave_speed * wave_number

    kx, ky, _kz = wave_vector
    polarization = jnp.asarray((ky, -kx, 0.0)) / jnp.sqrt(kx**2 + ky**2)
    magnetic_polarization = jnp.cross(wave_vector, polarization) / omega

    E = []
    for amplitude, locations in zip(polarization, E_FIELD_LOCATIONS):
        x, y, z = _component_coordinates(grids, locations)
        phase = wave_vector[0] * x + wave_vector[1] * y + wave_vector[2] * z - omega * time_value
        E.append(amplitude * jnp.cos(phase))

    B = []
    for amplitude, locations in zip(magnetic_polarization, B_FIELD_LOCATIONS):
        x, y, z = _component_coordinates(grids, locations)
        phase = wave_vector[0] * x + wave_vector[1] * y + wave_vector[2] * z - omega * time_value
        B.append(amplitude * jnp.cos(phase))

    return tuple(E), tuple(B)


def _active_vector(field, guard_cells):
    g = int(guard_cells)
    active = slice(g, -g)
    return tuple(component[:, :, :, active, active, active] for component in field)


def _initialize_analytic_fields(
    E_levels,
    B_levels,
    static_parameters,
    dynamic_parameters,
    analytic_fields,
):
    g = int(static_parameters.guard_cells)
    active = slice(g, -g)
    initialized_E = []
    initialized_B = []

    for E_level, B_level, level_data in zip(
        E_levels,
        B_levels,
        dynamic_parameters.fmr.levels,
    ):
        exact_E, exact_B = analytic_fields(
            level_data.grids,
            0.0,
            dynamic_parameters.C,
        )
        initialized_E.append(tuple(
            component.at[:, :, :, active, active, active].set(
                exact[:, :, :, active, active, active]
            )
            for component, exact in zip(E_level, exact_E)
        ))
        initialized_B.append(tuple(
            component.at[:, :, :, active, active, active].set(
                exact[:, :, :, active, active, active] * active_mask
            )
            for component, exact, active_mask in zip(
                B_level,
                exact_B,
                level_data.b_active_masks,
            )
        ))

    initialized_E[1] = prolong_e_to_fine_interface(
        initialized_E[0],
        initialized_E[1],
        dynamic_parameters.fmr.levels[1].e_interface_maps,
    )
    return tuple(initialized_E), tuple(initialized_B)


def _coordinate_mesh(grids, locations, level, guard_cells):
    g = int(guard_cells)
    axes = _component_coordinate_axes(grids, locations)
    axes = tuple(
        axis[g:g + cells]
        for axis, cells in zip(axes, (level.Nx, level.Ny, level.Nz))
    )
    return jnp.meshgrid(*axes, indexing="ij")


def _inside_box(coordinates, bounds, padding=0.0):
    inside = jnp.ones(coordinates[0].shape, dtype=bool)
    for coordinate, (lower, upper) in zip(coordinates, bounds):
        inside &= (coordinate > lower + padding) & (coordinate < upper - padding)
    return inside


def _inside_outer_box(coordinates, bounds, padding):
    inside = jnp.ones(coordinates[0].shape, dtype=bool)
    for coordinate, (lower, upper) in zip(coordinates, bounds):
        inside &= (coordinate >= lower - padding) & (coordinate <= upper + padding)
    return inside


def _physical_coordinate_mask(coordinates, periodic):
    mask = jnp.ones(coordinates[0].shape, dtype=bool)
    tolerance = 1.0e-13
    for coordinate, length in zip(coordinates, DOMAIN_LENGTHS):
        if periodic:
            mask &= (coordinate >= -tolerance) & (coordinate < length - tolerance)
        else:
            mask &= (coordinate >= -tolerance) & (coordinate <= length + tolerance)
    return mask


def _component_region_mask(
    coordinates,
    level_index,
    region,
    interface_width,
    periodic,
):
    physical = _physical_coordinate_mask(coordinates, periodic)
    inside_patch = _inside_box(coordinates, PATCH_BOUNDS)
    inside_inner = _inside_box(coordinates, PATCH_BOUNDS, interface_width)
    inside_outer = _inside_outer_box(coordinates, PATCH_BOUNDS, interface_width)

    # Root values covered by the patch never enter an error region. The root
    # supplies the outside half of the shell, while the fine level supplies
    # its inside half, so the physical regions do not overlap.
    if level_index == 0 and region == "coarse":
        return physical & ~inside_outer
    if level_index == 0 and region == "interface":
        return physical & inside_outer & ~inside_patch
    if level_index == 1 and region == "fine":
        return physical & inside_inner
    if level_index == 1 and region == "interface":
        return physical & inside_patch & ~inside_inner
    return jnp.zeros(coordinates[0].shape, dtype=bool)


def _build_vector_region_masks(
    locations,
    static_parameters,
    dynamic_parameters,
    periodic,
    field_name,
):
    interface_width = 2.0 * float(dynamic_parameters.dx)
    masks = {region: [] for region in REGIONS}

    for level_index, (level, level_data) in enumerate(zip(
        static_parameters.fmr_levels,
        dynamic_parameters.fmr.levels,
    )):
        for region in REGIONS:
            level_masks = []
            for component_index, component_locations in enumerate(locations):
                coordinates = _coordinate_mesh(
                    level_data.grids,
                    component_locations,
                    level,
                    static_parameters.guard_cells,
                )
                mask = _component_region_mask(
                    coordinates,
                    level_index,
                    region,
                    interface_width,
                    periodic,
                )
                weights = (
                    level_data.e_weights
                    if field_name == "E"
                    else level_data.b_weights
                )
                mask &= weights[component_index][0, 0, 0] != 0.0
                level_masks.append(mask[jnp.newaxis, jnp.newaxis, jnp.newaxis])
            masks[region].append(tuple(level_masks))

    return {region: tuple(level_masks) for region, level_masks in masks.items()}


def _regional_vector_error(
    numerical_levels,
    exact_levels,
    region_masks,
    level_weights,
    guard_cells,
):
    numerical_active = tuple(_active_vector(level, guard_cells) for level in numerical_levels)
    exact_active = tuple(_active_vector(level, guard_cells) for level in exact_levels)
    errors = {}

    for region in REGIONS:
        squared_error = 0.0
        total_weight = 0.0
        for numerical, exact, masks, weights in zip(
            numerical_active,
            exact_active,
            region_masks[region],
            level_weights,
        ):
            for numerical_component, exact_component, mask, weight in zip(
                numerical,
                exact,
                masks,
                weights,
            ):
                regional_weight = weight * mask
                squared_error += jnp.sum(regional_weight * (numerical_component - exact_component) ** 2)
                total_weight += jnp.sum(regional_weight)
        errors[region] = float(jnp.sqrt(squared_error / total_weight))

    return errors


def _electromagnetic_energy(E_levels, B_levels, dynamic_parameters, guard_cells):
    E_active = tuple(_active_vector(level, guard_cells) for level in E_levels)
    B_active = tuple(_active_vector(level, guard_cells) for level in B_levels)
    electric_energy = 0.0
    magnetic_energy = 0.0

    for E_level, B_level, level_data in zip(
        E_active,
        B_active,
        dynamic_parameters.fmr.levels,
    ):
        electric_energy += sum(
            jnp.sum(weight * component**2)
            for component, weight in zip(E_level, level_data.e_weights)
        )
        magnetic_energy += sum(
            jnp.sum(weight * component**2)
            for component, weight in zip(B_level, level_data.b_weights)
        )

    return (
        0.5 * dynamic_parameters.eps * electric_energy
        + 0.5 / dynamic_parameters.mu * magnetic_energy
    )


def _magnetic_divergence(
    B_level,
    level_index,
    static_parameters,
    spacing,
):
    """Apply the level's forward Yee divergence at V-V-V locations.

    The root uses the same MAD derivative family as its production curl. The
    fine level uses ordinary forward Yee differences. This makes div(curl(E))
    the relevant level-local algebraic identity for the evolved FMR fields.
    """

    g = int(static_parameters.guard_cells)
    if level_index == 0:
        B_work = ghost_cells.update_tiled_vector_ghost_cells(
            B_level,
            static_parameters,
            g,
        )
        alpha = 1.0 / static_parameters.fmr_levels[1].refinement_ratio**2
        derivative = _mesh_adapted_difference
        derivative_arguments = (g, alpha)
    else:
        fine_level = static_parameters.fmr_levels[1]
        fine_static_parameters = static_parameters._replace(
            tile_shape=fine_level.tile_shape,
            boundary_conditions=(BC_CONSTANT, BC_CONSTANT, BC_CONSTANT),
            fmr_enabled=False,
            fmr_levels=(),
        )
        B_work = ghost_cells.update_tiled_vector_ghost_cells(
            B_level,
            fine_static_parameters,
            g,
        )
        derivative = _yee_forward_difference
        derivative_arguments = (g,)

    Bx, By, Bz = B_work
    return (
        derivative(Bx, 0, spacing[0], *derivative_arguments)
        + derivative(By, 1, spacing[1], *derivative_arguments)
        + derivative(Bz, 2, spacing[2], *derivative_arguments)
    )


def _divergence_valid_mask(level_data, level_index):
    """Keep divergence points whose complete B stencil is level-owned."""

    Bx_mask, By_mask, Bz_mask = level_data.b_active_masks
    if level_index == 0:
        def stencil_is_owned(mask, axis):
            return (
                jnp.roll(mask, 1, axis=axis)
                & mask
                & jnp.roll(mask, -1, axis=axis)
                & jnp.roll(mask, -2, axis=axis)
            )

        valid = (
            stencil_is_owned(Bx_mask, 3)
            & stencil_is_owned(By_mask, 4)
            & stencil_is_owned(Bz_mask, 5)
        )
    else:
        valid = (
            Bx_mask
            & jnp.roll(Bx_mask, -1, axis=3)
            & By_mask
            & jnp.roll(By_mask, -1, axis=4)
            & Bz_mask
            & jnp.roll(Bz_mask, -1, axis=5)
        )
    return valid


def _build_divergence_region_masks(
    static_parameters,
    dynamic_parameters,
    periodic,
):
    interface_width = 2.0 * float(dynamic_parameters.dx)
    masks = {region: [] for region in REGIONS}
    divergence_locations = ("V", "V", "V")

    for level_index, (level, level_data) in enumerate(zip(
        static_parameters.fmr_levels,
        dynamic_parameters.fmr.levels,
    )):
        coordinates = _coordinate_mesh(
            level_data.grids,
            divergence_locations,
            level,
            static_parameters.guard_cells,
        )
        valid = _divergence_valid_mask(
            level_data,
            level_index,
        )
        for region in REGIONS:
            region_mask = _component_region_mask(
                coordinates,
                level_index,
                region,
                interface_width,
                periodic,
            )
            masks[region].append(
                valid & region_mask[jnp.newaxis, jnp.newaxis, jnp.newaxis]
            )

    return {region: tuple(level_masks) for region, level_masks in masks.items()}


def _magnetic_divergence_norms(
    B_levels,
    static_parameters,
    dynamic_parameters,
    region_masks,
):
    divergence_levels = tuple(
        _magnetic_divergence(
            B_level,
            level_index,
            static_parameters,
            level.spacing,
        )
        for level_index, (B_level, level) in enumerate(zip(
            B_levels,
            static_parameters.fmr_levels,
        ))
    )
    volumes = tuple(jnp.prod(jnp.asarray(level.spacing)) for level in static_parameters.fmr_levels)
    l2_norms = []
    infinity_norms = []

    for region in REGIONS:
        squared_divergence = 0.0
        total_volume = 0.0
        maximum = 0.0
        for divergence, mask, volume in zip(
            divergence_levels,
            region_masks[region],
            volumes,
        ):
            squared_divergence += volume * jnp.sum(jnp.where(mask, divergence**2, 0.0))
            total_volume += volume * jnp.sum(mask)
            maximum = jnp.maximum(maximum, jnp.max(jnp.where(mask, jnp.abs(divergence), 0.0)))
        l2_norms.append(jnp.sqrt(squared_divergence / total_volume))
        infinity_norms.append(maximum)

    return jnp.stack(l2_norms), jnp.stack(infinity_norms)


def _interface_constraint_residual(E_levels, dynamic_parameters):
    prolonged = prolong_e_to_fine_interface(
        E_levels[0],
        E_levels[1],
        dynamic_parameters.fmr.levels[1].e_interface_maps,
    )
    residual = 0.0
    for actual, expected, interpolation_map in zip(
        E_levels[1],
        prolonged,
        dynamic_parameters.fmr.levels[1].e_interface_maps,
    ):
        target = interpolation_map.target_indices
        difference = actual[
            0,
            0,
            0,
            target[:, 0],
            target[:, 1],
            target[:, 2],
        ] - expected[
            0,
            0,
            0,
            target[:, 0],
            target[:, 1],
            target[:, 2],
        ]
        residual = jnp.maximum(residual, jnp.max(jnp.abs(difference)))
    return float(residual)


def _build_fmr_case(problem, resolution):
    periodic = problem == "periodic"
    root_points = resolution if periodic else resolution + 1
    spacing = 1.0 / resolution
    boundary_condition = BC_PERIODIC if periodic else BC_CONDUCTING

    static_parameters, dynamic_parameters = kernel_parameters(
        Nx=root_points,
        Ny=root_points,
        Nz=root_points,
        x_wind=DOMAIN_LENGTHS[0],
        y_wind=DOMAIN_LENGTHS[1],
        z_wind=DOMAIN_LENGTHS[2],
        x_min=0.0,
        y_min=0.0,
        z_min=0.0,
        dx=spacing,
        dy=spacing,
        dz=spacing,
        dt=1.0,
        tile_shape=(root_points, root_points, root_points),
        guard_cells=GUARD_CELLS,
        boundary_conditions=(boundary_condition,) * 3,
        current_filter="none",
        C=1.0,
        eps=1.0,
        mu=1.0,
        pml_active=False,
        supergaussian_active=False,
    )

    patch_start = (resolution // 4,) * 3
    patch_stop = (3 * resolution // 4,) * 3
    config = {
        "fmr": {
            "enabled": True,
            "interpolation_order": INTERPOLATION_ORDER,
            "levels": [
                {
                    "parent": 0,
                    "refinement_ratio": REFINEMENT_RATIO,
                    "coarse_start": list(patch_start),
                    "coarse_stop": list(patch_stop),
                }
            ],
        }
    }
    geometry_values = {
        "Nx": root_points,
        "Ny": root_points,
        "Nz": root_points,
        "dx": spacing,
        "dy": spacing,
        "dz": spacing,
        "x_min": 0.0,
        "x_max": DOMAIN_LENGTHS[0],
        "y_min": 0.0,
        "y_max": DOMAIN_LENGTHS[1],
        "z_min": 0.0,
        "z_max": DOMAIN_LENGTHS[2],
    }
    levels = load_fmr_from_toml(
        config,
        geometry_values,
        static_parameters.tile_shape,
    )
    static_parameters = static_parameters._replace(
        fmr_enabled=True,
        fmr_levels=levels,
        fmr_interpolation_order=load_fmr_interpolation_order(config),
    )
    dynamic_parameters = dynamic_parameters._replace(
        fmr=build_fmr_parameters(static_parameters, dynamic_parameters)
    )

    wave_number = math.sqrt(3.0) * (2.0 * math.pi if periodic else math.pi)
    period = 2.0 * math.pi / (float(dynamic_parameters.C) * wave_number)
    final_time = period / 3.0
    fine_spacing = static_parameters.fmr_levels[-1].spacing
    dt_max = float(courant_condition(CFL, *fine_spacing, dynamic_parameters))
    number_of_steps = math.ceil(final_time / dt_max)
    dynamic_parameters = dynamic_parameters._replace(
        dt=jnp.asarray(final_time / number_of_steps)
    )

    return static_parameters, dynamic_parameters, final_time, number_of_steps


def _run_problem(problem, resolution):
    periodic = problem == "periodic"
    analytic_fields = _periodic_fields if periodic else _tm111_fields
    static_parameters, dynamic_parameters, final_time, number_of_steps = _build_fmr_case(
        problem,
        resolution,
    )

    E0, B0, J0, phi, rho = initialize_fields(static_parameters, dynamic_parameters)
    E_levels, B_levels, J_levels = build_fmr_fields(
        E0,
        B0,
        J0,
        static_parameters,
        dynamic_parameters,
    )
    E_levels, B_levels = _initialize_analytic_fields(
        E_levels,
        B_levels,
        static_parameters,
        dynamic_parameters,
        analytic_fields,
    )
    external_fields = (
        tuple(tuple(jnp.zeros_like(component) for component in level) for level in E_levels),
        tuple(tuple(jnp.zeros_like(component) for component in level) for level in B_levels),
    )
    fields = (
        E_levels,
        B_levels,
        J_levels,
        rho,
        phi,
        external_fields,
        None,
        jnp.asarray(False),
    )

    E_region_masks = _build_vector_region_masks(
        E_FIELD_LOCATIONS,
        static_parameters,
        dynamic_parameters,
        periodic,
        "E",
    )
    B_region_masks = _build_vector_region_masks(
        B_FIELD_LOCATIONS,
        static_parameters,
        dynamic_parameters,
        periodic,
        "B",
    )
    divergence_masks = _build_divergence_region_masks(
        static_parameters,
        dynamic_parameters,
        periodic,
    )

    initial_energy = _electromagnetic_energy(
        E_levels,
        B_levels,
        dynamic_parameters,
        static_parameters.guard_cells,
    )
    initial_divergence = _magnetic_divergence_norms(
        B_levels,
        static_parameters,
        dynamic_parameters,
        divergence_masks,
    )

    def diagnostics(field_state):
        E_now, B_now = field_state[:2]
        energy = _electromagnetic_energy(
            E_now,
            B_now,
            dynamic_parameters,
            static_parameters.guard_cells,
        )
        div_l2, div_infinity = _magnetic_divergence_norms(
            B_now,
            static_parameters,
            dynamic_parameters,
            divergence_masks,
        )
        return energy, div_l2, div_infinity

    def step(field_state, _unused):
        _, field_state = time_loop_electrodynamic_fmr_fields(
            (),
            (),
            field_state,
            static_parameters,
            dynamic_parameters,
        )
        return field_state, diagnostics(field_state)

    @jax.jit
    def evolve(field_state):
        return jax.lax.scan(
            step,
            field_state,
            xs=None,
            length=number_of_steps,
        )

    start_time = time.perf_counter()
    final_fields, history = evolve(fields)
    final_fields, history = jax.block_until_ready((final_fields, history))
    runtime = time.perf_counter() - start_time

    E_final, B_final = final_fields[:2]
    energy_history, divergence_l2_history, divergence_infinity_history = history
    exact_levels = tuple(
        analytic_fields(level_data.grids, final_time, dynamic_parameters.C)
        for level_data in dynamic_parameters.fmr.levels
    )
    exact_E_levels = tuple(fields_at_time[0] for fields_at_time in exact_levels)
    exact_B_levels = tuple(fields_at_time[1] for fields_at_time in exact_levels)
    E_weights = tuple(level_data.e_weights for level_data in dynamic_parameters.fmr.levels)
    B_weights = tuple(level_data.b_weights for level_data in dynamic_parameters.fmr.levels)

    E_l2 = _regional_vector_error(
        E_final,
        exact_E_levels,
        E_region_masks,
        E_weights,
        static_parameters.guard_cells,
    )
    B_l2 = _regional_vector_error(
        B_final,
        exact_B_levels,
        B_region_masks,
        B_weights,
        static_parameters.guard_cells,
    )
    EM_l2 = {
        region: math.sqrt(
            float(dynamic_parameters.eps) * E_l2[region] ** 2
            + B_l2[region] ** 2 / float(dynamic_parameters.mu)
        )
        for region in REGIONS
    }

    initial_energy_value = float(initial_energy)
    energy_history = np.asarray(energy_history)
    relative_energy = energy_history / initial_energy_value - 1.0
    initial_divergence_l2, initial_divergence_infinity = (
        np.asarray(value) for value in initial_divergence
    )
    divergence_l2_history = np.asarray(divergence_l2_history)
    divergence_infinity_history = np.asarray(divergence_infinity_history)

    leaves = jax.tree_util.tree_leaves((E_final, B_final))
    return {
        "H": 1.0 / resolution,
        "h": 1.0 / (REFINEMENT_RATIO * resolution),
        "dt": float(dynamic_parameters.dt),
        "steps": number_of_steps,
        "energy_initial": initial_energy_value,
        "energy_final": float(energy_history[-1]),
        "energy_final_relative_error": float(relative_energy[-1]),
        "energy_max_relative_error": float(np.max(np.abs(relative_energy))),
        "divB_initial": dict(zip(REGIONS, initial_divergence_l2.tolist())),
        "divB_initial_infinity": dict(zip(REGIONS, initial_divergence_infinity.tolist())),
        "divB": dict(zip(REGIONS, divergence_l2_history[-1].tolist())),
        "divB_infinity": dict(zip(REGIONS, divergence_infinity_history[-1].tolist())),
        "divB_max": dict(zip(REGIONS, np.max(divergence_l2_history, axis=0).tolist())),
        "divB_max_infinity": dict(
            zip(REGIONS, np.max(divergence_infinity_history, axis=0).tolist())
        ),
        "E_l2": E_l2,
        "B_l2": B_l2,
        "EM_l2": EM_l2,
        "interface_residual": _interface_constraint_residual(E_final, dynamic_parameters),
        "finite": all(bool(jnp.all(jnp.isfinite(component))) for component in leaves),
        "runtime": runtime,
    }


def _observed_orders(results, diagnostic):
    return {
        region: tuple(
            math.log(
                results[index][diagnostic][region]
                / results[index + 1][diagnostic][region],
                2.0,
            )
            for index in range(len(results) - 1)
        )
        for region in REGIONS
    }


def _result_table(problem, results, orders):
    lines = [
        f"{problem} FMR convergence",
        "",
        "N      H           dt          steps   coarse EM      fine EM        interface EM",
    ]
    for resolution, result in zip(ROOT_RESOLUTIONS, results):
        lines.append(
            f"{resolution:<6d} {result['H']:<11.4e} {result['dt']:<11.4e} "
            f"{result['steps']:<7d} {result['EM_l2']['coarse']:<14.6e} "
            f"{result['EM_l2']['fine']:<14.6e} {result['EM_l2']['interface']:<14.6e}"
        )
    for diagnostic in ("E_l2", "B_l2", "EM_l2"):
        lines.extend(("", f"Observed {diagnostic} orders:"))
        for region in REGIONS:
            lines.append(
                f"{region:<10s} p01={orders[diagnostic][region][0]:.6f}  "
                f"p12={orders[diagnostic][region][1]:.6f}"
            )
    lines.extend(("", "max |dU/U0|:"))
    lines.append("  ".join(f"{value['energy_max_relative_error']:.6e}" for value in results))
    lines.extend(("", "final div(B) L2 [coarse, fine, interface]:"))
    for result in results:
        lines.append("  ".join(f"{result['divB'][region]:.6e}" for region in REGIONS))
    return "\n".join(lines)


class TestFMRMaxwellConvergence(unittest.TestCase):
    def _assert_problem(self, problem):
        results = [_run_problem(problem, resolution) for resolution in ROOT_RESOLUTIONS]
        orders = {
            diagnostic: _observed_orders(results, diagnostic)
            for diagnostic in ("E_l2", "B_l2", "EM_l2")
        }
        message = _result_table(problem, results, orders)

        for result in results:
            self.assertTrue(result["finite"], msg=message)
            self.assertLess(result["interface_residual"], 2.0e-12, msg=message)
            self.assertLess(result["energy_max_relative_error"], 5.0e-2, msg=message)

        for region in REGIONS:
            self.assertGreater(orders["EM_l2"][region][0], 1.7, msg=message)
            self.assertGreater(orders["EM_l2"][region][1], 1.7, msg=message)
            self.assertLess(results[-1]["divB_max"][region], 2.0e-10, msg=message)

        energy_errors = [result["energy_max_relative_error"] for result in results]
        self.assertGreater(energy_errors[0], energy_errors[1], msg=message)
        self.assertGreater(energy_errors[1], energy_errors[2], msg=message)
        final_energy_errors = [
            abs(result["energy_final_relative_error"])
            for result in results
        ]
        self.assertGreater(final_energy_errors[0], final_energy_errors[1], msg=message)
        self.assertGreater(final_energy_errors[1], final_energy_errors[2], msg=message)
        return results, orders

    def test_tm111_pec_cavity_converges_with_one_fine_patch(self):
        self._assert_problem("TM111 PEC cavity")

    def test_periodic_plane_wave_converges_with_one_fine_patch(self):
        self._assert_problem("periodic")


if __name__ == "__main__":
    unittest.main()
