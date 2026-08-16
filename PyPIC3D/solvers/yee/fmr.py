from itertools import product
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONSTANT, BC_CONDUCTING
from PyPIC3D.utilities.grids import build_tiled_yee_grids, build_yee_grid
from PyPIC3D.utilities.parameters import GridParameters

from .first_order_yee import assemble_yee_curl, yee_derivatives_e_to_b_refreshed


E_FIELD_LOCATIONS = (("V", "C", "C"), ("C", "V", "C"), ("C", "C", "V"))
B_FIELD_LOCATIONS = (("C", "V", "V"), ("V", "C", "V"), ("V", "V", "C"))


class FMRLevel(NamedTuple):
    """Small, hashable geometry record for one fixed refinement level."""

    level: int
    parent: int
    refinement_ratio: int
    parent_start: tuple
    parent_stop: tuple
    Nx: int
    Ny: int
    Nz: int
    spacing: tuple
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    tile_shape: tuple


class FMRInterpolationMap(NamedTuple):
    """Precomputed tensor-product interpolation data for one E component."""

    target_indices: jax.Array
    source_indices: jax.Array
    weights: jax.Array


class FMRLevelData(NamedTuple):
    """JAX-array data associated with one statically described level."""

    grids: GridParameters
    e_interface_maps: tuple
    b_active_masks: tuple
    e_weights: tuple
    b_weights: tuple


class FMRParameters(NamedTuple):
    """Dynamic FMR maps, masks, and coordinates, ordered by level."""

    levels: tuple


def _three_ints(values, name):
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"FMR {name} must contain exactly three integer indices.")
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in values):
        raise ValueError(f"FMR {name} must contain exactly three integer indices.")
    return tuple(int(value) for value in values)


def _fmr_enabled(config):
    raw_fmr = config.get("fmr")
    if not raw_fmr:
        return False

    enabled = raw_fmr.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("FMR enabled must be true or false.")
    return enabled


def validate_fmr_configuration(config, static_config, plotting_parameters):
    """Reject runtime combinations outside the first field-only FMR scope."""

    raw_fmr = config.get("fmr")
    if not _fmr_enabled(config):
        return

    unsupported_options = sorted(set(raw_fmr) - {"enabled", "levels"})
    if unsupported_options:
        names = ", ".join(unsupported_options)
        raise NotImplementedError(f"Unsupported FMR option(s): {names}.")

    if static_config["solver"] != "electrodynamic_yee":
        raise NotImplementedError("FMR currently supports only solver='electrodynamic_yee'.")
    if config.get("pml"):
        raise NotImplementedError("PML is not supported with FMR yet.")
    if config.get("supergaussian"):
        raise NotImplementedError("The supergaussian absorber is not supported with FMR yet.")
    if any(key.startswith("particle") for key in config):
        raise NotImplementedError("Particle species cannot be coupled to FMR fields yet.")
    if any(key.startswith("field") for key in config):
        raise NotImplementedError("External or loaded fields cannot populate FMR levels yet.")
    if any(key.startswith("previous_field") for key in config):
        raise NotImplementedError("Previous-field restart cannot populate FMR levels yet.")

    unsupported_diagnostics = (
        "dump_fields",
        "plot_openpmd_fields",
        "plotvelocities",
        "plotchargedensity",
    )
    enabled_diagnostics = [name for name in unsupported_diagnostics if plotting_parameters.get(name, False)]
    if enabled_diagnostics:
        names = ", ".join(enabled_diagnostics)
        raise NotImplementedError(f"FMR field diagnostics are not level-aware yet: {names}.")


def load_fmr_from_toml(config, dynamic_config, root_tile_shape):
    """Parse one interior rectangular fine patch and derive its geometry."""

    raw_fmr = config.get("fmr")
    if not _fmr_enabled(config):
        return ()

    raw_levels = raw_fmr.get("levels", ())
    if len(raw_levels) != 1:
        raise ValueError("The first FMR implementation requires exactly one [[fmr.levels]] entry.")

    root_shape = tuple(int(dynamic_config[name]) for name in ("Nx", "Ny", "Nz"))
    root_tile_shape = tuple(int(width) for width in root_tile_shape)
    if root_tile_shape != root_shape:
        raise NotImplementedError(
            "FMR currently requires root tile grid = (1, 1, 1); "
            f"root shape is {root_shape} but tile shape is {root_tile_shape}."
        )

    raw_level = raw_levels[0]
    level_keys = {"parent", "refinement_ratio", "coarse_start", "coarse_stop"}
    unsupported_options = sorted(set(raw_level) - level_keys)
    if unsupported_options:
        names = ", ".join(unsupported_options)
        raise NotImplementedError(f"Unsupported FMR level option(s): {names}.")

    parent = raw_level.get("parent", -1)
    if isinstance(parent, bool) or not isinstance(parent, (int, np.integer)):
        raise ValueError("The first FMR fine level must have integer parent = 0.")
    parent = int(parent)
    if parent != 0:
        raise ValueError("The first FMR fine level must have integer parent = 0.")

    refinement_ratio = raw_level.get("refinement_ratio")
    if isinstance(refinement_ratio, bool) or not isinstance(refinement_ratio, (int, np.integer)):
        raise ValueError("FMR refinement_ratio must be an even integer greater than or equal to 2.")
    refinement_ratio = int(refinement_ratio)
    if refinement_ratio < 2 or refinement_ratio % 2 != 0:
        raise ValueError("FMR refinement_ratio must be an even integer greater than or equal to 2.")

    parent_start = _three_ints(raw_level.get("coarse_start"), "coarse_start")
    parent_stop = _three_ints(raw_level.get("coarse_stop"), "coarse_stop")
    for start, stop, cells in zip(parent_start, parent_stop, root_shape):
        if not 0 <= start < stop <= cells:
            raise ValueError("FMR bounds must satisfy 0 <= coarse_start < coarse_stop <= parent shape.")
        if start == 0 or stop == cells:
            raise ValueError("The FMR fine patch must be strictly interior to the root domain.")

    spacing = tuple(float(dynamic_config[name]) for name in ("dx", "dy", "dz"))
    lower = tuple(float(dynamic_config[f"{axis}_min"]) for axis in ("x", "y", "z"))
    upper = tuple(float(dynamic_config[f"{axis}_max"]) for axis in ("x", "y", "z"))

    root_level = FMRLevel(
        level=0,
        parent=-1,
        refinement_ratio=1,
        parent_start=(0, 0, 0),
        parent_stop=root_shape,
        Nx=root_shape[0],
        Ny=root_shape[1],
        Nz=root_shape[2],
        spacing=spacing,
        x_min=lower[0],
        x_max=upper[0],
        y_min=lower[1],
        y_max=upper[1],
        z_min=lower[2],
        z_max=upper[2],
        tile_shape=root_shape,
    )

    fine_shape = tuple(
        refinement_ratio * (stop - start)
        for start, stop in zip(parent_start, parent_stop)
    )
    fine_lower = tuple(
        root_lower + start * root_spacing
        for root_lower, start, root_spacing in zip(lower, parent_start, spacing)
    )
    fine_upper = tuple(
        root_lower + stop * root_spacing
        for root_lower, stop, root_spacing in zip(lower, parent_stop, spacing)
    )
    fine_spacing = tuple(root_spacing / refinement_ratio for root_spacing in spacing)

    fine_level = FMRLevel(
        level=1,
        parent=parent,
        refinement_ratio=refinement_ratio,
        parent_start=parent_start,
        parent_stop=parent_stop,
        Nx=fine_shape[0],
        Ny=fine_shape[1],
        Nz=fine_shape[2],
        spacing=fine_spacing,
        x_min=fine_lower[0],
        x_max=fine_upper[0],
        y_min=fine_lower[1],
        y_max=fine_upper[1],
        z_min=fine_lower[2],
        z_max=fine_upper[2],
        tile_shape=fine_shape,
    )

    return root_level, fine_level


def _build_level_grids(level, guard_cells):
    dynamic_setup = SimpleNamespace(
        dx=level.spacing[0],
        dy=level.spacing[1],
        dz=level.spacing[2],
        Nx=level.Nx,
        Ny=level.Ny,
        Nz=level.Nz,
        x_wind=level.x_max - level.x_min,
        y_wind=level.y_max - level.y_min,
        z_wind=level.z_max - level.z_min,
        x_min=level.x_min,
        y_min=level.y_min,
        z_min=level.z_min,
    )
    center_grid, vertex_grid = build_yee_grid(dynamic_setup)

    static_setup = SimpleNamespace(tile_shape=level.tile_shape, guard_cells=guard_cells)
    tiled_setup = SimpleNamespace(
        **dynamic_setup.__dict__,
        grids=SimpleNamespace(center=center_grid, vertex=vertex_grid),
    )
    tiled_center_grid, tiled_vertex_grid = build_tiled_yee_grids(static_setup, tiled_setup)

    return GridParameters(
        vertex=vertex_grid,
        center=center_grid,
        tiled_vertex_grid=tiled_vertex_grid,
        tiled_center_grid=tiled_center_grid,
    )


def _component_coordinate_axes(grids, locations):
    axes = []
    for axis, location in enumerate(locations):
        tiled_grid = grids.tiled_vertex_grid if location == "V" else grids.tiled_center_grid
        axes.append(np.asarray(jax.device_get(tiled_grid[axis][0, 0, 0])))
    return tuple(axes)


def _coordinate_tolerance(*axes):
    scale = max(1.0, *(float(np.max(np.abs(axis))) for axis in axes))
    dtype = np.result_type(*(axis.dtype for axis in axes))
    return 32.0 * np.finfo(dtype).eps * scale


def _axis_interpolation_sources(coarse_axis, target_coordinate, tolerance):
    coincident = np.flatnonzero(np.isclose(coarse_axis, target_coordinate, rtol=0.0, atol=tolerance))
    if coincident.size:
        return ((int(coincident[0]), 1.0),)

    right = int(np.searchsorted(coarse_axis, target_coordinate, side="right"))
    left = right - 1
    if left < 0 or right >= coarse_axis.size:
        raise ValueError("The FMR patch does not have enough parent cells for interface interpolation.")

    weight_right = (target_coordinate - coarse_axis[left]) / (coarse_axis[right] - coarse_axis[left])
    return ((left, 1.0 - weight_right), (right, weight_right))


def _build_component_interpolation_map(
    coarse_axes,
    fine_axes,
    fine_level,
    parent_shape,
    guard_cells,
):
    bounds = (
        (fine_level.x_min, fine_level.x_max),
        (fine_level.y_min, fine_level.y_max),
        (fine_level.z_min, fine_level.z_max),
    )
    tolerance = _coordinate_tolerance(*coarse_axes, *fine_axes)

    x, y, z = np.meshgrid(*fine_axes, indexing="ij")
    coordinate_mesh = (x, y, z)
    in_closed_patch = np.ones(x.shape, dtype=bool)
    on_interface = np.zeros(x.shape, dtype=bool)
    for coordinate, (lower, upper) in zip(coordinate_mesh, bounds):
        in_closed_patch &= (coordinate >= lower - tolerance) & (coordinate <= upper + tolerance)
        on_interface |= np.isclose(coordinate, lower, rtol=0.0, atol=tolerance)
        on_interface |= np.isclose(coordinate, upper, rtol=0.0, atol=tolerance)

    target_indices = np.argwhere(in_closed_patch & on_interface)
    if np.unique(target_indices, axis=0).shape[0] != target_indices.shape[0]:
        raise ValueError("FMR interpolation target indices must be unique.")

    source_indices = np.zeros((target_indices.shape[0], 8, 3), dtype=np.int32)
    weights = np.zeros((target_indices.shape[0], 8), dtype=np.float64)
    g = int(guard_cells)

    for target_number, target_index in enumerate(target_indices):
        target_coordinate = tuple(
            fine_axes[axis][target_index[axis]]
            for axis in range(3)
        )
        axis_sources = tuple(
            _axis_interpolation_sources(coarse_axes[axis], target_coordinate[axis], tolerance)
            for axis in range(3)
        )
        donors = tuple(product(*axis_sources))

        for donor_number, donor in enumerate(donors):
            donor_index = tuple(axis_donor[0] for axis_donor in donor)
            donor_weight = float(np.prod([axis_donor[1] for axis_donor in donor]))
            if donor_weight != 0.0:
                for index, cells in zip(donor_index, parent_shape):
                    if not g <= index < g + int(cells):
                        raise ValueError(
                            "The FMR patch does not have enough parent cells around it "
                            "for all Yee interpolation stencils."
                        )
            source_indices[target_number, donor_number] = donor_index
            weights[target_number, donor_number] = donor_weight

        source_indices[target_number, len(donors):] = source_indices[target_number, 0]

    if not np.allclose(np.sum(weights, axis=1), 1.0):
        raise ValueError("FMR interpolation weights must sum to one for every target.")

    weight_dtype = np.result_type(*(axis.dtype for axis in coarse_axes), np.float32)
    return FMRInterpolationMap(
        target_indices=jnp.asarray(target_indices, dtype=jnp.int32),
        source_indices=jnp.asarray(source_indices, dtype=jnp.int32),
        weights=jnp.asarray(weights, dtype=weight_dtype),
    )


def build_e_interface_maps(parent_level, fine_level, parent_grids, fine_grids, guard_cells):
    """Build separate tensor-product interpolation maps for Ex, Ey, and Ez."""

    parent_shape = (parent_level.Nx, parent_level.Ny, parent_level.Nz)
    return tuple(
        _build_component_interpolation_map(
            _component_coordinate_axes(parent_grids, locations),
            _component_coordinate_axes(fine_grids, locations),
            fine_level,
            parent_shape,
            guard_cells,
        )
        for locations in E_FIELD_LOCATIONS
    )


def _component_inside_mask(grids, locations, level, refined_bounds, guard_cells):
    g = int(guard_cells)
    axes = _component_coordinate_axes(grids, locations)
    active_axes = tuple(axis[g:g + cells] for axis, cells in zip(axes, (level.Nx, level.Ny, level.Nz)))
    x, y, z = np.meshgrid(*active_axes, indexing="ij")

    tolerance = _coordinate_tolerance(*active_axes)
    inside = np.ones(x.shape, dtype=bool)
    for coordinate, (lower, upper) in zip((x, y, z), refined_bounds):
        inside &= (coordinate > lower + tolerance) & (coordinate < upper - tolerance)

    return jnp.asarray(inside[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :, :])


def build_b_active_masks(parent_level, fine_level, parent_grids, fine_grids, guard_cells):
    """Build stagger-aware coarse-active B masks for the two-level composite grid."""

    refined_bounds = (
        (fine_level.x_min, fine_level.x_max),
        (fine_level.y_min, fine_level.y_max),
        (fine_level.z_min, fine_level.z_max),
    )
    parent_masks = []
    fine_masks = []
    for locations in B_FIELD_LOCATIONS:
        parent_inside = _component_inside_mask(
            parent_grids,
            locations,
            parent_level,
            refined_bounds,
            guard_cells,
        )
        fine_inside = _component_inside_mask(
            fine_grids,
            locations,
            fine_level,
            refined_bounds,
            guard_cells,
        )
        parent_masks.append(~parent_inside)
        fine_masks.append(fine_inside)

    return tuple(parent_masks), tuple(fine_masks)


def build_fmr_metric_weights(
    parent_level,
    fine_level,
    e_interface_maps,
    parent_b_masks,
    fine_b_masks,
    guard_cells,
):
    """Build active-grid Cartesian volume weights for the FMR fields."""

    parent_volume = np.prod(parent_level.spacing)
    fine_volume = np.prod(fine_level.spacing)

    parent_e_weights = tuple(
        jnp.full(mask.shape, parent_volume)
        for mask in parent_b_masks
    )

    g = int(guard_cells)
    fine_shape = np.asarray((fine_level.Nx, fine_level.Ny, fine_level.Nz))
    fine_e_weights = []
    for interpolation_map, mask in zip(e_interface_maps, fine_b_masks):
        weight = jnp.full(mask.shape, fine_volume)
        target = np.asarray(interpolation_map.target_indices) - g
        physical = np.all((target >= 0) & (target < fine_shape), axis=1)
        target = target[physical]
        weight = weight.at[
            0,
            0,
            0,
            target[:, 0],
            target[:, 1],
            target[:, 2],
        ].set(0.0, unique_indices=True)
        fine_e_weights.append(weight)

    parent_b_weights = tuple(parent_volume * mask for mask in parent_b_masks)
    fine_b_weights = tuple(fine_volume * mask for mask in fine_b_masks)

    return (
        parent_e_weights,
        parent_b_weights,
        tuple(fine_e_weights),
        fine_b_weights,
    )


def build_fmr_parameters(static_parameters, dynamic_parameters):
    """Build the static FMR interpolation, activity, and metric data once."""

    if not static_parameters.fmr_enabled:
        return None
    if len(static_parameters.fmr_levels) != 2:
        raise ValueError("The first FMR implementation requires root and one fine level.")

    parent_level, fine_level = static_parameters.fmr_levels
    fine_grids = _build_level_grids(fine_level, static_parameters.guard_cells)
    e_interface_maps = build_e_interface_maps(
        parent_level,
        fine_level,
        dynamic_parameters.grids,
        fine_grids,
        static_parameters.guard_cells,
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


def _interpolate_component(parent_component, interpolation_map):
    source = interpolation_map.source_indices
    source_values = parent_component[
        0,
        0,
        0,
        source[:, :, 0],
        source[:, :, 1],
        source[:, :, 2],
    ]
    return jnp.sum(interpolation_map.weights * source_values, axis=1)


def prolong_e_to_fine_interface(parent_E, fine_E, e_interface_maps):
    """Overwrite only coarse-controlled fine E interface locations."""

    prolonged = []
    for parent_component, fine_component, interpolation_map in zip(parent_E, fine_E, e_interface_maps):
        target = interpolation_map.target_indices
        values = _interpolate_component(parent_component, interpolation_map)
        fine_component = fine_component.at[
            0,
            0,
            0,
            target[:, 0],
            target[:, 1],
            target[:, 2],
        ].set(values, unique_indices=True)
        prolonged.append(fine_component)
    return tuple(prolonged)


def _active_vector(field_tiles, guard_cells):
    g = int(guard_cells)
    active = slice(g, -g)
    return tuple(component[:, :, :, active, active, active] for component in field_tiles)


def _apply_weights(values, weights):
    return tuple(value * weight for value, weight in zip(values, weights))


def _apply_inverse_weights(values, weights):
    weighted_values = []
    for value, weight in zip(values, weights):
        safe_weight = jnp.where(weight != 0.0, weight, 1.0)
        weighted_values.append(jnp.where(weight != 0.0, value / safe_weight, 0.0))
    return tuple(weighted_values)


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
    """Apply the metric-weighted FMR adjoint M_E^-1 C.T M_B."""

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


__all__ = [
    "FMRInterpolationMap",
    "FMRLevel",
    "FMRLevelData",
    "FMRParameters",
    "build_b_active_masks",
    "build_e_interface_maps",
    "build_fmr_fields",
    "build_fmr_metric_weights",
    "build_fmr_parameters",
    "fmr_curl_b_to_e",
    "fmr_curl_e_to_b",
    "load_fmr_from_toml",
    "prolong_e_to_fine_interface",
    "time_loop_electrodynamic_fmr_fields",
    "update_B_fmr",
    "update_E_fmr",
    "validate_fmr_configuration",
]
