import unittest
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONSTANT
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.solvers.yee.first_order_yee import (
    assemble_yee_curl,
    yee_derivatives_e_to_b_refreshed,
)
from PyPIC3D.solvers.yee.fmr import (
    build_fmr_fields,
    build_fmr_parameters,
    fmr_curl_b_to_e,
    fmr_curl_e_to_b,
    load_fmr_from_toml,
    prolong_e_to_fine_interface,
    time_loop_electrodynamic_fmr_fields,
    update_B_fmr,
    update_E_fmr,
)
from tests.kernel_fixtures import initialized_fields, kernel_parameters


jax.config.update("jax_enable_x64", True)


E_FIELD_LOCATIONS = (("V", "C", "C"), ("C", "V", "C"), ("C", "C", "V"))
B_FIELD_LOCATIONS = (("C", "V", "V"), ("V", "C", "V"), ("V", "V", "C"))

AFFINE_E_COEFFICIENTS = (
    (0.7, -1.1, 0.3, 2.0),
    (-0.2, 0.9, 1.3, -0.4),
    (1.5, -0.6, 0.8, 0.1),
)
AFFINE_E_CURL = (-1.9, -1.2, 0.9)


@lru_cache(maxsize=2)
def _fmr_case(refinement_ratio):
    static_parameters, dynamic_parameters = kernel_parameters(
        Nx=8,
        Ny=8,
        Nz=8,
        x_wind=8.0,
        y_wind=8.0,
        z_wind=8.0,
        x_min=0.0,
        y_min=0.0,
        z_min=0.0,
        tile_shape=(8, 8, 8),
        guard_cells=2,
        dt=2.0e-3,
        C=1.0,
        eps=1.0,
        mu=1.0,
    )

    config = {
        "fmr": {
            "enabled": True,
            "levels": [
                {
                    "parent": 0,
                    "refinement_ratio": refinement_ratio,
                    "coarse_start": [2, 2, 2],
                    "coarse_stop": [6, 6, 6],
                }
            ],
        }
    }
    dynamic_config = {
        "Nx": 8,
        "Ny": 8,
        "Nz": 8,
        "dx": 1.0,
        "dy": 1.0,
        "dz": 1.0,
        "x_min": 0.0,
        "x_max": 8.0,
        "y_min": 0.0,
        "y_max": 8.0,
        "z_min": 0.0,
        "z_max": 8.0,
    }

    levels = load_fmr_from_toml(config, dynamic_config, static_parameters.tile_shape)
    static_parameters = static_parameters._replace(
        fmr_enabled=True,
        fmr_levels=levels,
    )
    fmr_parameters = build_fmr_parameters(static_parameters, dynamic_parameters)
    dynamic_parameters = dynamic_parameters._replace(fmr=fmr_parameters)

    E0, B0, J0, phi, rho = initialized_fields(static_parameters, dynamic_parameters)
    E, B, J = build_fmr_fields(
        E0,
        B0,
        J0,
        static_parameters,
        dynamic_parameters,
    )
    return static_parameters, dynamic_parameters, E, B, J, rho, phi


def _component_axes(grids, locations):
    axes = []
    for axis, location in enumerate(locations):
        grid = grids.tiled_vertex_grid if location == "V" else grids.tiled_center_grid
        axes.append(grid[axis][0, 0, 0])
    return tuple(axes)


def _component_coordinates(grids, locations):
    x_axis, y_axis, z_axis = _component_axes(grids, locations)
    return (
        x_axis[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, jnp.newaxis, jnp.newaxis],
        y_axis[jnp.newaxis, jnp.newaxis, jnp.newaxis, jnp.newaxis, :, jnp.newaxis],
        z_axis[jnp.newaxis, jnp.newaxis, jnp.newaxis, jnp.newaxis, jnp.newaxis, :],
    )


def _constant_e_field(grids, values=(1.25, -2.0, 0.75)):
    fields = []
    for locations, value in zip(E_FIELD_LOCATIONS, values):
        x, y, z = _component_coordinates(grids, locations)
        fields.append(jnp.broadcast_to(jnp.asarray(value), jnp.broadcast_shapes(x.shape, y.shape, z.shape)))
    return tuple(fields)


def _affine_e_field(grids):
    fields = []
    for locations, (ax, ay, az, constant) in zip(E_FIELD_LOCATIONS, AFFINE_E_COEFFICIENTS):
        x, y, z = _component_coordinates(grids, locations)
        fields.append(ax * x + ay * y + az * z + constant)
    return tuple(fields)


def _map_target_values(component, interpolation_map):
    target = interpolation_map.target_indices
    return component[
        0,
        0,
        0,
        target[:, 0],
        target[:, 1],
        target[:, 2],
    ]


def _active_vector(field, guard_cells):
    g = int(guard_cells)
    active = slice(g, -g)
    return tuple(component[:, :, :, active, active, active] for component in field)


def _field_dot_levels(left, right):
    return sum(
        jnp.vdot(left_component, right_component)
        for left_level, right_level in zip(left, right)
        for left_component, right_component in zip(left_level, right_level)
    )


def _random_levels_like(levels, seed):
    leaves = [component for level in levels for component in level]
    keys = iter(jax.random.split(jax.random.key(seed), len(leaves)))
    return tuple(
        tuple(
            jax.random.normal(next(keys), component.shape, dtype=component.dtype)
            for component in level
        )
        for level in levels
    )


def _random_physical_levels_like(levels, active_masks, seed, guard_cells):
    g = int(guard_cells)
    active = slice(g, -g)
    leaves = [component for level in levels for component in level]
    keys = iter(jax.random.split(jax.random.key(seed), len(leaves)))

    random_levels = []
    for level, level_masks in zip(levels, active_masks):
        random_components = []
        for component, mask in zip(level, level_masks):
            values = jax.random.normal(next(keys), mask.shape, dtype=component.dtype)
            random_component = jnp.zeros_like(component)
            random_component = random_component.at[:, :, :, active, active, active].set(
                values * mask
            )
            random_components.append(random_component)
        random_levels.append(tuple(random_components))
    return tuple(random_levels)


def _weighted_field_dot_levels(left, right, weights):
    return sum(
        jnp.vdot(left_component, weight * right_component)
        for left_level, right_level, level_weights in zip(left, right, weights)
        for left_component, right_component, weight in zip(
            left_level,
            right_level,
            level_weights,
        )
    )


def _raw_fmr_curl_b_to_e(B_levels, E_template, static_parameters, dynamic_parameters):
    g = int(static_parameters.guard_cells)
    B_active = tuple(_active_vector(level, g) for level in B_levels)
    transpose = jax.linear_transpose(
        lambda E: fmr_curl_e_to_b(E, static_parameters, dynamic_parameters),
        E_template,
    )
    transposed_E, = transpose(B_active)
    return tuple(_active_vector(level, g) for level in transposed_E)


def _expected_b_mask(grids, locations, level, fine_level, guard_cells, active_inside):
    g = int(guard_cells)
    axes = _component_axes(grids, locations)
    active_axes = tuple(
        np.asarray(axis[g:g + cells])
        for axis, cells in zip(axes, (level.Nx, level.Ny, level.Nz))
    )
    x, y, z = np.meshgrid(*active_axes, indexing="ij")
    bounds = (
        (fine_level.x_min, fine_level.x_max),
        (fine_level.y_min, fine_level.y_max),
        (fine_level.z_min, fine_level.z_max),
    )

    inside = np.ones(x.shape, dtype=bool)
    for coordinate, (lower, upper) in zip((x, y, z), bounds):
        inside &= (coordinate > lower + 1.0e-13) & (coordinate < upper - 1.0e-13)

    mask = inside if active_inside else ~inside
    return jnp.asarray(mask[jnp.newaxis, jnp.newaxis, jnp.newaxis])


def _fine_raw_curl(E_levels, static_parameters, dynamic_parameters):
    E0, E1 = E_levels
    g = int(static_parameters.guard_cells)
    fine_level = static_parameters.fmr_levels[1]
    fine_data = dynamic_parameters.fmr.levels[1]
    fine_static_parameters = static_parameters._replace(
        tile_shape=fine_level.tile_shape,
        boundary_conditions=(BC_CONSTANT, BC_CONSTANT, BC_CONSTANT),
        fmr_enabled=False,
        fmr_levels=(),
    )

    E0 = ghost_cells.update_tiled_vector_ghost_cells(E0, static_parameters, g)
    E1 = ghost_cells.update_tiled_vector_ghost_cells(E1, fine_static_parameters, g)
    E1 = prolong_e_to_fine_interface(E0, E1, fine_data.e_interface_maps)
    derivatives = yee_derivatives_e_to_b_refreshed(E1, fine_level.spacing, g)
    return assemble_yee_curl(derivatives)


def _empty_particles():
    particles = TiledParticles(
        x=jnp.zeros((1, 1, 1, 0, 0, 3)),
        u=jnp.zeros((1, 1, 1, 0, 0, 3)),
        active=jnp.zeros((1, 1, 1, 0, 0), dtype=bool),
    )
    species_config = SpeciesConfig(
        charge=jnp.zeros((0,)),
        mass=jnp.zeros((0,)),
        weight=jnp.zeros((0,)),
        update_x=jnp.zeros((0, 3), dtype=bool),
    )
    return particles, species_config


def _runtime_fields(E, B, J, rho, phi):
    return E, B, J, rho, phi, ((), ()), None, jnp.asarray(False)


class TestFMRGeometryAndInterpolation(unittest.TestCase):
    def test_geometry_and_interface_map_metadata_for_ratio_two_and_four(self):
        for ratio in (2, 4):
            with self.subTest(refinement_ratio=ratio):
                static_parameters, dynamic_parameters, E, B, J, *_ = _fmr_case(ratio)
                parent_level, fine_level = static_parameters.fmr_levels

                self.assertEqual((parent_level.Nx, parent_level.Ny, parent_level.Nz), (8, 8, 8))
                self.assertEqual(
                    (fine_level.Nx, fine_level.Ny, fine_level.Nz),
                    (4 * ratio, 4 * ratio, 4 * ratio),
                )
                self.assertEqual(fine_level.spacing, (1.0 / ratio,) * 3)
                self.assertEqual(
                    (
                        fine_level.x_min,
                        fine_level.x_max,
                        fine_level.y_min,
                        fine_level.y_max,
                        fine_level.z_min,
                        fine_level.z_max,
                    ),
                    (2.0, 6.0, 2.0, 6.0, 2.0, 6.0),
                )

                fine_data = dynamic_parameters.fmr.levels[1]
                for level_fields in (E, B, J):
                    for level, field in zip((parent_level, fine_level), level_fields):
                        for component in field:
                            self.assertEqual(component.shape[:3], (1, 1, 1))
                            self.assertEqual(
                                component.shape[-3:],
                                (
                                    level.Nx + 2 * static_parameters.guard_cells,
                                    level.Ny + 2 * static_parameters.guard_cells,
                                    level.Nz + 2 * static_parameters.guard_cells,
                                ),
                            )

                bounds = (
                    (fine_level.x_min, fine_level.x_max),
                    (fine_level.y_min, fine_level.y_max),
                    (fine_level.z_min, fine_level.z_max),
                )
                for locations, interpolation_map in zip(
                    E_FIELD_LOCATIONS,
                    fine_data.e_interface_maps,
                ):
                    target = np.asarray(interpolation_map.target_indices)
                    source = np.asarray(interpolation_map.source_indices)
                    weights = np.asarray(interpolation_map.weights)

                    self.assertEqual(interpolation_map.source_indices.shape[1:], (8, 3))
                    self.assertEqual(weights.shape, (target.shape[0], 8))
                    self.assertEqual(np.unique(target, axis=0).shape[0], target.shape[0])
                    self.assertTrue(np.allclose(np.sum(weights, axis=1), 1.0, rtol=0.0, atol=1.0e-14))
                    self.assertTrue(np.all(weights >= 0.0))
                    self.assertTrue(np.all(np.count_nonzero(weights, axis=1) <= 8))

                    nonzero_source = source[weights > 0.0]
                    self.assertTrue(np.all(nonzero_source >= static_parameters.guard_cells))
                    self.assertTrue(np.all(
                        nonzero_source
                        < np.asarray((parent_level.Nx, parent_level.Ny, parent_level.Nz))
                        + static_parameters.guard_cells
                    ))

                    fine_axes = tuple(
                        np.asarray(axis)
                        for axis in _component_axes(fine_data.grids, locations)
                    )
                    target_coordinates = tuple(
                        fine_axes[axis][target[:, axis]]
                        for axis in range(3)
                    )
                    inside = np.ones(target.shape[0], dtype=bool)
                    on_interface = np.zeros(target.shape[0], dtype=bool)
                    for coordinate, (lower, upper) in zip(target_coordinates, bounds):
                        inside &= (coordinate >= lower - 1.0e-13) & (coordinate <= upper + 1.0e-13)
                        on_interface |= np.isclose(coordinate, lower, rtol=0.0, atol=1.0e-13)
                        on_interface |= np.isclose(coordinate, upper, rtol=0.0, atol=1.0e-13)
                    self.assertTrue(np.all(inside))
                    self.assertTrue(np.all(on_interface))

    def test_constant_and_affine_interface_interpolation_uses_component_coordinates(self):
        for ratio in (2, 4):
            with self.subTest(refinement_ratio=ratio):
                _, dynamic_parameters, E, *_ = _fmr_case(ratio)
                parent_grids = dynamic_parameters.fmr.levels[0].grids
                fine_grids = dynamic_parameters.fmr.levels[1].grids
                interpolation_maps = dynamic_parameters.fmr.levels[1].e_interface_maps

                constant_parent = _constant_e_field(parent_grids)
                constant_expected = _constant_e_field(fine_grids)
                constant_fine = tuple(jnp.zeros_like(component) for component in E[1])
                constant_actual = prolong_e_to_fine_interface(
                    constant_parent,
                    constant_fine,
                    interpolation_maps,
                )

                affine_parent = _affine_e_field(parent_grids)
                affine_expected = _affine_e_field(fine_grids)
                affine_fine = tuple(jnp.full_like(component, -999.0) for component in E[1])
                affine_actual = prolong_e_to_fine_interface(
                    affine_parent,
                    affine_fine,
                    interpolation_maps,
                )

                for component_index, interpolation_map in enumerate(interpolation_maps):
                    with self.subTest(component=component_index):
                        self.assertTrue(jnp.allclose(
                            _map_target_values(constant_actual[component_index], interpolation_map),
                            _map_target_values(constant_expected[component_index], interpolation_map),
                            rtol=1.0e-13,
                            atol=1.0e-13,
                        ))
                        self.assertTrue(jnp.allclose(
                            _map_target_values(affine_actual[component_index], interpolation_map),
                            _map_target_values(affine_expected[component_index], interpolation_map),
                            rtol=1.0e-13,
                            atol=1.0e-13,
                        ))

    def test_coincident_axes_copy_directly_and_ratio_four_has_quarter_weights(self):
        static_parameters, dynamic_parameters, *_ = _fmr_case(4)
        parent_grids = dynamic_parameters.fmr.levels[0].grids
        fine_grids = dynamic_parameters.fmr.levels[1].grids
        interpolation_maps = dynamic_parameters.fmr.levels[1].e_interface_maps

        direct_axis_count = 0
        observed_fractions = set()
        for locations, interpolation_map in zip(E_FIELD_LOCATIONS, interpolation_maps):
            coarse_axes = tuple(np.asarray(axis) for axis in _component_axes(parent_grids, locations))
            fine_axes = tuple(np.asarray(axis) for axis in _component_axes(fine_grids, locations))
            target = np.asarray(interpolation_map.target_indices)
            source = np.asarray(interpolation_map.source_indices)
            weights = np.asarray(interpolation_map.weights)

            for row, target_index in enumerate(target):
                nonzero = weights[row] > 1.0e-14
                for axis in range(3):
                    target_coordinate = fine_axes[axis][target_index[axis]]
                    donor_indices = np.unique(source[row, nonzero, axis])
                    donor_coordinates = coarse_axes[axis][donor_indices]

                    coincident = np.flatnonzero(np.isclose(
                        coarse_axes[axis],
                        target_coordinate,
                        rtol=0.0,
                        atol=1.0e-13,
                    ))
                    if coincident.size:
                        direct_axis_count += 1
                        self.assertEqual(donor_indices.shape[0], 1)
                        self.assertAlmostEqual(float(donor_coordinates[0]), float(target_coordinate), places=13)

                    if donor_indices.shape[0] != 2:
                        continue

                    order = np.argsort(donor_coordinates)
                    donor_indices = donor_indices[order]
                    donor_coordinates = donor_coordinates[order]
                    marginal_weights = np.asarray([
                        np.sum(weights[row, nonzero][source[row, nonzero, axis] == donor_index])
                        for donor_index in donor_indices
                    ])
                    fraction = (
                        (target_coordinate - donor_coordinates[0])
                        / (donor_coordinates[1] - donor_coordinates[0])
                    )
                    for expected_fraction in (0.25, 0.5, 0.75):
                        if np.isclose(fraction, expected_fraction, rtol=0.0, atol=1.0e-13):
                            observed_fractions.add(expected_fraction)
                            self.assertTrue(np.allclose(
                                marginal_weights,
                                (1.0 - expected_fraction, expected_fraction),
                                rtol=0.0,
                                atol=1.0e-14,
                            ))

        self.assertGreater(direct_axis_count, 0)
        self.assertEqual(observed_fractions, {0.25, 0.5, 0.75})

    def test_b_active_masks_use_each_components_staggering(self):
        static_parameters, dynamic_parameters, *_ = _fmr_case(2)
        parent_level, fine_level = static_parameters.fmr_levels
        parent_data, fine_data = dynamic_parameters.fmr.levels
        g = int(static_parameters.guard_cells)

        for component_index, locations in enumerate(B_FIELD_LOCATIONS):
            with self.subTest(component=component_index):
                expected_parent = _expected_b_mask(
                    parent_data.grids,
                    locations,
                    parent_level,
                    fine_level,
                    g,
                    active_inside=False,
                )
                expected_fine = _expected_b_mask(
                    fine_data.grids,
                    locations,
                    fine_level,
                    fine_level,
                    g,
                    active_inside=True,
                )
                self.assertTrue(jnp.array_equal(parent_data.b_active_masks[component_index], expected_parent))
                self.assertTrue(jnp.array_equal(fine_data.b_active_masks[component_index], expected_fine))
                self.assertGreater(int(jnp.sum(~expected_parent)), 0)
                self.assertGreater(int(jnp.sum(~expected_fine)), 0)

    def test_metric_weights_use_level_volumes_and_zero_constrained_dofs(self):
        static_parameters, dynamic_parameters, *_ = _fmr_case(2)
        parent_level, fine_level = static_parameters.fmr_levels
        parent_data, fine_data = dynamic_parameters.fmr.levels
        g = int(static_parameters.guard_cells)

        parent_volume = np.prod(parent_level.spacing)
        fine_volume = np.prod(fine_level.spacing)
        self.assertAlmostEqual(parent_volume / fine_volume, 8.0, places=14)

        for weight in parent_data.e_weights:
            self.assertTrue(jnp.all(jnp.isfinite(weight)))
            self.assertTrue(jnp.all(weight == parent_volume))

        fine_shape = np.asarray((fine_level.Nx, fine_level.Ny, fine_level.Nz))
        for weight, interpolation_map in zip(
            fine_data.e_weights,
            fine_data.e_interface_maps,
        ):
            target = np.asarray(interpolation_map.target_indices) - g
            physical = np.all((target >= 0) & (target < fine_shape), axis=1)
            target = target[physical]
            interface_weights = weight[
                0,
                0,
                0,
                target[:, 0],
                target[:, 1],
                target[:, 2],
            ]
            self.assertTrue(jnp.all(interface_weights == 0.0))
            self.assertTrue(jnp.all(jnp.isfinite(weight)))
            self.assertTrue(jnp.all(weight[weight != 0.0] == fine_volume))

        for level_data, volume in (
            (parent_data, parent_volume),
            (fine_data, fine_volume),
        ):
            for weight, active_mask in zip(
                level_data.b_weights,
                level_data.b_active_masks,
            ):
                self.assertTrue(jnp.all(weight[~active_mask] == 0.0))
                self.assertTrue(jnp.all(jnp.isfinite(weight[active_mask])))
                self.assertTrue(jnp.all(weight[active_mask] == volume))


class TestFMRCurl(unittest.TestCase):
    def test_constant_field_has_zero_fine_and_composite_curl(self):
        static_parameters, dynamic_parameters, *_ = _fmr_case(2)
        E_levels = tuple(
            _constant_e_field(level_data.grids)
            for level_data in dynamic_parameters.fmr.levels
        )

        raw_fine_curl = _fine_raw_curl(E_levels, static_parameters, dynamic_parameters)
        composite_curl = fmr_curl_e_to_b(E_levels, static_parameters, dynamic_parameters)

        for component in raw_fine_curl:
            self.assertLess(float(jnp.max(jnp.abs(component))), 1.0e-12)
        for level in composite_curl:
            for component in level:
                self.assertLess(float(jnp.max(jnp.abs(component))), 1.0e-12)

    def test_affine_fine_curl_is_exact_at_interface_faces_edges_and_corners(self):
        for ratio in (2, 4):
            with self.subTest(refinement_ratio=ratio):
                static_parameters, dynamic_parameters, *_ = _fmr_case(ratio)
                E_levels = tuple(
                    _affine_e_field(level_data.grids)
                    for level_data in dynamic_parameters.fmr.levels
                )
                raw_fine_curl = _fine_raw_curl(E_levels, static_parameters, dynamic_parameters)
                nx, ny, nz = raw_fine_curl[0].shape[-3:]
                i, j, k = np.indices((nx, ny, nz))
                boundary_count = (
                    ((i == 0) | (i == nx - 1)).astype(int)
                    + ((j == 0) | (j == ny - 1)).astype(int)
                    + ((k == 0) | (k == nz - 1)).astype(int)
                )
                masks = {
                    "-x": i == 0,
                    "+x": i == nx - 1,
                    "-y": j == 0,
                    "+y": j == ny - 1,
                    "-z": k == 0,
                    "+z": k == nz - 1,
                    "edges": boundary_count >= 2,
                    "corners": boundary_count == 3,
                }

                for component_index, (actual, expected) in enumerate(zip(raw_fine_curl, AFFINE_E_CURL)):
                    error = np.abs(np.asarray(actual[0, 0, 0]) - expected)
                    self.assertLess(float(np.max(error)), 2.0e-12)
                    for region_name, mask in masks.items():
                        with self.subTest(component=component_index, region=region_name):
                            self.assertLess(float(np.max(error[mask])), 2.0e-12)

    def test_affine_composite_curl_applies_parent_and_fine_b_masks(self):
        static_parameters, dynamic_parameters, *_ = _fmr_case(2)
        E_levels = tuple(
            _affine_e_field(level_data.grids)
            for level_data in dynamic_parameters.fmr.levels
        )
        composite_curl = fmr_curl_e_to_b(E_levels, static_parameters, dynamic_parameters)

        for level_index, (curl_level, level_data) in enumerate(
            zip(composite_curl, dynamic_parameters.fmr.levels)
        ):
            for component_index, (actual, expected_value, active_mask) in enumerate(
                zip(curl_level, AFFINE_E_CURL, level_data.b_active_masks)
            ):
                expected = expected_value * active_mask
                if level_index == 0:
                    # The affine root field is intentionally not periodic.  Its
                    # wrapped upper physical halo is unrelated to FMR, so keep
                    # this comparison away from those three outer planes.
                    actual = actual[:, :, :, :-1, :-1, :-1]
                    expected = expected[:, :, :, :-1, :-1, :-1]
                with self.subTest(level=level_index, component=component_index):
                    self.assertTrue(jnp.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12))

    def test_prolongation_satisfies_adjoint_identity(self):
        _, dynamic_parameters, E, *_ = _fmr_case(2)
        interpolation_maps = dynamic_parameters.fmr.levels[1].e_interface_maps
        parent_E = _random_levels_like((E[0],), seed=101)[0]
        fine_cotangent = _random_levels_like((E[1],), seed=102)[0]
        fine_zeros = tuple(jnp.zeros_like(component) for component in E[1])

        prolongation = lambda parent: prolong_e_to_fine_interface(
            parent,
            fine_zeros,
            interpolation_maps,
        )
        prolonged = prolongation(parent_E)
        transpose = jax.linear_transpose(prolongation, parent_E)
        transposed_parent, = transpose(fine_cotangent)

        lhs = sum(jnp.vdot(a, b) for a, b in zip(prolonged, fine_cotangent))
        rhs = sum(jnp.vdot(a, b) for a, b in zip(parent_E, transposed_parent))
        scale = max(1.0, float(jnp.abs(lhs)), float(jnp.abs(rhs)))
        self.assertLess(float(jnp.abs(lhs - rhs)) / scale, 2.0e-12)

    def test_complete_fmr_curl_satisfies_adjoint_identity(self):
        static_parameters, dynamic_parameters, E, B, *_ = _fmr_case(2)
        E_levels = _random_levels_like(E, seed=201)
        B_levels = _random_levels_like(B, seed=202)
        g = int(static_parameters.guard_cells)

        curl_E = fmr_curl_e_to_b(E_levels, static_parameters, dynamic_parameters)
        raw_curl_B = _raw_fmr_curl_b_to_e(
            B_levels,
            E_levels,
            static_parameters,
            dynamic_parameters,
        )
        E_active = tuple(_active_vector(level, g) for level in E_levels)
        B_active = tuple(_active_vector(level, g) for level in B_levels)

        lhs = _field_dot_levels(curl_E, B_active)
        rhs = _field_dot_levels(E_active, raw_curl_B)
        scale = max(1.0, float(jnp.abs(lhs)), float(jnp.abs(rhs)))
        self.assertLess(float(jnp.abs(lhs - rhs)) / scale, 5.0e-12)

    def test_discrete_electromagnetic_power_balance(self):
        static_parameters, dynamic_parameters, E, B, *_ = _fmr_case(2)
        g = int(static_parameters.guard_cells)

        e_weights = tuple(
            level_data.e_weights
            for level_data in dynamic_parameters.fmr.levels
        )
        b_weights = tuple(
            level_data.b_weights
            for level_data in dynamic_parameters.fmr.levels
        )
        e_active_masks = tuple(
            tuple(weight != 0.0 for weight in level_weights)
            for level_weights in e_weights
        )
        b_active_masks = tuple(
            tuple(weight != 0.0 for weight in level_weights)
            for level_weights in b_weights
        )
        E_levels = _random_physical_levels_like(
            E,
            e_active_masks,
            seed=211,
            guard_cells=g,
        )
        B_levels = _random_physical_levels_like(
            B,
            b_active_masks,
            seed=212,
            guard_cells=g,
        )

        curl_E = fmr_curl_e_to_b(E_levels, static_parameters, dynamic_parameters)
        raw_curl_B = _raw_fmr_curl_b_to_e(
            B_levels,
            E_levels,
            static_parameters,
            dynamic_parameters,
        )
        curl_B = fmr_curl_b_to_e(
            B_levels,
            E_levels,
            static_parameters,
            dynamic_parameters,
        )
        E_active = tuple(_active_vector(level, g) for level in E_levels)
        B_active = tuple(_active_vector(level, g) for level in B_levels)

        algebraic_left = _weighted_field_dot_levels(
            curl_E,
            B_active,
            b_active_masks,
        )
        algebraic_right = _weighted_field_dot_levels(
            E_active,
            raw_curl_B,
            e_active_masks,
        )
        algebraic_residual = algebraic_left - algebraic_right
        algebraic_relative_residual = jnp.abs(algebraic_residual) / jnp.maximum(
            jnp.abs(algebraic_left) + jnp.abs(algebraic_right),
            1.0e-30,
        )

        electric_power = _weighted_field_dot_levels(E_active, curl_B, e_weights)
        magnetic_power = _weighted_field_dot_levels(curl_E, B_active, b_weights)
        weighted_residual = electric_power - magnetic_power
        weighted_relative_residual = jnp.abs(weighted_residual) / jnp.maximum(
            jnp.abs(electric_power) + jnp.abs(magnetic_power),
            1.0e-30,
        )

        print(
            "\nFMR algebraic transpose:\n"
            f"    lhs = {float(algebraic_left):.16e}\n"
            f"    rhs = {float(algebraic_right):.16e}\n"
            f"    relative residual = {float(algebraic_relative_residual):.16e}\n\n"
            "FMR weighted power balance:\n"
            f"    electric power = {float(electric_power):.16e}\n"
            f"    magnetic power = {float(magnetic_power):.16e}\n"
            f"    relative residual = {float(weighted_relative_residual):.16e}\n\n"
            f"coarse cell volume = {np.prod(static_parameters.fmr_levels[0].spacing):.16e}\n"
            f"fine cell volume = {np.prod(static_parameters.fmr_levels[1].spacing):.16e}"
        )

        self.assertTrue(jnp.allclose(
            algebraic_left,
            algebraic_right,
            rtol=1.0e-11,
            atol=1.0e-12,
        ))
        self.assertTrue(jnp.allclose(
            electric_power,
            magnetic_power,
            rtol=1.0e-11,
            atol=1.0e-12,
        ))


class TestFMRFieldUpdates(unittest.TestCase):
    def test_covered_parent_b_is_not_updated(self):
        static_parameters, dynamic_parameters, _, B, *_ = _fmr_case(2)
        E_levels = tuple(
            _affine_e_field(level_data.grids)
            for level_data in dynamic_parameters.fmr.levels
        )
        B_levels = _random_levels_like(B, seed=301)
        g = int(static_parameters.guard_cells)
        curl_E = fmr_curl_e_to_b(E_levels, static_parameters, dynamic_parameters)
        B_after = update_B_fmr(E_levels, B_levels, static_parameters, dynamic_parameters)

        for level_index, (before_level, after_level, curl_level) in enumerate(
            zip(B_levels, B_after, curl_E)
        ):
            before_active = _active_vector(before_level, g)
            after_active = _active_vector(after_level, g)
            for component_index, (before, after, curl_component) in enumerate(
                zip(before_active, after_active, curl_level)
            ):
                expected = before - 0.5 * dynamic_parameters.dt * curl_component
                self.assertTrue(jnp.allclose(after, expected, rtol=1.0e-13, atol=1.0e-13))

                if level_index == 0:
                    active_mask = dynamic_parameters.fmr.levels[0].b_active_masks[component_index]
                    self.assertTrue(jnp.array_equal(after[~active_mask], before[~active_mask]))

    def test_e_update_synchronizes_constrained_fine_interface(self):
        static_parameters, dynamic_parameters, E, B, J, *_ = _fmr_case(2)
        E_levels = _random_levels_like(E, seed=401)
        B_levels = _random_levels_like(B, seed=402)

        E_after = update_E_fmr(
            E_levels,
            B_levels,
            J,
            static_parameters,
            dynamic_parameters,
        )
        prolonged = prolong_e_to_fine_interface(
            E_after[0],
            E_after[1],
            dynamic_parameters.fmr.levels[1].e_interface_maps,
        )

        for actual, expected, interpolation_map in zip(
            E_after[1],
            prolonged,
            dynamic_parameters.fmr.levels[1].e_interface_maps,
        ):
            self.assertTrue(jnp.allclose(
                _map_target_values(actual, interpolation_map),
                _map_target_values(expected, interpolation_map),
                rtol=1.0e-13,
                atol=1.0e-13,
            ))

    def test_field_only_loop_matches_one_b_e_b_step(self):
        static_parameters, dynamic_parameters, E, B, J, rho, phi = _fmr_case(2)
        E = _random_levels_like(E, seed=501)
        B = _random_levels_like(B, seed=502)
        particles, species_config = _empty_particles()
        fields = _runtime_fields(E, B, J, rho, phi)

        B_expected = update_B_fmr(E, B, static_parameters, dynamic_parameters)
        E_expected = update_E_fmr(E, B_expected, J, static_parameters, dynamic_parameters)
        B_expected = update_B_fmr(E_expected, B_expected, static_parameters, dynamic_parameters)

        particles_after, fields_after = time_loop_electrodynamic_fmr_fields(
            particles,
            species_config,
            fields,
            static_parameters,
            dynamic_parameters,
        )
        E_after, B_after = fields_after[:2]

        for actual, expected in zip(jax.tree_util.tree_leaves(E_after), jax.tree_util.tree_leaves(E_expected)):
            self.assertTrue(jnp.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12))
        for actual, expected in zip(jax.tree_util.tree_leaves(B_after), jax.tree_util.tree_leaves(B_expected)):
            self.assertTrue(jnp.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12))
        for actual, expected in zip(
            jax.tree_util.tree_leaves(particles_after),
            jax.tree_util.tree_leaves(particles),
        ):
            self.assertTrue(jnp.array_equal(actual, expected))

    def test_repeated_field_steps_preserve_shapes_constraints_and_finiteness(self):
        static_parameters, dynamic_parameters, E, B, J, rho, phi = _fmr_case(2)
        E = tuple(
            _affine_e_field(level_data.grids)
            for level_data in dynamic_parameters.fmr.levels
        )
        particles, species_config = _empty_particles()
        fields = _runtime_fields(E, B, J, rho, phi)
        initial_shapes = tuple(leaf.shape for leaf in jax.tree_util.tree_leaves((E, B, J)))

        for _ in range(3):
            particles, fields = time_loop_electrodynamic_fmr_fields(
                particles,
                species_config,
                fields,
                static_parameters,
                dynamic_parameters,
            )

        E_after, B_after, J_after = fields[:3]
        final_shapes = tuple(leaf.shape for leaf in jax.tree_util.tree_leaves((E_after, B_after, J_after)))
        self.assertEqual(final_shapes, initial_shapes)
        for leaf in jax.tree_util.tree_leaves((E_after, B_after, J_after)):
            self.assertTrue(jnp.all(jnp.isfinite(leaf)))

        prolonged = prolong_e_to_fine_interface(
            E_after[0],
            E_after[1],
            dynamic_parameters.fmr.levels[1].e_interface_maps,
        )
        for actual, expected, interpolation_map in zip(
            E_after[1],
            prolonged,
            dynamic_parameters.fmr.levels[1].e_interface_maps,
        ):
            self.assertTrue(jnp.allclose(
                _map_target_values(actual, interpolation_map),
                _map_target_values(expected, interpolation_map),
                rtol=1.0e-12,
                atol=1.0e-12,
            ))

    def test_complete_field_only_timestep_jits(self):
        static_parameters, dynamic_parameters, E, B, J, rho, phi = _fmr_case(2)
        E = tuple(
            _affine_e_field(level_data.grids)
            for level_data in dynamic_parameters.fmr.levels
        )
        particles, species_config = _empty_particles()
        fields = _runtime_fields(E, B, J, rho, phi)

        def step(particles, fields, dynamic_parameters):
            return time_loop_electrodynamic_fmr_fields(
                particles,
                species_config,
                fields,
                static_parameters,
                dynamic_parameters,
            )

        compiled_step = jax.jit(step).lower(particles, fields, dynamic_parameters).compile()
        particles_after, fields_after = compiled_step(particles, fields, dynamic_parameters)
        for leaf in jax.tree_util.tree_leaves((particles_after, fields_after)):
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()

        self.assertEqual(
            tuple(leaf.shape for leaf in jax.tree_util.tree_leaves(fields_after[:3])),
            tuple(leaf.shape for leaf in jax.tree_util.tree_leaves(fields[:3])),
        )
        for leaf in jax.tree_util.tree_leaves(fields_after[:3]):
            self.assertTrue(jnp.all(jnp.isfinite(leaf)))


if __name__ == "__main__":
    unittest.main()
