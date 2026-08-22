import inspect
import unittest
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

import PyPIC3D.solvers.yee.fmr.curls as curls_module
from PyPIC3D.initialization import initialize_fields
from PyPIC3D.solvers.yee.fmr import (
    B_FIELD_LOCATIONS,
    E_FIELD_LOCATIONS,
    build_fmr_hierarchy,
    initialize_fmr_field_levels,
    load_fmr_levels,
    synchronize_b_levels,
    synchronize_e_levels,
    validate_fmr_configuration,
)
from PyPIC3D.solvers.yee.fmr.curls import fmr_curl_b_to_e, fmr_curl_e_to_b
from PyPIC3D.solvers.yee.fmr.grids import _coordinate_tolerance
from PyPIC3D.solvers.yee.fmr.grids import component_coordinate_axes
from PyPIC3D.solvers.yee.fmr.time_loop import update_B_fmr, update_E_fmr
from PyPIC3D.solvers.yee.fmr.transfers import (
    _active_component_indices,
    _curl_read_indices,
    _indices_strictly_inside,
    _strict_interior_indices,
    interpolate_coarse_to_fine,
    interpolate_fine_to_coarse,
)
from tests.kernel_fixtures import kernel_parameters


jax.config.update("jax_enable_x64", True)


@lru_cache(maxsize=1)
def _fmr_case():
    n = 12
    static_parameters, dynamic_parameters = kernel_parameters(
        Nx=n,
        Ny=n,
        Nz=n,
        x_wind=1.0,
        y_wind=1.0,
        z_wind=1.0,
        x_min=0.0,
        y_min=0.0,
        z_min=0.0,
        dx=1.0/n,
        dy=1.0/n,
        dz=1.0/n,
        dt=1.0e-3,
        tile_shape=(n, n, n),
        guard_cells=2,
        C=1.0,
        eps=1.0,
        mu=1.0,
    )
    config = {
        "fmr": {
            "enabled": True,
            "levels": [{
                "parent": 0,
                "refinement_ratio": 2,
                "coarse_start": [3, 3, 3],
                "coarse_stop": [9, 9, 9],
            }],
        }
    }
    geometry = {
        "Nx": n,
        "Ny": n,
        "Nz": n,
        "dx": 1.0/n,
        "dy": 1.0/n,
        "dz": 1.0/n,
        "x_min": 0.0,
        "x_max": 1.0,
        "y_min": 0.0,
        "y_max": 1.0,
        "z_min": 0.0,
        "z_max": 1.0,
    }
    levels = load_fmr_levels(config, geometry, static_parameters.tile_shape)
    static_parameters = static_parameters._replace(
        fmr_enabled=True,
        fmr_levels=levels,
    )
    dynamic_parameters = dynamic_parameters._replace(
        fmr=build_fmr_hierarchy(static_parameters, dynamic_parameters)
    )
    E0, B0, J0, phi, rho = initialize_fields(static_parameters, dynamic_parameters)
    E, B, J = initialize_fmr_field_levels(
        E0,
        B0,
        J0,
        static_parameters,
        dynamic_parameters,
    )
    return static_parameters, dynamic_parameters, E, B, J, rho, phi


def _coordinates(grids, locations):
    axes = component_coordinate_axes(grids, locations)
    return (
        axes[0][None, None, None, :, None, None],
        axes[1][None, None, None, None, :, None],
        axes[2][None, None, None, None, None, :],
    )


def _polynomial_vector(grids, locations, degree):
    result = []
    for component, component_locations in enumerate(locations):
        x, y, z = _coordinates(grids, component_locations)
        value = 1.0 + (component + 1.0)*x - 0.3*y + 0.2*z + x*y + y*z + x*z
        value += x**2 - 0.5*y**2 + 0.25*z**2
        if degree == 3:
            value += 0.2*x**3 - 0.1*y**3 + 0.05*z**3 + 0.3*x*y*z
        result.append(value)
    return tuple(result)


def _map_values(component, transfer_map):
    target = transfer_map.target_indices
    return component[
        0, 0, 0, target[:, 0], target[:, 1], target[:, 2]
    ]


def _deep_shadow_indices(static_parameters, dynamic_parameters, locations, maps):
    """Derive test-only inactive coarse indices not refreshed by a transfer."""

    parent_runtime = dynamic_parameters.fmr.levels[0]
    fine_level = static_parameters.fmr_levels[1]
    bounds = tuple(zip(fine_level.lower, fine_level.upper))
    result = []
    for component_locations, transfer_map in zip(locations, maps):
        axes = component_coordinate_axes(parent_runtime.grids, component_locations)
        tolerance = _coordinate_tolerance(*axes)
        covered = {
            tuple(index)
            for index in np.asarray(_strict_interior_indices(axes, bounds, tolerance))
        }
        refreshed = {
            tuple(index)
            for index in np.asarray(transfer_map.target_indices)
        }
        result.append(jnp.asarray(sorted(covered - refreshed), dtype=jnp.int32))
    return tuple(result)


class TestFMRConfiguration(unittest.TestCase):
    def test_only_fixed_fourth_order_transfer_and_ratio_two_are_supported(self):
        static, _, *_ = _fmr_case()
        geometry = {
            "Nx": 12, "Ny": 12, "Nz": 12,
            "dx": 1/12, "dy": 1/12, "dz": 1/12,
            "x_min": 0.0, "x_max": 1.0,
            "y_min": 0.0, "y_max": 1.0,
            "z_min": 0.0, "z_max": 1.0,
        }

        config = {
            "fmr": {
                "enabled": True,
                "interpolation_order": 4,
                "levels": [{
                    "parent": 0,
                    "refinement_ratio": 2,
                    "coarse_start": [3, 3, 3],
                    "coarse_stop": [9, 9, 9],
                }],
            }
        }
        with self.assertRaisesRegex(NotImplementedError, "interpolation_order"):
            validate_fmr_configuration(config, {"solver": "electrodynamic_yee"}, {})
        with self.assertRaisesRegex(NotImplementedError, "interpolation_order"):
            load_fmr_levels(config, geometry, static.tile_shape)

        for ratio in (3, 4):
            config = {"fmr": {"enabled": True, "levels": [{
                "parent": 0,
                "refinement_ratio": ratio,
                "coarse_start": [3, 3, 3],
                "coarse_stop": [9, 9, 9],
            }]}}
            with self.subTest(ratio=ratio):
                with self.assertRaisesRegex(ValueError, "refinement_ratio = 2"):
                    load_fmr_levels(config, geometry, static.tile_shape)

    def test_fourth_order_transfer_requires_three_parent_cells_per_axis(self):
        geometry = {
            "Nx": 8, "Ny": 8, "Nz": 8,
            "dx": 1/8, "dy": 1/8, "dz": 1/8,
            "x_min": 0.0, "x_max": 1.0,
            "y_min": 0.0, "y_max": 1.0,
            "z_min": 0.0, "z_max": 1.0,
        }

        config = {"fmr": {"enabled": True, "levels": [{
            "parent": 0,
            "refinement_ratio": 2,
            "coarse_start": [2, 2, 2],
            "coarse_stop": [4, 5, 5],
        }]}}
        with self.assertRaisesRegex(ValueError, "at least three parent cells"):
            load_fmr_levels(config, geometry, (8, 8, 8))

        config["fmr"]["levels"][0]["coarse_stop"] = [5, 5, 5]
        levels = load_fmr_levels(config, geometry, (8, 8, 8))
        self.assertEqual(levels[1].shape, (6, 6, 6))

    def test_scope_validation_still_rejects_non_field_fmr(self):
        config = {"fmr": {"enabled": True, "levels": [{}]}}
        validate_fmr_configuration(config, {"solver": "electrodynamic_yee"}, {})
        with self.assertRaises(NotImplementedError):
            validate_fmr_configuration(config, {"solver": "electrostatic"}, {})


class TestFMRTransfers(unittest.TestCase):
    def test_fourth_order_coarse_to_fine_maps_are_exact_through_degree_three(self):
        _, dynamic, E, B, *_ = _fmr_case()
        parent_data, fine_data = dynamic.fmr.levels
        interface = dynamic.fmr.interface
        for locations, templates, maps in (
            (E_FIELD_LOCATIONS, E, interface.e_coarse_to_fine_maps),
            (B_FIELD_LOCATIONS, B, interface.b_coarse_to_fine_maps),
        ):
            parent = _polynomial_vector(parent_data.grids, locations, degree=3)
            exact = _polynomial_vector(fine_data.grids, locations, degree=3)
            actual = interpolate_coarse_to_fine(
                parent,
                tuple(jnp.zeros_like(component) for component in templates[1]),
                maps,
            )

            for component, (computed, expected, transfer_map, component_locations) in enumerate(zip(
                actual,
                exact,
                maps,
                locations,
            )):
                with self.subTest(locations=locations, component=component):
                    self.assertEqual(transfer_map.source_indices.shape[1:], (64, 3))
                    self.assertTrue(jnp.allclose(jnp.sum(transfer_map.weights, axis=1), 1.0))
                    self.assertTrue(jnp.allclose(
                        _map_values(computed, transfer_map),
                        _map_values(expected, transfer_map),
                        rtol=2.0e-12,
                        atol=2.0e-12,
                    ))

                    parent_axes = component_coordinate_axes(parent_data.grids, component_locations)
                    fine_axes = component_coordinate_axes(fine_data.grids, component_locations)
                    source = transfer_map.source_indices
                    for axis in range(3):
                        interpolated_coordinate = jnp.sum(
                            transfer_map.weights * parent_axes[axis][source[:, :, axis]],
                            axis=1,
                        )
                        target_coordinate = fine_axes[axis][transfer_map.target_indices[:, axis]]
                        self.assertTrue(jnp.allclose(
                            interpolated_coordinate,
                            target_coordinate,
                            rtol=0.0,
                            atol=2.0e-14,
                        ))

    def test_fourth_order_fine_to_coarse_maps_are_exact_through_degree_three(self):
        _, dynamic, E, B, *_ = _fmr_case()
        parent_data, fine_data = dynamic.fmr.levels
        interface = dynamic.fmr.interface
        for locations, templates, maps in (
            (E_FIELD_LOCATIONS, E, interface.e_fine_to_coarse_maps),
            (B_FIELD_LOCATIONS, B, interface.b_fine_to_coarse_maps),
        ):
            fine = _polynomial_vector(fine_data.grids, locations, degree=3)
            exact = _polynomial_vector(parent_data.grids, locations, degree=3)
            actual = interpolate_fine_to_coarse(
                fine,
                tuple(jnp.zeros_like(component) for component in templates[0]),
                maps,
            )
            for component, (computed, expected, transfer_map) in enumerate(zip(actual, exact, maps)):
                with self.subTest(locations=locations, component=component):
                    self.assertEqual(transfer_map.source_indices.shape[1:], (64, 3))
                    self.assertTrue(jnp.allclose(jnp.sum(transfer_map.weights, axis=1), 1.0))
                    self.assertTrue(jnp.allclose(
                        _map_values(computed, transfer_map),
                        _map_values(expected, transfer_map),
                        rtol=2.0e-12,
                        atol=2.0e-12,
                    ))

    def test_fine_ghost_cells_cover_face_edge_and_corner_neighborhoods(self):
        static, dynamic, *_ = _fmr_case()
        fine_level = static.fmr_levels[1]
        fine_data = dynamic.fmr.levels[1]
        interface = dynamic.fmr.interface
        bounds = tuple(zip(fine_level.lower, fine_level.upper))

        for locations_tuple, coarse_to_fine_maps in (
            (E_FIELD_LOCATIONS, interface.e_coarse_to_fine_maps),
            (B_FIELD_LOCATIONS, interface.b_coarse_to_fine_maps),
        ):
            for locations, transfer_map in zip(locations_tuple, coarse_to_fine_maps):
                axes = component_coordinate_axes(fine_data.grids, locations)
                target = np.asarray(transfer_map.target_indices)
                near_interface = np.zeros(target.shape[0], dtype=np.int32)
                for axis, (lower, upper) in enumerate(bounds):
                    coordinates = np.asarray(axes[axis])[target[:, axis]]
                    distance = np.minimum(np.abs(coordinates - lower), np.abs(coordinates - upper))
                    near_interface += distance <= 0.5 * fine_level.spacing[axis] + 2.0e-14

                # Yee components do not generally lie exactly at geometric
                # corners.  The half-cell staggered values adjacent to all
                # three faces are the component-specific corner ghost values.
                self.assertTrue({1, 2, 3}.issubset(set(near_interface.tolist())))

    def test_active_curl_reads_and_fine_ghost_donors_are_refreshed(self):
        static, dynamic, *_ = _fmr_case()
        parent_level, fine_level = static.fmr_levels
        parent_data = dynamic.fmr.levels[0]
        fine_data = dynamic.fmr.levels[1]
        interface = dynamic.fmr.interface
        g = static.guard_cells
        bounds = tuple(zip(fine_level.lower, fine_level.upper))

        e_deep = _deep_shadow_indices(
            static,
            dynamic,
            E_FIELD_LOCATIONS,
            interface.e_fine_to_coarse_maps,
        )
        b_deep = _deep_shadow_indices(
            static,
            dynamic,
            B_FIELD_LOCATIONS,
            interface.b_fine_to_coarse_maps,
        )

        for locations_tuple, output_locations, offset, fine_maps, coarse_maps, deep_sets in (
            (
                E_FIELD_LOCATIONS,
                B_FIELD_LOCATIONS,
                1,
                interface.e_coarse_to_fine_maps,
                interface.e_fine_to_coarse_maps,
                e_deep,
            ),
            (
                B_FIELD_LOCATIONS,
                E_FIELD_LOCATIONS,
                -1,
                interface.b_coarse_to_fine_maps,
                interface.b_fine_to_coarse_maps,
                b_deep,
            ),
        ):
            parent_output_active = _active_component_indices(
                parent_level, parent_data.grids, output_locations, bounds, g, fine=False
            )
            fine_output_active = _active_component_indices(
                fine_level, fine_data.grids, output_locations, bounds, g, fine=True
            )

            for component, (locations, fine_map, coarse_map, deep) in enumerate(zip(
                locations_tuple, fine_maps, coarse_maps, deep_sets
            )):
                fine_axes = component_coordinate_axes(fine_data.grids, locations)
                parent_axes = component_coordinate_axes(parent_data.grids, locations)
                tolerance = _coordinate_tolerance(*parent_axes, *fine_axes)

                fine_reads = _curl_read_indices(fine_output_active, component, offset)
                fine_ghost_reads = fine_reads[~_indices_strictly_inside(
                    fine_reads, fine_axes, bounds, tolerance
                )]
                fine_targets = {tuple(index) for index in np.asarray(fine_map.target_indices)}
                self.assertTrue(
                    {tuple(index) for index in np.asarray(fine_ghost_reads)}.issubset(fine_targets)
                )

                parent_reads = _curl_read_indices(parent_output_active, component, offset)
                covered_parent_reads = parent_reads[_indices_strictly_inside(
                    parent_reads, parent_axes, bounds, tolerance
                )]
                coarse_targets = {tuple(index) for index in np.asarray(coarse_map.target_indices)}
                self.assertTrue(
                    {tuple(index) for index in np.asarray(covered_parent_reads)}.issubset(coarse_targets)
                )

                donors = jnp.unique(fine_map.source_indices.reshape((-1, 3)), axis=0)
                covered_donors = donors[_indices_strictly_inside(
                    donors, parent_axes, bounds, tolerance
                )]
                covered_donors = {tuple(index) for index in np.asarray(covered_donors)}
                deep = {tuple(index) for index in np.asarray(deep)}
                self.assertTrue(covered_donors.issubset(coarse_targets))
                self.assertTrue(covered_donors.isdisjoint(deep))
                self.assertTrue(deep)

                covered = {
                    tuple(index)
                    for index in np.asarray(_strict_interior_indices(
                        parent_axes, bounds, tolerance
                    ))
                }
                self.assertEqual(covered, coarse_targets | deep)
                self.assertTrue(coarse_targets.isdisjoint(deep))

    def test_manufactured_e_interface_constraint_is_at_roundoff(self):
        _, dynamic, *_ = _fmr_case()
        parent_data, fine_data = dynamic.fmr.levels
        interface = dynamic.fmr.interface
        exact_fine = _polynomial_vector(fine_data.grids, E_FIELD_LOCATIONS, degree=3)
        E = (
            _polynomial_vector(parent_data.grids, E_FIELD_LOCATIONS, degree=3),
            exact_fine,
        )
        synchronized = synchronize_e_levels(E, dynamic)

        residual = 0.0
        for actual, exact, transfer_map in zip(
            synchronized[1],
            exact_fine,
            interface.e_coarse_to_fine_maps,
        ):
            residual = max(
                residual,
                float(jnp.max(jnp.abs(
                    _map_values(actual, transfer_map)
                    - _map_values(exact, transfer_map)
                ))),
            )
        self.assertLess(residual, 2.0e-12)

    def test_synchronization_preserves_deep_shadow_and_is_idempotent(self):
        static, dynamic, E, B, *_ = _fmr_case()
        parent_data, fine_data = dynamic.fmr.levels
        interface = dynamic.fmr.interface
        e_deep = _deep_shadow_indices(
            static, dynamic, E_FIELD_LOCATIONS, interface.e_fine_to_coarse_maps
        )
        b_deep = _deep_shadow_indices(
            static, dynamic, B_FIELD_LOCATIONS, interface.b_fine_to_coarse_maps
        )

        for locations, deep_sets, synchronize in (
            (E_FIELD_LOCATIONS, e_deep, synchronize_e_levels),
            (B_FIELD_LOCATIONS, b_deep, synchronize_b_levels),
        ):
            parent = list(_polynomial_vector(parent_data.grids, locations, degree=2))
            fine = _polynomial_vector(fine_data.grids, locations, degree=3)
            for component, deep in enumerate(deep_sets):
                parent[component] = parent[component].at[
                    0, 0, 0, deep[:, 0], deep[:, 1], deep[:, 2]
                ].set(1000.0 + component, unique_indices=True)

            synchronized = synchronize((tuple(parent), fine), dynamic)
            synchronized_twice = synchronize(synchronized, dynamic)
            for component, deep in enumerate(deep_sets):
                self.assertTrue(jnp.all(
                    synchronized[0][component][
                        0, 0, 0, deep[:, 0], deep[:, 1], deep[:, 2]
                    ] == 1000.0 + component
                ))
            for actual, expected in zip(
                jax.tree_util.tree_leaves(synchronized),
                jax.tree_util.tree_leaves(synchronized_twice),
            ):
                self.assertTrue(jnp.allclose(actual, expected, rtol=0.0, atol=2.0e-14))

    def test_deep_shadow_sentinels_do_not_affect_active_curls_or_fine_ghosts(self):
        static, dynamic, *_ = _fmr_case()
        parent_data, fine_data = dynamic.fmr.levels
        interface = dynamic.fmr.interface
        e_deep = _deep_shadow_indices(
            static, dynamic, E_FIELD_LOCATIONS, interface.e_fine_to_coarse_maps
        )
        b_deep = _deep_shadow_indices(
            static, dynamic, B_FIELD_LOCATIONS, interface.b_fine_to_coarse_maps
        )
        E = (
            _polynomial_vector(parent_data.grids, E_FIELD_LOCATIONS, degree=2),
            _polynomial_vector(fine_data.grids, E_FIELD_LOCATIONS, degree=2),
        )
        B = (
            _polynomial_vector(parent_data.grids, B_FIELD_LOCATIONS, degree=2),
            _polynomial_vector(fine_data.grids, B_FIELD_LOCATIONS, degree=2),
        )

        def with_sentinels(levels, deep_sets, base):
            parent = list(levels[0])
            for component, deep in enumerate(deep_sets):
                parent[component] = parent[component].at[
                    0, 0, 0, deep[:, 0], deep[:, 1], deep[:, 2]
                ].set(base + 10.0 * component, unique_indices=True)
            return tuple(parent), levels[1]

        E_sentinel = with_sentinels(E, e_deep, 10000.0)
        B_sentinel = with_sentinels(B, b_deep, -10000.0)

        for reference, perturbed in (
            (fmr_curl_e_to_b(E, static, dynamic), fmr_curl_e_to_b(E_sentinel, static, dynamic)),
            (fmr_curl_b_to_e(B, static, dynamic), fmr_curl_b_to_e(B_sentinel, static, dynamic)),
        ):
            for actual, expected in zip(
                jax.tree_util.tree_leaves(reference),
                jax.tree_util.tree_leaves(perturbed),
            ):
                self.assertTrue(jnp.allclose(actual, expected, rtol=0.0, atol=2.0e-14))

        for reference, perturbed, maps in (
            (
                synchronize_e_levels(E, dynamic),
                synchronize_e_levels(E_sentinel, dynamic),
                interface.e_coarse_to_fine_maps,
            ),
            (
                synchronize_b_levels(B, dynamic),
                synchronize_b_levels(B_sentinel, dynamic),
                interface.b_coarse_to_fine_maps,
            ),
        ):
            for reference_component, perturbed_component, transfer_map in zip(
                reference[1], perturbed[1], maps
            ):
                self.assertTrue(jnp.allclose(
                    _map_values(reference_component, transfer_map),
                    _map_values(perturbed_component, transfer_map),
                    rtol=0.0,
                    atol=2.0e-14,
                ))


class TestFMRExplicitCurlsAndUpdates(unittest.TestCase):
    def test_production_fmr_modules_have_no_generated_transpose(self):
        source = inspect.getsource(curls_module)
        for forbidden in ("linear_transpose", "jax.vjp", "jacobian"):
            self.assertNotIn(forbidden, source)

    def test_coarse_curl_has_no_refinement_ratio_scaling(self):
        static, dynamic, E, *_ = _fmr_case()
        parent_data, fine_data = dynamic.fmr.levels
        E_levels = (
            _polynomial_vector(parent_data.grids, E_FIELD_LOCATIONS, degree=2),
            _polynomial_vector(fine_data.grids, E_FIELD_LOCATIONS, degree=2),
        )
        curl0, _ = fmr_curl_e_to_b(E_levels, static, dynamic)
        # For the chosen polynomial, curl_x = dEz/dy - dEy/dz. At the domain
        # center this is O(1); an accidental 1/r^2 factor would miss by 75%.
        i = j = k = 1
        x, y, z = (
            axis[static.guard_cells + index]
            for axis, index in zip(
                component_coordinate_axes(parent_data.grids, B_FIELD_LOCATIONS[0]),
                (i, j, k),
            )
        )
        exact = (-0.3 + z + x - y) - (0.2 + y + x + 0.5*z)
        self.assertAlmostEqual(float(curl0[0][0, 0, 0, i, j, k]), float(exact), places=2)

    def test_updates_preserve_shapes_ownership_and_interface_constraints(self):
        static, dynamic, E, B, J, *_ = _fmr_case()
        parent_data, fine_data = dynamic.fmr.levels
        E = (
            _polynomial_vector(parent_data.grids, E_FIELD_LOCATIONS, degree=2),
            _polynomial_vector(fine_data.grids, E_FIELD_LOCATIONS, degree=2),
        )
        B = (
            _polynomial_vector(parent_data.grids, B_FIELD_LOCATIONS, degree=2),
            _polynomial_vector(fine_data.grids, B_FIELD_LOCATIONS, degree=2),
        )
        shapes = tuple(leaf.shape for leaf in jax.tree_util.tree_leaves((E, B)))
        dtypes = tuple(leaf.dtype for leaf in jax.tree_util.tree_leaves((E, B)))

        advance_B = jax.jit(lambda E_levels, B_levels: update_B_fmr(
            E_levels, B_levels, static, dynamic
        ))
        advance_E = jax.jit(lambda E_levels, B_levels, J_levels: update_E_fmr(
            E_levels, B_levels, J_levels, static, dynamic
        ))
        B_after = advance_B(E, B)
        E_after = advance_E(E, B_after, J)
        B_after = advance_B(E_after, B_after)

        self.assertEqual(
            tuple(leaf.shape for leaf in jax.tree_util.tree_leaves((E_after, B_after))),
            shapes,
        )
        self.assertEqual(
            tuple(leaf.dtype for leaf in jax.tree_util.tree_leaves((E_after, B_after))),
            dtypes,
        )
        for leaf in jax.tree_util.tree_leaves((E_after, B_after)):
            self.assertTrue(jnp.all(jnp.isfinite(leaf)))

        for actual, expected in zip(
            jax.tree_util.tree_leaves(E_after),
            jax.tree_util.tree_leaves(synchronize_e_levels(E_after, dynamic)),
        ):
            self.assertTrue(jnp.allclose(actual, expected, rtol=0.0, atol=2.0e-14))
        for actual, expected in zip(
            jax.tree_util.tree_leaves(B_after),
            jax.tree_util.tree_leaves(synchronize_b_levels(B_after, dynamic)),
        ):
            self.assertTrue(jnp.allclose(actual, expected, rtol=0.0, atol=2.0e-14))


if __name__ == "__main__":
    unittest.main()
