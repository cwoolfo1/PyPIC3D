import inspect
import unittest
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

import PyPIC3D.solvers.yee.fmr.B_fmr as B_fmr_module
import PyPIC3D.solvers.yee.fmr.E_fmr as E_fmr_module
from PyPIC3D.initialization import initialize_fields
from PyPIC3D.solvers.yee.fmr import (
    B_FIELD_LOCATIONS,
    E_FIELD_LOCATIONS,
    build_fmr_fields,
    build_fmr_parameters,
    fmr_curl_b_to_e,
    fmr_curl_e_to_b,
    load_fmr_from_toml,
    prolong_b_to_fine_interface,
    prolong_e_to_fine_interface,
    restrict_b_to_coarse_shadow,
    restrict_e_to_coarse_shadow,
    synchronize_b_levels,
    synchronize_e_levels,
    update_B_fmr,
    update_E_fmr,
    validate_fmr_configuration,
)
from PyPIC3D.solvers.yee.fmr.grids import _component_coordinate_axes
from PyPIC3D.solvers.yee.fmr.grids import _coordinate_tolerance
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
    levels = load_fmr_from_toml(config, geometry, static_parameters.tile_shape)
    static_parameters = static_parameters._replace(
        fmr_enabled=True,
        fmr_levels=levels,
    )
    dynamic_parameters = dynamic_parameters._replace(
        fmr=build_fmr_parameters(static_parameters, dynamic_parameters)
    )
    E0, B0, J0, phi, rho = initialize_fields(static_parameters, dynamic_parameters)
    E, B, J = build_fmr_fields(
        E0,
        B0,
        J0,
        static_parameters,
        dynamic_parameters,
    )
    return static_parameters, dynamic_parameters, E, B, J, rho, phi


def _coordinates(grids, locations):
    axes = _component_coordinate_axes(grids, locations)
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
            load_fmr_from_toml(config, geometry, static.tile_shape)

        for ratio in (3, 4):
            config = {"fmr": {"enabled": True, "levels": [{
                "parent": 0,
                "refinement_ratio": ratio,
                "coarse_start": [3, 3, 3],
                "coarse_stop": [9, 9, 9],
            }]}}
            with self.subTest(ratio=ratio):
                with self.assertRaisesRegex(ValueError, "refinement_ratio = 2"):
                    load_fmr_from_toml(config, geometry, static.tile_shape)

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
            load_fmr_from_toml(config, geometry, (8, 8, 8))

        config["fmr"]["levels"][0]["coarse_stop"] = [5, 5, 5]
        levels = load_fmr_from_toml(config, geometry, (8, 8, 8))
        self.assertEqual((levels[1].Nx, levels[1].Ny, levels[1].Nz), (6, 6, 6))

    def test_scope_validation_still_rejects_non_field_fmr(self):
        config = {"fmr": {"enabled": True, "levels": [{}]}}
        validate_fmr_configuration(config, {"solver": "electrodynamic_yee"}, {})
        with self.assertRaises(NotImplementedError):
            validate_fmr_configuration(config, {"solver": "electrostatic"}, {})


class TestFMRTransfers(unittest.TestCase):
    def test_fourth_order_interface_maps_are_exact_through_degree_three(self):
        _, dynamic, E, B, *_ = _fmr_case()
        parent_data, fine_data = dynamic.fmr.levels
        for locations, templates, maps, prolong in (
            (E_FIELD_LOCATIONS, E, fine_data.e_interface_maps, prolong_e_to_fine_interface),
            (B_FIELD_LOCATIONS, B, fine_data.b_interface_maps, prolong_b_to_fine_interface),
        ):
            parent = _polynomial_vector(parent_data.grids, locations, degree=3)
            exact = _polynomial_vector(fine_data.grids, locations, degree=3)
            actual = prolong(
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

                    parent_axes = _component_coordinate_axes(parent_data.grids, component_locations)
                    fine_axes = _component_coordinate_axes(fine_data.grids, component_locations)
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

    def test_fourth_order_shadow_maps_are_exact_through_degree_three(self):
        _, dynamic, E, B, *_ = _fmr_case()
        parent_data, fine_data = dynamic.fmr.levels
        for locations, templates, maps, restrict in (
            (E_FIELD_LOCATIONS, E, fine_data.e_restriction_maps, restrict_e_to_coarse_shadow),
            (B_FIELD_LOCATIONS, B, fine_data.b_restriction_maps, restrict_b_to_coarse_shadow),
        ):
            fine = _polynomial_vector(fine_data.grids, locations, degree=3)
            exact = _polynomial_vector(parent_data.grids, locations, degree=3)
            actual = restrict(
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

    def test_every_shadow_interface_donor_is_synchronized(self):
        _, dynamic, *_ = _fmr_case()
        parent_data = dynamic.fmr.levels[0]
        fine_data = dynamic.fmr.levels[1]
        fine_level = _fmr_case()[0].fmr_levels[1]
        bounds = (
            (fine_level.x_min, fine_level.x_max),
            (fine_level.y_min, fine_level.y_max),
            (fine_level.z_min, fine_level.z_max),
        )

        for locations_tuple, interface_maps, restriction_maps in (
            (E_FIELD_LOCATIONS, fine_data.e_interface_maps, fine_data.e_restriction_maps),
            (B_FIELD_LOCATIONS, fine_data.b_interface_maps, fine_data.b_restriction_maps),
        ):
            for locations, interface_map, restriction_map in zip(
                locations_tuple,
                interface_maps,
                restriction_maps,
            ):
                synchronized = {tuple(index) for index in np.asarray(restriction_map.target_indices)}
                axes = _component_coordinate_axes(parent_data.grids, locations)
                tolerance = float(_coordinate_tolerance(*axes))
                donors = {
                    tuple(index)
                    for index in np.asarray(interface_map.source_indices).reshape((-1, 3))
                }
                shadow_donors = {
                    index
                    for index in donors
                    if all(
                        lower + tolerance < float(axes[axis][index[axis]]) < upper - tolerance
                        for axis, (lower, upper) in enumerate(bounds)
                    )
                }

                # Every donor under the patch is reconstructed from the current
                # fine solution before it is used for interface prolongation.
                self.assertTrue(shadow_donors)
                self.assertTrue(shadow_donors.issubset(synchronized))


class TestFMRExplicitCurlsAndUpdates(unittest.TestCase):
    def test_production_fmr_modules_have_no_generated_transpose(self):
        source = inspect.getsource(B_fmr_module) + inspect.getsource(E_fmr_module)
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
                _component_coordinate_axes(parent_data.grids, B_FIELD_LOCATIONS[0]),
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

        B_after = update_B_fmr(E, B, static, dynamic)
        E_after = update_E_fmr(E, B_after, J, static, dynamic)
        B_after = update_B_fmr(E_after, B_after, static, dynamic)

        self.assertEqual(
            tuple(leaf.shape for leaf in jax.tree_util.tree_leaves((E_after, B_after))),
            shapes,
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
