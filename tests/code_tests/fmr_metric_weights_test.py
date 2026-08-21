import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from PyPIC3D.solvers.yee.fmr import (
    B_FIELD_LOCATIONS,
    E_FIELD_LOCATIONS,
    build_fmr_parameters,
    load_fmr_from_toml,
)
from PyPIC3D.solvers.yee.fmr.grids import _component_coordinate_axes
from tests.kernel_fixtures import kernel_parameters


jax.config.update("jax_enable_x64", True)


DOMAIN_LENGTH = 1.0
GUARD_CELLS = 2


def _metric_case(resolution, refinement_ratio):
    spacing = DOMAIN_LENGTH / resolution
    static_parameters, dynamic_parameters = kernel_parameters(
        Nx=resolution,
        Ny=resolution,
        Nz=resolution,
        x_wind=DOMAIN_LENGTH,
        y_wind=DOMAIN_LENGTH,
        z_wind=DOMAIN_LENGTH,
        x_min=0.0,
        y_min=0.0,
        z_min=0.0,
        dx=spacing,
        dy=spacing,
        dz=spacing,
        tile_shape=(resolution, resolution, resolution),
        guard_cells=GUARD_CELLS,
    )

    patch_start = (resolution // 4,) * 3
    patch_stop = (3 * resolution // 4,) * 3
    config = {
        "fmr": {
            "enabled": True,
            "levels": [
                {
                    "parent": 0,
                    "refinement_ratio": refinement_ratio,
                    "coarse_start": list(patch_start),
                    "coarse_stop": list(patch_stop),
                }
            ],
        }
    }
    geometry = {
        "Nx": resolution,
        "Ny": resolution,
        "Nz": resolution,
        "dx": spacing,
        "dy": spacing,
        "dz": spacing,
        "x_min": 0.0,
        "x_max": DOMAIN_LENGTH,
        "y_min": 0.0,
        "y_max": DOMAIN_LENGTH,
        "z_min": 0.0,
        "z_max": DOMAIN_LENGTH,
    }
    levels = load_fmr_from_toml(
        config,
        geometry,
        static_parameters.tile_shape,
    )
    static_parameters = static_parameters._replace(
        fmr_enabled=True,
        fmr_levels=levels,
    )
    dynamic_parameters = dynamic_parameters._replace(
        fmr=build_fmr_parameters(static_parameters, dynamic_parameters)
    )
    return static_parameters, dynamic_parameters


def _active_component_axes(grids, locations, level):
    g = GUARD_CELLS
    return tuple(
        np.asarray(axis[g:g + cells])
        for axis, cells in zip(
            _component_coordinate_axes(grids, locations),
            (level.Nx, level.Ny, level.Nz),
        )
    )


def _component_quadrature(grids, locations, level, weight):
    axes = _active_component_axes(grids, locations, level)
    x, y, z = np.meshgrid(*axes, indexing="ij")
    field = (
        1.0
        + 0.1
        * np.sin(2.0 * np.pi * x + 0.2)
        * np.cos(2.0 * np.pi * y + 0.4)
        * np.sin(2.0 * np.pi * z + 0.7)
    )
    return float(np.sum(np.asarray(weight)[0, 0, 0] * field**2))


def _index_at_coordinate(axis, coordinate):
    matches = np.flatnonzero(np.isclose(axis, coordinate, rtol=0.0, atol=1.0e-13))
    if matches.size != 1:
        raise AssertionError(f"Expected one grid point at {coordinate}, found {matches.size}.")
    return int(matches[0])


class TestFMRCompositeMetricWeights(unittest.TestCase):
    def test_constant_fields_partition_the_domain_for_all_components(self):
        for resolution in (8, 16):
            for refinement_ratio in (2,):
                with self.subTest(
                    resolution=resolution,
                    refinement_ratio=refinement_ratio,
                ):
                    _, dynamic_parameters = _metric_case(
                        resolution,
                        refinement_ratio,
                    )
                    parent_data, fine_data = dynamic_parameters.fmr.levels

                    for field_name in ("e_weights", "b_weights"):
                        parent_weights = getattr(parent_data, field_name)
                        fine_weights = getattr(fine_data, field_name)
                        totals = [
                            float(jnp.sum(parent) + jnp.sum(fine))
                            for parent, fine in zip(parent_weights, fine_weights)
                        ]
                        self.assertTrue(np.allclose(
                            totals,
                            DOMAIN_LENGTH**3,
                            rtol=0.0,
                            atol=2.0e-13,
                        ))

    def test_ratio_two_interface_weights_follow_live_yee_coordinates(self):
        static_parameters, dynamic_parameters = _metric_case(8, 2)
        parent_level, fine_level = static_parameters.fmr_levels
        parent_data = dynamic_parameters.fmr.levels[0]
        coarse_volume = np.prod(parent_level.spacing)
        lower = fine_level.x_min

        for field_name, locations_tuple in (
            ("e_weights", E_FIELD_LOCATIONS),
            ("b_weights", B_FIELD_LOCATIONS),
        ):
            for component, locations in enumerate(locations_tuple):
                with self.subTest(field=field_name, component=component):
                    axes = _active_component_axes(
                        parent_data.grids,
                        locations,
                        parent_level,
                    )
                    weights = np.asarray(getattr(parent_data, field_name)[component])[0, 0, 0]

                    outside = tuple(0 for _ in range(3))
                    self.assertAlmostEqual(weights[outside] / coarse_volume, 1.0, places=14)

                    interior_coordinates = tuple(
                        lower + (spacing if location == "C" else 0.5 * spacing)
                        for location, spacing in zip(locations, parent_level.spacing)
                    )
                    interior = tuple(
                        _index_at_coordinate(axis, coordinate)
                        for axis, coordinate in zip(axes, interior_coordinates)
                    )
                    self.assertEqual(weights[interior], 0.0)

                    # In the legacy grid builder C is the integer-coordinate
                    # axis and V is the half-cell axis.  Interface partial
                    # volumes therefore occur on the C axes in this checkout.
                    interface_axes = [
                        axis for axis, location in enumerate(locations)
                        if location == "C"
                    ]
                    face = list(interior)
                    face_axis = interface_axes[0]
                    face[face_axis] = _index_at_coordinate(axes[face_axis], lower)
                    self.assertAlmostEqual(
                        weights[tuple(face)] / coarse_volume,
                        3.0 / 4.0,
                        places=14,
                    )

                    if len(interface_axes) == 2:
                        edge = list(interior)
                        for axis in interface_axes:
                            edge[axis] = _index_at_coordinate(axes[axis], lower)
                        self.assertAlmostEqual(
                            weights[tuple(edge)] / coarse_volume,
                            15.0 / 16.0,
                            places=14,
                        )

    def test_active_weights_are_strictly_positive_and_inactive_weights_are_zero(self):
        for refinement_ratio in (2,):
            with self.subTest(refinement_ratio=refinement_ratio):
                _, dynamic_parameters = _metric_case(8, refinement_ratio)
                parent_data, fine_data = dynamic_parameters.fmr.levels

                for weights in (
                    parent_data.e_weights,
                    parent_data.b_weights,
                    fine_data.e_weights,
                    fine_data.b_weights,
                ):
                    for weight in weights:
                        values = np.asarray(weight)
                        self.assertTrue(np.all(np.isfinite(values)))
                        self.assertTrue(np.all(values >= 0.0))
                        self.assertTrue(np.all(values[values != 0.0] > 0.0))

                coarse_volume = float(np.max(np.asarray(parent_data.e_weights[0])))
                for weights in (parent_data.e_weights, parent_data.b_weights):
                    for weight in weights:
                        positive = np.asarray(weight)[np.asarray(weight) > 0.0]
                        self.assertTrue(np.all(positive >= 0.5 * coarse_volume))

                for weight, mask in zip(
                    parent_data.b_weights,
                    parent_data.b_active_masks,
                ):
                    self.assertTrue(np.all(np.asarray(weight)[~np.asarray(mask)] == 0.0))
                for weight, mask in zip(
                    fine_data.b_weights,
                    fine_data.b_active_masks,
                ):
                    self.assertTrue(np.all(np.asarray(weight)[~np.asarray(mask)] == 0.0))

    def test_smooth_composite_quadrature_converges_for_all_components(self):
        resolutions = (8, 16, 32, 64)
        exact_integral = 1.0 + 0.1**2 / 8.0
        errors = {name: [[] for _ in range(3)] for name in ("E", "B")}

        for resolution in resolutions:
            static_parameters, dynamic_parameters = _metric_case(resolution, 2)
            parent_level, fine_level = static_parameters.fmr_levels
            parent_data, fine_data = dynamic_parameters.fmr.levels

            for name, locations_tuple, weight_name in (
                ("E", E_FIELD_LOCATIONS, "e_weights"),
                ("B", B_FIELD_LOCATIONS, "b_weights"),
            ):
                parent_weights = getattr(parent_data, weight_name)
                fine_weights = getattr(fine_data, weight_name)
                for component, locations in enumerate(locations_tuple):
                    quadrature = _component_quadrature(
                        parent_data.grids,
                        locations,
                        parent_level,
                        parent_weights[component],
                    )
                    quadrature += _component_quadrature(
                        fine_data.grids,
                        locations,
                        fine_level,
                        fine_weights[component],
                    )
                    errors[name][component].append(abs(quadrature - exact_integral))

        lines = ["Composite metric quadrature convergence"]
        for name in ("E", "B"):
            for component in range(3):
                component_errors = errors[name][component]
                orders = tuple(
                    math.log(component_errors[index] / component_errors[index + 1], 2.0)
                    for index in range(len(component_errors) - 1)
                )
                lines.append(
                    f"{name}{component}: errors={component_errors}, orders={orders}"
                )
                self.assertGreater(min(orders), 1.8, msg="\n".join(lines))

        print("\n" + "\n".join(lines))


if __name__ == "__main__":
    unittest.main()
