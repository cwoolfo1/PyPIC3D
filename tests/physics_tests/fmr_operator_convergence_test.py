import math
import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.solvers.yee.fmr import (
    B_FIELD_LOCATIONS,
    E_FIELD_LOCATIONS,
    fmr_curl_b_to_e,
    fmr_curl_e_to_b,
    interpolate_coarse_to_fine,
    interpolate_fine_to_coarse,
    update_B_fmr,
    update_E_fmr,
)
from tests.physics_tests.fmr_maxwell_convergence_test import (
    REGIONS,
    _active_vector,
    _build_fmr_case,
    _build_vector_region_masks,
    _component_coordinates,
    _periodic_fields,
)


jax.config.update("jax_enable_x64", True)


RESOLUTIONS = (12, 24, 48)
WAVE_NUMBER = 2.0 * jnp.pi


def _manufactured_e(grids):
    values = []
    for component, locations in enumerate(E_FIELD_LOCATIONS):
        x, y, z = _component_coordinates(grids, locations)
        if component == 0:
            value = 0.7*jnp.sin(WAVE_NUMBER*x) + 0.2*jnp.cos(WAVE_NUMBER*y) + 0.3*jnp.sin(WAVE_NUMBER*z)
        elif component == 1:
            value = -0.4*jnp.cos(WAVE_NUMBER*x) + 0.6*jnp.sin(WAVE_NUMBER*y) + 0.5*jnp.cos(WAVE_NUMBER*z)
        else:
            value = 0.8*jnp.sin(WAVE_NUMBER*x) - 0.3*jnp.cos(WAVE_NUMBER*y) + 0.9*jnp.sin(WAVE_NUMBER*z)
        values.append(value + 0.0*(x + y + z))
    return tuple(values)


def _exact_curl_e(grids):
    values = []
    for component, locations in enumerate(B_FIELD_LOCATIONS):
        x, y, z = _component_coordinates(grids, locations)
        if component == 0:
            value = 0.3*WAVE_NUMBER*jnp.sin(WAVE_NUMBER*y) + 0.5*WAVE_NUMBER*jnp.sin(WAVE_NUMBER*z)
        elif component == 1:
            value = 0.3*WAVE_NUMBER*jnp.cos(WAVE_NUMBER*z) - 0.8*WAVE_NUMBER*jnp.cos(WAVE_NUMBER*x)
        else:
            value = 0.4*WAVE_NUMBER*jnp.sin(WAVE_NUMBER*x) + 0.2*WAVE_NUMBER*jnp.sin(WAVE_NUMBER*y)
        values.append(value + 0.0*(x + y + z))
    return tuple(values)


def _manufactured_b(grids):
    values = []
    for component, locations in enumerate(B_FIELD_LOCATIONS):
        x, y, z = _component_coordinates(grids, locations)
        if component == 0:
            value = 0.3*jnp.cos(WAVE_NUMBER*x) + 0.4*jnp.sin(WAVE_NUMBER*y) - 0.2*jnp.cos(WAVE_NUMBER*z)
        elif component == 1:
            value = 0.8*jnp.sin(WAVE_NUMBER*x) - 0.6*jnp.cos(WAVE_NUMBER*y) + 0.7*jnp.sin(WAVE_NUMBER*z)
        else:
            value = -0.5*jnp.cos(WAVE_NUMBER*x) + 0.9*jnp.sin(WAVE_NUMBER*y) + 0.2*jnp.cos(WAVE_NUMBER*z)
        values.append(value + 0.0*(x + y + z))
    return tuple(values)


def _exact_curl_b(grids):
    values = []
    for component, locations in enumerate(E_FIELD_LOCATIONS):
        x, y, z = _component_coordinates(grids, locations)
        if component == 0:
            value = 0.9*WAVE_NUMBER*jnp.cos(WAVE_NUMBER*y) - 0.7*WAVE_NUMBER*jnp.cos(WAVE_NUMBER*z)
        elif component == 1:
            value = 0.2*WAVE_NUMBER*jnp.sin(WAVE_NUMBER*z) - 0.5*WAVE_NUMBER*jnp.sin(WAVE_NUMBER*x)
        else:
            value = 0.8*WAVE_NUMBER*jnp.cos(WAVE_NUMBER*x) - 0.4*WAVE_NUMBER*jnp.cos(WAVE_NUMBER*y)
        values.append(value + 0.0*(x + y + z))
    return tuple(values)


def _region_norms(numerical, exact, masks, guard_cells, numerical_is_active=False):
    if not numerical_is_active:
        numerical = tuple(_active_vector(level, guard_cells) for level in numerical)
    exact = tuple(_active_vector(level, guard_cells) for level in exact)
    errors = {}

    for region in REGIONS:
        squared_error = 0.0
        count = 0.0
        maximum_error = 0.0
        for numerical_level, exact_level, level_masks in zip(
            numerical,
            exact,
            masks[region],
        ):
            for numerical_component, exact_component, mask in zip(
                numerical_level,
                exact_level,
                level_masks,
            ):
                difference = numerical_component - exact_component
                squared_error += float(jnp.sum(jnp.where(mask, difference**2, 0.0)))
                count += float(jnp.sum(mask))
                maximum_error = max(
                    maximum_error,
                    float(jnp.max(jnp.where(mask, jnp.abs(difference), 0.0))),
                )
        errors[region] = (math.sqrt(squared_error / count), maximum_error)
    return errors


def _orders(results, diagnostic, norm_index):
    return {
        region: tuple(
            math.log(
                results[index][diagnostic][region][norm_index]
                / results[index + 1][diagnostic][region][norm_index],
                2.0,
            )
            for index in range(len(results) - 1)
        )
        for region in REGIONS
    }


def _transfer_norms(actual, exact, transfer_maps):
    squared_error = 0.0
    count = 0
    maximum_error = 0.0
    for actual_component, exact_component, transfer_map in zip(
        actual,
        exact,
        transfer_maps,
    ):
        target = transfer_map.target_indices
        difference = actual_component[
            0, 0, 0, target[:, 0], target[:, 1], target[:, 2]
        ] - exact_component[
            0, 0, 0, target[:, 0], target[:, 1], target[:, 2]
        ]
        squared_error += float(jnp.sum(difference**2))
        count += difference.size
        maximum_error = max(maximum_error, float(jnp.max(jnp.abs(difference))))
    return math.sqrt(squared_error / count), maximum_error


def _run_transfers(resolution):
    static, dynamic, _, _ = _build_fmr_case("periodic", resolution)
    parent_data, fine_data = dynamic.fmr.levels

    parent_E = _manufactured_e(parent_data.grids)
    exact_fine_E = _manufactured_e(fine_data.grids)
    fine_E = tuple(jnp.zeros_like(component) for component in exact_fine_E)
    supplied_fine_E = interpolate_coarse_to_fine(
        parent_E,
        fine_E,
        fine_data.e_coarse_to_fine_maps,
    )

    parent_B = _manufactured_b(parent_data.grids)
    exact_fine_B = _manufactured_b(fine_data.grids)
    fine_B = tuple(jnp.zeros_like(component) for component in exact_fine_B)
    supplied_fine_B = interpolate_coarse_to_fine(
        parent_B,
        fine_B,
        fine_data.b_coarse_to_fine_maps,
    )

    fine_E = _manufactured_e(fine_data.grids)
    exact_parent_E = _manufactured_e(parent_data.grids)
    parent_E = tuple(jnp.zeros_like(component) for component in exact_parent_E)
    supplied_parent_E = interpolate_fine_to_coarse(
        fine_E,
        parent_E,
        fine_data.e_fine_to_coarse_maps,
    )

    fine_B = _manufactured_b(fine_data.grids)
    exact_parent_B = _manufactured_b(parent_data.grids)
    parent_B = tuple(jnp.zeros_like(component) for component in exact_parent_B)
    supplied_parent_B = interpolate_fine_to_coarse(
        fine_B,
        parent_B,
        fine_data.b_fine_to_coarse_maps,
    )
    return {
        "E_prolongation": _transfer_norms(
            supplied_fine_E,
            exact_fine_E,
            fine_data.e_coarse_to_fine_maps,
        ),
        "B_prolongation": _transfer_norms(
            supplied_fine_B,
            exact_fine_B,
            fine_data.b_coarse_to_fine_maps,
        ),
        "E_restriction": _transfer_norms(
            supplied_parent_E,
            exact_parent_E,
            fine_data.e_fine_to_coarse_maps,
        ),
        "B_restriction": _transfer_norms(
            supplied_parent_B,
            exact_parent_B,
            fine_data.b_fine_to_coarse_maps,
        ),
    }


def _run_curls(resolution):
    static, dynamic, _, _ = _build_fmr_case("periodic", resolution)
    E = tuple(_manufactured_e(data.grids) for data in dynamic.fmr.levels)
    B = tuple(_manufactured_b(data.grids) for data in dynamic.fmr.levels)
    exact_curl_E = tuple(_exact_curl_e(data.grids) for data in dynamic.fmr.levels)
    exact_curl_B = tuple(_exact_curl_b(data.grids) for data in dynamic.fmr.levels)
    E_masks = _build_vector_region_masks(E_FIELD_LOCATIONS, static, dynamic, True, "E")
    B_masks = _build_vector_region_masks(B_FIELD_LOCATIONS, static, dynamic, True, "B")

    curl_E = fmr_curl_e_to_b(E, static, dynamic)
    curl_B = fmr_curl_b_to_e(B, E, static, dynamic)
    return {
        "curl_E": _region_norms(
            curl_E,
            exact_curl_E,
            B_masks,
            static.guard_cells,
            numerical_is_active=True,
        ),
        "curl_B": _region_norms(
            curl_B,
            exact_curl_B,
            E_masks,
            static.guard_cells,
            numerical_is_active=True,
        ),
    }


def _run_stages(resolution):
    static, dynamic, _, _ = _build_fmr_case("periodic", resolution)
    dt = 0.1 / resolution
    dynamic = dynamic._replace(dt=jnp.asarray(dt))
    initial = tuple(
        _periodic_fields(data.grids, 0.0, dynamic.C)
        for data in dynamic.fmr.levels
    )
    E0 = tuple(fields[0] for fields in initial)
    B0 = tuple(fields[1] for fields in initial)
    J = tuple(tuple(jnp.zeros_like(component) for component in level) for level in E0)
    E_masks = _build_vector_region_masks(E_FIELD_LOCATIONS, static, dynamic, True, "E")
    B_masks = _build_vector_region_masks(B_FIELD_LOCATIONS, static, dynamic, True, "B")

    B_half = update_B_fmr(E0, B0, static, dynamic)
    exact_B_half = tuple(
        _periodic_fields(data.grids, dt/2.0, dynamic.C)[1]
        for data in dynamic.fmr.levels
    )
    E_full = update_E_fmr(E0, B_half, J, static, dynamic)
    exact_E_full = tuple(
        _periodic_fields(data.grids, dt, dynamic.C)[0]
        for data in dynamic.fmr.levels
    )
    B_full = update_B_fmr(E_full, B_half, static, dynamic)
    exact_B_full = tuple(
        _periodic_fields(data.grids, dt, dynamic.C)[1]
        for data in dynamic.fmr.levels
    )

    return {
        "B_half": _region_norms(B_half, exact_B_half, B_masks, static.guard_cells),
        "E_full": _region_norms(E_full, exact_E_full, E_masks, static.guard_cells),
        "B_full": _region_norms(B_full, exact_B_full, B_masks, static.guard_cells),
    }


class TestFMROperatorConvergence(unittest.TestCase):
    def test_interface_transfers_have_sufficient_point_value_order(self):
        results = [_run_transfers(resolution) for resolution in RESOLUTIONS]
        for diagnostic in (
            "E_prolongation",
            "B_prolongation",
            "E_restriction",
            "B_restriction",
        ):
            for norm_index, norm_name in enumerate(("L2", "Linf")):
                errors = [result[diagnostic][norm_index] for result in results]
                orders = tuple(
                    math.log(errors[index] / errors[index + 1], 2.0)
                    for index in range(len(errors) - 1)
                )
                self.assertGreater(
                    orders[-1],
                    3.5,
                    msg=f"{diagnostic} {norm_name}: errors={errors}, orders={orders}",
                )

    def test_both_explicit_curls_are_second_order_in_every_region(self):
        results = [_run_curls(resolution) for resolution in RESOLUTIONS]
        lines = []
        for diagnostic in ("curl_E", "curl_B"):
            l2_orders = _orders(results, diagnostic, 0)
            linf_orders = _orders(results, diagnostic, 1)
            for region in REGIONS:
                lines.append(
                    f"{diagnostic} {region}: L2={l2_orders[region]}, "
                    f"Linf={linf_orders[region]}"
                )
                self.assertGreater(l2_orders[region][-1], 1.8, msg="\n".join(lines))

    def test_b_e_b_stages_are_second_order_consistent_in_every_region(self):
        results = [_run_stages(resolution) for resolution in RESOLUTIONS]
        lines = []
        for diagnostic in ("B_half", "E_full", "B_full"):
            l2_orders = _orders(results, diagnostic, 0)
            linf_orders = _orders(results, diagnostic, 1)
            for region in REGIONS:
                lines.append(
                    f"{diagnostic} {region}: L2={l2_orders[region]}, "
                    f"Linf={linf_orders[region]}"
                )
                self.assertGreater(l2_orders[region][-1], 1.8, msg="\n".join(lines))


if __name__ == "__main__":
    unittest.main()
