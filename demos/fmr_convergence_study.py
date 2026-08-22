"""Reproduce the two-level FMR Maxwell convergence tables."""

import argparse
import math

from tests.fmr_support import (
    REGIONS,
    _observed_orders,
    _regional_vector_error,
    _run_problem,
)


RESOLUTIONS = (12, 24, 48)
DIAGNOSTICS = ("E_l2", "E_linf", "B_l2", "B_linf")


def _orders(results, diagnostic):
    return _observed_orders(results, diagnostic)


def _print_resolution_study(results):
    for diagnostic in DIAGNOSTICS:
        orders = _orders(results, diagnostic)
        print(f"\n{diagnostic}")
        print("N coarse fine interface")
        for resolution, result in zip(RESOLUTIONS, results):
            print(
                resolution,
                *(f"{result[diagnostic][region]:.12e}" for region in REGIONS),
            )
        for region in REGIONS:
            print(region, "orders", *(f"{order:.8f}" for order in orders[region]))


def fixed_cfl_study(exact_e_interface=False):
    results = [
        _run_problem("periodic", resolution, exact_e_interface=exact_e_interface)
        for resolution in RESOLUTIONS
    ]
    _print_resolution_study(results)


def spatial_study():
    results = [
        _run_problem(
            "periodic",
            resolution,
            dt_value=2.5e-4,
            final_time_value=2.0e-2,
        )
        for resolution in RESOLUTIONS
    ]
    _print_resolution_study(results)

    half_dt = _run_problem(
        "periodic",
        RESOLUTIONS[-1],
        dt_value=1.25e-4,
        final_time_value=2.0e-2,
    )
    print("\nN=48 half-dt check")
    for diagnostic in DIAGNOSTICS:
        print(
            diagnostic,
            *(f"{half_dt[diagnostic][region]:.12e}" for region in REGIONS),
        )


def _state_differences(results):
    differences = []
    for coarse_dt, fine_dt in zip(results[:-1], results[1:]):
        E_l2, E_linf, _, _ = _regional_vector_error(
            coarse_dt["E_final"],
            fine_dt["E_final"],
            coarse_dt["E_region_masks"],
            coarse_dt["E_weights"],
            coarse_dt["guard_cells"],
        )
        B_l2, B_linf, _, _ = _regional_vector_error(
            coarse_dt["B_final"],
            fine_dt["B_final"],
            coarse_dt["B_region_masks"],
            coarse_dt["B_weights"],
            coarse_dt["guard_cells"],
        )
        differences.append({
            "E_l2": E_l2,
            "E_linf": E_linf,
            "B_l2": B_l2,
            "B_linf": B_linf,
        })
    return differences


def temporal_study():
    time_steps = (4.0e-3, 2.0e-3, 1.0e-3, 5.0e-4)
    for resolution in (24, 48):
        results = [
            _run_problem(
                "periodic",
                resolution,
                dt_value=dt,
                final_time_value=2.0e-2,
                return_state=True,
            )
            for dt in time_steps
        ]
        differences = _state_differences(results)
        print(f"\nN={resolution}")
        for diagnostic in DIAGNOSTICS:
            print(f"\n{diagnostic}")
            for dt, difference in zip(time_steps[:-1], differences):
                print(
                    f"dt={dt:.7f}",
                    *(f"{difference[diagnostic][region]:.12e}" for region in REGIONS),
                )
            for region in REGIONS:
                errors = [difference[diagnostic][region] for difference in differences]
                orders = tuple(
                    math.log(errors[index] / errors[index + 1], 2.0)
                    for index in range(len(errors) - 1)
                )
                print(region, "orders", *(f"{order:.8f}" for order in orders))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "study",
        choices=("fixed-cfl", "spatial", "temporal", "exact-interface-oracle"),
    )
    arguments = parser.parse_args()

    if arguments.study == "fixed-cfl":
        fixed_cfl_study()
    elif arguments.study == "spatial":
        spatial_study()
    elif arguments.study == "temporal":
        temporal_study()
    else:
        fixed_cfl_study(exact_e_interface=True)


if __name__ == "__main__":
    main()
