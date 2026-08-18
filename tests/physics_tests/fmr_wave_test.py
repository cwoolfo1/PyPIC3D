import unittest

import jax
import jax.numpy as jnp
import numpy as np

from PyPIC3D.boundary_conditions.grid_and_stencil import BC_PERIODIC
from PyPIC3D.initialization import initialize_fields
from PyPIC3D.fmr import (
    B_FIELD_LOCATIONS,
    E_FIELD_LOCATIONS,
    build_fmr_fields,
    build_fmr_parameters,
    load_fmr_from_toml,
    time_loop_electrodynamic_fmr_fields,
)
from tests.kernel_fixtures import kernel_parameters


jax.config.update("jax_enable_x64", True)


# Diagnostic-only x64 CPU reference.  This records the first interface
# baseline without making its reflection amplitude a production threshold.
FMR_WAVE_AMPLITUDE_BASELINE = {
    "incident": 1.00000000,
    "reflected": 0.370517662,
    "transmitted": 0.832441070,
}


class TestFMRWaveCrossing(unittest.TestCase):
    def test_smooth_wave_crosses_coarse_fine_coarse_under_jit(self):
        # Keep a small transverse extent so this remains a practical regression,
        # while retaining a genuinely three-dimensional interior fine patch.
        nx, ny, nz = 96, 6, 6
        guard_cells = 2
        # The finest spacing is 0.5, so dt=0.15 remains below the 3-D CFL
        # limit used by the production initialization path.
        dt = 0.15
        patch_start = (36, 1, 1)
        patch_stop = (52, 5, 5)

        static_parameters, dynamic_parameters = kernel_parameters(
            Nx=nx,
            Ny=ny,
            Nz=nz,
            x_wind=float(nx),
            y_wind=float(ny),
            z_wind=float(nz),
            x_min=0.0,
            y_min=0.0,
            z_min=0.0,
            dx=1.0,
            dy=1.0,
            dz=1.0,
            dt=dt,
            tile_shape=(nx, ny, nz),
            guard_cells=guard_cells,
            boundary_conditions=(BC_PERIODIC, BC_PERIODIC, BC_PERIODIC),
            C=1.0,
            eps=1.0,
        )
        fmr_config = {
            "fmr": {
                "enabled": True,
                "levels": [
                    {
                        "parent": 0,
                        "refinement_ratio": 2,
                        "coarse_start": list(patch_start),
                        "coarse_stop": list(patch_stop),
                    }
                ],
            }
        }
        geometry_values = {
            "Nx": nx,
            "Ny": ny,
            "Nz": nz,
            "dx": 1.0,
            "dy": 1.0,
            "dz": 1.0,
            "x_min": 0.0,
            "x_max": float(nx),
            "y_min": 0.0,
            "y_max": float(ny),
            "z_min": 0.0,
            "z_max": float(nz),
        }
        fmr_levels = load_fmr_from_toml(
            fmr_config,
            geometry_values,
            static_parameters.tile_shape,
        )
        static_parameters = static_parameters._replace(
            fmr_enabled=True,
            fmr_levels=fmr_levels,
        )
        dynamic_parameters = dynamic_parameters._replace(
            fmr=build_fmr_parameters(static_parameters, dynamic_parameters)
        )

        E0, B0, J0, phi, rho = initialize_fields(static_parameters, dynamic_parameters)
        E_levels, B_levels, J_levels = build_fmr_fields(
            E0,
            B0,
            J0,
            static_parameters,
            dynamic_parameters,
        )

        pulse_center = 20.0
        pulse_width = 5.0
        wavelength = 16.0
        wave_number = 2.0 * jnp.pi / wavelength

        def pulse(x):
            envelope = jnp.exp(-0.5 * ((x - pulse_center) / pulse_width) ** 2)
            return envelope * jnp.cos(wave_number * (x - pulse_center))

        def component_x_coordinates(grids, locations):
            tiled_grid = grids.tiled_vertex_grid if locations[0] == "V" else grids.tiled_center_grid
            return tiled_grid[0][..., :, jnp.newaxis, jnp.newaxis]

        initialized_E = []
        initialized_B = []
        for E_level, B_level, level_data in zip(
            E_levels,
            B_levels,
            dynamic_parameters.fmr.levels,
        ):
            x_Ey = component_x_coordinates(level_data.grids, E_FIELD_LOCATIONS[1])
            x_Bz = component_x_coordinates(level_data.grids, B_FIELD_LOCATIONS[2])

            Ex, Ey, Ez = E_level
            Bx, By, Bz = B_level
            Ey = jnp.broadcast_to(pulse(x_Ey), Ey.shape)
            Bz = jnp.broadcast_to(pulse(x_Bz), Bz.shape)
            initialized_E.append((Ex, Ey, Ez))
            initialized_B.append((Bx, By, Bz))

        E_levels = tuple(initialized_E)
        B_levels = tuple(initialized_B)
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

        g = guard_cells
        root_Ey_initial = E_levels[0][1][0, 0, 0, g:-g, g:-g, g:-g]
        incident_amplitude = float(jnp.max(jnp.abs(root_Ey_initial[:patch_start[0]])))

        def advance_one_step(field_state):
            _, field_state = time_loop_electrodynamic_fmr_fields(
                (),
                (),
                field_state,
                static_parameters,
                dynamic_parameters,
            )
            return field_state

        # Compiling the production loop inside a fixed-count JAX loop tests the
        # complete FMR timestep while avoiding Python dispatch for every step.
        # At t=40.05 the pulse center is near x=60, eight coarse cells beyond
        # the upper refinement interface at x=52.
        number_of_steps = 267
        advance_wave = jax.jit(
            lambda field_state: jax.lax.fori_loop(
                0,
                number_of_steps,
                lambda _step, state: advance_one_step(state),
                field_state,
            )
        )
        fields = advance_wave(fields)

        E_levels, B_levels, J_levels = fields[:3]
        root_Ey = E_levels[0][1][0, 0, 0, g:-g, g:-g, g:-g]
        central_transverse_slice = (slice(2, 4), slice(2, 4))
        reflected_region = root_Ey[
            : patch_start[0] - 4,
            central_transverse_slice[0],
            central_transverse_slice[1],
        ]
        transmitted_region = root_Ey[
            patch_stop[0] + 4 :,
            central_transverse_slice[0],
            central_transverse_slice[1],
        ]
        reflected_amplitude = float(jnp.max(jnp.abs(reflected_region)))
        transmitted_amplitude = float(jnp.max(jnp.abs(transmitted_region)))

        amplitude_baseline = (
            f"incident={incident_amplitude:.8e}, "
            f"reflected={reflected_amplitude:.8e}, "
            f"transmitted={transmitted_amplitude:.8e}"
        )
        evolved_components = jax.tree_util.tree_leaves((E_levels, B_levels, J_levels))
        all_fields_finite = all(bool(jnp.all(jnp.isfinite(component))) for component in evolved_components)

        self.assertTrue(all_fields_finite, msg=amplitude_baseline)
        self.assertTrue(
            np.isfinite([incident_amplitude, reflected_amplitude, transmitted_amplitude]).all(),
            msg=amplitude_baseline,
        )
        self.assertGreater(incident_amplitude, 1.0e-3, msg=amplitude_baseline)
        self.assertGreater(transmitted_amplitude, 1.0e-8, msg=amplitude_baseline)
        # Reflection is intentionally recorded but not thresholded in the first
        # FMR baseline; later interface schemes can make that a convergence test.


if __name__ == "__main__":
    unittest.main()
