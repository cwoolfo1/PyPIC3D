"""Run a small z-invariant x-y plane wave on the static FMR hierarchy."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import toml
from tqdm import tqdm


DEMO_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = DEMO_DIR.parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from PyPIC3D.diagnostics.async_writer import (
    create_async_fmr_openpmd_field_writer,
    enqueue_fmr_openpmd_field_output,
)
from PyPIC3D.initialization import initialize_simulation
from PyPIC3D.solvers.yee.fmr import (
    B_FIELD_LOCATIONS,
    E_FIELD_LOCATIONS,
    synchronize_e_levels,
)
from PyPIC3D.solvers.yee.fmr.grids import _component_coordinate_axes


CONFIG_PATH = DEMO_DIR / "fmr_linear_wave_2d.toml"


def component_coordinates(grids, locations):
    """Return broadcastable coordinates for one live Yee component grid."""

    x_axis, y_axis, z_axis = _component_coordinate_axes(grids, locations)
    return (
        x_axis[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, jnp.newaxis, jnp.newaxis],
        y_axis[jnp.newaxis, jnp.newaxis, jnp.newaxis, jnp.newaxis, :, jnp.newaxis],
        z_axis[jnp.newaxis, jnp.newaxis, jnp.newaxis, jnp.newaxis, jnp.newaxis, :],
    )


def linear_wave_fields(grids, wave_speed):
    """Evaluate Ez and its transverse B field on the staggered Yee grids."""

    kx = 2.0 * jnp.pi
    ky = 2.0 * jnp.pi
    omega = wave_speed * jnp.sqrt(kx**2 + ky**2)

    E = []
    for amplitude, locations in zip((0.0, 0.0, 1.0), E_FIELD_LOCATIONS):
        x, y, z = component_coordinates(grids, locations)
        phase = kx * x + ky * y + 0.0 * z
        E.append(amplitude * jnp.cos(phase))

    B = []
    magnetic_amplitudes = (ky / omega, -kx / omega, 0.0)
    for amplitude, locations in zip(magnetic_amplitudes, B_FIELD_LOCATIONS):
        x, y, z = component_coordinates(grids, locations)
        phase = kx * x + ky * y + 0.0 * z
        B.append(amplitude * jnp.cos(phase))

    return tuple(E), tuple(B)


def initialize_linear_wave(fields, static_parameters, dynamic_parameters):
    """Populate both FMR levels at t=0 without changing the production layout."""

    E_levels, B_levels = fields[:2]
    g = int(static_parameters.guard_cells)
    active = slice(g, -g)

    initialized_E = []
    initialized_B = []
    for E_level, B_level, level_data in zip(
        E_levels,
        B_levels,
        dynamic_parameters.fmr.levels,
    ):
        exact_E, exact_B = linear_wave_fields(level_data.grids, dynamic_parameters.C)

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

    E_levels = synchronize_e_levels(tuple(initialized_E), dynamic_parameters)
    return (E_levels, tuple(initialized_B), *fields[2:])


def run_demo():
    """Initialize, evolve, and write the complete openPMD FMR series."""

    config = toml.load(CONFIG_PATH)
    config["simulation_parameters"]["output_dir"] = str(DEMO_DIR)

    (
        evolve_loop,
        particles,
        fields,
        static_parameters,
        dynamic_parameters,
        plotting_parameters,
        _plasma_parameters,
        species_config,
    ) = initialize_simulation(config)
    fields = initialize_linear_wave(fields, static_parameters, dynamic_parameters)

    def advance_one_step(particles_now, fields_now):
        return evolve_loop(
            particles_now,
            species_config,
            fields_now,
            static_parameters,
            dynamic_parameters,
        )

    advance_one_step_jit = jax.jit(advance_one_step)
    output_dir = DEMO_DIR / "data"
    field_writer = create_async_fmr_openpmd_field_writer(
        static_parameters,
        dynamic_parameters,
        str(output_dir),
        queue_size=int(plotting_parameters["openpmd_field_queue_size"]),
    )

    loop_error = None
    try:
        plotting_interval = int(plotting_parameters["plotting_interval"])
        for timestep in tqdm(range(static_parameters.Nt), desc="FMR linear wave"):
            if timestep % plotting_interval == 0:
                plot_num = timestep // plotting_interval
                E_levels, B_levels, J_levels = fields[:3]
                enqueue_fmr_openpmd_field_output(
                    field_writer,
                    {"E": E_levels, "B": B_levels, "J": J_levels},
                    dynamic_parameters,
                    plot_num,
                    timestep,
                )

            particles, fields = advance_one_step_jit(particles, fields)

        jax.block_until_ready(fields)
    except BaseException as exc:
        loop_error = exc
        raise
    finally:
        field_writer.close(raise_errors=loop_error is None)

    print(f"\nVisIt collection: {output_dir / 'fields.visit'}")
    return static_parameters, dynamic_parameters, fields


def main():
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    run_demo()


if __name__ == "__main__":
    main()
