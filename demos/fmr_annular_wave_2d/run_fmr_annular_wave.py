"""Focus a z-invariant annular electromagnetic pulse through an FMR patch."""

import argparse
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
    synchronize_b_levels,
    synchronize_e_levels,
)
from PyPIC3D.solvers.yee.fmr.grids import component_coordinate_axes


CONFIG_PATH = DEMO_DIR / "fmr_annular_wave_2d.toml"

PULSE_RADIUS = 0.62
PULSE_WIDTH = 0.08
PULSE_WAVELENGTH = 0.16


def component_coordinates(grids, locations):
    """Return broadcastable coordinates for one live Yee component grid."""

    x_axis, y_axis, z_axis = component_coordinate_axes(grids, locations)
    return (
        x_axis[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, jnp.newaxis, jnp.newaxis],
        y_axis[jnp.newaxis, jnp.newaxis, jnp.newaxis, jnp.newaxis, :, jnp.newaxis],
        z_axis[jnp.newaxis, jnp.newaxis, jnp.newaxis, jnp.newaxis, jnp.newaxis, :],
    )


def annular_electric_field(x, y, z):
    """Evaluate the radial wave packet while retaining the full slab shape."""

    radius = jnp.sqrt(x**2 + y**2)
    radial_offset = radius - PULSE_RADIUS
    envelope = jnp.exp(-0.5 * (radial_offset / PULSE_WIDTH) ** 2)
    carrier = jnp.cos(2.0 * jnp.pi * radial_offset / PULSE_WAVELENGTH)
    return envelope * carrier + 0.0 * z


def annular_wave_fields(grids, wave_speed, templates):
    """Evaluate the inward TMz pulse on the staggered Yee component grids."""

    E_template, B_template = templates
    Ez_coordinates = component_coordinates(grids, E_FIELD_LOCATIONS[2])
    Ez = annular_electric_field(*Ez_coordinates)
    E = (
        jnp.zeros_like(E_template[0]),
        jnp.zeros_like(E_template[1]),
        jnp.broadcast_to(Ez, E_template[2].shape),
    )

    x_Bx, y_Bx, z_Bx = component_coordinates(grids, B_FIELD_LOCATIONS[0])
    radius_Bx = jnp.sqrt(x_Bx**2 + y_Bx**2)
    safe_radius_Bx = jnp.where(radius_Bx > 0.0, radius_Bx, 1.0)
    Bphi_Bx = annular_electric_field(x_Bx, y_Bx, z_Bx) / wave_speed
    Bx = -y_Bx * Bphi_Bx / safe_radius_Bx

    x_By, y_By, z_By = component_coordinates(grids, B_FIELD_LOCATIONS[1])
    radius_By = jnp.sqrt(x_By**2 + y_By**2)
    safe_radius_By = jnp.where(radius_By > 0.0, radius_By, 1.0)
    Bphi_By = annular_electric_field(x_By, y_By, z_By) / wave_speed
    By = x_By * Bphi_By / safe_radius_By

    B = (
        jnp.broadcast_to(Bx, B_template[0].shape),
        jnp.broadcast_to(By, B_template[1].shape),
        jnp.zeros_like(B_template[2]),
    )
    return E, B


def initialize_annular_wave(fields, static_parameters, dynamic_parameters):
    """Populate and synchronize both FMR levels at t=0."""

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
        exact_E, exact_B = annular_wave_fields(
            level_data.grids,
            dynamic_parameters.C,
            (E_level, B_level),
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

    E_levels = synchronize_e_levels(tuple(initialized_E), dynamic_parameters)
    B_levels = synchronize_b_levels(tuple(initialized_B), dynamic_parameters)
    return (E_levels, B_levels, *fields[2:])


def run_demo(output_dir=None, number_of_steps=None):
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
    fields = initialize_annular_wave(fields, static_parameters, dynamic_parameters)

    plotting_interval = int(plotting_parameters["plotting_interval"])

    def advance_output_interval(state):
        def advance_one_step(_step, state_now):
            particles_now, fields_now = state_now
            return evolve_loop(
                particles_now,
                species_config,
                fields_now,
                static_parameters,
                dynamic_parameters,
            )

        return jax.lax.fori_loop(0, plotting_interval, advance_one_step, state)

    advance_output_interval_jit = jax.jit(advance_output_interval)
    output_dir = DEMO_DIR / "data" if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    field_writer = create_async_fmr_openpmd_field_writer(
        static_parameters,
        dynamic_parameters,
        str(output_dir),
        queue_size=int(plotting_parameters["openpmd_field_queue_size"]),
    )

    loop_error = None
    try:
        E_levels, B_levels, J_levels = fields[:3]
        enqueue_fmr_openpmd_field_output(
            field_writer,
            {"E": E_levels, "B": B_levels, "J": J_levels},
            dynamic_parameters,
            0,
            0,
        )

        if number_of_steps is None:
            number_of_steps = int(static_parameters.Nt)
        number_of_steps = min(int(number_of_steps), int(static_parameters.Nt))
        number_of_outputs = number_of_steps // plotting_interval
        for output_index in tqdm(
            range(1, number_of_outputs + 1),
            desc="Inward FMR annular wave",
        ):
            particles, fields = advance_output_interval_jit((particles, fields))
            timestep = output_index * plotting_interval

            E_levels, B_levels, J_levels = fields[:3]
            enqueue_fmr_openpmd_field_output(
                field_writer,
                {"E": E_levels, "B": B_levels, "J": J_levels},
                dynamic_parameters,
                output_index,
                timestep,
            )

        jax.block_until_ready(fields)
    except BaseException as exc:
        loop_error = exc
        raise
    finally:
        field_writer.close(raise_errors=loop_error is None)

    print(f"\nVisIt collection: {output_dir / 'fields.visit'}")
    return static_parameters, dynamic_parameters, fields


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--steps", type=int)
    arguments = parser.parse_args()

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    run_demo(arguments.output_dir, arguments.steps)


if __name__ == "__main__":
    main()
