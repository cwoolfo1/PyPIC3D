# Christopher Woolford July 2024
# 3D PIC code in Python using the JAX library

import csv
import os
import time

import jax
from jax import block_until_ready
from tqdm import tqdm

from PyPIC3D.diagnostics.plotting import write_data
from PyPIC3D.diagnostics.async_writer import (
    create_async_tiled_openpmd_field_writer,
    create_async_tiled_openpmd_particle_writer,
    enqueue_openpmd_field_output,
    enqueue_openpmd_particle_output,
)
from PyPIC3D.diagnostics.output_adapters import build_field_output_map, field_map_for_output
from PyPIC3D.diagnostics.diagnostic_quantities import (
    compute_energy,
    compute_gauss_residual,
    compute_total_momentum,
)
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.utilities.field_helpers import add_external_fields
from PyPIC3D.utilities.simulation_helpers import setup_pmd_files
from PyPIC3D.utilities.toml_helpers import (
    dump_parameters_to_toml,
    load_config_file,
)
from PyPIC3D.initialization import initialize_simulation


def _raise_if_tiled_particles_overflowed(fields):
    if len(fields) < 8:
        return

    overflow = fields[-1]
    if bool(jax.device_get(overflow)):
        raise RuntimeError("tiled particle tile capacity overflowed during periodic retile")


def _gpu_requested(config):
    """Return the backend requested by either supported parameter section."""

    for section_name in ("simulation_parameters", "static_parameters"):
        section = config.get(section_name, {})
        if "GPUs" in section:
            return bool(section["GPUs"])
    return False


def _validate_backend_devices(use_gpu, devices):
    expected = {"gpu", "cuda"} if use_gpu else {"cpu"}
    matching = [device for device in devices if str(device.platform).lower() in expected]
    if not matching:
        requested = "CUDA GPU" if use_gpu else "CPU"
        available = ", ".join(str(device) for device in devices) or "none"
        raise RuntimeError(
            f"PyPIC3D requested the {requested} backend, but no matching JAX device was found. "
            f"Visible devices: {available}."
        )
    return matching


def configure_jax_backend(config):
    """Configure JAX before the first device query and report the selected device."""

    use_gpu = _gpu_requested(config)
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platforms", "cuda" if use_gpu else "cpu")
    try:
        devices = jax.devices()
    except Exception as exc:
        requested = "CUDA GPU" if use_gpu else "CPU"
        raise RuntimeError(
            f"PyPIC3D requested the {requested} backend, but JAX could not initialize it: {exc}"
        ) from exc

    matching = _validate_backend_devices(use_gpu, devices)
    print(f"JAX backend: {'GPU' if use_gpu else 'CPU'}")
    print(f"JAX devices: {', '.join(str(device) for device in matching)}")
    return tuple(matching)


def _initialize_conservation_file(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as stream:
        csv.writer(stream).writerow(
            (
                "time_s",
                "gauss_residual",
                "total_energy_j",
                "relative_energy_drift",
                "mean_charge_density_c_m3",
                "mean_jx_a_m2",
                "mean_jy_a_m2",
                "mean_jz_a_m2",
            )
        )


def _write_conservation_sample(path, values):
    with open(path, "a", newline="") as stream:
        csv.writer(stream).writerow(values)


def run_PyPIC3D(config_file):
    (
        loop,
        particles,
        fields,
        static_parameters,
        dynamic_parameters,
        plotting_parameters,
        plasma_parameters,
        species_config,
    ) = initialize_simulation(config_file)

    dt = dynamic_parameters.dt
    Nt = static_parameters.Nt
    output_dir = static_parameters.output_dir
    particle_species_names = plotting_parameters.get("particle_species_names")
    static_metric = static_parameters.solver == "static_metric"

    def loop_with_static_parameters(
        particles,
        species_config,
        fields,
        dynamic_parameters,
    ):
        return loop(
            particles,
            species_config,
            fields,
            static_parameters,
            dynamic_parameters,
        )

    jit_loop = jax.jit(loop_with_static_parameters)

    initial_energy = None
    conservation_path = os.path.join(output_dir, "data", "conservation.csv")
    conservation_enabled = bool(plotting_parameters.get("conservation_diagnostics", False))
    if conservation_enabled and not static_metric:
        _initialize_conservation_file(conservation_path)
    if not static_metric:
        E, B, J, rho, phi, external_fields, *rest = fields
        total_E, total_B = add_external_fields(E, B, external_fields)
        e_energy, b_energy, kinetic_energy = compute_energy(
            particles,
            total_E,
            total_B,
            static_parameters,
            dynamic_parameters,
            species_config=species_config,
        )
        initial_energy = e_energy + b_energy + kinetic_energy

    field_writer = None
    particle_writer = None
    if plotting_parameters["plot_openpmd_fields"]:
        setup_pmd_files(os.path.join(output_dir, "data"), "fields", ".h5")
        field_writer = create_async_tiled_openpmd_field_writer(
            static_parameters,
            dynamic_parameters,
            os.path.join(output_dir, "data"),
            filename="fields",
            file_extension=".h5",
            queue_size=int(plotting_parameters.get("openpmd_field_queue_size", 2)),
        )
    if plotting_parameters["plot_openpmd_particles"]:
        setup_pmd_files(os.path.join(output_dir, "data"), "particles", ".h5")
        particle_writer = create_async_tiled_openpmd_particle_writer(
            static_parameters,
            dynamic_parameters,
            os.path.join(output_dir, "data"),
            filename="particles",
            file_extension=".h5",
            queue_size=int(plotting_parameters.get("openpmd_particle_queue_size", 2)),
        )

    loop_error = None
    try:
        for t in tqdm(range(Nt)):
            if t % plotting_parameters["plotting_interval"] == 0:
                plot_num = t // plotting_parameters["plotting_interval"]

                if not static_metric:
                    E, B, J, rho, phi, external_fields, *rest = fields
                    total_E, total_B = add_external_fields(E, B, external_fields)
                    e_energy, b_energy, kinetic_energy = compute_energy(
                        particles,
                        total_E,
                        total_B,
                        static_parameters,
                        dynamic_parameters,
                        species_config=species_config,
                    )
                    total_energy = e_energy + b_energy + kinetic_energy
                    relative_energy_drift = abs(initial_energy - total_energy) / max(initial_energy, 1e-10)
                    write_data(f"{output_dir}/data/total_energy.txt", t * dt, total_energy)
                    write_data(
                        f"{output_dir}/data/energy_error.txt",
                        t * dt,
                        relative_energy_drift,
                    )
                    write_data(f"{output_dir}/data/electric_field_energy.txt", t * dt, e_energy)
                    write_data(f"{output_dir}/data/magnetic_field_energy.txt", t * dt, b_energy)
                    write_data(f"{output_dir}/data/kinetic_energy.txt", t * dt, kinetic_energy)

                    total_momentum = compute_total_momentum(particles, species_config=species_config)
                    write_data(f"{output_dir}/data/total_momentum.txt", t * dt, total_momentum)

                    if conservation_enabled:
                        rho_diagnostic = compute_rho(
                            particles,
                            species_config,
                            rho,
                            static_parameters,
                            dynamic_parameters,
                        )
                        diagnostic_fields = field_map_for_output(
                            {"E": total_E, "J": J, "rho": rho_diagnostic},
                            static_parameters,
                        )
                        gauss_residual = compute_gauss_residual(
                            diagnostic_fields["E"],
                            diagnostic_fields["rho"],
                            dynamic_parameters,
                        )
                        rho_interior = diagnostic_fields["rho"][1:-1, 1:-1, 1:-1]
                        current_interior = tuple(
                            component[1:-1, 1:-1, 1:-1]
                            for component in diagnostic_fields["J"]
                        )
                        _write_conservation_sample(
                            conservation_path,
                            (
                                float(t * dt),
                                float(gauss_residual),
                                float(total_energy),
                                float(relative_energy_drift),
                                float(jax.numpy.mean(rho_interior)),
                                *(float(jax.numpy.mean(component)) for component in current_interior),
                            ),
                        )

                if particle_writer is not None:
                    enqueue_openpmd_particle_output(
                        particle_writer,
                        particles,
                        dynamic_parameters,
                        plot_num,
                        t,
                        species_config=species_config,
                        species_names=particle_species_names,
                    )

                if field_writer is not None:
                    field_map = build_field_output_map(
                        fields,
                        particles,
                        species_config,
                        static_parameters,
                        dynamic_parameters,
                        include_fluid_velocity=bool(plotting_parameters["plotvelocities"]),
                        include_charge_density=bool(plotting_parameters["plotchargedensity"]),
                    )
                    plotting_parameters["field_map"] = field_map
                    enqueue_openpmd_field_output(field_writer, field_map, dynamic_parameters, plot_num, t)

            particles, fields = jit_loop(
                particles,
                species_config,
                fields,
                dynamic_parameters,
            )
            _raise_if_tiled_particles_overflowed(fields)

    except BaseException as exc:
        loop_error = exc
        raise
    finally:
        writer_error = None
        for writer in (particle_writer, field_writer):
            if writer is None:
                continue
            try:
                writer.close(raise_errors=loop_error is None)
            except BaseException as exc:
                if writer_error is None:
                    writer_error = exc
        if loop_error is None and writer_error is not None:
            raise writer_error

    return (
        static_parameters,
        dynamic_parameters,
        plotting_parameters,
        plasma_parameters,
        particles,
        fields,
        species_config,
    )


def main():
    toml_file = load_config_file()
    configure_jax_backend(toml_file)

    start = time.time()
    (
        static_parameters,
        dynamic_parameters,
        plotting_parameters,
        plasma_parameters,
        particles,
        fields,
        species_config,
    ) = block_until_ready(run_PyPIC3D(toml_file))
    end = time.time()

    if static_parameters.solver == "static_metric":
        print("Skipping final flat-space energy diagnostics for static_metric fields and covariant particle u_i\n")
    else:
        E, B, J, rho, phi, external_fields, *rest = fields
        total_E, total_B = add_external_fields(E, B, external_fields)
        e_energy, b_energy, kinetic_energy = compute_energy(
            particles,
            total_E,
            total_B,
            static_parameters,
            dynamic_parameters,
            species_config=species_config,
        )
        print(f"Final Electric Field Energy: {e_energy}")
        print(f"Final Magnetic Field Energy: {b_energy}")
        print(f"Final Kinetic Energy: {kinetic_energy}")
        print(f"Total Final Energy: {e_energy + b_energy + kinetic_energy}\n")

    duration = end - start
    Nt = static_parameters.Nt
    simulation_stats = {
        "total_time": duration,
        "total_iterations": Nt,
        "time_per_iteration": duration / Nt,
    }

    dump_parameters_to_toml(
        simulation_stats,
        static_parameters,
        dynamic_parameters,
        plasma_parameters,
        plotting_parameters,
        particles,
    )

    print("\nSimulation Complete")
    print(f"Total Simulation Time: {duration} s")
    print(f"Time Per Iteration: {duration / Nt} s")


if __name__ == "__main__":
    main()
