"""Constraint-solved, periodic warm electron-ion two-stream initial data.

Gaussian units have c = m_e = |q_e| = k_B = 1. Temperatures are k_B T,
not T/m, and NUMBER_DENSITY is the total electron (and ion) number density
per initial coordinate volume. The proper box volume is normalized to the
coordinate volume by the metric solve. The conformally flat periodic matter
constraints require nonzero mean curvature; we choose its expanding branch.
No homogeneous source is subtracted.

Particles and physical fields returned here are at a common time. Native
PyPIC3D initialization seeds the leapfrog histories once after this solve.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import toml

from JAX_BSSN.bssn.constraints import (
    compute_all_constraints_with_matter,
    compute_hamiltonian_constraint_with_matter,
)
from JAX_BSSN.diagnostics.constraints import compute_constraint_norms
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.diagnostics.static_metric import divergence
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.particles.velocity_distribution import maxwell_juttner_momentum
from PyPIC3D.relativity.bssn_parameters import build_bssn_parameters
from PyPIC3D.relativity.dynamical_relativity import (
    bssn_to_tiles,
    build_bssn_metric,
    tiles_to_bssn,
)
from PyPIC3D.relativity.energy_momentum_tensor import (
    compute_electromagnetic_energy_momentum,
    compute_energy_momentum_tensor,
    compute_particle_energy_momentum,
    electromagnetic_moments,
)
from PyPIC3D.relativity.interpolate_metric import interpolate_metric
from PyPIC3D.utilities.grids import build_tiled_yee_grids, build_yee_grid
from PyPIC3D.utilities.parameters import (
    build_dynamic_parameters,
    build_static_parameters,
    dynamic_parameters_for_output,
    static_parameters_for_output,
)


# Plasma, perturbation, and numerical controls. Change these to define a run.
NX = 128
PARTICLES_PER_CELL = 64                    # Per electron beam; ions have twice as many.
NUMBER_DENSITY = 1.0 / (4.0 * math.pi)      # omega_pe = sqrt(4 pi n_e) = 1.
BEAM_SPEED = 0.5                          # Local orthonormal bulk speed, in units of c.
ELECTRON_TEMPERATURE = 0.0025
ION_TEMPERATURE = 0.0025
MASS_RATIO = 1836.0
WAVE_NUMBER = 1.0
MODE_NUMBER = 1
PERTURBATION = 0.01                       # x_e = q - epsilon sin(k q)/k.
GRAVITY_G = 8.16e-8
SEED = 42
CFL = 0.2
END_TIME = 50.0
GUARD_CELLS = 3
PARTICLE_BATCH_SIZE = 256
KO_DISSIPATION = 0.002
MAX_NEWTON_ITERATIONS = 20
VOLUME_TOLERANCE = 1.0e-12


def build_parameters(nx, length, end_time, cfl, gravity_G, output_dir):
    """One periodic tile with one resolved spatial axis and 3D momenta."""
    dx = length / nx
    nt = max(1, math.ceil(end_time / (cfl * dx)))
    dt = end_time / nt
    config = dict(
        Nx=nx, Ny=1, Nz=1, x_wind=length, y_wind=1.0, z_wind=1.0,
        x_min=0.0, y_min=0.0, z_min=0.0, dx=dx, dy=1.0, dz=1.0, dt=dt,
        C=1.0, eps=1.0, mu=1.0, kb=1.0,
    )
    static = build_static_parameters(dict(
        **config, name="full_relativistic_two_stream", output_dir=str(output_dir),
        Nt=nt, solver="dynamic_metric", metric="numerical", relativistic=True,
        particle_pusher="hybrid_boris_geodesic", current_deposition="esirkepov",
        current_filter="none", shape_factor=2, guard_cells=GUARD_CELLS,
        tile_shape=(nx, 1, 1), particle_batch_size=PARTICLE_BATCH_SIZE,
        boundary_conditions=(0, 0, 0), particle_boundary_conditions=(0, 0, 0),
    ))
    center, vertex = build_yee_grid(SimpleNamespace(**config))
    tiled_center, tiled_vertex = build_tiled_yee_grids(
        static, SimpleNamespace(**config, grids=SimpleNamespace(center=center, vertex=vertex)),
    )
    config["grids"] = dict(center=center, vertex=vertex, tiled_center_grid=tiled_center,
                           tiled_vertex_grid=tiled_vertex)
    dynamic = build_dynamic_parameters(config)
    bssn = build_bssn_parameters(
        static, dynamic, gravity_G=gravity_G,
        evolution_parameters=dict(gauge=0, zero_shift=1, nu=KO_DISSIPATION,
                                  xl_bc=0, xr_bc=0, yl_bc=0, yr_bc=0, zl_bc=0, zr_bc=0),
    )
    return static, dynamic._replace(bssn_parameters=bssn)


def load_plasma(nx, particles_per_cell, length, wave_number, perturbation,
                number_density, beam_speed, electron_temperature, ion_temperature,
                mass_ratio, seed):
    """Load quiet positions and mirrored local orthonormal warm momenta.

    Matching electron positions with opposite momenta cancel gravitational
    momentum exactly. Ions have thermal +/- pairs at the same unperturbed
    positions. Every particle moves during the subsequent evolution.
    """
    count = nx * particles_per_cell
    q = (jnp.arange(count, dtype=jnp.float64) + 0.5) * length / count
    electron_x = jnp.mod(q - perturbation * jnp.sin(wave_number * q) / wave_number, length)
    electron_key, sign_key, ion_key = jax.random.split(jax.random.PRNGKey(seed), 3)
    thermal = jnp.stack(maxwell_juttner_momentum(
        count, 1.0, electron_temperature, electron_temperature, electron_temperature,
        1.0, electron_key,
    ), axis=-1)
    gamma_thermal = jnp.sqrt(1.0 + jnp.sum(thermal**2, axis=-1))

    # Equal-time drifting Juttner sampling needs the momentum Jacobian
    # 1 + beta*u_x/Gamma before the Lorentz boost. Bias the sign of |u_x|
    # in an isotropic draw; this implements that factor without rejection.
    abs_ux = jnp.abs(thermal[:, 0])
    positive_probability = 0.5 * (1.0 + beam_speed * abs_ux / gamma_thermal)
    sign = jnp.where(jax.random.uniform(sign_key, (count,), dtype=jnp.float64)
                     < positive_probability, 1.0, -1.0)
    thermal = thermal.at[:, 0].set(sign * abs_ux)
    beam = thermal.at[:, 0].set(
        (thermal[:, 0] + beam_speed * gamma_thermal) / math.sqrt(1.0 - beam_speed**2),
    )
    ion = jnp.stack(maxwell_juttner_momentum(
        count, mass_ratio, ion_temperature, ion_temperature, ion_temperature,
        1.0, ion_key,
    ), axis=-1)
    ion_pairs = jnp.stack((ion, -ion), axis=1).reshape(2 * count, 3)

    # A common species capacity keeps the existing tiled particle layout.
    # Electron slots above count are inactive, not extra physical particles.
    x = jnp.zeros((1, 1, 1, 3, 2 * count, 3), dtype=jnp.float64)
    u = jnp.zeros_like(x)
    active = jnp.zeros(x.shape[:-1], dtype=bool)
    for species_index, direction in ((0, 1.0), (1, -1.0)):
        x = x.at[0, 0, 0, species_index, :count, 0].set(electron_x)
        u = u.at[0, 0, 0, species_index, :count].set(direction * beam)
        active = active.at[0, 0, 0, species_index, :count].set(True)
    x = x.at[0, 0, 0, 2, :, 0].set(jnp.repeat(q, 2))
    u = u.at[0, 0, 0, 2].set(ion_pairs)
    active = active.at[0, 0, 0, 2].set(True)
    weight = number_density * length / (2 * count)
    species = SpeciesConfig(
        jnp.array([-1.0, -1.0, 1.0]), jnp.array([1.0, 1.0, mass_ratio]),
        jnp.full((3,), weight), jnp.ones((3, 3), dtype=bool),
    )
    return TiledParticles(x, u, active), species


def solve_initial_metric(bssn, particle_energy, electromagnetic_energy, omega_pe,
                         *, constraint_tolerance=None,
                         max_newton_iterations=MAX_NEWTON_ITERATIONS):
    """Solve discrete Hamiltonian plus mean(W^-3)=1 on the expanding branch.

    Local orthonormal particle momenta are fixed initial-data inputs. Their
    Lorentz factors therefore remain fixed, so production deposition gives
    rho_particle(W)=W^3*rho_particle(1). The native densitized electric field
    is also fixed, giving rho_EM(W)=W^4*rho_EM(1). The final production-source
    check below verifies these identities after coordinate momentum conversion.
    """
    flat = bssn.state
    gravity_G = float(bssn.gravity_G)
    source_scale = float(16.0 * jnp.pi * gravity_G * jnp.mean(particle_energy + electromagnetic_energy))
    tolerance = (max(1.0e-11 * omega_pe**2, 1.0e-8 * source_scale)
                 if constraint_tolerance is None else float(constraint_tolerance))
    if gravity_G == 0.0:
        return flat, [0.0], [0.0], tolerance

    nx = flat.conformal_factor.size

    def state_and_density(unknown):
        W = jnp.exp(unknown[:-1]).reshape(flat.conformal_factor.shape)
        # Scale K^2 by the matter source to condition the last Newton column.
        K = -jnp.sqrt(source_scale * unknown[-1])
        state = flat._replace(conformal_factor=W, trace_K=jnp.full_like(W, K))
        rho = particle_energy * W**3 + electromagnetic_energy * W**4
        return state, rho

    def residual(unknown):
        state, rho = state_and_density(unknown)
        hamiltonian = compute_hamiltonian_constraint_with_matter(
            state, bssn.evolution_parameters, gravity_G * rho,
        )
        volume = jnp.mean(state.conformal_factor**-3) - 1.0
        return jnp.concatenate((hamiltonian.ravel() / source_scale, jnp.asarray([volume])))

    evaluate = jax.jit(residual)
    jacobian = jax.jit(jax.jacfwd(residual))
    unknown = jnp.concatenate((jnp.zeros(nx, dtype=jnp.float64), jnp.asarray([1.5])))
    residual_history, volume_history = [], []
    for iteration in range(max_newton_iterations + 1):
        value = evaluate(unknown)
        h_norm = float(jnp.max(jnp.abs(value[:-1]))) * source_scale
        volume_error = float(jnp.abs(value[-1]))
        residual_history.append(h_norm)
        volume_history.append(volume_error)
        if h_norm <= tolerance and volume_error <= VOLUME_TOLERANCE:
            return state_and_density(unknown)[0], residual_history, volume_history, tolerance
        if iteration == max_newton_iterations:
            break

        correction = jnp.linalg.solve(jacobian(unknown), -value)
        merit = max(h_norm / tolerance, volume_error / VOLUME_TOLERANCE)
        for backtrack in range(20):
            trial = unknown + (0.5**backtrack) * correction
            if float(trial[-1]) <= 0.0 or not bool(jnp.all(jnp.isfinite(trial))):
                continue
            trial_value = evaluate(trial)
            trial_merit = max(
                float(jnp.max(jnp.abs(trial_value[:-1]))) * source_scale / tolerance,
                float(jnp.abs(trial_value[-1])) / VOLUME_TOLERANCE,
            )
            if math.isfinite(trial_merit) and trial_merit < merit:
                unknown = trial
                break
        else:
            raise RuntimeError(f"Initial metric Newton step failed: H_linf={h_norm:.3e}, "
                               f"volume_error={volume_error:.3e}")
    raise RuntimeError(f"Initial metric did not converge: H_linf={h_norm:.3e}, "
                       f"volume_error={volume_error:.3e}")


def covariant_particle_momenta(particles, metric, static, dynamic):
    """Map fixed orthonormal u_hat to covariant u using the pusher's metric."""
    sampled = interpolate_metric(
        jax.tree.map(lambda value: value[0, 0, 0], metric.center),
        particles.x.reshape(-1, 3),
        tuple(axis[0, 0, 0] for axis in dynamic.grids.tiled_center_grid),
        static.metric, (True, False, False), (static.guard_cells,) * 3,
        derivatives=False,
    )
    scale = jnp.sqrt(sampled.gamma[:, 0, 0]).reshape(particles.active.shape)
    u = jnp.where(particles.active[..., None], particles.u * scale[..., None], 0.0)
    return particles._replace(u=u)


def build_initial_data(*, nx=NX, particles_per_cell=PARTICLES_PER_CELL,
                       end_time=END_TIME, cfl=CFL, wave_number=WAVE_NUMBER,
                       mode_number=MODE_NUMBER, perturbation=PERTURBATION,
                       number_density=NUMBER_DENSITY, beam_speed=BEAM_SPEED,
                       electron_temperature=ELECTRON_TEMPERATURE,
                       ion_temperature=ION_TEMPERATURE, mass_ratio=MASS_RATIO,
                       gravity_G=GRAVITY_G, seed=SEED, output_dir=".",
                       constraint_tolerance=None,
                       max_newton_iterations=MAX_NEWTON_ITERATIONS):
    """Return particles, species, physical fields, static, dynamic, metadata.

    Everything is at t=0. Do not evolve this state until the production
    initialize_dynamical_relativity routine has seeded its leapfrog histories.
    """
    jax.config.update("jax_enable_x64", True)
    if nx < 8 or particles_per_cell < 1 or mode_number < 1 or int(mode_number) != mode_number:
        raise ValueError("Use nx>=8, particles_per_cell>=1, and a positive integer mode_number")
    if 2 * mode_number >= nx:
        raise ValueError("The seeded wave must be below the grid Nyquist mode")
    if wave_number <= 0 or number_density <= 0 or gravity_G < 0 or end_time <= 0 or cfl <= 0:
        raise ValueError("The wave number, density, duration, and CFL must be positive; gravity_G>=0")
    if abs(perturbation) >= 1 or abs(beam_speed) >= 1 or mass_ratio <= 0:
        raise ValueError("The displacement map must be monotone, |beam_speed|<1, and mass_ratio>0")
    if electron_temperature < 0 or ion_temperature < 0:
        raise ValueError("Physical electron and ion temperatures must be nonnegative")

    length = 2.0 * math.pi * mode_number / wave_number
    omega_pe = math.sqrt(4.0 * math.pi * number_density)
    static, dynamic = build_parameters(nx, length, end_time, cfl, gravity_G, output_dir)
    particles, species = load_plasma(
        nx, particles_per_cell, length, wave_number, perturbation, number_density,
        beam_speed, electron_temperature, ion_temperature, mass_ratio, seed,
    )
    flat = dynamic.bssn_parameters.state
    flat_metric = build_bssn_metric(flat, static, dynamic)
    zero = tuple(jnp.zeros_like(component.lapse) for component in flat_metric.D)

    # Solve the native backward-difference Gauss law for densitized D_x.
    charge = compute_rho(particles, species, zero[0], static, dynamic)
    owned_charge = tiles_to_bssn(charge, static)
    if abs(float(jnp.mean(owned_charge))) > 1.0e-12 * number_density:
        raise RuntimeError("Periodic Gauss solve requires globally neutral particle loading")
    electric_x = 4.0 * jnp.pi * dynamic.dx * jnp.cumsum(owned_charge[:, 0, 0])
    electric_x = electric_x - jnp.mean(electric_x)
    displacement = (bssn_to_tiles(electric_x[:, None, None], static), zero[1], zero[2])
    magnetic = zero
    moments = electromagnetic_moments(displacement, magnetic, static)
    particle_energy = compute_particle_energy_momentum(
        particles, species, flat_metric, static, dynamic,
    )[0]
    electromagnetic_energy = compute_electromagnetic_energy_momentum(moments, flat)[0]
    state, solve_residuals, volume_residuals, tolerance = solve_initial_metric(
        dynamic.bssn_parameters, particle_energy, electromagnetic_energy, omega_pe,
        constraint_tolerance=constraint_tolerance, max_newton_iterations=max_newton_iterations,
    )
    dynamic = dynamic._replace(bssn_parameters=dynamic.bssn_parameters._replace(state=state))
    metric = build_bssn_metric(state, static, dynamic)
    particles = covariant_particle_momenta(particles, metric, static, dynamic)

    # Re-evaluate the actual production sources after coordinate conversion.
    rho, momentum, _ = compute_energy_momentum_tensor(
        particles, species, moments, state, static, dynamic,
    )
    predicted_rho = particle_energy * state.conformal_factor**3 + electromagnetic_energy * state.conformal_factor**4
    source_error = float(jnp.max(jnp.abs(rho - predicted_rho)))
    if not bool(jnp.all(jnp.isfinite(rho))) or source_error > 1.0e-12 * float(jnp.max(rho)):
        raise RuntimeError("Final production energy density disagrees with the initial metric source")
    violations = compute_all_constraints_with_matter(
        state, dynamic.bssn_parameters.evolution_parameters, gravity_G * rho, gravity_G * momentum,
    )
    constraints = {name: float(value) for name, value in compute_constraint_norms(violations).items()}
    constraints["gauss_linf"] = float(jnp.max(jnp.abs(tiles_to_bssn(
        divergence(displacement, dynamic) - 4.0 * jnp.pi * charge, static,
    ))))
    constraints["div_b_linf"] = float(jnp.max(jnp.abs(tiles_to_bssn(
        divergence(magnetic, dynamic, forward=True), static,
    ))))
    volume_error = float(jnp.abs(jnp.mean(state.conformal_factor**-3) - 1.0))
    if (not all(math.isfinite(value) for value in constraints.values())
            or constraints["hamiltonian_linf"] > tolerance
            or constraints["momentum_linf"] > tolerance
            or constraints["gauss_linf"] > 1.0e-11 * omega_pe**2
            or volume_error > VOLUME_TOLERANCE):
        raise RuntimeError(f"Initial data failed production constraint verification: {constraints}")

    physical_D = tuple(value / component.sqrt_gamma for value, component in zip(displacement, metric.D))
    physical_B = tuple(value / component.sqrt_gamma for value, component in zip(magnetic, metric.B))
    fields = (physical_D, physical_B, zero, charge, zero[0], (zero, zero), metric, None, jnp.asarray(False))
    metadata = dict(
        nx=nx, particles_per_cell=particles_per_cell, number_density=number_density,
        omega_pe=omega_pe, wave_number=wave_number, mode_number=mode_number,
        perturbation=perturbation, beam_speed=beam_speed,
        electron_temperature=electron_temperature, ion_temperature=ion_temperature,
        mass_ratio=mass_ratio, gravity_G=gravity_G, seed=seed, cfl=cfl, end_time=end_time,
        length_x=length, length_y=1.0, length_z=1.0, dt=float(dynamic.dt), nt=static.Nt,
        species_names=["electron_plus", "electron_minus", "ion"],
        species_counts=[nx * particles_per_cell, nx * particles_per_cell, 2 * nx * particles_per_cell],
        units="Gaussian, c=m_e=|q_e|=k_B=1", initial_branch="expanding",
        initial_trace_K=float(state.trace_K[0, 0, 0]),
        gauge="harmonic", shift="zero", boundaries="periodic on every face",
        particle_momentum="common-time covariant u_i derived from local orthonormal warm momenta",
        field_time="common-time physical fields; no leapfrog histories",
        density_convention="mean particle number per initial coordinate volume; proper volume normalized equal",
        temperature_convention="physical k_B*T; isotropic rest-frame Maxwell-Juttner",
        perturbation_convention="electron displacement x=q-epsilon*sin(k*q)/k; uniform ions",
        source_scale=float(16.0 * jnp.pi * gravity_G * jnp.mean(rho)),
        constraint_tolerance=tolerance, source_verification_linf=source_error,
        volume_error=volume_error, initial_constraints=constraints,
        solve_residuals=solve_residuals, solve_volume_residuals=volume_residuals,
        newton_iterations=len(solve_residuals) - 1,
    )
    return particles, species, fields, static, dynamic, metadata


def export_native_initial_data(output_dir, particles, species, fields, static, dynamic, metadata):
    """Write native particle/field files and a TOML loading the solved BSSN state."""
    directory = Path(output_dir).resolve()
    array_directory = directory / "initial_data"
    array_directory.mkdir()
    config_path = directory / "two_stream.toml"
    config = toml.load(Path(__file__).with_name(config_path.name))

    simulation = config["simulation_parameters"]
    simulation.update(
        name=static.name, output_dir=str(Path(static.output_dir).resolve()),
        solver=static.solver, metric=static.metric, Nt=static.Nt,
        t_wind=float(dynamic.dt) * static.Nt, dt=float(dynamic.dt), cfl=metadata["cfl"],
        particle_tile_capacity_factor=static.particle_tile_capacity_factor,
        particle_batch_size=static.particle_batch_size, shape_factor=static.shape_factor,
        guard_cells=static.guard_cells, relativistic=static.relativistic,
        particle_pusher=static.particle_pusher, current_calculation="esirkepov",
        filter_j=static.current_filter, verbose=static.verbose,
    )
    for axis, width in zip("xyz", static.tile_shape):
        simulation["N" + axis] = int(getattr(dynamic, "N" + axis))
        # Preserve the builder's exact origin, not roundoff in a sampled node.
        simulation[axis + "_min"] = 0.0
        simulation[axis + "_max"] = float(getattr(dynamic, axis + "_wind"))
        simulation[axis + "_wind"] = float(getattr(dynamic, axis + "_wind"))
        simulation["particle_tile_n" + axis] = width
    for name in ("C", "eps", "mu", "kb"):
        simulation[name] = float(getattr(dynamic, name))

    # Strip inactive padding, retaining each species' exact quiet-start order.
    for index, name in enumerate(metadata["species_names"]):
        active = particles.active[:, :, :, index, :]
        position = particles.x[:, :, :, index, :, :][active]
        momentum = particles.u[:, :, :, index, :, :][active]
        block = config[f"particle{index + 1}"]
        block.update(name=name, N_particles=position.shape[0],
                     charge=float(species.charge[index]), mass=float(species.mass[index]),
                     weight=float(species.weight[index]),
                     temperature=metadata["electron_temperature" if index < 2 else "ion_temperature"])
        for component, axis in enumerate("xyz"):
            block["update_" + axis] = bool(species.update_x[index, component])
            for key, values in (("initial_" + axis, position[:, component]),
                                ("initial_v" + axis, momentum[:, component])):
                path = array_directory / f"{name}_{key}.npy"
                jnp.save(path, values)
                block[key] = str(path)

    # Native field files contain owned physical Yee samples, without guards.
    for vector_index, name in enumerate(("D", "B")):
        for component, axis in enumerate("xyz"):
            path = array_directory / f"physical_{name}_{axis}.npy"
            jnp.save(path, tiles_to_bssn(fields[vector_index][component], static))
            config[f"field{3 * vector_index + component + 1}"]["path"] = str(path)

    bssn = dynamic.bssn_parameters
    metric = config["metric"]
    for name in ("conformal_metric", "conformal_factor", "trace_K", "lapse"):
        path = array_directory / f"{name}.npy"
        jnp.save(path, getattr(bssn.state, name))
        metric[name] = str(path)
    metric.update(traceless_K=False, conformal_connection=False, shift=False,
                  gravity_G=float(bssn.gravity_G), time=float(bssn.time),
                  gauge=("harmonic", "1+log")[int(bssn.evolution_parameters.gauge)],
                  zero_shift=bool(bssn.evolution_parameters.zero_shift))
    for name in ("eta", "kappa", "nu", "g", "mad_q"):
        metric[name] = float(getattr(bssn.evolution_parameters, name))
    for face in ("xl_bc", "xr_bc", "yl_bc", "yr_bc", "zl_bc", "zr_bc"):
        metric[face] = ("periodic", "sommerfeld")[int(getattr(bssn.evolution_parameters, face))]
    config["diagnostics"].update(
        number_density=metadata["number_density"], omega_pe=metadata["omega_pe"],
        wave_number=metadata["wave_number"],
        initial_metadata=str(directory / "initial_metadata.json"),
    )
    with config_path.open("w") as stream:
        toml.dump(config, stream)
    return config_path


def save_initial_data(output_dir, particles, species, fields, static, dynamic, metadata):
    """Save the common-time archive, metadata, and native TOML/.npy inputs."""
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    arrays = dict(particle_x=particles.x, particle_u=particles.u, particle_active=particles.active,
                  charge_density=fields[3])
    arrays.update({f"species_{name}": value for name, value in species._asdict().items()})
    arrays.update({f"bssn_{name}": value for name, value in dynamic.bssn_parameters.state._asdict().items()})
    for component, axis in enumerate("xyz"):
        arrays[f"physical_D_{axis}"] = fields[0][component]
        arrays[f"physical_B_{axis}"] = fields[1][component]
    for axis, grid in zip("xyz", dynamic.grids.center):
        arrays[f"center_grid_{axis}"] = grid
    array_path = directory / "initial_data.npz"
    metadata_path = directory / "initial_metadata.json"
    if any(path.exists() for path in (array_path, metadata_path,
                                      directory / "two_stream.toml",
                                      directory / "initial_data")):
        raise FileExistsError(f"Initial-data artifacts already exist in {directory}")
    jnp.savez(array_path, **arrays)
    record = dict(metadata, static_parameters=static_parameters_for_output(static),
                  dynamic_parameters=dynamic_parameters_for_output(dynamic))
    metadata_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    export_native_initial_data(directory, particles, species, fields, static, dynamic, metadata)
    return array_path, metadata_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--nx", type=int, default=NX)
    parser.add_argument("--particles-per-cell", type=int, default=PARTICLES_PER_CELL)
    parser.add_argument("--end-time", type=float, default=END_TIME)
    parser.add_argument("--gravity-G", type=float, default=GRAVITY_G)
    args = parser.parse_args()
    demo_directory = Path(__file__).resolve().parent
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    bundle_directory = args.output_dir or demo_directory / "outputs" / f"initial_{stamp}"
    run_directory = demo_directory / "runs" / stamp
    bundle_directory.mkdir(parents=True, exist_ok=False)
    result = build_initial_data(nx=args.nx, particles_per_cell=args.particles_per_cell,
                                end_time=args.end_time, gravity_G=args.gravity_G,
                                output_dir=run_directory)
    paths = save_initial_data(bundle_directory, *result)

    # Publish only after every array and its configuration have been written.
    # Earlier bundles keep their own configuration and native output path.
    launch_config = demo_directory / "two_stream.toml"
    temporary_config = demo_directory / f".two_stream.{stamp}.tmp"
    temporary_config.write_text((paths[0].parent / "two_stream.toml").read_text())
    temporary_config.replace(launch_config)
    print(f"Saved initial data: {paths[0]}")
    print(f"Native configuration: {launch_config}")
    print(f"Simulation output: {run_directory}")
    print(f"Expanding K = {result[-1]['initial_trace_K']:.8e}")
    for name, value in result[-1]["initial_constraints"].items():
        print(f"{name}: {value:.6e}")


if __name__ == "__main__":
    main()
