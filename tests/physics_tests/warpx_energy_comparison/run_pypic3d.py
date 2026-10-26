"""Standalone energy audit using native PyPIC3D kernels and common particles.

No evolution kernel or production diagnostic is changed. Diagnostic half pushes
operate on copies; electrostatic solve orchestration exposes its native residual.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from contextlib import redirect_stdout, redirect_stderr
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np  # NPZ interchange only; numerical work below uses JAX.

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from PyPIC3D.initialization import initialize_fields
from PyPIC3D.utilities.parameters import build_static_parameters, build_dynamic_parameters, resolve_field_gather
from PyPIC3D.utilities.grids import build_collocated_grid, build_yee_grid, build_tiled_yee_grids
from PyPIC3D.particles.particle_class import TiledParticles, SpeciesConfig
from PyPIC3D.particles.particle_tile_communication import (
    shard_tiled_particles, update_tiled_particle_positions, refresh_tiled_particle_tiles,
)
from PyPIC3D.pusher.particle_push import particle_push, seed_leapfrog_velocity
from PyPIC3D.solvers.yee.time_loop import time_loop_electrodynamic
from PyPIC3D.solvers.yee.first_order_yee import yee_curl_e_to_b
from PyPIC3D.solvers.electrostatic.electrostatic_yee import (
    solve_poisson_with_tiled_local_schwarz, electrostatic_gradient,
    _apply_tiled_phi_constant_boundaries, _poisson_residual,
)
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.utilities.field_helpers import add_external_fields, yee_push_fields
from PyPIC3D.utilities.filters import digital_filter
from PyPIC3D.diagnostics.diagnostic_quantities import compute_energy

C = 299792458.0
Q = 1.602176634e-19
EPS = 8.8541878188e-12
MU = 1.2566370612685e-6
ME = 9.1093837139e-31
MP = 1.67262192595e-27
N0 = 1.0e30
WPE = Q * math.sqrt(N0 / (ME * EPS))
LENGTH = 10.0 * C / WPE
RHO_SCALE = N0 * Q
RESIDUAL_SCALE = RHO_SCALE / EPS


def parameters(dim, solver, dt_wpe, relative_tol, shape, gather):
    cells = (1, 1, 8) if dim == 1 else (8, 1, 8)
    widths = (1.0, 1.0, LENGTH) if dim == 1 else (LENGTH, 1.0, LENGTH)
    minima = (-0.5, -0.5, 0.0) if dim == 1 else (0.0, -0.5, 0.0)
    spacing = tuple(width / n for width, n in zip(widths, cells))
    s = build_static_parameters(dict(
        name="WarpX energy comparison", solver="electrostatic" if solver == "es" else "electrodynamic_yee",
        electrostatic=solver == "es", relativistic=True, particle_pusher="boris",
        current_deposition="esirkepov" if solver == "esirkepov" else "direct",
        current_filter="none", shape_factor=shape, field_gather=gather, guard_cells=2, tile_shape=cells,
        Nx=cells[0], Ny=cells[1], Nz=cells[2], boundary_conditions=(0, 0, 0),
        particle_boundary_conditions=(0, 0, 0), particle_batch_size=256,
        electrostatic_schwarz_tol=relative_tol * RESIDUAL_SCALE,
        electrostatic_local_cg_tol=relative_tol * RESIDUAL_SCALE,
        electrostatic_schwarz_max_iterations=500, electrostatic_local_cg_max_iterations=500,
    ))
    data = dict(dt=dt_wpe / WPE, dx=spacing[0], dy=spacing[1], dz=spacing[2],
                Nx=cells[0], Ny=cells[1], Nz=cells[2], x_wind=widths[0],
                y_wind=widths[1], z_wind=widths[2], x_min=minima[0],
                y_min=minima[1], z_min=minima[2], C=C, eps=EPS, mu=MU,
                kb=1.380649e-23, alpha=1.0)
    grid = build_collocated_grid if s.electrostatic and resolve_field_gather(s.solver, s.current_deposition, s.field_gather) == "uniform" else build_yee_grid
    center, vertex = grid(SimpleNamespace(**data))
    data["grids"] = dict(center=center, vertex=vertex, tiled_center_grid=(), tiled_vertex_grid=())
    d = build_dynamic_parameters(data)
    tiled_center, tiled_vertex = build_tiled_yee_grids(s, d)
    data["grids"].update(tiled_center_grid=tiled_center, tiled_vertex_grid=tiled_vertex)
    d = build_dynamic_parameters(data)
    return s, d


def initial_particles(root, dim, s):
    source = root / "initial" / f"{dim}d.npz"
    with np.load(source) as data:
        positions, velocities, weights = [], [], []
        for name in ("electrons", "protons"):
            x = jnp.stack([jnp.asarray(data[f"{name}_{axis}"]) for axis in "xyz"], axis=-1)
            u = jnp.stack([jnp.asarray(data[f"{name}_u{axis}"]) for axis in "xyz"], axis=-1)
            gamma = jnp.sqrt(1.0 + jnp.sum(u * u, axis=-1) / C**2)
            positions.append(x)
            velocities.append(u / gamma[:, None])
            weight = jnp.asarray(data[f"{name}_w"])
            if not bool(jnp.all(weight == weight[0])):
                raise ValueError("Native species configuration requires constant particle weight")
            weights.append(float(weight[0]))
    x, v = jnp.stack(positions), jnp.stack(velocities)
    particles = TiledParticles(x=x[None, None, None], u=v[None, None, None],
                               active=jnp.ones((1, 1, 1) + x.shape[:-1], dtype=bool))
    species = SpeciesConfig(charge=jnp.asarray([-Q, Q]), mass=jnp.asarray([ME, MP]),
                            weight=jnp.asarray(weights), update_x=jnp.ones((2, 3), dtype=bool))
    return shard_tiled_particles(particles, s), species, source


def make_kernels(s, species):
    g = s.guard_cells
    interior = (..., slice(g, -g), slice(g, -g), slice(g, -g))

    def electrostatic_fields(particles, fields, d):
        E, B, J, rho, phi, external, pml, overflow = fields
        rho = compute_rho(particles, species, rho, s, d)
        phi, info = solve_poisson_with_tiled_local_schwarz(
            rho, phi, s, d, schwarz_tol=s.electrostatic_schwarz_tol,
            schwarz_max_iterations=s.electrostatic_schwarz_max_iterations,
            local_cg_tol=s.electrostatic_local_cg_tol,
            local_cg_max_iterations=s.electrostatic_local_cg_max_iterations,
            return_diagnostics=True)
        phi = digital_filter(phi, d.alpha, num_guard_cells=g)
        phi = _apply_tiled_phi_constant_boundaries(phi, s, g)
        E = electrostatic_gradient(phi, s, d, g)
        return (E, B, J, rho, phi, external, pml, overflow), info

    @jax.jit
    def initial_fields(particles, fields, d):
        if s.electrostatic:
            return electrostatic_fields(particles, fields, d)
        rho = compute_rho(particles, species, fields[3], s, d)
        return fields[:3] + (rho,) + fields[4:], (jnp.zeros((1, 1, 1)), jnp.asarray(0.0), jnp.asarray(0))

    @jax.jit
    def seed(particles, fields, d):
        gather = add_external_fields(fields[0], fields[1], fields[5]) if s.electrostatic else yee_push_fields(fields[0], fields[1], fields[5], s, d)
        return seed_leapfrog_velocity(particles, species, *gather, s, d)

    @jax.jit
    def step(particles, fields, d):
        if not s.electrostatic:
            particles, fields = time_loop_electrodynamic(particles, species, fields, s, d)
            return particles, fields, (jnp.zeros((1, 1, 1)), jnp.asarray(0.0), jnp.asarray(0))
        gather = add_external_fields(fields[0], fields[1], fields[5])
        particles = particle_push(particles, species, *gather, s, d)
        particles = update_tiled_particle_positions(particles, species, d.dt)
        particles, overflow = refresh_tiled_particle_tiles(particles, s, d)
        fields = fields[:-1] + (fields[-1] | overflow,)
        fields, info = electrostatic_fields(particles, fields, d)
        return particles, fields, info

    def kinetic(particles):
        v2 = jnp.sum(particles.u * particles.u, axis=-1)
        gamma = jax.lax.rsqrt(1.0 - v2 / C**2)
        mass = (species.mass * species.weight)[None, None, None, :, None]
        return jnp.sum(jnp.where(particles.active, mass * gamma * gamma * v2 / (gamma + 1.0), 0.0))

    @jax.jit
    def sample(particles, fields, info, d):
        E, B, J, rho, phi, external, _, overflow = fields
        electric, magnetic, kinetic_legacy = compute_energy(particles, E, B, s, d, species)
        gather = add_external_fields(E, B, external) if s.electrostatic else yee_push_fields(E, B, external, s, d)
        common = particle_push(particles, species, *gather, s, d._replace(dt=d.dt / 2.0))
        kinetic_raw, kinetic_sync = kinetic(particles), kinetic(common)
        volume = d.dx * d.dy * d.dz
        correction = jnp.asarray(0.0)
        if not s.electrostatic:
            curl = yee_curl_e_to_b(E, s, d)
            correction = volume * d.dt**2 / (8.0 * d.mu) * sum(jnp.sum(c * c) for c in curl)
        rho_phi = 0.5 * volume * jnp.sum(rho[interior] * phi[interior])
        gradient2 = jnp.zeros_like(phi[interior])
        for axis, spacing in enumerate((d.dx, d.dy, d.dz)):
            ahead = [slice(None)] * 3 + [slice(g, -g)] * 3
            ahead[axis + 3] = slice(g + 1, -g + 1)
            gradient2 = gradient2 + ((phi[tuple(ahead)] - phi[interior]) / spacing)**2
        gradient_energy = 0.5 * d.eps * volume * jnp.sum(gradient2)
        residual = jnp.max(jnp.abs(_poisson_residual(rho, phi, d, g))) / RESIDUAL_SCALE if s.electrostatic else jnp.asarray(0.0)
        count = jnp.sum(particles.active)
        finite = jnp.all(jnp.isfinite(particles.x)) & jnp.all(jnp.isfinite(particles.u))
        for value in E + B + (rho, phi):
            finite = finite & jnp.all(jnp.isfinite(value))
        return dict(electric=electric, magnetic=magnetic,
            kinetic_raw=kinetic_legacy, kinetic_raw_stable=kinetic_raw, kinetic_sync=kinetic_sync,
            total_raw=electric + magnetic + kinetic_legacy,
            total_raw_stable=electric + magnetic + kinetic_raw,
            total_sync=electric + magnetic + kinetic_sync,
            total_modified=electric + magnetic + kinetic_sync - correction,
            curl_correction=correction, electrostatic_rho_phi=rho_phi,
            electrostatic_forward_gradient=gradient_energy,
            total_poisson=rho_phi + kinetic_sync,
            poisson_residual_normalized=residual,
            mean_rho_normalized=jnp.abs(jnp.mean(rho[interior])) / RHO_SCALE,
            poisson_local_residual_normalized=jnp.max(info[0]) / RESIDUAL_SCALE,
            poisson_schwarz_residual_normalized=info[1] / RESIDUAL_SCALE,
            poisson_iterations=info[2], poisson_maxiter_hit=info[2] >= s.electrostatic_schwarz_max_iterations,
            active_count=count, finite=finite, overflow=overflow,
            max_speed_over_c=jnp.max(jnp.linalg.norm(particles.u, axis=-1)) / C)

    return initial_fields, seed, step, sample


def run_case(root, dim, solver, dt_wpe, relative_tol, end_time, shape, gather):
    suffix = "_tight" if relative_tol < 1e-12 else ""
    case = f"{dim}d_{solver}_p{shape}_{gather}_dt{dt_wpe:g}{suffix}"
    output = root / "pypic3d" / case
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    status_file = output / "status.json"
    status_file.write_text(json.dumps(dict(status="running", case=case, started_unix=started), indent=2))
    print(f"START {case}", flush=True)
    s, d = parameters(dim, solver, dt_wpe, relative_tol, shape, gather)
    particles, species, source = initial_particles(root, dim, s)
    E, B, J, phi, rho = initialize_fields(s, d)
    zero_fields = (E, B)
    fields = (E, B, J, rho, phi, zero_fields, None, jnp.asarray(False))
    initial_fields, seed, step, sample = make_kernels(s, species)
    fields, info = initial_fields(particles, fields, d)
    g = s.guard_cells
    interior = (..., slice(g, -g), slice(g, -g), slice(g, -g))
    initial_rho_normalized = float(jnp.max(jnp.abs(fields[3][interior])) / RHO_SCALE)
    initial_electric = float(sum(jnp.sum(e[interior]**2) for e in fields[0]))
    if initial_rho_normalized > 1e-13:
        raise ValueError(f"Common initial plasma is not neutral: {initial_rho_normalized}")
    if initial_electric != 0.0:
        raise ValueError(f"Common initial electric field is not zero: {initial_electric}")
    particles = seed(particles, fields, d)
    initial_count = int(jnp.sum(particles.active))
    steps = int(round(end_time / dt_wpe))
    manifest = dict(case=case, dimension=dim, solver=solver, dt_wpe=dt_wpe,
        dt=float(d.dt), end_time_wpe=end_time, steps=steps,
        cells=list(s.tile_shape), tile_shape=list(s.tile_shape), guard_cells=s.guard_cells,
        shape_factor=shape, field_gather_requested=gather, field_gather=resolve_field_gather(s.solver, s.current_deposition, s.field_gather),
        current_filter="none", alpha=1.0, relativistic=True,
        particle_pusher="boris", current_deposition=s.current_deposition, initial_particles=str(source),
        initial_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        initial_count=initial_count, initial_rho_normalized=initial_rho_normalized,
        initial_electric_squared_sum=initial_electric,
        poisson_relative_tolerance=relative_tol, poisson_absolute_tolerance=relative_tol*RESIDUAL_SCALE,
        poisson_iteration_limit=500, initial_electrostatic_solve=True,
        x64=jax.config.x64_enabled, backend=jax.default_backend(), python=sys.executable,
        jax_version=jax.__version__, repo_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        eps_mu_c2=EPS*MU*C*C, constants=dict(C=C,eps=EPS,mu=MU,me=ME,mp=MP,e=Q,n0=N0,omega_pe=WPE,L=LENGTH),
        source_hashes={str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (REPO/"PyPIC3D").rglob("*.py")},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note="Native production kernels; ES solve sequence exposes native diagnostics. Diagnostic copies do not feed trajectories. total_poisson is relevant only for ES.")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    totals = ("total_raw", "total_raw_stable", "total_sync", "total_modified", "total_poisson")
    reference = None
    maximum_errors = {name: 0.0 for name in totals}
    max_residual, max_iterations, maxiter_hits = 0.0, 0, 0
    with (output / "history.csv").open("w", newline="") as handle:
        writer = None
        for n in range(steps + 1):
            values = jax.device_get(sample(particles, fields, info, d))
            row = {"step": n, "time": n*float(d.dt), "wpe_t": n*dt_wpe}
            row.update({name: value.item() for name, value in values.items()})
            if reference is None:
                reference = {name: row[name] for name in totals}
            for name in totals:
                error = (row[name] - reference[name]) / abs(reference[name])
                row[name + "_relative_error"] = error
                maximum_errors[name] = max(maximum_errors[name], abs(error))
            row["legacy_reported_error"] = abs(row["total_raw"] - reference["total_raw"]) / max(reference["total_raw"], 1e-10)
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            max_residual = max(max_residual, row["poisson_residual_normalized"])
            max_iterations = max(max_iterations, row["poisson_iterations"])
            maxiter_hits += int(row["poisson_maxiter_hit"])
            if not row["finite"] or row["overflow"] or row["active_count"] != initial_count:
                handle.flush()
                raise RuntimeError(f"Invalid state at step {n}: finite={row['finite']} overflow={row['overflow']} count={row['active_count']}")
            if n % 100 == 0:
                handle.flush()
                print(f"{case} step={n}/{steps} sync_error={row['total_sync_relative_error']:.6e} residual={row['poisson_residual_normalized']:.3e} iterations={row['poisson_iterations']}", flush=True)
            if n < steps:
                particles, fields, info = step(particles, fields, d)
    final = dict(status="complete", case=case, elapsed_seconds=time.time()-started,
        steps=steps, initial_totals=reference, final_totals={name: row[name] for name in totals},
        final_relative_errors={name: row[name+"_relative_error"] for name in totals},
        maximum_absolute_relative_errors=maximum_errors,
        max_poisson_residual_normalized=max_residual, max_poisson_iterations=max_iterations,
        poisson_maxiter_hits=maxiter_hits, finite=True, overflow=False, active_count=initial_count)
    status_file.write_text(json.dumps(final, indent=2))
    (output / "summary.json").write_text(json.dumps(final, indent=2))
    print(f"DONE {case}: {json.dumps(final)}", flush=True)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dimension", type=int, choices=(1, 2))
    parser.add_argument("--solver", choices=("es", "direct", "esirkepov"))
    parser.add_argument("--dt-wpe", type=float)
    parser.add_argument("--shape", type=int, choices=(1, 2))
    parser.add_argument("--gather", choices=("auto", "uniform", "galerkin"), default="galerkin")
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--end-time", type=float, default=100.0)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.dimension:
        cases = [(args.dimension, args.solver, args.dt_wpe, args.shape)]
    else:
        cases = [(dim, solver, dt, shape)
                 for shape in (1, 2)
                 for dim in (1, 2)
                 for solver in ("es", "esirkepov")
                 for dt in ((0.2, 0.1, 0.05) if shape == 1 else (0.2, 0.1))]
    failed = []
    for dim, solver, dt, shape in cases:
        case = f"{dim}d_{solver}_p{shape}_{args.gather}_dt{dt:g}" + ("_tight" if args.tol < 1e-12 else "")
        output = root / "pypic3d" / case
        output.mkdir(parents=True, exist_ok=True)
        if (output / "status.json").exists():
            old = json.loads((output / "status.json").read_text())
            if old.get("status") == "complete":
                print(f"Preserving completed {case}", flush=True)
                continue
        print(f"Running {case}; log {output / 'run.log'}", flush=True)
        try:
            with (output / "run.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
                result = run_case(root, dim, solver, dt, args.tol, args.end_time, shape, args.gather)
            print(f"Completed {case} in {result['elapsed_seconds']:.1f}s", flush=True)
        except Exception as error:
            traceback.print_exc()
            failed.append(case)
            (output / "status.json").write_text(json.dumps(dict(status="failed", case=case, error=repr(error)), indent=2))
    if failed:
        raise SystemExit(f"Failed cases: {failed}")


if __name__ == "__main__":
    main()
