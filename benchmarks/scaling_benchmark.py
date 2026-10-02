#!/usr/bin/env python3
"""Reproducible CPU scaling and kernel timing benchmark for PyPIC3D.

The controller launches a fresh process for every case because JAX reads the
CPU device/thread configuration before the first backend is initialized.
Compilation is always warmed up and excluded from the reported medians.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _config(path: Path, cells: int, particles: int, cores: int) -> None:
    if cells % cores:
        raise ValueError(f"cells ({cells}) must be divisible by cores ({cores})")
    path.write_text(
        f"""[simulation_parameters]
name = "scaling benchmark"
solver = "electrodynamic_yee"
Nx = {cells}
Ny = 1
Nz = 1
particle_tile_nx = {cells // cores}
particle_tile_ny = 1
particle_tile_nz = 1
particle_tile_capacity_factor = 1.20
guard_cells = 2
Nt = 2
dt = 1.0e-12
x_wind = 1.0
y_wind = 1.0
z_wind = 1.0
shape_factor = 1
current_calculation = "j_from_rhov"
filter_j = "none"
particle_pusher = "boris"
relativistic = false
verbose = false

[plotting]
plotting_interval = 1000000
plot_openpmd_particles = false
plot_openpmd_fields = false

[particle1]
name = "electron"
N_particles = {particles}
charge = -1.602176634e-19
mass = 9.1093837e-31
vth = 1.0e5
number_density = 1.0e10
"""
    )


def _ready(value):
    import jax

    return jax.block_until_ready(value)


def _median_time(fn, args, repeats: int) -> float:
    _ready(fn(*args))  # compile and warm caches
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ready(fn(*args))
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def worker(args) -> None:
    import jax
    import toml

    from PyPIC3D.boundary_conditions.ghost_cells import update_tiled_vector_ghost_cells
    from PyPIC3D.deposition.J_from_rhov import J_from_rhov
    from PyPIC3D.initialization import initialize_simulation
    from PyPIC3D.particles.particle_tile_communication import (
        refresh_tiled_particle_tiles,
        update_tiled_particle_positions,
    )
    from PyPIC3D.pusher.particle_push import particle_push
    from PyPIC3D.solvers.first_order_yee import update_B, update_E
    from PyPIC3D.utils import add_external_fields

    config = Path(args.config)
    loop, particles, fields, static, dynamic, _, _, species = initialize_simulation(toml.load(config))
    E, B, J, _, _, external, pml, _ = fields
    push_E, push_B = add_external_fields(E, B, external)

    full = jax.jit(lambda p, f: loop(p, species, f, static, dynamic))
    push = jax.jit(lambda p: particle_push(p, species, push_E, push_B, static, dynamic))

    def deposit(p, current):
        p = update_tiled_particle_positions(p, species, dynamic.dt / 2)
        p, _ = refresh_tiled_particle_tiles(p, static, dynamic)
        current = J_from_rhov(p, species, current, static, dynamic)
        p = update_tiled_particle_positions(p, species, dynamic.dt / 2)
        p, _ = refresh_tiled_particle_tiles(p, static, dynamic)
        return p, current

    def fields_step(e, b, current):
        b, state = update_B(e, b, static, dynamic, pml, do_filter=False)
        e, state = update_E(e, b, current, static, dynamic, state)
        b, state = update_B(e, b, static, dynamic, state, do_filter=True)
        return e, b, state

    deposition = jax.jit(deposit)
    field_evolution = jax.jit(fields_step)
    ghost = jax.jit(lambda vector: update_tiled_vector_ghost_cells(vector, static, static.guard_cells))

    timings = {
        "step_s": _median_time(full, (particles, fields), args.repeats),
        "particle_push_s": _median_time(push, (particles,), args.repeats),
        "deposition_retile_s": _median_time(deposition, (particles, J), args.repeats),
        "field_evolution_s": _median_time(field_evolution, (E, B, J), args.repeats),
        "one_ghost_exchange_s": _median_time(ghost, (E,), args.repeats),
    }
    print("BENCHMARK_JSON=" + json.dumps({
        "cells": args.cells, "particles": args.particles, "cores": args.cores,
        "repeats": args.repeats, **timings,
    }, sort_keys=True))


def _run_case(cells: int, particles: int, cores: int, repeats: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="pypic3d-bench-") as directory:
        config = Path(directory) / "case.toml"
        _config(config, cells, particles, cores)
        env = os.environ.copy()
        env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={cores}"
        env["OMP_NUM_THREADS"] = str(cores)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--config", str(config),
                   "--cells", str(cells), "--particles", str(particles), "--cores", str(cores),
                   "--repeats", str(repeats)]
        result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(
                f"benchmark worker failed ({result.returncode})\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        marker = next(line for line in result.stdout.splitlines() if line.startswith("BENCHMARK_JSON="))
        return json.loads(marker.split("=", 1)[1])


def controller(args) -> None:
    cases = []
    for particles in args.particle_counts:
        cases.append(("particle", args.fixed_cells, particles, 1))
    for cells in args.cell_counts:
        cases.append(("cell", cells, args.fixed_particles, 1))
    for cores in args.core_counts:
        cases.append(("strong", args.fixed_cells, args.fixed_particles, cores))
        cases.append(("weak", args.weak_cells_per_core * cores, args.weak_particles_per_core * cores, cores))

    rows = []
    for kind, cells, particles, cores in cases:
        print(f"running {kind}: cells={cells}, particles={particles}, cores={cores}", flush=True)
        rows.append({"scaling": kind, **_run_case(cells, particles, cores, args.repeats)})

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with (output / "timings.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (output / "metadata.json").write_text(json.dumps({
        "python": sys.version, "platform": sys.platform, "cpu_count": os.cpu_count(),
        "command": sys.argv,
    }, indent=2))
    plot_results(output / "timings.csv", output)


def plot_results(csv_path: Path, output: Path) -> None:
    import matplotlib.pyplot as plt

    with csv_path.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in row:
            if key != "scaling":
                row[key] = float(row[key])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, kind, xkey, label in ((axes[0], "particle", "particles", "Particles"),
                                  (axes[1], "cell", "cells", "Cells")):
        selected = [r for r in rows if r["scaling"] == kind]
        ax.plot([r[xkey] for r in selected], [r["step_s"] for r in selected], "o-")
        ax.set(xlabel=label, ylabel="Median step time (s)", title=f"{label} scaling")
        ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(output / "problem_size_scaling.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, kind, title in ((axes[0], "strong", "Strong scaling"), (axes[1], "weak", "Weak scaling")):
        selected = [r for r in rows if r["scaling"] == kind]
        cores = [r["cores"] for r in selected]; times = [r["step_s"] for r in selected]
        ax.plot(cores, times, "o-", label="measured")
        if kind == "strong": ax.plot(cores, [times[0] / c for c in cores], "--", label="ideal")
        else: ax.axhline(times[0], ls="--", label="ideal")
        ax.set(xlabel="JAX CPU devices", ylabel="Median step time (s)", title=title)
        ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(output / "core_scaling.png", dpi=180); plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config")
    parser.add_argument("--cells", type=int); parser.add_argument("--particles", type=int)
    parser.add_argument("--cores", type=int)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--particle-counts", type=int, nargs="+", default=[256, 1024, 4096, 16384])
    parser.add_argument("--cell-counts", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--core-counts", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--fixed-cells", type=int, default=128)
    parser.add_argument("--fixed-particles", type=int, default=8192)
    parser.add_argument("--weak-cells-per-core", type=int, default=32)
    parser.add_argument("--weak-particles-per-core", type=int, default=2048)
    parser.add_argument("--output", default="benchmarks/results")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    worker(parsed) if parsed.worker else controller(parsed)
