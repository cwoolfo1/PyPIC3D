"""Run the charged electrovacuum puncture on a CUDA device.

All numerical defaults are in initial_data.py. Checkpoints retain native time
histories and sponge references; only the derived Yee metric is reconstructed.
"""

import argparse
from datetime import datetime, timezone
import csv
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("JAX_PLATFORMS", "cuda")
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from PyPIC3D.relativity.dynamical_relativity import build_bssn_metric
from PyPIC3D.solvers.GR_yee.time_loop_dynamical_relativity import time_loop_dynamical_relativity
from demos.dynamical_relativity.charged_vacuum_puncture.initial_data import (
    build_initial_data, MASS, CHARGE, NUM_POINTS, HALF_WIDTH, CFL, FINAL_TIME,
    KO_DISSIPATION, MOMENTUM_DAMPING,
)
from demos.dynamical_relativity.charged_vacuum_puncture.diagnostics import (
    diagnostic_values, centered_fields, valid_state, failure_description,
)


def write_json(path, data):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def checkpoint_tree(fields, dynamic):
    return fields[:6], fields[7], fields[8], dynamic.bssn_parameters


def save_checkpoint(path, fields, dynamic, config, step):
    leaves = jax.tree.leaves(checkpoint_tree(fields, dynamic))
    temporary = Path(str(path) + ".tmp.npz")
    metadata = dict(version=1, config=config, step=step, time=float(dynamic.bssn_parameters.time))
    jnp.savez(temporary, metadata=json.dumps(metadata), **{f"q{i}": q for i, q in enumerate(leaves)})
    temporary.replace(path)


def load_checkpoint(path, fields, static, dynamic, config):
    with jnp.load(path, allow_pickle=False) as saved:
        metadata = json.loads(saved["metadata"].item())
        if metadata["version"] != 1 or metadata["config"] != config:
            raise ValueError("Restart requires identical physics, grid, dt, and source-filter settings")
        template, tree = jax.tree.flatten(checkpoint_tree(fields, dynamic))
        arrays = [jnp.asarray(saved[f"q{i}"]) for i in range(len(template))]
        if any(a.shape != b.shape or a.dtype != b.dtype for a, b in zip(arrays, template)):
            raise ValueError("Checkpoint arrays do not match this grid/state layout")
        initial, history, overflow, bssn = jax.tree.unflatten(tree, arrays)
    dynamic = dynamic._replace(bssn_parameters=bssn)
    metric = build_bssn_metric(bssn.state, static, dynamic)
    return initial + (metric, history, overflow), dynamic, metadata["step"]


def save_snapshot(path, fields, static, dynamic):
    D, B = centered_fields(fields, static)
    q = dynamic.bssn_parameters.state
    temporary = Path(str(path) + ".tmp.npz")
    jnp.savez(temporary, **q._asdict(), D=D, B=B, time=dynamic.bssn_parameters.time,
              axis=dynamic.grids.center[0][1:-1], dx=dynamic.dx)
    temporary.replace(path)


def available_storage(output, required):
    free = shutil.disk_usage(output).free
    if free < required + 5*1024**3:
        raise RuntimeError(f"storage reserve: {free/1024**3:.2f} GiB free, "
                           f"need {required/1024**3:.2f} GiB plus 5 GiB reserve")


def run(args):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and args.restart is None:
        raise FileExistsError(f"Refusing to overwrite existing run {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = dict(n=args.n, mass=args.mass, charge=args.charge, half_width=args.half_width,
                  cfl=args.cfl, nu=args.nu, kappa=args.kappa, source_filter=args.source_filter)
    if args.duration < 0 or args.snapshot_interval <= 0 or args.checkpoint_interval <= 0:
        raise ValueError("Duration must be nonnegative and output intervals positive")
    requested_dt = args.cfl*2*args.half_width/args.n
    if args.restart:
        with jnp.load(args.restart, allow_pickle=False) as saved:
            config["dt"] = json.loads(saved["metadata"].item())["config"]["dt"]
    else:
        # Shorten dt once, before seeding, to hit the requested endpoint exactly.
        config["dt"] = (args.duration/math.ceil(args.duration/requested_dt)
                        if args.duration > 0 else requested_dt)
    write_json(output/"status.json", dict(state="initializing", config=config, target_time=args.duration))
    print("Devices:", jax.devices(), flush=True)
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("This campaign requires the Jetstream CUDA device")
    builder_config = {k: v for k, v in config.items() if k != "source_filter"}
    static, dynamic, particles, species, fields = build_initial_data(**builder_config, output_dir=output)
    step = 0
    if args.restart:
        fields, dynamic, step = load_checkpoint(args.restart, fields, static, dynamic, config)
    if not bool(valid_state(fields, dynamic.bssn_parameters)):
        raise RuntimeError(failure_description(fields, dynamic.bssn_parameters))
    dt = float(dynamic.dt)
    end_step = round(args.duration/dt)
    if abs(end_step*dt-args.duration) > 1.e-10 or end_step < step:
        raise ValueError("Duration must be a multiple of the fixed dt, at or after restart time")
    def due(interval):
        # Cross fixed physical output times without accumulating rounding drift
        # when the native timestep does not divide the requested interval.
        return (step == 0 or step == end_step
                or math.floor((step*dt + 1.e-10)/interval)
                > math.floor(((step-1)*dt + 1.e-10)/interval))
    # Upper bound includes full histories, plus temporary atomic-file writes.
    checkpoint_bytes = sum(q.size*q.dtype.itemsize for q in jax.tree.leaves(checkpoint_tree(fields, dynamic)))
    snapshot_bytes = (sum(q.size for q in dynamic.bssn_parameters.state) + 6*args.n**3)*8
    remaining_snapshots = int((end_step-step)*dt/args.snapshot_interval) + 2
    available_storage(output, remaining_snapshots*snapshot_bytes + 3*checkpoint_bytes)
    write_json(output/"metadata.json", dict(
        config=config, dt=dt, native_dt=dt/2, target_time=args.duration,
        restart=str(args.restart) if args.restart else None,
        paper="https://arxiv.org/abs/1311.6483v2", equations="2.5-2.13; u=1",
        units="Gaussian G=c=1; rho_EM=(E^2+B^2)/(8*pi); charge=flux(D)/(4*pi)",
        masks="exterior: 2<=coordinate r<=8; sponge: max(abs(x_i))>=half_width-3",
        jax_version=jax.__version__, devices=[str(d) for d in jax.devices()],
        snapshot_interval=args.snapshot_interval, checkpoint_interval=args.checkpoint_interval,
        estimated_remaining_bytes=remaining_snapshots*snapshot_bytes + 3*checkpoint_bytes))

    advance = jax.jit(lambda p, f, d: time_loop_dynamical_relativity(
        p, species, f, static, d, source_filter=args.source_filter))
    # The diagnostics JIT is separate from evolution to bound compile/memory.
    started = time.monotonic()
    checkpoint_index = int(step*dt/args.checkpoint_interval)
    diagnostics_path = output/"diagnostics.csv"
    last_diagnostic = -1.
    if diagnostics_path.exists():
        with diagnostics_path.open() as stream:
            rows = list(csv.DictReader(stream))
        if rows:
            last_diagnostic = float(rows[-1]["time"])
            if last_diagnostic > float(dynamic.bssn_parameters.time) + 1.e-10:
                raise ValueError("Output extends beyond restart; choose a fresh output directory")

    def record():
        nonlocal last_diagnostic, checkpoint_index
        now = float(dynamic.bssn_parameters.time)
        if due(.5):
            available_storage(output, checkpoint_bytes)
            if now > last_diagnostic + 1.e-10:
                values = {k: float(v) for k, v in diagnostic_values(particles, species, fields, static, dynamic).items()}
                with diagnostics_path.open("a") as stream:
                    writer = csv.DictWriter(stream, fieldnames=sorted(values))
                    if stream.tell() == 0:
                        writer.writeheader()
                    writer.writerow(values)
                last_diagnostic = now
            print(f"step={step} t={now:.6f} elapsed={time.monotonic()-started:.1f}s", flush=True)
            write_json(output/"status.json", dict(state="running", step=step, time=now,
                       target_time=args.duration, elapsed_seconds=time.monotonic()-started))
        if due(args.snapshot_interval):
            path = output/f"snapshot_{step:07d}.npz"
            if not path.exists():
                available_storage(output, snapshot_bytes + checkpoint_bytes)
                save_snapshot(path, fields, static, dynamic)
        if due(args.checkpoint_interval):
            available_storage(output, checkpoint_bytes)
            save_checkpoint(output/f"checkpoint_{checkpoint_index % 2}.npz", fields, dynamic, config, step)
            checkpoint_index += 1

    try:
        record()
        while step < end_step:
            next_particles, next_fields, next_dynamic = advance(particles, fields, dynamic)
            if not bool(valid_state(next_fields, next_dynamic.bssn_parameters)):
                reason = failure_description(next_fields, next_dynamic.bssn_parameters)
                available_storage(output, checkpoint_bytes)
                save_checkpoint(output/f"checkpoint_{checkpoint_index % 2}.npz", fields, dynamic, config, step)
                write_json(output/"failure.json", dict(step=step+1, time=(step+1)*dt,
                           reason=reason, last_valid_step=step))
                raise RuntimeError(f"Invalid evolved state at step {step+1}: {reason}")
            particles, fields, dynamic = next_particles, next_fields, next_dynamic
            step += 1
            record()
        write_json(output/"status.json", dict(state="complete", step=step,
                   time=float(dynamic.bssn_parameters.time), elapsed_seconds=time.monotonic()-started))
    except BaseException as error:
        write_json(output/"status.json", dict(state="failed", step=step,
                   time=float(dynamic.bssn_parameters.time), error=str(error),
                   elapsed_seconds=time.monotonic()-started))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=NUM_POINTS)
    parser.add_argument("--mass", type=float, default=MASS)
    parser.add_argument("--charge", type=float, default=CHARGE)
    parser.add_argument("--half-width", type=float, default=HALF_WIDTH)
    parser.add_argument("--cfl", type=float, default=CFL)
    parser.add_argument("--nu", type=float, default=KO_DISSIPATION)
    parser.add_argument("--kappa", type=float, default=MOMENTUM_DAMPING)
    parser.add_argument("--duration", type=float, default=FINAL_TIME)
    parser.add_argument("--source-filter", choices=("none", "binomial"), default="binomial")
    parser.add_argument("--output", type=Path, default=Path("/media/volume/JAXBSSN") / (
        "charged_puncture_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--snapshot-interval", type=float, default=4.)
    parser.add_argument("--checkpoint-interval", type=float, default=5.)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as error:
        if args.output.exists() and (args.output/"status.json").exists():
            status = json.loads((args.output/"status.json").read_text())
            if status.get("state") != "failed":
                status.update(state="failed", error=str(error))
                write_json(args.output/"status.json", status)
        raise


if __name__ == "__main__":
    main()
