"""Summarize GPU usage and project the two full OT run times."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import toml


C = 2.99792458e8
TARGET_CONFIGS = {
    "pilot": "orszag_tang.toml",
    "spectrum": "orszag_tang_spectrum.toml",
}


def configured_steps(config):
    simulation = config["simulation_parameters"]
    if "Nt" in simulation:
        return int(simulation["Nt"])
    dx = float(simulation["x_wind"]) / int(simulation["Nx"])
    dy = float(simulation["y_wind"]) / int(simulation["Ny"])
    inverse_spacing = 0.0
    if int(simulation["Nx"]) > 1:
        inverse_spacing += 1.0 / dx
    if int(simulation["Ny"]) > 1:
        inverse_spacing += 1.0 / dy
    dt = float(simulation["cfl"]) / (C * inverse_spacing)
    return int(float(simulation["t_wind"]) / dt)


def peak_gpu_memory_mib(path):
    if not path.exists():
        return None
    peak = None
    with open(path, newline="") as stream:
        for row in csv.DictReader(stream):
            raw = row.get(" memory.used [MiB]") or row.get("memory.used [MiB]")
            if raw is None:
                continue
            value = float(raw.strip().split()[0])
            peak = value if peak is None else max(peak, value)
    return peak


def projection_class(seconds):
    if seconds < 10 * 3600:
        return "short"
    if seconds < 20 * 3600:
        return "medium"
    if seconds < 6 * 24 * 3600:
        return "long"
    return "checkpointing required"


def build_report(demo_dir, run_output, gpu_log):
    run = toml.load(run_output)
    seconds_per_step = float(run["simulation_stats"]["time_per_iteration"])
    projections = {}
    for name, filename in TARGET_CONFIGS.items():
        steps = configured_steps(toml.load(demo_dir / filename))
        seconds = seconds_per_step * steps
        projections[name] = {
            "steps": steps,
            "projected_seconds": seconds,
            "projected_hours": seconds / 3600.0,
            "recommended_gpu_job_length": projection_class(seconds),
        }
    return {
        "measured_seconds_per_step": seconds_per_step,
        "peak_gpu_memory_mib": peak_gpu_memory_mib(gpu_log),
        "projections": projections,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", required=True, type=Path)
    parser.add_argument("--gpu-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    demo_dir = Path(__file__).resolve().parents[1]
    report = build_report(demo_dir, args.run_output, args.gpu_log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
