"""Sequential Jetstream qualification and 200M continuation; no job overlap."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, JAX_PLATFORMS="cuda", JAX_ENABLE_X64="true",
                       XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_ALLOCATOR="platform")
    stages = []

    def status(state, stage):
        data = dict(state=state, stage=stage, updated=datetime.now(timezone.utc).isoformat(), stages=stages)
        temporary = output/"campaign_status.tmp"
        temporary.write_text(json.dumps(data, indent=2)+"\n")
        temporary.replace(output/"campaign_status.json")

    def execute(name, command):
        status("running", name)
        print(name, " ".join(map(str, command)), flush=True)
        with (output/(name+".log")).open("w") as log:
            result = subprocess.run(list(map(str, command)), cwd=ROOT, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT)
        stages.append(dict(name=name, returncode=result.returncode))
        status("running" if result.returncode == 0 else "stage_failed", name)
        return result.returncode == 0

    if not args.skip_tests:
        tests = ["charged_vacuum_puncture", "dynamical_relativity", "bssn_parameters",
                 "gr_filtered_sources", "gr_matched_filter", "gr_filtered_two_stream"]
        # Separate processes release executables and allocator state between suites.
        for test in tests:
            if not execute(test+"_tests", [sys.executable, "-u", "-m", "unittest",
                                           "tests.code_tests."+test+"_test", "-v"]):
                status("failed", test+"_tests")
                return 1

    refinement = []
    for n in (64, 96, 128):
        report = output/f"initial_{n}.json"
        if not execute(f"initial_{n}", [sys.executable, "-u", HERE/"refinement.py", "--n", n, "--output", report]):
            status("failed", f"initial_{n}")
            return 1
        refinement.append(json.loads(report.read_text()))
    checks = {name: all(refinement[i+1][name] < refinement[i][name] for i in range(2))
              for name in ("exterior_raw_hamiltonian_l2", "exterior_filtered_hamiltonian_l2", "exterior_gauss_l2")}
    (output/"refinement_checks.json").write_text(json.dumps(checks, indent=2)+"\n")
    if not all(checks.values()):
        status("failed", "initial_refinement")
        return 1

    def simulation(name, n, duration, filter_name="binomial", restart=None):
        command = [sys.executable, "-u", HERE/"run.py", "--n", n, "--duration", duration,
                   "--source-filter", filter_name, "--output", output/name]
        if name == "smoke":
            command += ["--half-width", 4]
        if name not in ("production", "pilot96_filtered"):
            command += ["--snapshot-interval", 10]
        if restart:
            command += ["--restart", restart]
        success = execute(name, command)
        if (output/name/"diagnostics.csv").exists() and list((output/name).glob("snapshot_*.npz")):
            plotted = execute(name+"_plots", [sys.executable, HERE/"plot.py", output/name])
            if not plotted:
                print("Plotting failed; preserved numerical results", flush=True)
        return success

    if not simulation("smoke", 16, 1.):
        status("failed", "smoke")
        return 1
    # A failed unfiltered control does not prevent the filtered comparison.
    simulation("pilot96_unfiltered", 96, 10., "none")
    filtered96 = simulation("pilot96_filtered", 96, 10.)
    filtered128 = simulation("pilot128_filtered", 128, 10.)
    if not (filtered96 and filtered128):
        status("failed", "filtered_qualification")
        return 1
    checkpoints = []
    # Read only metadata: never infer the newest state from rotating filenames.
    import numpy as np
    for path in (output/"pilot96_filtered").glob("checkpoint_*.npz"):
        with np.load(path, allow_pickle=False) as saved:
            info = json.loads(saved["metadata"].item())
        checkpoints.append((info["step"], path))
    restart = max(checkpoints)[1]
    production = output/"production"
    production.mkdir()
    # Retain the pilot segment of the same trajectory without duplicating
    # large snapshots. The extra pilot endpoint at 10M stays in the pilot.
    import shutil
    shutil.copyfile(output/"pilot96_filtered"/"diagnostics.csv", production/"diagnostics.csv")
    for snapshot in (output/"pilot96_filtered").glob("snapshot_*.npz"):
        step = int(snapshot.stem.split("_")[-1])
        if step % 80 == 0:
            os.link(snapshot, production/snapshot.name)
    if not simulation("production", 96, 200., restart=restart):
        status("failed", "production")
        return 1
    status("complete", "production")
    return 0


if __name__ == "__main__":
    sys.exit(main())
