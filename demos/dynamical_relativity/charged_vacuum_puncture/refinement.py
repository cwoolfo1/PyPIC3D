"""Evaluate one initial-data resolution, writing only scalar diagnostics."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import jax
jax.config.update("jax_enable_x64", True)

from demos.dynamical_relativity.charged_vacuum_puncture.initial_data import build_initial_data
from demos.dynamical_relativity.charged_vacuum_puncture.diagnostics import diagnostic_values, valid_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    static, dynamic, particles, species, fields = build_initial_data(args.n)
    if not bool(valid_state(fields, dynamic.bssn_parameters)):
        raise RuntimeError("Invalid initial data")
    result = {key: float(value) for key, value in diagnostic_values(particles, species, fields, static, dynamic).items()}
    result.update(n=args.n, dx=float(dynamic.dx))
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)
