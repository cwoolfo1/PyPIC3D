"""Validate the CUDA image's JAX API and copied demo permissions."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

from PyPIC3D.utilities.jax_compat import SHARD_MAP_API, shard_map


def main():
    if jax.__version__ != "0.4.38":
        raise RuntimeError(f"expected JAX 0.4.38, found {jax.__version__}")

    devices = jax.devices("cpu")
    mesh = Mesh(np.asarray(devices[:1]).reshape((1, 1, 1)), ("x", "y", "z"))
    mapped = shard_map(
        lambda value: value + 1,
        mesh=mesh,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    result = mapped(jnp.asarray(1)).block_until_ready()
    if int(result) != 2:
        raise RuntimeError("shard_map preflight returned an unexpected result")

    expected_modes = {
        Path("/opt/PyPIC3D/demos/ot_vortex/chtc"): 0o755,
        Path("/opt/PyPIC3D/demos/ot_vortex/initial_conditions.py"): 0o644,
        Path("/opt/PyPIC3D/demos/ot_vortex/orszag_tang.toml"): 0o644,
    }
    for path, required_mode in expected_modes.items():
        actual_mode = path.stat().st_mode & 0o777
        if actual_mode & required_mode != required_mode:
            raise PermissionError(
                f"{path} has mode {actual_mode:o}; required bits are {required_mode:o}"
            )
        print(f"mode {actual_mode:o} {path}")

    print(f"JAX {jax.__version__}")
    print(f"shard_map API: {SHARD_MAP_API}")
    print(f"CPU preflight devices: {devices}")


if __name__ == "__main__":
    main()
