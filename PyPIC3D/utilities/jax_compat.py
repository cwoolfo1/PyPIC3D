"""Compatibility helpers for JAX APIs that moved between releases."""

import jax


if hasattr(jax, "shard_map"):
    SHARD_MAP_API = "jax.shard_map"

    def shard_map(function, *, mesh, in_specs, out_specs, check_vma=True):
        return jax.shard_map(
            function,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=check_vma,
        )

else:
    from jax.experimental.shard_map import shard_map as _experimental_shard_map

    SHARD_MAP_API = "jax.experimental.shard_map.shard_map"

    def shard_map(function, *, mesh, in_specs, out_specs, check_vma=True):
        return _experimental_shard_map(
            function,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_rep=check_vma,
        )
