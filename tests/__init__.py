"""Shared unittest package for PyPIC3D tests."""

import os

import jax

# Multi-tile tests need one JAX device per tile. Default to 16 CPU devices
# unless the caller chose a platform or device count; this runs before any
# test module touches the JAX backend.
if "JAX_PLATFORMS" not in os.environ:
    jax.config.update("jax_platforms", "cpu")
if "JAX_NUM_CPU_DEVICES" not in os.environ:
    jax.config.update("jax_num_cpu_devices", 16)

jax.config.update("jax_enable_x64", True)
