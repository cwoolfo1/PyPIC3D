# Orszag--Tang GPU spectrum cases

The original 200-by-200 setup is retained as `orszag_tang_legacy.toml`. The
default `orszag_tang.toml` is a 512-by-512 GPU pilot spanning 25.6 ion inertial
lengths. `orszag_tang_spectrum.toml` is the 1024-by-1024 production case
spanning 51.2 ion inertial lengths.

For the legacy density of \(10^{14}\,\mathrm{m^{-3}}\), the original one-meter
box spans only 1.88 electron inertial lengths and 0.044 physical-proton ion
inertial lengths. Its frozen ions and light-speed electron-only flow therefore
cannot provide an MHD inertial range.

Both new cases use mobile reduced-mass ions, a five-to-one guide field,
species beta 0.08, a common Alfvenic OT flow, and unfiltered direct current
deposition (`j_from_rhov`). Because direct deposition is not exactly charge
conserving, `conservation.csv` records the Gauss residual and total-energy
drift at every field output.

## Local preparation

Run these commands from this directory. Generating the production particles
requires several GiB of local disk space.

```bash
python3 initial_conditions.py --case pilot
python3 -m PyPIC3D --config orszag_tang.toml
python3 analyze_spectrum.py --config orszag_tang.toml
```

Replace `pilot` with `spectrum` and use `orszag_tang_spectrum.toml` for the
production run. The analyzer always uses snapshots between four and six
turnover times and never changes the configured fit interval after seeing the
result.

## CHTC GPU Lab

Build the pinned CUDA image from the repository root with a unique tag:

```bash
docker build --no-cache --progress=plain \
  -f demos/ot_vortex/chtc/Dockerfile \
  -t <registry-user>/pypic3d:cuda12.3.2-smoke-v1 .
```

Before submitting to CHTC, configure NVIDIA Container Toolkit and confirm
that Docker can expose the local GPU:

```bash
docker run --rm --gpus all \
  nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
```

Run the image preflight, followed by the five-step 64-by-64 smoke case. The
runner uses the same ``run_gpu_job`` entry point as Condor and writes results
to the optional second argument.

```bash
docker run --rm --gpus all \
  <registry-user>/pypic3d:cuda12.3.2-smoke-v1 \
  bash -lc 'ptxas --version && python3 /opt/PyPIC3D/demos/ot_vortex/chtc/container_preflight.py && python3 -c '\''import jax; print(jax.devices("gpu"))'\'''

docker run --rm \
  --env JAX_PLATFORMS=cpu \
  --env XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  <registry-user>/pypic3d:cuda12.3.2-smoke-v1 \
  python3 -m unittest \
    tests.code_tests.distributed_ghost_cells_test \
    tests.code_tests.distributed_filters_test \
    tests.code_tests.distributed_particle_refresh_test

demos/ot_vortex/chtc/run_local_gpu_smoke \
  <registry-user>/pypic3d:cuda12.3.2-smoke-v1 \
  "$PWD/local_ot_vortex_smoke" \
  0
```

The optional final argument selects one host GPU, matching CHTC's one-GPU
allocation. It defaults to device 0.

The smoke run must produce ``ot_vortex_smoke.tar.zst`` in the mounted work
directory plus ``output.toml``, ``gpu_usage.csv``, and
``runtime_projection.json`` under
``local_ot_vortex_smoke/ot_vortex_work/runs/smoke/data``. Inspect the archive
before publishing:

```bash
tar --zstd -tf local_ot_vortex_smoke/ot_vortex_smoke.tar.zst
docker push <registry-user>/pypic3d:cuda12.3.2-smoke-v1
docker pull <registry-user>/pypic3d:cuda12.3.2-smoke-v1
```

Repeat the preflight against the pulled tag. Do not reuse an existing tag;
execution nodes may retain an older image under a mutable tag.

Copy the `chtc` directory into your CHTC `/home` directory, then submit the
200-step benchmark:

```bash
condor_submit ot_vortex_gpu.sub \
  case=benchmark job_length=short \
  image=<registry-user>/pypic3d:cuda12.3.2-smoke-v1 \
  staging=osdf:///chtc/staging/<initial>/<netid>
```

Use `case=pilot` after the benchmark. Select `job_length=short` below ten
projected hours, `medium` below twenty hours, and `long` otherwise. Submit
`case=spectrum job_length=long` only if the pilot projects below six days and
passes the conservation checks. HTCondor controls `CUDA_VISIBLE_DEVICES`; the
wrapper deliberately does not modify it. Every result archive includes
`gpu_usage.csv` and `runtime_projection.json`; use the latter to select the
pilot and production job classes.
