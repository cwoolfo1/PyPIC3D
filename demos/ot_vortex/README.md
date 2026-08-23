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

Build and push the pinned CUDA image from the repository root:

```bash
docker build -f demos/ot_vortex/chtc/Dockerfile \
  -t <registry-user>/pypic3d:cuda12.1.1-v1 .
docker push <registry-user>/pypic3d:cuda12.1.1-v1
```

Copy the `chtc` directory into your CHTC `/home` directory, then submit the
200-step benchmark:

```bash
condor_submit ot_vortex_gpu.sub \
  case=benchmark job_length=short \
  image=<registry-user>/pypic3d:cuda12.1.1-v1 \
  staging=osdf:///chtc/staging/<initial>/<netid>
```

Use `case=pilot` after the benchmark. Select `job_length=short` below ten
projected hours, `medium` below twenty hours, and `long` otherwise. Submit
`case=spectrum job_length=long` only if the pilot projects below six days and
passes the conservation checks. HTCondor controls `CUDA_VISIBLE_DEVICES`; the
wrapper deliberately does not modify it. Every result archive includes
`gpu_usage.csv` and `runtime_projection.json`; use the latter to select the
pilot and production job classes.
