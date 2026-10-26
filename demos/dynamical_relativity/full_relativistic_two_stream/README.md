# Fully relativistic two-stream demo

From this directory, run:

```bash
python initial_data.py
PyPIC3D --config two_stream.toml
```

Edit the plasma, perturbation, and numerical globals near the top of
`initial_data.py`. Each generation saves an independent input bundle under
`outputs/initial_<timestamp>/`, assigns a fresh simulation directory under
`runs/<timestamp>/`, and publishes a launch configuration beside the script.
Generated input paths are absolute. Native initialization converts the
common-time physical fields and covariant particle momenta to their staggered
runtime state exactly once.

The default is a periodic, warm electron-ion plasma with electron beams at
opposite speeds. The electron positions share a sinusoidal displacement;
colocated opposite momentum pairs give zero initial momentum density. The
boosted Maxwell–Jüttner loading includes the
[equal-time correction](https://arxiv.org/abs/1504.03910).

The conformally flat expanding metric solves the discrete Hamiltonian
constraint with constant negative trace of the extrinsic curvature and
`mean(W**-3) = 1`. The full matter density is retained, following the
[periodic constant-mean-curvature construction](https://link.aps.org/accepted/10.1103/PhysRevD.93.124059).
Evolution uses harmonic slicing, frozen zero shift, and periodic boundaries
for particles, electromagnetic fields, and all six BSSN faces.

`CURRENT_FILTER = "bilinear"` selects matched periodic filtering. The native
configuration records this as `filter_j = "bilinear"`; `"none"` and
`"digital"` are also available. Native GR configurations retain `"none"` as
their default; newly generated demo configurations explicitly opt in to
`"bilinear"`. Matched GR filtering requires periodic Cartesian grids.
Selecting `"digital"` now filters all gathers and sources described here,
extending its former current-only GR behavior. Set `CURRENT_FILTER_ALPHA`
before generating data to select the digital stencil's central weight: its
native default `1` is the identity, while `0.5` gives nonzero smoothing.
The importable builders accept `filter_alpha` for the same setting, and
generated configurations record the coefficient used by the solve as `alpha`.
Charge, current, and coordinate stress-energy
densities use the same filter as the particle field and metric gathers.
The Gauss solve uses filtered charge. Holding local orthonormal momenta and
densitized electric fields fixed, the Newton solve uses

```text
rho(W) = W**3 * F[rho_particle_flat + W * rho_EM_flat].
```

Covariant momenta are constructed using the filtered metric sampled by the
pusher. The evolved BSSN geometry and the metric used for field densitization
remain raw. The final initial-data check evaluates the actual production
sources and constraints. Particle Lorentz factors used in the sources follow
the filtered particle metric; physical Lorentz-factor output uses the raw
geometry.

Filtering suppresses poorly resolved particle noise and preserves periodic
charge continuity when matched to the current filter. It changes finite-grid
particle coupling; it does not establish discrete stress-energy conservation
or repair the BSSN derivative-stencil mismatch. Compare constraints, the
work-energy residual, and resolved instability growth independently, and
refine the grid and particle loading before treating a run as converged.

Native field output includes BSSN variables and constraints, with initial and
final snapshots included for matched GR filtering. The companion `plot.py`
can read the native output to plot constraints and make particle
phase-space movies. There is no separate evolution runner in this demo.
