# Blandford–Znajek monopole

This demo evolves pair plasma and electromagnetic fields in a fixed Kerr–Schild
spacetime. It reproduces the macroscopic monopole in Entity II, section 4.5,
Figure 6 ([paper](https://doi.org/10.3847/1538-4365/ae6592)): radial electromagnetic
outflow powered by black-hole rotation. It does not model a collimated jet.

## Run

From the repository root, using an environment with CUDA-enabled JAX and the
project dependencies:

```bash
CUDA_VISIBLE_DEVICES=0 python -m demos.bz_monopole.run_bz_monopole \
  --mode run --output demos/bz_monopole/runs/demo
```

The accepted defaults are 64×64 cells, one GPU, sixteen nominal pairs per cell,
spin 0.2, magnetization 2500, skin depth 0.02 M, Larmor radius 0.0004 M,
timestep cap 0.004 M, injection interval 0.1 M, and end time 200 M.
They use Cartesian particle coordinates, four implicit iterations, Entity
field interpolation, four conservative filter passes, five inner field planes,
and batches of 8192 particles. Outputs are saved every 5 M.

For a short check, add `--end-time 1 --output-interval 1`. Without `--mode run`,
the driver checks initialization only. `--backend cpu` requires
`JAX_PLATFORMS=cpu`. Use `--help` for resolution and runtime controls.

Restart into a fresh output directory, keeping the physical and numerical
settings identical; end time and output cadence may change:

```bash
CUDA_VISIBLE_DEVICES=0 python -m demos.bz_monopole.run_bz_monopole \
  --mode run --restart demos/bz_monopole/runs/short/checkpoint.npz \
  --end-time 200 --output demos/bz_monopole/runs/continued
```

The checkpoint contains particles, fields, RNG, and staggered field history.
Version-3 checkpoints from the accepted configuration remain supported. Signal
interruptions request a checked checkpointed stop. Metric errors, nonfinite
states, failed implicit solves, unsupported displacements, and capacity failures
stop the run. Exterior Gauss and magnetic-divergence tolerances default to 1e-10;
interior-horizon and sponge residuals are reported separately.

## Inspect results

Plot a run or the bundled measured result without JAX or a GPU:

```bash
python -m demos.bz_monopole.plot_entity_bz \
  --data demos/bz_monopole/reference --time 200 --output-dir /tmp/bz-reference
python -m demos.bz_monopole.animate_bz demos/bz_monopole/runs/demo \
  --times 0 50 100 150 200 --output /tmp/bz-evolution.gif
```

The animator uses only snapshots in the supplied directory; all requested times
must exist. Figures retain the full angular profiles and measured flux contours.
Snapshots, `comparison.json`, and the run manifest contain numerical diagnostics;
a completed run is not automatically a new validation claim.

![Measured fields at 200 M](reference/entity_bz_diagnostics.png)

[Measured evolution from 0 to 200 M](reference/evolution.gif).
The reference includes final arrays, normalization metadata, a compact summary,
and SHA-256 hashes for the unchanged numerical/image artifacts.

## Numerical method and validation

Particles use one supplied-grid cubic Hermite metric reconstruction: inverse
metrics, derivatives, and connections derive from the same interpolant. Static
metric runs require three guard cells. The Cartesian particle option regularizes
axisymmetric spherical geometry by transforming the supplied metric and particle
momenta; it does not substitute an analytic metric. Stored staggered momenta
are expressed in the spherical basis at their stored positions, including birth
seeding and centered deposition states. Cartesian and native checkpoints are
not interchangeable. Core library defaults remain independently configurable.

The demo retains electromagnetic/geodesic splitting, implicit midpoint residual
checks, metric-weighted auxiliary fields, and matching filters on integrated
charge and current. No particle cooling, energy clipping, or imposed BZ solution
is used. Invalid metric samples are reported rather than repaired or clamped.

At 200 M, mean power over 2.2–7.5 M is 1.00244 L_BZ. Toroidal-field profile
errors at radii 2, 3, 4, and 5 M are 0.86–1.28%; rotation errors are 2.16–3.07%,
using a 15-degree polar exclusion for those profile measurements. The plotted
curves retain the poles. Eleven states from 150–200 M have power standard
deviation 0.0342%. Exterior Gauss/divergence residuals are 5.72e-13/5.71e-14.
Normalization is Omega_H=a/(2 r_H), L_BZ=B0² Omega_H²/6.

Independent tests cover metric consistency, close-axis free motion, magnetic
energy conservation, curved-geodesic refinement, charge continuity, batching,
and restart equivalence. Settled timestep, grid, particle-count, seed, and
injection-cadence comparisons pass. These finite-resolution comparisons do not
establish a uniform convergence order for the full coupled PIC simulation.

With pytest and SciPy installed, run the focused demo and metric tests on two
virtual CPU devices (needed for tile-boundary tests):

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true XLA_FLAGS=--xla_force_host_platform_device_count=2 \
  python -m pytest demos/bz_monopole tests/code_tests/bz_*test.py \
  tests/code_tests/hermite_*test.py
```

The user accepted this macroscopic demonstration on 2026-09-18. Kinetic lengths
are eight times the paper's, and the grid is 64×64 rather than 1024×1024.
Kinetic-scale robustness and exact-parameter kinetic reproduction are outside
that acceptance; smaller-scale trials included failures. Fixed particle weights
and correlated pair birth momenta differ from Entity's reference implementation.
Boundary-region residuals are larger than exterior residuals. Complete historical
records and raw runs are archived outside the repository at
`../PyPIC3D-bz-campaign-archive/20260918/`.
