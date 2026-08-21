# Fixed fourth-order FMR transfer report

Date: 2026-08-21

## Result

Every coarse/fine transfer now uses one tensor-product four-point Lagrange
formula.  The one-dimensional interpolation is exact through polynomial degree
three and has `O(h^4)` point error; each three-dimensional map has 64 donors.
The interface prolongation formula is unchanged, while E and B shadow
restriction has moved from three to four donors per axis.

The focused transfer, metric, initialization, parameter, openPMD, wave-crossing,
curl, and B-E-B stage tests pass.  All four transfer families converge above
the required 3.5 order in RMS and L-infinity norms.  The full TM111 PEC Maxwell
gate passes.  The periodic Maxwell gate remains below its required fine-region
order and is left failing without a threshold change.

## Production contract

- Refinement is fixed at 2:1 with one strictly interior rectangular patch.
- A patch spans at least three parent cells along every stored axis so that
  shadow restriction always reads four fine-owned interior donors.
- E and B prolongation and restriction use the same fourth-order transfer-map
  construction; no runtime interpolation-order option remains.
- Fine-owned values reconstruct the inactive coarse shadow before the current
  coarse state supplies constrained fine-interface values.
- Yee staggering, ownership masks, explicit curls, physical level spacing, and
  B-E-B time-centering are unchanged.

## Transfer validation

The table reports smooth manufactured-field errors at root resolutions
`N = 12, 24, 48`.

| Transfer/norm | N=12 | N=24 | N=48 | orders | Result |
|---|---:|---:|---:|---:|---|
| E prolongation RMS | 7.69910e-4 | 4.89575e-5 | 3.07308e-6 | 3.975, 3.994 | pass |
| E prolongation L-inf | 2.45977e-3 | 1.58660e-4 | 9.99456e-6 | 3.955, 3.989 | pass |
| B prolongation RMS | 7.92193e-4 | 5.03768e-5 | 3.16220e-6 | 3.975, 3.994 | pass |
| B prolongation L-inf | 1.88212e-3 | 1.19718e-4 | 7.51533e-6 | 3.975, 3.994 | pass |
| E restriction RMS | 7.82258e-5 | 4.49601e-6 | 2.56897e-7 | 4.121, 4.129 | pass |
| E restriction L-inf | 1.46183e-4 | 1.00201e-5 | 6.40378e-7 | 3.867, 3.968 | pass |
| B restriction RMS | 9.22781e-5 | 5.28738e-6 | 3.04094e-7 | 4.125, 4.120 | pass |
| B restriction L-inf | 2.43638e-4 | 1.67001e-5 | 1.06730e-6 | 3.867, 3.968 | pass |

Degree-three polynomial values are reproduced to `2e-12` for every E/B
prolongation and restriction map.  Interface coordinate reconstruction agrees
with the live staggered Yee coordinates to `2e-14`.

## Full Maxwell convergence

The production acceptance uses `CFL = 0.8` on the fine grid, halves space and
time together, and requires last-pair composite EM RMS order greater than 1.8
in the coarse, fine, and interface regions.

The TM111 PEC cavity test passes all unchanged gates.  The periodic wave is
finite and retains near-roundoff divergence, but its fine-region order remains
below threshold:

| N | coarse EM RMS | fine EM RMS | interface EM RMS |
|---:|---:|---:|---:|
| 12 | 1.316309e-2 | 3.660133e-3 | 1.202168e-2 |
| 24 | 3.263274e-3 | 1.225709e-3 | 2.680916e-3 |
| 48 | 8.056302e-4 | 3.794145e-4 | 6.540994e-4 |

| Region | EM orders | Result |
|---|---:|---|
| coarse | 2.012, 2.018 | pass |
| fine | 1.578, 1.692 | **fail** |
| interface | 2.165, 2.035 | pass |

The periodic last-pair fine orders are 1.621 for E and 1.765 for B.  Maximum
relative energy error decreases from `1.674e-4` to `1.688e-7`, and the final
`div(B)` L2 norms remain between `8.98e-15` and `9.11e-14`.  The surviving
failure is therefore still an accumulated fine-region interface error, not a
non-finite instability or divergence defect.

## Demo and reproduction

The z-invariant linear-wave demo completed all 64 steps.  Its readback contains
two blocks and 16 iterations, with root shape `(32, 32, 5)` and fine shape
`(32, 32, 6)`.

Run the principal gates from the repository root with the checkout's validated
Python environment:

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=16 /home/christopherwoolford/.conda/envs/pypic/bin/python -m unittest -v tests.code_tests.fmr_test tests.code_tests.fmr_metric_weights_test tests.code_tests.parameters_test tests.code_tests.initialization_test tests.code_tests.fmr_openpmd_test
XLA_FLAGS=--xla_force_host_platform_device_count=16 /home/christopherwoolford/.conda/envs/pypic/bin/python -m unittest -v tests.physics_tests.fmr_operator_convergence_test
XLA_FLAGS=--xla_force_host_platform_device_count=16 /home/christopherwoolford/.conda/envs/pypic/bin/python -m unittest -v tests.physics_tests.fmr_maxwell_convergence_test
/home/christopherwoolford/.conda/envs/pypic/bin/python demos/fmr_linear_wave_2d/run_fmr_linear_wave.py
```
