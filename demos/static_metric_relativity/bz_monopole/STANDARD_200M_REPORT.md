# BZ monopole: standard interpolation through 200 M

The default production demo completed **t=200 M (50,000 steps)** using the
standard metric-weighted field interpolation and **10° excised polar caps**.
No runtime checks failed and no constraint waiver was used. All final evolved
field, current, charge, potential, position, and momentum arrays are finite.
The final active particle count is **18,692**.

## Configuration and provenance

- Grid: 64×64, one GPU; spherical Kerr–Schild metric, spin a/M=0.2.
- Domain: r/M=1…10, θ=10°…170°; sponge starts at r/M=9.
- Conducting cap field boundaries and specular particle reflection; three
  guard cells, with metric samples strictly away from both axes.
- dt=0.004 M; seed 20260908; `pairs_per_cell=16`; capacity factor 8;
  injection interval 0.1 M; four current-filter passes; remaining production
  parameters are recorded in the archived `result.json`.
- Standard E/H constitutive interpolation only. Particle metric interpolation
  and polar update/deposition geometry retain the migrated demo behavior.
- NVIDIA RTX A4000, physical GPU 1; elapsed wall time including setup/output:
  **4.42 hours**. The GPU was shared during part of the run.
- Base commit `946331a15641064b09303350c5ccbe3222baaea8` plus the saved `source.diff` of the working
  tree at launch. The instrumentation records progress and checked residuals
  without changing the numerical kernels.

## Acceptance checks

| Relative exterior residual | Maximum at runtime checks | Maximum in saved snapshots | Final snapshot | Limit |
|---|---:|---:|---:|---:|
| Gauss | 3.89899276e-12 | 3.89337755e-12 | 3.89337755e-12 | 1e-10 |
| Magnetic divergence | 6.99172364e-14 | 6.87263859e-14 | 6.87263859e-14 | 1e-10 |

Constraints were checked every 100 steps, with `allow_divergence_errors=False`.
Finite-state, particle displacement, injection capacity, and migration checks
remained enabled throughout. The saved final state was also checked directly.

These are the demo's normalized integrated-flux **exterior** residuals. The
masks exclude the horizon interior, two cells next to physical boundaries,
and the sponge plus two cells. Runtime checks include 2,928 Gauss cells and
2,880 magnetic-divergence cells. Saved snapshots occur every 5 M; their
magnetic field is reconstructed at integer time, whereas runtime checks use
the evolved staggered field. Sampling times and magnetic time placement can
therefore give slightly different maxima. The constraint history grows slowly
but stays below the original limits through 200 M.

## Physical diagnostics

| r/M | Final L/L_BZ | L/L_BZ, 150–200 M mean ± std | Final median Ω/Ω_H | Median Ω/Ω_H, 150–200 M mean ± std |
|---:|---:|---:|---:|---:|
| 2 | 1.011424 | 1.011615 ± 0.000351 | 0.509487 | 0.509782 ± 0.000958 |
| 3 | 0.999389 | 0.998897 ± 0.000693 | 0.504663 | 0.505143 ± 0.001057 |
| 5 | 0.991758 | 0.990591 ± 0.001096 | 0.503139 | 0.501926 ± 0.001095 |
| 8 | 0.986522 | 0.985749 ± 0.001214 | 0.501331 | 0.500380 ± 0.001062 |

Late statistics use 11 saved snapshots, spaced by 5 M;
standard deviations describe temporal variation, not independent statistical
errors. Luminosity is integrated over the retained angular domain and uses
the demo's full-sphere normalization L_BZ=B₀²Ω_H²/6. Rotation medians exclude
the two angular cells nearest each cap. Radial values are linearly interpolated
from the saved diagnostic grids.

The luminosity approaches its late-time level by roughly 40 M and remains
near L_BZ. Field-line rotation away from the cap walls remains near Ω_H/2.
The final toroidal Hφ profile has a **0.81–0.94%**
relative L2 discrepancy from the demo's leading-order reference
Hφ/B₀=−a sin²θ/8 at the four sampled radii, using the same angular mask.
Rotation shows larger departures near the conducting cap walls; those regions
are visible in the final figure and are not represented by the interior medians.

Across the 2,928 sampled exterior diagnostic cells at t=200 M:

- D²/B²: median **0.030372**, maximum
  **0.085218**; fraction with D²>B²:
  **0.000%**.
- |D·B|/B²: median **0.000987**,
  95th percentile **0.003263**, maximum
  **0.008128**.
- Magnetization σ: median **485.95**; all 2,928 sampled cells have finite σ.
- Active particle counts in saved 150–200 M snapshots range from
  **18,672 to 19,411**.

Relative L2 changes from 190 to 200 M over the exterior diagnostic mask:

| Quantity | Relative change |
|---|---:|
| `Br` | 0.066% |
| `Btheta` | 19.025% |
| `Hphi` | 1.002% |
| `omega` | 2.748% |
| `density` | 38.050% |
| `D2_over_B2` | 1.888% |
| `DdotB_over_B2` | 124.274% |
| `luminosity` | 0.192% |

Local PIC density and weak field components continue to fluctuate. These
changes should not be read as pointwise convergence of every diagnostic merely
because integrated power is steady. For scale, the coordinate-component L2
ratio ‖Bθ‖/‖Br‖ is 0.00112489; this is not an
invariant physical-field ratio.

## Conclusion and scope

Standard interpolation works through the full requested 200 M for this
64×64, 10°-cap demo, with finite evolved state, accepted exterior constraints,
and the expected approximately steady BZ power and field rotation. This run
does not establish cap-size or grid convergence, nor a universal treatment of
coordinate singularities. The interpolation still requires a nonsingular
coordinate domain. No Entity run to 200 M was performed for this comparison.

## Historical artifacts

The original report, comparison scripts, source diff, result records, and raw
arrays are preserved in the ignored repository-local
`research_archive/bz_monopole/` archive. They are absent from a fresh checkout.
The archive includes a checksum inventory and original-location notes. Archived
launch scripts contain historical machine paths and are not supported entry points.

This report describes the pre-refactor run at the commit and working-tree diff
listed above. Later cleanup is verified separately by short-run equivalence and
regression tests; it does not constitute a second 200 M validation run.
