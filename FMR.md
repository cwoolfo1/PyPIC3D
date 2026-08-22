# Field-only fixed mesh refinement

PyPIC3D supports a deliberately narrow FMR runtime for developing and testing
the electromagnetic interface method. It evolves one root level and one
strictly interior rectangular fine patch on a single logical tile.

## Supported contract

- field-only electrodynamic Yee evolution;
- two levels with a fixed 2:1 refinement ratio;
- one fine patch that does not touch the root boundary;
- fourth-order tensor-product transfers with 64 donors per target;
- component-specific Yee staggering and ownership masks;
- fine-to-coarse shadow restriction followed by coarse-to-fine interface fill;
- explicit forward and backward Yee curls;
- one global `B(dt/2) -> E(dt) -> B(dt/2)` timestep;
- periodic or conducting root boundaries; and
- independent root/fine openPMD patch series with guard stripping and Yee
  positions preserved.

The public runtime entry points are configuration validation/loading, hierarchy
construction, field initialization, E/B synchronization, the field-only
timestep, and the Yee-location constants. Transfer maps, quadrature weights,
coordinate helpers, and explicit curls remain internal implementation details.

Multitile field evolution is rejected when the hierarchy is constructed. The
output adapter already accepts tiled patch layouts so output topology can grow
without changing the on-disk patch-series contract.

## Numerical status

The transfer maps reproduce degree-three polynomials to roundoff and retain
greater than 3.5 observed order on the smooth transfer tests. Ownership,
face/edge/corner ghost coverage, deep-shadow isolation, synchronization
idempotence, explicit curls, and individual B-E-B stages have strict tests.

The TM111 PEC Maxwell convergence gate passes. The periodic plane-wave gate is
intentionally still strict and failing: the verified last-pair fine-region
composite electromagnetic order is `1.6917696665`, below the required `1.8`.
Coarse and interface regions remain above second order. This is known numerical
debt in the fine-region interface coupling, not part of this organizational
cleanup.

## Explicitly unsupported

- multitile evolution;
- particles and deposition on an FMR hierarchy;
- PML and super-Gaussian absorbers;
- external or restart fields;
- patches touching the root boundary;
- multiple patches at one level; and
- more than two levels.

These features should replace the single-tile or one-patch seams directly;
compatibility wrappers around the current internal records are not intended.

## Runnable examples and diagnostics

The minimal reference is:

```bash
python demos/fmr_linear_wave_2d/run_fmr_linear_wave.py
```

The larger interface-crossing example is:

```bash
python demos/fmr_annular_wave_2d/run_fmr_annular_wave.py
```

Both write two ordered openPMD blocks per saved time under their ignored
`data/` directories. The exploratory convergence studies are developer tools,
not tests:

```bash
python demos/fmr_convergence_study.py fixed-cfl
```

The acceptance gates live in `tests/code_tests/` and `tests/physics_tests/`;
shared analytic fields, hierarchy setup, quadrature, and regional norms live in
`tests/fmr_support.py`.
