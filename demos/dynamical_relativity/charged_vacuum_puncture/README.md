# Charged electrovacuum puncture

The single-puncture specialization of Zilhão et al.,
[arXiv:1311.6483v2](https://arxiv.org/abs/1311.6483v2), Eqs. (2.5)--(2.13), has
`u=1`, `psi²=(1+M/(2r))²-Q²/(4r²)`, and `W=psi^-2`. Initially the conformal
metric is the identity, `K_ij=0`, `B=0`, `beta=0`, and the chosen precollapsed
lapse is `alpha=W`. The lapse choice is a gauge choice, not the static RN lapse.

PyPIC3D uses Gaussian units: the native electric density is
`D^i=Q*x^i/r³`, its flux is `4*pi*Q`, and the electromagnetic energy density is
`(E²+B²)/(8*pi)`. Do **not** apply the rationalized `1/sqrt(4*pi)` conversion
from the historical JAX_BSSN demo. Empty particle storage bypasses particle
push and deposition while retaining the electromagnetic gravitational sources.

## Evolution

Edit constants in `initial_data.py`, or use the runner's CLI. Defaults are
`M=1`, `Q=.5`, `96³` C nodes on a half-cell offset in `[-12,12]³`, `CFL=.2`,
`nu=.02`, `kappa=.002`, 1+log slicing and the evolving Gamma-driver shift.
The coupled scheme remains second order in time with RK4 for the BSSN step.
The runner shortens the requested CFL timestep once, if necessary, to reach
the endpoint; a restart retains that exact timestep.

`source_filter='binomial'` applies one separable `[1,2,1]/4` pass to
`sqrt(gamma)*(rho,S_i,S_ij)` at each BSSN stage and divides by the raw stage
volume. Native field products precede smoothing, retaining checkerboard
energy. Isolated source edges replicate the nearest sample. The filter never
smooths stored Maxwell fields or BSSN state. It modifies the gravitational
source approximation; the analytic initial metric is not re-solved for the
filtered sources. Both raw and filtered constraints are therefore recorded.

This source-only option is separate from the periodic matched PIC filter:
`current_filter` remains `none`, particle storage must be empty, and combining
the two filters is rejected. Existing matched-filter defaults are unchanged.

BSSN uses Sommerfeld faces. Maxwell uses a width-3, order-4 reference sponge
with attenuation `1e-3`, relaxing toward the initial *densitized* Coulomb field
and zero magnetic field; exterior reference samples stay fixed. The sponge
can affect flux and divergence, so its diagnostics are separate.

## Jetstream commands

Use isolated source copies and the CUDA environment, with both repository
roots on `PYTHONPATH`. All simulation and test execution for this study belongs
on Jetstream. For example, from the staged PyPIC3D repository:

```bash
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
PYTHON=/home/exouser/Documents/pair_injection_20261019/venv/bin/python
# Bare source snapshots also need distribution metadata for openPMD output.
# Keep build tools and metadata local to this campaign, outside the shared venv.
$PYTHON -m pip install --target ../../build_tools setuptools wheel
mkdir -p ../../metadata
PYTHONPATH=../../build_tools $PYTHON -c 'from setuptools.build_meta import prepare_metadata_for_build_wheel; prepare_metadata_for_build_wheel("../../metadata")'
export PYTHONPATH=.:../JAX_BSSN:../../metadata
$PYTHON demos/dynamical_relativity/charged_vacuum_puncture/campaign.py /media/volume/JAXBSSN/charged_puncture_TIMESTAMP
```

The campaign runs focused tests in separate processes, initial-data refinement
at 64³/96³/128³, a small smoke, paired 96³ filtered/unfiltered 10M pilots, a
filtered 128³ pilot, then continuation of the filtered 96³ state to 200M.
`campaign_status.json` distinguishes stage failures from completion. A failed
unfiltered control is retained without preventing the filtered qualification.

Single runs accept `--n`, `--duration`, `--output`, `--restart`, and
`--source-filter none|binomial`. Restart requires the same physical/grid/filter
settings; specify the new absolute end time. Output directories are never
silently overwritten. A restart into an existing directory rejects diagnostic
history extending beyond the checkpoint.

## Outputs and interpretation

- `metadata.json`: units, grid, timestep, parameters, masks and dependency data.
- `diagnostics.csv`: raw/filtered Einstein constraints, native Maxwell
  divergences, gauge extrema, field maxima, short-wavelength measures and
  enclosed cubical flux charges at half-widths 2, 4 and 6.
- `snapshot_*.npz`: all BSSN arrays and common-time C-interpolated electric and
  magnetic densities. These visualization arrays are not restart histories.
- `checkpoint_0.npz`, `checkpoint_1.npz`: rotating atomic full restart files,
  including native fields, previous time levels, synchronized B, sponge rates,
  references and masks, BSSN state/parameters and time. Only the derived Yee
  metric is rebuilt. Checkpoint metadata, not filenames, identifies the latest.
- `status.json` and, on invalid evolution, `failure.json`: exact failed step,
  first nonfinite field/index or geometry validity predicate, last valid step.

The exterior mask is fixed at `2<=coordinate r<=8`; the sponge mask is
`max(abs(x_i))>=half_width-3`. Neither is a measured apparent horizon. Electric
Gauss norms exclude the unresolved puncture; the enclosed flux includes it.
The initial RN horizon has isotropic radius `sqrt(M²-Q²)/2`, only about 1.73
baseline cells. Stability on this grid is not evidence of horizon resolution
or continuum convergence.

`plot.py RUN_DIRECTORY` reads saved outputs to produce constraint, charge and
oscillation histories and radial comparisons with fixed input M,Q. The radial
plots interpolate the axis and assume spherical geometry to estimate areal
radius. The diagonal metric reconstruction additionally assumes stationarity;
the plots do not establish that assumption or fit away discrepancies.

Production retains 51 snapshots at 4M intervals, reusing the pilot's early
snapshots as hard links. The pilot also retains its 10M endpoint. The runner
reserves 5 GiB and estimates snapshot/checkpoint storage before evolving.
Historical output is never deleted to make room.
