# WarpX thermal-plasma comparison

These manually invoked scripts compare PyPIC3D's production field gathering and
electrostatic gradient with WarpX. They are not imported by the automated test
suite, and WarpX is not a PyPIC3D runtime or test dependency.

The common initial state comes from WarpX's seed-1
`energy_conserving_thermal_plasma` test: periodic 1D and 2D domains, eight cells per
active axis, 100 eV electrons and protons, density `1e30 m^-3`, and domain length
`10 c / omega_pe`. The saved NPZ files contain SI positions, proper velocities
`gamma*v`, and physical macroparticle weights. Both codes load precisely these
arrays; no random particles are redrawn for a solver or timestep.

The campaign uses relativistic Boris, no filtering, quadratic or linear particle
shapes, and Esirkepov current deposition for the electromagnetic solver. The
electrostatic runner uses the production-selected grid and
`electrostatic_gradient` dispatcher. Its native Poisson solve is instrumented to
record residuals and iteration limits without changing the solve or trajectory.

The primary energy is native-field electric plus magnetic energy and stable
relativistic kinetic energy with particle velocities synchronized to the field
time. Diagnostic half pushes operate on copies. The actual initial total is the
denominator. Raw staggered energy, the historical diagnostic, and the modified
Yee field-energy expression remain separate diagnostic columns.

Run from the repository root with the recorded local environments, or substitute
equivalent environments and pass the WarpX source location with `--source`:

```bash
energy_source=runs/warpx_energy_comparison/20261026T194051Z
energy_repeat=runs/warpx_energy_comparison/reproduce_$(date -u +%Y%m%dT%H%M%SZ)_galerkin
mkdir -p "$energy_repeat/initial"

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/pypic-galerkin-mpl
energy_warpx_python=/home/christopherwoolford/.conda/envs/warpx-clean/bin/python
energy_pypic_python=/home/christopherwoolford/.conda/envs/num_rel/bin/python
energy_scripts=tests/physics_tests/warpx_energy_comparison

if [ -d "$energy_source/initial" ]; then
    cp "$energy_source"/initial/* "$energy_repeat/initial/"
else
    # Export once per geometry, then reuse exactly these particles for every
    # shape, solver, and timestep.
    for energy_dim in 1 2; do
        "$energy_warpx_python" "$energy_scripts/run_warpx.py" \
            --root "$energy_repeat" --mode export --dim "$energy_dim" --shape 2
    done
fi

for energy_shape in 1 2; do
    energy_timesteps="0.2 0.1"
    if [ "$energy_shape" = 1 ]; then
        energy_timesteps="0.2 0.1 0.05"
    fi
    for energy_dim in 1 2; do
        for energy_solver in es em; do
            for energy_dt in $energy_timesteps; do
                "$energy_warpx_python" "$energy_scripts/run_warpx.py" \
                    --root "$energy_repeat" --dim "$energy_dim" \
                    --solver "$energy_solver" --shape "$energy_shape" \
                    --dt-wpe "$energy_dt"
            done
        done
    done
done

# No case arguments runs all 20 matched Galerkin cases to omega_pe*t = 100.
"$energy_pypic_python" "$energy_scripts/run_pypic3d.py" --root "$energy_repeat"

# Explicit uniform gathering should reproduce the archived production method.
for energy_dim in 1 2; do
    for energy_solver in es esirkepov; do
        "$energy_pypic_python" "$energy_scripts/run_pypic3d.py" \
            --root "$energy_repeat" --dimension "$energy_dim" \
            --solver "$energy_solver" --shape 2 --gather uniform --dt-wpe 0.2
    done
done
if [ -f "$energy_source/pypic3d/1d_es_dt0.2/history.csv" ]; then
    "$energy_pypic_python" "$energy_scripts/analyze_results.py" \
        --root "$energy_repeat" --legacy-root "$energy_source"
else
    "$energy_pypic_python" "$energy_scripts/analyze_results.py" --root "$energy_repeat"
fi
```

Each run saves its input, log, precision, environment, initial-state hash, method
selection, and energy history. PyPIC3D manifests also save production source
hashes. Existing completed cases are preserved; use a fresh root after changing
code or parameters. The original archive, when available, is read-only input to
this workflow. Historical-history comparisons are optional; fresh exports and
current uniform controls do not require the ignored prior campaign directory.

Validation compares every synchronized total-energy sample: maximum difference
divided by WarpX's actual initial energy must be below `1e-8` for electrostatic
and `1e-10` for electromagnetic runs. Quadratic cases must also remain below
`0.003` maximum relative error, with the half-timestep maximum less than `0.30`
times the coarse maximum. Linear cases are reported empirically at all three
timesteps: their lowest-order component gather is discontinuous, so the quadratic
refinement gate does not apply. When an archive is supplied, uniform controls
must match its histories within `1e-10` of the archived initial total.

`summary.csv`, `summary.json`, `checks.json`, and PDF/PNG plots contain the
postprocessed results. These benchmarks test the flat periodic thermal-plasma
case, not exact energy conservation or coverage of every supported solver.

The 2026-10-26 CPU float64 campaign completed 44 runs using WarpX 26.7 and
JAX 0.9.2. Quadratic maximum synchronized energy errors were:

| Case | Uniform, dt wpe = 0.2 | Galerkin, dt wpe = 0.2 | Galerkin, dt wpe = 0.1 |
|---|---:|---:|---:|
| 1D electrostatic | 4.38868% | 0.0221275% | 0.00554209% |
| 2D electrostatic | 373.131% | 0.0120880% | 0.00300945% |
| 1D Yee–Esirkepov | 1.40708% | 0.0614990% | 0.0152366% |
| 2D Yee–Esirkepov | 1.60382% | 0.0306235% | 0.00751588% |

All 20 matched energy histories met their targets: maximum discrepancy was
`3.217e-10 H0` for electrostatics and `9.499e-13 H0` for electromagnetics.
All four uniform controls reproduced the archived energy histories exactly.
Linear shapes showed nonmonotonic refinement in some cases in both codes;
the quadratic rate is not a valid blanket expectation for their discontinuous
component gathers. Raw histories, every-step and upstream-cadence summaries,
plots, source hashes, and validation records are retained locally under
`runs/warpx_energy_comparison/20261026T205451Z_galerkin/`.

GR tests separately cover physical-field reconstruction and propagation of the
selection through initialization, synchronization, metric rephasing, and both
PIC half-steps. The charged curved-chart probe and its limitations are documented
in `docs/particles.rst`; these flat thermal-plasma numbers are not GR energy
conservation measurements. Production energy diagnostics and temporal E/J
centering were not changed.
