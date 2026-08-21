# PyPIC3D scaling benchmark

Run the complete benchmark from the repository root:

```bash
python benchmarks/scaling_benchmark.py --output benchmarks/results
```

The runner creates a clean process per case, forces the requested number of
JAX host devices before backend initialization, performs one unmeasured
compile/warm-up call, and reports the median of five synchronized calls. It
produces:

* `timings.csv`: raw, machine-readable measurements;
* `metadata.json`: interpreter, platform, CPU count, and invocation;
* `problem_size_scaling.png`: fixed-grid particle and fixed-particle cell scans;
* `core_scaling.png`: strong and weak CPU-device scaling.

Use longer runs for publication-quality results, for example:

```bash
python benchmarks/scaling_benchmark.py \
  --particle-counts 1024 4096 16384 65536 \
  --cell-counts 32 64 128 256 \
  --core-counts 1 2 4 8 \
  --fixed-cells 256 --fixed-particles 65536 \
  --weak-cells-per-core 64 --weak-particles-per-core 16384 \
  --repeats 10 --output benchmarks/results
```

## Reading the component columns

`step_s` is the authoritative end-to-end steady-state time. The other columns
are independently compiled diagnostic kernels:

* `particle_push_s`: field interpolation and Boris velocity update;
* `deposition_retile_s`: two half position updates, tile refreshes, and direct
  current deposition;
* `field_evolution_s`: the B/E/B Yee sequence, including its required halo
  refreshes and filters;
* `one_ghost_exchange_s`: one three-component tiled halo refresh.

The ghost measurement is a **subset diagnostic**, not an additional mutually
exclusive phase: deposition and field evolution invoke halo operations
internally. Consequently, component values must not be summed to reconstruct
`step_s`. This design avoids modifying production kernels merely to insert
timers and makes JAX synchronization explicit with `block_until_ready`.

## Analysis checklist

1. Particle scaling should be approximately linear once launch overhead is
   amortized. Super-linear growth points to scatter pressure in deposition or
   excessive fixed tile capacity.
2. Cell scaling isolates the Yee curls, filters, and halo surface/volume cost.
3. Strong-scaling efficiency is `T(1)/(p*T(p))`; flattening indicates launch,
   communication, or memory-bandwidth saturation.
4. Weak scaling is ideal when time remains constant as cells and particles per
   device remain fixed.
5. Compare the component columns at every size. The direct deposition kernel
   performs repeated indexed scatters for every shape-stencil combination;
   particle push performs six field interpolations; and every field update
   performs multiple halo refreshes. These are the principal expected
   bottlenecks to validate on the target system.

CPU-device counts are logical JAX devices, not MPI ranks. Do not oversubscribe:
request only cores physically allocated by the scheduler. GPU scaling requires
a separate launcher and device allocation; this runner intentionally measures
the CPU path.
