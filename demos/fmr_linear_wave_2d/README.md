# Single-level FMR 2D linear wave

This demo evolves a source-free electromagnetic plane wave with

```text
Ez = cos(2 pi x + 2 pi y)
Bx =  cos(2 pi x + 2 pi y) / sqrt(2)
By = -cos(2 pi x + 2 pi y) / sqrt(2)
```

on the production Yee FMR timestep. The fields depend on `x` and `y` and are
constant in `z`. A five-cell z slab keeps the fine patch strictly interior and
provides the four fine-owned donors required by the fixed fourth-order FMR
transfer along every stored axis.

From the repository root, run:

```bash
python demos/fmr_linear_wave_2d/run_fmr_linear_wave.py
```

The runner always resolves its configuration and output location relative to
this directory. Open the generated collection in VisIt:

```text
demos/fmr_linear_wave_2d/data/fields.visit
```

The collection has two blocks at every saved time: the full root grid and the
interior fine grid. Plot the `E/z` component and apply a Slice operator normal
to z to see the x-y wave. The root and fine blocks overlap spatially; this
collection aligns the two domains but does not remove root cells covered by the
fine patch.

The per-patch ParaView helpers are also written in `data/` as
`fields_level_00_patch_000.pmd` and `fields_level_01_patch_000.pmd`.
