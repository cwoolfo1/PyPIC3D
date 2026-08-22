# Inward FMR annular electromagnetic wave

This demo evolves a source-free, z-invariant TMz electromagnetic wave packet
on the production Yee FMR timestep. The initial electric field is an annulus,

```text
Ez(r) = exp[-(r - r0)^2 / (2 sigma^2)] cos[2 pi (r - r0) / wavelength],
```

with `r0 = 0.62`, `sigma = 0.08`, and `wavelength = 0.16`. Its azimuthal
magnetic field has `Bphi = Ez / c`; this sign selects inward propagation. The
pulse begins outside the origin-centered fine patch, crosses the coarse-fine
interface, focuses near the origin, and then starts to expand again.

From the repository root, run:

```bash
python demos/fmr_annular_wave_2d/run_fmr_annular_wave.py
```

For a short headless smoke run without writing into the repository:

```bash
python demos/fmr_annular_wave_2d/run_fmr_annular_wave.py --steps 8 --output-dir /tmp/fmr-annular
```

The runner resolves its configuration and output directory relative to its own
location, so it can also be invoked by its path from another working directory.
It writes 55 output times to:

```text
demos/fmr_annular_wave_2d/data/
```

## View in VisIt

Open `data/fields.visit`, add a Pseudocolor plot of `E/z`, and apply a Slice
operator normal to z. Add a Mesh plot or display the block boundaries to show
the square fine patch from `x,y = -0.375` to `0.375`, then play the time slider.

Every saved time contains the full root block and the interior fine block. The
blocks overlap spatially: the collection aligns them but does not remove root
cells covered by the fine patch. Use a block selection if you want to inspect
one resolution separately.

The per-patch ParaView helpers are also written as
`fields_level_00_patch_000.pmd` and `fields_level_01_patch_000.pmd`.
