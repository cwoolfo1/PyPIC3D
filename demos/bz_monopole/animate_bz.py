"""Animate measured Figure-6 diagnostics from one explicitly selected run directory.

CPU only. Fixed limits include every selected finite profile value; no angular
cut or temporal interpolation is applied. The GIF is a visualization, not an
acceptance result. Its JSON sidecar records the exact snapshots and limits.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .plot_entity_bz import (
    Normalization, load_snapshot, make_figure, radial_profile, snapshot_catalog,
)


def padded_limits(values):
    finite = np.concatenate([np.asarray(v).ravel() for v in values])
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError('No finite values for a profile panel.')
    lo, hi = float(finite.min()), float(finite.max())
    pad = .05*max(hi-lo, abs(hi)*.01, 1e-6)
    return [lo-pad, hi+pad]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--times', type=float, nargs='+', required=True,
                        help='Exact saved times, in increasing order; no nearest-time substitution')
    parser.add_argument('--output', type=Path, required=True, help='New .gif file')
    parser.add_argument('--fps', type=float, default=2.)
    parser.add_argument('--dpi', type=int, default=100)
    args = parser.parse_args()
    if args.output.suffix.lower() != '.gif' or not np.isfinite(args.fps) or not 0 < args.fps <= 50:
        parser.error('Use a .gif output and 0 < fps <= 50.')
    if args.dpi <= 0 or not np.isfinite(args.times).all() or any(b <= a for a, b in zip(args.times, args.times[1:])):
        parser.error('DPI must be positive; times must be finite and strictly increasing.')
    outputs = [args.output, args.output.with_suffix('.png'), args.output.with_suffix('.json')]
    if any(p.exists() for p in outputs):
        parser.error('An output or sidecar already exists; choose a fresh output name.')
    catalog = snapshot_catalog(args.run)
    manifest = json.loads((args.run/'manifest.json').read_text())
    selected = []
    for requested in args.times:
        matches = [(t, p) for t, p in catalog if abs(t-requested) < 1e-9]
        if len(matches) != 1:
            parser.error(f'No unique saved snapshot at exactly t={requested:g}.')
        selected.append(matches[0])
    norm = Normalization.from_manifest(manifest)
    radii = (2., 3., 4., 5.)
    data = [load_snapshot(path) for _, path in selected]
    h, omega, power, map_max = [[-norm.spin/8, 0.]], [[0., .5]], [[0., 1.]], .03
    for row in data:
        h.extend(radial_profile(row['r'], row['Hphi']/norm.B0, r) for r in radii)
        omega.extend(radial_profile(row['r'], row['omega']/norm.omega_h, r) for r in radii)
        exterior = row['r'] >= norm.horizon
        power.append(row['luminosity'][exterior]/norm.luminosity_bz)
        signal = -row['Hphi'][exterior]/norm.B0
        positive = signal[np.isfinite(signal) & (signal > 0)]
        if positive.size:
            map_max = max(map_max, float(positive.max()))
    limits = dict(Hphi=padded_limits(h), omega=padded_limits(omega), luminosity=padded_limits(power))
    from PIL import Image
    import matplotlib.pyplot as plt
    frames, provenance = [], []
    for (time, path), row in zip(selected, data):
        fig, notes = make_figure(row, norm, radii=radii)
        fig.set_dpi(args.dpi)
        fig.axes[0].collections[0].set_clim(1e-4, map_max)
        for axis, bounds in zip(fig.axes[1:4], limits.values()):
            axis.set_ylim(bounds)
        p = manifest['parameters']
        fig.suptitle(f'Measured BZ monopole: t={time:g} M   '
                     f'{p["nr"]}×{p["ntheta"]}, d0={p["skin_depth"]:g} M, '
                     f'{p["pairs_per_cell"]} nominal pairs/cell', fontsize=12)
        fig.supxlabel('Saved states; fixed scales; no temporal interpolation. '
                      'Gray map cells: nonpositive/undefined signal. Validation remains separate.', fontsize=8)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB'))
        plt.close(fig)
        provenance.append(dict(time=time, snapshot=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), notes=notes))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    duration = max(20, round(1000/args.fps/10)*10)
    frames[0].save(args.output, save_all=True, append_images=frames[1:],
                   duration=[duration]*(len(frames)-1)+[duration+2000], loop=0)
    frames[-1].save(outputs[1])
    report = dict(run=str(args.run.resolve()), frames=provenance,
        parameters=manifest['parameters'], normalization=asdict(norm),
        fixed_profile_limits=limits, fixed_map_limits=[1e-4, map_max],
        frame_duration_ms=duration, final_hold_extra_ms=2000,
        note='Each measured state gets equal playback time; simulation-time gaps may differ. '
             'No interpolation or angular exclusion. This is not an acceptance report.')
    outputs[2].write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'Saved {len(frames)} measured frames: {args.output}')


if __name__ == '__main__':
    main()
