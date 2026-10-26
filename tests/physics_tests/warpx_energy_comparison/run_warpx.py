#!/usr/bin/env python3
"""Run the native WarpX energy test or a matched thermal-plasma comparison.

Use a separate process for each geometry. Initial NPZ arrays contain SI positions,
proper velocities gamma*v (m/s), and physical particle weights. No PyPIC3D solver
or diagnostic is changed by this script.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np


SOURCE = Path('/home/christopherwoolford/Documents/Research/scratch/warpx_pic/warpx')
CONSTANTS = dict(c=299792458.0, epsilon0=8.8541878188e-12,
                 mu0=1.2566370612685e-6, e=1.602176634e-19,
                 m_e=9.1093837139e-31, m_p=1.67262192595e-27)
N0 = 1.e30
WPE = CONSTANTS['e'] * np.sqrt(N0 / (CONSTANTS['m_e'] * CONSTANTS['epsilon0']))
DE0 = CONSTANTS['c'] / WPE


def input_text(dim, mode, solver, dt_wpe, t_end_wpe, sync_interval, gather, galerkin_scheme, shape, source_root):
    source = source_root / 'Examples/Tests/energy_conserving_thermal_plasma' / f'inputs_test_{dim}d_energy_conserving_thermal_plasma'
    text = source.read_text()
    if mode == 'native':
        return source, text

    # Native loading is retained for export. Matched runs load the exact exported
    # arrays before the initial field solve and before the baseline diagnostics.
    if mode == 'matched':
        lines = []
        for line in text.splitlines():
            if re.match(r'^(electrons|protons)\.', line) and '.species_type' not in line:
                continue
            lines.append(line)
        text = '\n'.join(lines) + '\nelectrons.injection_style = none\nprotons.injection_style = none\n'

    replace = {
        'max_step': 0 if mode == 'export' else round(t_end_wpe / dt_wpe),
        'my_constants.dt': f'{dt_wpe:.17g}/wpe',
        'warpx.do_electrostatic': 'labframe' if solver == 'es' else 'none',
        'warpx.verbose': 0,
        'EP.intervals': sync_interval,
        'EF.intervals': sync_interval,
        'algo.field_gathering': gather,
        'algo.particle_shape': shape,
    }
    for name, value in replace.items():
        text = re.sub(r'^' + re.escape(name) + r'\s*=.*$', f'{name} = {value}', text, flags=re.M)
    # Reduced diagnostics suffice; remove bulky full plotfiles.
    text = '\n'.join(line for line in text.splitlines()
                     if not line.startswith(('diagnostics.', 'diag1.')))
    text += '\nwarpx.random_seed = 1\nwarpx.synchronize_velocity_for_diagnostics = 1\n'
    text += 'algo.particle_pusher = boris\nreduced_diags.precision = 17\n'
    if solver == 'em':
        text += 'algo.maxwell_solver = yee\nalgo.current_deposition = esirkepov\n'
    if galerkin_scheme is not None:
        text += f'interpolation.galerkin_scheme = {galerkin_scheme}\n'
    return source, text


def particle_arrays(dim, libwarpx):
    arrays = {}
    for species in ('electrons', 'protons'):
        pc = libwarpx.warpx.multi_particle_container().get(species)
        count = pc.total_number_of_particles(True, False)
        active_coordinates = ('z',) if dim == 1 else ('x', 'z')
        for name in ('x', 'y', 'z', 'ux', 'uy', 'uz', 'w'):
            if name in ('x', 'y', 'z') and name not in active_coordinates:
                value = np.zeros(count, dtype=np.float64)
            else:
                index = pc.get_real_comp_index(name)
                pieces = [pti.soa().get_real_data(index).to_numpy(copy=True)
                          for pti in libwarpx.libwarpx_so.WarpXParIter(pc, 0)]
                value = np.concatenate(pieces) if pieces else np.empty(0)
            arrays[f'{species}_{name}'] = np.asarray(value, dtype=np.float64)
    return arrays


def particle_summary(arrays):
    summary = {}
    for species, mass in (('electrons', CONSTANTS['m_e']), ('protons', CONSTANTS['m_p'])):
        u = np.stack([arrays[f'{species}_u{axis}'] for axis in 'xyz'], axis=1)
        w = arrays[f'{species}_w']
        u2 = np.sum(u*u, axis=1)
        gamma = np.sqrt(1 + u2 / CONSTANTS['c']**2)
        summary[species] = dict(count=len(w), weight_min=float(w.min()),
                                weight_max=float(w.max()), weight_sum=float(w.sum()),
                                kinetic=float(np.sum(w * mass * u2 / (1 + gamma))))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('export', 'native', 'matched'), default='matched')
    parser.add_argument('--dim', type=int, choices=(1, 2), required=True)
    parser.add_argument('--solver', choices=('es', 'em'), default='es')
    parser.add_argument('--dt-wpe', type=float, default=.2)
    parser.add_argument('--t-end-wpe', type=float, default=100.)
    parser.add_argument('--sync-interval', type=int, default=1)
    parser.add_argument('--gather', choices=('energy-conserving', 'momentum-conserving'),
                        default='energy-conserving')
    parser.add_argument('--galerkin-scheme', type=int, choices=(0, 1))
    parser.add_argument('--shape', type=int, choices=(1, 2), default=2)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    source_root = args.source.resolve()
    label = f'{args.dim}d_{args.solver}_p{args.shape}_dt{args.dt_wpe:g}'
    if args.gather == 'momentum-conserving':
        label += '_momentum'
    elif args.galerkin_scheme == 0:
        label += '_uniform'
    out = args.output or (root/'warpx'/('export_' + str(args.dim) + 'd' if args.mode == 'export' else label))
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'energy.csv').exists():
        raise FileExistsError(f'Preserving existing completed run: {out}')
    start = time.perf_counter()
    # Capture the C++ and Python streams together for reproducible audit records.
    sys.stdout.flush()
    log = (out/'run.log').open('w')
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    os.chdir(out)
    source, contents = input_text(args.dim, args.mode, args.solver, args.dt_wpe,
                                  args.t_end_wpe, args.sync_interval, args.gather,
                                  args.galerkin_scheme, args.shape, source_root)
    (out/'upstream_inputs').write_text(source.read_text())
    (out/'inputs').write_text(contents)

    from pywarpx import callbacks, geometry, libwarpx
    geometry.dims = str(args.dim)
    initial_path = root/'initial'/f'{args.dim}d.npz'
    expected = None
    if args.mode == 'matched':
        with np.load(initial_path) as data:
            expected = {name: data[name].copy() for name in data.files}

        def load_particles():
            for species in ('electrons', 'protons'):
                pc = libwarpx.warpx.multi_particle_container().get(species)
                pc.add_particles(**{name: expected[f'{species}_{name}']
                                    for name in ('x', 'y', 'z', 'ux', 'uy', 'uz', 'w')})

        callbacks.installbeforeInitEsolve(load_particles)

    libwarpx.initialize([sys.executable, str(out/'inputs')])
    arrays = particle_arrays(args.dim, libwarpx)
    expected_count = 32 if args.dim == 1 else 256
    assert all(len(arrays[f'{species}_w']) == expected_count for species in ('electrons', 'protons'))
    # The exported order is the local tile order. Single-box/single-rank insertion
    # preserves it, so compare every loaded component exactly.
    if expected is not None:
        for name in arrays:
            assert np.array_equal(arrays[name], expected[name]), name
    summary = particle_summary(arrays)
    if args.mode == 'export':
        initial_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(initial_path, **arrays)
        meta_path = initial_path.parent/'metadata.json'
        metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        metadata.update(constants=CONSTANTS, n0=N0, temperature_eV=100., omega_pe=float(WPE),
                        skin_depth=float(DE0), length_active=float(10*DE0), seed=1,
                        phase_space_time='t=0; positions and proper velocities synchronized',
                        coordinates='SI metres; 1D z, 2D x,z; inactive coordinates zero',
                        velocity='ux,uy,uz are proper velocities gamma*v in m/s',
                        weights='physical particle counts with unit inactive measures',
                        schema='species_component keys; species electrons/protons; components x,y,z,ux,uy,uz,w')
        metadata[f'{args.dim}d'] = dict(particles=summary,
            sha256=hashlib.sha256(initial_path.read_bytes()).hexdigest(), cells=[8]*args.dim,
            initial_field_energy=0., kinetic0=sum(v['kinetic'] for v in summary.values()))
        meta_path.write_text(json.dumps(metadata, indent=2)+'\n')
    else:
        libwarpx.warpx.evolve(-1)

    field = np.atleast_2d(np.loadtxt(out/'diags/reducedfiles/EF.txt'))
    particle = np.atleast_2d(np.loadtxt(out/'diags/reducedfiles/EP.txt'))
    assert np.array_equal(field[:, 0], particle[:, 0])
    total = field[:, 2] + particle[:, 2]
    history = np.column_stack((field[:, 0], field[:, 1], field[:, 1]*WPE,
                               field[:, 3], field[:, 4], particle[:, 2], total))
    np.savetxt(out/'energy.csv', history, delimiter=',', comments='',
               header='step,time,wpe_t,electric,magnetic,kinetic,total', fmt='%.17e')
    kinetic0 = sum(v['kinetic'] for v in summary.values())
    assert np.isclose(particle[0, 2], kinetic0, rtol=1.e-12, atol=0.)
    error = (total-total[0])/total[0]
    metadata = dict(mode=args.mode, dimension=args.dim, solver=args.solver,
                    package_version=importlib.metadata.version('pywarpx'),
                    compiled_version=libwarpx.__version__, python=sys.executable,
                    source=str(source), constants=CONSTANTS, n0=N0, omega_pe=float(WPE),
                    dt_wpe=args.dt_wpe, t_end_wpe=float(history[-1,2]),
                    diagnostics_interval=100 if args.mode == 'native' else args.sync_interval,
                    field_gathering=args.gather, particle_shape=args.shape,
                    galerkin_scheme=args.galerkin_scheme if args.galerkin_scheme is not None else int(args.gather == 'energy-conserving'),
                    synchronization=True, omp_threads=os.environ.get('OMP_NUM_THREADS'),
                    initial_particles=summary, initial_kinetic=kinetic0, initial_total=float(total[0]),
                    command=sys.argv,
                    initial_array_sha256=hashlib.sha256(initial_path.read_bytes()).hexdigest() if initial_path.exists() else None,
                    final_relative_error=float(error[-1]), max_relative_error=float(np.max(np.abs(error))),
                    rms_relative_error=float(np.sqrt(np.mean(error*error))),
                    elapsed_seconds=time.perf_counter()-start,
                    native_threshold_pass=bool(np.max(np.abs(error)) < .003))
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
