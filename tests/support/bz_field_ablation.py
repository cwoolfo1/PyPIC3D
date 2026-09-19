"""Controlled field-only BZ ablations; no changes to production numerics.

Run with JAX_PLATFORMS=cpu python -m tests.support.bz_field_ablation.
Each case keeps the existing field time loop, grid, dt, sponge, and initial
monopole. Vacuum isolates field instability from particle noise and injection;
passing this experiment is not a plasma BZ validation.
"""
from contextlib import ExitStack, contextmanager
from pathlib import Path
import math
import time
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from demos.bz_monopole import run_bz_monopole as runner
from demos.bz_monopole.simulation_parameters import SimulationParameters, build_runtime
from demos.bz_monopole.plasma_injector import empty_particles
from PyPIC3D.solvers.gr_static import static_metric as solver
from PyPIC3D.solvers.gr_static import time_loop as loop


@contextmanager
def averaging(method):
    """Use main's gather on the same polar geometry, isolating interpolation.

    original_regular only removes singular unused gather values: an unchanged
    location is an identity; a zero determinant at a different location returns
    zero. This is an experimental extension, not a proven general axis limit.
    """
    with ExitStack() as stack:
        if method.startswith('original'):
            if method == 'original_regular':
                def gather(field, source, target, source_location, target_location):
                    if source_location == target_location:
                        return field
                    root = target.sqrt_gamma
                    numerator = solver._location_interpolate(
                        source.sqrt_gamma * field, source_location, target_location)
                    return jnp.where(root != 0, numerator / jnp.where(root != 0, root, 1.), 0.)
                stack.enter_context(patch.object(solver, '_metric_weighted_interpolate', gather))
            for name in ('compute_covariant_E', 'compute_covariant_H'):
                original = getattr(solver, name)
                def auxiliary(D, B, metric, interpolation='physical', original=original):
                    return original(D, B, metric._replace(geometry=None))
                for module in (solver, loop, runner):
                    stack.enter_context(patch.object(module, name, auxiliary))
        yield


def vacuum_case(output, method='entity', horizon_cells=5, end_time=50., nr=64, ntheta=64):
    """Run the actual production time loop with zero-capacity particle arrays."""
    output = runner.prepare_output_directory(output)
    p = SimulationParameters(nr=nr, ntheta=ntheta, pairs_per_cell=1, capacity_factor=0, backend='cpu',
                             end_time=end_time, vacuum=True, current_filter_passes=0,
                             horizon_field_cells=horizon_cells,
                             field_interpolation=method if method in ('entity', 'physical') else 'physical')
    print(f'Building {method}, inner={horizon_cells}, {nr}x{ntheta}', flush=True)
    static, dynamic, metric, _ = build_runtime(
        p, horizon_field_cells=horizon_cells, field_interpolation=p.field_interpolation,
        particle_coordinates=p.particle_coordinates)
    particles, species = empty_particles(p, static)
    dt = float(dynamic.dt)
    with averaging(method):
        fields, background = runner.initialize_fields(p, static, dynamic, metric)
        measure = jax.jit(lambda fs: runner.constraint_residuals(
            particles, species, fs, static, dynamic, p))
        def step(_, fs):
            _, fs = loop.time_loop_static_metric(particles, species, fs, static, dynamic)
            return runner.apply_sponge(fs, background, p, static, dynamic)
        chunk = jax.jit(lambda fs, count: jax.lax.fori_loop(0, count, step, fs))
        target = math.ceil(end_time/dt)
        interval = max(1, round(1./dt))
        records = []
        def record(index):
            values = {key: float(value) for key, value in measure(fields).items()}
            finite = all(np.isfinite(np.asarray(a)).all() for a in (*fields[0], *fields[1]))
            maximum = max(float(jnp.max(jnp.abs(a))) for a in (*fields[0], *fields[1])) / p.B0
            records.append((index*dt, finite, maximum, values['gauss'], values['magnetic_divergence'],
                            values['horizon_gauss'], values['boundary_gauss']))
            return finite and maximum < 1e6
        healthy = record(0)
        start = time.monotonic()
        index = 0
        with tqdm(total=target, desc=f'{method} inner={horizon_cells}', mininterval=1., unit='step') as bar:
            while index < target and healthy:
                count = min(interval, target-index)
                fields = chunk(fields, count)
                jax.block_until_ready(fields)
                index += count
                healthy = record(index)
                bar.update(count)
                bar.set_postfix(time=index*dt, amplitude=records[-1][2], refresh=False)
        metadata = runner.output_metadata(p, static, dynamic)
        metadata.update(experiment_method=np.asarray(method), vacuum=np.asarray(True))
        np.savez(output/'history.npz', measurements=np.asarray(records),
                 columns=np.asarray(['time', 'finite', 'max_field_over_B0', 'gauss', 'divB',
                                     'horizon_gauss', 'boundary_gauss']),
                 wall_seconds=time.monotonic()-start, **metadata)
        np.savez(output/'fields.npz', **{f'{name}_{i}': np.asarray(a)
                 for name, vector in zip(('D','B'),fields[:2]) for i,a in enumerate(vector)},
                 actual_time=index*dt, **metadata)
        print(f'{method} inner={horizon_cells}: final={records[-1]}', flush=True)
    return np.asarray(records)


if __name__ == '__main__':
    root = Path('demos/bz_monopole/runs/field_ablation')
    for method, cells in [('entity', 5), ('physical', 5),
                           ('original_regular', 5), ('entity', 0), ('physical', 0),
                           ('original_regular', 0)]:
        vacuum_case(root/f'{method}_inner{cells}', method, cells)
