"""Explicit-only configuration and portable checkpoint compatibility."""
import json
import sys

import jax
import numpy as np
import pytest

from PyPIC3D.utilities.parameters import build_static_parameters
from demos.bz_monopole import run_bz_monopole as runner
from demos.bz_monopole.plasma_injector import empty_particles
from demos.bz_monopole.simulation_parameters import (
    PARTICLE_INTEGRATOR, SimulationParameters, build_runtime,
)


@pytest.fixture(scope='module', params=['native', 'cartesian'])
def checkpoint(request, tmp_path_factory):
    p = SimulationParameters(nr=16, ntheta=16, devices=1, r_max=4., sponge_start=3.)
    s, d, m, report = build_runtime(p, particle_coordinates=request.param)
    particles, _ = empty_particles(p, s)
    fields, _ = runner.initialize_fields(p, s, d, m)
    key = jax.random.PRNGKey(17)
    path = tmp_path_factory.mktemp(request.param)/'checkpoint.npz'
    runner.save_checkpoint(path, particles, fields, key, 7, p, report)
    with np.load(path, allow_pickle=False) as data:
        arrays = dict(data)
    return p, s, d, particles, fields, key, report, arrays


def legacy_arrays(checkpoint, changes):
    *_, report, arrays = checkpoint
    metadata = dict(report)
    metadata.pop('particle_integrator')
    metadata.update(changes)
    return dict(arrays, run_metadata=np.asarray(json.dumps(metadata)))


@pytest.mark.parametrize('kind', ['missing', 'zero', 'legacy_solver', 'new', 'consistent_mixed'])
def test_explicit_checkpoint_restarts(checkpoint, tmp_path, kind):
    p, s, d, particles, fields, key, _, _ = checkpoint
    changes = {}
    if kind in ('zero', 'legacy_solver', 'consistent_mixed'):
        changes['geodesic_iterations'] = 0
    if kind in ('legacy_solver', 'consistent_mixed'):
        changes['geodesic_nonlinear_solver'] = (
            'cartesian_explicit_midpoint_v1' if s.particle_coordinates == 'cartesian'
            else 'explicit_midpoint')
    if kind in ('new', 'consistent_mixed'):
        changes['particle_integrator'] = PARTICLE_INTEGRATOR
    path = tmp_path/'legacy.npz'
    np.savez(path, **legacy_arrays(checkpoint, changes))
    restored = runner.load_checkpoint(path, particles, fields, p, s, expected_dt=d.dt)
    assert restored[3] == 7
    for actual, expected in zip(jax.tree.leaves(restored[:3]),
                                jax.tree.leaves((particles, fields, key))):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('changes', [
    {'geodesic_iterations': 4},
    {'geodesic_iterations': -1},
    {'geodesic_iterations': False},
    {'geodesic_iterations': 0.0},
    {'geodesic_iterations': None},
    {'geodesic_nonlinear_solver': 'cartesian_implicit_midpoint_v1'},
    {'geodesic_nonlinear_solver': 'picard_newton_cyclic_chart_retry_v4'},
    {'geodesic_nonlinear_solver': 'unknown'},
    {'geodesic_nonlinear_solver': None},
    {'particle_integrator': 'unknown'},
    {'particle_integrator': None},
    {'particle_integrator': PARTICLE_INTEGRATOR, 'geodesic_iterations': 4},
    {'geodesic_iterations': 0, 'geodesic_nonlinear_solver': 'cartesian_implicit_midpoint_v1'},
])
def test_incompatible_checkpoint_is_rejected(checkpoint, tmp_path, changes):
    p, s, d, particles, fields, _, _, _ = checkpoint
    path = tmp_path/'incompatible.npz'
    np.savez(path, **legacy_arrays(checkpoint, changes))
    with pytest.raises(ValueError, match='particle integrator'):
        runner.load_checkpoint(path, particles, fields, p, s, expected_dt=d.dt)


def test_legacy_solver_must_match_chart(checkpoint, tmp_path):
    p, s, d, particles, fields, _, _, _ = checkpoint
    other = ('explicit_midpoint' if s.particle_coordinates == 'cartesian'
             else 'cartesian_explicit_midpoint_v1')
    path = tmp_path/'conflicting.npz'
    np.savez(path, **legacy_arrays(checkpoint, {'geodesic_nonlinear_solver': other}))
    with pytest.raises(ValueError, match='coordinate chart'):
        runner.load_checkpoint(path, particles, fields, p, s, expected_dt=d.dt)


def test_writer_emits_only_explicit_metadata(checkpoint, tmp_path):
    p, s, d, particles, fields, key, report, _ = checkpoint
    assert report['particle_integrator'] == PARTICLE_INTEGRATOR
    assert 'geodesic_iterations' not in report
    assert 'geodesic_nonlinear_solver' not in report
    legacy = dict(report, geodesic_iterations=0,
                  geodesic_nonlinear_solver=('cartesian_explicit_midpoint_v1'
                      if s.particle_coordinates == 'cartesian' else 'explicit_midpoint'))
    path = tmp_path/'new.npz'
    runner.save_checkpoint(path, particles, fields, key, 7, p, legacy)
    assert 'geodesic_iterations' in legacy  # Do not mutate the caller's metadata.
    with np.load(path, allow_pickle=False) as data:
        assert int(data['checkpoint_version']) == 3
        metadata = json.loads(str(data['run_metadata']))
    assert metadata['particle_integrator'] == PARTICLE_INTEGRATOR
    assert 'geodesic_iterations' not in metadata
    assert 'geodesic_nonlinear_solver' not in metadata
    assert runner.load_checkpoint(path, particles, fields, p, s, expected_dt=d.dt)[3] == 7
    with pytest.raises(ValueError, match='particle integrator'):
        runner.save_checkpoint(tmp_path/'bad.npz', particles, fields, key, 7, p,
                               dict(report, geodesic_iterations=4))
    assert not (tmp_path/'bad.npz').exists()


@pytest.mark.parametrize('value', [0, 4, True, None])
def test_removed_configuration_key_is_rejected(value):
    with pytest.raises(ValueError, match='geodesic_iterations was removed'):
        build_static_parameters({'geodesic_iterations': value})


def test_removed_cli_option_is_rejected(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['bz', '--geodesic-iterations', '0'])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert 'unrecognized arguments: --geodesic-iterations 0' in capsys.readouterr().err


def test_removed_runtime_argument_is_rejected():
    with pytest.raises(TypeError, match='geodesic_iterations'):
        build_runtime(geodesic_iterations=0)
