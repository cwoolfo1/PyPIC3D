"""Removed solver settings and runner command-line options stay unavailable."""
import sys
import pytest
from PyPIC3D.utilities.parameters import build_static_parameters
from demos.bz_monopole import run_bz_monopole as runner
from demos.bz_monopole.simulation_parameters import build_runtime


@pytest.mark.parametrize('value', [0, 4, True, None])
def test_removed_configuration_key_is_rejected(value):
    with pytest.raises(ValueError, match='geodesic_iterations was removed'):
        build_static_parameters({'geodesic_iterations': value})


@pytest.mark.parametrize('arguments', [
    ['--restart', 'old.npz'], ['--end-time', '1'], ['--mode', 'run'], ['--output', 'data'],
])
def test_runner_rejects_command_line_arguments(monkeypatch, arguments):
    monkeypatch.setattr(sys, 'argv', ['bz', *arguments])
    with pytest.raises(SystemExit, match='takes no arguments'):
        runner.main()


def test_removed_runtime_argument_is_rejected():
    with pytest.raises(TypeError, match='geodesic_iterations'):
        build_runtime(geodesic_iterations=0)
