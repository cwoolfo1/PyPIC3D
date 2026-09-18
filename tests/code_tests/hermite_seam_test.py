"""Metric halo width supports the existing migration/capacity contract."""

import pytest
from tests.kernel_fixtures import kernel_parameters
from PyPIC3D.utilities.parameters import build_static_parameters


def test_two_guard_hybrid_configuration_is_rejected():
    s, _ = kernel_parameters(
        solver="static_metric", particle_pusher="hybrid_boris_geodesic"
    )
    config = dict(s._asdict(), guard_cells=2)
    with pytest.raises(ValueError, match="guard_cells >= 3"):
        build_static_parameters(config)
