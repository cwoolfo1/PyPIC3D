"""Small standard-metric BZ runtimes, independent of polar geometry tests."""
from functools import lru_cache
import math
from demos.static_metric_relativity.bz_monopole.simulation_parameters import SimulationParameters, build_runtime
from PyPIC3D.relativity.metrics.flat import initialize_flat_spherical_metric
from tests.support.polar_fixtures import particle


@lru_cache(None)
def bz_runtime(devices=1, order=1, flat=False, nt=16):
    p = SimulationParameters(nr=16, ntheta=nt, devices=devices, r_max=4., sponge_start=3.,
            skin_depth=.0025, pairs_per_cell=4, maximum_timestep=None,
            end_time=5., output_interval=1., horizon_field_cells=0,
            theta_start=math.radians(30), theta_end=math.radians(150), backend='cpu')
    s, d, m, _ = build_runtime(p)
    s = s._replace(shape_factor=order)
    if flat:
        s = s._replace(metric='flat_spherical', metric_mass=0., metric_spin=0.)
        m = initialize_flat_spherical_metric(s, d)
    return p, s, d, m
