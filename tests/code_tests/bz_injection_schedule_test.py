"""Physical injection events retain their random stream under timestep refinement."""
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import jax
import numpy as np
from demos.bz_monopole import run_bz_monopole as runner

class TestInjectionSchedule(unittest.TestCase):
    def draw_events(self, dt, policy='event_ordinal_v1'):
        p = SimpleNamespace(injection_interval=.1, devices=1)
        fields = (None, None, None, None, None, None, None, None, False)
        boundary = SimpleNamespace(invalid_push=False)
        errors = SimpleNamespace(throw=lambda: None)
        draws = []
        def inject(pts, species, mag, D, B, metric, static, dynamic, params, key, event):
            key, event_key = jax.random.split(key)
            draws.append(np.asarray(jax.random.uniform(jax.random.fold_in(event_key, event), (8,))))
            return pts, key, SimpleNamespace(errors=(), requested=np.zeros(1),
                                            inserted=np.zeros(1), rejected=np.zeros(1))
        with ExitStack() as stack:
            for name, replacement in (
                ('inject_pairs', inject), ('measure_magnetization', lambda *a: None),
                ('finite_state', lambda *a: True), ('check_sharding', lambda *a: None),
                ('time_loop_static_metric', lambda pts, *a, **kw: (errors, (pts, fields, boundary)))):
                stack.enter_context(patch.object(runner, name, replacement))
            stack.enter_context(patch.object(runner.jax, 'jit', lambda f: f))
            step = runner.make_step(p, None, None, SimpleNamespace(dt=dt), None,
                                    sponge=False, injection_rng_policy=policy)
            key = jax.random.PRNGKey(17)
            for index in range(round(.3/dt)):
                _, _, key, _ = step(None, fields, key, index)
        return np.array(draws)

    def test_matched_physical_events_share_random_candidates(self):
        np.testing.assert_array_equal(self.draw_events(.002), self.draw_events(.001))
        # Preserve the old stream when restarting checkpoints without a policy.
        old = self.draw_events(.002, 'step_v0')
        self.assertFalse(np.array_equal(old[1:], self.draw_events(.001, 'step_v0')[1:]))


