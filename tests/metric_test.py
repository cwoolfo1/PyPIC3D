import os
import tempfile
import unittest
import numpy as np
import jax
import jax.numpy as jnp

from PyPIC3D.metric import (
    build_metric_from_parameters,
    metric_terms_at_position,
    geodesic_acceleration,
    relativistic_metric_rhs,
    manufactured_geodesic_residual,
)
from PyPIC3D.gr_particle_pusher import relativistic_metric_single_particle


jax.config.update("jax_enable_x64", True)


class TestMetricMethods(unittest.TestCase):
    def test_predefined_metrics(self):
        minkowski = build_metric_from_parameters({"metric": "minkowski"})
        self.assertEqual(int(minkowski["metric_type"]), 0)
        self.assertTrue(jnp.allclose(minkowski["spatial_cov"], jnp.eye(3)))

        cylindrical = build_metric_from_parameters({"metric": "cylindrical"})
        self.assertEqual(int(cylindrical["metric_type"]), 1)
        g_cov, _, gamma, _, _, _ = metric_terms_at_position(2.0, 0.0, 0.0, cylindrical)
        self.assertTrue(jnp.isclose(g_cov[1, 1], 4.0))
        self.assertTrue(jnp.isclose(gamma[0, 1, 1], -2.0))
        self.assertTrue(jnp.isclose(gamma[1, 0, 1], 0.5))
        self.assertTrue(jnp.isclose(gamma[1, 1, 0], 0.5))

    def test_static_metric_from_file(self):
        metric_tensor = np.diag([1.0, 1.5, 2.0])
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            metric_path = tmp.name
        try:
            np.save(metric_path, metric_tensor)
            metric = build_metric_from_parameters({"metric": "static", "metric_file": metric_path})
            self.assertEqual(int(metric["metric_type"]), 2)
            self.assertTrue(jnp.allclose(metric["spatial_cov"], jnp.asarray(metric_tensor)))
            self.assertTrue(jnp.allclose(metric["spatial_cov"] @ metric["spatial_contra"], jnp.eye(3), atol=1e-12))
        finally:
            os.remove(metric_path)

    def test_mms_geodesic_residual_cylindrical(self):
        metric = build_metric_from_parameters({"metric": "cylindrical"})

        # Manufactured trajectory in cylindrical coordinates:
        # r(t)=r0+a*t, phi(t)=omega*t, z(t)=z0+b*t
        # Source is chosen so that residual == 0.
        t = 0.35
        r0 = 2.0
        a = 0.2
        omega = 0.3
        b = -0.1
        r = r0 + a * t

        x = jnp.array([r, omega * t, b * t])
        v = jnp.array([a, omega, b])
        accel = jnp.array([0.0, 0.0, 0.0])

        source = jnp.array([-r * omega**2, 2.0 * a * omega / r, 0.0])
        residual = manufactured_geodesic_residual(v, accel, x, source, metric)
        self.assertTrue(jnp.allclose(residual, jnp.zeros(3), atol=1e-12))

    def test_metric_rhs_and_pusher_manufactured_source(self):
        metric = build_metric_from_parameters({"metric": "cylindrical"})
        constants = {"C": jnp.asarray(1e12)}
        q = 1.0
        m = 1.0

        x = jnp.array([2.0, 0.25, 0.0])
        v = jnp.array([0.1, 0.2, 0.0])
        geo = geodesic_acceleration(v, x, metric)

        # With E chosen equal to geodesic term (and very large C so gamma ~ 1),
        # the manufactured RHS is approximately zero.
        rhs = relativistic_metric_rhs(v, x, geo, jnp.zeros(3), q, m, constants, metric)
        self.assertTrue(jnp.allclose(rhs, jnp.zeros(3), atol=1e-10))

        dt = 1e-4
        newv = relativistic_metric_single_particle(
            v[0], v[1], v[2],
            x[0], x[1], x[2],
            geo[0], geo[1], geo[2],
            0.0, 0.0, 0.0,
            q, m, dt, constants, metric
        )
        newv = jnp.asarray(newv)
        self.assertTrue(jnp.allclose(newv, v, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
