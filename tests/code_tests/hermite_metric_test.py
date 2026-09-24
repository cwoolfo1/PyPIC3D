"""Independent checks of the supplied-grid Hermite reconstruction."""

import unittest
import jax
import jax.numpy as jnp
import numpy as np
from PyPIC3D.relativity.interpolate_metric import interpolate_hermite, interpolate_metric
from tests.code_tests.particle_metric_consistency_test import manufactured

jax.config.update("jax_enable_x64", True)


class TestHermite(unittest.TestCase):
    def test_tensor_quadratic_reproduction(self):
        grid = tuple(jnp.arange(-3, 14, dtype=float) * h for h in (0.1, 0.13, 0.17))

        def f(p):
            x, y, z = jnp.moveaxis(p, -1, 0)
            return (
                1
                + 0.3 * x
                - 0.2 * y
                + 0.1 * z
                + x * x
                + 2 * y * y
                + 0.4 * z * z
                + 0.7 * x * y * z
                + 0.8 * x * x * y * y
            )

        values = f(jnp.stack(jnp.meshgrid(*grid, indexing="ij"), axis=-1))
        q = jnp.array([[0.247, 0.381, 0.418], [0.41, 0.53, 0.67]])
        get = lambda q: interpolate_hermite(
            values, q, grid, (True, True, True), (3, 3, 3)
        )
        np.testing.assert_allclose(get(q), f(q), atol=3e-14, rtol=3e-14)
        for k in range(3):
            direction = jnp.broadcast_to(jnp.eye(3)[k], q.shape)
            np.testing.assert_allclose(
                jax.jvp(get, (q,), (direction,))[1],
                jax.jvp(f, (q,), (direction,))[1],
                atol=1e-12,
            )

    def test_nodal_values_and_shared_slopes(self):
        x = jnp.arange(-3, 13, dtype=float) * 0.125
        grid = (x, jnp.arange(7, dtype=float), jnp.arange(7, dtype=float))
        f = jnp.sin(x) + 0.3 * jnp.cos(3 * x)
        data = jnp.broadcast_to(f[:, None, None], (len(x), 7, 7))
        get = lambda t: interpolate_hermite(
            data, jnp.array([t, 3.0, 3.0]), grid, (True, False, False), (3, 3, 3)
        )
        derivative = jax.grad(get)
        for i in range(2, len(x) - 3):
            self.assertAlmostEqual(float(get(x[i])), float(f[i]), places=13)
            expected = (f[i + 1] - f[i - 1]) / (2 * 0.125)
            self.assertLess(abs(float(derivative(x[i]) - expected)), 1e-12)
            self.assertLess(
                abs(float(derivative(x[i] - 1e-10) - derivative(x[i] + 1e-10))), 1e-8
            )

    def test_invalid_stencil_is_reported(self):
        grid, m = manufactured()
        q = jnp.array([[-1.0, 0.3, 3.0]])
        a = interpolate_metric(m, q, grid, "numerical", (True, True, False), (3, 3, 3))
        self.assertTrue(np.isnan(np.asarray(a.gamma)).all())


if __name__ == "__main__":
    unittest.main()
