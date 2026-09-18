"""Grid-only metric representation, derivative and polar sampling regressions."""

import unittest
import jax
import jax.numpy as jnp
import numpy as np
from PyPIC3D.relativity.core import Metric
from PyPIC3D.relativity.particle_metric import (
    sample_particle_metric,
    particle_metric_valid,
)
from tests.support.particle_metric_fixtures import (
    make_runtime,
    sample_metric,
    sample_gradient,
    sampled_rotation_checks,
)

jax.config.update("jax_enable_x64", True)


def manufactured(offset=0.0):
    grid = (
        jnp.arange(-3, 12, dtype=float) * 0.1 + offset,
        jnp.arange(-3, 12, dtype=float) * 0.1,
        jnp.arange(7, dtype=float),
    )
    x, y, z = jnp.meshgrid(*grid, indexing="ij")
    shape = x.shape
    g = jnp.broadcast_to(
        jnp.array([[2.0, 0.2, 0.1], [0.2, 3.0, -0.1], [0.1, -0.1, 1.5]]), shape + (3, 3)
    )
    g = (
        g.at[..., 0, 0]
        .add(0.2 * x + 0.1 * y * y)
        .at[..., 0, 1]
        .add(0.03 * x * y)
        .at[..., 1, 0]
        .add(0.03 * x * y)
    )
    lapse = 1.0 + 0.03 * x + 0.02 * y * y
    shift = jnp.stack((0.04 * x * y, 0.02 * y, 0.01 * x), axis=-1)
    return grid, Metric(
        lapse,
        shift,
        g,
        jnp.zeros_like(g),
        jnp.zeros(shape),
        jnp.zeros(shape + (3, 3, 3)),
        jnp.zeros(shape + (3,)),
        jnp.zeros(shape + (3, 3)),
    )


class TestParticleMetricConsistency(unittest.TestCase):
    def test_inverse_and_derivatives(self):
        grid, m = manufactured()
        q = jnp.array([[0.247, 0.381, 3.0], [0.415, 0.274, 3.0]])
        for order in (1, 2):
            evaluate = lambda q: sample_particle_metric(
                m, q, grid, order, "numerical", (True, True, False), (3, 3, 3)
            )
            a, dh = evaluate(q)
            np.testing.assert_allclose(
                a.gamma @ a.gamma_inv,
                jnp.broadcast_to(jnp.eye(3), a.gamma.shape),
                atol=1e-12,
            )
            np.testing.assert_allclose(
                a.sqrt_gamma**2, jnp.linalg.det(a.gamma), rtol=1e-12
            )
            np.testing.assert_allclose(
                jnp.linalg.det(a.gamma) * jnp.linalg.det(a.gamma_inv), 1.0, atol=1e-12
            )
            for k in range(3):
                tangent = jnp.broadcast_to(jnp.eye(3)[k], q.shape)
                actual = jax.jvp(lambda x: evaluate(x)[0].gamma_inv, (q,), (tangent,))[
                    1
                ]
                np.testing.assert_allclose(dh[:, k], actual, atol=1e-12, rtol=1e-12)
                eps = 1e-5
                plus = evaluate(q + eps * tangent)[0]
                minus = evaluate(q - eps * tangent)[0]
                np.testing.assert_allclose(
                    dh[:, k],
                    (plus.gamma_inv - minus.gamma_inv) / (2 * eps),
                    atol=1e-10,
                    rtol=1e-8,
                )
                np.testing.assert_allclose(
                    a.grad_lapse[:, k],
                    (plus.lapse - minus.lapse) / (2 * eps),
                    atol=1e-10,
                )
                np.testing.assert_allclose(
                    a.grad_shift[:, :, k],
                    (plus.shift - minus.shift) / (2 * eps),
                    atol=1e-10,
                )
            np.testing.assert_array_equal(dh[:, 2], 0.0)
            self.assertTrue(np.asarray(particle_metric_valid(a, q, "numerical")).all())

    def test_adjacent_radial_tiles_agree(self):
        g1, m1 = manufactured()
        g2, m2 = manufactured(0.5)
        q = jnp.array([[0.499, 0.381, 3.0], [0.501, 0.381, 3.0]])
        for order in (1, 2):
            a, da = sample_particle_metric(
                m1, q, g1, order, "numerical", (True, True, False), (3, 3, 3)
            )
            b, db = sample_particle_metric(
                m2, q, g2, order, "numerical", (True, True, False), (3, 3, 3)
            )
            for v, w in zip(jax.tree.leaves((a, da)), jax.tree.leaves((b, db))):
                np.testing.assert_allclose(v, w, rtol=1e-12, atol=1e-12)

    def test_polar_masks_do_not_enter_particle_geometry(self):
        s, d, m, D, B = make_runtime("spherical", 16, 32)
        q = jnp.array(
            [
                [2.37, 0.05 * d.dy, 0.1],
                [2.37, 0.5 * d.dy, 0.1],
                [2.37, jnp.pi - 0.05 * d.dy, 0.1],
                [2.37, jnp.pi - 0.5 * d.dy, 0.1],
            ]
        )
        for order in (1, 2):
            static = s._replace(shape_factor=order)
            a = sample_metric(q, m, static, d)
            dh = sample_gradient(q, m, static, d)
            np.testing.assert_allclose(a.gamma_inv[:, 0, 0], 1.0, atol=1e-12)
            np.testing.assert_allclose(dh[:, :, 0, 0], 0.0, atol=1e-12)
            poisoned = m._replace(
                center=m.center._replace(
                    gamma_inv=jnp.full_like(m.center.gamma_inv, jnp.nan),
                    grad_lapse=jnp.full_like(m.center.grad_lapse, jnp.nan),
                    grad_shift=jnp.full_like(m.center.grad_shift, jnp.nan),
                    christoffel=jnp.full_like(m.center.christoffel, jnp.nan),
                ),
                center_grad_gamma_inv=jnp.full_like(m.center_grad_gamma_inv, jnp.nan),
            )
            for v, w in zip(
                jax.tree.leaves((a, dh)),
                jax.tree.leaves(
                    (
                        sample_metric(q, poisoned, static, d),
                        sample_gradient(q, poisoned, static, d),
                    )
                ),
            ):
                np.testing.assert_allclose(v, w, atol=1e-12)
            changed = m._replace(
                center=m.center._replace(
                    gamma=2 * m.center.gamma,
                    lapse=0.7 * m.center.lapse,
                    shift=m.center.shift + 0.1,
                )
            )
            b = sample_metric(q, changed, static, d)
            np.testing.assert_allclose(b.gamma_inv, a.gamma_inv / 2, atol=1e-12)
            np.testing.assert_allclose(b.lapse, 0.7 * a.lapse, atol=1e-12)
            np.testing.assert_allclose(b.shift, a.shift + 0.1, atol=1e-12)
        axes = q.at[:, 1].set(jnp.array([0.0, jnp.pi, 0.0, jnp.pi]))
        self.assertFalse(
            np.asarray(
                particle_metric_valid(sample_metric(axes, m, s, d), axes, s.metric)
            ).any()
        )

    def test_sampled_rotation_invariants(self):
        for row in sampled_rotation_checks(32):
            for key in (
                "inverse_defect",
                "norm_error",
                "parallel_error",
                "reversal_error",
            ):
                self.assertLess(row[key], 1e-12, (key, row))

    def test_signed_orientation_and_invalid_metrics(self):
        grid, m = manufactured()
        for chart, q in [
            ("flat_cylindrical", jnp.array([[-0.15, 0.25, 3.0]])),
            ("flat_spherical", jnp.array([[0.25, -0.15, 3.0]])),
        ]:
            for order in (1, 2):
                a, _ = sample_particle_metric(
                    m, q, grid, order, chart, (True, True, False), (3, 3, 3)
                )
                self.assertLess(float(a.sqrt_gamma[0]), 0.0)
                np.testing.assert_allclose(
                    a.sqrt_gamma**2, jnp.linalg.det(a.gamma), rtol=1e-12
                )
                self.assertTrue(bool(particle_metric_valid(a, q, chart)[0]))
        q = jnp.array([[0.25, 0.35, 3.0]])
        for tensor in (jnp.diag(jnp.array([-1.0, 1.0, 1.0])), jnp.zeros((3, 3))):
            bad = m._replace(gamma=jnp.broadcast_to(tensor, m.gamma.shape))
            a, _ = sample_particle_metric(
                bad, q, grid, 1, "numerical", (True, True, False), (3, 3, 3)
            )
            self.assertFalse(bool(particle_metric_valid(a, q, "numerical")[0]))

    def test_inactive_singular_slot_is_preserved(self):
        from PyPIC3D.particles.particle_class import TiledParticles, SpeciesConfig
        from PyPIC3D.pusher.hybrid_boris_geodesic import hybrid_boris_geodesic_push

        s, d, m, D, B = make_runtime("spherical", 16, 32)
        x = jnp.array([[[[[[2.0, 0.7, 0.0], [0.0, 0.0, 0.0]]]]]])
        u = jnp.zeros_like(x)
        active = jnp.array([[[[[True, False]]]]])
        p = TiledParticles(x, u, active)
        sp = SpeciesConfig(
            jnp.ones(1), jnp.ones(1), jnp.ones(1), jnp.ones((1, 3), bool)
        )
        new, mid = hybrid_boris_geodesic_push(p, sp, D, B, m, s, d)
        self.assertTrue(np.isfinite(np.asarray(new.x)).all())
        self.assertTrue(np.isfinite(np.asarray(new.u)).all())
        np.testing.assert_array_equal(new.x[..., 1, :], x[..., 1, :])
        np.testing.assert_array_equal(mid.x[..., 1, :], x[..., 1, :])


if __name__ == "__main__":
    unittest.main()
