"""Particle paths must consume supplied metric arrays, including in polar mode."""
import unittest
import jax.numpy as jnp
import numpy as np
from tests.code_tests.polar_test import polar_runtime as polar_setup, particle
from tests.support.particle_metric_fixtures import sample_metric
from PyPIC3D.deposition.GR_direct_deposition import GR_direct_deposition
from PyPIC3D.boundary_conditions.polar import current_factors


class TestParticleMetricGrid(unittest.TestCase):
    def supplied_metric(self):
        p,s,d,m=polar_setup()
        shape=m.center.sqrt_gamma.shape
        gamma=jnp.diag(jnp.array([2.,3.,4.]))
        center=m.center._replace(
            gamma=jnp.broadcast_to(gamma,shape+(3,3)),
            gamma_inv=jnp.broadcast_to(jnp.linalg.inv(gamma),shape+(3,3)),
            lapse=jnp.full(shape,.7),shift=jnp.broadcast_to(jnp.array([.1,0.,0.]),shape+(3,)),
            sqrt_gamma=jnp.full(shape,jnp.sqrt(24.)))
        return p,s,d,m._replace(center=center)

    def test_metric_and_derivatives_follow_grid(self):
        p,s,d,m=self.supplied_metric()
        position=jnp.array([[2.,.1*p.dtheta,0.],[2.1,jnp.pi-.1*p.dtheta,0.]])
        sampled=sample_metric(position,m,s,d)
        np.testing.assert_allclose(sampled.gamma,np.broadcast_to(np.diag([2.,3.,4.]),(2,3,3)),atol=1e-12)
        np.testing.assert_allclose(sampled.lapse,.7,atol=1e-12)
        # Derivatives are those of the interpolated supplied metric.
        np.testing.assert_allclose(sampled.grad_gamma_inv,0.,atol=1e-12)
        np.testing.assert_allclose(sampled.grad_lapse,0.,atol=1e-12)

    def test_direct_current_follows_supplied_metric(self):
        p,s,d,m=self.supplied_metric()
        old,sp=particle(s,d,.6)
        old=old._replace(u=old.u.at[...,0].set(1.))
        z=jnp.zeros_like(m.center.sqrt_gamma)
        current=GR_direct_deposition(old,sp,(z,)*3,m,s,d)
        conformal=current[0]*current_factors(m.geometry,d)[0]
        total=jnp.where(m.geometry.D_owned[0],conformal,0).sum()*d.dx*d.dy*d.dz
        expected=.7*.5/jnp.sqrt(1.5)-.1
        self.assertAlmostEqual(float(total),float(expected),places=12)


if __name__ == '__main__':
    unittest.main()
