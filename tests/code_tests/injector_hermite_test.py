"""Birth-frame invariants at polar stencils and a radial tile seam."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import checkify
from demos.static_metric_relativity.bz_monopole.plasma_injector import birth_covariant_momentum
from PyPIC3D.relativity.particle_metric import sample_particle_metric
from tests.code_tests.polar_test import polar_runtime as polar_setup

class TestBirthMetric(unittest.TestCase):
    def test_polar_seam_norm_and_placeholder_independence(self):
        p,s,d,m=polar_setup(devices=2,nt=32)
        momentum=jnp.array([[.3,-.4,.7],[-.2,.9,.1],[.8,.1,-.2]])
        seam=p.r_min+p.dr*s.tile_shape[0]
        positions=jnp.array([[seam,.1*p.dtheta,0.],[seam,np.pi-.1*p.dtheta,0.],[np.nan,np.nan,np.nan]])
        active=jnp.array([True,True,False]);results=[]
        for tile in range(2):
            grid=tuple(a[tile,0,0] for a in d.grids.tiled_center_grid)
            metric=jax.tree.map(lambda a:a[tile,0,0],m.center)
            f=lambda met:birth_covariant_momentum(momentum,positions,active,met,grid,s,jnp.array([tile,0,0]))
            error,cov=jax.jit(checkify.checkify(f))(metric);error.throw()
            sampled,_=sample_particle_metric(metric,positions[:2],grid,1,s.metric,(True,True,False),(3,3,3),derivatives=False)
            norm=jnp.einsum('ni,nij,nj->n',cov[:2],sampled.gamma_inv,cov[:2])
            np.testing.assert_allclose(norm,(momentum[:2]**2).sum(-1),rtol=1e-12,atol=1e-12)
            poisoned=metric._replace(gamma_inv=jnp.full_like(metric.gamma_inv,jnp.nan),grad_lapse=jnp.full_like(metric.grad_lapse,jnp.nan),grad_shift=jnp.full_like(metric.grad_shift,jnp.nan))
            np.testing.assert_array_equal(f(poisoned),cov)
            np.testing.assert_array_equal(f(metric),birth_covariant_momentum(momentum,positions,active,metric,grid,s._replace(shape_factor=2)))
            np.testing.assert_array_equal(cov[2],jnp.zeros(3));results.append(cov)
            changed=metric._replace(gamma=metric.gamma*2)
            np.testing.assert_allclose(f(changed)[:2],cov[:2]*np.sqrt(2),rtol=1e-12)
        np.testing.assert_allclose(results[0],results[1],rtol=1e-12,atol=1e-12)

    def test_timestep_cap_and_invalid_override(self):
        from dataclasses import replace
        from demos.static_metric_relativity.bz_monopole.simulation_parameters import SimulationParameters,build_runtime
        p=SimulationParameters(nr=16,ntheta=16,devices=1,r_max=4.,sponge_start=3.,
                               maximum_timestep=None)
        _,d,_,report=build_runtime(p)
        self.assertEqual(float(d.dt),report['cfl_dt'])
        self.assertEqual(report['timestep_policy'],'cfl')
        self.assertNotIn('gyro_dt',report)
        self.assertNotIn('polar_dt',report)
        _,changed,_,_=build_runtime(replace(p,skin_depth=p.skin_depth/10))
        self.assertEqual(float(changed.dt),float(d.dt))
        cap=report['cfl_dt']/2
        _,d2,_,_=build_runtime(replace(p,maximum_timestep=cap))
        self.assertEqual(float(d2.dt),cap)
        for value in (-1.,0.,np.inf,np.nan):
            with self.assertRaises(ValueError):replace(p,maximum_timestep=value).validate()

    def test_active_axis_reports_birth_stage(self):
        p,s,d,m=polar_setup()
        grid=tuple(a[0,0,0] for a in d.grids.tiled_center_grid)
        metric=jax.tree.map(lambda a:a[0,0,0],m.center)
        f=lambda:birth_covariant_momentum(jnp.ones((1,3)),jnp.array([[2.,0.,0.]]),jnp.ones(1,bool),metric,grid,s)
        err,_=jax.jit(checkify.checkify(f))()
        self.assertIn('injection birth transform',err.get())

if __name__=='__main__':unittest.main()
