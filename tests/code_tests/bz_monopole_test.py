"""Independent tests for the fresh half-domain BZ components (no long run)."""
from dataclasses import replace
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from tests.code_tests.polar_test import polar_runtime as polar_setup,particle
from demos.bz_monopole import magnetization as mag
from demos.bz_monopole.simulation_parameters import SimulationParameters
from demos.bz_monopole.plasma_injector import empty_particles,inject_pairs,thermal_momentum,orthonormal_to_covariant
from demos.bz_monopole.run_bz_monopole import initialize_fields,diagnostics,apply_sponge
from PyPIC3D.boundary_conditions.polar import refresh_vector
from PyPIC3D.relativity.core import B_FIELD_LOCATIONS
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.relativity.kerr_schild import _kerr_schild_spherical_metric_at_position

class TestParameters(unittest.TestCase):
    def test_production_defaults(self):
        p=SimulationParameters().validate()
        self.assertEqual((p.nr,p.ntheta,p.devices,p.guard_cells),(64,64,1,3))
        self.assertEqual(p.courant,.2)
        self.assertEqual(p.end_time,200.)
        self.assertEqual((p.skin_depth,p.pairs_per_cell,p.maximum_timestep),(.02,16,.004))
    def test_invalid(self):
        for changes in (dict(ntheta=7),dict(devices=3),dict(guard_cells=2),dict(spin=1.),dict(skin_depth=0.)):
            with self.assertRaises(ValueError):replace(SimulationParameters(),**changes).validate()
    def test_horizon_normalization(self):
        p,*_=polar_setup()
        for theta in (.2,1.,2.7):
            _,shift,gamma,_,_=_kerr_schild_spherical_metric_at_position(jnp.array([p.horizon,theta,0.]),1.,p.spin)
            self.assertAlmostEqual(float(-jnp.dot(gamma[2],shift)/gamma[2,2]),p.omega_h,places=13)

class TestMagnetization(unittest.TestCase):
    def test_constant_off_diagonal(self):mag.self_test()
    def test_number_and_inactive(self):
        p,s,d,m=polar_setup(order=2);zero=jnp.zeros_like(m.center.sqrt_gamma)
        for theta in (.01*p.dtheta,np.pi-.01*p.dtheta):
            pts,sp=particle(s,d,theta)
            sp=sp._replace(weight=jnp.array([2.5]))
            n=mag.deposit_number_density(pts,sp,zero,m,s,d)
            self.assertTrue(bool(jnp.all(n>=0)))
            total=jnp.sum(jnp.where(m.geometry.charge_owned,n[0]*m.geometry.volume,0))
            self.assertAlmostEqual(float(total),2.5,places=12)
            n=mag.deposit_number_density(pts._replace(active=jnp.zeros_like(pts.active)),sp,zero,m,s,d)
            self.assertEqual(float(n.sum()),0.)
    def test_manufactured_convergence(self):
        errors=[]
        for nt in (16,32,64):
            p,s,d,m=polar_setup(flat=True,nt=nt);shape=m.center.sqrt_gamma.shape
            tc=d.grids.tiled_center_grid[1][...,None,:,None]
            tv=d.grids.tiled_vertex_grid[1][...,None,:,None]
            rc=d.grids.tiled_center_grid[0][..., :,None,None]
            B=(jnp.broadcast_to(jnp.cos(2*tv),shape),jnp.broadcast_to(jnp.sin(tc),shape),jnp.zeros(shape))
            B=refresh_vector(B,s,B_FIELD_LOCATIONS,'B')
            n=jnp.ones((2,)+shape)
            result=mag.magnetization_from_density(B,n,jnp.ones(2),m)
            exact=jnp.cos(2*tc)**2+rc**2*jnp.sin(tc)**2
            error=jnp.max(jnp.where(m.geometry.charge_owned,jnp.abs(result.magnetic_squared-exact),0))
            errors.append(float(error))
            self.assertTrue(bool(jnp.allclose(result.sigma,result.magnetic_squared/(8*jnp.pi))))
        self.assertGreater(errors[0]/errors[1],3.7);self.assertGreater(errors[1]/errors[2],3.7)

class TestInjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p,cls.s,cls.d,cls.m=polar_setup();cls.pts,cls.sp=empty_particles(cls.p,cls.s)
        shape=cls.m.center.sqrt_gamma.shape;z=jnp.zeros(shape);cls.zero=(z,)*3
        cls.vac=mag.Magnetization(jnp.zeros((2,)+shape),jnp.ones(shape),jnp.full(shape,jnp.inf),jnp.ones(shape,bool))
        cls.call=jax.jit(lambda pts,key,vac:inject_pairs(pts,cls.sp,vac,cls.zero,cls.zero,cls.m,cls.s,cls.d,cls.p,key,0))
        cls.result,cls.key,cls.report=cls.call(cls.pts,jax.random.PRNGKey(41),cls.vac)
    def test_neutral_reproducible(self):
        r,k,rep=type(self).call(self.pts,jax.random.PRNGKey(41),self.vac)
        self.assertTrue(bool(jnp.array_equal(r.x,self.result.x)))
        self.assertTrue(bool(jnp.array_equal(r.u,self.result.u)))
        self.assertGreater(int(rep.inserted.sum()),0);self.assertEqual(int(rep.rejected.sum()),0)
        rho=compute_rho(r,self.sp,self.zero[0],self.s,self.d)
        absolute_charge=compute_rho(r,self.sp._replace(charge=jnp.abs(self.sp.charge)),
                                    self.zero[0],self.s,self.d)
        # Depositing opposite species cancels to reduction roundoff. Scale the
        # bound by the charge being cancelled, rather than dimensional units.
        tolerance=16*np.finfo(np.float64).eps*float(jnp.max(absolute_charge))
        self.assertLessEqual(float(jnp.max(jnp.abs(rho))),tolerance)
        x=np.asarray(r.x)[np.asarray(r.active)]
        self.assertTrue(np.all((x[:,1]>0)&(x[:,1]<np.pi)&(x[:,0]>=self.p.r_min)&(x[:,0]<self.p.sponge_start)))
        n=mag.deposit_number_density(r,self.sp,self.zero[0],self.m,self.s,self.d)
        total=float(jnp.sum(jnp.where(self.m.geometry.charge_owned,n.sum(0)*self.m.geometry.volume,0)))
        np.testing.assert_allclose(total,2*float(self.sp.weight[0])*int(rep.inserted.sum()),
                                   rtol=16*np.finfo(np.float64).eps,atol=0.)
    def test_threshold_and_capacity(self):
        v=self.vac._replace(sigma=jnp.full_like(self.vac.sigma,self.p.sigma_threshold))
        r,k,rep=type(self).call(self.pts,jax.random.PRNGKey(1),v)
        self.assertEqual(int(rep.requested.sum()),0)
        active=self.pts.active.at[...,0,:].set(True)
        pts=self.pts._replace(active=active)
        r,k,rep=type(self).call(pts,jax.random.PRNGKey(1),self.vac)
        self.assertEqual(int(rep.inserted.sum()),0);self.assertGreater(int(rep.rejected.sum()),0)
        self.assertTrue(bool(jnp.array_equal(r.active,active)))
        self.assertTrue(bool(jnp.array_equal(r.u,pts.u)))
    def test_thermal_and_transform(self):
        from scipy.integrate import quad
        keys=jax.random.split(jax.random.PRNGKey(1),12000)
        u=jax.jit(jax.vmap(lambda k:thermal_momentum(k,.5)))(keys)
        norm=quad(lambda p:p*p*np.exp(-np.sqrt(1+p*p)/.5),0,np.inf)[0]
        expected=quad(lambda p:p*p*np.sqrt(1+p*p)*np.exp(-np.sqrt(1+p*p)/.5),0,np.inf)[0]/norm
        self.assertLess(abs(float(jnp.mean(jnp.sqrt(1+jnp.sum(u*u,axis=1))))-expected)/expected,.015)
        self.assertLess(float(jnp.max(jnp.abs(jnp.mean(u,axis=0)))),.04)
        gamma=jnp.array([[2.,.3,.1],[.3,3.,.2],[.1,.2,4.]])
        cov=orthonormal_to_covariant(u[0],gamma)
        inv=jnp.linalg.inv(gamma)
        self.assertAlmostEqual(float(cov@inv@cov),float(u[0]@u[0]),places=11)

    def test_newborn_only_and_slot_reuse(self):
        old=self.result
        new,key,report=type(self).call(old,self.key,self.vac)
        np.testing.assert_array_equal(np.asarray(new.x)[np.asarray(old.active)],np.asarray(old.x)[np.asarray(old.active)])
        np.testing.assert_array_equal(np.asarray(new.u)[np.asarray(old.active)],np.asarray(old.u)[np.asarray(old.active)])
        self.assertEqual(new.x.shape,old.x.shape)
        self.assertTrue(bool(jnp.all(~report.newborn|~old.active)))
        # Deactivate all occupied slots; the next event reuses the same capacity.
        cleared=old._replace(active=jnp.zeros_like(old.active))
        again,key,report=type(self).call(cleared,jax.random.PRNGKey(41),self.vac)
        np.testing.assert_array_equal(again.active,self.result.active)
        np.testing.assert_array_equal(again.x,self.result.x)


class TestRunner(unittest.TestCase):
    def test_initialize_sponge(self):
        p,s,d,m=polar_setup();particles,sp=empty_particles(p,s);fields,b=initialize_fields(p,s,d,m)
        self.assertTrue(all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(fields)))
        stationary=(fields[0],b)+fields[2:]
        damped=apply_sponge(stationary,b,p,s,d)
        for a,bb in zip(damped[1],b):np.testing.assert_array_equal(a,bb)
        snap=diagnostics(particles,sp,fields,p,s,d)
        self.assertEqual(snap['Hphi'].shape,(p.nr,p.ntheta+1))
        self.assertEqual(snap['divB'].shape,(p.nr,p.ntheta))


if __name__ == '__main__':
    unittest.main()
