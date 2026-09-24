"""Independent polar geometry, finite-volume field and unmodified-shape tests."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from PyPIC3D.boundary_conditions.polar import current_factors,divergence,refresh_vector,divide
from PyPIC3D.relativity.core import B_FIELD_LOCATIONS
from PyPIC3D.relativity.flat import initialize_flat_spherical_metric
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.deposition.GR_Esirkepov import GR_Esirkepov_current
from PyPIC3D.solvers.gr_static.static_metric import update_D_relativity,compute_covariant_E,compute_covariant_H
from demos.static_metric_relativity.bz_monopole.run_bz_monopole import monopole_field
from tests.support.polar_fixtures import polar_runtime,particle
jax.config.update('jax_enable_x64',True)

class TestPolar(unittest.TestCase):
    def test_polar_metric_parameters_must_match_geometry(self):
        from PyPIC3D.relativity.kerr_schild import initialize_kerr_schild_spherical_metric
        p,s,d,m=polar_runtime()
        with self.assertRaisesRegex(ValueError,'match static parameters'):
            initialize_kerr_schild_spherical_metric(s,d,mass=2.,spin=p.spin)
        with self.assertRaisesRegex(ValueError,'match static parameters'):
            initialize_flat_spherical_metric(s,d)

    def test_constant_field_gather_at_polar_halos(self):
        from PyPIC3D.pusher.hybrid_boris_geodesic import gather_vector
        for order in (1,2):
            p,s,d,m=polar_runtime(order=order,flat=True)
            z=jnp.zeros_like(m.center.sqrt_gamma)
            B=refresh_vector((z+1,z,z+.3),s,B_FIELD_LOCATIONS,'B')
            center_grid=tuple(axis[0,0,0] for axis in d.grids.tiled_center_grid)
            vertex_grid=tuple(axis[0,0,0] for axis in d.grids.tiled_vertex_grid)
            theta=jnp.array([-.1*p.dtheta,.1*p.dtheta,np.pi-.1*p.dtheta,np.pi+.1*p.dtheta])
            position=jnp.stack((jnp.full(4,2.),theta,jnp.zeros(4)),axis=-1)
            value=gather_vector(tuple(a[0,0,0] for a in B),B_FIELD_LOCATIONS,position,
                                center_grid,vertex_grid,order,(True,True,False),(3,3,3))
            np.testing.assert_allclose(value,np.broadcast_to([1.,0.,.3],(4,3)),rtol=1e-12,atol=1e-12)

    def test_stationary_cloud_overlapping_poles(self):
        for order in (1,2):
            p,s,d,m=polar_runtime(order=order);z=jnp.zeros_like(m.center.sqrt_gamma)
            for theta in (.1*p.dtheta,np.pi-.1*p.dtheta):
                old,sp=particle(s,d,theta)
                q=compute_rho(old,sp,z,s,d)*d.dx*d.dy*d.dz
                self.assertAlmostEqual(float(jnp.where(m.geometry.charge_owned,q,0).sum()),1.,places=12)
                current=GR_Esirkepov_current(old,old,sp,(z,)*3,m,s,d)
                self.assertEqual(max(float(jnp.max(jnp.abs(a))) for a in current),0.)

    def test_geometry(self):
        p,s,d,m=polar_runtime(flat=True);geo=m.geometry;g=s.guard_cells
        self.assertTrue(all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(m)))
        self.assertTrue(np.all(np.asarray(geo.volume)[np.asarray(geo.charge_owned)]>0))
        r=float(d.grids.tiled_center_grid[0][0,0,0,g+4])
        expected=2*np.pi*((r+p.dr/2)**3-(r-p.dr/2)**3)/3*(1-np.cos(p.dtheta/2))
        self.assertAlmostEqual(float(geo.volume[0,0,0,g+4,g,g])/expected,1.,places=12)
        self.assertEqual(int(geo.charge_owned.sum()),p.nr*(p.ntheta+1))

    def test_continuity(self):
        for order in (1,2):
            p,s,d,m=polar_runtime(order=order);zero=jnp.zeros_like(m.center.sqrt_gamma)
            for pole in (0.,np.pi):
                for direction in (-1,1):
                    for sign in (-1.,1.):
                        start=pole+(1 if pole==0 else -1)*.1*p.dtheta
                        old,sp=particle(s,d,start,charge=sign)
                        new=old._replace(x=old.x.at[...,1].add(direction*.85*p.dtheta).at[...,0].add(.2*p.dr))
                        q0=compute_rho(old,sp,zero,s,d)*d.dx*d.dy*d.dz
                        q1=compute_rho(new,sp,zero,s,d)*d.dx*d.dy*d.dz
                        j=GR_Esirkepov_current(old,new,sp,(zero,)*3,m,s,d)
                        residual=(q1-q0)/d.dt+divergence(j,m.geometry,s)
                        mask=m.geometry.charge_owned
                        scale=max(float(jnp.max(jnp.abs((q1-q0)/d.dt))),1.)
                        error=float(jnp.max(jnp.where(mask,jnp.abs(residual),0)))/scale
                        self.assertLess(error,1e-12,(order,pole,direction,sign,error))
                        self.assertAlmostEqual(float(jnp.sum(jnp.where(mask,q1,0))),sign,places=12)

    def test_curl_divergence(self):
        p,s,d,m=polar_runtime();zero=jnp.zeros_like(m.center.sqrt_gamma)
        random=tuple(jax.random.normal(jax.random.PRNGKey(i),zero.shape) for i in range(3))
        h=refresh_vector(random,s,B_FIELD_LOCATIONS)
        dd=update_D_relativity((zero,)*3,h,(zero,)*3,m,s,d,1.)
        mask=m.geometry.charge_owned.at[:,:,:,s.guard_cells,:,:].set(False)
        self.assertLess(float(jnp.max(jnp.where(mask,jnp.abs(divergence(dd,m.geometry,s)),0))),1e-12)
        b=refresh_vector(monopole_field(p,m,d),s,B_FIELD_LOCATIONS,'B')
        mask=m.geometry.B_owned[2].at[:,:,:,s.guard_cells+s.tile_shape[0]-1,:,:].set(False)
        # Integrated flux scales with B0=20000. Measure cancellation in
        # units of that flux, so roundoff is independent of field amplitude.
        flux_scale=max(float(jnp.max(jnp.abs(v*a))) for v,a in zip(b,m.geometry.B_area))
        residual=float(jnp.max(jnp.where(mask,jnp.abs(divergence(b,m.geometry,s,True)),0)))
        self.assertLess(residual/flux_scale,16*np.finfo(np.float64).eps)
        for a in (*compute_covariant_E(dd,b,m),*compute_covariant_H(dd,b,m)):
            self.assertTrue(bool(jnp.all(jnp.isfinite(a))))


class TestDistributedPolar(unittest.TestCase):
    @unittest.skipIf(len(jax.devices())<2,'requires two devices')
    def test_seam_and_pole(self):
        results=[]
        for devices in (1,2):
            p,s,d,m=polar_runtime(devices=devices,order=2)
            zero=jnp.zeros_like(m.center.sqrt_gamma);r=2.5-.05*p.dr
            old,sp=particle(s,d,.1*p.dtheta,r=r)
            new=old._replace(x=old.x.at[...,0].add(.2*p.dr).at[...,1].add(-.8*p.dtheta))
            q0=compute_rho(old,sp,zero,s,d)*d.dx*d.dy*d.dz
            q1=compute_rho(new,sp,zero,s,d)*d.dx*d.dy*d.dz
            j=GR_Esirkepov_current(old,new,sp,(zero,)*3,m,s,d)
            res=(q1-q0)/d.dt+divergence(j,m.geometry,s)
            err=float(jnp.max(jnp.where(m.geometry.charge_owned,jnp.abs(res),0)))/max(float(jnp.max(jnp.abs((q1-q0)/d.dt))),1.)
            self.assertLess(err,1e-12)
            g=s.guard_cells
            results.append(np.concatenate([np.asarray(q1)[i,0,0,g:-g,g:-g+1,g] for i in range(devices)]))
        np.testing.assert_allclose(*results,atol=1e-12,rtol=1e-12)

class TestPolarFieldCoupling(unittest.TestCase):
    def test_prescribed_32_steps(self):
        # Isolate the source/field contract from the independently failing pusher.
        p,s,d,m=polar_runtime(order=2);z=jnp.zeros_like(m.center.sqrt_gamma)
        old,sp=particle(s,d,.7*p.dtheta);D=(z,)*3
        qstart=compute_rho(old,sp,z,s,d)*d.dx*d.dy*d.dz
        D=(divide(4*jnp.pi*jnp.cumsum(qstart,axis=3),m.geometry.D_area[0]),z,z)
        g=s.guard_cells;mask=m.geometry.charge_owned.at[:,:,:,g,:,:].set(False)
        maximum=0.
        for k in range(32):
            newtheta=.7*p.dtheta-.06*p.dtheta*(k+1)
            # Raw unfolded endpoints; represented charge is folded onto the chart.
            new=old._replace(x=old.x.at[...,1].set(newtheta))
            j=GR_Esirkepov_current(old,new,sp,(z,)*3,m,s,d)
            D=update_D_relativity(D,(z,)*3,j,m,s,d,d.dt)
            q=compute_rho(new,sp,z,s,d)*d.dx*d.dy*d.dz
            err=float(jnp.max(jnp.where(mask,jnp.abs(divergence(D,m.geometry,s)-4*jnp.pi*q),0)))
            maximum=max(maximum,err);old=new
        self.assertLess(maximum,1e-10)

class TestAbsorption(unittest.TestCase):
    def test_boundary_cloud_budget(self):
        from PyPIC3D.boundary_conditions.polar import step_diagnostics
        p,s,d,m=polar_runtime();z=jnp.zeros_like(m.center.sqrt_gamma)
        for r,delta in ((p.r_min+.05*p.dr,-.1*p.dr),(p.r_max-.05*p.dr,.1*p.dr)):
            old,sp=particle(s,d,.3,r=r)
            new=old._replace(x=old.x.at[...,0].add(delta))
            j=GR_Esirkepov_current(old,new,sp,(z,)*3,m,s,d)
            report=step_diagnostics(old,new,j,sp,m,s,d)
            self.assertEqual(int(report.absorbed_count.sum()),1)
            self.assertAlmostEqual(float(report.absorbed_charge.sum()),1.)
            q0=compute_rho(old,sp,z,s,d)*d.dx*d.dy*d.dz
            q1=compute_rho(new._replace(active=jnp.zeros_like(new.active)),sp,z,s,d)*d.dx*d.dy*d.dz
            rate=jnp.sum(jnp.where(m.geometry.charge_owned,(q1-q0)/d.dt,0))
            balance=rate+report.radial_current_outflow.sum()+report.removed_grid_charge.sum()/d.dt
            self.assertLess(abs(float(balance)),1e-10)

class TestDirectAndCharts(unittest.TestCase):
    def test_direct_and_chart_transition(self):
        from PyPIC3D.deposition.GR_direct_deposition import GR_direct_deposition
        from PyPIC3D.particles.particle_tile_communication import refresh_tiled_particle_tiles
        p,s,d,m=polar_runtime();z=jnp.zeros_like(m.center.sqrt_gamma)
        for theta in (-.1*p.dtheta,np.pi+.1*p.dtheta):
            old,sp=particle(s,d,theta)
            old=old._replace(u=old.u.at[...,1].set(.2))
            new,overflow=refresh_tiled_particle_tiles(old,s,d)
            self.assertFalse(bool(overflow))
            self.assertGreaterEqual(float(new.x[...,1].min()),0.)
            self.assertLessEqual(float(new.x[...,1].max()),np.pi)
            self.assertAlmostEqual(float(new.u[0,0,0,0,0,1]),-.2)
            self.assertAlmostEqual(float(new.x[0,0,0,0,0,2]),np.pi)
            current=GR_direct_deposition(new,sp,(z,)*3,m,s,d)
            self.assertTrue(all(bool(jnp.all(jnp.isfinite(a))) for a in current))

    def test_direct_current_follows_supplied_metric(self):
        from PyPIC3D.deposition.GR_direct_deposition import GR_direct_deposition
        p,s,d,m=polar_runtime()
        shape=m.center.sqrt_gamma.shape
        gamma=jnp.diag(jnp.array([2.,3.,4.]))
        m=m._replace(center=m.center._replace(
            gamma=jnp.broadcast_to(gamma,shape+(3,3)),
            gamma_inv=jnp.broadcast_to(jnp.linalg.inv(gamma),shape+(3,3)),
            lapse=jnp.full(shape,.7),shift=jnp.broadcast_to(jnp.array([.1,0.,0.]),shape+(3,)),
            sqrt_gamma=jnp.full(shape,jnp.sqrt(24.))))
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
