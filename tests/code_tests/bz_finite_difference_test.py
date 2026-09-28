"""Independent finite-difference BZ wall, conservation, and solver checks."""
from contextlib import ExitStack
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from PyPIC3D.boundary_conditions.staggered import refresh_fields, scalar_boundaries
from PyPIC3D.diagnostics.static_metric import divergence, node_weights, step_diagnostics
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.deposition.GR_Esirkepov import GR_Esirkepov_current
from PyPIC3D.particles.particle_tile_communication import refresh_tiled_particle_tiles
from PyPIC3D.relativity.core import D_FIELD_LOCATIONS, B_FIELD_LOCATIONS
from PyPIC3D.solvers.gr_static.static_metric import update_D_relativity
from demos.static_metric_relativity.bz_monopole.simulation_parameters import SimulationParameters, build_runtime
from demos.static_metric_relativity.bz_monopole.current_filter import smooth_conformal, filter_current
from demos.static_metric_relativity.bz_monopole import run_bz_monopole as runner
from demos.static_metric_relativity.bz_monopole.plasma_injector import empty_particles
from tests.support.bz_fixtures import bz_runtime, particle

jax.config.update('jax_enable_x64', True)


class TestFiniteDifferenceBZ(unittest.TestCase):
    def test_ranges_and_regular_metric(self):
        p = SimulationParameters(nr=16, ntheta=32, r_max=4., horizon_field_cells=0,
                                 theta_start=.4, theta_end=2.5)
        s, d, m, _ = build_runtime(p)
        self.assertIsNone(m.geometry)
        self.assertEqual(s.boundary_conditions, (3, 1, 0))
        self.assertEqual(s.particle_boundary_conditions, (2, 1, 0))
        self.assertAlmostEqual(float(d.grids.center[1][1]), p.theta_start)
        self.assertAlmostEqual(float(d.grids.center[1][-1]), p.theta_end)
        for item in (m.center, m.vertex, *m.D, *m.B):
            self.assertTrue(np.isfinite(np.asarray(item.gamma_inv)).all())
            self.assertTrue(np.all(np.asarray(item.sqrt_gamma) > 0))
        for endpoint, sign in ((p.theta_start, -1), (p.theta_end, 1)):
            points, _ = particle(s, d, endpoint+sign*.1*p.dtheta)
            reflected, overflow = refresh_tiled_particle_tiles(points, s, d)
            self.assertFalse(bool(overflow))
            self.assertAlmostEqual(float(reflected.x[0,0,0,0,0,1]), endpoint-sign*.1*p.dtheta)
        for start, end in ((0., 3.), (.4, math.pi), (1., .5), (np.nan, 2.), (.01, 3.)):
            with patch('demos.static_metric_relativity.bz_monopole.simulation_parameters.initialize_kerr_schild_spherical_metric') as init:
                with self.assertRaises(ValueError):
                    build_runtime(replace(p, theta_start=start, theta_end=end))
                init.assert_not_called()
        for lower, upper in ((.1, 4.), (4., 1.), (np.nan, 4.)):
            with patch('demos.static_metric_relativity.bz_monopole.simulation_parameters.initialize_kerr_schild_spherical_metric') as init:
                with self.assertRaises(ValueError):
                    build_runtime(replace(p, r_min=lower, r_max=upper))
                init.assert_not_called()

    def test_component_walls_are_at_actual_endpoints(self):
        p, s, d, m = bz_runtime()
        g, n = s.guard_cells, s.tile_shape[1]
        def plane(i): return (0, 0, 0, slice(None), i, slice(None))
        for locs, kind in ((D_FIELD_LOCATIONS, 'D'), (B_FIELD_LOCATIONS, 'B')):
            vector = tuple(jnp.ones_like(m.center.lapse)*(i+1) for i in range(3))
            result = refresh_fields(vector, s, locs, kind)
            for i, (value, loc) in enumerate(zip(result, locs)):
                parity = (1 if i == 1 else -1) if kind == 'D' else (-1 if i == 1 else 1)
                if loc[1] == 'C' and parity == -1:
                    np.testing.assert_array_equal(value[plane(g)], 0.)
                    np.testing.assert_array_equal(value[plane(g+n)], 0.)
                    # The final interior node is not silently turned into a wall.
                    self.assertGreater(float(jnp.max(jnp.abs(value[plane(g+n-1)]))), 0.)
                low_owner = g if loc[1] == 'V' else g+1
                np.testing.assert_array_equal(value[plane(g-1)], parity*value[plane(low_owner)])
                high_ghost = g+n if loc[1] == 'V' else g+n+1
                np.testing.assert_array_equal(value[plane(high_ghost)], parity*value[plane(g+n-1)])

    def test_cartesian_staggered_walls_and_source_integral(self):
        from PyPIC3D.boundary_conditions.staggered import source_boundaries
        _, s, _, _ = bz_runtime()
        for nx in (1, 8):
            s = s._replace(metric='flat_cartesian', tile_shape=(nx,8,8),
                           boundary_conditions=(1,1,1), particle_boundary_conditions=(1,1,1))
            g=s.guard_cells
            shape=(1,1,1)+(nx+2*g,8+2*g,8+2*g)
            source=jnp.arange(np.prod(shape),dtype=float).reshape(shape)/np.prod(shape)
            folded=source_boundaries(source,s,fold=True)
            # Even image sources on endpoint nodes are integrated with ordinary
            # nodal quadrature, independently of the coordinate metric.
            self.assertAlmostEqual(float(jnp.sum(folded*node_weights(s,source))),float(source.sum()),places=10)
            for locs,kind in ((D_FIELD_LOCATIONS,'D'),(B_FIELD_LOCATIONS,'B')):
                result=refresh_fields((jnp.ones(shape),)*3,s,locs,kind)
                for axis in range(3):
                    for i,loc in enumerate(locs):
                        odd=(i!=axis) if kind=='D' else (i==axis)
                        if loc[axis]!='C' or not odd:continue
                        for wall in (g,g+s.tile_shape[axis]):
                            plane=[slice(None)]*6;plane[axis+3]=wall
                            np.testing.assert_array_equal(result[i][tuple(plane)],0.)

    def test_frozen_layers_leave_auxiliary_fields_unflattened(self):
        _,s,_,m=bz_runtime()
        s=s._replace(horizon_field_cells=3)
        g=s.guard_cells
        value=jnp.broadcast_to(jnp.arange(m.center.lapse.shape[3])[None,None,None,:,None,None],m.center.lapse.shape).astype(float)
        for locs,kind in ((D_FIELD_LOCATIONS,'D'),(B_FIELD_LOCATIONS,'B')):
            fields=refresh_fields((value,)*3,s,locs,kind)
            auxiliary=refresh_fields((value,)*3,s,locs)
            for a,b in zip(fields,auxiliary):
                np.testing.assert_array_equal(np.asarray(a)[0,0,0,:g+3,g+5,g],g+3)
                self.assertNotEqual(float(b[0,0,0,g,g+5,g]),float(b[0,0,0,g+2,g+5,g]))

    def test_reflection_and_single_cell_periodicity(self):
        p, s, d, _ = bz_runtime()
        for wall, sign in ((p.theta_start, -1), (p.theta_end, 1)):
            old, _ = particle(s, d, wall+sign*.1*p.dtheta)
            old = old._replace(x=old.x.at[..., 2].set(2*math.pi+.7),
                               u=old.u.at[..., 1].set(.3))
            new, overflow = refresh_tiled_particle_tiles(old, s, d)
            self.assertFalse(bool(overflow))
            self.assertAlmostEqual(float(new.x[0,0,0,0,0,1]), wall-sign*.1*p.dtheta)
            self.assertAlmostEqual(float(new.x[0,0,0,0,0,2]), .7)
            self.assertAlmostEqual(float(new.u[0,0,0,0,0,1]), -.3)

    def test_reflected_continuity_and_filter_commutation(self):
        single_device = {}
        for devices in (1, 2):
            p, s, d, m = bz_runtime(devices=devices)
            z = jnp.zeros_like(m.center.lapse)
            g = s.guard_cells
            def assemble(value):
                return np.concatenate([
                    np.asarray(value)[tile,0,0,g:g+s.tile_shape[0],g:g+p.ntheta+1,g]
                    for tile in range(devices)], axis=0)
            for wall, sign in ((p.theta_start, 1), (p.theta_end, -1)):
                old, species = particle(s, d, wall+sign*.1*p.dtheta, r=2.5-.05*p.dr)
                new = old._replace(x=old.x.at[..., 1].add(-sign*.3*p.dtheta).at[...,0].add(.1*p.dr))
                current = GR_Esirkepov_current(old, new, species, (z,)*3, m, s, d)
                q0 = compute_rho(old, species, z, s, d)
                q1 = compute_rho(new, species, z, s, d)
                reflected, _ = refresh_tiled_particle_tiles(new, s, d)
                np.testing.assert_allclose(compute_rho(reflected, species, z, s, d), q1, rtol=1e-12, atol=1e-12)
                for passes in (0, 4):
                    flux = filter_current(current, m, s, passes)
                    rate = (smooth_conformal(q1, s, passes)-smooth_conformal(q0, s, passes))/d.dt
                    error = rate+divergence(flux, m.D, d)
                    r = d.grids.tiled_center_grid[0][..., :,None,None]
                    mask = (node_weights(s,z)>0)&(r>p.r_min+5*p.dr)&(r<p.r_max-5*p.dr)
                    scale = max(float(jnp.max(jnp.abs(rate))), 1.)
                    self.assertLess(float(jnp.max(jnp.where(mask,jnp.abs(error),0.)))/scale, 2e-12)
                    values = [assemble(q1), *(assemble(component) for component in flux)]
                    if devices == 1:
                        single_device[wall, passes] = values
                    else:
                        for actual, expected in zip(values, single_device[wall, passes]):
                            np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
                total = jnp.sum(q1*node_weights(s,z))*d.dx*d.dy*d.dz
                self.assertAlmostEqual(float(total), 1., places=12)

    def test_absorbing_radial_charge_budget(self):
        p, s, d, m = bz_runtime()
        z = jnp.zeros_like(m.center.lapse)
        for radius, delta in ((p.r_min+.05*p.dr, -.1*p.dr), (p.r_max-.05*p.dr, .1*p.dr)):
            for angle in (1.3, p.theta_start+.01*p.dtheta, p.theta_end-.01*p.dtheta):
                old, species = particle(s, d, angle, r=radius)
                new = old._replace(x=old.x.at[...,0].add(delta))
                current = GR_Esirkepov_current(old,new,species,(z,)*3,m,s,d)
                report = step_diagnostics(old,new,current,species,m,s,d)
                q0 = compute_rho(old,species,z,s,d)
                vanished = new._replace(active=jnp.zeros_like(new.active))
                q1 = compute_rho(vanished,species,z,s,d)
                rate = jnp.sum((q1-q0)*node_weights(s,z))*d.dx*d.dy*d.dz/d.dt
                self.assertEqual(int(report.absorbed_count.sum()),1)
                self.assertLess(abs(float(rate+report.radial_current_outflow.sum()+report.removed_grid_charge.sum()/d.dt)),1e-10)

    def test_monopole_and_pic_step_do_not_use_polar_helpers(self):
        p, s, d, m = bz_runtime()
        p = replace(p, maximum_timestep=.004)
        d = d._replace(dt=jnp.asarray(.004))
        particles, species = empty_particles(p,s)
        names = ('build_geometry','safe_grid_provider','refresh_vector','update_fields','physical_current')
        with ExitStack() as stack:
            for name in names:
                stack.enter_context(patch('PyPIC3D.boundary_conditions.polar.'+name, side_effect=AssertionError(name)))
            stack.enter_context(patch('PyPIC3D.deposition.GR_Esirkepov.physical_current', side_effect=AssertionError('polar current')))
            fields, background = runner.initialize_fields(p,s,d,m)
            error = divergence(background,m.B,d,forward=True)
            g=s.guard_cells
            self.assertLess(float(jnp.max(jnp.abs(error[:,:, :,g+2:-g-2,g+2:-g-2,g]))),1e-10)
            execute = runner.make_step(p,species,s,d,background,current_filter_passes=0)
            particles,fields,_,_ = execute(particles,fields,jax.random.PRNGKey(p.seed),0)
            self.assertTrue(bool(runner.finite_state(particles,fields)))
            runner.check_constraints(runner.constraint_residuals(particles,species,fields,s,d,p))

    def test_manufactured_curved_metric_curl_is_second_order(self):
        errors=[]
        for n in (16,32,64):
            p=SimulationParameters(nr=n,ntheta=n,r_max=4.,horizon_field_cells=0,
                    theta_start=math.pi/6,theta_end=5*math.pi/6)
            s,d,m,_=build_runtime(p)
            shape=m.center.lapse.shape
            rc=d.grids.tiled_center_grid[0][..., :,None,None]
            rv=d.grids.tiled_vertex_grid[0][..., :,None,None]
            tc=d.grids.tiled_center_grid[1][...,None,:,None]
            tv=d.grids.tiled_vertex_grid[1][...,None,:,None]
            z=jnp.zeros(shape)
            H=(z,z,jnp.broadcast_to(jnp.sin(rv)*jnp.cos(tv),shape))
            value=update_D_relativity((z,)*3,H,(z,)*3,m,s,d,1.)
            exact=(-jnp.sin(rv)*jnp.sin(tc)/m.D[0].sqrt_gamma,
                   -jnp.cos(rc)*jnp.cos(tv)/m.D[1].sqrt_gamma,z)
            g=s.guard_cells
            # Use the same physical region on every grid. A fixed number of
            # excluded cells moves the norm toward the high-curvature inner
            # radius as n increases and biases the measured order.
            mask=np.broadcast_to(np.asarray((rc>=2.)&(rc<=3.)&(tc>=1.)&(tc<=2.)),shape).copy()
            mask[..., :g]=False;mask[..., g+1:]=False
            numerator=sum(np.mean(np.asarray(a-b)[mask]**2) for a,b in zip(value,exact))
            denominator=sum(np.mean(np.asarray(b)[mask]**2) for b in exact)
            errors.append(np.sqrt(numerator/denominator))
        orders = [math.log(coarse/fine,2) for coarse,fine in zip(errors,errors[1:])]
        print(f'FD curl relative errors: {errors}; orders: {orders}', flush=True)
        for order in orders:
            self.assertGreater(order,1.8,errors)


if __name__ == '__main__':
    unittest.main()
