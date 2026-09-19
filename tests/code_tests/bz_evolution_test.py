"""The self-contained BZ evolution driver preserves checked restart state."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from demos.bz_monopole import run_bz_monopole as runner
from demos.bz_monopole.simulation_parameters import SimulationParameters,build_runtime
from demos.bz_monopole.plasma_injector import empty_particles

class TestEvolution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=SimulationParameters(nr=16,ntheta=16,devices=2,r_max=4.,sponge_start=3.,
                                  skin_depth=.0025,pairs_per_cell=4,maximum_timestep=None,
                                  end_time=5.,output_interval=1.)
        cls.s,cls.d,cls.m,_=build_runtime(cls.p)
        cls.pts,cls.sp=empty_particles(cls.p,cls.s)
        cls.fields,cls.bg=runner.initialize_fields(cls.p,cls.s,cls.d,cls.m)
        cls.key=jax.random.PRNGKey(5)

    def test_evolution_restart_and_failure(self):
        p,s,d=self.p,self.s,self.d
        execute=runner.make_step(p,self.sp,s,d,self.bg)
        expected_pts,expected_fields,expected_key=self.pts,self.fields,self.key
        for step in range(4):
            expected_pts,expected_fields,expected_key,_=execute(expected_pts,expected_fields,expected_key,step)
        with TemporaryDirectory() as directory:
            root=Path(directory)
            # Avoid recompilation from a fresh closure for each invocation.
            with patch.object(runner,'make_step',return_value=execute):
                a=runner.evolve(self.pts,self.sp,self.fields,self.key,p,s,d,self.bg,root/'first',
                    steps=2,allow_divergence_errors=True,checkpoint_seconds=0,constraint_check_interval=1)
                saved=root/'first/checkpoint.npz'
                self.assertTrue((root/'first/checkpoint_previous.npz').exists())
                restored=runner.load_checkpoint(saved,self.pts,self.fields,p,s,expected_dt=d.dt)
                self.assertEqual(restored[3],2)
                result=runner.evolve(restored[0],self.sp,restored[1],restored[2],p,s,d,self.bg,root/'resumed',
                    start_step=2,steps=2,allow_divergence_errors=True)
            for x,y in zip(jax.tree.leaves(result[:3]),jax.tree.leaves((expected_pts,expected_fields,expected_key))):
                np.testing.assert_allclose(x,y,rtol=1e-12,atol=1e-12)
            self.assertEqual(result[3],4);runner.check_sharding(result[0],result[1],s)
            metadata=json.loads((root/'resumed/manifest.json').read_text())
            self.assertEqual(metadata['particle_integrator'],'explicit_midpoint_strang_v1')
            with self.assertRaisesRegex(ValueError,'timestep differs'):
                runner.load_checkpoint(saved,self.pts,self.fields,p,s,expected_dt=2*float(d.dt))
            first=execute(self.pts,self.fields,self.key,0)
            def fail_second(pts,fields,key,index):
                if index==1:raise RuntimeError('injected test failure')
                return execute(pts,fields,key,index)
            with patch.object(runner,'make_step',return_value=fail_second):
                with self.assertRaisesRegex(RuntimeError,'injected test failure'):
                    runner.evolve(self.pts,self.sp,self.fields,self.key,p,s,d,self.bg,root/'failed',
                        steps=3,allow_divergence_errors=True)
            failed=runner.load_checkpoint(root/'failed/failed_checkpoint.npz',self.pts,self.fields,p,s,expected_dt=d.dt)
            self.assertEqual(failed[3],1)
            for x,y in zip(jax.tree.leaves(failed[:3]),jax.tree.leaves(first[:3])):
                np.testing.assert_array_equal(x,y)
            self.assertEqual(json.loads((root/'failed/manifest.json').read_text())['status'],'failed')
        broken=first[0]._replace(x=first[0].x.at[0,0,0,0,0,1].set(0.))
        self.assertTrue(bool(broken.active[0,0,0,0,0]))
        with self.assertRaises(Exception):execute(broken,first[1],first[2],1)

    def test_divergence_waiver_does_not_allow_nonfinite(self):
        with self.assertRaisesRegex(RuntimeError,'[Dd]ivergence'):
            runner.check_constraints(dict(gauss=.01,magnetic_divergence=0.),False)
        runner.check_constraints(dict(gauss=.01,magnetic_divergence=0.),True)
        runner.check_constraints(dict(gauss=0.,magnetic_divergence=0.,
                                      boundary_gauss=1.,boundary_magnetic_divergence=1.,
                                      full_gauss=1.,full_magnetic_divergence=1.),False)
        with self.assertRaises(FloatingPointError):
            runner.check_constraints(dict(gauss=np.nan,magnetic_divergence=0.),True)

    def test_regions_exclude_physical_boundaries_but_keep_tile_seams(self):
        from dataclasses import replace
        p=replace(self.p,sponge_start=3.8)
        geometry=self.m.geometry
        zero=jnp.zeros_like(self.fields[3])
        dm=runner.constraint_masks(geometry,self.s,self.d,p)
        bm=runner.constraint_masks(geometry,self.s,self.d,p,magnetic=True)
        g=self.s.guard_cells
        for masks in (dm,bm):
            owned,inside,boundary=map(np.asarray,masks)
            self.assertTrue(inside[0,0,0,g+self.s.tile_shape[0]-1,g+5,g])
            self.assertTrue(inside[1,0,0,g,g+5,g])
            self.assertFalse(inside[0,0,0,g,g+5,g])
            self.assertFalse(inside[0,0,0,g+5,g,g])
            self.assertFalse(inside[1,0,0,g+self.s.tile_shape[0]-1,g+5,g])
            np.testing.assert_array_equal(inside|boundary,owned)
        def measure(dd,bb):
            with patch('PyPIC3D.boundary_conditions.polar.divergence',side_effect=[dd,bb]), \
                 patch.object(runner,'compute_rho',return_value=zero):
                return {k:float(v) for k,v in runner.constraint_residuals(
                    self.pts,self.sp,self.fields,self.s,self.d,p).items()}
        boundary_d=jnp.where(dm[2],1e12,0.)
        boundary_b=jnp.where(bm[2],1e12,0.)
        r=measure(boundary_d,boundary_b)
        runner.check_constraints(r,False)
        self.assertGreater(r['boundary_gauss'],0)
        self.assertGreater(r['boundary_magnetic_divergence'],0)
        # A seam error must fail and cannot be diluted by a huge boundary error.
        seam=(1,0,0,g,g+5,g)
        r=measure(boundary_d.at[seam].set(1e-5),boundary_b)
        self.assertAlmostEqual(r['gauss'],1e-5,places=12)
        with self.assertRaisesRegex(RuntimeError,'Exterior'):runner.check_constraints(r,False)
        r=measure(boundary_d,boundary_b.at[seam].set(1.))
        self.assertGreater(r['magnetic_divergence'],1e-10)

    def test_horizon_and_outer_errors_cannot_hide_exterior_errors(self):
        zero=jnp.zeros_like(self.fields[3])
        regions=[runner._constraint_regions(self.m.geometry,self.s,self.d,self.p,magnetic=m)
                 for m in (False,True)]
        for magnetic, masks in zip((False,True),regions):
            grid=self.d.grids.tiled_vertex_grid if magnetic else self.d.grids.tiled_center_grid
            radius=np.broadcast_to(np.asarray(grid[0])[..., :,None,None],zero.shape)
            owned=np.asarray(masks['full_'])
            self.assertTrue(np.all(np.asarray(masks['horizon_'])[owned & (radius<self.p.horizon)]))
            self.assertFalse(np.any(np.asarray(masks[''])[radius<self.p.horizon]))
            # The first owned radial plane outside the horizon has accepted equatorial cells.
            first=radius[owned & (radius>=self.p.horizon)].min()
            self.assertTrue(np.any(np.asarray(masks['']) & (radius==first)))
            self.assertFalse(np.any(np.asarray(masks['']) & np.asarray(masks['outer_boundary_'])))
        excluded=[m['horizon_']|m['outer_boundary_'] for m in regions]
        dd,bb=[jnp.where(m,1e12,0.) for m in excluded]
        # Enormous excluded-region B must not increase the exterior normalization.
        fields=list(self.fields)
        fields[1]=tuple(jnp.where(excluded[1],1e18,0.) for _ in range(3))
        def measure(d,b):
            with patch('PyPIC3D.boundary_conditions.polar.divergence',side_effect=[d,b]), \
                 patch.object(runner,'compute_rho',return_value=zero):
                return runner.constraint_residuals(self.pts,self.sp,fields,self.s,self.d,self.p)
        values=measure(dd,bb)
        runner.check_constraints(values)
        for prefix in ('horizon_','outer_boundary_'):
            self.assertGreater(float(values[prefix+'gauss']),0.)
            self.assertGreater(float(values[prefix+'magnetic_divergence']),0.)
        for i,name in enumerate(('gauss','magnetic_divergence')):
            cell=tuple(np.argwhere(np.asarray(regions[i]['']))[0])
            errors=[dd,bb];errors[i]=errors[i].at[cell].set(1e-5)
            values=measure(*errors)
            self.assertAlmostEqual(float(values[name]),1e-5,places=12)
            with self.assertRaisesRegex(RuntimeError,name):runner.check_constraints(values)

    def test_empty_exterior_and_snapshot_regions(self):
        from dataclasses import replace
        p=replace(self.p,sponge_start=1.9)
        values=runner.constraint_residuals(self.pts,self.sp,self.fields,self.s,self.d,p)
        with self.assertRaises(FloatingPointError):runner.check_constraints(values,True)
        snap=runner.diagnostics(self.pts,self.sp,self.fields,self.p,self.s,self.d)
        for prefix in ('','horizon_','outer_boundary_','boundary_','full_'):
            for name in ('gauss','divB'):
                self.assertIn(prefix+name+'_relative',snap)
                self.assertGreater(int(snap[prefix+name+'_cells']),0)
                self.assertTrue(bool(snap[prefix+name+'_valid']))
        self.assertEqual(int(snap['constraint_policy_version']),2)


class TestConstraintPolicy(unittest.TestCase):
    def test_independent_thresholds_and_equality(self):
        for name in ('gauss','magnetic_divergence'):
            values=dict(gauss=0.,magnetic_divergence=0.)
            values[name]=1e-10
            with self.assertRaisesRegex(RuntimeError,name):runner.check_constraints(values)
            runner.check_constraints(values,**{name+'_tolerance':1e-9})
            runner.check_constraints(values,True)

    def test_invalid_settings(self):
        for value in (0.,-1.,float('inf'),float('nan')):
            for name in ('gauss_tolerance','magnetic_divergence_tolerance'):
                args=dict(gauss_tolerance=1e-10,magnetic_divergence_tolerance=1e-10)
                args[name]=value
                with self.assertRaises(ValueError):runner.validate_constraint_settings(**args)
        for value in (0,-1,1.5,True):
            with self.assertRaises(ValueError):runner.validate_constraint_settings(1e-10,1e-10,value)

    def test_reporting_only_nonfinite_and_empty_regions(self):
        values=dict(gauss=0.,magnetic_divergence=0.,horizon_gauss=np.inf,
                    horizon_magnetic_divergence=np.nan,horizon_magnetic_divergence_cells=0)
        runner.check_constraints(values)
        payload=runner.constraint_payload(values)
        self.assertIsNone(payload['constraints']['horizon_gauss'])
        self.assertFalse(payload['constraint_validity']['horizon_gauss'])
        self.assertFalse(payload['constraint_validity']['horizon_magnetic_divergence'])
        json.dumps(payload,allow_nan=False)
        for name in ('gauss','magnetic_divergence'):
            for invalid in (np.nan,np.inf):
                with self.assertRaises(FloatingPointError):runner.check_constraints(dict(values,**{name:invalid}),True)
            with self.assertRaises(FloatingPointError):runner.check_constraints(dict(values,**{name+'_cells':0}),True)


class TestConstraintSchedule(unittest.TestCase):
    """Exercise host scheduling and checkpoint transactions without particle kernels."""
    def run_driver(self,directory,*,start=0,steps=205,fail_at=None,stop_at=None,waive=False):
        from contextlib import ExitStack
        from types import SimpleNamespace
        import signal
        p=SimulationParameters(devices=1)
        static=SimpleNamespace(field_mesh=SimpleNamespace(devices=np.array([],dtype=object)))
        pts=SimpleNamespace(active=np.zeros((1,1,1,2,1),bool))
        boundary=SimpleNamespace(absorbed_count=np.zeros((2,2)),absorbed_charge=np.zeros((2,2)),
                                 removed_grid_charge=0.,radial_current_outflow=np.zeros(2))
        handlers={}
        def install(sig,handler):
            old=handlers.get(sig,signal.SIG_DFL);handlers[sig]=handler;return old
        def execute(particles,fields,key,index):
            if stop_at==index+1:handlers[signal.SIGTERM](signal.SIGTERM,None)
            return particles,index+1,index+1,(np.zeros(1),np.zeros(1),np.zeros(1),boundary)
        def measure(particles,species,fields,*args):
            return dict(gauss=1e-5 if fail_at is not None and fields>=fail_at else 0.,
                        magnetic_divergence=0.,horizon_gauss=np.inf,outer_boundary_gauss=1.)
        def save(path,particles,fields,key,step,p,run_metadata=None):
            np.savez(path,step=step,marker=fields,key=key,run_metadata=json.dumps(run_metadata or {}))
        with ExitStack() as stack:
            for name,replacement in [('make_step',lambda *a:execute),('constraint_residuals',measure),
                    ('check_species',lambda *a:None),('check_sharding',lambda *a:None),
                    ('finite_state',lambda *a:True),('save_checkpoint',save),
                    ('diagnostics',lambda *a:dict(marker=a[2])),('plot_diagnostics',lambda *a:None),
                    ('comparison_errors',lambda *a:{})]:
                stack.enter_context(patch.object(runner,name,replacement))
            stack.enter_context(patch.object(runner.jax,'jit',lambda f:f))
            stack.enter_context(patch.object(runner.signal,'signal',install))
            return runner.evolve(pts,None,start,start,p,static,SimpleNamespace(dt=.001),None,directory,
                                 start_step=start,steps=steps,allow_divergence_errors=waive)

    def test_absolute_schedule_and_restart(self):
        with TemporaryDirectory() as out:
            root=Path(out)
            for label,start,steps,expected in [('first',0,205,[0,100,200,205]),
                                              ('restart',153,52,[153,200,205]),
                                              ('initialize',0,0,[0])]:
                directory=root/label
                self.run_driver(directory,start=start,steps=steps)
                rows=[json.loads(s) for s in (directory/'progress.jsonl').read_text().splitlines()]
                records=[row for row in rows if row['event']=='constraints']
                self.assertEqual([row['step'] for row in records],expected)
                self.assertTrue(all(row['status']=='passed' for row in records))
                self.assertTrue(all(row['constraints']['horizon_gauss'] is None for row in records))
                manifest=json.loads((directory/'manifest.json').read_text())
                self.assertEqual(manifest['constraint_check_interval'],100)
                self.assertEqual(manifest['constraints_step'],start+steps)
                self.assertEqual(manifest['gauss_tolerance'],1e-10)
                with np.load(directory/'checkpoint.npz') as saved:
                    self.assertEqual(int(saved['step']),start+steps)

    def test_rejected_candidate_and_waiver(self):
        with TemporaryDirectory() as out:
            root=Path(out)
            with self.assertRaisesRegex(RuntimeError,'gauss'):
                self.run_driver(root/'failed',fail_at=200)
            manifest=json.loads((root/'failed/manifest.json').read_text())
            self.assertEqual(manifest['step'],199)
            self.assertEqual(manifest['failed_candidate_step'],200)
            self.assertEqual(manifest['failure_constraints']['gauss'],1e-5)
            self.assertEqual(manifest['constraints_step'],100)
            with np.load(root/'failed/failed_checkpoint.npz') as saved:
                self.assertEqual(int(saved['step']),199)
                self.assertEqual(int(saved['marker']),199)
                self.assertEqual(int(saved['key']),199)
            rows=[json.loads(s) for s in (root/'failed/progress.jsonl').read_text().splitlines()]
            self.assertEqual(rows[-1]['status'],'failed')
            self.assertEqual(rows[-1]['step'],200)
            self.run_driver(root/'waived',fail_at=200,waive=True)
            rows=[json.loads(s) for s in (root/'waived/progress.jsonl').read_text().splitlines()]
            self.assertEqual([r['step'] for r in rows if r.get('status')=='waived'],[200,205])

    def test_signal_stop_checks_final_state(self):
        with TemporaryDirectory() as out:
            self.run_driver(Path(out),stop_at=17)
            rows=[json.loads(s) for s in (Path(out)/'progress.jsonl').read_text().splitlines()]
            self.assertEqual([r['step'] for r in rows if r['event']=='constraints'],[0,17])
            manifest=json.loads((Path(out)/'manifest.json').read_text())
            self.assertEqual(manifest['status'],'stopped')
            self.assertEqual(manifest['constraints_step'],17)


if __name__=='__main__':unittest.main()
