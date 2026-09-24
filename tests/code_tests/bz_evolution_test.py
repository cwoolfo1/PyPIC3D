"""Fresh BZ evolution, constraint checks, and injection/output scheduling."""
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import sys
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from demos.static_metric_relativity.bz_monopole import run_bz_monopole as runner
from demos.static_metric_relativity.bz_monopole.simulation_parameters import SimulationParameters,build_runtime
from demos.static_metric_relativity.bz_monopole.plasma_injector import empty_particles

class TestEvolution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=SimulationParameters(nr=16,ntheta=16,devices=2,r_max=4.,sponge_start=3.,
                                  skin_depth=.0025,pairs_per_cell=4,maximum_timestep=None,
                                  end_time=5.,output_interval=1.,backend='cpu',current_filter_passes=0,
                                  horizon_field_cells=0)
        cls.s,cls.d,cls.m,_=build_runtime(cls.p)
        cls.pts,cls.sp=empty_particles(cls.p,cls.s)
        cls.fields,cls.bg=runner.initialize_fields(cls.p,cls.s,cls.d,cls.m)
        cls.key=jax.random.PRNGKey(5)

    def test_fresh_evolution_outputs_and_failure(self):
        s,d=self.s,self.d
        p=replace(self.p,end_time=3.5*float(d.dt),output_interval=2*float(d.dt),
                  allow_divergence_errors=True,constraint_check_interval=1)
        execute=runner.make_step(p,self.sp,s,d,self.bg)
        expected_pts,expected_fields,expected_key=self.pts,self.fields,self.key
        for step in range(4):
            expected_pts,expected_fields,expected_key,_=execute(expected_pts,expected_fields,expected_key,step)
        with TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(runner,'make_step',return_value=execute):
                result=runner.evolve(self.pts,self.sp,self.fields,self.key,p,s,d,self.bg,root/'fresh')
            for actual,expected in zip(jax.tree.leaves(result[:3]),
                                       jax.tree.leaves((expected_pts,expected_fields,expected_key))):
                np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12,equal_nan=True)
            self.assertEqual(result[3],4)
            output=root/'fresh'
            self.assertEqual(len(list(output.glob('snapshot_*.npz'))),3)
            self.assertFalse(list(output.glob('*.json*')))
            self.assertFalse(list(output.glob('*checkpoint*')))
            with np.load(output/'final_state.npz',allow_pickle=False) as saved:
                self.assertEqual(int(saved['step']),4)
                self.assertEqual(float(saved['time']),4*float(d.dt))
                np.testing.assert_array_equal(saved['active'],expected_pts.active)
                np.testing.assert_array_equal(saved['x'],expected_pts.x)
                np.testing.assert_array_equal(saved['u'],expected_pts.u)
                for name,vector in (('D',expected_fields[0]),('B',expected_fields[1]),('J',expected_fields[2])):
                    for i,value in enumerate(vector):
                        np.testing.assert_array_equal(saved[f'{name}_{i}'],value)
                for name,value in zip(('charge','mass','weight','update_x'),self.sp):
                    np.testing.assert_array_equal(saved['species_'+name],value)
                self.assertTrue(all(saved[key].dtype.kind!='O' for key in saved.files))
            from demos.static_metric_relativity.bz_monopole.plot_entity_bz import load_snapshot,Normalization
            data=load_snapshot(output/'diagnostics.npz')
            self.assertAlmostEqual(Normalization.from_snapshot(data).B0,p.B0)
            self.assertTrue((output/'figure6_diagnostics.png').exists())
            def fail_second(pts,fields,key,index):
                if index==1:raise RuntimeError('injected test failure')
                return execute(pts,fields,key,index)
            with patch.object(runner,'make_step',return_value=fail_second):
                with self.assertRaisesRegex(RuntimeError,'injected test failure'):
                    runner.evolve(self.pts,self.sp,self.fields,self.key,p,s,d,self.bg,root/'failed')
            self.assertEqual([f.name for f in (root/'failed').iterdir()],['snapshot_000000000000.npz'])
        first=execute(self.pts,self.fields,self.key,0)
        broken=first[0]._replace(x=first[0].x.at[0,0,0,0,0,1].set(0.))
        self.assertTrue(bool(broken.active[0,0,0,0,0]))
        with self.assertRaises(Exception):execute(broken,first[1],first[2],1)

    def test_divergence_waiver_does_not_allow_nonfinite(self):
        with self.assertRaisesRegex(RuntimeError,'[Dd]ivergence'):
            runner.check_constraints(dict(gauss=.01,magnetic_divergence=0.),False)
        runner.check_constraints(dict(gauss=.01,magnetic_divergence=0.),True)
        with self.assertRaises(FloatingPointError):
            runner.check_constraints(dict(gauss=np.nan,magnetic_divergence=0.),True)

    def test_exterior_mask_excludes_boundaries_but_keeps_tile_seams(self):
        p=replace(self.p,sponge_start=3.8)
        g=self.s.guard_cells
        zero=jnp.zeros_like(self.fields[3])
        dm=runner.exterior_mask(self.m.geometry,self.s,self.d,p)
        bm=runner.exterior_mask(self.m.geometry,self.s,self.d,p,magnetic=True)
        for magnetic,mask in ((False,dm),(True,bm)):
            inside=np.asarray(mask)
            self.assertTrue(inside[0,0,0,g+self.s.tile_shape[0]-1,g+5,g])
            self.assertTrue(inside[1,0,0,g,g+5,g])
            self.assertFalse(inside[0,0,0,g,g+5,g])
            self.assertFalse(inside[0,0,0,g+5,g,g])
            self.assertFalse(inside[1,0,0,g+self.s.tile_shape[0]-1,g+5,g])
            grid=self.d.grids.tiled_vertex_grid if magnetic else self.d.grids.tiled_center_grid
            radius=np.broadcast_to(np.asarray(grid[0])[..., :,None,None],zero.shape)
            self.assertFalse(np.any(inside[radius<self.p.horizon]))
        # the first owned radial plane outside the horizon has accepted cells
        owned=np.asarray(self.m.geometry.charge_owned)
        radius=np.broadcast_to(np.asarray(self.d.grids.tiled_center_grid[0])[..., :,None,None],zero.shape)
        first=radius[owned & (radius>=self.p.horizon)].min()
        self.assertTrue(np.any(np.asarray(dm) & (radius==first)))
        # Enormous errors and fields outside the exterior neither fail the check
        # nor dilute an exterior error through the normalization.
        fields=list(self.fields)
        fields[1]=tuple(jnp.where(bm,0.,1e18) for _ in range(3))
        outside_d=jnp.where(dm,0.,1e12)
        outside_b=jnp.where(bm,0.,1e12)
        def measure(dd,bb):
            with patch.object(runner,'divergence',side_effect=[dd,bb]), \
                 patch.object(runner,'compute_rho',return_value=zero):
                return {k:float(v) for k,v in runner.constraint_residuals(
                    self.pts,self.sp,fields,self.s,self.d,p).items()}
        runner.check_constraints(measure(outside_d,outside_b),False)
        seam=(1,0,0,g,g+5,g)
        r=measure(outside_d.at[seam].set(1e-5),outside_b)
        self.assertAlmostEqual(r['gauss'],1e-5,places=12)
        with self.assertRaisesRegex(RuntimeError,'Exterior.*gauss'):runner.check_constraints(r,False)
        r=measure(outside_d,outside_b.at[seam].set(1e-5))
        self.assertAlmostEqual(r['magnetic_divergence'],1e-5,places=12)
        with self.assertRaisesRegex(RuntimeError,'magnetic_divergence'):runner.check_constraints(r,False)

    def test_empty_exterior_and_snapshot_residuals(self):
        p=replace(self.p,sponge_start=1.9)
        values=runner.constraint_residuals(self.pts,self.sp,self.fields,self.s,self.d,p)
        with self.assertRaises(FloatingPointError):runner.check_constraints(values,True)
        snap=runner.diagnostics(self.pts,self.sp,self.fields,self.p,self.s,self.d)
        for name in ('gauss_relative','divB_relative'):
            self.assertTrue(np.isfinite(snap[name]))


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

    def test_nonfinite_or_empty_exterior_always_fails(self):
        values=dict(gauss=0.,magnetic_divergence=0.)
        runner.check_constraints(values)
        for name in ('gauss','magnetic_divergence'):
            for invalid in (np.nan,np.inf):
                with self.assertRaises(FloatingPointError):runner.check_constraints(dict(values,**{name:invalid}),True)
            with self.assertRaises(FloatingPointError):runner.check_constraints(dict(values,**{name+'_cells':0}),True)


class TestConstraintSchedule(unittest.TestCase):
    """Exercise fresh-run scheduling and failure paths without particle kernels."""
    def run_driver(self,directory,*,end_time=.2045,interval=.1,fail_at=None,interrupt_at=None,waive=False):
        p=SimulationParameters(end_time=end_time,output_interval=interval,
                               allow_divergence_errors=waive)
        pts=SimpleNamespace(active=np.zeros((1,1,1,2,1),bool))
        boundary=SimpleNamespace(absorbed_count=np.zeros((2,2)),absorbed_charge=np.zeros((2,2)),
                                 removed_grid_charge=0.,radial_current_outflow=np.zeros(2))
        self.checked=[]
        def execute(particles,fields,key,index):
            if interrupt_at==index+1:raise KeyboardInterrupt()
            return particles,index+1,index+1,(np.zeros(1),np.zeros(1),np.zeros(1),boundary)
        def measure(particles,species,fields,*args,**kwargs):
            self.checked.append(fields)
            return dict(gauss=1e-5 if fail_at is not None and fields>=fail_at else 0.,
                        magnetic_divergence=0.)
        def save(path,particles,species,fields,step,*args):
            np.savez(path,step=step,marker=fields)
        with ExitStack() as stack:
            for name,replacement in [('make_step',lambda *a,**kw:execute),('constraint_residuals',measure),
                    ('check_sharding',lambda *a:None),
                    ('finite_state',lambda *a:True),('save_final_state',save),
                    ('output_metadata',lambda *a:dict(dt=.001)),
                    ('diagnostics',lambda *a,**kw:dict(marker=a[2])),('plot_diagnostics',lambda *a:None)]:
                stack.enter_context(patch.object(runner,name,replacement))
            stack.enter_context(patch.object(runner.jax,'jit',lambda f:f))
            self.bar=stack.enter_context(patch.object(runner,'tqdm'))
            return runner.evolve(pts,None,0,0,p,None,SimpleNamespace(dt=.001),None,directory)

    def test_fresh_schedule_and_output_cadence(self):
        with TemporaryDirectory() as out:
            directory=Path(out)
            self.run_driver(directory)
            self.assertEqual(self.checked,[0,100,200,205])
            self.assertEqual(self.bar.call_args.kwargs['total'],205)
            self.bar.return_value.__exit__.assert_called_once()
            self.assertEqual(self.bar.return_value.__enter__.return_value.update.call_count,205)
            names={f.name for f in directory.iterdir()}
            self.assertEqual(names,{'snapshot_000000000000.npz','snapshot_000000000100.npz',
                                    'snapshot_000000000200.npz','snapshot_000000000205.npz',
                                    'diagnostics.npz','final_state.npz'})
            with np.load(directory/'diagnostics.npz',allow_pickle=False) as data:
                self.assertEqual(int(data['marker']),205)
                self.assertAlmostEqual(float(data['time']),.205)
        with TemporaryDirectory() as out:
            self.run_driver(Path(out),end_time=.002,interval=.0001)
            self.assertEqual(len(list(Path(out).glob('snapshot_*.npz'))),3)

    def test_rejected_candidate_and_waiver(self):
        with TemporaryDirectory() as out:
            root=Path(out)
            with self.assertRaisesRegex(RuntimeError,'Step 200.*gauss'):
                self.run_driver(root/'failed',fail_at=200)
            self.assertEqual(self.checked,[0,100,200])
            self.bar.return_value.__exit__.assert_called_once()
            self.assertEqual({p.name for p in (root/'failed').iterdir()},
                             {'snapshot_000000000000.npz','snapshot_000000000100.npz'})
            self.run_driver(root/'waived',fail_at=200,waive=True)
            self.assertEqual(self.checked,[0,100,200,205])

    def test_interruption_leaves_only_completed_snapshots(self):
        with TemporaryDirectory() as out:
            with self.assertRaises(KeyboardInterrupt):
                self.run_driver(Path(out),interrupt_at=17)
            self.bar.return_value.__exit__.assert_called_once()
            self.assertEqual([p.name for p in Path(out).iterdir()],['snapshot_000000000000.npz'])

    def test_existing_output_is_rejected_before_initialization(self):
        with TemporaryDirectory() as out:
            root=Path(out);original=root/'keep.txt';original.write_text('keep')
            with patch.object(runner,'build_runtime') as build:
                with self.assertRaises(FileExistsError):
                    runner.run(SimulationParameters(output_directory=str(root)))
                build.assert_not_called()
            self.assertEqual(original.read_text(),'keep')


class TestInjectionSchedule(unittest.TestCase):
    def draw_events(self, dt):
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
                                    sponge=False)
            key = jax.random.PRNGKey(17)
            for index in range(round(.3/dt)):
                _, _, key, _ = step(None, fields, key, index)
        return np.array(draws)

    def test_matched_physical_events_share_random_candidates(self):
        np.testing.assert_array_equal(self.draw_events(.002), self.draw_events(.001))


class TestRunnerArguments(unittest.TestCase):
    def test_runner_rejects_command_line_arguments(self):
        for arguments in (['--restart','old.npz'],['--end-time','1'],['--mode','run'],['--output','data']):
            with patch.object(sys,'argv',['bz',*arguments]), \
                 self.assertRaisesRegex(SystemExit,'takes no arguments'):
                runner.main()

if __name__=='__main__':unittest.main()
