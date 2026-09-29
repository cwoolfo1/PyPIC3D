"""Self-contained NumPy output drives figures without sidecar files."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from PIL import Image
from demos.static_metric_relativity.bz_monopole import plot_entity_bz as plotter
from demos.static_metric_relativity.bz_monopole.simulation_parameters import SimulationParameters
from demos.static_metric_relativity.bz_monopole import animate_bz


def snapshot(p, time):
    r=np.linspace(p.r_min,p.r_max,64)
    theta=np.linspace(p.theta_start,p.theta_end,65)
    shape=(len(r),len(theta))
    return dict(r=r,theta=theta,time=time,
        Hphi=np.broadcast_to(-p.B0*p.spin*np.sin(theta)**2/8,shape),
        omega=np.full(shape,p.omega_h/2),radial_flux=np.broadcast_to(np.sin(theta),shape),
        luminosity=np.full(len(r),p.B0**2*p.omega_h**2/6),
        spin=p.spin,B0=p.B0,omega_h=p.omega_h,r_min=p.r_min,r_max=p.r_max,
        sponge_start=p.sponge_start,nr=p.nr,ntheta=p.ntheta,skin_depth=p.skin_depth,
        pairs_per_cell=p.pairs_per_cell,theta_start=p.theta_start,theta_end=p.theta_end)


class TestBZOutput(unittest.TestCase):
    def test_radial_budget_retains_saved_order_and_accumulates(self):
        from types import SimpleNamespace
        from demos.static_metric_relativity.bz_monopole.output import RadialBoundaryBudget

        budget = RadialBoundaryBudget()
        report = SimpleNamespace(
            absorbed_count=np.array([[1, 2], [3, 4]]),
            absorbed_charge=np.array([[-2., 3.], [-4., 5.]]),
            removed_grid_charge=np.array([.25, -.125]),
            radial_current_outflow=np.array([-6., 7.]),
        )
        for _ in range(2):
            budget.accumulate(report, .5)
        np.testing.assert_array_equal(
            budget.as_array(), [2., 4., 6., 8., -4., 6., -8., 10., .25, -6., 7.]
        )

    def test_diagnostics_respects_configured_guard_depth(self):
        from demos.static_metric_relativity.bz_monopole.simulation_parameters import build_runtime
        from demos.static_metric_relativity.bz_monopole.plasma_injector import empty_particles
        from demos.static_metric_relativity.bz_monopole.run_bz_monopole import initialize_fields, diagnostics

        p = SimulationParameters(nr=16, ntheta=24, r_max=4., sponge_start=3.,
                                 guard_cells=4, horizon_field_cells=0,
                                 theta_start=np.pi/6, theta_end=5*np.pi/6, backend='cpu')
        static, dynamic, metric, _ = build_runtime(p)
        particles, species = empty_particles(p, static)
        fields, _ = initialize_fields(p, static, dynamic, metric)
        data = diagnostics(particles, species, fields, p, static, dynamic)
        self.assertEqual(data['gauss'].shape, (p.nr, p.ntheta + 1))
        self.assertEqual(data['divB'].shape, (p.nr, p.ntheta))
        self.assertEqual(data['Br'].shape, (len(data['r']), len(data['theta'])))

    def test_plots_and_animation_need_only_numpy_files(self):
        with TemporaryDirectory() as directory:
            tmp_path=Path(directory)
            p=SimulationParameters()
            for step in (0,1):
                np.savez(tmp_path/f'snapshot_{step:012d}.npz',**snapshot(p,float(step)))
            self.assertEqual(plotter.main(['--data',str(tmp_path),'--all','--dpi','35']),0)
            self.assertEqual(len(list((tmp_path/'entity_plots').glob('*.png'))),2)
            animate_bz.main([str(tmp_path),'--times','0','1','--output',str(tmp_path/'evolution.gif'),
                             '--dpi','35'])
            with Image.open(tmp_path/'evolution.gif') as gif:
                self.assertEqual(gif.n_frames,2)
            self.assertTrue((tmp_path/'evolution.png').is_file())
            self.assertFalse(list(tmp_path.rglob('*.json*')))

    def test_cap_domain_and_legacy_full_sphere_snapshots(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'snapshot.npz'
            p=SimulationParameters()
            data=snapshot(p,0.)
            np.savez(path,**data)
            loaded=plotter.load_snapshot(path)
            np.testing.assert_array_equal(loaded['theta'],data['theta'])
            figure,_=plotter.make_figure(loaded,plotter.Normalization.from_snapshot(loaded))
            coordinates=figure.axes[0].collections[0].get_coordinates()
            # The map ends on the cap surfaces, without painting the excised wedges.
            angles=np.arctan2(coordinates[...,0],coordinates[...,1])
            np.testing.assert_allclose(angles[:,0],p.theta_start,rtol=0,atol=1e-14)
            np.testing.assert_allclose(angles[:,-1],p.theta_end,rtol=0,atol=1e-14)
            import matplotlib.pyplot as plt
            plt.close(figure)
            for theta in ([np.nan, 1.], [.4, .3], [-.1, 1.], [.1, 3.2]):
                invalid = dict(data, theta=np.asarray(theta))
                np.savez(path, **invalid)
                with self.assertRaises(ValueError):
                    plotter.load_snapshot(path)
            # Historical cap metadata is ignored; coordinates are authoritative.
            np.savez(path, **dict(data, polar_cap_angle=.2))
            np.testing.assert_array_equal(plotter.load_snapshot(path)['theta'], data['theta'])
            legacy=snapshot(replace(p,theta_start=0.,theta_end=np.pi),0.)
            del legacy['theta_start'], legacy['theta_end']
            legacy['allow_divergence_errors'] = np.asarray(True)
            np.savez(path,**legacy)
            np.testing.assert_array_equal(plotter.load_snapshot(path)['theta'],legacy['theta'])

    def test_snapshot_normalization_is_required_and_validated(self):
        with TemporaryDirectory() as directory:
            tmp_path=Path(directory)
            p=SimulationParameters();data=snapshot(p,0.)
            del data['B0']
            path=tmp_path/'missing.npz';np.savez(path,**data)
            with self.assertRaisesRegex(ValueError,'B0'):
                plotter.load_snapshot(path)
            data=snapshot(p,0.);data['B0']=-1
            with self.assertRaisesRegex(ValueError,'positive'):
                plotter.Normalization.from_snapshot(data)


if __name__=='__main__':
    unittest.main()
