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
    theta=np.linspace(p.polar_cap_angle,np.pi-p.polar_cap_angle,65)
    shape=(len(r),len(theta))
    return dict(r=r,theta=theta,time=time,
        Hphi=np.broadcast_to(-p.B0*p.spin*np.sin(theta)**2/8,shape),
        omega=np.full(shape,p.omega_h/2),radial_flux=np.broadcast_to(np.sin(theta),shape),
        luminosity=np.full(len(r),p.B0**2*p.omega_h**2/6),
        spin=p.spin,B0=p.B0,omega_h=p.omega_h,r_min=p.r_min,r_max=p.r_max,
        sponge_start=p.sponge_start,nr=p.nr,ntheta=p.ntheta,skin_depth=p.skin_depth,
        pairs_per_cell=p.pairs_per_cell,polar_cap_angle=p.polar_cap_angle)


class TestBZOutput(unittest.TestCase):
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
            np.testing.assert_allclose(angles[:,0],p.polar_cap_angle,rtol=0,atol=1e-14)
            np.testing.assert_allclose(angles[:,-1],np.pi-p.polar_cap_angle,rtol=0,atol=1e-14)
            import matplotlib.pyplot as plt
            plt.close(figure)
            for cap in (-.1,np.pi/2,np.nan,[.1]):
                np.savez(path,**dict(data,polar_cap_angle=cap))
                with self.assertRaisesRegex(ValueError,'polar_cap_angle'):
                    plotter.load_snapshot(path)
            np.savez(path,**dict(data,polar_cap_angle=.2))
            with self.assertRaisesRegex(ValueError,'declared domain'):
                plotter.load_snapshot(path)
            legacy=snapshot(replace(p,polar_cap_angle=0.),0.)
            del legacy['polar_cap_angle']
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
