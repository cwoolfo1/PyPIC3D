"""Self-contained NumPy output drives figures without sidecar files."""
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from demos.bz_monopole import plot_entity_bz as plotter, animate_bz
from demos.bz_monopole.simulation_parameters import SimulationParameters


def snapshot(p, time):
    r=np.linspace(p.r_min,p.r_max,64)
    theta=np.linspace(0,np.pi,65)
    shape=(len(r),len(theta))
    return dict(r=r,theta=theta,time=time,
        Hphi=np.broadcast_to(-p.B0*p.spin*np.sin(theta)**2/8,shape),
        omega=np.full(shape,p.omega_h/2),radial_flux=np.broadcast_to(np.sin(theta),shape),
        luminosity=np.full(len(r),p.B0**2*p.omega_h**2/6),
        spin=p.spin,B0=p.B0,omega_h=p.omega_h,r_min=p.r_min,r_max=p.r_max,
        sponge_start=p.sponge_start,nr=p.nr,ntheta=p.ntheta,skin_depth=p.skin_depth,
        pairs_per_cell=p.pairs_per_cell)


def test_plots_and_animation_need_only_numpy_files(tmp_path):
    p=SimulationParameters()
    for step in (0,1):
        np.savez(tmp_path/f'snapshot_{step:012d}.npz',**snapshot(p,float(step)))
    assert plotter.main(['--data',str(tmp_path),'--all','--dpi','35'])==0
    assert len(list((tmp_path/'entity_plots').glob('*.png')))==2
    animate_bz.main([str(tmp_path),'--times','0','1','--output',str(tmp_path/'evolution.gif'),
                     '--dpi','35'])
    with Image.open(tmp_path/'evolution.gif') as gif:
        assert gif.n_frames==2
    assert (tmp_path/'evolution.png').is_file()
    assert not list(tmp_path.rglob('*.json*'))


def test_snapshot_normalization_is_required_and_validated(tmp_path):
    p=SimulationParameters();data=snapshot(p,0.)
    del data['B0']
    path=tmp_path/'missing.npz';np.savez(path,**data)
    with pytest.raises(ValueError,match='B0'):
        plotter.load_snapshot(path)
    data=snapshot(p,0.);data['B0']=-1
    with pytest.raises(ValueError,match='positive'):
        plotter.Normalization.from_snapshot(data)
