"""Off-axis BZ field experiment using the original point-metric solver.

Both Yee node families miss the poles by a quarter cell. No polar geometry,
Entity averaging, polar-cap curl, or inner field-plane copying is used.
Reflection uses physical coordinates, not an axis-node index convention.
This field-only experiment does not yet supply off-axis particle deposition.
"""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import math

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from demos.bz_monopole import run_bz_monopole as runner
from demos.bz_monopole.simulation_parameters import SimulationParameters, shard_array
from demos.bz_monopole.plasma_injector import empty_particles
from PyPIC3D.relativity.core import D_FIELD_LOCATIONS, B_FIELD_LOCATIONS
from PyPIC3D.relativity.kerr_schild import initialize_kerr_schild_spherical_metric
from PyPIC3D.solvers.gr_static import static_metric as solver
from PyPIC3D.solvers.gr_static import time_loop as loop
from PyPIC3D.utilities.grids import build_yee_grid, build_tiled_yee_grids
from PyPIC3D.utilities.parameters import build_static_parameters, build_dynamic_parameters


def build_runtime(p):
    jax.config.update('jax_enable_x64', True)
    config = dict(Nx=p.nr, Ny=p.ntheta, Nz=1, x_wind=p.r_max-p.r_min,
                  y_wind=math.pi, z_wind=2*math.pi, x_min=p.r_min,
                  y_min=p.dtheta/4, z_min=0., dx=p.dr, dy=p.dtheta,
                  dz=2*math.pi, dt=p.maximum_timestep)
    static = build_static_parameters(dict(
        **config, name='bz_offset_experiment', solver='static_metric',
        metric='kerr_schild_spherical', metric_mass=1., metric_spin=p.spin,
        particle_pusher='hybrid_boris_geodesic', current_deposition='GR_esirkepov',
        current_filter='none', shape_factor=1, guard_cells=3,
        tile_shape=(p.nr,p.ntheta,1), boundary_conditions=(3,3,0),
        particle_boundary_conditions=(2,1,0), particle_batch_size=8192,
        particle_coordinates='cartesian'))
    center, vertex = build_yee_grid(SimpleNamespace(**config))
    tc,tv = build_tiled_yee_grids(static, SimpleNamespace(
        **config, grids=SimpleNamespace(center=center,vertex=vertex)))
    config['grids'] = dict(center=center,vertex=vertex,tiled_center_grid=tc,tiled_vertex_grid=tv)
    dynamic = build_dynamic_parameters(config)
    metric = initialize_kerr_schild_spherical_metric(static,dynamic,spin=p.spin)
    metric = jax.tree.map(lambda a:shard_array(a,static),metric)
    assert metric.geometry is None
    assert all(bool(jnp.all(m.sqrt_gamma != 0)) for m in (*metric.D,*metric.B,metric.center,metric.vertex))
    return static,dynamic,metric


def reflection(vector, locations, static, dynamic):
    """Quadratic extension from the reflected physical point, theta -> -theta.

    Quarter-cell offsets put the reflected point between same-component nodes.
    At most half-cell extrapolation is needed for the nearest ghost point.
    """
    result=[]
    g,n=static.guard_cells,static.tile_shape[1]
    for component,(value,loc) in enumerate(zip(vector,locations)):
        grids=dynamic.grids.tiled_center_grid if loc[1]=='C' else dynamic.grids.tiled_vertex_grid
        theta=np.asarray(grids[1])[0,0,0]
        indices=np.concatenate((np.arange(g),np.arange(g+n,len(theta))))
        reflected=np.where(theta[indices]<0,-theta[indices],2*np.pi-theta[indices])
        left=np.clip(np.floor((reflected-theta[g])/float(dynamic.dy)).astype(int)+g-1,g,g+n-3)
        nodes=left[:,None]+np.arange(3)
        x=theta[nodes]
        weights=np.ones_like(x)
        for k in range(3):
            for j in range(3):
                if k!=j:weights[:,k]*=(reflected-x[:,j])/(x[:,k]-x[:,j])
        gathered=jnp.take(value,jnp.asarray(nodes),axis=4)
        ghosts=jnp.sum(gathered*jnp.asarray(weights)[None,None,None,None,:,:,None],axis=5)
        result.append(value.at[:,:,:,:,indices,:].set((-1 if component==1 else 1)*ghosts))
    return tuple(result)


@contextmanager
def reflected_boundaries(static,dynamic):
    """Patch only experiment boundaries; production field arithmetic is intact."""
    original_D,original_B=solver.update_D_relativity,solver.update_B_relativity
    original_refresh=runner.refresh_fields
    def update_D(D,H,J,metric,s,d,dt):
        return reflection(original_D(D,H,J,metric,s,d,dt),D_FIELD_LOCATIONS,s,d)
    def update_B(E,B,metric,s,d,dt):
        return reflection(original_B(E,B,metric,s,d,dt),B_FIELD_LOCATIONS,s,d)
    def refresh(vector,locations,metric,s):
        return reflection(original_refresh(vector,locations,metric,s),locations,s,dynamic)
    from contextlib import ExitStack
    with ExitStack() as stack:
        for module in (solver,loop,runner):
            stack.enter_context(patch.object(module,'update_D_relativity',update_D))
            stack.enter_context(patch.object(module,'update_B_relativity',update_B))
        stack.enter_context(patch.object(runner,'refresh_fields',refresh))
        yield


def vacuum_case(output, end_time=50., spin=.2):
    output=runner.prepare_output_directory(output)
    p=SimulationParameters(pairs_per_cell=1,capacity_factor=0,backend='cpu',
                           spin=spin,field_interpolation='physical',horizon_field_cells=0,
                           current_filter_passes=0,end_time=end_time)
    print(f'Building quarter-cell offset, spin={spin}',flush=True)
    s,d,m=build_runtime(p)
    pts,sp=empty_particles(p,s)
    g=s.guard_cells
    r=d.grids.tiled_center_grid[0][..., :,None,None]
    theta=d.grids.tiled_center_grid[1][...,None,:,None]
    owned=jnp.zeros_like(m.center.sqrt_gamma,dtype=bool).at[:,:,:,g:-g,g:-g,g:g+1].set(True)
    exterior=owned&(r>=p.horizon)&(r<p.sponge_start-2*p.dr)&(theta>=2*p.dtheta)&(theta<=np.pi-2*p.dtheta)
    def divergence(vector,metrics,forward=False):
        result=jnp.zeros_like(vector[0])
        for i,h in enumerate((d.dx,d.dy,d.dz)):
            flux=vector[i]*metrics[i].sqrt_gamma
            result+=(jnp.roll(flux,-1,axis=i+3)-flux)/h if forward else (flux-jnp.roll(flux,1,axis=i+3))/h
        return result
    with reflected_boundaries(s,d):
        fields,background=runner.initialize_fields(p,s,d,m)
        def advance(_,fs):
            # The metric is fixed; close over it to avoid carrying its storage.
            fs=fs[:6]+(m,)+fs[7:]
            _,fs=loop.time_loop_static_metric(pts,sp,fs,s,d)
            return runner.apply_sponge(fs,background,p,s,d)
        chunk=jax.jit(lambda fs,count:jax.lax.fori_loop(0,count,advance,fs))
        measure=jax.jit(lambda fs:(jnp.max(jnp.where(exterior,jnp.abs(divergence(fs[0],m.D)),0.))/p.B0,
                                 jnp.max(jnp.where(exterior,jnp.abs(divergence(fs[1],m.B,True)),0.))/p.B0))
        records=[]
        target=math.ceil(end_time/float(d.dt));interval=round(1/float(d.dt))
        def record(index):
            arrays=[np.asarray(a) for a in (*fields[0],*fields[1])]
            finite=all(np.isfinite(a).all() for a in arrays)
            amplitude=np.max(np.abs(np.asarray(arrays)))/p.B0
            errors=measure(fields)
            records.append((index*float(d.dt),finite,amplitude,*map(float,errors)))
            np.savez(output/'history.npz',measurements=np.asarray(records),
                     columns=np.asarray(['time','finite','max_field_over_B0','gauss_over_B0','divB_over_B0']),
                     theta_offset_cells=.25,spin=p.spin,dt=float(d.dt),nr=p.nr,ntheta=p.ntheta)
            return finite and amplitude<1e6
        index=0;healthy=record(index)
        with tqdm(total=target,desc=f'offset spin={spin}',mininterval=1.,unit='step') as bar:
            while index<target and healthy:
                count=min(interval,target-index)
                fields=chunk(fields,count);jax.block_until_ready(fields)
                index+=count;healthy=record(index)
                bar.update(count);bar.set_postfix(time=index*float(d.dt),amplitude=records[-1][2],refresh=False)
        np.savez(output/'fields.npz',**{f'{name}_{i}':np.asarray(a)
                 for name,v in zip(('D','B'),fields[:2]) for i,a in enumerate(v)},
                 time=index*float(d.dt),theta=np.asarray(d.grids.center[1]),r=np.asarray(d.grids.center[0]))
        print(f'Quarter offset spin={spin}: {records[-1]}',flush=True)
    return np.asarray(records)


if __name__=='__main__':
    root=Path('demos/bz_monopole/runs/offset_grid_ablation_v3')
    vacuum_case(root/'spin0',spin=0.,end_time=10.)
    vacuum_case(root/'spin02',spin=.2,end_time=50.)
