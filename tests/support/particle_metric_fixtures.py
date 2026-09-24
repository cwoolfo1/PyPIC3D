"""Small supplied-metric fixtures for production regression tests."""
from types import SimpleNamespace
from functools import lru_cache
import numpy as np
import jax
import jax.numpy as jnp
from PyPIC3D.utilities.parameters import build_static_parameters, build_dynamic_parameters
from PyPIC3D.utilities.grids import build_yee_grid, build_tiled_yee_grids
from PyPIC3D.relativity.core import B_FIELD_LOCATIONS
from PyPIC3D.relativity.interpolate_metric import interpolate_metric
from PyPIC3D.relativity.flat import initialize_flat_cartesian_metric, initialize_flat_spherical_metric
from PyPIC3D.boundary_conditions.polar import refresh_vector
from PyPIC3D.pusher.hybrid_boris_geodesic import magnetic_boris_rotation, gather_vector

jax.config.update('jax_enable_x64', True)
PERIOD = 2*np.pi*np.sqrt(1.16)

@lru_cache(maxsize=1)
def make_runtime(chart, nr=64, ntheta=128):
    spherical=chart=='spherical'
    nx,ny=(nr,ntheta) if spherical else (ntheta,ntheta)
    cfg=dict(Nx=nx,Ny=ny,Nz=1,x_min=1. if spherical else -4.,y_min=0. if spherical else -4.,
             z_min=0. if spherical else -4.,x_wind=3. if spherical else 8.,
             y_wind=np.pi if spherical else 8.,z_wind=2*np.pi if spherical else 8.,dt=PERIOD/4096)
    cfg.update(dx=cfg['x_wind']/nx,dy=cfg['y_wind']/ny,dz=cfg['z_wind'])
    s=build_static_parameters(dict(**cfg,solver='static_metric',metric='flat_spherical' if spherical else 'flat_cartesian',
        metric_mass=0.,metric_spin=0.,particle_pusher='hybrid_boris_geodesic',current_deposition='GR_esirkepov',
        current_filter='none',shape_factor=1,guard_cells=3,tile_shape=(nx,ny,1),
        boundary_conditions=(3,4,0) if spherical else (3,3,3),
        particle_boundary_conditions=(2,4,0) if spherical else (2,2,2)))
    center,vertex=build_yee_grid(SimpleNamespace(**cfg))
    if spherical:
        t=jnp.arange(-1,ny+1,dtype=jnp.float64)*cfg['dy']
        center=(center[0],t,center[2]);vertex=(vertex[0],t+cfg['dy']/2,vertex[2])
    tc,tv=build_tiled_yee_grids(s,SimpleNamespace(**cfg,grids=SimpleNamespace(center=center,vertex=vertex)))
    d=build_dynamic_parameters(dict(**cfg,grids=dict(center=center,vertex=vertex,tiled_center_grid=tc,tiled_vertex_grid=tv)))
    m=(initialize_flat_spherical_metric if spherical else initialize_flat_cartesian_metric)(s,d)
    z=jnp.zeros_like(m.center.sqrt_gamma);B=[]
    for i,loc in enumerate(B_FIELD_LOCATIONS):
        rg=(tc if loc[0]=='C' else tv)[0][...,:,None,None]
        tg=(tc if loc[1]=='C' else tv)[1][...,None,:,None]
        b=(jnp.cos(tg) if i==0 else -jnp.sin(tg)/rg if i==1 else z) if spherical else (z+1 if i==2 else z)
        B.append(jnp.broadcast_to(b,z.shape))
    B=refresh_vector(tuple(B),s,B_FIELD_LOCATIONS,'B') if spherical else tuple(B)
    return s,d,m,(z,)*3,B


def tile_grids(d):
    return (tuple(a[0,0,0] for a in d.grids.tiled_center_grid),
            tuple(a[0,0,0] for a in d.grids.tiled_vertex_grid))


def sample_metric(q,m,s,d):
    center=jax.tree.map(lambda a:a[0,0,0],m.center)
    return interpolate_metric(center,q,tile_grids(d)[0],s.metric,(True,True,False),(3,3,3))


def gather_B(q,B,s,d):
    return gather_vector(tuple(b[0,0,0] for b in B),B_FIELD_LOCATIONS,q,*tile_grids(d),
                         s.shape_factor,(True,True,False),(3,3,3))


def norm(u,m):
    return jnp.einsum('...i,...ij,...j->...',u,m.gamma_inv,u)


def sampled_rotation_checks(ntheta=128):
    rows=[];s,d,m,D,B=make_runtime('spherical',64,ntheta)
    for order in (1,2):
        s=s._replace(shape_factor=order)
        for label,t in [('away',.8),('north_half',float(d.dy)/2),('north_close',float(d.dy)*.05),
                        ('south_half',np.pi-float(d.dy)/2),('south_close',np.pi-float(d.dy)*.05)]:
            q=jnp.array([[2.37,t,.1]]);a=sample_metric(q,m,s,d)
            b=gather_B(q,B,s,d);u=jnp.einsum('...ij,j->...i',jnp.linalg.cholesky(a.gamma),jnp.array([.4,.3,.2]))
            derivative=a.grad_gamma_inv[0]
            differentiated=jnp.moveaxis(jax.jacfwd(lambda x:sample_metric(x[None,:],m,s,d).gamma_inv[0])(q[0]),-1,0)
            derivative_error=float(jnp.linalg.norm(derivative-differentiated)/jnp.maximum(jnp.linalg.norm(differentiated),1e-30))
            for control,met in [('production_sample',a),('inverse_consistent_rotation_only',a._replace(gamma_inv=jnp.linalg.inv(a.gamma)))]:
                for charge in (-1.,1.):
                    for dt in (.01,.5,5.,50.):
                        v=magnetic_boris_rotation(u,b,met,jnp.array([charge]),dt)
                        back=magnetic_boris_rotation(v,b,met,jnp.array([charge]),-dt)
                        scale=jnp.sqrt(norm(u,met)*jnp.einsum('...i,...ij,...j->...',b,met.gamma,b))
                        rows.append(dict(shape=order,location=label,control=control,charge=charge,dt=dt,
                            inverse_defect=float(jnp.linalg.norm(met.gamma@met.gamma_inv-jnp.eye(3))),
                            norm_error=float(jnp.max(jnp.abs(norm(v,met)/norm(u,met)-1))),
                            parallel_error=float(jnp.max(jnp.abs(jnp.sum((v-u)*b,axis=-1))/scale)),
                            reversal_error=float(jnp.linalg.norm(back-u)/jnp.linalg.norm(u)),
                            derivative_discrepancy=derivative_error))
    return rows

