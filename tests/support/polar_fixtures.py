"""Small polar spherical Kerr-Schild runtimes shared by polar and BZ tests."""
from functools import lru_cache
import jax
import jax.numpy as jnp
from tests.support.polar_runtime import SimulationParameters,build_runtime
from PyPIC3D.particles.particle_class import TiledParticles,SpeciesConfig
from PyPIC3D.relativity.metrics.flat import initialize_flat_spherical_metric
jax.config.update('jax_enable_x64',True)

@lru_cache(None)
def polar_runtime(devices=1,order=1,flat=False,nt=16,cap=0.):
    """Keep true axes by default; request caps explicitly for field evolution."""
    p=SimulationParameters(nr=16,ntheta=nt,devices=devices,r_max=4.,sponge_start=3.,
                           skin_depth=.0025,pairs_per_cell=4,maximum_timestep=None,end_time=5.,output_interval=1.,
                           horizon_field_cells=0,polar_cap_angle=cap)
    s,d,m,_=build_runtime(p);s=s._replace(shape_factor=order)
    if flat:
        s=s._replace(metric='flat_spherical',metric_mass=0.,metric_spin=0.)
        m=initialize_flat_spherical_metric(s,d)
    return p,s,d,m

def particle(s,d,theta,r=2.1,charge=1.):
    x=jnp.broadcast_to(jnp.array([r,theta,0.]),s.field_mesh.devices.shape+(1,1,3))
    active=jnp.zeros(x.shape[:-1],bool)
    tx=min(int((r-float(d.grids.center[0][1]))/float(d.dx))//s.tile_shape[0],x.shape[0]-1)
    active=active.at[tx,0,0,0,0].set(True)
    return TiledParticles(x,jnp.zeros_like(x),active),SpeciesConfig(jnp.array([charge]),jnp.ones(1),jnp.ones(1),jnp.ones((1,3),bool))

from PyPIC3D.boundary_conditions.polar import divide

def monopole_field(p, metric, dynamic):
    """Discrete curl of A_phi on D_phi=(C,C,V), yielding B_r=(C,V,V)."""
    theta = dynamic.grids.tiled_center_grid[1][..., None, :, None]
    flux = (-p.B0*jnp.cos(theta+dynamic.dy)+p.B0*jnp.cos(theta))/dynamic.dy
    Br = divide(jnp.broadcast_to(flux, metric.B[0].sqrt_gamma.shape)*dynamic.dy*dynamic.dz,
                metric.geometry.B_area[0])
    return (Br, jnp.zeros_like(Br), jnp.zeros_like(Br))
