"""Axisymmetric polar boundaries in a single physical spherical chart.

C nodes include both 0 and pi; V nodes are halfway between them. The upper
C boundary is an owned plane in the first high halo slot. Source folding is
additive and acts on integrated charge/flux, never on already normalized J.
"""
from typing import NamedTuple
import numpy as np
import jax.numpy as jnp
from PyPIC3D.relativity.core import D_FIELD_LOCATIONS, B_FIELD_LOCATIONS

BC_POLAR = 4


class PolarGeometry(NamedTuple):
    volume: object
    D_area: tuple
    B_area: tuple
    charge_owned: object
    D_owned: tuple
    B_owned: tuple
    center_regular: object
    D_regular: tuple
    B_regular: tuple
    theta_width: object


def enabled(static):
    return static.boundary_conditions[1] == BC_POLAR


def plane(array, index):
    return (slice(None),)*4 + (index, slice(None))


def fold(array, g, n, staggered=False, parity=1):
    """Push ghost contributions onto their reflected owner; keep axis nodes once."""
    upper = g+n-1 if staggered else g+n
    result = array
    for i in list(range(g)) + list(range(upper+1, array.shape[4])):
        j = i-g
        target = (-j-1 if staggered else -j) if i < g else (2*n-1-j if staggered else 2*n-j)
        result = result.at[plane(result,g+target)].add(parity*array[plane(array,i)])
        result = result.at[plane(result,i)].set(0)
    return result


def refresh(array, g, n, staggered=False, parity=1):
    upper = g+n-1 if staggered else g+n
    result = array
    for i in list(range(g)) + list(range(upper+1, array.shape[4])):
        j = i-g
        target = (-j-1 if staggered else -j) if i < g else (2*n-1-j if staggered else 2*n-j)
        result = result.at[plane(result,i)].set(parity*array[plane(array,g+target)])
    return result


def refresh_vector(vector, static, locations, field_kind=None):
    from PyPIC3D.boundary_conditions.ghost_cells import update_tiled_vector_ghost_cells
    # Generic exchange skips theta in polar mode, retaining the owned upper C plane.
    vector = update_tiled_vector_ghost_cells(vector, static, static.guard_cells)
    g, n = static.guard_cells, static.tile_shape[1]
    result=[]
    for i,(a,loc) in enumerate(zip(vector,locations)):
        cells = static.horizon_field_cells
        if cells and field_kind in ('D', 'B'):
            # Entity II sec. 3.3.3: freeze the inner n_filter+1 physical
            # planes and their ghosts to the next plane, inside the horizon.
            # Auxiliary E/H have metric factors and must not be flattened.
            if cells >= static.tile_shape[0]:
                raise ValueError('Horizon boundary reference must lie in the first radial tile')
            a = a.at[0, :, :, :g+cells].set(a[0, :, :, g+cells:g+cells+1])
        if (field_kind == 'D' and i == 2) or (field_kind == 'B' and i == 1):
            a=a.at[plane(a,g)].set(0).at[plane(a,g+n)].set(0)
        result.append(refresh(a,g,n,loc[1]=='V',-1 if i==1 else 1))
    return tuple(result)


def axes(position):
    """Exact coordinate axes (with floating-point pi endpoint recognition)."""
    theta=position[...,1]
    return (jnp.abs(theta)<1e-14) | (jnp.abs(jnp.abs(theta)-jnp.pi)<1e-14)


def safe_grid_provider(provider):
    """Finite unused inverse/derivative storage at singular grid nodes.

    Covariant metric and determinant retain their analytic axis values. No
    particle is evaluated with this wrapper: particle singular stages fail.
    """
    def metric(position):
        axis=axes(position)
        safe=position.at[1].set(jnp.where(axis,jnp.pi/2,position[1]))
        lapse,shift,gamma,inverse,root=provider(safe)
        # At axes evaluate regular lapse and radial/theta metric from a
        # theta=0 covariant expression supplied by the spherical KS formula.
        if hasattr(provider,'polar_values'):
            l0,s0,g0=provider.polar_values(position)
        else:
            l0,s0,g0=lapse,shift,gamma.at[2,:].set(0).at[:,2].set(0)
        return (jnp.where(axis,l0,lapse),jnp.where(axis,s0,shift),
                jnp.where(axis,g0,gamma),jnp.where(axis,jnp.zeros_like(inverse),inverse),
                jnp.where(axis,0.,root))
    return metric


def build_geometry(static,dynamic,metric):
    g=static.guard_cells; nr,nt,nz=static.tile_shape
    shape=metric.center.sqrt_gamma.shape
    rc=dynamic.grids.tiled_center_grid[0][..., :,None,None]
    tc=dynamic.grids.tiled_center_grid[1][..., None,:,None]
    dr,dt,dp=dynamic.dx,dynamic.dy,dynamic.dz
    mass=static.metric_mass if static.metric=='kerr_schild_spherical' else 0.
    a=static.metric_spin if static.metric=='kerr_schild_spherical' else 0.
    nodes,weights=np.polynomial.legendre.leggauss(8)
    def root(r,t):
        sig=r*r+a*a*jnp.cos(t)**2
        return sig*jnp.sqrt(1+2*mass*r/sig)*jnp.sin(t)
    def integrate(r,t,radial,angular):
        lo=jnp.clip(t-dt/2,0,jnp.pi); hi=jnp.clip(t+dt/2,0,jnp.pi)
        result=jnp.zeros_like(r+t)
        for x,w in zip(nodes,weights) if radial else [(0.,2.)]:
            rr=r+dr*x/2 if radial else r
            for y,v in zip(nodes,weights) if angular else [(0.,2.)]:
                tt=(lo+hi)/2+(hi-lo)*y/2 if angular else t
                result=result+w*v*root(rr,tt)/4
        return result*(dr if radial else 1)*((hi-lo) if angular else 1)
    def expand(x): return jnp.broadcast_to(x,shape)
    def own(loc):
        ri=jnp.arange(shape[3])[None,None,None,:,None,None]
        ti=jnp.arange(shape[4])[None,None,None,None,:,None]
        zi=jnp.arange(shape[5])[None,None,None,None,None,:]
        return expand((ri>=g)&(ri<g+nr)&(ti>=g)&(ti<g+nt+(loc[1]=='C'))&(zi==g))
    def regular(loc):
        t=tc+(dt/2 if loc[1]=='V' else 0)
        return expand((jnp.abs(jnp.sin(t))>1e-14))
    def area(loc,i):
        r=rc+(dr/2 if loc[0]=='V' else 0)
        t=tc+(dt/2 if loc[1]=='V' else 0)
        return expand(integrate(r,t,i!=0,i!=1)*(dp if i!=2 else 1))
    volume=expand(integrate(rc,tc,True,True)*dp)
    return PolarGeometry(volume,tuple(area(l,i) for i,l in enumerate(D_FIELD_LOCATIONS)),
                         tuple(area(l,i) for i,l in enumerate(B_FIELD_LOCATIONS)),own(('C',)*3),
                         tuple(own(l) for l in D_FIELD_LOCATIONS),tuple(own(l) for l in B_FIELD_LOCATIONS),
                         regular(('C',)*3),tuple(regular(l) for l in D_FIELD_LOCATIONS),
                         tuple(regular(l) for l in B_FIELD_LOCATIONS),
                         expand(jnp.clip(tc+dt/2,0,jnp.pi)-jnp.clip(tc-dt/2,0,jnp.pi)))


def current_factors(geometry,dynamic):
    # The Cartesian kernel deposits coordinate flux per unit transverse area.
    dr,dt,dp=dynamic.dx,dynamic.dy,dynamic.dz
    return tuple(a/s for a,s in zip(geometry.D_area,(dt*dp,dr*dp,dr*dt)))


def divide(numerator,denominator):
    return jnp.where(denominator!=0,numerator/jnp.where(denominator!=0,denominator,1),0.)


def curl_integrals(vector,static,dynamic,forward):
    """Stokes circulation on each staggered face, including half polar caps."""
    g,n=static.guard_cells,static.tile_shape[1]
    dr,dt,dp=dynamic.dx,dynamic.dy,dynamic.dz
    def diff(a,axis):
        return jnp.roll(a,-1,axis=axis)-a if forward else a-jnp.roll(a,1,axis=axis)
    theta=diff(vector[2],4)
    if not forward:
        theta=theta.at[plane(theta,g)].set(vector[2][plane(theta,g)])
        theta=theta.at[plane(theta,g+n)].set(-vector[2][plane(theta,g+n-1)])
    # Axisymmetric: phi derivatives vanish but all three source components remain.
    return (theta*dp,-diff(vector[2],3)*dp,
            diff(vector[1],3)*dt-diff(vector[0],4)*dr)


def divergence(vector,geometry,static,forward=False):
    area=geometry.B_area if forward else geometry.D_area
    f=tuple(v*a for v,a in zip(vector,area))
    diff=lambda a,ax: jnp.roll(a,-1,axis=ax)-a if forward else a-jnp.roll(a,1,axis=ax)
    theta=diff(f[1],4)
    if not forward:
        g,n=static.guard_cells,static.tile_shape[1]
        theta=theta.at[plane(theta,g)].set(f[1][plane(theta,g)])
        theta=theta.at[plane(theta,g+n)].set(-f[1][plane(theta,g+n-1)])
    return diff(f[0],3)+theta


def update_fields(vector,covariant,current,metric,static,dynamic,dt,magnetic=False):
    geometry=metric.geometry
    # Auxiliary fields need their own halo refresh at every intermediate stage.
    covariant=refresh_vector(covariant,static,
                             D_FIELD_LOCATIONS if magnetic else B_FIELD_LOCATIONS)
    area=geometry.B_area if magnetic else geometry.D_area
    owned=geometry.B_owned if magnetic else geometry.D_owned
    circulation=curl_integrals(covariant,static,dynamic,magnetic)
    result=[]
    for i in range(3):
        increment=(-circulation[i] if magnetic else circulation[i]-4*jnp.pi*area[i]*current[i])
        value=vector[i]+dt*divide(increment,area[i])
        result.append(jnp.where(owned[i],value,vector[i]))
    return refresh_vector(tuple(result),static,B_FIELD_LOCATIONS if magnetic else D_FIELD_LOCATIONS,
                          'B' if magnetic else 'D')


class PolarStepDiagnostics(NamedTuple):
    absorbed_count: object  # (lower/upper radial side, species)
    absorbed_charge: object
    removed_grid_charge: object  # instantaneous charge discarded by slot removal
    radial_current_outflow: object  # outward integrated current through mesh faces
    invalid_push: object


def step_diagnostics(old,new,current,species,metric,static,dynamic):
    """Keep particle loss, cloud removal and mesh flux distinct at absorbers.

    A particle center can leave while its shape still overlaps the mesh. The
    removed-cloud term must be included in a discrete charge budget; treating
    particle-count loss as identical to face flux would hide that boundary sink.
    """
    from PyPIC3D.deposition.rho import compute_rho
    g=static.guard_cells
    rmin=dynamic.grids.center[0][1];rmax=rmin+dynamic.x_wind
    low=new.active&(new.x[...,0]<rmin);high=new.active&(new.x[...,0]>=rmax)
    counts=jnp.stack([jnp.sum(a,axis=(0,1,2,4)) for a in (low,high)])
    charges=counts*(species.charge*species.weight)[None,:]
    lost=new._replace(active=low|high)
    raw=compute_rho(lost,species,jnp.zeros_like(current[0]),static,dynamic)
    removed=jnp.where(metric.geometry.charge_owned,raw*dynamic.dx*dynamic.dy*dynamic.dz,0.)
    flux=current[0]*metric.geometry.D_area[0]
    lower=-jnp.sum(flux[0,0,0,g-1,g:-g+1,g])
    upper=jnp.sum(flux[-1,0,0,g+static.tile_shape[0]-1,g:-g+1,g])
    displacement=jnp.abs(new.x[...,:2]-old.x[...,:2])/jnp.array([dynamic.dx,dynamic.dy])
    invalid=jnp.any(old.active & (jnp.any(displacement>1+1e-12,axis=-1)|jnp.any(~jnp.isfinite(new.x),axis=-1)|jnp.any(~jnp.isfinite(new.u),axis=-1)))
    return PolarStepDiagnostics(counts,charges,removed,jnp.array([lower,upper]),invalid)
