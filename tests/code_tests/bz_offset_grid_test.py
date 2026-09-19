"""Physical-pole reflection on a quarter-cell-shifted angular grid."""
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np

from PyPIC3D.relativity.core import D_FIELD_LOCATIONS, B_FIELD_LOCATIONS
from tests.support.bz_offset_grid import reflection

jax.config.update('jax_enable_x64', True)


def test_shifted_reflection_converges_to_smooth_even_and_odd_fields():
    for locations in (D_FIELD_LOCATIONS,B_FIELD_LOCATIONS):
        errors=[]
        for n in (16,32,64):
            g=3;step=np.pi/n
            center=(np.arange(n+2*g)-g+.25)*step
            vertex=center+step/2
            grids=SimpleNamespace(tiled_center_grid=(None,center[None,None,None,:],None),
                                  tiled_vertex_grid=(None,vertex[None,None,None,:],None))
            static=SimpleNamespace(guard_cells=g,tile_shape=(4,n,1))
            dynamic=SimpleNamespace(grids=grids,dy=step)
            functions=(np.cos,np.sin,lambda x:np.cos(2*x))
            expected=tuple(jnp.asarray(fun(center if loc[1]=='C' else vertex))[None,None,None,None,:,None]
                           for fun,loc in zip(functions,locations))
            # Poison the halos: the extension must use only owned physical data.
            source=tuple(a.at[:,:,:,:,:g,:].set(jnp.nan).at[:,:,:,:,g+n:,:].set(jnp.nan) for a in expected)
            actual=reflection(source,locations,static,dynamic)
            errors.append(max(float(jnp.max(jnp.abs(a-b))) for a,b in zip(actual,expected)))
            for a,b in zip(actual,expected):
                np.testing.assert_array_equal(a[:,:,:,:,g:g+n,:],b[:,:,:,:,g:g+n,:])
        assert errors[0]/errors[1]>7.
        assert errors[1]/errors[2]>7.


def test_reflection_preserves_stability_of_flat_angular_maxwell_operator():
    # At r=1 the source-free angular subsystem is
    # d_t D^phi = -d_theta B^r/sin(theta),
    # d_t B^r = -d_theta(sin(theta)^2 D^phi)/sin(theta).
    # Construct its spatial operator from reflected basis vectors. The metric
    # factor must multiply the extended primitive, including in ghost cells.
    n=32;g=3;h=np.pi/n
    c=(np.arange(n+2*g)-g+.25)*h;v=c+h/2
    grids=SimpleNamespace(tiled_center_grid=(None,c[None,None,None,:],None),
                          tiled_vertex_grid=(None,v[None,None,None,:],None))
    s=SimpleNamespace(guard_cells=g,tile_shape=(n,n,1))
    d=SimpleNamespace(grids=grids,dy=h)
    basis=jnp.zeros((1,1,1,n,n+2*g,1)).at[0,0,0,:,g:g+n,0].set(jnp.eye(n))
    rc=np.asarray(reflection((basis,)*3,D_FIELD_LOCATIONS,s,d)[0])[0,0,0,:,:,0].T
    rv=np.asarray(reflection((basis,)*3,B_FIELD_LOCATIONS,s,d)[0])[0,0,0,:,:,0].T
    a=-(rv[g:g+n]-rv[g-1:g+n-1])/h/np.sin(c[g:g+n,None])
    weighted=np.sin(c[:,None])**2*rc
    b=-(weighted[g+1:g+n+1]-weighted[g:g+n])/h/np.sin(v[g:g+n,None])
    operator=np.block([[np.zeros((n,n)),a],[b,np.zeros((n,n))]])
    assert np.linalg.eigvals(operator).real.max()<1e-6
