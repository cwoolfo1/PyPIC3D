"""Surface placement and distributed staggered PEC contracts."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from PyPIC3D.boundary_conditions.pec import _pec_setup, _prepare_normal_D, _project_native_nodes
from PyPIC3D.boundary_conditions.staggered import refresh_fields
from PyPIC3D.boundary_conditions.ghost_cells import apply_tiled_pec_boundary
from PyPIC3D.relativity.core import Metric, YeeMetric, D_FIELD_LOCATIONS, B_FIELD_LOCATIONS, build_yee_metric
from PyPIC3D.relativity.field_interpolation import metric_weighted_interpolate
from PyPIC3D.solvers.gr_static.static_metric import compute_covariant_E, compute_covariant_H, update_B_relativity, update_D_relativity
from PyPIC3D.diagnostics.static_metric import divergence
from tests.kernel_fixtures import kernel_parameters


def constant_metric(shape, gamma=None, shift=(0., 0., 0.)):
    gamma = jnp.eye(3) if gamma is None else jnp.asarray(gamma)
    m = Metric(jnp.ones(shape), jnp.broadcast_to(jnp.asarray(shift), shape+(3,)),
               jnp.broadcast_to(gamma, shape+(3,3)),
               jnp.broadcast_to(jnp.linalg.inv(gamma), shape+(3,3)),
               jnp.full(shape, jnp.sqrt(jnp.linalg.det(gamma))))
    return YeeMetric((m,)*3, (m,)*3, m, m)


def make_setup(n=(6, 5, 1), g=2, bc=(1, 0, 0), tiles=None, gamma=None, shift=(0.,0.,0.)):
    s,d = kernel_parameters(Nx=n[0], Ny=n[1], Nz=n[2], tile_shape=tiles or n,
                            guard_cells=g, boundary_conditions=bc, solver='static_metric',
                            x_min=0., y_min=0., z_min=0., x_wind=1., y_wind=1., z_wind=1.)
    shape = tuple(a//b for a,b in zip(n,s.tile_shape))+tuple(a+2*g for a in s.tile_shape)
    return s,d,constant_metric(shape,gamma,shift)


def coupled_metric(position):
    """Smooth periodic metric whose off-diagonal terms couple every wall pair."""
    X, Y, Z = (2*jnp.pi*value for value in position)
    gamma = jnp.array([[2.+.2*jnp.sin(X), .3+.05*jnp.cos(Y), .15+.03*jnp.sin(Z)],
                       [.3+.05*jnp.cos(Y), 3.+.1*jnp.cos(Y), -.2+.02*jnp.cos(X)],
                       [.15+.03*jnp.sin(Z), -.2+.02*jnp.cos(X), 1.5+.1*jnp.sin(Z)]])
    lapse = 1.+.03*jnp.sin(X)
    shift = jnp.array([.04*jnp.sin(X), .02*jnp.sin(Y), .01*jnp.cos(Z)])
    return lapse, shift, gamma, jnp.linalg.inv(gamma), jnp.sqrt(jnp.linalg.det(gamma))


def coupled_setup(n, tiles, bc):
    s, d = kernel_parameters(Nx=n[0], Ny=n[1], Nz=n[2], tile_shape=tiles, boundary_conditions=bc,
                             solver='static_metric', metric='numerical',
                             x_min=0., y_min=0., z_min=0., x_wind=1., y_wind=1., z_wind=1.)
    return s, d, build_yee_metric(d, coupled_metric)


class TestEdgeCoupledProjection(unittest.TestCase):
    """Where two walls meet in a non-orthogonal metric, the wall rows couple."""

    def test_refresh_is_idempotent_with_coupled_edges(self):
        cases = (((16, 8, 1), (8, 8, 1), (1, 1, 0)),    # x/y walls, two tiles
                 ((8, 6, 8), (4, 6, 4), (1, 1, 1)))     # x/y/z walls, 2x1x2 tiles
        rng = np.random.default_rng(0)
        for n, tiles, bc in cases:
            s, d, m = coupled_setup(n, tiles, bc)
            shape = m.center.lapse.shape
            for kind, locations in (('D', D_FIELD_LOCATIONS), ('B', B_FIELD_LOCATIONS)):
                with self.subTest(walls=bc, field=kind):
                    vector = tuple(jnp.asarray(rng.normal(size=shape)) for _ in range(3))
                    once = refresh_fields(vector, s, locations, kind, m)
                    twice = refresh_fields(once, s, locations, kind, m)
                    for first, second in zip(once, twice):
                        np.testing.assert_allclose(second, first, rtol=0, atol=1e-12 * float(jnp.abs(first).max()))

    def test_D_edge_rows_hold_simultaneously(self):
        """Recomputing the x-wall normal row of D^y from the refreshed y-wall D^x reproduces it."""
        s, d, m = coupled_setup((16, 8, 1), (8, 8, 1), (1, 1, 0))
        vector = tuple(jnp.asarray(np.random.default_rng(1).normal(size=m.center.lapse.shape)) for _ in range(3))
        refreshed = refresh_fields(vector, s, D_FIELD_LOCATIONS, 'D', m)
        axes, exchange = _pec_setup(refreshed[0].shape, s, D_FIELD_LOCATIONS)
        snapshot = exchange(_prepare_normal_D(exchange(refreshed), s, axes))
        one_pass = _project_native_nodes(snapshot, s, D_FIELD_LOCATIONS, 'D', m, axes)
        g = s.guard_cells
        for component in (0, 1):
            np.testing.assert_allclose(one_pass[component][0, 0, 0, g:g+2, g:g+2, g],
                                       refreshed[component][0, 0, 0, g:g+2, g:g+2, g], atol=1e-13)


class TestStaggeredProjectors(unittest.TestCase):
    def test_endpoints_all_axes_guards_and_narrow_tiles(self):
        for g in (1,2,3):
            for axis in range(3):
                n=[1,1,1];n[axis]=2
                bc=[0,0,0];bc[axis]=1
                s,d,m=make_setup(tuple(n),g,tuple(bc))
                original=tuple(jnp.full(m.center.lapse.shape,i+1.) for i in range(3))
                for locs,kind in ((D_FIELD_LOCATIONS,'D'),(B_FIELD_LOCATIONS,'B')):
                    result=refresh_fields(original,s,locs,kind,m)
                    for i,loc in enumerate(locs):
                        index=[0,0,0,g,g,g]
                        if loc[axis]=='C':
                            for wall in (g,g+n[axis]):
                                index[axis+3]=wall
                                self.assertEqual(float(result[i][tuple(index)]),0.)
                            index[axis+3]=g+n[axis]-1
                            self.assertEqual(float(result[i][tuple(index)]),i+1.)
                        # Diagonal metric reflection is exactly the C/V parity.
                        index[axis+3]=g-1; owner=index.copy()
                        owner[axis+3]=g if loc[axis]=='V' else g+1
                        parity=(1 if i==axis else -1) if kind=='D' else (-1 if i==axis else 1)
                        self.assertEqual(float(result[i][tuple(index)]),parity*float(result[i][tuple(owner)]))

    def test_offdiagonal_normal_D_and_normal_shift_auxiliary(self):
        gamma=np.array([[1.25,.5,0.],[.5,1.,0.],[0.,0.,1.]])
        s,d,m=make_setup(gamma=gamma,shift=(.3,0.,0.))
        shape=m.center.lapse.shape;g=s.guard_cells;n=s.tile_shape[0]
        D=tuple(jnp.full(shape,v) for v in (2.,7.,8.))
        B=tuple(jnp.full(shape,v) for v in (3.,0.,4.))
        Dp=refresh_fields(D,s,D_FIELD_LOCATIONS,'D',m)
        Bp=refresh_fields(B,s,B_FIELD_LOCATIONS,'B',m)
        for x in (g,g+n):
            idx=(0,0,0,x,g+2,g)
            self.assertAlmostEqual(float(Dp[1][idx]),-1.)
            self.assertEqual(float(Dp[2][idx]),0.)
            self.assertEqual(float(Bp[0][idx]),0.)
        E=compute_covariant_E(Dp,Bp,m)
        self.assertGreater(abs(float(E[1][0,0,0,g,g+2,g])),.1)
        Er=refresh_fields(E,s,D_FIELD_LOCATIONS)
        np.testing.assert_array_equal(Er[1][0,0,0,:g,g+2,g],E[1][0,0,0,:g,g+2,g])
        for i in range(3):
            np.testing.assert_array_equal(D[i],(2.,7.,8.)[i]);np.testing.assert_array_equal(B[i],(3.,0.,4.)[i])
        with self.assertRaisesRegex(ValueError,'require a Yee metric'):
            refresh_fields(D,s,D_FIELD_LOCATIONS,'D')

    def test_corners_intersect_and_yee_last_interior_is_free(self):
        s,d,m=make_setup((4,4,4),2,(1,1,1),gamma=[[2.,.4,.2],[.4,1.5,.3],[.2,.3,1.]])
        values=tuple(jnp.full(m.center.lapse.shape,i+1.) for i in range(3))
        D=refresh_fields(values,s,D_FIELD_LOCATIONS,'D',m)
        E=apply_tiled_pec_boundary(values,s)
        for i,loc in enumerate(D_FIELD_LOCATIONS):
            others=[a for a in range(3) if a!=i]
            for lo in (2,6):
                for hi in (2,6):
                    idx=[0,0,0,3,3,3];idx[others[0]+3]=lo;idx[others[1]+3]=hi
                    self.assertEqual(float(D[i][tuple(idx)]),0.)
            idx=[0,0,0,3,3,3];idx[others[0]+3]=5
            self.assertEqual(float(E[i][tuple(idx)]),i+1.)
            idx[others[0]+3]=6;self.assertEqual(float(E[i][tuple(idx)]),0.)

    def test_transverse_endpoint_exchange_matches_single_device(self):
        if jax.device_count()<4:self.skipTest('requires four CPU devices')
        gamma=[[1.25,.5,0.],[.5,1.,0.],[0.,0.,1.]]
        results=[]
        tiled_results=[]
        for tile in ((4,4,1),(2,2,1)):
            s,d,m=make_setup((4,4,1),3,(1,1,0),tile,gamma)
            x=d.grids.tiled_center_grid[0][..., :,None,None]
            y=d.grids.tiled_center_grid[1][...,None,:,None]
            fields=tuple(jnp.broadcast_to((i+1)*(1+x+2*y),m.center.lapse.shape) for i in range(3))
            pair=[]
            tile_pair=[]
            for locs,kind in ((D_FIELD_LOCATIONS,'D'),(B_FIELD_LOCATIONS,'B')):
                values=refresh_fields(fields,s,locs,kind,m)
                tile_pair.extend(np.asarray(v) for v in values)
                joined=[]
                for value,loc in zip(values,locs):
                    rows=[]
                    for tx in range(value.shape[0]):
                        cols=[]
                        for ty in range(value.shape[1]):
                            ex=int(tx==value.shape[0]-1 and loc[0]=='C')
                            ey=int(ty==value.shape[1]-1 and loc[1]=='C')
                            cols.append(np.asarray(value)[tx,ty,0,3:3+tile[0]+ex,3:3+tile[1]+ey,3])
                        rows.append(np.concatenate(cols,axis=1))
                    joined.append(np.concatenate(rows,axis=0))
                pair.extend(joined)
            results.append(pair)
            tiled_results.append(tile_pair)
        for a,b in zip(*results):np.testing.assert_allclose(a,b,atol=1e-13)
        # Check endpoint planes in transverse halos as well as owned nodes.
        for single,distributed in zip(*tiled_results):
            for tx in range(2):
                for ty in range(2):
                    np.testing.assert_allclose(distributed[tx,ty,0],
                        single[0,0,0,2*tx:2*tx+8,2*ty:2*ty+8,:],atol=1e-13)

    def test_varying_metric_boundary_residual_converges(self):
        errors=[]
        for n in (8,16,32):
            s,d,_=make_setup((n,n,1),2)
            def metric_at(p):
                shear=.3+.15*jnp.cos(2*jnp.pi*p[1])+.1*p[0]
                gamma=jnp.array([[1+shear**2,shear,0.],[shear,1.,0.],[0.,0.,1.]])
                return 1.,jnp.zeros(3),gamma,jnp.linalg.inv(gamma),1.
            m=build_yee_metric(d,metric_at)
            v=tuple(jnp.ones_like(m.center.lapse)*(i+1) for i in range(3))
            result=refresh_fields(v,s,D_FIELD_LOCATIONS,'D',m)
            collocated=jnp.stack([metric_weighted_interpolate(result[i],m.D[i],m.center,loc,('C',)*3)
                                 for i,loc in enumerate(D_FIELD_LOCATIONS)],axis=-1)
            residual=jnp.einsum('...ij,...j->...i',m.center.gamma,collocated)[...,1:]
            errors.append(float(jnp.max(jnp.abs(residual[0,0,0,2,3:2+n-1,2]))))
        for coarse,fine in zip(errors,errors[1:]):self.assertGreater(coarse/fine,3.5,errors)

    def test_updated_stages_sponge_and_projected_divergence(self):
        """Uniform fields, flat metric, x walls: projection adds no div D and only wall div B."""
        s,d,m=make_setup((6,6,1),2)
        s=s._replace(supergaussian_active=True,supergaussian_layers=((1,1,2,4.,2.),))
        shape=m.center.lapse.shape;g=s.guard_cells;n=s.tile_shape[0]
        fields=tuple(jnp.ones(shape)*(i+1.) for i in range(3));zero=(jnp.zeros(shape),)*3
        B=update_B_relativity(zero,fields,m,s,d,.01)
        D=update_D_relativity(fields,zero,zero,m,s,d,.01)
        for wall in (g,g+n):
            np.testing.assert_array_equal(B[0][0,0,0,wall,g:-g,g],0.)
            for i in (1,2):np.testing.assert_array_equal(D[i][0,0,0,wall,g:-g,g],0.)
        projected_B=refresh_fields(fields,s,B_FIELD_LOCATIONS,'B',m)
        projected_D=refresh_fields(fields,s,D_FIELD_LOCATIONS,'D',m)
        # Zeroing the uniform tangential D^y, D^z on the walls leaves Gauss's law unchanged.
        gauss=divergence(projected_D,m.D,d,forward=False)-divergence(fields,m.D,d,forward=False)
        np.testing.assert_allclose(gauss[0,0,0,g:-g,g:-g,g],0.,atol=1e-12)
        # Zeroing normal B^x=1 on the wall nodes changes div B by +-B^x/dx in the
        # first and last interior cells only.
        change=(divergence(projected_B,m.B,d,forward=True)-divergence(fields,m.B,d,forward=True))
        expected=np.zeros((n,s.tile_shape[1]))
        expected[0],expected[-1]=1./d.dx,-1./d.dx
        np.testing.assert_allclose(change[0,0,0,g:-g,g:-g,g],expected,rtol=1e-12,atol=1e-12)

    def test_initial_normal_trace_ignores_stale_exterior(self):
        s,d,m=make_setup(gamma=[[1.25,.5,0.],[.5,1.,0.],[0.,0.,1.]])
        g=s.guard_cells;n=s.tile_shape[0]
        normal=jnp.full(m.center.lapse.shape,2.)
        normal=normal.at[:,:,:,:g].set(123.).at[:,:,:,g+n:].set(-456.)
        D=refresh_fields((normal,jnp.zeros_like(normal),jnp.zeros_like(normal)),s,D_FIELD_LOCATIONS,'D',m)
        for wall in (g,g+n):self.assertAlmostEqual(float(D[1][0,0,0,wall,g+2,g]),-1.)

    def test_particle_gather_and_previous_levels_are_projected(self):
        from unittest.mock import patch
        from PyPIC3D.solvers.gr_static.time_loop import time_loop_static_metric
        s,d,m=make_setup()
        value=jnp.ones(m.center.lapse.shape);z=(value*0,)*3
        # the loop expects every incoming D and B to be refreshed already
        vD=refresh_fields((value,)*3,s,D_FIELD_LOCATIONS,'D',m)
        vB=refresh_fields((value,)*3,s,B_FIELD_LOCATIONS,'B',m)
        fields=(vD,vB,z,value*0,value*0,(vD,vB),m,(vD,vB),False)
        captured=[]
        def push(particles,species,D,B,*args):
            captured.append((D,B));return particles,particles
        module='PyPIC3D.solvers.gr_static.time_loop.'
        with patch(module+'hybrid_boris_geodesic_push',side_effect=push), \
             patch(module+'GR_direct_deposition',return_value=z), \
             patch(module+'refresh_tiled_particle_tiles',side_effect=lambda p,*args:(p,False)):
            _,result=time_loop_static_metric(None,None,fields,s,d)
        for D,B in (*captured,(result[0],result[1]),result[7]):
            for wall in (s.guard_cells,s.guard_cells+s.tile_shape[0]):
                idx=(0,0,0,wall,s.guard_cells+2,s.guard_cells)
                self.assertEqual(float(B[0][idx]),0.)
                self.assertEqual(float(D[1][idx]),0.)
                self.assertEqual(float(D[2][idx]),0.)

    def test_flat_stages_agree_with_yee_for_compatible_fields(self):
        from PyPIC3D.solvers.yee.first_order_yee import update_E, update_B
        s,d,m=make_setup((8,8,1),2,(1,1,0))
        shape=m.center.lapse.shape
        x=d.grids.tiled_center_grid[0][..., :,None,None]
        y=d.grids.tiled_center_grid[1][...,None,:,None]
        z=jnp.zeros(shape)
        E=(z,z,jnp.broadcast_to(jnp.sin(jnp.pi*x)*jnp.sin(jnp.pi*y),shape))
        E=apply_tiled_pec_boundary(E,s);D=refresh_fields(E,s,D_FIELD_LOCATIONS,'D',m)
        By=(z,)*3;Bg=By;J=By
        for _ in range(3):
            By,_=update_B(E,By,s,d)
            Bg=update_B_relativity(compute_covariant_E(D,Bg,m),Bg,m,s,d,d.dt/2)
            E,_=update_E(E,By,J,s,d)
            D=update_D_relativity(D,compute_covariant_H(D,Bg,m),J,m,s,d,d.dt)
            for a,b in zip((*E,*By),(*compute_covariant_E(D,Bg,m),*Bg)):
                np.testing.assert_allclose(a[:,:,:,2:-2,2:-2,2],b[:,:,:,2:-2,2:-2,2],atol=1e-12)

    def test_reflection_uses_surface_metric_not_ghost_metric(self):
        s,d,_=make_setup((8,4,1),3)
        def metric_at(p):
            shear=.2+.4*p[0]
            gamma=jnp.array([[1+shear**2,shear,0.],[shear,1.,0.],[0.,0.,1.]])
            return 1.,jnp.zeros(3),gamma,jnp.linalg.inv(gamma),1.
        m=build_yee_metric(d,metric_at);g=s.guard_cells;n=s.tile_shape[0]
        v=tuple(jnp.full(m.center.lapse.shape,x) for x in (2.,5.,0.))
        D=refresh_fields(v,s,D_FIELD_LOCATIONS,'D',m)
        for wall,ghost in ((g,g-1),(g+n,g+n+1)):
            x=float(d.grids.tiled_center_grid[0][0,0,0,wall])
            shear=.2+.4*x
            self.assertAlmostEqual(float(D[1][0,0,0,wall,g+1,g]),-2*shear)
            self.assertAlmostEqual(float(D[1][0,0,0,ghost,g+1,g]),-4*shear-5.)

    def test_auxiliary_refresh_preserves_all_computed_exterior_faces(self):
        s,d,m=make_setup((4,4,1),2,(1,3,0))
        shape=m.center.lapse.shape
        x=jnp.arange(shape[3])[None,None,None,:,None,None]
        y=jnp.arange(shape[4])[None,None,None,None,:,None]
        value=jnp.broadcast_to(x+10*y,shape).astype(float)
        for locs in (D_FIELD_LOCATIONS,B_FIELD_LOCATIONS):
            result=refresh_fields((value,)*3,s,locs)
            for component in result:
                np.testing.assert_array_equal(component[0,0,0,:,:,2],value[0,0,0,:,:,2])
