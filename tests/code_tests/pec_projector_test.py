"""Metric algebra, surface placement, and distributed staggered PEC contracts."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from PyPIC3D.boundary_conditions.pec import normal_projector, tangent_intersection_projector
from PyPIC3D.boundary_conditions.staggered import refresh_fields
from PyPIC3D.boundary_conditions.ghost_cells import apply_tiled_pec_boundary
from PyPIC3D.relativity.core import Metric, YeeMetric, D_FIELD_LOCATIONS, B_FIELD_LOCATIONS, build_yee_metric
from PyPIC3D.relativity.field_interpolation import metric_weighted_interpolate
from PyPIC3D.solvers.gr_static.static_metric import compute_covariant_E, compute_covariant_H, update_B_relativity, update_D_relativity
from PyPIC3D.diagnostics.static_metric import divergence
from tests.kernel_fixtures import kernel_parameters

jax.config.update('jax_enable_x64', True)


def constant_metric(shape, gamma=None, shift=(0., 0., 0.)):
    gamma = jnp.eye(3) if gamma is None else jnp.asarray(gamma)
    m = Metric(jnp.ones(shape), jnp.broadcast_to(jnp.asarray(shift), shape+(3,)),
               jnp.broadcast_to(gamma, shape+(3,3)),
               jnp.broadcast_to(jnp.linalg.inv(gamma), shape+(3,3)),
               jnp.full(shape, jnp.sqrt(jnp.linalg.det(gamma))))
    return YeeMetric((m,)*3, (m,)*3, m, m)


def setup(n=(6, 5, 1), g=2, bc=(1, 0, 0), tiles=None, gamma=None, shift=(0.,0.,0.)):
    s,d = kernel_parameters(Nx=n[0], Ny=n[1], Nz=n[2], tile_shape=tiles or n,
                            guard_cells=g, boundary_conditions=bc, solver='static_metric',
                            x_min=0., y_min=0., z_min=0., x_wind=1., y_wind=1., z_wind=1.)
    shape = tuple(a//b for a,b in zip(n,s.tile_shape))+tuple(a+2*g for a in s.tile_shape)
    return s,d,constant_metric(shape,gamma,shift)


class TestProjectorAlgebra(unittest.TestCase):
    def test_metric_projectors_and_intersections(self):
        for gamma in (np.diag([2.,3.,4.]), np.array([[2.,.4,.2],[.4,1.5,.3],[.2,.3,1.]])):
            inverse = jnp.linalg.inv(gamma)
            for a in range(3):
                cov = jnp.eye(3)[a]/jnp.sqrt(inverse[a,a])
                raised = inverse@cov
                self.assertAlmostEqual(float(cov@raised),1.)
                G = normal_projector(inverse,cov); F=jnp.eye(3)-G
                for actual, expected in ((G@G,G),(F@F,F),(F@G,jnp.zeros((3,3))),
                                         (G@raised,raised),(F@raised,jnp.zeros(3))):
                    np.testing.assert_allclose(actual,expected,atol=1e-14)
                v=jnp.array([.5,-1.,2.]); tangent=F@v
                self.assertAlmostEqual(float(cov@tangent),0.)
                np.testing.assert_allclose(F@tangent,tangent,atol=1e-14)
                np.testing.assert_allclose((2*G-jnp.eye(3))@(2*G-jnp.eye(3)),jnp.eye(3),atol=1e-14)
            normals = jnp.array([[1.,0.,0.],[0.,1.,0.]])
            P = tangent_intersection_projector(inverse,normals)
            np.testing.assert_allclose(P@P,P,atol=1e-14)
            np.testing.assert_allclose(normals@P,0.,atol=1e-14)
            np.testing.assert_allclose(P@jnp.array([0.,0.,2.]),[0.,0.,2.],atol=1e-14)


class TestStaggeredProjectors(unittest.TestCase):
    def test_endpoints_all_axes_guards_and_narrow_tiles(self):
        for g in (1,2,3):
            for axis in range(3):
                n=[1,1,1];n[axis]=2
                bc=[0,0,0];bc[axis]=1
                s,d,m=setup(tuple(n),g,tuple(bc))
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
        s,d,m=setup(gamma=gamma,shift=(.3,0.,0.))
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
        s,d,m=setup((4,4,4),2,(1,1,1),gamma=[[2.,.4,.2],[.4,1.5,.3],[.2,.3,1.]])
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
            s,d,m=setup((4,4,1),3,(1,1,0),tile,gamma)
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
            s,d,_=setup((n,n,1),2)
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

    def test_updated_stages_sponge_and_constraint_change(self):
        s,d,m=setup((6,6,1),2)
        s=s._replace(supergaussian_active=True,supergaussian_layers=((1,1,2,4.,2.),))
        shape=m.center.lapse.shape;g=s.guard_cells
        fields=tuple(jnp.ones(shape)*(i+1.) for i in range(3));zero=(jnp.zeros(shape),)*3
        before=divergence(fields,m.B,d,forward=True)
        B=update_B_relativity(zero,fields,m,s,d,.01)
        D=update_D_relativity(fields,zero,zero,m,s,d,.01)
        for wall in (g,g+s.tile_shape[0]):
            np.testing.assert_array_equal(B[0][0,0,0,wall,g:-g,g],0.)
            for i in (1,2):np.testing.assert_array_equal(D[i][0,0,0,wall,g:-g,g],0.)
        projected_B=refresh_fields(fields,s,B_FIELD_LOCATIONS,'B',m)
        projected_D=refresh_fields(fields,s,D_FIELD_LOCATIONS,'D',m)
        after=divergence(projected_B,m.B,d,forward=True)
        change=float(jnp.max(jnp.abs((after-before)[0,0,0,g:-g,g:-g,g])))
        self.assertGreater(change,0.)
        self.assertTrue(np.isfinite(change))
        gauss=divergence(projected_D,m.D,d,forward=False)-divergence(fields,m.D,d,forward=False)
        self.assertTrue(np.isfinite(np.asarray(gauss)).all())
        print(f'PEC projection residual changes: div B={change:.6g}, div D={float(jnp.max(jnp.abs(gauss))):.6g}')

    def test_initial_normal_trace_ignores_stale_exterior(self):
        s,d,m=setup(gamma=[[1.25,.5,0.],[.5,1.,0.],[0.,0.,1.]])
        g=s.guard_cells;n=s.tile_shape[0]
        normal=jnp.full(m.center.lapse.shape,2.)
        normal=normal.at[:,:,:,:g].set(123.).at[:,:,:,g+n:].set(-456.)
        D=refresh_fields((normal,jnp.zeros_like(normal),jnp.zeros_like(normal)),s,D_FIELD_LOCATIONS,'D',m)
        for wall in (g,g+n):self.assertAlmostEqual(float(D[1][0,0,0,wall,g+2,g]),-1.)

    def test_particle_gather_and_previous_levels_are_projected(self):
        from unittest.mock import patch
        from PyPIC3D.solvers.gr_static.time_loop import time_loop_static_metric
        s,d,m=setup()
        value=jnp.ones(m.center.lapse.shape);v=(value,)*3;z=(value*0,)*3
        fields=(v,v,z,value*0,value*0,(v,v),m,(v,v),False)
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
        s,d,m=setup((8,8,1),2,(1,1,0))
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
        s,d,_=setup((8,4,1),3)
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
        s,d,m=setup((4,4,1),2,(1,3,0))
        shape=m.center.lapse.shape
        x=jnp.arange(shape[3])[None,None,None,:,None,None]
        y=jnp.arange(shape[4])[None,None,None,None,:,None]
        value=jnp.broadcast_to(x+10*y,shape).astype(float)
        for locs in (D_FIELD_LOCATIONS,B_FIELD_LOCATIONS):
            result=refresh_fields((value,)*3,s,locs)
            for component in result:
                np.testing.assert_array_equal(component[0,0,0,:,:,2],value[0,0,0,:,:,2])
