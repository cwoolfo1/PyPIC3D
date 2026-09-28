"""Excised spherical caps: metric support, wall conditions, and charge folding."""
import math
import unittest
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from tests.support.polar_runtime import SimulationParameters, build_runtime
from PyPIC3D.boundary_conditions.polar import refresh_vector, divergence, plane
from PyPIC3D.deposition.GR_Esirkepov import GR_Esirkepov_current
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.particles.particle_tile_communication import refresh_tiled_particle_tiles
from PyPIC3D.relativity.core import B_FIELD_LOCATIONS, D_FIELD_LOCATIONS
from PyPIC3D.relativity.flat import initialize_flat_spherical_metric
from PyPIC3D.solvers.gr_static.static_metric import compute_covariant_E, compute_covariant_H
from tests.support.polar_fixtures import particle

jax.config.update('jax_enable_x64', True)


class TestExcisedCaps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = SimulationParameters(nr=16, ntheta=32, r_max=4., sponge_start=3.,
                                    current_filter_passes=0, horizon_field_cells=0,
                                    pairs_per_cell=2, capacity_factor=2, particle_batch_size=256,
                                    polar_cap_angle=math.radians(20), backend='cpu')
        cls.s, cls.d, cls.m, _ = build_runtime(cls.p)

    def test_all_metric_nodes_including_halos_are_regular(self):
        for m in (*self.m.D, *self.m.B, self.m.center, self.m.vertex):
            self.assertTrue(np.all(np.asarray(m.sqrt_gamma) > 0))
            for leaf in jax.tree.leaves(m):
                self.assertTrue(np.isfinite(np.asarray(leaf)).all())
        self.assertAlmostEqual(float(self.d.grids.center[1][1]), self.p.polar_cap_angle)
        self.assertAlmostEqual(float(self.d.grids.center[1][-1]), math.pi-self.p.polar_cap_angle)
        with self.assertRaisesRegex(ValueError, 'guard nodes'):
            build_runtime(replace(self.p, polar_cap_angle=math.radians(1)))

    def test_cap_volume_is_clipped_at_wall(self):
        s, d, p = self.s, self.d, self.p
        flat_s = s._replace(metric='flat_spherical', metric_mass=0., metric_spin=0.)
        m = initialize_flat_spherical_metric(flat_s, d)
        g = s.guard_cells
        r = float(d.grids.tiled_center_grid[0][0, 0, 0, g+4])
        expected = 2*math.pi*((r+p.dr/2)**3-(r-p.dr/2)**3)/3
        expected *= math.cos(p.polar_cap_angle)-math.cos(p.polar_cap_angle+p.dtheta/2)
        self.assertAlmostEqual(float(m.geometry.volume[0, 0, 0, g+4, g, g]), expected, places=12)

    def test_reflection_does_not_rotate_azimuth(self):
        p, s, d = self.p, self.s, self.d
        for wall, sign in ((p.polar_cap_angle, -1), (math.pi-p.polar_cap_angle, 1)):
            pts, _ = particle(s, d, wall+sign*.1*p.dtheta)
            pts = pts._replace(u=pts.u.at[..., 1].set(.3), x=pts.x.at[..., 2].set(.7))
            new, overflow = refresh_tiled_particle_tiles(pts, s, d)
            self.assertFalse(bool(overflow))
            self.assertAlmostEqual(float(new.x[0, 0, 0, 0, 0, 1]), wall-sign*.1*p.dtheta)
            self.assertAlmostEqual(float(new.x[0, 0, 0, 0, 0, 2]), .7)
            self.assertAlmostEqual(float(new.u[0, 0, 0, 0, 0, 1]), -.3)

    def test_conducting_wall_and_geometry_independent_interpolation(self):
        s, m = self.s, self.m
        values = tuple(jnp.ones_like(m.center.lapse)*(i+1) for i in range(3))
        D = refresh_vector(values, s, D_FIELD_LOCATIONS, 'D')
        B = refresh_vector(values, s, B_FIELD_LOCATIONS, 'B')
        # Constitutive interpolation must be independent of curl geometry.
        for fn in (compute_covariant_E, compute_covariant_H):
            for a, b in zip(fn(D, B, m), fn(D, B, m._replace(geometry=None))):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        for metric in (m, m._replace(geometry=None)):
            E, H = compute_covariant_E(D, B, metric), compute_covariant_H(D, B, metric)
            self.assertTrue(all(np.isfinite(np.asarray(x)).all() for x in (*E, *H)))
            for k in (s.guard_cells, s.guard_cells+s.tile_shape[1]):
                for x in (D[0], D[2], B[1], E[0], E[2]):
                    np.testing.assert_array_equal(np.asarray(x[plane(x, k)]), 0.)

    def test_reflected_particle_current_continuity_at_cap(self):
        p, s, d, m = self.p, self.s, self.d, self.m
        z = jnp.zeros_like(m.center.lapse)
        old, sp = particle(s, d, p.polar_cap_angle+.1*p.dtheta)
        new = old._replace(x=old.x.at[..., 1].add(-.3*p.dtheta))
        J = GR_Esirkepov_current(old, new, sp, (z,)*3, m, s, d)
        q0 = compute_rho(old, sp, z, s, d)*d.dx*d.dy*d.dz
        q1 = compute_rho(new, sp, z, s, d)*d.dx*d.dy*d.dz
        residual = (q1-q0)/d.dt+divergence(J, m.geometry, s)
        self.assertLess(float(jnp.max(jnp.abs(jnp.where(m.geometry.charge_owned, residual, 0)))), 1e-11)

    def test_full_sphere_metric_retains_axis_values(self):
        from tests.support.polar_fixtures import polar_runtime
        p, s, d, m = polar_runtime()
        self.assertEqual(p.polar_cap_angle, 0.)
        self.assertTrue(bool(jnp.any(m.D[0].sqrt_gamma == 0.)))
        self.assertTrue(all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(m)))


if __name__ == '__main__':
    unittest.main()
