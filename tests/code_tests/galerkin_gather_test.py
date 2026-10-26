"""Mixed-order field gathering on the native Yee component grids."""

import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.pusher.boris import field_component_orders, interpolate_field_to_particles
from PyPIC3D.pusher.hybrid_boris_geodesic import (
    gather_vector, hybrid_boris_geodesic_push, tile_fields,
)
from PyPIC3D.pusher.particle_push import particle_push, seed_leapfrog_velocity
from PyPIC3D.relativity.core import B_FIELD_LOCATIONS, D_FIELD_LOCATIONS, location_grid
from PyPIC3D.relativity.metrics.flat import initialize_flat_cartesian_metric
from tests.kernel_fixtures import kernel_parameters


def cardinal_weights(position, axis, order):
    """Independent full-grid cardinal splines, including periodic images."""
    spacing = axis[1] - axis[0]
    r = (position[:, None] - axis[None, :]) / spacing
    r = jnp.mod(r + len(axis) / 2, len(axis)) - len(axis) / 2
    if order == 0:
        return ((r >= -0.5) & (r < 0.5)).astype(position.dtype)
    if order == 1:
        return jnp.maximum(1.0 - jnp.abs(r), 0.0)
    distance = jnp.abs(r)
    return jnp.where(distance < 0.5, 0.75 - r**2,
                     jnp.where(distance < 1.5, 0.5 * (1.5 - distance)**2, 0.0))


def reference_gather(field, position, grid, orders):
    weights = tuple(cardinal_weights(position[:, a], grid[a], orders[a]) for a in range(3))
    return jnp.einsum("pi,pj,pk,ijk...->p...", *weights, field)


class TestGalerkinGather(unittest.TestCase):
    def test_all_component_orders_match_cardinal_splines_with_trailing_dimensions(self):
        shape = (8, 7, 6)
        center = tuple(jnp.arange(n, dtype=float) * 0.5 - 1.0 for n in shape)
        vertex = tuple(axis + 0.25 for axis in center)
        field = jax.random.normal(jax.random.PRNGKey(17), shape + (2, 2), dtype=jnp.float64)
        position = jnp.asarray(((-1.0, 0.125, 1.75), (0.25, -1.25, 0.625),
                                (2.875, 2.375, 1.875), (3.0, -1.0, -1.0)))
        for p in (1, 2):
            electric, magnetic = field_component_orders(p, "galerkin")
            self.assertEqual(electric, ((p-1, p, p), (p, p-1, p), (p, p, p-1)))
            self.assertEqual(magnetic, ((p, p-1, p-1), (p-1, p, p-1), (p-1, p-1, p)))
            for location, orders in zip(D_FIELD_LOCATIONS + B_FIELD_LOCATIONS, electric + magnetic):
                with self.subTest(p=p, location=location):
                    grid = location_grid(center, vertex, location)
                    expected = reference_gather(field, position, grid, orders)
                    actual = interpolate_field_to_particles(
                        field, *position.T, grid, p, axis_orders=orders,
                    )
                    self.assertEqual(actual.shape, (4, 2, 2))
                    self.assertTrue(jnp.allclose(actual, expected, rtol=2e-14, atol=2e-14))

    def test_uniform_default_preserves_cardinal_interpolation(self):
        grid = (jnp.arange(8, dtype=float),) * 3
        field = jax.random.normal(jax.random.PRNGKey(29), (8, 8, 8), dtype=jnp.float64)
        position = jnp.asarray(((0.5, 1.5, 2.5), (7.875, -0.25, 0.0)))
        for p in (1, 2):
            self.assertEqual(field_component_orders(p, "uniform"), ((None,) * 3, (None,) * 3))
            actual = interpolate_field_to_particles(field, *position.T, grid, p)
            expected = reference_gather(field, position, grid, (p, p, p))
            self.assertTrue(jnp.allclose(actual, expected, rtol=2e-14, atol=2e-14))

    def test_zero_order_half_cell_ties_and_periodic_images(self):
        x = jnp.asarray((-0.5, 0.5, 1.5, 7.5, 8.5))
        singleton = jnp.asarray((0.0,))
        grid = (jnp.arange(8, dtype=float), singleton, singleton)
        field = jnp.arange(8, dtype=float).reshape(8, 1, 1)
        actual = interpolate_field_to_particles(
            field, x, jnp.zeros_like(x), jnp.zeros_like(x), grid, 1, axis_orders=(0, 1, 1),
        )
        self.assertTrue(jnp.array_equal(actual, jnp.asarray((0., 1., 2., 0., 1.))))

    def test_reduced_axes_use_owned_cell_not_ghost_values(self):
        axis = jnp.arange(9, dtype=float)
        grid = (axis, jnp.arange(7, dtype=float), jnp.arange(7, dtype=float))
        field = jnp.full((9, 7, 7), 1e6).at[:, 3, 3].set(axis**2)
        x = jnp.asarray((3.25, 4.75))
        for orders in ((0, 1, 1), (2, 1, 2), (1, 0, 0)):
            actual = interpolate_field_to_particles(
                field, x, jnp.full_like(x, -12.25), jnp.full_like(x, 27.5), grid, 2,
                ghost_cells=True, active_axes=(True, False, False),
                inactive_axis_indices=(3, 3, 3), axis_orders=orders,
            )
            expected = cardinal_weights(x, axis, orders[0]) @ axis**2
            self.assertTrue(jnp.allclose(actual, expected, rtol=2e-14, atol=2e-14))

    def test_tile_faces_and_periodic_edges_match_global_gather(self):
        g = 2
        global_axis = jnp.arange(8, dtype=float) - 3.5
        global_field = jnp.sin(2 * jnp.pi * global_axis / 8.)
        for p in (1, 2):
            for tx, x in ((0, jnp.asarray((-3.99, -0.01))), (1, jnp.asarray((0.0, 3.99)))):
                # Populate a tile and its periodic halos explicitly: gathering
                # itself must not require multiple execution devices.
                indices = tx * 4 + jnp.arange(-g, 4+g)
                field = jnp.broadcast_to(global_field[jnp.mod(indices, 8)][:, None, None], (8, 5, 5))
                grid = (global_axis[0] + indices, jnp.arange(-g, g+1, dtype=float),
                        jnp.arange(-g, g+1, dtype=float))
                actual = interpolate_field_to_particles(
                    field, x, jnp.zeros_like(x), jnp.zeros_like(x), grid, p,
                    ghost_cells=True, active_axes=(True, False, False),
                    inactive_axis_indices=(g, g, g), axis_orders=(p-1, p, p),
                )
                expected = cardinal_weights(x, global_axis, p-1) @ global_field
                self.assertTrue(jnp.allclose(actual, expected, rtol=2e-14, atol=2e-14))

    def test_flat_and_gr_pushers_use_galerkin_electric_gather_and_seed(self):
        position = jnp.asarray(((-0.31, 0.17, 0.28), (0.41, -0.34, 0.12)))
        particles = TiledParticles(
            x=position.reshape(1, 1, 1, 1, 2, 3),
            u=jnp.zeros((1, 1, 1, 1, 2, 3)),
            active=jnp.ones((1, 1, 1, 1, 2), dtype=bool),
        )
        species = SpeciesConfig(jnp.asarray([2.]), jnp.asarray([3.]), jnp.asarray([1.]),
                                jnp.ones((1, 3), dtype=bool))
        for p in (1, 2):
            s, d = kernel_parameters(Nx=4, Ny=4, Nz=4, x_wind=2., y_wind=2., z_wind=2.,
                                     shape_factor=p, guard_cells=3, dt=0.02,
                                     current_deposition="esirkepov", field_gather="galerkin",
                                     relativistic=False, particle_batch_size=2)
            center = tuple(axis[0, 0, 0] for axis in d.grids.tiled_center_grid)
            vertex = tuple(axis[0, 0, 0] for axis in d.grids.tiled_vertex_grid)
            electric, _ = field_component_orders(p, "galerkin")
            E, expected = [], []
            for i, location in enumerate(D_FIELD_LOCATIONS):
                grid = location_grid(center, vertex, location)
                X, Y, Z = jnp.meshgrid(*grid, indexing="ij")
                field = jnp.sin(jnp.pi * X) + 0.3 * jnp.cos(jnp.pi * Y) + (i+1) * jnp.sin(jnp.pi * Z)
                E.append(field[None, None, None])
                owned = (slice(3, -3),) * 3
                expected.append(reference_gather(field[owned], position, tuple(a[3:-3] for a in grid), electric[i]))
            E = tuple(E)
            B = tuple(jnp.zeros_like(component) for component in E)
            expected_u = (2. / 3.) * d.dt * jnp.stack(expected, axis=-1)
            flat = particle_push(particles, species, E, B, s, d)
            self.assertTrue(jnp.allclose(flat.u.reshape(2, 3), expected_u, rtol=2e-14, atol=2e-14))

            gr_s = s._replace(solver="static_metric", particle_pusher="hybrid_boris_geodesic")
            metric = initialize_flat_cartesian_metric(gr_s, d)
            gr, _ = hybrid_boris_geodesic_push(particles, species, E, B, metric, gr_s, d, advance_positions=False)
            self.assertTrue(jnp.allclose(gr.u.reshape(2, 3), expected_u, rtol=2e-14, atol=2e-14))
            for params, geometry in ((s, None), (gr_s, metric)):
                seeded = seed_leapfrog_velocity(particles, species, E, B, params, d, metric=geometry)
                self.assertTrue(jnp.allclose(seeded.u.reshape(2, 3), -0.5 * expected_u, rtol=2e-14, atol=2e-14))
                self.assertTrue(jnp.array_equal(seeded.x, particles.x))

    def test_gr_undensitizes_each_native_component_before_gather(self):
        s, d = kernel_parameters(Nx=4, Ny=4, Nz=4, guard_cells=3,
                                 solver="static_metric", current_deposition="esirkepov",
                                 particle_pusher="hybrid_boris_geodesic", field_gather="galerkin")
        metric = initialize_flat_cartesian_metric(s, d)
        # Distinct positive node factors expose gathering the density first.
        def varying_factor(sample, offset):
            factor = 1.0 + offset + jnp.arange(sample.sqrt_gamma.size, dtype=float).reshape(sample.sqrt_gamma.shape) / 1000.
            return sample._replace(sqrt_gamma=factor)
        metric = metric._replace(D=tuple(varying_factor(m, i) for i, m in enumerate(metric.D)),
                                 B=tuple(varying_factor(m, i+3) for i, m in enumerate(metric.B)))
        D = tuple((i+1.) * m.sqrt_gamma for i, m in enumerate(metric.D))
        B = tuple((i+4.) * m.sqrt_gamma for i, m in enumerate(metric.B))
        tile = tile_fields(0, 0, 0, metric, D, B, d)
        position = jnp.asarray(((0.31, 0.17, -0.11), (-0.18, 0.22, 0.14)))
        for p in (1, 2):
            electric, magnetic = field_component_orders(p, "galerkin")
            for field, locations, orders, values in ((tile.D, D_FIELD_LOCATIONS, electric, (1., 2., 3.)),
                                                       (tile.B, B_FIELD_LOCATIONS, magnetic, (4., 5., 6.))):
                actual = gather_vector(field, locations, position, tile.center_grid, tile.vertex_grid,
                                       p, (True, True, True), (3, 3, 3), orders)
                self.assertTrue(jnp.allclose(actual, jnp.asarray(values), rtol=2e-14, atol=2e-14))

    def test_dynamic_nonconstant_fields_keep_gather_through_sync_rephase_and_half_steps(self):
        from tests.code_tests.dynamical_relativity_test import coupled_state
        from PyPIC3D.relativity.dynamical_relativity import (
            initialize_dynamical_relativity, rephase_pic_state,
            synchronize_particles, synchronized_magnetic_field,
        )
        from PyPIC3D.solvers.GR_yee.time_loop import time_loop_static_metric
        from PyPIC3D.solvers.GR_yee.time_loop_dynamical_relativity import time_loop_dynamical_relativity

        s, d, particles, species, fields = coupled_state(particles=True)
        x_grid = d.grids.tiled_vertex_grid[0]
        Ex = jnp.broadcast_to(0.1 * jnp.sin(2 * jnp.pi * x_grid[..., :, None, None] / 8.), fields[0][0].shape)
        zero = tuple(jnp.zeros_like(q) for q in fields[0])
        E = (Ex, zero[1], zero[2])
        initial_u = jnp.zeros_like(particles.u).at[..., 0].set(0.15)
        initial = particles._replace(u=initial_u)
        raw_fields = (E, zero, zero, fields[3], fields[4], (zero, zero), None,
                      (E, zero), jnp.asarray(False))
        outcomes = []

        for mode in ("uniform", "galerkin"):
            with self.subTest(field_gather=mode):
                static = s._replace(field_gather=mode, shape_factor=2)
                seeded, start, dynamic = initialize_dynamical_relativity(
                    initial, species, raw_fields, static, d,
                )
                h = start[7][3]
                order = 1 if mode == "galerkin" else 2
                axis = x_grid[0, 0, 0, 3:-3]
                x = initial.x.reshape(-1, 3)[:, 0]
                E_at_x = cardinal_weights(x, axis, order) @ Ex[0, 0, 0, 3:-3, 3, 3]
                force = jnp.zeros_like(initial.u).at[..., 0].set(
                    (species.charge[0] / species.mass[0]) * E_at_x.reshape(initial.u.shape[:-1]))
                self.assertTrue(jnp.allclose(seeded.u, initial.u - h / 2 * force, rtol=2e-13, atol=2e-13))
                common = synchronize_particles(seeded, species, start, static, dynamic)
                self.assertTrue(jnp.allclose(common.u, initial.u, rtol=2e-13, atol=2e-13))
                self.assertTrue(jnp.array_equal(common.x, initial.x))

                metric = start[6]
                new_metric = metric._replace(
                    D=tuple(m._replace(lapse=1.2 * m.lapse) for m in metric.D),
                    B=tuple(m._replace(lapse=1.2 * m.lapse) for m in metric.B),
                    center=metric.center._replace(lapse=1.2 * metric.center.lapse),
                    vertex=metric.vertex._replace(lapse=1.2 * metric.vertex.lapse),
                )
                rephased, _ = rephase_pic_state(seeded, species, start, new_metric, static, dynamic)
                self.assertTrue(jnp.allclose(rephased.u, seeded.u - 0.1 * h * force, rtol=2e-13, atol=2e-13))

                half = dynamic._replace(dt=h)
                ordinary = start[:7] + (start[7][:2], start[8])
                advance = jax.jit(lambda p, f: time_loop_static_metric(p, species, f, static, half))
                expected_p, midpoint = advance(seeded, ordinary)
                expected_p, expected_f = advance(expected_p, midpoint)
                before_second = midpoint[:7] + ((*midpoint[7], start[7][2], h, None), midpoint[8])
                common_B = synchronized_magnetic_field(before_second, expected_f, static, half)
                expected_f = expected_f[:7] + ((*expected_f[7], common_B, h, None), expected_f[8])
                actual_p, actual_f, _ = jax.jit(
                    lambda p, f, d: time_loop_dynamical_relativity(p, species, f, static, d)
                )(seeded, start, dynamic)
                for actual, expected in zip(jax.tree.leaves((actual_p, actual_f)),
                                            jax.tree.leaves((expected_p, expected_f))):
                    self.assertTrue(jnp.allclose(actual, expected, rtol=2e-12, atol=2e-12))
                outcomes.append(actual_p.u)

        # Nonconstant longitudinal E must actually distinguish the two gathers.
        self.assertGreater(float(jnp.max(jnp.abs(outcomes[1] - outcomes[0]))), 1e-6)


if __name__ == "__main__":
    unittest.main()
