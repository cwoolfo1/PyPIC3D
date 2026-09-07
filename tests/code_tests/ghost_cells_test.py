import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_CONSTANT, BC_PERIODIC
from PyPIC3D.boundary_conditions import ghost_cells

jax.config.update("jax_enable_x64", True)


def _assert_allclose(test_case, actual, expected, **kwargs):
    test_case.assertTrue(
        bool(jnp.allclose(jnp.asarray(actual), jnp.asarray(expected), **kwargs))
    )


class TestGhostCells(unittest.TestCase):
    """Tests for the tiled ghost-cell boundary condition approach."""

    def setUp(self):
        self.tile_shape = (2, 2, 2)
        self.g = 1
        self.parameter_set = {
            "tile_shape": self.tile_shape,
            "boundary_conditions": {"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC},
        }

    def _parameters_with_field_mesh(self, tile_grid_shape):
        parameter_set = dict(self.parameter_set)
        try:
            parameter_set["field_mesh"] = ghost_cells.make_field_mesh(tile_grid_shape)
        except ValueError as exc:
            self.skipTest(str(exc))
        return SimpleNamespace(
            tile_shape=tuple(int(width) for width in parameter_set["tile_shape"]),
            guard_cells=self.g,
            boundary_conditions=(
                int(parameter_set["boundary_conditions"]["x"]),
                int(parameter_set["boundary_conditions"]["y"]),
                int(parameter_set["boundary_conditions"]["z"]),
            ),
            field_mesh=parameter_set["field_mesh"],
        )

    def test_update_tiled_ghost_cells_periodic_refreshes_neighbor_halos(self):
        # this tests the communication of ghost cells between two tiles in a periodic domain

        parameter_set = self._parameters_with_field_mesh((2, 1, 1))
        field_tiles = jnp.zeros((2, 1, 1, 4, 4, 4))
        field_tiles = field_tiles.at[0, 0, 0, 1:3, 1:3, 1:3].set(1.0)
        field_tiles = field_tiles.at[1, 0, 0, 1:3, 1:3, 1:3].set(2.0)
        # create two tiles, one with a value of 1.0 and one with a value of 2.0 constant 
        # across the tile

        result = ghost_cells.update_tiled_ghost_cells(field_tiles, parameter_set, self.g)
        # call the update ghost cells method to communicate the ghost cells between the two tiles

        self.assertTrue(jnp.all(result[0, 0, 0, -1, 1:3, 1:3] == 2.0))
        # make sure the ghost cell on the first tile has been updated from 1.0 to 2.0
        self.assertTrue(jnp.all(result[1, 0, 0, 0, 1:3, 1:3] == 1.0))
        # make sure the ghost cell on the second tile has been updated from 2.0 to 1.0

    def test_update_tiled_ghost_cells_requires_static_field_mesh(self):
        # tiled halo exchange should use the startup-owned field mesh, not infer
        # a device mesh from the current array shape.

        field_tiles = jnp.zeros((1, 1, 1, 4, 4, 4))

        incomplete_parameters = SimpleNamespace(
            tile_shape=self.tile_shape,
            guard_cells=self.g,
            boundary_conditions=(BC_PERIODIC, BC_PERIODIC, BC_PERIODIC),
        )

        with self.assertRaises(AttributeError):
            ghost_cells.update_tiled_ghost_cells(field_tiles, incomplete_parameters, self.g)

    def test_fold_tiled_ghost_cells_periodic_adds_to_owner_tile(self):
        # this tests the folding of ghost cells back to the owner tile in a periodic domain
        # this is used to confirm the current and charge deposition is correct when using ghost cells

        parameter_set = self._parameters_with_field_mesh((2, 1, 1))
        field_tiles = jnp.zeros((2, 1, 1, 4, 4, 4))
        # create two tiles with a shape of (4, 4, 4) and a ghost cell width of 1
        field_tiles = field_tiles.at[0, 0, 0, -1, 2, 2].set(3.0)
        # set the ghost cell on the first tile at the right x boundary to a value of 3.0
        field_tiles = field_tiles.at[1, 0, 0, 0, 2, 2].set(5.0)
        # set the ghost cell on the second tile at the left x boundary to a value of 5.0

        result = ghost_cells.fold_tiled_ghost_cells(field_tiles, parameter_set, self.g)
        # call the fold ghost cells method to add the ghost cell values back to the owner tile

        self.assertAlmostEqual(float(result[1, 0, 0, 1, 2, 2]), 3.0)
        # make sure the ghost cell value of 3.0 on the first tile has been added to the owner tile
        self.assertAlmostEqual(float(result[0, 0, 0, -2, 2, 2]), 5.0)
        # make sure the ghost cell value of 5.0 on the second tile has been added to the owner tile
        self.assertEqual(float(result[0, 0, 0, -1, 2, 2]), 0.0)
        self.assertEqual(float(result[1, 0, 0, 0, 2, 2]), 0.0)
        # make sure the ghost cell values have been reset to 0.0 after folding

    def test_apply_tiled_zero_boundary_zeros_global_tangential_faces(self):
        # this tests the application of axis-wise conducting boundary conditions to a tiled electric field

        parameter_set = SimpleNamespace(
            tile_shape=self.tile_shape,
            guard_cells=self.g,
            field_mesh=ghost_cells.make_field_mesh((1, 1, 1)),
            boundary_conditions=(BC_CONDUCTING, BC_CONDUCTING, BC_CONDUCTING),
        )
        # create a parameter_set with conducting boundary conditions in all directions

        E = tuple(jnp.ones((1, 1, 1, 4, 4, 4)) for _ in range(3))
        # create a tuple of three electric field components (Ex, Ey, Ez) with shape (1, 1, 1, 4, 4, 4) and all values set to 1.0

        Ex, Ey, Ez = E
        Ey = ghost_cells.apply_tiled_zero_boundary(Ey, parameter_set, axis=0, num_guard_cells=self.g)
        Ez = ghost_cells.apply_tiled_zero_boundary(Ez, parameter_set, axis=0, num_guard_cells=self.g)
        Ex = ghost_cells.apply_tiled_zero_boundary(Ex, parameter_set, axis=1, num_guard_cells=self.g)
        Ez = ghost_cells.apply_tiled_zero_boundary(Ez, parameter_set, axis=1, num_guard_cells=self.g)
        Ex = ghost_cells.apply_tiled_zero_boundary(Ex, parameter_set, axis=2, num_guard_cells=self.g)
        Ey = ghost_cells.apply_tiled_zero_boundary(Ey, parameter_set, axis=2, num_guard_cells=self.g)
        # call the shared scalar zero boundary method for each tangential electric component

        self.assertTrue(jnp.all(Ey[0, 0, 0, 1, :, :] == 0.0))
        self.assertTrue(jnp.all(Ez[0, 0, 0, 1, :, :] == 0.0))
        self.assertTrue(jnp.all(Ex[0, 0, 0, :, 1, :] == 0.0))
        self.assertTrue(jnp.all(Ez[0, 0, 0, :, 1, :] == 0.0))
        self.assertTrue(jnp.all(Ex[0, 0, 0, :, :, 1] == 0.0))
        self.assertTrue(jnp.all(Ey[0, 0, 0, :, :, 1] == 0.0))
        # make sure the tangential faces of the electric field components have been zeroed out

    def test_apply_tiled_zero_boundary_periodic_axis_keeps_refreshed_field(self):
        parameter_set = self._parameters_with_field_mesh((1, 1, 1))
        field = jnp.arange(4 * 4 * 4, dtype=float).reshape((1, 1, 1, 4, 4, 4))
        # create a scalar field with shape (1, 1, 1, 4, 4, 4) and values from 0 to 63
        refreshed = ghost_cells.update_tiled_ghost_cells(field, parameter_set, self.g)
        result = ghost_cells.apply_tiled_zero_boundary(refreshed, parameter_set, axis=0, num_guard_cells=self.g)
        # periodic field boundaries do not zero the physical plane

        self.assertTrue(jnp.allclose(result, refreshed))
        # confirm the field has not been modified after an already consistent periodic halo refresh

    def test_apply_tiled_constant_boundary_copies_adjacent_interior_to_global_ghosts(self):
        parameter_set = SimpleNamespace(
            tile_shape=self.tile_shape,
            guard_cells=self.g,
            field_mesh=ghost_cells.make_field_mesh((1, 1, 1)),
            boundary_conditions=(BC_CONDUCTING, BC_PERIODIC, BC_PERIODIC),
        )
        field = jnp.arange(4 * 4 * 4, dtype=float).reshape((1, 1, 1, 4, 4, 4))

        result = ghost_cells.apply_tiled_constant_boundary(field, parameter_set, axis=0, num_guard_cells=self.g)

        self.assertTrue(jnp.allclose(result[0, 0, 0, 0, :, :], result[0, 0, 0, 1, :, :]))
        self.assertTrue(jnp.allclose(result[0, 0, 0, -1, :, :], result[0, 0, 0, -2, :, :]))
        self.assertTrue(jnp.allclose(result[0, 0, 0, 1:-1, 1:-1, 1:-1], field[0, 0, 0, 1:-1, 1:-1, 1:-1]))

    def test_apply_tiled_constant_boundary_periodic_axis_keeps_refreshed_field(self):
        parameter_set = self._parameters_with_field_mesh((1, 1, 1))
        field = jnp.arange(4 * 4 * 4, dtype=float).reshape((1, 1, 1, 4, 4, 4))
        refreshed = ghost_cells.update_tiled_ghost_cells(field, parameter_set, self.g)

        result = ghost_cells.apply_tiled_constant_boundary(field, parameter_set, axis=0, num_guard_cells=self.g)

        self.assertTrue(jnp.allclose(result, refreshed))

    def test_update_tiled_ghost_cells_constant_copies_adjacent_global_interiors(self):
        parameter_set = SimpleNamespace(
            tile_shape=self.tile_shape,
            guard_cells=self.g,
            field_mesh=ghost_cells.make_field_mesh((1, 1, 1)),
            boundary_conditions=(BC_CONSTANT, BC_PERIODIC, BC_PERIODIC),
        )
        field = jnp.zeros((1, 1, 1, 4, 4, 4), dtype=float)
        field = field.at[0, 0, 0, 1, :, :].set(2.0)
        field = field.at[0, 0, 0, 2, :, :].set(7.0)

        result = ghost_cells.update_tiled_ghost_cells(field, parameter_set, self.g)

        self.assertTrue(jnp.allclose(result[0, 0, 0, 0, :, :], result[0, 0, 0, 1, :, :]))
        self.assertTrue(jnp.allclose(result[0, 0, 0, -1, :, :], result[0, 0, 0, -2, :, :]))
        self.assertTrue(jnp.allclose(result[0, 0, 0, 1, :, :], 2.0))
        self.assertTrue(jnp.allclose(result[0, 0, 0, 2, :, :], 7.0))

    def test_particle_scalar_fold_and_refresh_mirror_both_reflecting_walls(self):
        g = 2
        parameters = SimpleNamespace(
            tile_shape=(2, 2, 4),
            guard_cells=g,
            field_mesh=ghost_cells.make_field_mesh((1, 1, 1)),
            boundary_conditions=(BC_PERIODIC, BC_PERIODIC, BC_CONDUCTING),
            particle_boundary_conditions=(BC_PERIODIC, BC_PERIODIC, BC_CONDUCTING),
        )
        deposits = jnp.zeros((1, 1, 1, 6, 6, 8), dtype=float)
        line = (0, 0, 0, g, g)
        deposits = deposits.at[line + (slice(g, 2 * g),)].set(jnp.array([10.0, 20.0]))
        deposits = deposits.at[line + (slice(-2 * g, -g),)].set(jnp.array([30.0, 40.0]))
        deposits = deposits.at[line + (slice(0, g),)].set(jnp.array([1.0, 2.0]))
        deposits = deposits.at[line + (slice(-g, None),)].set(jnp.array([3.0, 4.0]))

        even = ghost_cells.fold_tiled_ghost_cells(
            deposits,
            parameters,
            g,
            bc_type=ghost_cells.BC_TYPE_PARTICLE,
        )
        odd = ghost_cells.fold_tiled_ghost_cells(
            deposits,
            parameters,
            g,
            bc_type=ghost_cells.BC_TYPE_PARTICLE,
            reflecting_parity=(1, 1, -1),
        )

        # Lower storage is [far, near] and upper storage is [near, far].
        # Both must be reversed so nearest ghost maps to nearest interior.
        _assert_allclose(self, even[line + (slice(g, 2 * g),)], [12.0, 21.0])
        _assert_allclose(self, even[line + (slice(-2 * g, -g),)], [34.0, 43.0])
        _assert_allclose(self, odd[line + (slice(g, 2 * g),)], [8.0, 19.0])
        _assert_allclose(self, odd[line + (slice(-2 * g, -g),)], [26.0, 37.0])

        refreshed_even = ghost_cells.update_tiled_ghost_cells(
            even,
            parameters,
            g,
            bc_type=ghost_cells.BC_TYPE_PARTICLE,
        )
        refreshed_odd = ghost_cells.update_tiled_ghost_cells(
            odd,
            parameters,
            g,
            bc_type=ghost_cells.BC_TYPE_PARTICLE,
            reflecting_parity=(1, 1, -1),
        )
        _assert_allclose(self, refreshed_even[line + (slice(0, g),)], [21.0, 12.0])
        _assert_allclose(self, refreshed_even[line + (slice(-g, None),)], [43.0, 34.0])
        _assert_allclose(self, refreshed_odd[line + (slice(0, g),)], [-19.0, -8.0])
        _assert_allclose(self, refreshed_odd[line + (slice(-g, None),)], [-37.0, -26.0])

    def test_particle_vector_has_tangential_even_and_normal_odd_wall_parity(self):
        g = 2
        parameters = SimpleNamespace(
            tile_shape=(2, 2, 4),
            guard_cells=g,
            field_mesh=ghost_cells.make_field_mesh((1, 1, 1)),
            boundary_conditions=(BC_PERIODIC, BC_PERIODIC, BC_CONDUCTING),
            particle_boundary_conditions=(BC_PERIODIC, BC_PERIODIC, BC_CONDUCTING),
        )
        scalar = jnp.zeros((1, 1, 1, 6, 6, 8), dtype=float)
        line = (0, 0, 0, g, g)
        scalar = scalar.at[line + (slice(g, 2 * g),)].set(jnp.array([10.0, 20.0]))
        scalar = scalar.at[line + (slice(-2 * g, -g),)].set(jnp.array([30.0, 40.0]))
        scalar = scalar.at[line + (slice(0, g),)].set(jnp.array([1.0, 2.0]))
        scalar = scalar.at[line + (slice(-g, None),)].set(jnp.array([3.0, 4.0]))
        vector = (scalar, scalar, scalar)

        folded = ghost_cells.fold_tiled_vector_ghost_cells(
            vector,
            parameters,
            g,
            bc_type=ghost_cells.BC_TYPE_PARTICLE,
        )

        for component in (0, 1):
            _assert_allclose(self, folded[component][line + (slice(g, 2 * g),)], [12.0, 21.0])
            _assert_allclose(self, folded[component][line + (slice(-2 * g, -g),)], [34.0, 43.0])
        _assert_allclose(self, folded[2][line + (slice(g, 2 * g),)], [8.0, 19.0])
        _assert_allclose(self, folded[2][line + (slice(-2 * g, -g),)], [26.0, 37.0])

        refreshed = ghost_cells.update_tiled_vector_ghost_cells(
            folded,
            parameters,
            g,
            bc_type=ghost_cells.BC_TYPE_PARTICLE,
        )
        for component in (0, 1):
            _assert_allclose(self, refreshed[component][line + (slice(0, g),)], [21.0, 12.0])
            _assert_allclose(self, refreshed[component][line + (slice(-g, None),)], [43.0, 34.0])
        _assert_allclose(self, refreshed[2][line + (slice(0, g),)], [-19.0, -8.0])
        _assert_allclose(self, refreshed[2][line + (slice(-g, None),)], [-37.0, -26.0])

    def test_particle_vector_reflection_parity_applies_on_every_axis(self):
        g = 1
        tile_shape = (2, 2, 2)
        scalar = jnp.zeros((1, 1, 1, 4, 4, 4), dtype=float)
        scalar = scalar.at[0, 0, 0, g:-g, g:-g, g:-g].set(7.0)
        vector = (scalar, scalar, scalar)

        for wall_axis in range(3):
            particle_boundaries = [BC_PERIODIC, BC_PERIODIC, BC_PERIODIC]
            particle_boundaries[wall_axis] = BC_CONDUCTING
            parameters = SimpleNamespace(
                tile_shape=tile_shape,
                guard_cells=g,
                field_mesh=ghost_cells.make_field_mesh((1, 1, 1)),
                boundary_conditions=(BC_PERIODIC, BC_PERIODIC, BC_PERIODIC),
                particle_boundary_conditions=tuple(particle_boundaries),
            )
            refreshed = ghost_cells.update_tiled_vector_ghost_cells(
                vector,
                parameters,
                g,
                bc_type=ghost_cells.BC_TYPE_PARTICLE,
            )
            lower_wall = [0, 0, 0, g, g, g]
            lower_wall[3 + wall_axis] = 0
            upper_wall = [0, 0, 0, g, g, g]
            upper_wall[3 + wall_axis] = -1
            for component in range(3):
                expected = -7.0 if component == wall_axis else 7.0
                self.assertEqual(float(refreshed[component][tuple(lower_wall)]), expected)
                self.assertEqual(float(refreshed[component][tuple(upper_wall)]), expected)

    def test_reduced_reflecting_axis_uses_scalar_parity(self):
        g = 1
        parameters = SimpleNamespace(
            tile_shape=(1, 2, 2),
            guard_cells=g,
            field_mesh=ghost_cells.make_field_mesh((1, 1, 1)),
            boundary_conditions=(BC_PERIODIC, BC_PERIODIC, BC_PERIODIC),
            particle_boundary_conditions=(BC_CONDUCTING, BC_PERIODIC, BC_PERIODIC),
        )
        deposits = jnp.zeros((1, 1, 1, 3, 4, 4), dtype=float)
        line = (0, 0, 0, slice(None), g, g)
        deposits = deposits.at[line].set(jnp.array([2.0, 10.0, 3.0]))

        even = ghost_cells.fold_tiled_ghost_cells(
            deposits,
            parameters,
            g,
            bc_type=ghost_cells.BC_TYPE_PARTICLE,
        )
        odd = ghost_cells.fold_tiled_ghost_cells(
            deposits,
            parameters,
            g,
            bc_type=ghost_cells.BC_TYPE_PARTICLE,
            reflecting_parity=(-1, 1, 1),
        )
        self.assertEqual(float(even[0, 0, 0, g, g, g]), 15.0)
        self.assertEqual(float(odd[0, 0, 0, g, g, g]), 5.0)

        refreshed_even = ghost_cells.update_tiled_ghost_cells(
            even,
            parameters,
            g,
            bc_type=ghost_cells.BC_TYPE_PARTICLE,
        )
        refreshed_odd = ghost_cells.update_tiled_ghost_cells(
            odd,
            parameters,
            g,
            bc_type=ghost_cells.BC_TYPE_PARTICLE,
            reflecting_parity=(-1, 1, 1),
        )
        _assert_allclose(self, refreshed_even[line], [15.0, 15.0, 15.0])
        _assert_allclose(self, refreshed_odd[line], [-5.0, 5.0, -5.0])

    def test_reflecting_corner_composes_axis_parity(self):
        g = 1
        parameters = SimpleNamespace(
            tile_shape=(2, 2, 2),
            guard_cells=g,
            field_mesh=ghost_cells.make_field_mesh((1, 1, 1)),
            boundary_conditions=(BC_PERIODIC, BC_PERIODIC, BC_PERIODIC),
            particle_boundary_conditions=(BC_CONDUCTING, BC_PERIODIC, BC_CONDUCTING),
        )
        parity = (-1, 1, -1)
        field = jnp.zeros((1, 1, 1, 4, 4, 4), dtype=float)
        field = field.at[0, 0, 0, g, g, g].set(7.0)
        refreshed = ghost_cells.update_tiled_ghost_cells(
            field,
            parameters,
            g,
            bc_type=ghost_cells.BC_TYPE_PARTICLE,
            reflecting_parity=parity,
        )
        self.assertEqual(float(refreshed[0, 0, 0, 0, g, g]), -7.0)
        self.assertEqual(float(refreshed[0, 0, 0, g, g, 0]), -7.0)
        self.assertEqual(float(refreshed[0, 0, 0, 0, g, 0]), 7.0)

        deposits = jnp.zeros_like(field).at[0, 0, 0, 0, g, 0].set(3.0)
        folded = ghost_cells.fold_tiled_ghost_cells(
            deposits,
            parameters,
            g,
            bc_type=ghost_cells.BC_TYPE_PARTICLE,
            reflecting_parity=parity,
        )
        self.assertEqual(float(folded[0, 0, 0, g, g, g]), 3.0)

    def test_reflecting_parity_validation_and_field_rejection(self):
        parameters = SimpleNamespace(
            tile_shape=(2, 2, 2),
            guard_cells=1,
            field_mesh=ghost_cells.make_field_mesh((1, 1, 1)),
            boundary_conditions=(BC_CONDUCTING, BC_PERIODIC, BC_PERIODIC),
            particle_boundary_conditions=(BC_CONDUCTING, BC_PERIODIC, BC_PERIODIC),
        )
        scalar = jnp.zeros((1, 1, 1, 4, 4, 4), dtype=float)

        with self.assertRaisesRegex(ValueError, "three values"):
            ghost_cells.update_tiled_ghost_cells(
                scalar,
                parameters,
                bc_type=ghost_cells.BC_TYPE_PARTICLE,
                reflecting_parity=(1, -1),
            )
        with self.assertRaisesRegex(ValueError, "either -1 or 1"):
            ghost_cells.fold_tiled_ghost_cells(
                scalar,
                parameters,
                bc_type=ghost_cells.BC_TYPE_PARTICLE,
                reflecting_parity=(1, 0, 1),
            )
        with self.assertRaisesRegex(ValueError, "either -1 or 1"):
            ghost_cells.fold_tiled_ghost_cells(
                scalar,
                parameters,
                bc_type=ghost_cells.BC_TYPE_PARTICLE,
                reflecting_parity=(1, 1.5, 1),
            )
        with self.assertRaisesRegex(ValueError, "three values"):
            ghost_cells.update_tiled_ghost_cells(
                scalar,
                parameters,
                bc_type=ghost_cells.BC_TYPE_PARTICLE,
                reflecting_parity=1,
            )
        with self.assertRaisesRegex(ValueError, "only valid for particle"):
            ghost_cells.update_tiled_ghost_cells(
                scalar,
                parameters,
                reflecting_parity=(1, 1, 1),
            )
        with self.assertRaisesRegex(ValueError, "one parity tuple per component"):
            ghost_cells.update_tiled_vector_ghost_cells(
                (scalar, scalar, scalar),
                parameters,
                bc_type=ghost_cells.BC_TYPE_PARTICLE,
                reflecting_parity=((1, 1, 1), (1, -1, 1)),
            )


if __name__ == '__main__':
    unittest.main()
