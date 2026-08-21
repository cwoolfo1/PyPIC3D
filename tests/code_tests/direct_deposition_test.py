import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import (
    BC_CONDUCTING,
    BC_CONSTANT,
    BC_PERIODIC,
    prepare_particle_axis_stencil,
)
from PyPIC3D.deposition.J_from_rhov import J_from_rhov
from PyPIC3D.deposition.shapes import get_first_order_weights
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.particles.particle_tile_communication import refresh_tiled_particle_tiles
from PyPIC3D.diagnostics.output_adapters import assemble_tiled_vector_field
from PyPIC3D.utilities.grids import build_tiled_yee_grids, build_yee_grid
from tests.kernel_fixtures import field_tiles_from_global, kernel_parameters_from_values


jax.config.update("jax_enable_x64", True)


def _tile_axis_count(n_cells, cells_per_tile):
    if int(n_cells) % int(cells_per_tile) != 0:
        raise ValueError("Shared tile sizes must divide the physical grid dimensions exactly.")
    return int(n_cells) // int(cells_per_tile)
# compute the number of tiles along each axis, ensuring that the number of cells is divisible by the number of cells per tile.


def tile_scalar_field(field, parameter_set, tile_shape, num_guard_cells=2):
    parameter_set = dict(parameter_set)
    parameter_set["tile_shape"] = tuple(int(width) for width in tile_shape)
    parameter_set["field_mesh"] = ghost_cells.make_field_mesh(
        tuple(
            int(parameter_set[axis]) // int(width)
            for axis, width in zip(("Nx", "Ny", "Nz"), tile_shape)
        )
    )
    static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set)
    return field_tiles_from_global(
        field,
        static_parameters,
        dynamic_parameters,
        num_guard_cells=num_guard_cells,
    )


def tile_vector_field(field, parameter_set, tile_shape, num_guard_cells=2):
    return tuple(tile_scalar_field(component, parameter_set, tile_shape, num_guard_cells) for component in field)


def _field_static_parameters(parameter_set):
    static_parameters, _ = kernel_parameters_from_values(parameter_set)
    return static_parameters
    # call tile_scalar_field for each component of the vector field and return a tuple of tiled components


def _update_ghost_cells(field, bc_x, bc_y, bc_z):
    field = jax.lax.cond(
        bc_x == BC_PERIODIC,
        lambda f: f.at[0, :, :].set(f[-2, :, :]).at[-1, :, :].set(f[1, :, :]),
        lambda f: f.at[0, :, :].set(0.0).at[-1, :, :].set(0.0),
        operand=field,
    )
    field = jax.lax.cond(
        bc_y == BC_PERIODIC,
        lambda f: f.at[:, 0, :].set(f[:, -2, :]).at[:, -1, :].set(f[:, 1, :]),
        lambda f: f.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0),
        operand=field,
    )
    field = jax.lax.cond(
        bc_z == BC_PERIODIC,
        lambda f: f.at[:, :, 0].set(f[:, :, -2]).at[:, :, -1].set(f[:, :, 1]),
        lambda f: f.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0),
        operand=field,
    )
    return field


def _fold_ghost_cells(field, bc_x, bc_y, bc_z):
    field = jax.lax.cond(
        bc_x == BC_PERIODIC,
        lambda f: f.at[1, :, :].add(f[-1, :, :]).at[-2, :, :].add(f[0, :, :]),
        lambda f: f.at[1, :, :].add(-f[0, :, :]).at[-2, :, :].add(-f[-1, :, :]),
        operand=field,
    )
    field = field.at[0, :, :].set(0.0).at[-1, :, :].set(0.0)
    field = jax.lax.cond(
        bc_y == BC_PERIODIC,
        lambda f: f.at[:, 1, :].add(f[:, -1, :]).at[:, -2, :].add(f[:, 0, :]),
        lambda f: f.at[:, 1, :].add(-f[:, 0, :]).at[:, -2, :].add(-f[:, -1, :]),
        operand=field,
    )
    field = field.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0)
    field = jax.lax.cond(
        bc_z == BC_PERIODIC,
        lambda f: f.at[:, :, 1].add(f[:, :, -1]).at[:, :, -2].add(f[:, :, 0]),
        lambda f: f.at[:, :, 1].add(-f[:, :, 0]).at[:, :, -2].add(-f[:, :, -1]),
        operand=field,
    )
    field = field.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
    return field


class TestDirectDeposition(unittest.TestCase):
    def test_face_stencil_at_grid_node_is_independent_of_tile_origin(self):
        dx = 0.5
        position = jnp.asarray([-1.1102230246251565e-16])
        grid_axes = (
            jnp.asarray((-1.0, -0.5, 0.0, 0.5)),
            jnp.asarray((-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)),
        )
        face_stencils = []

        for grid_axis in grid_axes:
            face_grid_axis = grid_axis + 0.5 * dx
            _, _, delta_face, points_face = prepare_particle_axis_stencil(
                position,
                face_grid_axis,
                len(face_grid_axis),
                shape_factor=1,
                bc=BC_CONSTANT,
                ghost_cells=True,
            )
            weights_face, _, _ = get_first_order_weights(
                delta_face,
                delta_face,
                delta_face,
                dx,
                dx,
                dx,
            )
            weights_face = jnp.stack(weights_face)[:, 0]
            coordinates_face = face_grid_axis[0] + points_face[:, 0] * dx
            face_stencils.append((coordinates_face, weights_face))

            self.assertTrue(jnp.all(weights_face >= 0.0))
            self.assertAlmostEqual(float(jnp.sum(weights_face)), 1.0)

        for tiled_values, one_tile_values in zip(face_stencils[0], face_stencils[1]):
            self.assertTrue(jnp.allclose(tiled_values, one_tile_values))
        self.assertTrue(jnp.allclose(face_stencils[0][1], jnp.asarray((0.0, 0.5, 0.5))))

    def _build_parameter_values(self, Nx=8, Ny=6, Nz=4, dt=0.05, boundary_conditions=None):
        x_wind, y_wind, z_wind = 4.0, 3.0, 2.0
        if boundary_conditions is None:
            boundary_conditions = {"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC}
        parameter_set = {
            "dx": x_wind / Nx,
            "dy": y_wind / Ny,
            "dz": z_wind / Nz,
            "Nx": Nx,
            "Ny": Ny,
            "Nz": Nz,
            "x_wind": x_wind,
            "y_wind": y_wind,
            "z_wind": z_wind,
            "dt": dt,
            "shape_factor": 1,
            "guard_cells": 1,
            "boundary_conditions": boundary_conditions,
        }
        center_grid, vertex_grid = build_yee_grid(SimpleNamespace(**parameter_set))
        parameter_set["grids"] = {"center": center_grid, "vertex": vertex_grid}
        return parameter_set

    def _empty_J(self, parameter_set):
        shape = (parameter_set["Nx"] + 2, parameter_set["Ny"] + 2, parameter_set["Nz"] + 2)
        return (jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))

    def _empty_J_tiles(self, parameter_set):
        tile_shape = tuple(int(width) for width in parameter_set["tile_shape"])
        tile_nx, tile_ny, tile_nz = tile_shape
        g = int(parameter_set["guard_cells"])
        shape = (
            parameter_set["Nx"] // tile_nx,
            parameter_set["Ny"] // tile_ny,
            parameter_set["Nz"] // tile_nz,
            tile_nx + 2 * g,
            tile_ny + 2 * g,
            tile_nz + 2 * g,
        )
        return (jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))
    # create an empty tiled current density field with the appropriate shape based on the parameter_set and tile shape

    def _tile_shape(self, simulation_parameters):
        return (
            simulation_parameters["particle_tile_nx"],
            simulation_parameters["particle_tile_ny"],
            simulation_parameters["particle_tile_nz"],
        )
    # get the shape of the particle tiles from the simulation parameters

    def _parameters_with_tiled_grids(self, parameter_set, tile_shape):
        g = int(parameter_set["guard_cells"])
        parameter_set = dict(parameter_set)
        grids = dict(parameter_set["grids"])
        parameter_set["tile_shape"] = tile_shape
        parameter_set["field_mesh"] = ghost_cells.make_field_mesh((
            int(parameter_set["Nx"]) // int(tile_shape[0]),
            int(parameter_set["Ny"]) // int(tile_shape[1]),
            int(parameter_set["Nz"]) // int(tile_shape[2]),
        ))
        grid_static_parameters = SimpleNamespace(tile_shape=tile_shape, guard_cells=g)
        grid_dynamic_parameters = SimpleNamespace(
            dx=parameter_set["dx"],
            dy=parameter_set["dy"],
            dz=parameter_set["dz"],
            grids=SimpleNamespace(vertex=grids["vertex"], center=grids["center"]),
        )
        tiled_center_grid, tiled_vertex_grid = build_tiled_yee_grids(
            grid_static_parameters,
            grid_dynamic_parameters,
        )
        grids["tiled_center_grid"] = tiled_center_grid
        grids["tiled_vertex_grid"] = tiled_vertex_grid
        parameter_set["grids"] = grids
        return parameter_set
    # create a new parameter_set dictionary that includes the tiled grids based on the given tile shape

    def _one_tile_parameters(self, parameter_set):
        return {
            "particle_tile_nx": parameter_set["Nx"],
            "particle_tile_ny": parameter_set["Ny"],
            "particle_tile_nz": parameter_set["Nz"],
        }
    # create simulation parameters for a single tile that covers the entire parameter_set grid

    def _species_config(self, charges, masses, weights, update_x=None):
        n_species = len(charges)
        if update_x is None:
            update_x = [(True, True, True)] * n_species

        return SpeciesConfig(
            charge=jnp.asarray(charges, dtype=float),
            mass=jnp.asarray(masses, dtype=float),
            weight=jnp.asarray(weights, dtype=float),
            update_x=jnp.asarray(update_x, dtype=bool),
        )
    # create a SpeciesConfig object with the given charges, masses, weights, and directional update flags

    def _empty_tiled_particles(self, parameter_set, simulation_parameters, n_species, n_slots):
        tile_nx, tile_ny, tile_nz = self._tile_shape(simulation_parameters)
        ntx = _tile_axis_count(parameter_set["Nx"], tile_nx)
        nty = _tile_axis_count(parameter_set["Ny"], tile_ny)
        ntz = _tile_axis_count(parameter_set["Nz"], tile_nz)
        shape = (ntx, nty, ntz, n_species, n_slots, 3)

        return TiledParticles(
            x=jnp.zeros(shape),
            u=jnp.zeros(shape),
            active=jnp.zeros(shape[:-1], dtype=bool),
        )
    # create an empty TiledParticles object with the appropriate shape based on the parameter_set, simulation parameters, number of species, and number of slots

    def _set_tiled_particle(self, particles, tile, species, slot, x, u, active=True):
        tx, ty, tz = tile
        particles = particles._replace(
            x=particles.x.at[tx, ty, tz, species, slot].set(jnp.asarray(x, dtype=float)),
            u=particles.u.at[tx, ty, tz, species, slot].set(jnp.asarray(u, dtype=float)),
            active=particles.active.at[tx, ty, tz, species, slot].set(active),
        )
        return particles
    # set the position, velocity, and active status of a specific particle in the TiledParticles object based on the given tile, species, slot, position, velocity, and active flag

    def _particles_from_slots(self, parameter_set, simulation_parameters, n_species, n_slots, slots):
        particles = self._empty_tiled_particles(parameter_set, simulation_parameters, n_species, n_slots)
        for tile, species, slot, position, velocity, active in slots:
            particles = self._set_tiled_particle(
                particles,
                tile,
                species,
                slot,
                position,
                velocity,
                active,
            )
        return particles
    # create a TiledParticles object from a list of slots, where each slot specifies the tile, species, slot index, position, velocity, and active status of a particle

    def _one_tile_particles_from_tiled(self, particles):
        n_species = particles.active.shape[3]
        n_slots = (
            particles.active.shape[0]
            * particles.active.shape[1]
            * particles.active.shape[2]
            * particles.active.shape[4]
        )
        return TiledParticles(
            x=particles.x.transpose(3, 0, 1, 2, 4, 5).reshape(1, 1, 1, n_species, n_slots, 3),
            u=particles.u.transpose(3, 0, 1, 2, 4, 5).reshape(1, 1, 1, n_species, n_slots, 3),
            active=particles.active.transpose(3, 0, 1, 2, 4).reshape(1, 1, 1, n_species, n_slots),
        )
    # convert a TiledParticles object into a single-tile representation by transposing and reshaping the arrays to have a single tile dimension, while preserving the species and slot dimensions

    def _centered_tiled_particles(self, particles, parameter_set, simulation_parameters):
        """
        Build the tiled particle view expected by direct tiled deposition.

        ``J_from_rhov`` expects particles at the centered direct-current
        deposition position.  These fixtures start from the forward position,
        so the test view applies the half-step before deposition and refreshes
        tile ownership at the centered position.
        """

        particles = particles._replace(x=particles.x - 0.5 * particles.u * parameter_set["dt"])
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set)

        centered_particles, overflow = refresh_tiled_particle_tiles(
            particles,
            static_parameters,
            dynamic_parameters,
        )
        self.assertFalse(bool(overflow))
        # ensure that no particles have overflowed their tiles after centering

        return centered_particles

    def _assembled_tiled_current(self, particles, species_config, parameter_set, simulation_parameters, dynamic_values, filter="none"):
        tile_shape = self._tile_shape(simulation_parameters)
        parameter_set = self._parameters_with_tiled_grids(parameter_set, tile_shape)
        tiled_particles = self._centered_tiled_particles(particles, parameter_set, simulation_parameters)
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)
        static_parameters = static_parameters._replace(current_filter=filter)

        J_tiles = J_from_rhov(
            tiled_particles,
            species_config,
            self._empty_J_tiles(parameter_set),
            static_parameters,
            dynamic_parameters,
        )
        g = int(parameter_set["guard_cells"])
        J_from_tiles = assemble_tiled_vector_field(J_tiles, parameter_set, tile_shape, num_guard_cells=g)
        # assemble the tiled current density field into a global field for comparison

        return J_tiles, J_from_tiles

    def _compare_tiled_to_one_tile(self, particles, species_config, parameter_set, simulation_parameters, filter="none", alpha=1.0):
        dynamic_values = {"C": 3.0e8, "alpha": alpha}
        J_tiles, J_from_tiles = self._assembled_tiled_current(
            particles, species_config, parameter_set, simulation_parameters, dynamic_values, filter=filter
        )
        _, J_reference = self._assembled_tiled_current(
            self._one_tile_particles_from_tiled(particles),
            species_config,
            parameter_set,
            self._one_tile_parameters(parameter_set),
            dynamic_values,
            filter=filter,
        )

        for tile_component in J_tiles:
            self.assertEqual(tile_component.ndim, 6)
            # ensure that the tiled current density components have 6 dimensions (tile_x, tile_y, tile_z, tile_nx, tile_ny, tile_nz)
        for reference_component, tiled_component in zip(J_reference, J_from_tiles):
            error = jnp.max(jnp.abs(tiled_component - reference_component))
            self.assertTrue(jnp.allclose(tiled_component, reference_component, rtol=5.0e-15, atol=5.0e-15))
            # compare the assembled tiled current density components to the reference components from the single-tile deposition, ensuring they are close within a specified tolerance


    def test_tiled_direct_deposition_matches_quadratic_with_two_guard_cells(self):
        parameter_set = self._build_parameter_values(Nx=8, Ny=6, Nz=4)
        parameter_set["shape_factor"] = 2
        parameter_set["guard_cells"] = 2
        simulation_parameters = {
            "particle_tile_nx": 4,
            "particle_tile_ny": 3,
            "particle_tile_nz": 2,
        }

        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=3,
            slots=[
                ((0, 0, 0), 0, 0, (-1.55, -1.10, -0.70), (0.18, 0.03, -0.06), True),
                ((0, 0, 0), 0, 1, (-0.52, -0.55, -0.04), (-0.11, 0.17, 0.24), True),
                ((0, 0, 1), 0, 0, (-0.03, -0.03, 0.03), (0.07, -0.22, 0.11), True),
                ((1, 1, 1), 0, 0, (0.49, 0.02, 0.31), (-0.04, 0.19, -0.14), True),
                ((1, 1, 1), 0, 1, (0.55, 0.52, 0.49), (0.21, -0.08, 0.05), True),
                ((1, 1, 1), 0, 2, (1.45, 1.05, 0.72), (-0.16, 0.12, -0.19), True),
            ],
        )
        species_config = self._species_config(charges=[-1.0], masses=[1.0], weights=[0.5])

        self._compare_tiled_to_one_tile(particles, species_config, parameter_set, simulation_parameters)
        # ensure the direct deposition from tiled particles matches the deposition from a single tile representation, using quadratic shape factors and two guard cells

    def test_tiled_direct_deposition_matches_quadratic_saved_style_reduced_axes(self):
        parameter_set = self._build_parameter_values(Nx=20, Ny=1, Nz=1, dt=0.05)
        parameter_set["shape_factor"] = 2
        parameter_set["guard_cells"] = 2
        simulation_parameters = {
            "particle_tile_nx": 5,
            "particle_tile_ny": 1,
            "particle_tile_nz": 1,
        }

        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=3,
            slots=[
                ((0, 0, 0), 0, 0, (-1.95, 0.0, 0.0), (0.18, 0.03, -0.06), True),
                ((0, 0, 0), 0, 1, (-1.51, 0.0, 0.0), (-0.11, 0.17, 0.24), True),
                ((0, 0, 0), 0, 2, (-1.02, 0.0, 0.0), (0.07, -0.22, 0.11), True),
                ((1, 0, 0), 0, 0, (-0.48, 0.0, 0.0), (-0.04, 0.19, -0.14), True),
                ((2, 0, 0), 0, 0, (0.02, 0.0, 0.0), (0.21, -0.08, 0.05), True),
                ((2, 0, 0), 0, 1, (0.47, 0.0, 0.0), (-0.16, 0.12, -0.19), True),
                ((3, 0, 0), 0, 0, (1.04, 0.0, 0.0), (0.09, -0.15, 0.16), True),
                ((3, 0, 0), 0, 1, (1.88, 0.0, 0.0), (-0.13, 0.05, -0.07), True),
            ],
        )
        species_config = self._species_config(charges=[-1.0], masses=[1.0], weights=[0.5])

        self._compare_tiled_to_one_tile(particles, species_config, parameter_set, simulation_parameters)
        # test the direct deposition from tiled particles matches single tiled with reduced dimensions

    def test_tiled_direct_deposition_bilinear_matches_quadratic_reduced_axes(self):
        parameter_set = self._build_parameter_values(Nx=20, Ny=1, Nz=1, dt=0.05)
        parameter_set["shape_factor"] = 2
        parameter_set["guard_cells"] = 2
        simulation_parameters = {
            "particle_tile_nx": 5,
            "particle_tile_ny": 1,
            "particle_tile_nz": 1,
        }
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=3,
            slots=[
                ((0, 0, 0), 0, 0, (-1.95, 0.0, 0.0), (0.18, 0.03, -0.06), True),
                ((0, 0, 0), 0, 1, (-1.51, 0.0, 0.0), (-0.11, 0.17, 0.24), True),
                ((0, 0, 0), 0, 2, (-1.02, 0.0, 0.0), (0.07, -0.22, 0.11), True),
                ((1, 0, 0), 0, 0, (-0.48, 0.0, 0.0), (-0.04, 0.19, -0.14), True),
                ((2, 0, 0), 0, 0, (0.02, 0.0, 0.0), (0.21, -0.08, 0.05), True),
                ((2, 0, 0), 0, 1, (0.47, 0.0, 0.0), (-0.16, 0.12, -0.19), True),
                ((3, 0, 0), 0, 0, (1.04, 0.0, 0.0), (0.09, -0.15, 0.16), True),
                ((3, 0, 0), 0, 1, (1.88, 0.0, 0.0), (-0.13, 0.05, -0.07), True),
            ],
        )
        species_config = self._species_config(charges=[-1.0], masses=[1.0], weights=[0.5])

        self._compare_tiled_to_one_tile(particles, species_config, parameter_set, simulation_parameters, filter="bilinear")
        # test the bilinear filtered direct deposition from tiled particles matches single tiled with reduced dimensions

    def test_tiled_direct_deposition_returns_only_local_current_tiles(self):
        parameter_set = self._build_parameter_values()
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 3,
            "particle_tile_nz": 2,
        }
        dynamic_values = {"C": 3.0e8, "alpha": 1.0}
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=1,
            slots=[
                ((0, 0, 0), 0, 0, (-1.25, -1.0, -0.65), (0.2, 0.0, -0.05), True),
                ((1, 0, 0), 0, 0, (-0.25, -0.25, -0.15), (-0.1, 0.15, 0.25), True),
                ((2, 1, 1), 0, 0, (0.65, 0.35, 0.25), (0.05, -0.2, 0.1), True),
                ((3, 1, 1), 0, 0, (1.45, 1.05, 0.75), (0.3, 0.1, -0.15), True),
            ],
        )
        species_config = self._species_config(charges=[1.0], masses=[1.0], weights=[1.0])
        tile_shape = self._tile_shape(simulation_parameters)
        parameter_set = self._parameters_with_tiled_grids(parameter_set, tile_shape)
        tiled_particles = self._centered_tiled_particles(particles, parameter_set, simulation_parameters)
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)

        J_tiles = J_from_rhov(
            tiled_particles,
            species_config,
            self._empty_J_tiles(parameter_set),
            static_parameters,
            dynamic_parameters,
        )
        J_from_tiles = assemble_tiled_vector_field(J_tiles, parameter_set, tile_shape, num_guard_cells=int(parameter_set["guard_cells"]))

        _, J_reference = self._assembled_tiled_current(
            self._one_tile_particles_from_tiled(particles),
            species_config,
            parameter_set,
            self._one_tile_parameters(parameter_set),
            dynamic_values,
            filter="none",
        )

        for reference_component, tiled_component in zip(J_reference, J_from_tiles):
            self.assertTrue(jnp.allclose(tiled_component, reference_component, rtol=1.0e-15, atol=1.0e-15))
        # test that the direct deposition from tiled particles returns only the local current tiles, and that the assembled tiled current matches the reference current from a single-tile representation

    def test_tiled_direct_deposition_matches_J_from_rhov_for_dummy_species(self):
        parameter_set = self._build_parameter_values()
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 3,
            "particle_tile_nz": 2,
        }

        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=2,
            n_slots=1,
            slots=[
                ((0, 0, 0), 0, 0, (-1.25, -1.0, -0.65), (0.2, 0.0, -0.05), True),
                ((0, 1, 1), 1, 0, (-1.65, 1.15, 0.35), (-0.1, 0.3, 0.1), True),
                ((1, 0, 0), 0, 0, (-0.25, -0.25, -0.15), (-0.1, 0.15, 0.25), True),
                ((2, 0, 0), 1, 0, (0.15, -0.75, -0.45), (0.2, -0.05, 0.05), True),
                ((2, 1, 1), 0, 0, (0.65, 0.35, 0.25), (0.05, -0.2, 0.1), True),
                ((3, 1, 1), 1, 0, (1.75, 0.45, 0.85), (-0.25, 0.15, -0.2), True),
                ((3, 1, 1), 0, 0, (1.45, 1.05, 0.75), (0.3, 0.1, -0.15), True),
            ],
        )
        species_config = self._species_config(
            charges=[-1.0, 2.0],
            masses=[1.0, 4.0],
            weights=[0.5, 0.25],
        )

        self._compare_tiled_to_one_tile(particles, species_config, parameter_set, simulation_parameters)
        # test that the direct deposition from tiled particles with multiple species (including a dummy species) matches the deposition from a single-tile representation, ensuring consistency across species

    def test_tiled_direct_deposition_masks_current_per_species_direction(self):
        parameter_set = self._build_parameter_values(Nx=4, Ny=4, Nz=4)
        parameter_set["guard_cells"] = 2
        simulation_parameters = {
            "particle_tile_nx": 4,
            "particle_tile_ny": 4,
            "particle_tile_nz": 4,
        }
        parameter_set = self._parameters_with_tiled_grids(
            parameter_set,
            self._tile_shape(simulation_parameters),
        )
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=2,
            n_slots=1,
            slots=[
                ((0, 0, 0), 0, 0, (-0.75, -0.25, 0.25), (0.2, 0.3, 0.4), True),
                ((0, 0, 0), 1, 0, (0.75, 0.25, -0.25), (-0.5, -0.6, -0.7), True),
            ],
        )
        species_config = self._species_config(
            charges=[1.0, 2.0],
            masses=[1.0, 1.0],
            weights=[1.0, 1.0],
            update_x=[
                (False, True, False),
                (True, False, True),
            ],
        )
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set)
        static_parameters = static_parameters._replace(current_filter="none")

        masked_current = J_from_rhov(
            particles,
            species_config,
            self._empty_J_tiles(parameter_set),
            static_parameters,
            dynamic_parameters,
        )

        slot_mask = species_config.update_x.reshape((1, 1, 1, 2, 1, 3))
        reference_particles = particles._replace(u=jnp.where(slot_mask, particles.u, 0.0))
        reference_config = species_config._replace(update_x=jnp.ones_like(species_config.update_x))
        reference_current = J_from_rhov(
            reference_particles,
            reference_config,
            self._empty_J_tiles(parameter_set),
            static_parameters,
            dynamic_parameters,
        )

        for masked_component, reference_component in zip(masked_current, reference_current):
            self.assertTrue(jnp.allclose(masked_component, reference_component))
            self.assertGreater(float(jnp.max(jnp.abs(masked_component))), 0.0)

        disabled_config = species_config._replace(update_x=jnp.zeros_like(species_config.update_x))
        disabled_current = J_from_rhov(
            particles,
            disabled_config,
            self._empty_J_tiles(parameter_set),
            static_parameters,
            dynamic_parameters,
        )
        for component in disabled_current:
            self.assertTrue(jnp.allclose(component, 0.0))

    def test_public_J_from_rhov_dispatches_tiled_particles_to_tile_local_current(self):
        parameter_set = self._build_parameter_values()
        parameter_set["guard_cells"] = 2
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 3,
            "particle_tile_nz": 2,
        }
        tile_shape = self._tile_shape(simulation_parameters)
        parameter_set = self._parameters_with_tiled_grids(parameter_set, tile_shape)
        dynamic_values = {"C": 3.0e8, "alpha": 0.6}
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=1,
            slots=[
                ((0, 0, 0), 0, 0, (-1.25, -1.0, -0.65), (0.2, 0.0, -0.05), True),
                ((1, 0, 0), 0, 0, (-0.25, -0.25, -0.15), (-0.1, 0.15, 0.25), True),
                ((2, 1, 1), 0, 0, (0.65, 0.35, 0.25), (0.05, -0.2, 0.1), True),
                ((3, 1, 1), 0, 0, (1.45, 1.05, 0.75), (0.3, 0.1, -0.15), True),
            ],
        )
        species_config = self._species_config(charges=[-1.0], masses=[1.0], weights=[0.5])
        tiled_particles = self._centered_tiled_particles(particles, parameter_set, simulation_parameters)
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)
        static_parameters = static_parameters._replace(current_filter="digital")

        J_tiles = J_from_rhov(
            tiled_particles,
            species_config,
            self._empty_J_tiles(parameter_set),
            static_parameters,
            dynamic_parameters,
        )
        J_from_tiles = assemble_tiled_vector_field(
            J_tiles,
            parameter_set,
            tile_shape,
            num_guard_cells=int(parameter_set["guard_cells"]),
        )
        _, J_reference = self._assembled_tiled_current(
            self._one_tile_particles_from_tiled(particles),
            species_config,
            parameter_set,
            self._one_tile_parameters(parameter_set),
            dynamic_values,
            filter="digital",
        )

        for tile_component in J_tiles:
            self.assertEqual(tile_component.ndim, 6)
        for reference_component, tiled_component in zip(J_reference, J_from_tiles):
            self.assertTrue(jnp.allclose(tiled_component, reference_component, rtol=1.0e-15, atol=1.0e-15))
        # test that the public J_from_rhov function correctly dispatches tiled particles to the tile-local current deposition, and that the assembled tiled current matches the reference current from a single-tile representation

    def test_tiled_direct_deposition_digital_filter_matches_J_from_rhov(self):
        parameter_set = self._build_parameter_values()
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 3,
            "particle_tile_nz": 2,
        }
        dynamic_values = {"C": 3.0e8, "alpha": 0.6}
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=1,
            slots=[
                ((0, 0, 0), 0, 0, (-1.25, -1.0, -0.65), (0.2, 0.0, -0.05), True),
                ((1, 0, 0), 0, 0, (-0.25, -0.25, -0.15), (-0.1, 0.15, 0.25), True),
                ((2, 1, 1), 0, 0, (0.65, 0.35, 0.25), (0.05, -0.2, 0.1), True),
                ((3, 1, 1), 0, 0, (1.45, 1.05, 0.75), (0.3, 0.1, -0.15), True),
            ],
        )
        species_config = self._species_config(charges=[-1.0], masses=[1.0], weights=[0.5])

        self._compare_tiled_to_one_tile(particles, species_config, parameter_set, simulation_parameters, filter="digital", alpha=0.6)
        # test that the direct deposition from tiled particles with a digital filter matches the deposition from a single-tile representation, ensuring consistency between tiled and global deposition with filtering

    def test_tiled_direct_deposition_none_filter_does_not_use_alpha(self):
        parameter_set = self._build_parameter_values()
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 3,
            "particle_tile_nz": 2,
        }
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=1,
            slots=[
                ((0, 0, 0), 0, 0, (-1.25, -1.0, -0.65), (0.2, 0.1, -0.05), True),
                ((1, 0, 0), 0, 0, (-0.25, -0.25, -0.15), (-0.1, 0.15, 0.25), True),
                ((2, 1, 1), 0, 0, (0.65, 0.35, 0.25), (0.05, -0.2, 0.1), True),
            ],
        )
        species_config = self._species_config(charges=[-1.0], masses=[1.0], weights=[0.5])

        _, raw_alpha_06 = self._assembled_tiled_current(
            particles,
            species_config,
            parameter_set,
            simulation_parameters,
            {"C": 3.0e8, "alpha": 0.6},
            filter="none",
        )
        _, raw_alpha_10 = self._assembled_tiled_current(
            particles,
            species_config,
            parameter_set,
            simulation_parameters,
            {"C": 3.0e8, "alpha": 1.0},
            filter="none",
        )
        _, digital_alpha_06 = self._assembled_tiled_current(
            particles,
            species_config,
            parameter_set,
            simulation_parameters,
            {"C": 3.0e8, "alpha": 0.6},
            filter="digital",
        )

        for raw_06_component, raw_10_component in zip(raw_alpha_06, raw_alpha_10):
            self.assertTrue(jnp.allclose(raw_06_component, raw_10_component, rtol=1.0e-15, atol=1.0e-15))

        digital_difference = max(
            float(jnp.max(jnp.abs(raw_component - digital_component)))
            for raw_component, digital_component in zip(raw_alpha_06, digital_alpha_06)
        )
        self.assertGreater(digital_difference, 1.0e-12)

    def test_tiled_direct_deposition_bilinear_filter_matches_J_from_rhov(self):
        parameter_set = self._build_parameter_values(Nx=8, Ny=6, Nz=4)
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 3,
            "particle_tile_nz": 2,
        }
        dynamic_values = {"C": 3.0e8, "alpha": 1.0}
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=2,
            slots=[
                ((0, 0, 0), 0, 0, (-1.55, -1.10, -0.70), (0.18, 0.03, -0.06), True),
                ((1, 0, 0), 0, 0, (-0.52, -0.55, -0.04), (-0.11, 0.17, 0.24), True),
                ((1, 0, 1), 0, 0, (-0.03, -0.03, 0.03), (0.07, -0.22, 0.11), True),
                ((2, 1, 1), 0, 0, (0.49, 0.02, 0.31), (-0.04, 0.19, -0.14), True),
                ((2, 1, 1), 0, 1, (0.55, 0.52, 0.49), (0.21, -0.08, 0.05), True),
                ((3, 1, 1), 0, 0, (1.45, 1.05, 0.72), (-0.16, 0.12, -0.19), True),
            ],
        )
        species_config = self._species_config(charges=[-1.0], masses=[1.0], weights=[0.5])

        self._compare_tiled_to_one_tile(particles, species_config, parameter_set, simulation_parameters, filter="bilinear")
        # test that the direct deposition from tiled particles with a bilinear filter matches the deposition from a single-tile representation, ensuring consistency between tiled and global deposition with bilinear filtering

    def test_tiled_direct_deposition_respects_active_mask(self):
        parameter_set = self._build_parameter_values()
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 3,
            "particle_tile_nz": 2,
        }
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=1,
            slots=[
                ((0, 0, 0), 0, 0, (-1.25, -1.0, -0.65), (0.2, 0.0, -0.05), True),
                ((1, 1, 0), 0, 0, (-0.25, -0.25, -0.15), (-0.1, 0.15, 0.25), False),
                ((2, 1, 1), 0, 0, (0.65, 0.35, 0.25), (0.05, -0.2, 0.1), True),
                ((3, 2, 1), 0, 0, (1.45, 1.05, 0.75), (0.3, 0.1, -0.15), False),
            ],
        )
        species_config = self._species_config(charges=[1.0], masses=[1.0], weights=[1.0])

        self._compare_tiled_to_one_tile(particles, species_config, parameter_set, simulation_parameters)
        # test that the direct deposition from tiled particles respects the active mask, ensuring that only active particles contribute to the current deposition, and that the assembled tiled current matches the reference current from a single-tile representation

    def test_tiled_direct_deposition_periodic_boundary_crossing(self):
        parameter_set = self._build_parameter_values(Nx=10, Ny=1, Nz=1, dt=0.0)
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 1,
            "particle_tile_nz": 1,
        }
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=1,
            slots=[
                ((0, 0, 0), 0, 0, (-parameter_set["x_wind"] / 2 - 0.2 * parameter_set["dx"], 0.0, 0.0), (-0.25, -0.2, 0.15), True),
                ((4, 0, 0), 0, 0, (parameter_set["x_wind"] / 2 + 0.1 * parameter_set["dx"], 0.0, 0.0), (0.5, 0.1, 0.0), True),
            ],
        )
        species_config = self._species_config(charges=[1.0], masses=[1.0], weights=[1.0])

        self._compare_tiled_to_one_tile(particles, species_config, parameter_set, simulation_parameters)

    def test_tiled_direct_deposition_matches_J_from_rhov_for_conducting_boundaries(self):
        parameter_set = self._build_parameter_values(
            Nx=8,
            Ny=6,
            Nz=4,
            dt=0.0,
            boundary_conditions={"x": BC_CONDUCTING, "y": BC_CONDUCTING, "z": BC_CONDUCTING},
        )
        parameter_set["particle_boundary_conditions"] = {
            "x": BC_CONDUCTING,
            "y": BC_CONDUCTING,
            "z": BC_CONDUCTING,
        }
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 3,
            "particle_tile_nz": 2,
        }
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=1,
            slots=[
                (
                    (0, 0, 1),
                    0,
                    0,
                    (-parameter_set["x_wind"] / 2 + 0.1 * parameter_set["dx"], -parameter_set["y_wind"] / 2 + 0.1 * parameter_set["dy"], 0.0),
                    (0.5, 0.1, -0.15),
                    True,
                ),
                (
                    (3, 1, 0),
                    0,
                    0,
                    (parameter_set["x_wind"] / 2 - 0.1 * parameter_set["dx"], 0.0, -parameter_set["z_wind"] / 2 + 0.1 * parameter_set["dz"]),
                    (-0.25, -0.2, 0.35),
                    True,
                ),
                (
                    (2, 1, 1),
                    0,
                    0,
                    (0.0, parameter_set["y_wind"] / 2 - 0.1 * parameter_set["dy"], parameter_set["z_wind"] / 2 - 0.1 * parameter_set["dz"]),
                    (0.15, 0.3, -0.1),
                    True,
                ),
            ],
        )
        species_config = self._species_config(charges=[1.0], masses=[1.0], weights=[1.0])

        self._compare_tiled_to_one_tile(particles, species_config, parameter_set, simulation_parameters)

    def test_tiled_direct_deposition_matches_J_from_rhov_for_mixed_boundaries(self):
        parameter_set = self._build_parameter_values(
            Nx=8,
            Ny=6,
            Nz=4,
            dt=0.0,
            boundary_conditions={"x": BC_PERIODIC, "y": BC_CONDUCTING, "z": BC_PERIODIC},
        )
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 3,
            "particle_tile_nz": 2,
        }
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=1,
            slots=[
                (
                    (0, 0, 1),
                    0,
                    0,
                    (-parameter_set["x_wind"] / 2 - 0.1 * parameter_set["dx"], -parameter_set["y_wind"] / 2 + 0.1 * parameter_set["dy"], 0.0),
                    (0.2, 0.0, -0.05),
                    True,
                ),
                ((1, 0, 0), 0, 0, (-0.5, -0.25, -parameter_set["z_wind"] / 2 - 0.1 * parameter_set["dz"]), (0.05, -0.2, 0.1), True),
                ((2, 1, 1), 0, 0, (0.5, 0.25, parameter_set["z_wind"] / 2 + 0.2 * parameter_set["dz"]), (0.3, 0.1, -0.15), True),
                (
                    (3, 1, 1),
                    0,
                    0,
                    (parameter_set["x_wind"] / 2 + 0.2 * parameter_set["dx"], parameter_set["y_wind"] / 2 - 0.2 * parameter_set["dy"], 0.25),
                    (-0.1, 0.15, 0.25),
                    True,
                ),
            ],
        )
        species_config = self._species_config(charges=[-1.0], masses=[1.0], weights=[0.5])

        self._compare_tiled_to_one_tile(particles, species_config, parameter_set, simulation_parameters)

    def test_tiled_direct_deposition_reduced_dimensions(self):
        parameter_set = self._build_parameter_values(Nx=16, Ny=1, Nz=1, dt=0.02)
        simulation_parameters = {
            "particle_tile_nx": 4,
            "particle_tile_ny": 1,
            "particle_tile_nz": 1,
        }
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=1,
            slots=[
                ((0, 0, 0), 0, 0, (-1.25, 0.0, 0.0), (0.2, 0.3, -0.05), True),
                ((2, 0, 0), 0, 0, (0.15, 0.0, 0.0), (-0.1, 0.15, 0.25), True),
                ((3, 0, 0), 0, 0, (1.25, 0.0, 0.0), (0.05, -0.2, 0.1), True),
            ],
        )
        species_config = self._species_config(charges=[1.0], masses=[1.0], weights=[1.0])

        self._compare_tiled_to_one_tile(particles, species_config, parameter_set, simulation_parameters)


if __name__ == "__main__":
    unittest.main()
