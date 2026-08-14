import inspect
import unittest
import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import toml

from PyPIC3D.solvers.electrostatic.time_loop import time_loop_electrostatic
from PyPIC3D.initialization import initialize_simulation
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.solvers.electrostatic import electrostatic_yee
from PyPIC3D.solvers.electrostatic.electrostatic_yee import (
    _apply_tiled_phi_constant_boundaries,
    _poisson_residual,
    calculate_electrostatic_fields,
    solve_poisson_with_tiled_local_schwarz,
)
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_PERIODIC
from tests.kernel_fixtures import empty_tiled_scalar, kernel_parameters

jax.config.update("jax_enable_x64", True)


def apply_negative_laplacian(field, dx, dy, dz):
    laplacian_x = (jnp.roll(field, shift=1, axis=0) + jnp.roll(field, shift=-1, axis=0) - 2.0 * field) / (dx * dx)
    laplacian_y = (jnp.roll(field, shift=1, axis=1) + jnp.roll(field, shift=-1, axis=1) - 2.0 * field) / (dy * dy)
    laplacian_z = (jnp.roll(field, shift=1, axis=2) + jnp.roll(field, shift=-1, axis=2) - 2.0 * field) / (dz * dz)
    return -(laplacian_x + laplacian_y + laplacian_z)


class TestElectrostaticYeeMethods(unittest.TestCase):
    def setUp(self):
        self.Nx = 16
        self.Ny = 16
        self.Nz = 16
        self.x_wind = 2 * jnp.pi
        self.y_wind = 2 * jnp.pi
        self.z_wind = 2 * jnp.pi
        self.dx = self.x_wind / self.Nx
        self.dy = self.y_wind / self.Ny
        self.dz = self.z_wind / self.Nz

        # Interior coordinate grid (no ghost cells) for analytical solutions
        x = jnp.linspace(0, self.x_wind, self.Nx, endpoint=False)
        y = jnp.linspace(0, self.y_wind, self.Ny, endpoint=False)
        z = jnp.linspace(0, self.z_wind, self.Nz, endpoint=False)
        self.X, self.Y, self.Z = jnp.meshgrid(x, y, z, indexing='ij')

        self.static_parameters, self.dynamic_parameters = kernel_parameters(
            Nx=self.Nx,
            Ny=self.Ny,
            Nz=self.Nz,
            x_wind=self.x_wind,
            y_wind=self.y_wind,
            z_wind=self.z_wind,
            dx=self.dx,
            dy=self.dy,
            dz=self.dz,
            tile_shape=(self.Nx, self.Ny, self.Nz),
            guard_cells=2,
            shape_factor=1,
            boundary_conditions=(BC_PERIODIC, BC_PERIODIC, BC_PERIODIC),
            eps=1.0,
            alpha=1.0,
            electrostatic=True,
            solver="electrostatic",
        )

        self.g = int(self.static_parameters.guard_cells)
        self.active = slice(self.g, -self.g)
        tile_field_shape = (
            1,
            1,
            1,
            self.Nx + 2 * self.g,
            self.Ny + 2 * self.g,
            self.Nz + 2 * self.g,
        )
        self.initial_rho = jnp.zeros(tile_field_shape)
        self.initial_phi = jnp.zeros(tile_field_shape)

    def test_tiled_schwarz_poisson_single_mode(self):
        phi_true_interior = jnp.sin(self.X + self.Y + self.Z)
        rhs = apply_negative_laplacian(phi_true_interior, self.dx, self.dy, self.dz)
        rho_interior = rhs * self.dynamic_parameters.eps

        rho_tiles = self.initial_rho.at[
            0,
            0,
            0,
            self.active,
            self.active,
            self.active,
        ].set(rho_interior)

        phi_tiles = solve_poisson_with_tiled_local_schwarz(
            rho_tiles,
            self.initial_phi,
            self.static_parameters,
            self.dynamic_parameters,
            schwarz_tol=1.0e-10,
            schwarz_max_iterations=1000,
            local_cg_tol=1.0e-10,
            local_cg_max_iterations=1000,
        )

        phi_num = phi_tiles[0, 0, 0, self.active, self.active, self.active]
        phi_num = phi_num - jnp.mean(phi_num)
        phi_true = phi_true_interior - jnp.mean(phi_true_interior)

        self.assertEqual(phi_num.shape, phi_true.shape)
        self.assertTrue(jnp.allclose(phi_num, phi_true, atol=1e-7, rtol=1e-6))

        residual = _poisson_residual(
            rho_tiles,
            phi_tiles,
            self.dynamic_parameters,
            self.g,
        )
        self.assertLess(float(jnp.max(jnp.abs(residual))), 1.0e-9)

    def test_phi_refresh_uses_shared_constant_boundary_for_conducting_axis(self):
        static_parameters, dynamic_parameters = kernel_parameters(
            Nx=8,
            Ny=4,
            Nz=4,
            x_wind=1.0,
            y_wind=1.0,
            z_wind=1.0,
            tile_shape=(8, 4, 4),
            guard_cells=2,
            shape_factor=1,
            boundary_conditions=(BC_CONDUCTING, BC_PERIODIC, BC_PERIODIC),
            electrostatic=True,
            solver="electrostatic",
        )
        g = int(static_parameters.guard_cells)
        phi = jnp.zeros((1, 1, 1, 8 + 2 * g, 4 + 2 * g, 4 + 2 * g))
        interior = jnp.arange(8 * 4 * 4, dtype=float).reshape((8, 4, 4))
        phi = phi.at[0, 0, 0, g:-g, g:-g, g:-g].set(interior)

        refreshed_phi = _apply_tiled_phi_constant_boundaries(
            phi,
            static_parameters,
            g,
        )[0, 0, 0]

        self.assertTrue(jnp.allclose(refreshed_phi[:g, :, :], refreshed_phi[g:g + 1, :, :]))
        self.assertTrue(jnp.allclose(refreshed_phi[-g:, :, :], refreshed_phi[-g - 1:-g, :, :]))
        self.assertTrue(jnp.allclose(refreshed_phi[g:-g, g:-g, g:-g], interior))

    def test_tiled_electrostatic_single_active_particle_uses_constant_phi_boundary(self):
        static_parameters, dynamic_parameters = kernel_parameters(
            Nx=8,
            Ny=4,
            Nz=4,
            x_wind=1.0,
            y_wind=1.0,
            z_wind=1.0,
            tile_shape=(8, 4, 4),
            guard_cells=2,
            shape_factor=1,
            boundary_conditions=(BC_CONDUCTING, BC_PERIODIC, BC_PERIODIC),
            electrostatic=True,
            solver="electrostatic",
        )
        g = int(static_parameters.guard_cells)
        active = slice(g, -g)
        particles = TiledParticles(
            x=jnp.asarray([[[[[[0.0, 0.0, 0.0]]]]]], dtype=float),
            u=jnp.zeros((1, 1, 1, 1, 1, 3), dtype=float),
            active=jnp.asarray([[[[[True]]]]]),
        )
        species_config = SpeciesConfig(
            charge=jnp.asarray([0.0], dtype=float),
            mass=jnp.asarray([1.0], dtype=float),
            weight=jnp.asarray([1.0], dtype=float),
            update_x=jnp.asarray([[True, True, True]]),
        )
        rho_tiles = empty_tiled_scalar(static_parameters, dynamic_parameters)
        phi_tiles = empty_tiled_scalar(static_parameters, dynamic_parameters)
        phi_tiles = phi_tiles.at[0, 0, 0, active, active, active].set(3.0)

        E_tiles, phi_tiles, rho_tiles = calculate_electrostatic_fields(
            static_parameters,
            dynamic_parameters,
            particles,
            species_config,
            rho_tiles,
            phi_tiles,
        )
        phi = phi_tiles[0, 0, 0]

        self.assertEqual(int(jnp.sum(particles.active)), 1)
        self.assertTrue(jnp.allclose(rho_tiles, 0.0, rtol=1.0e-12, atol=1.0e-12))
        self.assertTrue(jnp.allclose(phi[:g, :, :], phi[g:g + 1, :, :], rtol=1.0e-12, atol=1.0e-12))
        self.assertTrue(jnp.allclose(phi[-g:, :, :], phi[-g - 1:-g, :, :], rtol=1.0e-12, atol=1.0e-12))
        self.assertTrue(jnp.allclose(phi[active, active, active], 3.0, rtol=1.0e-12, atol=1.0e-12))
        for component in E_tiles:
            self.assertTrue(jnp.allclose(component, 0.0, rtol=1.0e-12, atol=1.0e-12))

    def test_untiled_electrostatic_api_is_removed(self):
        solver_names = {
            name
            for name in vars(electrostatic_yee)
            if name.startswith("solve_poisson")
        }
        field_pipeline_names = {
            name
            for name in vars(electrostatic_yee)
            if name.startswith("calculate") and name.endswith("electrostatic_fields")
        }

        self.assertEqual(solver_names, {"solve_poisson_with_tiled_local_schwarz"})
        self.assertEqual(field_pipeline_names, {"calculate_electrostatic_fields"})
        self.assertNotIn("single_tile", inspect.getsource(electrostatic_yee))

    def test_initialize_simulation_preserves_requested_electrostatic_tiles(self):
        if jax.device_count() < 4:
            self.skipTest(f"Need 4 JAX devices, got {jax.device_count()}")

        with tempfile.TemporaryDirectory() as tmpdir:
            x_path = os.path.join(tmpdir, "x.npy")
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            vx_path = os.path.join(tmpdir, "vx.npy")
            np.save(x_path, np.array([-1.5, -0.5, 0.5, 1.5]))
            np.save(zeros_path, np.zeros(4))
            np.save(vx_path, np.array([0.10, -0.05, 0.07, -0.02]))

            config = {
                "simulation_parameters": {
                    "name": "electrostatic init smoke",
                    "output_dir": tmpdir,
                    "solver": "electrostatic",
                    "Nx": 8,
                    "Ny": 1,
                    "Nz": 1,
                    "x_wind": 4.0,
                    "y_wind": 1.0,
                    "z_wind": 1.0,
                    "dt": 0.01,
                    "Nt": 1,
                    "shape_factor": 1,
                    "guard_cells": 1,
                    "particle_tile_nx": 2,
                    "particle_tile_ny": 1,
                    "particle_tile_nz": 1,
                    "current_calculation": "j_from_rhov",
                    "filter_j": "none",
                    "particle_pusher": "boris",
                    "relativistic": False,
                },
                "plotting": {"plotting_interval": 1},
                "particle1": {
                    "name": "electrons",
                    "N_particles": 4,
                    "charge": -1.0,
                    "mass": 2.0,
                    "weight": 0.5,
                    "temperature": 1.0,
                    "initial_x": x_path,
                    "initial_y": zeros_path,
                    "initial_z": zeros_path,
                    "initial_vx": vx_path,
                    "initial_vy": zeros_path,
                    "initial_vz": zeros_path,
                },
            }

            (
                loop,
                particles,
                fields,
                static_parameters,
                dynamic_parameters,
                _,
                _,
                species_config,
            ) = initialize_simulation(toml.loads(toml.dumps(config)))

            self.assertIs(loop, time_loop_electrostatic)
            self.assertIsInstance(particles, TiledParticles)
            self.assertEqual(tuple(static_parameters.tile_shape), (2, 1, 1))
            self.assertEqual(static_parameters.guard_cells, 1)
            for vertex_axis, center_axis in zip(dynamic_parameters.grids.vertex, dynamic_parameters.grids.center):
                self.assertTrue(jnp.allclose(vertex_axis, center_axis))
            for tiled_vertex_axis, tiled_center_axis in zip(
                dynamic_parameters.grids.tiled_vertex_grid,
                dynamic_parameters.grids.tiled_center_grid,
            ):
                self.assertTrue(jnp.allclose(tiled_vertex_axis, tiled_center_axis))
            self.assertEqual(fields[0][0].shape[:3], (4, 1, 1))
            self.assertEqual(fields[3].shape[:3], (4, 1, 1))
            self.assertEqual(fields[4].shape[:3], (4, 1, 1))

            zero_charge_config = species_config._replace(
                charge=jnp.zeros_like(species_config.charge)
            )
            jitted_step = jax.jit(
                lambda particle_state, field_state: loop(
                    particle_state,
                    zero_charge_config,
                    field_state,
                    static_parameters,
                    dynamic_parameters,
                )
            )
            _, stepped_fields = jitted_step(particles, fields)
            self.assertTrue(jnp.all(jnp.isfinite(stepped_fields[4])))


if __name__ == "__main__":
    unittest.main()
