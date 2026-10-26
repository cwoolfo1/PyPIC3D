"""Discrete electrostatic energy, boundary, and startup contracts."""

import contextlib
import io
import os
import tempfile
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_PERIODIC
from PyPIC3D.initialization import initialize_simulation
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.pusher.boris import field_component_orders, interpolate_field_to_particles
from PyPIC3D.solvers.electrostatic.electrostatic_yee import (
    _apply_tiled_phi_constant_boundaries,
    _poisson_residual,
    _tiled_laplacian,
    calculate_electrostatic_fields,
    electrostatic_gradient,
)
from PyPIC3D.utilities.filters import digital_filter
from tests.kernel_fixtures import empty_tiled_scalar, field_tiles_from_global, kernel_parameters


class TestElectrostaticGalerkin(unittest.TestCase):
    def test_periodic_field_energy_is_the_discrete_poisson_energy(self):
        for shape in ((8, 1, 1), (8, 6, 1), (8, 6, 4)):
            for tile_nx in ((8, 4) if jax.device_count() >= 2 else (8,)):
                with self.subTest(shape=shape, tile_nx=tile_nx):
                    s, d = kernel_parameters(
                        Nx=shape[0], Ny=shape[1], Nz=shape[2],
                        tile_shape=(tile_nx, shape[1], shape[2]),
                        solver="electrostatic", electrostatic=True, eps=2.3,
                    )
                    i, j, k = jnp.meshgrid(*(jnp.arange(n) for n in shape), indexing="ij")
                    phi = (jnp.sin(2 * jnp.pi * i / shape[0]) + 0.2 * (-1.0)**i
                           + 0.3 * jnp.cos(2 * jnp.pi * j / shape[1])
                           + 0.4 * jnp.sin(2 * jnp.pi * k / shape[2]))
                    phi = field_tiles_from_global(phi, s, d)
                    g = s.guard_cells
                    gradient = jax.jit(lambda phi, d: electrostatic_gradient(phi, s, d, g))
                    E = gradient(phi, d)
                    rho = -d.eps * _tiled_laplacian(phi, d, g)
                    dV = d.dx * d.dy * d.dz
                    field_energy = d.eps * dV / 2 * sum(
                        jnp.sum(component[..., g:-g, g:-g, g:-g]**2) for component in E
                    )
                    poisson_energy = dV / 2 * jnp.sum(rho * phi[..., g:-g, g:-g, g:-g])
                    self.assertLess(float(jnp.abs(field_energy / poisson_energy - 1)), 2e-14)
                    for axis, n in enumerate(shape):
                        if n == 1:
                            self.assertTrue(jnp.all(E[axis] == 0))

    def test_uniform_retains_centered_gradient_and_nyquist_null(self):
        s, d = kernel_parameters(Nx=8, Ny=1, Nz=1, solver="electrostatic", electrostatic=True)
        phi = field_tiles_from_global((-1.0)**jnp.arange(8)[:, None, None], s, d)
        uniform = electrostatic_gradient(phi, s._replace(field_gather="uniform"), d, s.guard_cells)
        galerkin = electrostatic_gradient(phi, s, d, s.guard_cells)
        g = s.guard_cells
        self.assertTrue(jnp.all(uniform[0][..., g:-g, g:-g, g:-g] == 0))
        self.assertGreater(float(jnp.max(jnp.abs(galerkin[0]))), 0)

    def test_wall_and_tile_gather_is_derivative_of_interpolated_potential(self):
        # Include both sides of a tile seam and the physical upper/lower walls.
        for bc in (BC_PERIODIC, BC_CONDUCTING):
            for p in (1, 2):
                with self.subTest(boundary=bc, shape_factor=p):
                    tile_nx = 4 if jax.device_count() >= 2 else 8
                    s, d = kernel_parameters(
                        Nx=8, Ny=6, Nz=1, tile_shape=(tile_nx, 6, 1), shape_factor=p,
                        solver="electrostatic", electrostatic=True,
                        boundary_conditions=(bc, BC_PERIODIC, BC_PERIODIC),
                    )
                    i, j, _ = jnp.meshgrid(jnp.arange(8), jnp.arange(6), jnp.arange(1), indexing="ij")
                    phi = field_tiles_from_global(i**2 / 10 + jnp.sin(2 * jnp.pi * j / 6), s, d)
                    phi = _apply_tiled_phi_constant_boundaries(phi, s, s.guard_cells)
                    E = electrostatic_gradient(phi, s, d, s.guard_cells)
                    electric_orders, _ = field_component_orders(p, "galerkin")
                    for tx in range(8 // tile_nx):
                        center = tuple(axis[tx, 0, 0] for axis in d.grids.tiled_center_grid)
                        vertex = tuple(axis[tx, 0, 0] for axis in d.grids.tiled_vertex_grid)
                        g = s.guard_cells
                        x = jnp.array([center[0][g], center[0][g] + .07 * d.dx,
                                       center[0][g + tile_nx] - .09 * d.dx,
                                       center[0][g + tile_nx]])
                        y = jnp.full_like(x, center[1][g] + 2.31 * d.dy)
                        z = jnp.zeros_like(x)

                        def potential_at_particle(position):
                            return interpolate_field_to_particles(
                                phi[tx, 0, 0], *(position[axis:axis + 1] for axis in range(3)),
                                center, p, ghost_cells=True,
                                active_axes=(True, True, False), inactive_axis_indices=(g, g, g),
                            )[0]

                        grad_phi = jax.vmap(jax.grad(potential_at_particle))(jnp.stack((x, y, z), axis=1))
                        for axis in (0, 1):
                            component_grid = tuple(vertex[k] if k == axis else center[k] for k in range(3))
                            gathered = interpolate_field_to_particles(
                                E[axis][tx, 0, 0], x, y, z, component_grid, p,
                                ghost_cells=True, active_axes=(True, True, False),
                                inactive_axis_indices=(g, g, g), axis_orders=electric_orders[axis],
                            )
                            self.assertTrue(jnp.allclose(gathered, -grad_phi[:, axis], rtol=2e-13, atol=2e-13))
                    if bc == BC_CONDUCTING:
                        # Neumann phi extension leaves tangential E at the wall;
                        # imposing Maxwell PEC would incorrectly erase it.
                        self.assertGreater(float(jnp.max(jnp.abs(E[1][0, 0, 0, 0, :, g]))), 0)
                        self.assertTrue(jnp.all(E[0][0, 0, 0, :g] == 0))
                        self.assertTrue(jnp.all(E[0][-1, 0, 0, -g - 1:] == 0))

    def test_initial_self_solve_precedes_seed_and_preserves_external_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = {}
            for name, values in {
                "zero": jnp.zeros(2), "electron_x": jnp.array([-.37, .13]),
                "ion_x": jnp.array([-.12, .38]),
                "self_Ex": jnp.full((8, 1, 1), 7.0),
                "external_Ex": (-.5 + (jnp.arange(8) + .5) / 8)[:, None, None],
                "external_Ey": jnp.full((8, 1, 1), .25),
            }.items():
                paths[name] = os.path.join(tmpdir, name + ".npy")
                jnp.save(paths[name], values)
            config = {
                "simulation_parameters": {
                    "name": "ES initial self field", "output_dir": tmpdir,
                    "solver": "electrostatic", "Nx": 8, "Ny": 1, "Nz": 1,
                    "x_wind": 1.0, "y_wind": 1.0, "z_wind": 1.0,
                    "Nt": 1, "dt": .01, "C": 20.0, "eps": 1.0, "mu": 1.0,
                    "kb": 1.0, "relativistic": False, "filter_j": "none",
                    "electrostatic_schwarz_tol": 1e-11, "electrostatic_local_cg_tol": 1e-11,
                },
                "field1": {"name": "self Ex", "type": 0, "path": paths["self_Ex"]},
                "field2": {"name": "external Ey", "type": 1, "path": paths["external_Ey"], "evolve": False},
                "field3": {"name": "external Ex", "type": 0, "path": paths["external_Ex"], "evolve": False},
            }
            for number, (name, charge) in enumerate((("electron", -1.0), ("ion", 1.0)), 1):
                config[f"particle{number}"] = {
                    "name": name, "N_particles": 2, "charge": charge, "mass": 1.0,
                    "temperature": 0.0, "weight": .02, "initial_x": paths[name + "_x"],
                    **{key: paths["zero"] for key in ("initial_y", "initial_z", "initial_vx", "initial_vy", "initial_vz")},
                }
            for mode in ("auto", "uniform"):
                with self.subTest(field_gather=mode):
                    config["simulation_parameters"]["field_gather"] = mode
                    with patch("PyPIC3D.initialization.plot_initial_histograms"), contextlib.redirect_stdout(io.StringIO()):
                        _, particles, fields, s, d, _, _, species = initialize_simulation(config)
                    E, _, _, rho, phi, external, _, _ = fields
                    self.assertTrue(jnp.all(external[0][1] == .25))
                    if mode == "auto":
                        self.assertEqual(s.field_gather, "galerkin")
                        self.assertGreater(float(jnp.max(jnp.abs(rho))), 0)
                        self.assertGreater(float(jnp.max(jnp.abs(phi))), 0)
                        self.assertLess(float(jnp.max(jnp.abs(E[0]))), 1)
                        expected_offset = d.dx / 2
                    else:
                        self.assertEqual(s.field_gather, "uniform")
                        self.assertTrue(jnp.all(rho == 0))
                        self.assertTrue(jnp.all(phi == 0))
                        self.assertTrue(jnp.all(E[0] == 7))
                        expected_offset = 0
                    self.assertTrue(jnp.allclose(d.grids.vertex[0] - d.grids.center[0], expected_offset))

                    x = particles.x[0, 0, 0].reshape(-1, 3)
                    center = tuple(axis[0, 0, 0] for axis in d.grids.tiled_center_grid)
                    vertex = tuple(axis[0, 0, 0] for axis in d.grids.tiled_vertex_grid)
                    orders, _ = field_component_orders(s.shape_factor, s.field_gather)
                    external_Ex = interpolate_field_to_particles(
                        external[0][0][0, 0, 0], x[:, 0], x[:, 1], x[:, 2],
                        (vertex[0], center[1], center[2]), s.shape_factor,
                        ghost_cells=True, active_axes=(True, False, False),
                        inactive_axis_indices=(s.guard_cells,) * 3, axis_orders=orders[0],
                    )
                    if mode == "auto":
                        face_index = jnp.floor((x[:, 0] + .5) / d.dx)
                        expected_external = -.5 + (face_index + .5) * d.dx
                    else:
                        r = (x[:, 0] + .5) / d.dx
                        index = jnp.floor(r)
                        fraction = r - index
                        left = -.5 + (jnp.mod(index, 8) + .5) * d.dx
                        right = -.5 + (jnp.mod(index + 1, 8) + .5) * d.dx
                        expected_external = (1 - fraction) * left + fraction * right
                    self.assertTrue(jnp.allclose(external_Ex, expected_external, rtol=2e-13, atol=2e-13))
                    force = []
                    for axis in range(3):
                        grid = tuple(vertex[k] if k == axis else center[k] for k in range(3))
                        force.append(interpolate_field_to_particles(
                            E[axis][0, 0, 0] + external[0][axis][0, 0, 0],
                            x[:, 0], x[:, 1], x[:, 2], grid, s.shape_factor,
                            ghost_cells=True, active_axes=(True, False, False),
                            inactive_axis_indices=(s.guard_cells,) * 3, axis_orders=orders[axis],
                        ))
                    force = jnp.stack(force, axis=-1).reshape(particles.u.shape)
                    q_over_m = (species.charge / species.mass)[None, None, None, :, None, None]
                    expected_velocity = -d.dt / 2 * q_over_m * force
                    self.assertTrue(jnp.allclose(particles.u, expected_velocity, rtol=2e-13, atol=2e-13))

    def test_filter_and_conducting_solve_keep_the_potential_extension(self):
        for bc in (BC_PERIODIC, BC_CONDUCTING):
            with self.subTest(boundary=bc):
                s, d = kernel_parameters(
                    Nx=8, Ny=1, Nz=1, solver="electrostatic", electrostatic=True,
                    boundary_conditions=(bc, BC_PERIODIC, BC_PERIODIC),
                    electrostatic_schwarz_tol=1e-11, electrostatic_local_cg_tol=1e-11,
                )
                x = jnp.zeros((1, 1, 1, 2, 2, 3))
                x = x.at[0, 0, 0, :, :, 0].set(jnp.array([[-1.35, .63], [-.23, 1.27]]))
                particles = TiledParticles(x=x, u=jnp.zeros_like(x), active=jnp.ones(x.shape[:-1], dtype=bool))
                species = SpeciesConfig(charge=jnp.array([-1., 1.]), mass=jnp.ones(2),
                                        weight=jnp.full(2, .02), update_x=jnp.ones(2, dtype=bool))
                zero = empty_tiled_scalar(s, d)
                E, phi, rho = calculate_electrostatic_fields(s, d, particles, species, zero, zero)
                if bc == BC_PERIODIC:
                    g = s.guard_cells
                    owned = (..., slice(g, -g), slice(g, -g), slice(g, -g))
                    dV = d.dx * d.dy * d.dz
                    residual = jnp.max(jnp.abs(_poisson_residual(rho, phi, d, g)))
                    self.assertLess(float(residual), 2e-11)
                    field_energy = d.eps * dV / 2 * sum(jnp.sum(component[owned]**2) for component in E)
                    poisson_energy = dV / 2 * jnp.sum(rho[owned] * phi[owned])
                    tolerance = (d.eps * dV / 2 * jnp.sum(jnp.abs(phi[owned])) * residual
                                 + 2e-13 * jnp.abs(poisson_energy))
                    self.assertLessEqual(float(jnp.abs(field_energy - poisson_energy)), float(tolerance))
                filtered_d = d._replace(alpha=jnp.asarray(.5))
                E_filtered, phi_filtered, rho_filtered = calculate_electrostatic_fields(
                    s, filtered_d, particles, species, zero, zero,
                )
                expected_phi = digital_filter(phi, filtered_d.alpha, num_guard_cells=s.guard_cells)
                expected_phi = _apply_tiled_phi_constant_boundaries(expected_phi, s, s.guard_cells)
                self.assertTrue(jnp.allclose(phi_filtered, expected_phi, rtol=2e-12, atol=2e-12))
                self.assertTrue(jnp.array_equal(rho_filtered, rho))
                self.assertGreater(float(jnp.max(jnp.abs(phi_filtered - phi))), 1e-6)
                expected_E = electrostatic_gradient(expected_phi, s, d, s.guard_cells)
                for component, expected in zip(E_filtered, expected_E):
                    self.assertTrue(jnp.allclose(component, expected, rtol=2e-12, atol=2e-12))


if __name__ == "__main__":
    unittest.main()
