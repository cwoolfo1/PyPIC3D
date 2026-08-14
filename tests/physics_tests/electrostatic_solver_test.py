import inspect
import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.ghost_cells import update_tiled_ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_PERIODIC
from PyPIC3D.solvers.electrostatic.electrostatic_yee import (
    _apply_tiled_phi_constant_boundaries,
    _local_tile_cg_solve,
    _poisson_residual,
    _tiled_laplacian,
    calculate_electrostatic_fields,
    solve_poisson_with_tiled_local_schwarz,
)
from tests.kernel_fixtures import kernel_parameters


jax.config.update("jax_enable_x64", True)


def _tile_field(interior, tile_grid_shape, tile_shape, g):
    ntx, nty, ntz = tile_grid_shape
    tile_nx, tile_ny, tile_nz = tile_shape
    owned_tiles = interior.reshape(
        ntx,
        tile_nx,
        nty,
        tile_ny,
        ntz,
        tile_nz,
    ).transpose(0, 2, 4, 1, 3, 5)

    field_tiles = jnp.zeros(
        tile_grid_shape
        + (
            tile_nx + 2 * g,
            tile_ny + 2 * g,
            tile_nz + 2 * g,
        ),
        dtype=interior.dtype,
    )
    return field_tiles.at[..., g:-g, g:-g, g:-g].set(owned_tiles)


def _assemble_owned(field_tiles, g):
    ntx, nty, ntz = field_tiles.shape[:3]
    owned_tiles = field_tiles[..., g:-g, g:-g, g:-g]
    tile_nx, tile_ny, tile_nz = owned_tiles.shape[-3:]
    return owned_tiles.transpose(0, 3, 1, 4, 2, 5).reshape(
        ntx * tile_nx,
        nty * tile_ny,
        ntz * tile_nz,
    )


def _periodic_mode_problem(tile_grid_shape, g=1):
    Nx = Ny = Nz = 8
    tile_shape = tuple(
        cells // num_tiles
        for cells, num_tiles in zip((Nx, Ny, Nz), tile_grid_shape)
    )
    static_parameters, dynamic_parameters = kernel_parameters(
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        tile_shape=tile_shape,
        guard_cells=g,
        boundary_conditions=(BC_PERIODIC, BC_PERIODIC, BC_PERIODIC),
        electrostatic=True,
        solver="electrostatic",
    )

    ii, jj, kk = jnp.meshgrid(
        jnp.arange(Nx),
        jnp.arange(Ny),
        jnp.arange(Nz),
        indexing="ij",
    )
    phi_true = (
        jnp.sin(2.0 * jnp.pi * ii / Nx)
        + 0.3 * jnp.cos(2.0 * jnp.pi * jj / Ny)
        + 0.2 * jnp.sin(2.0 * jnp.pi * kk / Nz)
    )
    negative_laplacian_phi = -(
        (
            jnp.roll(phi_true, 1, axis=0)
            + jnp.roll(phi_true, -1, axis=0)
            - 2.0 * phi_true
        )
        / dynamic_parameters.dx**2
        + (
            jnp.roll(phi_true, 1, axis=1)
            + jnp.roll(phi_true, -1, axis=1)
            - 2.0 * phi_true
        )
        / dynamic_parameters.dy**2
        + (
            jnp.roll(phi_true, 1, axis=2)
            + jnp.roll(phi_true, -1, axis=2)
            - 2.0 * phi_true
        )
        / dynamic_parameters.dz**2
    )
    rho = dynamic_parameters.eps * negative_laplacian_phi
    rho_tiles = _tile_field(rho, tile_grid_shape, tile_shape, g)
    rho_tiles = update_tiled_ghost_cells(
        rho_tiles,
        static_parameters,
        g,
    )
    phi_tiles = _tile_field(
        jnp.zeros_like(phi_true),
        tile_grid_shape,
        tile_shape,
        g,
    )

    return static_parameters, dynamic_parameters, rho, rho_tiles, phi_tiles, phi_true


def _relative_phi_error(phi_tiles, phi_true, g):
    phi = _assemble_owned(phi_tiles, g)
    phi = phi - jnp.mean(phi)
    phi_true = phi_true - jnp.mean(phi_true)
    return jnp.linalg.norm(phi - phi_true) / jnp.linalg.norm(phi_true)


class TestTiledLocalSchwarz(unittest.TestCase):
    def _require_devices(self, count):
        if jax.device_count() < count:
            self.skipTest(f"Need {count} JAX devices, got {jax.device_count()}")

    def test_one_tile_matches_manufactured_discrete_solution(self):
        static_parameters, dynamic_parameters, _, rho_tiles, phi_tiles, phi_true = _periodic_mode_problem(
            (1, 1, 1)
        )
        g = int(static_parameters.guard_cells)

        phi_tiles = solve_poisson_with_tiled_local_schwarz(
            rho_tiles,
            phi_tiles,
            static_parameters,
            dynamic_parameters,
            schwarz_tol=1.0e-9,
            schwarz_max_iterations=1000,
            local_cg_tol=1.0e-9,
            local_cg_max_iterations=1000,
        )

        residual = _poisson_residual(
            rho_tiles,
            phi_tiles,
            dynamic_parameters,
            g,
        )
        self.assertLess(float(_relative_phi_error(phi_tiles, phi_true, g)), 2.0e-7)
        self.assertLess(float(jnp.max(jnp.abs(residual))), 1.0e-8)

    def test_two_tiles_couple_through_the_interface(self):
        self._require_devices(2)
        static_parameters, dynamic_parameters, _, rho_tiles, phi_tiles, phi_true = _periodic_mode_problem(
            (2, 1, 1)
        )

        phi_tiles, diagnostics = solve_poisson_with_tiled_local_schwarz(
            rho_tiles,
            phi_tiles,
            static_parameters,
            dynamic_parameters,
            return_diagnostics=True,
        )
        local_cg_residual, schwarz_residual, schwarz_iteration = diagnostics

        self.assertLess(float(_relative_phi_error(phi_tiles, phi_true, 1)), 2.0e-5)
        self.assertEqual(local_cg_residual.shape, (2, 1, 1))
        self.assertTrue(jnp.all(jnp.isfinite(local_cg_residual)))
        self.assertLessEqual(float(schwarz_residual), 1.0e-6)
        self.assertGreater(int(schwarz_iteration), 0)

    def test_eight_tile_convergence_and_warm_start_behavior(self):
        self._require_devices(8)
        static_parameters, dynamic_parameters, rho, rho_tiles, phi_zero, phi_true = _periodic_mode_problem(
            (2, 2, 2)
        )

        phi_first, first_diagnostics = solve_poisson_with_tiled_local_schwarz(
            rho_tiles,
            phi_zero,
            static_parameters,
            dynamic_parameters,
            return_diagnostics=True,
        )
        self.assertLess(float(_relative_phi_error(phi_first, phi_true, 1)), 1.0e-3)
        self.assertLessEqual(float(first_diagnostics[1]), 1.0e-6)

        phi_warm, warm_converged_diagnostics = solve_poisson_with_tiled_local_schwarz(
            rho_tiles,
            phi_first,
            static_parameters,
            dynamic_parameters,
            return_diagnostics=True,
        )
        self.assertEqual(int(warm_converged_diagnostics[2]), 0)
        self.assertTrue(jnp.allclose(phi_warm, phi_first))

        rho_shifted = jnp.roll(rho, 1, axis=0)
        rho_shifted_tiles = _tile_field(rho_shifted, (2, 2, 2), (4, 4, 4), 1)
        rho_shifted_tiles = update_tiled_ghost_cells(
            rho_shifted_tiles,
            static_parameters,
            1,
        )
        _, warm_diagnostics = solve_poisson_with_tiled_local_schwarz(
            rho_shifted_tiles,
            phi_first,
            static_parameters,
            dynamic_parameters,
            schwarz_max_iterations=0,
            return_diagnostics=True,
        )
        _, cold_diagnostics = solve_poisson_with_tiled_local_schwarz(
            rho_shifted_tiles,
            phi_zero,
            static_parameters,
            dynamic_parameters,
            schwarz_max_iterations=0,
            return_diagnostics=True,
        )

        warm_defect = jnp.max(warm_diagnostics[0])
        cold_defect = jnp.max(cold_diagnostics[0])
        self.assertLess(float(warm_defect), 0.5 * float(cold_defect))

    def test_conducting_boundaries_preserve_the_existing_constant_ghost_rule(self):
        Nx = Ny = Nz = 8
        static_parameters, dynamic_parameters = kernel_parameters(
            Nx=Nx,
            Ny=Ny,
            Nz=Nz,
            tile_shape=(Nx, Ny, Nz),
            guard_cells=1,
            boundary_conditions=(BC_CONDUCTING, BC_CONDUCTING, BC_CONDUCTING),
            electrostatic=True,
            solver="electrostatic",
        )
        ii, jj, kk = jnp.meshgrid(
            jnp.arange(Nx),
            jnp.arange(Ny),
            jnp.arange(Nz),
            indexing="ij",
        )
        phi_true = (
            jnp.cos(jnp.pi * ii / (Nx - 1))
            + 0.2 * jnp.cos(2.0 * jnp.pi * jj / (Ny - 1))
            + 0.1 * jnp.cos(jnp.pi * kk / (Nz - 1))
        )
        phi_true_tiles = _tile_field(phi_true, (1, 1, 1), (Nx, Ny, Nz), 1)
        phi_true_tiles = _apply_tiled_phi_constant_boundaries(
            phi_true_tiles,
            static_parameters,
            1,
        )
        rho_owned = -dynamic_parameters.eps * _tiled_laplacian(
            phi_true_tiles,
            dynamic_parameters,
            1,
        )
        rho_tiles = jnp.zeros_like(phi_true_tiles)
        rho_tiles = rho_tiles.at[..., 1:-1, 1:-1, 1:-1].set(rho_owned)

        phi_tiles, diagnostics = solve_poisson_with_tiled_local_schwarz(
            rho_tiles,
            jnp.zeros_like(phi_true_tiles),
            static_parameters,
            dynamic_parameters,
            return_diagnostics=True,
        )
        local_cg_residual, schwarz_residual, schwarz_iteration = diagnostics

        self.assertLess(float(_relative_phi_error(phi_tiles, phi_true, 1)), 2.0e-4)
        self.assertTrue(jnp.all(jnp.isfinite(local_cg_residual)))
        self.assertLessEqual(float(schwarz_residual), 1.0e-6)
        self.assertGreater(int(schwarz_iteration), 0)
        self.assertTrue(
            jnp.allclose(
                phi_tiles[..., 0, 1:-1, 1:-1],
                phi_tiles[..., 1, 1:-1, 1:-1],
            )
        )
        self.assertTrue(
            jnp.allclose(
                phi_tiles[..., -1, 1:-1, 1:-1],
                phi_tiles[..., -2, 1:-1, 1:-1],
            )
        )

    def test_zero_residual_is_nan_safe_and_jittable(self):
        static_parameters, dynamic_parameters, _, rho_tiles, phi_tiles, _ = _periodic_mode_problem(
            (1, 1, 1)
        )
        rho_tiles = jnp.zeros_like(rho_tiles)
        phi_tiles = jnp.zeros_like(phi_tiles)

        solve = jax.jit(
            lambda rho, phi: solve_poisson_with_tiled_local_schwarz(
                rho,
                phi,
                static_parameters,
                dynamic_parameters,
                return_diagnostics=True,
            )
        )
        phi_tiles, diagnostics = solve(rho_tiles, phi_tiles)

        self.assertTrue(jnp.all(jnp.isfinite(phi_tiles)))
        self.assertTrue(jnp.allclose(phi_tiles, 0.0))
        self.assertTrue(jnp.allclose(diagnostics[0], 0.0))
        self.assertEqual(float(diagnostics[1]), 0.0)
        self.assertEqual(int(diagnostics[2]), 0)

    def test_local_cg_reductions_keep_the_tile_axes(self):
        source = inspect.getsource(_local_tile_cg_solve)
        production_source = inspect.getsource(calculate_electrostatic_fields)

        self.assertEqual(source.count("axis=(-3, -2, -1)"), 3)
        self.assertIn("solve_poisson_with_tiled_local_schwarz", production_source)
        self.assertNotIn("conjugate_gradient", production_source)
        self.assertNotIn("rho_tiles[0", production_source)
        self.assertNotIn("phi_tiles[0", production_source)


if __name__ == "__main__":
    unittest.main()
