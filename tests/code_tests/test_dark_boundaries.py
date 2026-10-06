"""Proca PEC images, PML state contracts, and public runtime integration.

Run with JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 JAX_NUM_CPU_DEVICES=4.
"""
import contextlib
import io
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import openpmd_api as pmd

from PyPIC3D.__main__ import run_PyPIC3D
from PyPIC3D.boundary_conditions.PML import load_pml_from_toml
from PyPIC3D.diagnostics.diagnostic_quantities import compute_dark_energy
from PyPIC3D.diagnostics.output_adapters import build_field_output_map, scalar_field_for_output
from PyPIC3D.initialization import initialize_simulation
from PyPIC3D.relativity.core import D_FIELD_LOCATIONS, B_FIELD_LOCATIONS
from PyPIC3D.solvers.dark_matter_yee.boundaries import dark_field_boundaries
from PyPIC3D.solvers.dark_matter_yee.dark_photon_fields import (
    initialize_dark_photon_fields, synchronized_dark_fields, dark_divergence,
)
from PyPIC3D.solvers.dark_matter_yee.pml import initialize_dark_pml, advance_dark_pml, stretch_dark_derivatives
from PyPIC3D.solvers.dark_matter_yee.time_loop import time_loop_dark_photon
from PyPIC3D.solvers.yee.time_loop import time_loop_electrodynamic
from PyPIC3D.utilities.parameters import build_static_parameters
from tests.kernel_fixtures import kernel_parameters
from tests.code_tests.test_dark_photon import evolve, interior
from tests.code_tests import test_dark_photon as periodic_tests


def wall_mode(s, d, location, parity):
    """A separable mode with known PEC images on every guard plane."""
    value = 1.
    for axis, (length, lower) in enumerate(zip((d.x_wind, d.y_wind, d.z_wind), (-d.x_wind/2, -d.y_wind/2, -d.z_wind/2))):
        grid = d.grids.tiled_center_grid if location[axis] == "C" else d.grids.tiled_vertex_grid
        phase = np.pi * (grid[axis] - lower) / length
        if s.boundary_conditions[axis] == 0:
            factor = jnp.cos(2 * phase)
        else:
            factor = jnp.sin(phase) if parity[axis] == -1 else jnp.cos(phase)
        shape = list(factor.shape[:3]) + [1, 1, 1]
        shape[axis + 3] = factor.shape[-1]
        value = value * factor.reshape(shape)
    return value


def pml_evolve(fields, state, s, d, steps):
    J = tuple(jnp.zeros_like(v) for v in fields[0])
    return jax.jit(lambda f, p: jax.lax.fori_loop(
        0, steps, lambda _, pair: advance_dark_pml(pair[0], J, pair[1], s, d), (f, p),
    ))(fields, state)


class TestDarkBoundaries(unittest.TestCase):
    def assert_tree_close(self, a, b, atol=2e-12):
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            np.testing.assert_allclose(x, y, rtol=2e-12, atol=atol)

    def test_pec_images_faces_corners_and_tile_seams(self):
        for g in (1, 2):
            for tiles in (1, 4):
                if tiles > len(jax.devices()):
                    continue
                for bc in ((1, 1, 1), (1, 0, 1)):
                    s, d = kernel_parameters(Nx=8, Ny=6, Nz=4, tile_shape=(8//tiles, 6, 4),
                                             guard_cells=g, boundary_conditions=bc, solver="dark_matter_yee")
                    for kind in ("E", "B", "phi"):
                        locations = (("C",)*3,) if kind == "phi" else B_FIELD_LOCATIONS if kind == "B" else D_FIELD_LOCATIONS
                        exact = []
                        for component, location in enumerate(locations):
                            parity = tuple(-1 if kind == "phi" else
                                           (-1 if axis == component else 1) if kind == "B" else
                                           (1 if axis == component else -1) for axis in range(3))
                            exact.append(wall_mode(s, d, location, parity))
                        values = tuple(jnp.zeros_like(v).at[interior(s)].set(v[interior(s)]) for v in exact)
                        result = dark_field_boundaries(values[0] if kind == "phi" else values, s, kind)
                        self.assert_tree_close(result, exact[0] if kind == "phi" else tuple(exact))

    def test_ade_interval_average_and_zero_sigma(self):
        sigma = (jnp.array([0., .3, 20.]),)*3
        d, p = jnp.array([2., 3., 4.]), jnp.array([0., .2, -.1])
        for dt in (.1, -.05):
            result, memory = stretch_dark_derivatives((d,), (p,), sigma, (0,), dt)
            q = np.asarray(sigma[0]) * dt
            expected = np.exp(-q)*np.asarray(p)+np.expm1(-q)*np.asarray(d)
            np.testing.assert_allclose(memory[0], expected, atol=1e-15)
            np.testing.assert_allclose(result[0][1:], -(expected[1:]-p[1:])/q[1:], atol=1e-14)
            self.assertEqual(float(result[0][0]), 2.)

    def test_zero_sigma_matches_potential_solver_and_initial_reconstruction(self):
        for bc in ((0, 0, 0), (1, 0, 1)):
            s, d = kernel_parameters(Nx=8, Ny=6, Nz=4, dt=.002, dark_mu=.7,
                                     boundary_conditions=bc, solver="dark_matter_yee")
            zero = initialize_dark_photon_fields(s, d)[2]
            rng = np.random.default_rng(93)
            E, A = [tuple(jnp.asarray(rng.normal(size=zero.shape)) for _ in range(3)) for _ in range(2)]
            fields = initialize_dark_photon_fields(s, d, E, A, jnp.asarray(rng.normal(size=zero.shape)))
            for strength in (0., 3.):
                profiles = load_pml_from_toml(dict(wall="+x", thickness=3, sigma_max=strength), s, d)[4]
                seeded, state = initialize_dark_pml(fields, s, d, profiles)
                expected = synchronized_dark_fields(fields, s, d)
                self.assert_tree_close(synchronized_dark_fields(seeded, s, d, state), expected)
                if strength == 0.:
                    result, _ = pml_evolve(seeded, state, s, d, 8)
                    self.assert_tree_close(result, evolve(fields, s, d, 8))

    def test_pml_jit_tiles_diagnostics_and_interior_constraint(self):
        results = []
        for tiles in (1, 4):
            if tiles > len(jax.devices()):
                continue
            s, d = kernel_parameters(Nx=32, Ny=1, Nz=1, tile_shape=(32//tiles, 1, 1),
                                     boundary_conditions=(1, 0, 0), dt=.01, dark_mu=.7,
                                     solver="dark_matter_yee", pml_active=True)
            E, A, phi = initialize_dark_photon_fields(s, d)
            ax = wall_mode(s, d, D_FIELD_LOCATIONS[0], (1, -1, -1))
            fields = initialize_dark_photon_fields(s, d, E, (ax, A[1], A[2]), phi)
            profiles = load_pml_from_toml(dict(wall="+x", thickness=8), s, d)[4]
            fields, state = initialize_dark_pml(fields, s, d, profiles)
            J = tuple(jnp.zeros_like(v) for v in E)
            self.assert_tree_close(advance_dark_pml(fields, J, state, s, d), pml_evolve(fields, state, s, d, 1))
            fields, state = pml_evolve(fields, state, s, d, 20)
            before = jax.tree.map(np.array, (fields, state))
            first = synchronized_dark_fields(fields, s, d, state)
            compute_dark_energy(fields, s, d, state)
            self.assert_tree_close(first, synchronized_dark_fields(fields, s, d, state))
            self.assert_tree_close(before, (fields, state), atol=0.)
            residual = dark_divergence(fields[0], s, d) + s.dark_mu**2*fields[2]
            # Divergence returns interior-only data; output assembly reads
            # overlapping halos at tile seams, so refresh the diagnostic first.
            residual = dark_field_boundaries(residual, s, "phi")
            residual = scalar_field_for_output(residual, s)[1:-1, 1:-1, 1:-1]
            np.testing.assert_allclose(residual[1:22], 0., atol=2e-13)
            results.append(jax.tree.map(lambda v: scalar_field_for_output(v, s)[1:-1, 1:-1, 1:-1], first))
        if len(results) == 2:
            self.assert_tree_close(*results)

    def test_public_initialization_and_sector_history_independence(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            config = periodic_tests.TestDarkPhoton().config(directory, x_bc="conducting", sin_chi=0.)
            config["pml"] = [dict(wall="+x", thickness=3)]
            path = directory + "/dark_A.npy"
            np.save(path, np.sin(np.pi*np.arange(8)/8)[:, None, None])
            config["dark_field4"] = dict(type=4, path=path)
            loop, particles, fields, s, d, _, _, species = initialize_simulation(config)
            self.assertIs(loop, time_loop_dark_photon)
            self.assertEqual(len(fields), 9)
            self.assertEqual(s.boundary_conditions, (1, 0, 0))
            self.assertEqual(build_static_parameters(s._asdict()), s)
            eager = loop(particles, species, fields, s, d)
            compiled = jax.jit(lambda p, f: loop(p, species, f, s, d))(particles, fields)
            self.assert_tree_close(eager, compiled)
            fields = compiled[1]
            for history in jax.tree.leaves(fields[6][0][:2]):
                np.testing.assert_array_equal(history, 0.)
            # A starts nonzero but E starts zero: Ampere memory is excited on
            # this first kick; Faraday memory is first driven on the next drift.
            self.assertGreater(sum(float(jnp.linalg.norm(v)) for v in fields[6][1][4]), 0.)
            output = build_field_output_map(fields, compiled[0], species, s, d)
            expected = synchronized_dark_fields(fields[7], s, d, fields[6][1])
            self.assert_tree_close(output["dark_B"], expected[3])

            # Excite only Maxwell and compare the complete coupled production
            # step against the ordinary Yee loop, including its PML histories.
            zero_dark = initialize_dark_photon_fields(s, d)
            profiles = load_pml_from_toml(config["pml"], s, d)[4]
            zero_dark, dark_pml = initialize_dark_pml(zero_dark, s, d, profiles)
            ordinary = (expected[1], expected[3], *fields[2:6], fields[6][0], fields[8])
            coupled = ordinary[:6] + ((ordinary[6], dark_pml), zero_dark, ordinary[7])
            yee = jax.jit(lambda p, f: time_loop_electrodynamic(p, species, f, s, d))(particles, ordinary)
            coupled = jax.jit(lambda p, f: loop(p, species, f, s, d))(particles, coupled)
            actual = coupled[1]
            self.assert_tree_close(yee, (coupled[0], actual[:6] + (actual[6][0], actual[8])))
            for history in jax.tree.leaves(actual[6][1][1:5]):
                np.testing.assert_array_equal(history, 0.)

    def test_pml_driver_particle_seed_and_openpmd(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            config = periodic_tests.TestDarkPhoton().config(
                directory, relativistic=False, current_calculation="esirkepov",
            )
            config["pml"] = [dict(wall="+x", thickness=3)]
            config["plotting"] = dict(dump_fields=False, plot_openpmd_fields=True)
            path = directory + "/dark_E.npy"
            np.save(path, np.full((8, 1, 1), .75))
            config["dark_field0"] = dict(type=0, path=path)
            config["particle1"] = dict(name="p", N_particles=1, mass=4., charge=2., temperature=0.,
                                       initial_x=0., initial_y=0., initial_z=0.,
                                       initial_vx=0., initial_vy=0., initial_vz=0.)
            _, particles, fields, s, d, *_ = initialize_simulation(config)
            np.testing.assert_allclose(np.asarray(particles.u)[np.asarray(particles.active), 0],
                                       (2/4)*s.sin_chi*.75*float(d.dt)/2, atol=1e-15)
            result = run_PyPIC3D(config)
            self.assertEqual(len(result[5]), 9)
            self.assertIsNotNone(result[5][6])
            for value in jax.tree.leaves(result[5]):
                self.assertTrue(np.all(np.isfinite(value)))
            series = pmd.Series(directory + "/data/fields.h5", pmd.Access.read_only)
            for iteration in series.iterations:
                for name in ("dark_E", "dark_A", "dark_phi", "dark_B"):
                    self.assertEqual(series.iterations[iteration].meshes[name].time_offset, 0.)
            values = series.iterations[0].meshes["dark_E"]["x"].load_chunk()
            series.flush()
            np.testing.assert_allclose(values, .75, atol=1e-14)
            series.close()


if __name__ == "__main__":
    unittest.main()
