"""Matched-filter initial data and native periodic two-stream evolution."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import jax
import jax.numpy as jnp
import toml

try:
    import JAX_BSSN
except ModuleNotFoundError as error:
    if error.name == "JAX_BSSN":
        raise unittest.SkipTest("Optional JAX_BSSN dependency is not installed")
    raise

from demos.dynamical_relativity.full_relativistic_two_stream import initial_data as demo
from PyPIC3D.__main__ import run_PyPIC3D
from PyPIC3D.diagnostics.static_metric import electromagnetic_constraints
from PyPIC3D.initialization import initialize_simulation
from PyPIC3D.relativity.dynamical_relativity import (
    initialize_dynamical_relativity, synchronize_particles, tiles_to_bssn,
)
from PyPIC3D.relativity.filtering import particle_metric
from PyPIC3D.relativity.interpolate_metric import interpolate_metric
from PyPIC3D.solvers.GR_yee.time_loop_dynamical_relativity import time_loop_dynamical_relativity

jax.config.update("jax_enable_x64", True)


class FilteredTwoStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.initial = demo.build_initial_data(
            nx=16, particles_per_cell=2, end_time=.02, current_filter="bilinear")

    def assert_arrays_close(self, actual, expected, atol=2.e-12):
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
            self.assertTrue(bool(jnp.allclose(a, b, rtol=2.e-12, atol=atol)), a.shape)

    def test_filtered_solve_and_orthonormal_momenta(self):
        particles, species, fields, static, dynamic, metadata = self.initial
        self.assertEqual(static.current_filter, "bilinear")
        self.assertLess(metadata["initial_constraints"]["hamiltonian_linf"],
                        metadata["constraint_tolerance"])
        self.assertLess(metadata["initial_constraints"]["momentum_linf"], 1.e-12)
        self.assertLess(metadata["initial_constraints"]["gauss_linf"], 1.e-12)
        self.assertLess(metadata["source_verification_linf"], 1.e-11)
        self.assertLess(metadata["volume_error"], demo.VOLUME_TOLERANCE)

        orthonormal, _ = demo.load_plasma(
            16, 2, metadata["length_x"], demo.WAVE_NUMBER, demo.PERTURBATION,
            demo.NUMBER_DENSITY, demo.BEAM_SPEED, demo.ELECTRON_TEMPERATURE,
            demo.ION_TEMPERATURE, demo.MASS_RATIO, demo.SEED)
        center = particle_metric(fields[6], static, dynamic)
        sampled = interpolate_metric(
            jax.tree.map(lambda value: value[0, 0, 0], center),
            particles.x.reshape(-1, 3),
            tuple(axis[0, 0, 0] for axis in dynamic.grids.tiled_center_grid),
            static.metric, (True, False, False), (static.guard_cells,) * 3,
            derivatives=False)
        u = particles.u.reshape(-1, 3)
        norm = jnp.einsum("ni,nij,nj->n", u, sampled.gamma_inv, u)
        self.assert_arrays_close(norm.reshape(particles.active.shape),
                                 jnp.sum(orthonormal.u**2, axis=-1))

    def test_toml_round_trip_initializes_history_once(self):
        particles, species, fields, static, dynamic, metadata = self.initial
        expected_p, expected_f, expected_d = initialize_dynamical_relativity(
            particles, species, fields, static, dynamic)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            demo.save_initial_data(path / "inputs", particles, species, fields,
                                   static._replace(output_dir=str(path / "run")), dynamic, metadata)
            config = toml.load(path / "inputs" / "two_stream.toml")
            config["simulation_parameters"]["gr_work_energy_theorem"] = False
            config["plotting"].update(dump_particles=False, dump_fields=False)
            self.assertEqual(config["simulation_parameters"]["filter_j"], "bilinear")
            self.assertEqual(config["simulation_parameters"]["alpha"], float(dynamic.alpha))
            self.assertTrue(Path(config["metric"]["conformal_factor"]).is_absolute())
            with contextlib.redirect_stdout(io.StringIO()):
                _, actual_p, actual_f, _, actual_d, _, _, actual_species = initialize_simulation(config)
        self.assert_arrays_close(actual_species, species)
        self.assert_arrays_close(actual_p, expected_p)
        # rho is scratch storage in native EM initialization; sources are
        # recomputed from particles when needed, rather than loaded from TOML.
        self.assert_arrays_close(actual_f[:3] + actual_f[4:], expected_f[:3] + expected_f[4:])
        self.assert_arrays_close(actual_d.bssn_parameters.state, expected_d.bssn_parameters.state)
        self.assert_arrays_close(synchronize_particles(actual_p, actual_species, actual_f, static, actual_d),
                                 particles)

    def test_native_smoke_writes_initial_output_and_evolves_metric(self):
        import openpmd_api as pmd

        particles, species, fields, static, dynamic, metadata = self.initial
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            demo.save_initial_data(path / "inputs", particles, species, fields,
                                   static._replace(output_dir=str(path / "run")), dynamic, metadata)
            config = toml.load(path / "inputs" / "two_stream.toml")
            config["simulation_parameters"]["gr_work_energy_theorem"] = False
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = run_PyPIC3D(config)
            final_static, final_dynamic, _, _, final_particles, final_fields, final_species = result
            for name in ("fields", "particles"):
                series = pmd.Series(str(path / "run" / "data" / f"{name}.h5"),
                                    pmd.Access.read_only)
                try:
                    self.assertEqual(list(series.iterations), [0, static.Nt])
                    self.assertAlmostEqual(series.iterations[0].time, 0., places=13)
                    self.assertAlmostEqual(series.iterations[static.Nt].time,
                                           float(dynamic.dt) * static.Nt, places=13)
                finally:
                    series.close()
        self.assertEqual(int(jnp.sum(final_particles.active)), int(jnp.sum(particles.active)))
        self.assertEqual(final_static.boundary_conditions, (0, 0, 0))
        self.assertEqual(final_static.particle_boundary_conditions, (0, 0, 0))
        state = final_dynamic.bssn_parameters.state
        self.assertTrue(bool(jnp.all(state.shift == 0.)))
        self.assertGreater(float(jnp.max(jnp.abs(state.lapse - 1.))), 1.e-8)
        self.assertTrue(all(bool(jnp.all(jnp.isfinite(a))) for a in jax.tree.leaves(state)))
        gauss, div_b = electromagnetic_constraints(
            final_particles, final_species, final_fields, final_static, final_dynamic)
        self.assertLess(float(jnp.max(jnp.abs(tiles_to_bssn(gauss, final_static)))), 1.e-11)
        self.assertLess(float(jnp.max(jnp.abs(tiles_to_bssn(div_b, final_static)))), 1.e-11)

    def test_nonidentity_digital_initialization_round_trip(self):
        particles, species, fields, static, dynamic, metadata = demo.build_initial_data(
            nx=8, particles_per_cell=2, end_time=.02,
            current_filter="digital", filter_alpha=.5)
        self.assertEqual(metadata["filter_alpha"], .5)
        self.assertLess(metadata["initial_constraints"]["hamiltonian_linf"],
                        metadata["constraint_tolerance"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            demo.save_initial_data(path / "inputs", particles, species, fields,
                                   static._replace(output_dir=str(path / "run")), dynamic, metadata)
            config = toml.load(path / "inputs" / "two_stream.toml")
            config["simulation_parameters"]["gr_work_energy_theorem"] = False
            config["plotting"].update(dump_particles=False, dump_fields=False)
            self.assertEqual(config["simulation_parameters"]["filter_j"], "digital")
            self.assertEqual(config["simulation_parameters"]["alpha"], .5)
            with contextlib.redirect_stdout(io.StringIO()):
                _, actual_p, actual_f, actual_s, actual_d, _, _, actual_species = initialize_simulation(config)
        self.assertEqual(float(actual_d.alpha), .5)
        self.assertEqual(actual_s.current_filter, "digital")
        self.assert_arrays_close(actual_d.bssn_parameters.state, dynamic.bssn_parameters.state)
        gauss, _ = electromagnetic_constraints(actual_p, actual_species, actual_f, actual_s, actual_d)
        self.assertLess(float(jnp.max(jnp.abs(tiles_to_bssn(gauss, actual_s)))), 1.e-12)

    def test_zero_gravity_seeded_growth(self):
        particles, species, fields, static, dynamic, _ = demo.build_initial_data(
            nx=16, particles_per_cell=8, end_time=12., cfl=.2, gravity_G=0.,
            current_filter="bilinear")
        particles, fields, dynamic = initialize_dynamical_relativity(
            particles, species, fields, static, dynamic)
        initial_state = dynamic.bssn_parameters.state
        initial_Dx = tiles_to_bssn(fields[0][0], static)[:, 0, 0]
        initial_energy = jnp.sum(initial_Dx**2)
        initial_mode_power = jnp.abs(jnp.fft.rfft(initial_Dx)[1])**2

        @jax.jit
        def evolve(particles, fields, dynamic):
            def step(carry, _):
                p, f, d = carry
                p, f, d = time_loop_dynamical_relativity(p, species, f, static, d)
                energy = jnp.sum(tiles_to_bssn(f[0][0], static)**2)
                return (p, f, d), energy
            return jax.lax.scan(step, (particles, fields, dynamic), None, length=static.Nt)

        (final_particles, final_fields, final_dynamic), energies = evolve(particles, fields, dynamic)
        final_Dx = tiles_to_bssn(final_fields[0][0], static)[:, 0, 0]
        final_mode_power = jnp.abs(jnp.fft.rfft(final_Dx)[1])**2
        self.growth = dict(peak_electric_energy_ratio=float(jnp.max(energies) / initial_energy),
                           seeded_mode_power_ratio=float(final_mode_power / initial_mode_power))
        self.assertTrue(bool(jnp.all(jnp.isfinite(energies))))
        self.assertGreater(self.growth["peak_electric_energy_ratio"], 2.)
        self.assertGreater(self.growth["seeded_mode_power_ratio"], 2.)
        self.assertEqual(int(jnp.sum(final_particles.active)), int(jnp.sum(particles.active)))
        for actual, expected in zip(jax.tree.leaves(final_dynamic.bssn_parameters.state),
                                    jax.tree.leaves(initial_state)):
            self.assertTrue(bool(jnp.array_equal(actual, expected)))
        gauss, _ = electromagnetic_constraints(final_particles, species, final_fields, static, final_dynamic)
        self.assertLess(float(jnp.max(jnp.abs(tiles_to_bssn(gauss, static)))), 1.e-11)


if __name__ == "__main__":
    unittest.main()
