import unittest
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import jax
import numpy as np
import jax.numpy as jnp

from PyPIC3D.diagnostics import async_writer
from PyPIC3D.diagnostics import openPMD
from PyPIC3D.diagnostics.openPMD import _ensure_openpmd_array
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.relativity.core import build_yee_metric
from PyPIC3D.relativity.interpolate_metric import particle_lorentz_factor
from tests.kernel_fixtures import (
    build_tiled_particles,
    field_tiles_from_global,
    kernel_parameters,
    kernel_parameters_from_values,
    particle_parameters_from_values,
    particle_species,
    species_names as names_for_species,
    vector_tiles_from_global,
)


def _tile_axis_count(n_cells, cells_per_tile):
    if int(n_cells) % int(cells_per_tile) != 0:
        raise ValueError("Shared tile sizes must divide the physical grid dimensions exactly.")
    return int(n_cells) // int(cells_per_tile)


def tile_scalar_field(field, static_parameters, dynamic_parameters, num_guard_cells=None):
    return field_tiles_from_global(field, static_parameters, dynamic_parameters, num_guard_cells)


def tile_vector_field(field, static_parameters, dynamic_parameters, num_guard_cells=None):
    return vector_tiles_from_global(field, static_parameters, dynamic_parameters, num_guard_cells)


class FakeRecord:
    def __init__(self):
        self.shape = None
        self.dataset_shape = None
        self.unit_SI = None
        self.data = None
        self.chunks = []

    def reset_dataset(self, dataset):
        extent = getattr(dataset, "extent", None)
        if extent is not None:
            self.dataset_shape = tuple(extent)

    def store_chunk(self, array, offset, extent):
        self.shape = tuple(extent)
        self.data = np.array(array, copy=True)
        self.chunks.append((tuple(offset), tuple(extent), np.array(array, copy=True)))


class FakeMesh:
    def __init__(self):
        self.records = {}
        self.axis_labels = None
        self.grid_spacing = None
        self.grid_global_offset = None
        self.unit_dimension = None

    def __getitem__(self, component):
        if component not in self.records:
            self.records[component] = FakeRecord()
        return self.records[component]


class FakeMeshes(dict):
    def __getitem__(self, name):
        if name not in self:
            self[name] = FakeMesh()
        return dict.__getitem__(self, name)


class FakeParticleRecord(dict):
    def __getitem__(self, component):
        if component not in self:
            self[component] = FakeRecord()
        return dict.__getitem__(self, component)


class FakeParticleSpecies(dict):
    def __getitem__(self, record_name):
        if record_name not in self:
            if record_name in ("position", "positionOffset", "momentum"):
                self[record_name] = FakeParticleRecord()
            else:
                self[record_name] = FakeRecord()
        return dict.__getitem__(self, record_name)


class FakeParticles(dict):
    def __getitem__(self, species_name):
        if species_name not in self:
            self[species_name] = FakeParticleSpecies()
        return dict.__getitem__(self, species_name)


class FakeIteration:
    def __init__(self):
        self.meshes = FakeMeshes()
        self.particles = FakeParticles()
        self.time = None
        self.dt = None
        self.time_unit_SI = None


class FakeIterations(dict):
    def __getitem__(self, iteration):
        if iteration not in self:
            self[iteration] = FakeIteration()
        return dict.__getitem__(self, iteration)


class FakeSeries:
    def __init__(self):
        self.iterations = FakeIterations()

    def set_attribute(self, name, value):
        pass

    def flush(self):
        pass

    def close(self):
        pass


def _zero_field(shape):
    return tuple(jnp.zeros(shape) for _ in range(3))


def _parameter_values():
    return {
        "dt": 0.2,
        "dx": 1.0,
        "dy": 1.0,
        "dz": 1.0,
        "x_wind": 4.0,
        "y_wind": 2.0,
        "z_wind": 1.0,
        "Nx": 4,
        "Ny": 2,
        "Nz": 1,
        "tile_shape": (2, 1, 1),
        "guard_cells": 2,
        "boundary_conditions": {"x": 0, "y": 0, "z": 0},
        "particle_boundary_conditions": {"x": 0, "y": 0, "z": 0},
    }


def _record_data(record):
    if len(record.chunks) <= 1:
        return record.data

    n = sum(int(extent[0]) for _offset, extent, _data in record.chunks)
    out = jnp.zeros((n,), dtype=record.chunks[0][2].dtype)
    for offset, extent, data in record.chunks:
        start = int(offset[0])
        stop = start + int(extent[0])
        out = out.at[start:stop].set(data)
    return out


def _species(name, charge, mass, weight, x1):
    return particle_species(
        name=name,
        charge=charge,
        mass=mass,
        weight=weight,
        x1=jnp.asarray(x1),
        u1=jnp.ones_like(x1) * 0.1,
    )


class OpenPMDDiagnosticsTests(unittest.TestCase):

    def test_output_arrays_are_host_owned_writable_and_contiguous(self):
        original = np.arange(24, dtype=np.float64).reshape(4, 6)[:, ::2]
        original.setflags(write=False)
        for data in (original, jnp.asarray(original), np.empty((0, 3))):
            for dtype in (np.float32, np.float64):
                with self.subTest(input_type=type(data), dtype=dtype):
                    actual = _ensure_openpmd_array(data, dtype=dtype)
                    self.assertIsInstance(actual, np.ndarray)
                    self.assertEqual(actual.dtype, np.dtype(dtype))
                    self.assertTrue(actual.flags.c_contiguous)
                    self.assertTrue(actual.flags.writeable)
                    np.testing.assert_array_equal(actual, np.asarray(data, dtype=dtype))

    def test_host_float64_conversion_does_not_depend_on_jax_precision(self):
        original = np.array([1. + 2.**-40], dtype=np.float64)
        with jax.enable_x64(False), \
             patch.object(jnp, 'asarray', side_effect=AssertionError('host data sent to JAX')):
            actual = _ensure_openpmd_array(original)
        np.testing.assert_array_equal(actual, original)
        self.assertEqual(actual.dtype, np.dtype('float64'))

    def test_host_particle_snapshot_preserves_chunks_and_empty_species_without_jax_arrays(self):
        static, dynamic = particle_parameters_from_values(
            _parameter_values(), dynamic_values={'C': 10.})
        x = np.zeros((2, 1, 1, 2, 2, 3), dtype=np.float64)
        x[0, 0, 0, 0, 0, 0] = -1.5
        x[1, 0, 0, 0, 1, 0] = .5 + 2.**-40
        active = np.zeros(x.shape[:-1], dtype=bool)
        active[0, 0, 0, 0, 0] = active[1, 0, 0, 0, 1] = True
        index = (slice(None),)*6
        snapshot = SimpleNamespace(
            species_names=('live particles', 'empty species'),
            species_charge=np.array([-1., 1.]), species_mass=np.array([2., 3.]),
            species_weight=np.array([4., 5.]), x_shards=[(index, x)],
            u_shards=[(index, np.zeros_like(x))], active_shards=[(index[:-1], active)])
        iteration = FakeIteration()
        with jax.enable_x64(False), \
             patch.object(jnp, 'asarray', side_effect=AssertionError('host snapshot sent to JAX')), \
             patch.object(jax, 'device_put', side_effect=AssertionError('unexpected device transfer')):
            openPMD.write_tiled_particle_snapshot_to_iteration(iteration, snapshot, static, dynamic)
        record = iteration.particles['live_particles']['position']['x']
        self.assertEqual([(offset, extent) for offset, extent, _ in record.chunks],
                         [((0,), (1,)), ((1,), (1,))])
        actual = np.concatenate([data for _, _, data in record.chunks])
        np.testing.assert_array_equal(actual, [-1.5, .5 + 2.**-40])
        self.assertEqual(actual.dtype, np.dtype('float64'))
        empty = iteration.particles['empty_species']['position']['x']
        self.assertEqual(empty.dataset_shape, (0,))
        self.assertEqual(empty.chunks, [])
        self.assertEqual(record.unit_SI, 1.)

    def test_openpmd_field_array_preserves_thin_y_axis(self):
        field_component = jnp.ones((4, 1, 6))

        array = _ensure_openpmd_array(field_component)

        self.assertEqual(array.shape, (4, 1, 6))

    def test_initial_fields_preserve_thin_y_mesh_metadata(self):
        shape_with_ghosts = (6, 3, 8)
        E = _zero_field(shape_with_ghosts)
        B = _zero_field(shape_with_ghosts)
        J = _zero_field(shape_with_ghosts)
        static_parameters, dynamic_parameters = kernel_parameters(
            Nx=4,
            Ny=1,
            Nz=6,
            x_wind=1.0,
            y_wind=2.0,
            z_wind=3.0,
            dx=0.25,
            dy=0.5,
            dz=0.75,
            dt=1.0,
            tile_shape=(4, 1, 6),
            guard_cells=1,
        )
        series = FakeSeries()
        field_map = {
            "E": E,
            "B": B,
            "J": J,
        }

        with patch.object(openPMD.io, "Series", return_value=series):
            openPMD.write_openpmd_initial_fields(field_map, static_parameters, dynamic_parameters, "/tmp")

        B_mesh = series.iterations[0].meshes["B"]
        self.assertEqual(set(series.iterations[0].meshes), {"E", "B", "J"})
        self.assertEqual(B_mesh.axis_labels, ["x", "y", "z"])
        self.assertEqual(B_mesh.grid_spacing, [0.25, 0.5, 0.75])
        self.assertTrue(jnp.allclose(jnp.asarray(B_mesh.grid_global_offset), jnp.array([-0.5, -1.0, -1.5])))
        self.assertEqual(B_mesh.records["x"].shape, (4, 1, 6))

    def test_field_meshes_encode_physical_si_unit_dimensions(self):
        shape_with_ghosts = (6, 3, 8)
        vector = _zero_field(shape_with_ghosts)
        scalar = jnp.zeros(shape_with_ghosts)
        static_parameters, dynamic_parameters = kernel_parameters(
            Nx=4,
            Ny=1,
            Nz=6,
            x_wind=1.0,
            y_wind=2.0,
            z_wind=3.0,
            dx=0.25,
            dy=0.5,
            dz=0.75,
            dt=1.0,
            tile_shape=(4, 1, 6),
            guard_cells=1,
        )
        series = FakeSeries()
        field_map = {
            "E": vector,
            "B": vector,
            "J": vector,
            "rho": scalar,
            "phi": scalar,
            "fluid_velocity": vector,
        }

        with patch.object(openPMD.io, "Series", return_value=series):
            openPMD.write_openpmd_initial_fields(field_map, static_parameters, dynamic_parameters, "/tmp")

        dimensions = openPMD.io.Unit_Dimension
        expected = {
            "E": {dimensions.L: 1.0, dimensions.M: 1.0, dimensions.T: -3.0, dimensions.I: -1.0},
            "B": {dimensions.M: 1.0, dimensions.T: -2.0, dimensions.I: -1.0},
            "J": {dimensions.L: -2.0, dimensions.I: 1.0},
            "rho": {dimensions.L: -3.0, dimensions.T: 1.0, dimensions.I: 1.0},
            "phi": {dimensions.L: 2.0, dimensions.M: 1.0, dimensions.T: -3.0, dimensions.I: -1.0},
            "fluid_velocity": {dimensions.L: 1.0, dimensions.T: -1.0},
        }
        for name, unit_dimension in expected.items():
            self.assertEqual(
                series.iterations[0].meshes[name].unit_dimension,
                unit_dimension,
            )

    def test_initial_fields_use_shifted_grid_lower_bounds_for_offsets(self):
        shape_with_ghosts = (6, 3, 3)
        E = _zero_field(shape_with_ghosts)
        field_map = {"E": E}
        static_parameters, dynamic_parameters = kernel_parameters(
            Nx=4,
            Ny=1,
            Nz=1,
            x_wind=2.0,
            y_wind=1.0,
            z_wind=1.0,
            x_min=1.0,
            y_min=-0.5,
            z_min=2.0,
            dx=0.5,
            dy=1.0,
            dz=1.0,
            dt=1.0,
            tile_shape=(4, 1, 1),
            guard_cells=1,
        )
        series = FakeSeries()

        with patch.object(openPMD.io, "Series", return_value=series):
            openPMD.write_openpmd_initial_fields(field_map, static_parameters, dynamic_parameters, "/tmp")

        E_mesh = series.iterations[0].meshes["E"]
        self.assertEqual(E_mesh.grid_spacing, [0.5, 1.0, 1.0])
        self.assertEqual(E_mesh.grid_global_offset, [1.0, -0.5, 2.0])

    def test_initial_fields_assemble_tiled_fields_before_output(self):
        shape_with_ghosts = (6, 4, 4)
        E = _zero_field(shape_with_ghosts)
        B = _zero_field(shape_with_ghosts)
        J = _zero_field(shape_with_ghosts)
        rho = jnp.zeros(shape_with_ghosts)
        phi = jnp.zeros(shape_with_ghosts)
        external_fields = _zero_field(shape_with_ghosts), _zero_field(shape_with_ghosts)
        parameter_values = {
            "dt": 1.0,
            "dx": 0.25,
            "dy": 0.5,
            "dz": 0.75,
            "x_wind": 1.0,
            "y_wind": 2.0,
            "z_wind": 3.0,
            "Nx": 4,
            "Ny": 2,
            "Nz": 2,
            "tile_shape": (2, 1, 1),
            "guard_cells": 2,
            "boundary_conditions": {"x": 0, "y": 0, "z": 0},
        }
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_values)
        tiled_fields = (
            tile_vector_field(E, static_parameters, dynamic_parameters),
            tile_vector_field(B, static_parameters, dynamic_parameters),
            tile_vector_field(J, static_parameters, dynamic_parameters),
            tile_scalar_field(rho, static_parameters, dynamic_parameters),
            tile_scalar_field(phi, static_parameters, dynamic_parameters),
            (
                tile_vector_field(external_fields[0], static_parameters, dynamic_parameters),
                tile_vector_field(external_fields[1], static_parameters, dynamic_parameters),
            ),
            None,
        )
        series = FakeSeries()
        field_map = {
            "E": tiled_fields[0],
            "rho": tiled_fields[3],
        }

        with patch.object(openPMD.io, "Series", return_value=series):
            openPMD.write_openpmd_initial_fields(field_map, static_parameters, dynamic_parameters, "/tmp")

        E_mesh = series.iterations[0].meshes["E"]
        rho_mesh = series.iterations[0].meshes["rho"]
        self.assertEqual(E_mesh.records["x"].shape, (4, 2, 2))
        self.assertEqual(rho_mesh.records[openPMD.io.Mesh_Record_Component.SCALAR].shape, (4, 2, 2))

    def test_tiled_field_snapshot_writes_tile_chunks_without_global_assembly(self):
        parameter_values = _parameter_values()
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_values)
        shape_with_ghosts = (6, 4, 3)
        rho_global = jnp.arange(6 * 4 * 3, dtype=jnp.float64).reshape(shape_with_ghosts)
        phi_global = rho_global + 100.0
        E_global = tuple(rho_global + offset for offset in (10.0, 20.0, 30.0))
        tile_shape = tuple(int(width) for width in static_parameters.tile_shape)
        field_map = {
            "E": tile_vector_field(E_global, static_parameters, dynamic_parameters),
            "rho": tile_scalar_field(rho_global, static_parameters, dynamic_parameters),
            "phi": tile_scalar_field(phi_global, static_parameters, dynamic_parameters),
        }
        snapshot = async_writer.make_tiled_field_snapshot(
            field_map,
            step=3,
            time=0.6,
        )
        layout = openPMD.TiledMeshLayout(
            global_shape=(int(dynamic_parameters.Nx), int(dynamic_parameters.Ny), int(dynamic_parameters.Nz)),
            tile_shape=tile_shape,
            guard_cells=int(static_parameters.guard_cells),
        )
        series = FakeSeries()

        with patch.object(openPMD, "_open_openpmd_series", return_value=series):
            openPMD.write_tiled_field_snapshot_openpmd(
                snapshot,
                output_dir="/tmp",
                filename="fields",
                dynamic_parameters=dynamic_parameters,
                layout=layout,
                file_extension=".h5",
            )

        iteration = series.iterations[3]
        rho_record = iteration.meshes["rho"].records[openPMD.io.Mesh_Record_Component.SCALAR]
        E_record = iteration.meshes["E"].records["x"]

        self.assertEqual(iteration.time, 0.6)
        self.assertEqual(iteration.dt, float(dynamic_parameters.dt))
        self.assertEqual(iteration.meshes["rho"].axis_labels, ["x", "y", "z"])
        self.assertEqual(len(rho_record.chunks), 4)
        self.assertEqual(rho_record.chunks[0][0], (0, 0, 0))
        self.assertEqual(rho_record.chunks[0][1], tile_shape)
        self.assertEqual(len(E_record.chunks), 4)
        self.assertEqual(E_record.chunks[0][0], (0, 0, 0))
        dimensions = openPMD.io.Unit_Dimension
        self.assertEqual(
            iteration.meshes["E"].unit_dimension,
            {
                dimensions.L: 1.0,
                dimensions.M: 1.0,
                dimensions.T: -3.0,
                dimensions.I: -1.0,
            },
        )

    def test_enqueue_openpmd_field_output_preserves_selected_field_map(self):
        class RecordingFieldWriter:
            def enqueue_fields(self, field_map, *, step, time, block):
                self.field_map = field_map
                self.step = step
                self.time = time
                self.block = block
                return True

        _, dynamic_parameters = kernel_parameters_from_values(_parameter_values())
        scalar_field = jnp.zeros((1, 1, 1, 4, 4, 4))
        field_map = {
            "E": (scalar_field, scalar_field, scalar_field),
            "fluid_velocity": (scalar_field, scalar_field, scalar_field),
        }
        writer = RecordingFieldWriter()

        accepted = async_writer.enqueue_openpmd_field_output(
            writer,
            field_map,
            dynamic_parameters,
            plot_t=3,
            t=4,
        )

        self.assertTrue(accepted)
        self.assertIs(writer.field_map, field_map)
        self.assertEqual(tuple(writer.field_map), ("E", "fluid_velocity"))
        self.assertEqual(writer.step, 3)
        self.assertEqual(writer.time, 4 * float(dynamic_parameters.dt))
        self.assertTrue(writer.block)

    def test_async_field_writer_queue_size_caps_pending_snapshots(self):
        static_parameters, dynamic_parameters = kernel_parameters_from_values(_parameter_values())
        writer = async_writer.AsyncTiledOpenPMDFieldWriter(
            output_dir="/tmp",
            filename="fields",
            static_parameters=static_parameters,
            dynamic_parameters=dynamic_parameters,
            global_shape=(int(dynamic_parameters.Nx), int(dynamic_parameters.Ny), int(dynamic_parameters.Nz)),
            tile_shape=tuple(int(width) for width in static_parameters.tile_shape),
            guard_cells=int(static_parameters.guard_cells),
            queue_size=1,
        )
        snapshot = async_writer.TiledFieldSnapshot(step=0, time=0.0, fields={})

        self.assertTrue(writer.enqueue(snapshot, block=False))
        self.assertFalse(writer.enqueue(snapshot, block=False))
        writer.close(raise_errors=False)

    def test_async_field_writer_raises_worker_errors_on_close(self):
        static_parameters, dynamic_parameters = kernel_parameters_from_values(_parameter_values())
        writer = async_writer.AsyncTiledOpenPMDFieldWriter(
            output_dir="/tmp",
            filename="fields",
            static_parameters=static_parameters,
            dynamic_parameters=dynamic_parameters,
            global_shape=(int(dynamic_parameters.Nx), int(dynamic_parameters.Ny), int(dynamic_parameters.Nz)),
            tile_shape=tuple(int(width) for width in static_parameters.tile_shape),
            guard_cells=int(static_parameters.guard_cells),
            queue_size=1,
        )
        snapshot = async_writer.TiledFieldSnapshot(step=0, time=0.0, fields={})

        with patch.object(async_writer, "write_tiled_field_snapshot_openpmd", side_effect=RuntimeError("disk failed")):
            writer.start()
            self.assertTrue(writer.enqueue(snapshot))
            with self.assertRaisesRegex(RuntimeError, "Async openPMD writer failed"):
                writer.close()

    def test_tiled_particle_snapshot_writes_expected_flat_records(self):
        parameter_values = _parameter_values()
        dynamic_values = {"C": 10.0}
        static_parameters, dynamic_parameters = particle_parameters_from_values(
            parameter_values,
            dynamic_values=dynamic_values,
        )
        species = [
            _species("beam electrons", -1.0, 2.0, 3.0, jnp.array([-1.5, 0.5])),
            _species("background ions", 1.0, 4.0, 5.0, jnp.array([1.5, -0.5])),
        ]
        tiled_particles, species_config = build_tiled_particles(species, static_parameters, dynamic_parameters)
        species_names = names_for_species(species)

        snapshot = async_writer.make_tiled_particle_snapshot(
            tiled_particles,
            step=2,
            time=3 * float(dynamic_parameters.dt),
            species_names=species_names,
            species_config=species_config,
        )
        series = FakeSeries()
        with patch.object(openPMD, "_open_openpmd_series", return_value=series):
            openPMD.write_tiled_particle_snapshot_openpmd(
                snapshot,
                output_dir="/tmp",
                filename="particles",
                static_parameters=static_parameters,
                dynamic_parameters=dynamic_parameters,
                file_extension=".h5",
            )

        iteration = series.iterations[2]
        self.assertEqual(iteration.time, 3 * float(dynamic_parameters.dt))
        dt, velocity, C = float(dynamic_parameters.dt), 0.1, 10.0
        gamma = 1.0 / np.sqrt(1.0 - velocity**2 / C**2)
        for name, charge, mass, weight, x1 in (("beam_electrons", -1.0, 2.0, 3.0, [-1.5, 0.5]),
                                               ("background_ions", 1.0, 4.0, 5.0, [1.5, -0.5])):
            with self.subTest(species=name):
                group = iteration.particles[name]
                # flat particles store u^{n+1/2}; positions are moved back half a step
                np.testing.assert_allclose(np.sort(_record_data(group["position"]["x"])),
                                           np.sort(np.array(x1) - velocity * dt / 2))
                np.testing.assert_allclose(_record_data(group["position"]["y"]), 0.0)
                np.testing.assert_allclose(_record_data(group["positionOffset"]["x"]), 0.0)
                np.testing.assert_allclose(_record_data(group["momentum"]["x"]), mass * gamma * velocity)
                np.testing.assert_allclose(_record_data(group["momentum"]["z"]), 0.0)
                np.testing.assert_allclose(_record_data(group["gamma"]), gamma)
                np.testing.assert_allclose(_record_data(group["weighting"]), weight)
                np.testing.assert_allclose(_record_data(group["charge"]), charge)
                np.testing.assert_allclose(_record_data(group["mass"]), mass)

    def test_async_particle_writer_queue_size_caps_pending_snapshots(self):
        static_parameters, dynamic_parameters = particle_parameters_from_values(
            _parameter_values(),
            dynamic_values={"C": 10.0},
        )
        writer = async_writer.AsyncTiledOpenPMDParticleWriter(
            output_dir="/tmp",
            filename="particles",
            static_parameters=static_parameters,
            dynamic_parameters=dynamic_parameters,
            queue_size=1,
        )
        snapshot = async_writer.TiledParticleSnapshot(
            step=0,
            time=0.0,
            species_names=("electrons",),
            x_shards=[],
            u_shards=[],
            active_shards=[],
            species_charge=jnp.array([-1.0]),
            species_mass=jnp.array([1.0]),
            species_weight=jnp.array([1.0]),
        )

        self.assertTrue(writer.enqueue(snapshot, block=False))
        self.assertFalse(writer.enqueue(snapshot, block=False))
        writer.close(raise_errors=False)

    def test_async_particle_writer_raises_worker_errors_on_close(self):
        static_parameters, dynamic_parameters = particle_parameters_from_values(
            _parameter_values(),
            dynamic_values={"C": 10.0},
        )
        writer = async_writer.AsyncTiledOpenPMDParticleWriter(
            output_dir="/tmp",
            filename="particles",
            static_parameters=static_parameters,
            dynamic_parameters=dynamic_parameters,
            queue_size=1,
        )
        snapshot = async_writer.TiledParticleSnapshot(
            step=0,
            time=0.0,
            species_names=("electrons",),
            x_shards=[],
            u_shards=[],
            active_shards=[],
            species_charge=jnp.array([-1.0]),
            species_mass=jnp.array([1.0]),
            species_weight=jnp.array([1.0]),
        )

        with patch.object(async_writer, "write_tiled_particle_snapshot_openpmd", side_effect=RuntimeError("disk failed")):
            writer.start()
            self.assertTrue(writer.enqueue(snapshot))
            with self.assertRaisesRegex(RuntimeError, "Async openPMD particle writer failed"):
                writer.close()

    def test_field_and_particle_async_writers_serialize_openpmd_calls(self):
        static_parameters, dynamic_parameters = kernel_parameters_from_values(
            _parameter_values(),
            dynamic_values={"C": 10.0},
        )
        field_writer = async_writer.AsyncTiledOpenPMDFieldWriter(
            output_dir="/tmp",
            filename="fields",
            static_parameters=static_parameters,
            dynamic_parameters=dynamic_parameters,
            global_shape=(int(dynamic_parameters.Nx), int(dynamic_parameters.Ny), int(dynamic_parameters.Nz)),
            tile_shape=tuple(int(width) for width in static_parameters.tile_shape),
            guard_cells=int(static_parameters.guard_cells),
            queue_size=1,
        )
        particle_writer = async_writer.AsyncTiledOpenPMDParticleWriter(
            output_dir="/tmp",
            filename="particles",
            static_parameters=static_parameters,
            dynamic_parameters=dynamic_parameters,
            queue_size=1,
        )
        field_snapshot = async_writer.TiledFieldSnapshot(step=0, time=0.0, fields={})
        particle_snapshot = async_writer.TiledParticleSnapshot(
            step=0,
            time=0.0,
            species_names=("electrons",),
            x_shards=[],
            u_shards=[],
            active_shards=[],
            species_charge=jnp.array([-1.0]),
            species_mass=jnp.array([1.0]),
            species_weight=jnp.array([1.0]),
        )

        active_writes = 0
        overlaps = []
        counter_lock = threading.Lock()
        field_write_started = threading.Event()

        def write_snapshot(*args, **kwargs):
            nonlocal active_writes
            with counter_lock:
                active_writes += 1
                if active_writes > 1:
                    overlaps.append(active_writes)
            field_write_started.set()
            time.sleep(0.05)
            with counter_lock:
                active_writes -= 1

        with (
            patch.object(async_writer, "write_tiled_field_snapshot_openpmd", side_effect=write_snapshot),
            patch.object(async_writer, "write_tiled_particle_snapshot_openpmd", side_effect=write_snapshot),
        ):
            field_writer.start()
            particle_writer.start()
            self.assertTrue(field_writer.enqueue(field_snapshot))
            self.assertTrue(field_write_started.wait(timeout=1.0))
            self.assertTrue(particle_writer.enqueue(particle_snapshot))
            field_writer.close()
            particle_writer.close()

        self.assertEqual(overlaps, [])

    def test_write_openpmd_initial_particles_flattens_tiled_particles_and_preserves_names(self):
        static_parameters, dynamic_parameters = particle_parameters_from_values(
            _parameter_values(),
            dynamic_values={"C": 10.0},
        )
        species = [
            _species("beam electrons", -1.0, 2.0, 3.0, jnp.array([-1.5, 0.5])),
            _species("background ions", 1.0, 4.0, 5.0, jnp.array([1.5])),
        ]
        tiled_particles, species_config = build_tiled_particles(species, static_parameters, dynamic_parameters)
        species_names = names_for_species(species)
        series = FakeSeries()

        with patch.object(openPMD.io, "Series", return_value=series):
            openPMD.write_openpmd_initial_particles(
                tiled_particles,
                static_parameters,
                dynamic_parameters,
                "/tmp",
                species_config=species_config,
                species_names=species_names,
            )

        iteration = series.iterations[0]
        electron_group = iteration.particles["beam_electrons"]
        ion_group = iteration.particles["background_ions"]
        self.assertEqual(electron_group["position"]["x"].shape, (2,))
        self.assertEqual(ion_group["position"]["x"].shape, (1,))
        self.assertTrue(jnp.allclose(electron_group["weighting"].data, jnp.array([3.0, 3.0])))
        self.assertTrue(jnp.allclose(ion_group["charge"].data, jnp.array([1.0])))
        expected_electron_mass = jnp.array([2.0, 2.0])
        self.assertTrue(jnp.allclose(electron_group["mass"].data, expected_electron_mass))
        expected_gamma = 1.0 / jnp.sqrt(1.0 - 0.1**2 / 10.0**2)
        expected_momentum_x = expected_electron_mass * 0.1 * expected_gamma
        self.assertTrue(jnp.allclose(electron_group["momentum"]["x"].data, expected_momentum_x))


    def test_static_metric_snapshot_writes_covariant_gamma_momentum_and_stored_position(self):
        static, dynamic = kernel_parameters(
            Nx=8, Ny=8, Nz=1, x_min=1., y_min=.5, z_min=0.,
            x_wind=1., y_wind=1., z_wind=1., dt=.005,
            solver='static_metric', particle_pusher='hybrid_boris_geodesic',
            metric='numerical', tile_shape=(4, 8, 1),
        )
        gamma_metric = jnp.array([[2., .2, .1], [.2, 3., -.1], [.1, -.1, 1.5]])
        # a uniform metric is reproduced exactly by the Hermite interpolant

        def provider(position):
            return 1., jnp.zeros(3), gamma_metric, jnp.linalg.inv(gamma_metric), jnp.sqrt(jnp.linalg.det(gamma_metric))

        metric = build_yee_metric(dynamic, provider)
        # one live particle in each of the two x tiles, plus an inactive slot
        x = jnp.array([[[1.2, .8, 0.], [1.3, .9, 0.]], [[1.7, 1.1, 0.], [0., 0., 0.]]]).reshape(2, 1, 1, 1, 2, 3)
        u = jnp.array([[[.6, .2, -.1], [0., 0., 0.]], [[-2., 1., 3.], [5., 5., 5.]]]).reshape(2, 1, 1, 1, 2, 3)
        active = jnp.array([[True, False], [True, False]]).reshape(2, 1, 1, 1, 2)
        particles = TiledParticles(x, u, active)
        species = SpeciesConfig(jnp.array([-1.]), jnp.array([2.]), jnp.array([3.]), jnp.ones((1, 3), bool))

        gamma = particle_lorentz_factor(particles, metric, static, dynamic)
        snapshot = async_writer.make_tiled_particle_snapshot(
            particles, step=0, time=0., species_names=('electrons',), species_config=species, gamma=gamma)
        iteration = FakeIteration()
        openPMD.write_tiled_particle_snapshot_to_iteration(iteration, snapshot, static, dynamic)

        group = iteration.particles['electrons']
        x_live = np.array([[1.2, .8, 0.], [1.7, 1.1, 0.]])
        u_live = np.array([[.6, .2, -.1], [-2., 1., 3.]])
        expected_gamma = np.sqrt(1. + np.einsum('ni,ij,nj->n', u_live, np.linalg.inv(gamma_metric), u_live))
        written_gamma = np.concatenate([data for _, _, data in group['gamma'].chunks])
        np.testing.assert_allclose(written_gamma, expected_gamma, rtol=1e-6)
        self.assertTrue(np.all(written_gamma > 1.))
        for axis, component in enumerate(('x', 'y', 'z')):
            momentum = np.concatenate([data for _, _, data in group['momentum'][component].chunks])
            position = np.concatenate([data for _, _, data in group['position'][component].chunks])
            np.testing.assert_allclose(momentum, 2. * u_live[:, axis], rtol=1e-6)
            np.testing.assert_allclose(position, x_live[:, axis], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
