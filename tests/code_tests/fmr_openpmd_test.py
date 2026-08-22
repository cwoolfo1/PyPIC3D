import os
import tempfile
import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import openpmd_api as io

from PyPIC3D.diagnostics.async_writer import AsyncFMROpenPMDFieldWriter
from PyPIC3D.diagnostics import fmr_openpmd
from PyPIC3D.solvers.yee.fmr import load_fmr_levels


jax.config.update("jax_enable_x64", True)


def _fmr_levels():
    config = {
        "fmr": {
            "enabled": True,
            "levels": [{
                "parent": 0,
                "refinement_ratio": 2,
                "coarse_start": [1, 1, 1],
                "coarse_stop": [4, 4, 4],
            }],
        }
    }
    geometry = {
        "Nx": 5,
        "Ny": 5,
        "Nz": 5,
        "dx": 0.5,
        "dy": 1.0,
        "dz": 1.5,
        "x_min": 2.0,
        "x_max": 4.5,
        "y_min": -3.0,
        "y_max": 2.0,
        "z_min": 5.0,
        "z_max": 12.5,
    }
    return load_fmr_levels(config, geometry, root_tile_shape=(5, 5, 5))


def _component(level, guard_cells, value_offset):
    tile_grid = tuple(
        cells // tile_cells
        for cells, tile_cells in zip(level.shape, level.tile_shape)
    )
    shape = (
        *tile_grid,
        *(int(cells) + 2*guard_cells for cells in level.tile_shape),
    )
    values = jnp.arange(np.prod(shape), dtype=jnp.float64).reshape(shape)
    return values + float(value_offset)


def _field_map(levels, guard_cells=2):
    field_map = {}
    for field_index, name in enumerate(("E", "B", "J")):
        field_map[name] = tuple(
            tuple(
                _component(level, guard_cells, 10000*field_index + 1000*level.index + component)
                for component in range(3)
            )
            for level in levels
        )
    return field_map


def _read_patch(path, step):
    series = io.Series(path, io.Access.read_only)
    return series, series.iterations[int(step)]


class FMROpenPMDTests(unittest.TestCase):

    def _write_step(self, output_dir, levels, field_map, step, physical_time):
        static_parameters = SimpleNamespace(fmr_levels=levels, guard_cells=2)
        dynamic_parameters = SimpleNamespace(dt=0.1)
        writer = AsyncFMROpenPMDFieldWriter(
            output_dir=output_dir,
            static_parameters=static_parameters,
            dynamic_parameters=dynamic_parameters,
        )
        writer.start()
        self.assertTrue(
            writer.enqueue_fields(
                field_map,
                step=step,
                time=physical_time,
            )
        )
        writer.close()

    def test_fmr_patch_series_preserves_geometry_staggering_and_field_data(self):
        levels = _fmr_levels()
        patches = fmr_openpmd.build_fmr_patch_descriptors(levels, guard_cells=2)
        field_map = _field_map(levels)
        original_fields = tuple(
            np.array(component)
            for field_levels in field_map.values()
            for level in field_levels
            for component in level
        )

        with tempfile.TemporaryDirectory() as output_dir:
            self._write_step(output_dir, levels, field_map, step=0, physical_time=0.0)

            mesh_names = []
            for patch in patches:
                level = patch.level
                filename = fmr_openpmd.fmr_iteration_filename(patch, 0)
                series, iteration = _read_patch(os.path.join(output_dir, filename), step=0)
                mesh_names.append(set(iteration.meshes))

                self.assertEqual(mesh_names[-1], {"E", "B", "J"})
                self.assertEqual(int(iteration.get_attribute("fmrLevel")), level.index)
                self.assertEqual(int(iteration.get_attribute("fmrPatch")), 0)
                self.assertEqual(int(iteration.get_attribute("fmrParent")), level.parent)
                self.assertEqual(
                    int(iteration.get_attribute("refinementRatio")),
                    level.refinement_ratio,
                )
                np.testing.assert_array_equal(
                    iteration.get_attribute("coarseStart"),
                    level.parent_start,
                )
                np.testing.assert_array_equal(
                    iteration.get_attribute("coarseStop"),
                    level.parent_stop,
                )

                for name in ("E", "B", "J"):
                    mesh = iteration.meshes[name]
                    self.assertEqual(list(mesh.axis_labels), ["x", "y", "z"])
                    self.assertEqual(mesh.data_order, "C")
                    self.assertEqual(mesh.geometry, io.Geometry.cartesian)
                    self.assertEqual(float(mesh.grid_unit_SI), 1.0)
                    np.testing.assert_allclose(mesh.grid_spacing, level.spacing, rtol=0.0, atol=0.0)
                    np.testing.assert_allclose(
                        mesh.grid_global_offset,
                        level.lower,
                        rtol=0.0,
                        atol=0.0,
                    )

                    locations = (
                        fmr_openpmd.B_FIELD_LOCATIONS
                        if name == "B"
                        else fmr_openpmd.E_FIELD_LOCATIONS
                    )
                    for component_name, location in zip(("x", "y", "z"), locations):
                        record = mesh[component_name]
                        self.assertEqual(
                            list(record.position),
                            fmr_openpmd.fmr_component_position(location),
                        )
                        self.assertEqual(
                            tuple(record.shape),
                            level.shape,
                        )

                source = field_map["E"][level.index][0]
                expected = np.asarray(source[0, 0, 0, 2:-2, 2:-2, 2:-2])
                stored = iteration.meshes["E"]["x"].load_chunk()
                series.flush()
                np.testing.assert_array_equal(np.asarray(stored), expected)
                series.close()

            self.assertEqual(mesh_names[0], mesh_names[1])
            np.testing.assert_allclose(levels[1].spacing, np.asarray(levels[0].spacing)/2)
            np.testing.assert_allclose(
                levels[1].lower,
                np.asarray(levels[0].lower)
                + np.asarray(levels[1].parent_start)*np.asarray(levels[0].spacing),
            )

        for original, current in zip(
            original_fields,
            (
                np.asarray(component)
                for field_levels in field_map.values()
                for level in field_levels
                for component in level
            ),
        ):
            np.testing.assert_array_equal(current, original)

    def test_viewer_helpers_are_relative_ordered_and_restart_safe(self):
        levels = _fmr_levels()
        patches = fmr_openpmd.build_fmr_patch_descriptors(levels, guard_cells=2)
        field_map = _field_map(levels)

        with tempfile.TemporaryDirectory() as output_dir:
            self._write_step(output_dir, levels, field_map, step=1, physical_time=0.1)
            self._write_step(output_dir, levels, field_map, step=0, physical_time=0.0)
            self._write_step(output_dir, levels, field_map, step=0, physical_time=0.0)

            for patch in patches:
                h5_name = fmr_openpmd.fmr_iteration_filename(patch, 0)
                opmd_name = fmr_openpmd.fmr_iteration_filename(
                    patch,
                    0,
                    extension=".opmd",
                )
                self.assertTrue(os.path.islink(os.path.join(output_dir, opmd_name)))
                self.assertEqual(os.readlink(os.path.join(output_dir, opmd_name)), h5_name)

                pmd_path = os.path.join(
                    output_dir,
                    fmr_openpmd.fmr_series_name(patch) + ".pmd",
                )
                with open(pmd_path) as input_file:
                    self.assertEqual(
                        input_file.read(),
                        fmr_openpmd.fmr_series_pattern(patch) + "\n",
                    )

            manifest_path = os.path.join(output_dir, "fields.visit")
            with open(manifest_path) as input_file:
                expected = (
                    "!NBLOCKS 2\n"
                    "!TIME 0.0\n"
                    "fields_level_00_patch_000_00000000.opmd\n"
                    "fields_level_01_patch_000_00000000.opmd\n"
                    "!TIME 0.1\n"
                    "fields_level_00_patch_000_00000001.opmd\n"
                    "fields_level_01_patch_000_00000001.opmd\n"
                )
                self.assertEqual(input_file.read(), expected)

            fmr_openpmd.finalize_fmr_openpmd_viewer_step(
                output_dir,
                patches,
                step=0,
                physical_time=0.0,
            )
            with open(manifest_path) as input_file:
                self.assertEqual(input_file.read(), expected)

    def test_fmr_writer_assembles_multiple_tiled_chunks_per_patch(self):
        root, fine = _fmr_levels()
        root = root._replace(tile_shape=(1, 5, 5))
        fine = fine._replace(tile_shape=(3, 6, 6))
        levels = (root, fine)
        patches = fmr_openpmd.build_fmr_patch_descriptors(levels, guard_cells=2)
        field_map = _field_map(levels)

        with tempfile.TemporaryDirectory() as output_dir:
            self._write_step(output_dir, levels, field_map, step=0, physical_time=0.0)

            for patch in patches:
                filename = fmr_openpmd.fmr_iteration_filename(patch, 0)
                series, iteration = _read_patch(os.path.join(output_dir, filename), step=0)
                stored = iteration.meshes["E"]["x"].load_chunk()
                series.flush()

                source = np.asarray(field_map["E"][patch.level.index][0])
                tile_grid = source.shape[:3]
                expected = np.empty(patch.level.shape, dtype=source.dtype)
                for i in range(tile_grid[0]):
                    for j in range(tile_grid[1]):
                        for k in range(tile_grid[2]):
                            tile = source[i, j, k, 2:-2, 2:-2, 2:-2]
                            ni, nj, nk = tile.shape
                            expected[
                                i*ni:(i + 1)*ni,
                                j*nj:(j + 1)*nj,
                                k*nk:(k + 1)*nk,
                            ] = tile

                np.testing.assert_array_equal(np.asarray(stored), expected)
                series.close()

    def test_viewer_helpers_reject_conflicting_alias_and_topology(self):
        levels = _fmr_levels()
        patches = fmr_openpmd.build_fmr_patch_descriptors(levels, guard_cells=2)
        root_patch = patches[0]

        with tempfile.TemporaryDirectory() as output_dir:
            root_h5 = fmr_openpmd.fmr_iteration_filename(root_patch, 0)
            with open(os.path.join(output_dir, root_h5), "w"):
                pass
            root_alias = fmr_openpmd.fmr_iteration_filename(
                root_patch,
                0,
                extension=".opmd",
            )
            with open(os.path.join(output_dir, root_alias), "w"):
                pass

            with self.assertRaisesRegex(FileExistsError, "conflicting VisIt alias"):
                fmr_openpmd.ensure_fmr_visit_alias(output_dir, root_patch, 0)

        with tempfile.TemporaryDirectory() as output_dir:
            with open(os.path.join(output_dir, "fields.visit"), "w") as output_file:
                output_file.write("!NBLOCKS 3\n")

            with self.assertRaisesRegex(ValueError, "topology"):
                fmr_openpmd.update_fmr_visit_manifest(
                    output_dir,
                    patches,
                    step=0,
                    physical_time=0.0,
                )


if __name__ == "__main__":
    unittest.main()
