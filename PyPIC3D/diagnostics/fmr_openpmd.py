"""openPMD patch-series output and viewer helpers for field-only FMR."""

from dataclasses import dataclass
import importlib.metadata
import os

import numpy as np
import openpmd_api as io

from PyPIC3D.diagnostics.openPMD import (
    TiledMeshLayout,
    _ensure_openpmd_array,
    _iter_tile_chunks_from_host_shard,
)
from PyPIC3D.solvers.yee.fmr.types import B_FIELD_LOCATIONS, E_FIELD_LOCATIONS


@dataclass(frozen=True)
class FMRPatchDescriptor:
    """Geometry and tiled output layout for one FMR patch series."""

    level: object
    patch_index: int
    layout: TiledMeshLayout


def build_fmr_patch_descriptors(levels, guard_cells, dtype=np.float64):
    """Build the ordered output topology for the current one-patch levels."""

    root_level, fine_level = levels
    expected_fine_origin = tuple(
        root_origin + coarse_start * coarse_spacing
        for root_origin, coarse_start, coarse_spacing in zip(
            root_level.lower,
            fine_level.parent_start,
            root_level.spacing,
        )
    )
    if fine_level.lower != expected_fine_origin:
        raise ValueError("FMR fine-patch origin is inconsistent with its coarse bounds.")

    return tuple(
        FMRPatchDescriptor(
            level=level,
            patch_index=0,
            layout=TiledMeshLayout(
                global_shape=tuple(int(width) for width in level.shape),
                tile_shape=tuple(int(width) for width in level.tile_shape),
                guard_cells=int(guard_cells),
                active_dims=(1, 1, 1),
                dtype=dtype,
            ),
        )
        for level in levels
    )


def fmr_series_name(patch):
    return (
        f"fields_level_{int(patch.level.index):02d}_"
        f"patch_{int(patch.patch_index):03d}"
    )


def fmr_series_pattern(patch):
    return f"{fmr_series_name(patch)}_%08T.h5"


def fmr_iteration_filename(patch, step, extension=".h5"):
    return f"{fmr_series_name(patch)}_{int(step):08d}{extension}"


def _configure_fmr_mesh(mesh, level):
    mesh.geometry = io.Geometry.cartesian
    mesh.data_order = io.Data_Order.C if hasattr(io, "Data_Order") else "C"
    mesh.axis_labels = ["x", "y", "z"]
    mesh.grid_spacing = [float(spacing) for spacing in level.spacing]
    mesh.grid_global_offset = [float(origin) for origin in level.lower]
    mesh.grid_unit_SI = 1.0


def fmr_component_position(location):
    # In the live Yee grid, C is the collocated/base index and V is shifted
    # to the cell center. openPMD position is relative to the mesh cell.
    return [0.5 if axis_location == "V" else 0.0 for axis_location in location]


def _set_fmr_iteration_attributes(iteration, patch):
    level = patch.level
    iteration.set_attribute("fmrLevel", np.int64(level.index))
    iteration.set_attribute("fmrPatch", np.int64(patch.patch_index))
    iteration.set_attribute("fmrParent", np.int64(level.parent))
    iteration.set_attribute("refinementRatio", np.int64(level.refinement_ratio))
    iteration.set_attribute("coarseStart", np.asarray(level.parent_start, dtype=np.int64))
    iteration.set_attribute("coarseStop", np.asarray(level.parent_stop, dtype=np.int64))


def _reset_vector_record(iteration, name, component_name, position, patch):
    mesh = iteration.meshes[name]
    _configure_fmr_mesh(mesh, patch.level)
    record = mesh[component_name]
    record.reset_dataset(
        io.Dataset(np.dtype(patch.layout.dtype), list(patch.layout.global_shape))
    )
    record.position = fmr_component_position(position)
    record.unit_SI = 1.0
    return record


def _write_vector_chunks(iteration, name, component_host_shards, positions, patch):
    for component_name, host_shards, position in zip(
        ("x", "y", "z"),
        component_host_shards,
        positions,
    ):
        record = _reset_vector_record(
            iteration,
            name,
            component_name,
            position,
            patch,
        )
        for shard_index, shard_data in host_shards:
            for offset, tile in _iter_tile_chunks_from_host_shard(
                shard_index,
                shard_data,
                layout=patch.layout,
            ):
                tile = _ensure_openpmd_array(tile, dtype=patch.layout.dtype)
                record.store_chunk(tile, offset, list(tile.shape))


def _series_access(output_dir, patch):
    prefix = fmr_series_name(patch) + "_"
    has_existing_file = any(
        filename.startswith(prefix) and filename.endswith(".h5")
        for filename in os.listdir(output_dir)
    )
    return io.Access.append if has_existing_file else io.Access.create


def write_fmr_patch_snapshot_openpmd(snapshot, *, output_dir, patch, dt):
    """Write one guard-stripped field snapshot for one FMR patch."""

    series_path = os.path.join(output_dir, fmr_series_pattern(patch))
    series = io.Series(series_path, _series_access(output_dir, patch))
    series.set_attribute("software", "PyPIC3D")
    series.set_attribute("softwareVersion", importlib.metadata.version("PyPIC3D"))

    try:
        iteration = series.iterations[int(snapshot.step)]
        iteration.time = float(snapshot.time)
        iteration.dt = float(dt)
        iteration.time_unit_SI = 1.0
        _set_fmr_iteration_attributes(iteration, patch)

        for name, component_host_shards in snapshot.fields.items():
            positions = B_FIELD_LOCATIONS if name == "B" else E_FIELD_LOCATIONS
            _write_vector_chunks(
                iteration,
                name,
                component_host_shards,
                positions,
                patch,
            )
        series.flush()
    finally:
        series.close()


def _write_text_atomically(path, text):
    temporary_path = path + ".tmp"
    with open(temporary_path, "w") as output_file:
        output_file.write(text)
    os.replace(temporary_path, path)


def setup_fmr_openpmd_viewer_files(output_dir, patches):
    for patch in patches:
        pmd_path = os.path.join(output_dir, fmr_series_name(patch) + ".pmd")
        pmd_text = fmr_series_pattern(patch) + "\n"
        if os.path.exists(pmd_path):
            with open(pmd_path) as input_file:
                if input_file.read() == pmd_text:
                    continue
        _write_text_atomically(pmd_path, pmd_text)


def ensure_fmr_visit_alias(output_dir, patch, step):
    h5_name = fmr_iteration_filename(patch, step)
    opmd_name = fmr_iteration_filename(patch, step, extension=".opmd")
    h5_path = os.path.join(output_dir, h5_name)
    opmd_path = os.path.join(output_dir, opmd_name)

    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"Cannot create VisIt alias before patch file exists: {h5_path}")
    if os.path.lexists(opmd_path):
        if os.path.islink(opmd_path) and os.readlink(opmd_path) == h5_name:
            return opmd_name
        raise FileExistsError(f"Refusing to replace conflicting VisIt alias: {opmd_path}")

    os.symlink(h5_name, opmd_path)
    return opmd_name


def _visit_step_from_alias(alias):
    stem, extension = os.path.splitext(alias)
    if extension != ".opmd":
        raise ValueError(f"Invalid fields.visit block entry: {alias}")
    try:
        return int(stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid fields.visit block entry: {alias}") from exc


def _expected_visit_aliases(patches, step):
    return tuple(
        fmr_iteration_filename(patch, step, extension=".opmd")
        for patch in patches
    )


def _read_visit_manifest(path, patches):
    if not os.path.exists(path):
        return {}

    with open(path) as input_file:
        lines = [line.strip() for line in input_file if line.strip()]

    block_count = len(patches)
    expected_header = f"!NBLOCKS {block_count}"
    if not lines or lines[0] != expected_header:
        actual_header = lines[0] if lines else "<empty>"
        raise ValueError(
            f"Existing fields.visit topology does not match {expected_header}: {actual_header}"
        )

    entries = {}
    line_index = 1
    while line_index < len(lines):
        time_line = lines[line_index]
        if not time_line.startswith("!TIME "):
            raise ValueError(f"Expected !TIME in fields.visit, got: {time_line}")
        physical_time = float(time_line.split(maxsplit=1)[1])
        aliases = tuple(lines[line_index + 1:line_index + 1 + block_count])
        if len(aliases) != block_count:
            raise ValueError("Incomplete block group in fields.visit")

        steps = {_visit_step_from_alias(alias) for alias in aliases}
        if len(steps) != 1:
            raise ValueError("A fields.visit block group contains more than one output index")
        step = steps.pop()
        expected_aliases = _expected_visit_aliases(patches, step)
        if aliases != expected_aliases:
            raise ValueError(
                "Existing fields.visit block ordering does not match the FMR topology"
            )
        if step in entries and entries[step] != (physical_time, aliases):
            raise ValueError(f"Conflicting fields.visit entries for output index {step}")
        entries[step] = physical_time, aliases
        line_index += 1 + block_count

    return entries


def update_fmr_visit_manifest(output_dir, patches, step, physical_time):
    manifest_path = os.path.join(output_dir, "fields.visit")
    entries = _read_visit_manifest(manifest_path, patches)
    aliases = _expected_visit_aliases(patches, step)

    if int(step) in entries and entries[int(step)] != (float(physical_time), aliases):
        raise ValueError(f"Conflicting fields.visit entry for output index {int(step)}")
    entries[int(step)] = float(physical_time), aliases

    lines = [f"!NBLOCKS {len(patches)}"]
    for output_index in sorted(entries):
        time_value, block_aliases = entries[output_index]
        lines.append(f"!TIME {time_value}")
        lines.extend(block_aliases)
    _write_text_atomically(manifest_path, "\n".join(lines) + "\n")


def finalize_fmr_openpmd_viewer_step(output_dir, patches, step, physical_time):
    for patch in patches:
        ensure_fmr_visit_alias(output_dir, patch, step)
    update_fmr_visit_manifest(output_dir, patches, step, physical_time)
