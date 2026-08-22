"""FMR grid construction and component-coordinate helpers."""

from types import SimpleNamespace

import jax.numpy as jnp

from PyPIC3D.utilities.grids import build_tiled_yee_grids, build_yee_grid
from PyPIC3D.utilities.parameters import GridParameters


SINGLE_TILE_INDEX = (0, 0, 0)


def _build_level_grids(level, guard_cells):
    dynamic_setup = SimpleNamespace(
        dx=level.spacing[0],
        dy=level.spacing[1],
        dz=level.spacing[2],
        Nx=level.shape[0],
        Ny=level.shape[1],
        Nz=level.shape[2],
        x_wind=level.upper[0] - level.lower[0],
        y_wind=level.upper[1] - level.lower[1],
        z_wind=level.upper[2] - level.lower[2],
        x_min=level.lower[0],
        y_min=level.lower[1],
        z_min=level.lower[2],
    )
    center_grid, vertex_grid = build_yee_grid(dynamic_setup)

    static_setup = SimpleNamespace(tile_shape=level.tile_shape, guard_cells=guard_cells)
    tiled_setup = SimpleNamespace(
        **dynamic_setup.__dict__,
        grids=SimpleNamespace(center=center_grid, vertex=vertex_grid),
    )
    tiled_center_grid, tiled_vertex_grid = build_tiled_yee_grids(static_setup, tiled_setup)

    return GridParameters(
        vertex=vertex_grid,
        center=center_grid,
        tiled_vertex_grid=tiled_vertex_grid,
        tiled_center_grid=tiled_center_grid,
    )


def component_coordinate_axes(grids, locations):
    """Return the three logical axes for a component on the one live tile."""

    axes = []
    for axis, location in enumerate(locations):
        tiled_grid = grids.tiled_vertex_grid if location == "V" else grids.tiled_center_grid
        axes.append(jnp.asarray(tiled_grid[axis][SINGLE_TILE_INDEX]))
    return tuple(axes)


def _coordinate_tolerance(*axes):
    dtype = jnp.result_type(*(axis.dtype for axis in axes))
    axis_scales = jnp.stack(tuple(jnp.max(jnp.abs(axis)) for axis in axes))
    scale = jnp.maximum(jnp.asarray(1.0, dtype=dtype), jnp.max(axis_scales))
    return 32.0 * jnp.finfo(dtype).eps * scale
