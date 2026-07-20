from functools import partial

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.ghost_cells import (
    fold_tiled_vector_ghost_cells,
    update_tiled_vector_ghost_cells,
)
from PyPIC3D.boundary_conditions.grid_and_stencil import (
    collapse_axis_stencil,
    prepare_particle_axis_stencil,
)
from PyPIC3D.deposition.shapes import get_first_order_weights, get_second_order_weights
from PyPIC3D.pusher.boris import interpolate_field_to_particles
from PyPIC3D.relativity.core import Metric, contravariant_three_velocity
from PyPIC3D.utilities.filters import bilinear_filter_vector, digital_filter_vector


def _collapse_tiled_axis_stencil(points, weights, local_n, reduced_axis, g):
    if reduced_axis:
        collapsed_points = jnp.full((1, points.shape[1]), int(g), dtype=points.dtype)
        collapsed_weights = jnp.sum(weights, axis=0, keepdims=True)
        return collapsed_points, collapsed_weights
    return collapse_axis_stencil(points, weights, local_n, ghost_cells=True)


def _sample_scalar(field, x, y, z, grid, shape_factor):
    particle_shape = x.shape
    return interpolate_field_to_particles(
        field,
        x.reshape(-1),
        y.reshape(-1),
        z.reshape(-1),
        grid,
        shape_factor,
        ghost_cells=True,
    ).reshape(particle_shape)


def _sample_vector(field, x, y, z, grid, shape_factor):
    return jnp.stack(
        tuple(_sample_scalar(field[..., i], x, y, z, grid, shape_factor) for i in range(3)),
        axis=-1,
    )


def _sample_tensor(field, x, y, z, grid, shape_factor):
    rows = []
    for i in range(3):
        columns = []
        for j in range(3):
            columns.append(_sample_scalar(field[..., i, j], x, y, z, grid, shape_factor))
        rows.append(jnp.stack(tuple(columns), axis=-1))
    return jnp.stack(tuple(rows), axis=-2)


def _sample_current_metric(metric, x, y, z, grid, shape_factor):
    return Metric(
        lapse=_sample_scalar(metric.lapse, x, y, z, grid, shape_factor),
        shift=_sample_vector(metric.shift, x, y, z, grid, shape_factor),
        gamma=_sample_tensor(metric.gamma, x, y, z, grid, shape_factor),
        gamma_inv=_sample_tensor(metric.gamma_inv, x, y, z, grid, shape_factor),
        sqrt_gamma=_sample_scalar(metric.sqrt_gamma, x, y, z, grid, shape_factor),
        christoffel=metric.christoffel,
        grad_lapse=metric.grad_lapse,
        grad_shift=metric.grad_shift,
    )


def _metric_tile(metric, tx, ty, tz):
    return Metric(
        lapse=metric.lapse[tx, ty, tz],
        shift=metric.shift[tx, ty, tz],
        gamma=metric.gamma[tx, ty, tz],
        gamma_inv=metric.gamma_inv[tx, ty, tz],
        sqrt_gamma=metric.sqrt_gamma[tx, ty, tz],
        christoffel=metric.christoffel[tx, ty, tz],
        grad_lapse=metric.grad_lapse[tx, ty, tz],
        grad_shift=metric.grad_shift[tx, ty, tz],
    )


@partial(jax.jit, static_argnames="static_parameters")
def GR_direct_deposition(
    particles,
    species_config,
    J,
    metric,
    static_parameters,
    dynamic_parameters,
):
    """
    Direct current deposition for a fixed 3+1 metric.

    ``particles.u`` stores covariant spatial components ``u_i``.  The deposited
    current uses the contravariant FIDO three-velocity and the lapse scaling

        J^i = alpha(x) * (q / dV) * S(x) * v^i.
    """

    current_filter = static_parameters.current_filter
    tile_shape = tuple(int(width) for width in static_parameters.tile_shape)
    g = int(static_parameters.guard_cells)
    tiled_grid = dynamic_parameters.grids.tiled_center_grid
    dx = dynamic_parameters.dx
    dy = dynamic_parameters.dy
    dz = dynamic_parameters.dz

    Jx_tiles, Jy_tiles, Jz_tiles = J
    ntx, nty, ntz = Jx_tiles.shape[:3]
    tile_nx, tile_ny, tile_nz = tile_shape
    local_Nx = tile_nx + 2 * g
    local_Ny = tile_ny + 2 * g
    local_Nz = tile_nz + 2 * g
    shape_factor = static_parameters.shape_factor

    reduced_x = int(tile_nx) == 1 and int(ntx) == 1
    reduced_y = int(tile_ny) == 1 and int(nty) == 1
    reduced_z = int(tile_nz) == 1 and int(ntz) == 1

    Jx_template = jnp.zeros_like(Jx_tiles[0, 0, 0])
    Jy_template = jnp.zeros_like(Jy_tiles[0, 0, 0])
    Jz_template = jnp.zeros_like(Jz_tiles[0, 0, 0])
    local_bc = 2
    species_weighted_charge = species_config.charge * species_config.weight

    def deposit_one_tile(x_tile, u_tile, active_tile, tx, ty, tz):
        x = x_tile[..., 0].reshape(-1)
        y = x_tile[..., 1].reshape(-1)
        z = x_tile[..., 2].reshape(-1)
        u_cov = u_tile.reshape(-1, 3)
        active = active_tile.reshape(-1).astype(x.dtype)
        q = jnp.broadcast_to(species_weighted_charge[:, jnp.newaxis], active_tile.shape).reshape(-1)
        dq = q / (dx * dy * dz)

        x_grid = tiled_grid[0][tx, ty, tz]
        y_grid = tiled_grid[1][tx, ty, tz]
        z_grid = tiled_grid[2][tx, ty, tz]
        center_grid = (x_grid, y_grid, z_grid)

        metric_at_particles = _sample_current_metric(
            _metric_tile(metric.center, tx, ty, tz),
            x,
            y,
            z,
            center_grid,
            shape_factor,
        )
        v_con = contravariant_three_velocity(u_cov, metric_at_particles.gamma_inv)
        lapse_v = metric_at_particles.lapse[:, jnp.newaxis] * v_con
        vx = lapse_v[:, 0]
        vy = lapse_v[:, 1]
        vz = lapse_v[:, 2]

        x, x0, deltax_node, xpts = prepare_particle_axis_stencil(
            x,
            x_grid,
            local_Nx,
            shape_factor,
            local_bc,
            wind=tile_nx * dx,
            ghost_cells=True,
        )
        y, y0, deltay_node, ypts = prepare_particle_axis_stencil(
            y,
            y_grid,
            local_Ny,
            shape_factor,
            local_bc,
            wind=tile_ny * dy,
            ghost_cells=True,
        )
        z, z0, deltaz_node, zpts = prepare_particle_axis_stencil(
            z,
            z_grid,
            local_Nz,
            shape_factor,
            local_bc,
            wind=tile_nz * dz,
            ghost_cells=True,
        )

        deltax_face = (x - x_grid[0]) - (x0 + 0.5) * dx
        deltay_face = (y - y_grid[0]) - (y0 + 0.5) * dy
        deltaz_face = (z - z_grid[0]) - (z0 + 0.5) * dz

        x_weights_node, y_weights_node, z_weights_node = jax.lax.cond(
            shape_factor == 1,
            lambda _: get_first_order_weights(deltax_node, deltay_node, deltaz_node, dx, dy, dz),
            lambda _: get_second_order_weights(deltax_node, deltay_node, deltaz_node, dx, dy, dz),
            operand=None,
        )
        x_weights_face, y_weights_face, z_weights_face = jax.lax.cond(
            shape_factor == 1,
            lambda _: get_first_order_weights(deltax_face, deltay_face, deltaz_face, dx, dy, dz),
            lambda _: get_second_order_weights(deltax_face, deltay_face, deltaz_face, dx, dy, dz),
            operand=None,
        )

        xpts = jnp.asarray(xpts)
        ypts = jnp.asarray(ypts)
        zpts = jnp.asarray(zpts)
        x_weights_node = jnp.asarray(x_weights_node)
        y_weights_node = jnp.asarray(y_weights_node)
        z_weights_node = jnp.asarray(z_weights_node)
        x_weights_face = jnp.asarray(x_weights_face)
        y_weights_face = jnp.asarray(y_weights_face)
        z_weights_face = jnp.asarray(z_weights_face)

        xpts, x_weights_node = _collapse_tiled_axis_stencil(xpts, x_weights_node, local_Nx, reduced_x, g)
        _, x_weights_face = _collapse_tiled_axis_stencil(xpts, x_weights_face, local_Nx, reduced_x, g)
        ypts, y_weights_node = _collapse_tiled_axis_stencil(ypts, y_weights_node, local_Ny, reduced_y, g)
        _, y_weights_face = _collapse_tiled_axis_stencil(ypts, y_weights_face, local_Ny, reduced_y, g)
        zpts, z_weights_node = _collapse_tiled_axis_stencil(zpts, z_weights_node, local_Nz, reduced_z, g)
        _, z_weights_face = _collapse_tiled_axis_stencil(zpts, z_weights_face, local_Nz, reduced_z, g)

        tile_Jx = Jx_template
        tile_Jy = Jy_template
        tile_Jz = Jz_template

        for i in range(xpts.shape[0]):
            for j in range(ypts.shape[0]):
                for k in range(zpts.shape[0]):
                    ix = xpts[i]
                    iy = ypts[j]
                    iz = zpts[k]
                    tile_Jx = tile_Jx.at[ix, iy, iz].add(
                        active * dq * vx * x_weights_face[i] * y_weights_node[j] * z_weights_node[k],
                        mode="drop",
                    )
                    tile_Jy = tile_Jy.at[ix, iy, iz].add(
                        active * dq * vy * x_weights_node[i] * y_weights_face[j] * z_weights_node[k],
                        mode="drop",
                    )
                    tile_Jz = tile_Jz.at[ix, iy, iz].add(
                        active * dq * vz * x_weights_node[i] * y_weights_node[j] * z_weights_face[k],
                        mode="drop",
                    )

        return tile_Jx, tile_Jy, tile_Jz

    tx, ty, tz = jnp.meshgrid(
        jnp.arange(ntx),
        jnp.arange(nty),
        jnp.arange(ntz),
        indexing="ij",
    )

    deposit_tiles = deposit_one_tile
    deposit_tiles = jax.vmap(deposit_tiles, in_axes=(0, 0, 0, 0, 0, 0), out_axes=0)
    deposit_tiles = jax.vmap(deposit_tiles, in_axes=(0, 0, 0, 0, 0, 0), out_axes=0)
    deposit_tiles = jax.vmap(deposit_tiles, in_axes=(0, 0, 0, 0, 0, 0), out_axes=0)

    Jx, Jy, Jz = deposit_tiles(
        particles.x,
        particles.u,
        particles.active,
        tx,
        ty,
        tz,
    )

    J = fold_tiled_vector_ghost_cells((Jx, Jy, Jz), static_parameters, g, bc_type=1)
    J = update_tiled_vector_ghost_cells(J, static_parameters, g, bc_type=1)

    def bilinear_filtered_current(J):
        J = bilinear_filter_vector(J, num_guard_cells=g)
        return update_tiled_vector_ghost_cells(J, static_parameters, num_guard_cells=g, bc_type=1)

    def digital_filtered_current(J):
        J = digital_filter_vector(J, dynamic_parameters.alpha, num_guard_cells=g)
        return update_tiled_vector_ghost_cells(J, static_parameters, num_guard_cells=g, bc_type=1)

    J = jax.lax.cond(
        current_filter == "bilinear",
        bilinear_filtered_current,
        lambda J: jax.lax.cond(
            current_filter == "digital",
            digital_filtered_current,
            lambda J: J,
            J,
        ),
        J,
    )

    return J
