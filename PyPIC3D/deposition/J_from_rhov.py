from PyPIC3D.particles.particle_class import TiledParticles, SpeciesConfig

from PyPIC3D.boundary_conditions.grid_and_stencil import (
    collapse_axis_stencil,
    prepare_particle_axis_stencil,
)

from PyPIC3D.deposition.shapes import get_first_order_weights, get_second_order_weights
from PyPIC3D.boundary_conditions.ghost_cells import (
    fold_tiled_vector_ghost_cells,
    update_tiled_vector_ghost_cells,
)

from PyPIC3D.utilities.filters import (
    bilinear_filter_vector,
    digital_filter_vector,
)

import jax
import jax.numpy as jnp
from functools import partial

def _collapse_tiled_axis_stencil(points, weights, local_n, reduced_axis, g):
    if reduced_axis:
        collapsed_points = jnp.full((1, points.shape[1]), int(g), dtype=points.dtype)
        collapsed_weights = jnp.sum(weights, axis=0, keepdims=True)
        return collapsed_points, collapsed_weights
    return collapse_axis_stencil(points, weights, local_n, ghost_cells=True)


@partial(jax.jit, static_argnames="static_parameters")
def J_from_rhov(
    particles,
    species_config,
    J,
    static_parameters,
    dynamic_parameters,
):
    """Compute tile-local direct current from centered tiled particles."""


    current_filter = static_parameters.current_filter
    tile_shape = tuple(int(width) for width in static_parameters.tile_shape)
    g = int(static_parameters.guard_cells)
    g = int(g)
    # determine the number of guard cells and the shape of each of the tiles

    grid = dynamic_parameters.grids.center
    # use one global coordinate origin so stencil anchors are independent of tile decomposition

    dx = dynamic_parameters.dx
    dy = dynamic_parameters.dy
    dz = dynamic_parameters.dz
    # get the grid spacing

    Jx_tiles, Jy_tiles, Jz_tiles = J
    # unpack the current density tiles

    ntx, nty, ntz = Jx_tiles.shape[:3]
    # get the number of tiles in each dimension
    tile_nx, tile_ny, tile_nz = tile_shape
    # unpack the tile shape
    local_Nx = tile_nx + 2 * g
    local_Ny = tile_ny + 2 * g
    local_Nz = tile_nz + 2 * g
    # piece together the total local tile shape

    shape_factor = static_parameters.shape_factor
    # get the shape factor

    reduced_x = int(tile_nx) == 1 and int(ntx) == 1
    reduced_y = int(tile_ny) == 1 and int(nty) == 1
    reduced_z = int(tile_nz) == 1 and int(ntz) == 1
    # determine if any of the axes are dummy axes

    Jx_template = jnp.zeros_like(Jx_tiles[0, 0, 0])
    Jy_template = jnp.zeros_like(Jy_tiles[0, 0, 0])
    Jz_template = jnp.zeros_like(Jz_tiles[0, 0, 0])
    # build template tiles

    # Tile boundaries are not physical boundaries.  Deposits that cross a tile
    # edge should land in tile ghost cells and be exchanged by the tiled fold.
    local_bc = 2

    species_weighted_charge = species_config.charge * species_config.weight
    # compute the weighted charge for each species

    def deposit_one_tile(x_tile, u_tile, active_tile, tx, ty, tz):
        # deposit the current density for a single tile, given the particle positions, velocities, and active mask
        x = x_tile[..., 0].reshape(-1)
        y = x_tile[..., 1].reshape(-1)
        z = x_tile[..., 2].reshape(-1)
        # reshape the particle positions into 1D arrays for processing
        vx = u_tile[..., 0].reshape(-1)
        vy = u_tile[..., 1].reshape(-1)
        vz = u_tile[..., 2].reshape(-1)
        # reshape the particle velocities into 1D arrays for processing
        update_x1 = jnp.broadcast_to(species_config.update_x[:, 0, jnp.newaxis], active_tile.shape).reshape(-1)
        update_x2 = jnp.broadcast_to(species_config.update_x[:, 1, jnp.newaxis], active_tile.shape).reshape(-1)
        update_x3 = jnp.broadcast_to(species_config.update_x[:, 2, jnp.newaxis], active_tile.shape).reshape(-1)
        # broadcast the directional species masks to particle slots
        active = active_tile.reshape(-1).astype(x.dtype)
        # reshape the active particle mask into a 1D array for processing
        q = jnp.broadcast_to(species_weighted_charge[:, jnp.newaxis], active_tile.shape).reshape(-1)
        # reshape the particle charges into a 1D array for processing
        dq = q / (dx * dy * dz)
        # compute the charge density contribution of each particle

        x_grid, y_grid, z_grid = grid
        # all tiles calculate weights from the same global grid coordinates

        x, _, deltax_node, xpts_node = prepare_particle_axis_stencil(
            x,
            x_grid,
            x_grid.shape[0],
            shape_factor,
            local_bc,
            wind=tile_nx * dx,
            ghost_cells=True,
        )
        _, _, deltax_face, xpts_face = prepare_particle_axis_stencil(
            x,
            x_grid + 0.5 * dx,
            x_grid.shape[0],
            shape_factor,
            local_bc,
            wind=tile_nx * dx,
            ghost_cells=True,
        )
        y, _, deltay_node, ypts_node = prepare_particle_axis_stencil(
            y,
            y_grid,
            y_grid.shape[0],
            shape_factor,
            local_bc,
            wind=tile_ny * dy,
            ghost_cells=True,
        )
        _, _, deltay_face, ypts_face = prepare_particle_axis_stencil(
            y,
            y_grid + 0.5 * dy,
            y_grid.shape[0],
            shape_factor,
            local_bc,
            wind=tile_ny * dy,
            ghost_cells=True,
        )
        z, _, deltaz_node, zpts_node = prepare_particle_axis_stencil(
            z,
            z_grid,
            z_grid.shape[0],
            shape_factor,
            local_bc,
            wind=tile_nz * dz,
            ghost_cells=True,
        )
        _, _, deltaz_face, zpts_face = prepare_particle_axis_stencil(
            z,
            z_grid + 0.5 * dz,
            z_grid.shape[0],
            shape_factor,
            local_bc,
            wind=tile_nz * dz,
            ghost_cells=True,
        )
        # node- and face-centered quantities need independent anchors and stencil points

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
        # compute the weights for the node-centered and face-centered contributions based on the shape factor and deltas

        xpts_node = jnp.asarray(xpts_node)
        ypts_node = jnp.asarray(ypts_node)
        zpts_node = jnp.asarray(zpts_node)
        xpts_face = jnp.asarray(xpts_face)
        ypts_face = jnp.asarray(ypts_face)
        zpts_face = jnp.asarray(zpts_face)
        x_local_offset = tx * tile_nx - (g - 1)
        y_local_offset = ty * tile_ny - (g - 1)
        z_local_offset = tz * tile_nz - (g - 1)
        xpts_node = xpts_node - x_local_offset
        xpts_face = xpts_face - x_local_offset
        ypts_node = ypts_node - y_local_offset
        ypts_face = ypts_face - y_local_offset
        zpts_node = zpts_node - z_local_offset
        zpts_face = zpts_face - z_local_offset
        # translate global stencil ownership into the compact tile-local arrays
        x_weights_node = jnp.asarray(x_weights_node)
        y_weights_node = jnp.asarray(y_weights_node)
        z_weights_node = jnp.asarray(z_weights_node)
        x_weights_face = jnp.asarray(x_weights_face)
        y_weights_face = jnp.asarray(y_weights_face)
        z_weights_face = jnp.asarray(z_weights_face)
        # convert the stencil points and weights to JAX arrays for further processing

        xpts_node, x_weights_node = _collapse_tiled_axis_stencil(
            xpts_node, x_weights_node, local_Nx, reduced_x, g
        )
        xpts_face, x_weights_face = _collapse_tiled_axis_stencil(
            xpts_face, x_weights_face, local_Nx, reduced_x, g
        )
        ypts_node, y_weights_node = _collapse_tiled_axis_stencil(
            ypts_node, y_weights_node, local_Ny, reduced_y, g
        )
        ypts_face, y_weights_face = _collapse_tiled_axis_stencil(
            ypts_face, y_weights_face, local_Ny, reduced_y, g
        )
        zpts_node, z_weights_node = _collapse_tiled_axis_stencil(
            zpts_node, z_weights_node, local_Nz, reduced_z, g
        )
        zpts_face, z_weights_face = _collapse_tiled_axis_stencil(
            zpts_face, z_weights_face, local_Nz, reduced_z, g
        )
        # collapse the stencil points and weights for each axis, taking into account any reduced axes and guard cells

        tile_Jx = Jx_template
        tile_Jy = Jy_template
        tile_Jz = Jz_template

        for i in range(xpts_node.shape[0]):
            for j in range(ypts_node.shape[0]):
                for k in range(zpts_node.shape[0]):
                    tile_Jx = tile_Jx.at[xpts_face[i], ypts_node[j], zpts_node[k]].add(
                        active * update_x1 * dq * vx * x_weights_face[i] * y_weights_node[j] * z_weights_node[k],
                        mode="drop",
                    )
                    tile_Jy = tile_Jy.at[xpts_node[i], ypts_face[j], zpts_node[k]].add(
                        active * update_x2 * dq * vy * x_weights_node[i] * y_weights_face[j] * z_weights_node[k],
                        mode="drop",
                    )
                    tile_Jz = tile_Jz.at[xpts_node[i], ypts_node[j], zpts_face[k]].add(
                        active * update_x3 * dq * vz * x_weights_node[i] * y_weights_node[j] * z_weights_face[k],
                        mode="drop",
                    )

        return tile_Jx, tile_Jy, tile_Jz

    tx, ty, tz = jnp.meshgrid(
        jnp.arange(ntx),
        jnp.arange(nty),
        jnp.arange(ntz),
        indexing="ij",
    )
    # build the tile index arrays for each dimension

    deposit_tiles = deposit_one_tile
    deposit_tiles = jax.vmap(deposit_tiles, in_axes=(0, 0, 0, 0, 0, 0), out_axes=0)
    deposit_tiles = jax.vmap(deposit_tiles, in_axes=(0, 0, 0, 0, 0, 0), out_axes=0)
    deposit_tiles = jax.vmap(deposit_tiles, in_axes=(0, 0, 0, 0, 0, 0), out_axes=0)
    # vectorize the deposit_one_tile function over the tile indices using jax.vmap

    Jx, Jy, Jz = deposit_tiles(
        particles.x,
        particles.u,
        particles.active,
        tx,
        ty,
        tz,
    )
    # compute the current density contributions for all tiles by applying the vectorized deposit function to the particle data and tile indices

    J = fold_tiled_vector_ghost_cells((Jx, Jy, Jz), static_parameters, g, bc_type=1)
    # fold the ghost cells of the current density tiles to ensure continuity across tile boundaries
    J = update_tiled_vector_ghost_cells(J, static_parameters, g, bc_type=1)
    # update the ghost cells of the current density tiles to reflect the contributions from neighboring tiles



    ################# CURRENT FILTERING #################
    def bilinear_filtered_current(J):
        J = bilinear_filter_vector(J, num_guard_cells=g)
        J = update_tiled_vector_ghost_cells(J, static_parameters, num_guard_cells=g, bc_type=1)
        return J
    
    def digital_filtered_current(J):
        J = digital_filter_vector(J, dynamic_parameters.alpha, num_guard_cells=g)
        J = update_tiled_vector_ghost_cells(J, static_parameters, num_guard_cells=g, bc_type=1)
        return J


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
    # apply current filtering


    
    return J
