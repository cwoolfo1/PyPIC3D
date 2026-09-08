from functools import partial

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.ghost_cells import (
    BC_TYPE_PARTICLE,
    fold_tiled_vector_ghost_cells,
    update_tiled_vector_ghost_cells,
)
from PyPIC3D.deposition.Esirkepov import esirkepov_tile_currents
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles


__all__ = ["GR_Esirkepov_current"]


@partial(jax.jit, static_argnames="static_parameters")
def GR_Esirkepov_current(
    particles_old: TiledParticles,
    particles_new: TiledParticles,
    species_config: SpeciesConfig,
    J,
    metric,
    static_parameters,
    dynamic_parameters,
):
    """
    Charge-conserving Esirkepov current deposition for a fixed 3+1 metric.

    The scheme deposits the conformal current ``sqrt(gamma) J^i`` and returns the
    physical contravariant current ``J^i`` that ``update_D_relativity`` consumes,
    matching the convention of ``GR_direct_deposition``.

    Unlike the direct deposit, this one satisfies the discrete continuity
    equation exactly.  The conformal charge density carries no metric,

        sqrt(gamma) rho = q S(x) / (dx dy dz),

    so the conformal continuity equation

        d_t( sqrt(gamma) rho ) + d_i( sqrt(gamma) J^i ) = 0

    is the flat Esirkepov identity verbatim and the ordinary density
    decomposition applies unchanged.  Because the backward-difference divergence
    of the backward-difference curl in ``update_D_relativity`` vanishes
    identically, satisfying that equation preserves

        d_i( sqrt(gamma) D^i ) = 4 pi sqrt(gamma) rho

    to round-off, with no divergence cleaning.

    ``particles_old`` holds ``x^n`` and ``particles_new`` holds ``x^{n+1}``.
    Both must be supplied before the full-step retile, so that they share one
    tile frame and neither has been wrapped by the periodic boundary -- the
    deposition differences the two positions directly and a wrap would appear as
    a domain-sized displacement.  Passing the positions explicitly is required:
    ``particles.u`` stores covariant ``u_i``, so the flat shortcut
    ``x^n = x^{n+1} - dt u^{n+1/2}`` does not hold in a curved chart.

    No current filter is applied.  Filtering destroys the exact continuity that
    is the entire purpose of the scheme, so the configuration layer rejects it.
    """

    tile_shape = tuple(int(width) for width in static_parameters.tile_shape)
    g = int(static_parameters.guard_cells)
    tiled_grid = dynamic_parameters.grids.tiled_center_grid

    dx = dynamic_parameters.dx
    dy = dynamic_parameters.dy
    dz = dynamic_parameters.dz
    dt = dynamic_parameters.dt
    shape_factor = static_parameters.shape_factor

    Jx, Jy, Jz = J
    ntx, nty, ntz = Jx.shape[:3]
    tile_nx, tile_ny, tile_nz = tile_shape
    local_Nx = tile_nx + 2 * g
    local_Ny = tile_ny + 2 * g
    local_Nz = tile_nz + 2 * g

    x_active = ntx * tile_nx > 1
    y_active = nty * tile_ny > 1
    z_active = ntz * tile_nz > 1
    # determine which axes are actually active and which ones are redundant

    Jx_template = jnp.zeros_like(Jx[0, 0, 0])
    Jy_template = jnp.zeros_like(Jy[0, 0, 0])
    Jz_template = jnp.zeros_like(Jz[0, 0, 0])
    # build a template array for the local J tiles

    species_weighted_charge = species_config.charge * species_config.weight
    # compute the species weighted charge

    def deposit_one_tile(x_old_tile, x_new_tile, active_tile, tx, ty, tz):
        old_x = x_old_tile[..., 0].reshape(-1)
        old_y = x_old_tile[..., 1].reshape(-1)
        old_z = x_old_tile[..., 2].reshape(-1)
        # positions at time level n, before the geodesic position update

        new_x = x_new_tile[..., 0].reshape(-1)
        new_y = x_new_tile[..., 1].reshape(-1)
        new_z = x_new_tile[..., 2].reshape(-1)
        # positions at time level n+1, after the push but before the retile

        active = active_tile.reshape(-1).astype(old_x.dtype)
        q = jnp.broadcast_to(species_weighted_charge[:, jnp.newaxis], active_tile.shape).reshape(-1)
        update_x1 = jnp.broadcast_to(species_config.update_x[:, 0, jnp.newaxis], active_tile.shape).reshape(-1)
        update_x2 = jnp.broadcast_to(species_config.update_x[:, 1, jnp.newaxis], active_tile.shape).reshape(-1)
        update_x3 = jnp.broadcast_to(species_config.update_x[:, 2, jnp.newaxis], active_tile.shape).reshape(-1)
        # determine which axes are updated for each particle

        x = jnp.where(update_x1, new_x, old_x)
        y = jnp.where(update_x2, new_y, old_y)
        z = jnp.where(update_x3, new_z, old_z)
        # hold frozen axes fixed, so a species pinned on an axis deposits no
        # displacement there even if the pusher wrote one

        vx = (x - old_x) / dt
        vy = (y - old_y) / dt
        vz = (z - old_z) / dt
        # coordinate velocity dx^i/dt taken straight from the displacement the
        # position update actually produced.  Only the inactive axes read this,
        # where the component takes no part in the continuity equation.  Using
        # the displacement rather than alpha v^i - beta^i keeps the whole kernel
        # free of metric interpolation at particle positions, and reduces to
        # u^i identically in flat space.

        return esirkepov_tile_currents(
            (x, y, z),
            (old_x, old_y, old_z),
            (vx, vy, vz),
            q,
            active,
            (update_x1, update_x2, update_x3),
            (
                tiled_grid[0][tx, ty, tz],
                tiled_grid[1][tx, ty, tz],
                tiled_grid[2][tx, ty, tz],
            ),
            (x_active, y_active, z_active),
            (local_Nx, local_Ny, local_Nz),
            (Jx_template, Jy_template, Jz_template),
            shape_factor,
            dx,
            dy,
            dz,
            dt,
        )
        # the decomposition itself is metric-free and shared with the flat solver

    tx, ty, tz = jnp.meshgrid(
        jnp.arange(ntx),
        jnp.arange(nty),
        jnp.arange(ntz),
        indexing="ij",
    )
    # build a meshgrid of tile indices to pass to the deposit function

    deposit_tiles = deposit_one_tile
    deposit_tiles = jax.vmap(deposit_tiles, in_axes=(0, 0, 0, 0, 0, 0), out_axes=0)
    deposit_tiles = jax.vmap(deposit_tiles, in_axes=(0, 0, 0, 0, 0, 0), out_axes=0)
    deposit_tiles = jax.vmap(deposit_tiles, in_axes=(0, 0, 0, 0, 0, 0), out_axes=0)
    # nested vmap to vectorize the deposit function over the tile indices

    Jx, Jy, Jz = deposit_tiles(
        particles_old.x,
        particles_new.x,
        particles_new.active,
        tx,
        ty,
        tz,
    )
    # the push does not change particle activity, so either set's mask will do

    conformal_J = fold_tiled_vector_ghost_cells(
        (Jx, Jy, Jz),
        static_parameters,
        num_guard_cells=g,
        bc_type=BC_TYPE_PARTICLE,
    )
    # fold charge deposited into tile ghost cells back into the owner interiors

    conformal_J = update_tiled_vector_ghost_cells(
        conformal_J,
        static_parameters,
        num_guard_cells=g,
        bc_type=BC_TYPE_PARTICLE,
    )
    # refresh the halos so the folded current is consistent across tiles

    return tuple(
        conformal_J[i] / metric.D[i].sqrt_gamma
        for i in range(3)
    )
    # convert the conformal current to the physical contravariant current.  This
    # divides by the same sqrt(gamma) array that update_D_relativity multiplies
    # back in, so the round trip is exact and the quantity Esirkepov conserved is
    # precisely the one the D update differences.
