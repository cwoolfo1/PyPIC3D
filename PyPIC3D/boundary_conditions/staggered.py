"""Coordinate-independent endpoint walls for metric Yee fields and sources."""
from functools import partial
import jax
import jax.numpy as jnp

from .ghost_cells import make_distributed_ghost_updater, make_distributed_ghost_folder
from .grid_and_stencil import BC_CONDUCTING
from PyPIC3D.relativity.core import D_FIELD_LOCATIONS


@partial(jax.jit, static_argnames=('static', 'location', 'parity', 'fold', 'particle'))
def scalar_boundaries(value, static, location=('C', 'C', 'C'), parity=(1, 1, 1),
                      *, fold=False, particle=False):
    bc = static.particle_boundary_conditions if particle else static.boundary_conditions
    factory = make_distributed_ghost_folder if fold else make_distributed_ghost_updater
    return factory(static.field_mesh, static.tile_shape, bc, static.guard_cells,
                   reflecting_parity=parity, location=location)(value)


def source_boundaries(value, static, *, fold=False, vector=False, reflecting_parity=None):
    if not vector:
        parity = (1, 1, 1) if reflecting_parity is None else reflecting_parity
        return scalar_boundaries(value, static, parity=parity, fold=fold, particle=True)
    result = tuple(scalar_boundaries(a, static, loc,
                    tuple(-1 if axis == i else 1 for axis in range(3))
                    if reflecting_parity is None else reflecting_parity[i],
                    fold=fold, particle=True)
                   for i, (a, loc) in enumerate(zip(value, D_FIELD_LOCATIONS)))
    return jnp.stack(result) if hasattr(value, 'ndim') and value.ndim == 7 else result


@partial(jax.jit, static_argnums=(1, 2, 3))
def refresh_fields(vector, static, locations, field_kind=None):
    """Enforce conducting component parity and the configured inner layers.

    D/E use odd tangential, even normal parity; B/H use the opposite.
    The BZ spherical metric has no theta cross terms or theta shift, making
    the D/B wall conditions equivalent to tangential E=0 and normal B=0.
    """
    electric = locations == D_FIELD_LOCATIONS
    g = static.guard_cells
    result = []
    for i, (value, loc) in enumerate(zip(vector, locations)):
        cells = static.horizon_field_cells
        if cells and field_kind in ('D', 'B'):
            if cells >= static.tile_shape[0]:
                raise ValueError('Horizon boundary reference must lie in the first radial tile')
            value = value.at[0, :, :, :g+cells].set(value[0, :, :, g+cells:g+cells+1])
        parity = tuple((1 if i == axis else -1) if electric else
                       (-1 if i == axis else 1) for axis in range(3))
        for axis, bc in enumerate(static.boundary_conditions):
            if bc != BC_CONDUCTING or loc[axis] != 'C' or parity[axis] != -1:
                continue
            low = [slice(None)] * 6
            high = low.copy()
            low[axis] = 0
            high[axis] = -1
            low[axis+3] = g
            high[axis+3] = g+static.tile_shape[axis]
            value = value.at[tuple(low)].set(0.).at[tuple(high)].set(0.)
        result.append(scalar_boundaries(value, static, loc, parity))
    return tuple(result)
