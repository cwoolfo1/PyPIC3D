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


@partial(jax.jit, static_argnames=('static', 'locations', 'field_kind'))
def refresh_fields(vector, static, locations, field_kind=None, metric=None):
    """Refresh fields; conducting D/B require the local spatial metric.

    Auxiliary E/H retain their computed physical exterior values. Only D/B
    receive horizon treatment and the FIDO projector boundary contract.
    """
    from .ghost_cells import update_tiled_vector_ghost_cells
    if field_kind not in (None, 'D', 'B'):
        raise ValueError("field_kind must be D, B, or None for auxiliary fields")
    if field_kind is not None:
        cells, g = static.horizon_field_cells, static.guard_cells
        if cells:
            if cells >= static.tile_shape[0]:
                raise ValueError('Horizon boundary reference must lie in the first radial tile')
            vector = tuple(value.at[0, :, :, :g+cells].set(value[0, :, :, g+cells:g+cells+1])
                           for value in vector)
        if BC_CONDUCTING in static.boundary_conditions:
            if metric is None:
                raise ValueError('Conducting D/B boundaries require a Yee metric')
            from .pec import project_fields
            return project_fields(vector, static, locations, field_kind, metric)
    return update_tiled_vector_ghost_cells(
        vector, static, static.guard_cells, locations=locations,
        preserve_exterior=field_kind is None)
