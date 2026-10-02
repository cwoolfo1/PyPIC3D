"""Boundary staging for contravariant fields and computed auxiliary fields."""
from functools import partial

import jax

from .grid_and_stencil import BC_CONDUCTING
from .ghost_cells import update_tiled_vector_ghost_cells
from .pec import enforce_pec_B, enforce_pec_D


def freeze_horizon_layers(vector, static):
    """Extrapolate inner radial D/B layers from the first unfrozen plane.

    This spherical Kerr-Schild policy is validated during configuration and
    must precede conducting-wall projection. Auxiliary E/H do not use it.
    """
    cells, guard = static.horizon_field_cells, static.guard_cells
    if not cells:
        return vector
    if cells >= static.tile_shape[0]:
        raise ValueError('Horizon boundary reference must lie in the first radial tile')
    reference = guard + cells
    return tuple(
        value.at[0, :, :, :reference].set(value[0, :, :, reference:reference+1])
        for value in vector
    )


@partial(jax.jit, static_argnames=('static', 'locations', 'field_kind'))
def refresh_fields(vector, static, locations, field_kind=None, metric=None):
    """Refresh fields; conducting D/B require the local spatial metric.

    Auxiliary E/H retain their computed physical exterior values. Only D/B
    receive horizon treatment and the FIDO projector boundary contract.
    """
    if field_kind not in (None, 'D', 'B'):
        raise ValueError("field_kind must be D, B, or None for auxiliary fields")
    if field_kind is not None:
        vector = freeze_horizon_layers(vector, static)
        if BC_CONDUCTING in static.boundary_conditions:
            if metric is None:
                raise ValueError('Conducting D/B boundaries require a Yee metric')
            enforce_pec = enforce_pec_D if field_kind == 'D' else enforce_pec_B
            return enforce_pec(vector, static, locations, metric)
    return update_tiled_vector_ghost_cells(
        vector, static, static.guard_cells, locations=locations,
        preserve_exterior=field_kind is None)
