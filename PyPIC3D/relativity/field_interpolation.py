"""Density-weighted transfers between the eight C/V metric locations."""
import jax.numpy as jnp

from .core import D_FIELD_LOCATIONS, B_FIELD_LOCATIONS

def location_interpolate(field, source_location, target_location):
    """Average a field from one C/V location to another, one axis at a time.

    Axis ``a`` of the location is array axis ``a + 3`` of a tiled field.  A
    C-to-V move averages with the upper neighbour, a V-to-C move with the
    lower one.
    """
    interpolated = field
    for axis in range(3):
        if source_location[axis] == target_location[axis]:
            continue
        shift = -1 if source_location[axis] == "C" else 1
        interpolated = 0.5 * (interpolated + jnp.roll(interpolated, shift, axis=axis + 3))
    return interpolated


def metric_weighted_interpolate(field, source_metric, target_metric, source_location, target_location):
    weighted = source_metric.sqrt_gamma * field
    weighted = location_interpolate(weighted, source_location, target_location)
    return weighted / target_metric.sqrt_gamma


def reconstruct_vector(vector, locations, metric, target, *, preserve_native=True):
    """Transfer vector components to one C/V location using metric densities.

    PEC reconstruction retains native values exactly. Constitutive operators
    pass ``preserve_native=False`` to retain their density multiply/divide
    even for a component already at the target location.
    """
    target_metric = metric_at_location(metric, target)
    return tuple(
        value if preserve_native and source == target else metric_weighted_interpolate(
            value, metric_at_location(metric, source), target_metric, source, target
        )
        for value, source in zip(vector, locations)
    )


def metric_at_location(metric, location):
    if location == ('C', 'C', 'C'):
        return metric.center
    if location == ('V', 'V', 'V'):
        return metric.vertex
    if location in D_FIELD_LOCATIONS:
        return metric.D[D_FIELD_LOCATIONS.index(location)]
    return metric.B[B_FIELD_LOCATIONS.index(location)]
