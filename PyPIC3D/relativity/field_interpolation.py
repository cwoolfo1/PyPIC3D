"""Density-weighted transfers between the eight C/V metric locations."""
import jax.numpy as jnp

from .core import D_FIELD_LOCATIONS, B_FIELD_LOCATIONS

def location_interpolate_axis(field, source_location, target_location, axis):
    array_axis = axis + 3
    if source_location[axis] == target_location[axis]:
        return field
    if source_location[axis] == "C":
        return 0.5 * (field + jnp.roll(field, -1, axis=array_axis))
    return 0.5 * (field + jnp.roll(field, 1, axis=array_axis))


def location_interpolate(field, source_location, target_location):
    interpolated = field
    for axis in range(3):
        interpolated = location_interpolate_axis(interpolated, source_location, target_location, axis)
    return interpolated


def metric_weighted_interpolate(field, source_metric, target_metric, source_location, target_location):
    weighted = source_metric.sqrt_gamma * field
    weighted = location_interpolate(weighted, source_location, target_location)
    return weighted / target_metric.sqrt_gamma


def metric_at_location(metric, location):
    if location == ('C', 'C', 'C'):
        return metric.center
    if location == ('V', 'V', 'V'):
        return metric.vertex
    if location in D_FIELD_LOCATIONS:
        return metric.D[D_FIELD_LOCATIONS.index(location)]
    return metric.B[B_FIELD_LOCATIONS.index(location)]
