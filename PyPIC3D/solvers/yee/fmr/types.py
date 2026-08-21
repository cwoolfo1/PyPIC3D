"""Basic data structures and constants for field-only FMR."""

from typing import NamedTuple

import jax

from PyPIC3D.utilities.parameters import GridParameters


E_FIELD_LOCATIONS = (("V", "C", "C"), ("C", "V", "C"), ("C", "C", "V"))
B_FIELD_LOCATIONS = (("C", "V", "V"), ("V", "C", "V"), ("V", "V", "C"))


class FMRLevel(NamedTuple):
    """Small, hashable geometry record for one fixed refinement level."""

    level: int
    parent: int
    refinement_ratio: int
    parent_start: tuple
    parent_stop: tuple
    Nx: int
    Ny: int
    Nz: int
    spacing: tuple
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    tile_shape: tuple


class FMRInterpolationMap(NamedTuple):
    """Fixed transfer stencil for one staggered field component."""

    target_indices: jax.Array
    source_indices: jax.Array
    weights: jax.Array


class FMRLevelData(NamedTuple):
    """JAX-array data associated with one statically described level."""

    grids: GridParameters
    e_coarse_to_fine_maps: tuple
    b_coarse_to_fine_maps: tuple
    e_fine_to_coarse_maps: tuple
    b_fine_to_coarse_maps: tuple
    e_deep_shadow_indices: tuple
    b_deep_shadow_indices: tuple
    e_active_masks: tuple
    b_active_masks: tuple
    e_weights: tuple
    b_weights: tuple


class FMRParameters(NamedTuple):
    """Dynamic FMR maps, masks, and coordinates, ordered by level."""

    levels: tuple
