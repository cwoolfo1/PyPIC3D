"""Basic data structures and constants for field-only FMR."""

from typing import NamedTuple

import jax

from PyPIC3D.utilities.parameters import GridParameters


E_FIELD_LOCATIONS = (("V", "C", "C"), ("C", "V", "C"), ("C", "C", "V"))
B_FIELD_LOCATIONS = (("C", "V", "V"), ("V", "C", "V"), ("V", "V", "C"))

FMR_DEFAULT_INTERPOLATION_ORDER = 1
FMR_SUPPORTED_INTERPOLATION_ORDERS = (1, 2)
FMR_INTERPOLATION_ORDER = FMR_DEFAULT_INTERPOLATION_ORDER


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
    """Coarse-to-fine tensor-product prolongation map for one staggered E component."""

    target_indices: jax.Array
    source_indices: jax.Array
    weights: jax.Array


class FMRLevelData(NamedTuple):
    """JAX-array data associated with one statically described level."""

    grids: GridParameters
    e_interface_maps: tuple
    b_active_masks: tuple
    e_weights: tuple
    b_weights: tuple


class FMRParameters(NamedTuple):
    """Dynamic FMR maps, masks, and coordinates, ordered by level."""

    levels: tuple
