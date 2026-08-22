"""Geometry and runtime records for the single-patch field-only FMR solver."""

from typing import NamedTuple

import jax

from PyPIC3D.utilities.parameters import GridParameters


E_FIELD_LOCATIONS = (("V", "C", "C"), ("C", "V", "C"), ("C", "C", "V"))
B_FIELD_LOCATIONS = (("C", "V", "V"), ("V", "C", "V"), ("V", "V", "C"))


class FMRLevel(NamedTuple):
    """Hashable geometry for one fixed refinement level."""

    index: int
    parent: int
    refinement_ratio: int
    parent_start: tuple
    parent_stop: tuple
    shape: tuple
    spacing: tuple
    lower: tuple
    upper: tuple
    tile_shape: tuple


class FMRTransferMap(NamedTuple):
    """Fixed transfer stencil for one staggered field component."""

    target_indices: jax.Array
    source_indices: jax.Array
    weights: jax.Array


class FMRLevelRuntime(NamedTuple):
    """Grids and active degrees of freedom for one level."""

    grids: GridParameters
    e_active_masks: tuple
    b_active_masks: tuple


class FMRInterfaceData(NamedTuple):
    """Bidirectional E/B transfers across the one coarse-fine interface."""

    e_coarse_to_fine_maps: tuple
    b_coarse_to_fine_maps: tuple
    e_fine_to_coarse_maps: tuple
    b_fine_to_coarse_maps: tuple


class FMRHierarchy(NamedTuple):
    """Runtime data for one root, one fine level, and their interface."""

    levels: tuple
    interface: FMRInterfaceData
