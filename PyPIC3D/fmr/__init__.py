"""Public API for the field-only, two-level Yee FMR implementation."""

from .config import load_fmr_from_toml, validate_fmr_configuration
from .curls import fmr_curl_b_to_e, fmr_curl_e_to_b
from .fields import build_fmr_fields, build_fmr_parameters
from .interpolation import build_e_interface_maps, prolong_e_to_fine_interface
from .time_loop import (
    time_loop_electrodynamic_fmr_fields,
    update_B_fmr,
    update_E_fmr,
)
from .types import (
    B_FIELD_LOCATIONS,
    E_FIELD_LOCATIONS,
    FMR_INTERPOLATION_ORDER,
    FMRInterpolationMap,
    FMRLevel,
    FMRLevelData,
    FMRParameters,
)
from .weights import build_b_active_masks, build_fmr_metric_weights


__all__ = [
    "B_FIELD_LOCATIONS",
    "E_FIELD_LOCATIONS",
    "FMRInterpolationMap",
    "FMR_INTERPOLATION_ORDER",
    "FMRLevel",
    "FMRLevelData",
    "FMRParameters",
    "build_b_active_masks",
    "build_e_interface_maps",
    "build_fmr_fields",
    "build_fmr_metric_weights",
    "build_fmr_parameters",
    "fmr_curl_b_to_e",
    "fmr_curl_e_to_b",
    "load_fmr_from_toml",
    "prolong_e_to_fine_interface",
    "time_loop_electrodynamic_fmr_fields",
    "update_B_fmr",
    "update_E_fmr",
    "validate_fmr_configuration",
]
