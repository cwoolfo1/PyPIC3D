"""Public API for the field-only, two-level Yee FMR implementation."""

from .config import (
    load_fmr_from_toml,
    validate_fmr_configuration,
)
from .curls import fmr_curl_b_to_e, fmr_curl_e_to_b
from .fields import (
    build_fmr_fields,
    build_fmr_parameters,
    synchronize_b_levels,
    synchronize_e_levels,
)
from .interpolation import (
    build_b_transfer_maps,
    build_e_transfer_maps,
    fill_b_coarse_halo,
    fill_b_fine_halo,
    fill_e_coarse_halo,
    fill_e_fine_halo,
)
from .time_loop import (
    time_loop_electrodynamic_fmr_fields,
    update_B_fmr,
    update_E_fmr,
)
from .types import (
    B_FIELD_LOCATIONS,
    E_FIELD_LOCATIONS,
    FMRInterpolationMap,
    FMRLevel,
    FMRLevelData,
    FMRParameters,
)
from .weights import build_fmr_metric_weights


__all__ = [
    "B_FIELD_LOCATIONS",
    "E_FIELD_LOCATIONS",
    "FMRInterpolationMap",
    "FMRLevel",
    "FMRLevelData",
    "FMRParameters",
    "build_b_transfer_maps",
    "build_e_transfer_maps",
    "build_fmr_fields",
    "build_fmr_metric_weights",
    "build_fmr_parameters",
    "fill_b_coarse_halo",
    "fill_b_fine_halo",
    "fill_e_coarse_halo",
    "fill_e_fine_halo",
    "fmr_curl_b_to_e",
    "fmr_curl_e_to_b",
    "load_fmr_from_toml",
    "synchronize_b_levels",
    "synchronize_e_levels",
    "time_loop_electrodynamic_fmr_fields",
    "update_B_fmr",
    "update_E_fmr",
    "validate_fmr_configuration",
]
