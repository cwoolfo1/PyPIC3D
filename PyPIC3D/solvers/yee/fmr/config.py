"""Configuration parsing and validation for the two-level FMR geometry."""

from numbers import Integral

from .types import (
    FMR_DEFAULT_INTERPOLATION_ORDER,
    FMR_SUPPORTED_INTERPOLATION_ORDERS,
    FMRLevel,
)


def _three_ints(values, name):
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"FMR {name} must contain exactly three integer indices.")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
        raise ValueError(f"FMR {name} must contain exactly three integer indices.")
    return tuple(int(value) for value in values)


def _fmr_enabled(config):
    raw_fmr = config.get("fmr")
    if not raw_fmr:
        return False

    enabled = raw_fmr.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("FMR enabled must be true or false.")
    return enabled


def load_fmr_interpolation_order(config):
    """Read the coarse-to-fine interpolation order as a static integer."""

    raw_fmr = config.get("fmr") or {}
    interpolation_order = raw_fmr.get(
        "interpolation_order",
        FMR_DEFAULT_INTERPOLATION_ORDER,
    )
    if (
        isinstance(interpolation_order, bool)
        or not isinstance(interpolation_order, Integral)
        or interpolation_order not in FMR_SUPPORTED_INTERPOLATION_ORDERS
    ):
        raise ValueError(
            "FMR interpolation_order must be 1 (linear) or 2 (quadratic)."
        )
    return int(interpolation_order)


def validate_fmr_configuration(config, static_config, plotting_parameters):
    """Reject runtime combinations outside the first field-only FMR scope."""

    raw_fmr = config.get("fmr")
    if not _fmr_enabled(config):
        return

    unsupported_options = sorted(
        set(raw_fmr) - {"enabled", "levels", "interpolation_order"}
    )
    if unsupported_options:
        names = ", ".join(unsupported_options)
        raise NotImplementedError(f"Unsupported FMR option(s): {names}.")

    load_fmr_interpolation_order(config)

    if static_config["solver"] != "electrodynamic_yee":
        raise NotImplementedError("FMR currently supports only solver='electrodynamic_yee'.")
    if config.get("pml"):
        raise NotImplementedError("PML is not supported with FMR yet.")
    if config.get("supergaussian"):
        raise NotImplementedError("The supergaussian absorber is not supported with FMR yet.")
    if any(key.startswith("particle") for key in config):
        raise NotImplementedError("Particle species cannot be coupled to FMR fields yet.")
    if any(key.startswith("field") for key in config):
        raise NotImplementedError("External or loaded fields cannot populate FMR levels yet.")
    if any(key.startswith("previous_field") for key in config):
        raise NotImplementedError("Previous-field restart cannot populate FMR levels yet.")

    unsupported_diagnostics = (
        "dump_fields",
        "plot_openpmd_fields",
        "plotvelocities",
        "plotchargedensity",
    )
    enabled_diagnostics = [name for name in unsupported_diagnostics if plotting_parameters.get(name, False)]
    if enabled_diagnostics:
        names = ", ".join(enabled_diagnostics)
        raise NotImplementedError(f"FMR field diagnostics are not level-aware yet: {names}.")


def load_fmr_from_toml(config, dynamic_config, root_tile_shape):
    """Parse one interior rectangular fine patch and derive its geometry."""

    raw_fmr = config.get("fmr")
    if not _fmr_enabled(config):
        return ()

    load_fmr_interpolation_order(config)
    raw_levels = raw_fmr.get("levels", ())
    if len(raw_levels) != 1:
        raise ValueError("The first FMR implementation requires exactly one [[fmr.levels]] entry.")

    root_shape = tuple(int(dynamic_config[name]) for name in ("Nx", "Ny", "Nz"))
    root_tile_shape = tuple(int(width) for width in root_tile_shape)
    if root_tile_shape != root_shape:
        raise NotImplementedError(
            "FMR currently requires root tile grid = (1, 1, 1); "
            f"root shape is {root_shape} but tile shape is {root_tile_shape}."
        )

    raw_level = raw_levels[0]
    level_keys = {"parent", "refinement_ratio", "coarse_start", "coarse_stop"}
    unsupported_options = sorted(set(raw_level) - level_keys)
    if unsupported_options:
        names = ", ".join(unsupported_options)
        raise NotImplementedError(f"Unsupported FMR level option(s): {names}.")

    parent = raw_level.get("parent", -1)
    if isinstance(parent, bool) or not isinstance(parent, Integral):
        raise ValueError("The first FMR fine level must have integer parent = 0.")
    parent = int(parent)
    if parent != 0:
        raise ValueError("The first FMR fine level must have integer parent = 0.")

    refinement_ratio = raw_level.get("refinement_ratio")
    if isinstance(refinement_ratio, bool) or not isinstance(refinement_ratio, Integral):
        raise ValueError("FMR refinement_ratio must be an even integer greater than or equal to 2.")
    refinement_ratio = int(refinement_ratio)
    if refinement_ratio < 2 or refinement_ratio % 2 != 0:
        raise ValueError("FMR refinement_ratio must be an even integer greater than or equal to 2.")

    parent_start = _three_ints(raw_level.get("coarse_start"), "coarse_start")
    parent_stop = _three_ints(raw_level.get("coarse_stop"), "coarse_stop")
    for start, stop, cells in zip(parent_start, parent_stop, root_shape):
        if not 0 <= start < stop <= cells:
            raise ValueError("FMR bounds must satisfy 0 <= coarse_start < coarse_stop <= parent shape.")
        if start == 0 or stop == cells:
            raise ValueError("The FMR fine patch must be strictly interior to the root domain.")

    spacing = tuple(float(dynamic_config[name]) for name in ("dx", "dy", "dz"))
    lower = tuple(float(dynamic_config[f"{axis}_min"]) for axis in ("x", "y", "z"))
    upper = tuple(float(dynamic_config[f"{axis}_max"]) for axis in ("x", "y", "z"))

    root_level = FMRLevel(
        level=0,
        parent=-1,
        refinement_ratio=1,
        parent_start=(0, 0, 0),
        parent_stop=root_shape,
        Nx=root_shape[0],
        Ny=root_shape[1],
        Nz=root_shape[2],
        spacing=spacing,
        x_min=lower[0],
        x_max=upper[0],
        y_min=lower[1],
        y_max=upper[1],
        z_min=lower[2],
        z_max=upper[2],
        tile_shape=root_shape,
    )

    fine_shape = tuple(
        refinement_ratio * (stop - start)
        for start, stop in zip(parent_start, parent_stop)
    )
    fine_lower = tuple(
        root_lower + start * root_spacing
        for root_lower, start, root_spacing in zip(lower, parent_start, spacing)
    )
    fine_upper = tuple(
        root_lower + stop * root_spacing
        for root_lower, stop, root_spacing in zip(lower, parent_stop, spacing)
    )
    fine_spacing = tuple(root_spacing / refinement_ratio for root_spacing in spacing)

    fine_level = FMRLevel(
        level=1,
        parent=parent,
        refinement_ratio=refinement_ratio,
        parent_start=parent_start,
        parent_stop=parent_stop,
        Nx=fine_shape[0],
        Ny=fine_shape[1],
        Nz=fine_shape[2],
        spacing=fine_spacing,
        x_min=fine_lower[0],
        x_max=fine_upper[0],
        y_min=fine_lower[1],
        y_max=fine_upper[1],
        z_min=fine_lower[2],
        z_max=fine_upper[2],
        tile_shape=fine_shape,
    )

    return root_level, fine_level
