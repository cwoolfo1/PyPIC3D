#!/usr/bin/env python3
"""Create normalized movies from the 2-D Harris-sheet openPMD output.

The field writer stores every vector component on an ``(x, y, z)`` mesh, but
the components retain PyPIC3D's Yee staggering.  This module therefore moves
the magnetic field to the grid required by each diagnostic before plotting or
forming ``E + u x B``.  The field-line movie applies a clearly labeled,
two-cell display filter to suppress nonpersistent particle-noise null pairs;
all quantitative diagnostics use the raw dump.  Frames are read one at a time
so a long run is never held in memory.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pypic3d-reconnection-matplotlib")

import numpy as np


# Plasma parameters used by both the input generator and these diagnostics.
MU0 = 1.25663706e-6  # H m^-1
EPS0 = 8.85418782e-12  # F m^-1
ELECTRON_MASS = 9.1093837e-31  # kg
POSITRON_MASS = ELECTRON_MASS  # kg
ELEMENTARY_CHARGE = 1.602e-19  # C
LIGHT_SPEED = 2.99792458e8  # m s^-1

SHEET_DENSITY = 1.0e22  # m^-3, per species
BACKGROUND_DENSITY = 0.3 * SHEET_DENSITY  # m^-3, per species
THERMAL_SPEED = 0.05 * LIGHT_SPEED
B0 = np.sqrt(4.0 * MU0 * SHEET_DENSITY * ELECTRON_MASS * THERMAL_SPEED**2)
PLASMA_FREQUENCY = np.sqrt(
    SHEET_DENSITY * ELEMENTARY_CHARGE**2 / (EPS0 * ELECTRON_MASS)
)
ELECTRON_SKIN_DEPTH = LIGHT_SPEED / PLASMA_FREQUENCY
SHEET_HALF_WIDTH = 0.5 * ELECTRON_SKIN_DEPTH
CYCLOTRON_FREQUENCY = ELEMENTARY_CHARGE * B0 / ELECTRON_MASS
ALFVEN_SPEED = B0 / np.sqrt(
    MU0 * 2.0 * ELECTRON_MASS * BACKGROUND_DENSITY
)
CURRENT_DENSITY_SCALE = B0 / (MU0 * SHEET_HALF_WIDTH)

CURRENT_MASK_FRACTION = 0.02
DISPLAY_X_FRACTION = 0.80
ETA_PERCENTILE = 99.0
FIELD_LINE_SMOOTHING_SIGMA_CELLS = 2.0
GAUSSIAN_KERNEL_TRUNCATION = 4.0

VECTOR_COMPONENTS = ("x", "y", "z")
REQUIRED_MESHES = ("E", "B", "J", "fluid_velocity")


@dataclass(frozen=True)
class Grid2D:
    """Base (electric-field) x-z grid reconstructed from openPMD metadata."""

    x: np.ndarray
    z: np.ndarray
    dx: float
    dz: float


@dataclass(frozen=True)
class Frame:
    """One owned simulation frame in SI units, with arrays ordered x-z."""

    iteration: int
    time: float
    grid: Grid2D
    electric: tuple[np.ndarray, np.ndarray, np.ndarray]
    magnetic: tuple[np.ndarray, np.ndarray, np.ndarray]
    current: tuple[np.ndarray, np.ndarray, np.ndarray]
    velocity: tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class PlotScales:
    """Fixed scales shared by every frame of the four movies."""

    magnetic_max: float
    eta_max: float
    velocity_max: float


def _import_openpmd_api():
    try:
        import openpmd_api as io
    except ImportError as exc:  # pragma: no cover - depends on user environment
        raise RuntimeError(
            "openpmd-api is required to read the field dump. Install the project "
            "dependencies with `python -m pip install -e .`."
        ) from exc
    return io


def _matplotlib_modules():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import animation, colors
    except ImportError as exc:  # pragma: no cover - depends on user environment
        raise RuntimeError(
            "Matplotlib is required to create the reconnection movies. Install "
            "the project dependencies with `python -m pip install -e .`."
        ) from exc
    return plt, animation, colors


def resolve_series_path(path: str | Path) -> Path:
    """Resolve PyPIC3D's one-line ``.pmd`` pointer relative to its directory."""

    requested = Path(path).expanduser()
    if not requested.exists():
        raise FileNotFoundError(
            f"Field series {requested} does not exist. Run the reconnection demo first."
        )
    if requested.suffix != ".pmd":
        return requested.resolve()

    targets = [line.strip() for line in requested.read_text().splitlines() if line.strip()]
    if len(targets) != 1:
        raise ValueError(
            f"Expected {requested} to contain exactly one non-empty series path; "
            f"found {len(targets)}."
        )
    target = Path(targets[0]).expanduser()
    if not target.is_absolute():
        target = requested.parent / target
    if not target.exists() and "%T" not in str(target):
        raise FileNotFoundError(
            f"The openPMD pointer {requested} refers to missing series {target}."
        )
    return target.resolve()


def _decode_axis_label(label) -> str:
    if isinstance(label, bytes):
        label = label.decode("utf-8")
    return str(label).lower()


def to_xz(array: np.ndarray, axis_labels: Sequence[str]) -> np.ndarray:
    """Drop a singleton y axis and return a component ordered ``(x, z)``."""

    values = np.asarray(array)
    labels = [_decode_axis_label(label) for label in axis_labels]
    if len(labels) != values.ndim:
        raise ValueError(
            f"Mesh has {values.ndim} dimensions but axis labels are {labels}."
        )
    if len(set(labels)) != len(labels):
        raise ValueError(f"Mesh axis labels must be unique; received {labels}.")

    if "y" in labels:
        y_axis = labels.index("y")
        if values.shape[y_axis] != 1:
            raise ValueError(
                "The reconnection analysis expects a 2-D x-z dump with a singleton "
                f"y axis; received shape {values.shape}."
            )
        values = np.squeeze(values, axis=y_axis)
        labels.pop(y_axis)

    if set(labels) != {"x", "z"} or len(labels) != 2:
        raise ValueError(
            "The reconnection analysis requires x and z mesh axes; received "
            f"{labels}."
        )
    return np.transpose(values, (labels.index("x"), labels.index("z"))).copy()


def grid_from_metadata(
    shape: Sequence[int],
    axis_labels: Sequence[str],
    spacing: Sequence[float],
    offset: Sequence[float],
    grid_unit_si: float = 1.0,
) -> Grid2D:
    """Build the base x-z coordinate vectors from one openPMD mesh."""

    labels = [_decode_axis_label(label) for label in axis_labels]
    if not (len(shape) == len(labels) == len(spacing) == len(offset)):
        raise ValueError("Mesh shape, labels, spacing, and offset must have equal length.")
    if "x" not in labels or "z" not in labels:
        raise ValueError(f"Mesh metadata does not contain x and z axes: {labels}.")
    if "y" in labels and int(shape[labels.index("y")]) != 1:
        raise ValueError(f"Expected a singleton y mesh, received shape {tuple(shape)}.")

    scale = float(grid_unit_si)
    x_axis = labels.index("x")
    z_axis = labels.index("z")
    dx = float(spacing[x_axis]) * scale
    dz = float(spacing[z_axis]) * scale
    if not np.isfinite(dx) or not np.isfinite(dz) or dx <= 0.0 or dz <= 0.0:
        raise ValueError(f"Mesh spacings must be positive and finite; dx={dx}, dz={dz}.")
    x0 = float(offset[x_axis]) * scale
    z0 = float(offset[z_axis]) * scale
    x = x0 + np.arange(int(shape[x_axis]), dtype=float) * dx
    z = z0 + np.arange(int(shape[z_axis]), dtype=float) * dz
    return Grid2D(x=x, z=z, dx=dx, dz=dz)


class OpenPMDFieldReader:
    """Random-access, frame-at-a-time reader for PyPIC3D field diagnostics."""

    def __init__(self, path: str | Path):
        self.io = _import_openpmd_api()
        self.path = resolve_series_path(path)
        self.series = self.io.Series(str(self.path), self.io.Access.read_only)
        self.iterations = tuple(sorted(int(index) for index in self.series.iterations))
        if not self.iterations:
            self.close()
            raise ValueError(f"The field series {self.path} contains no iterations.")
        self._validate_meshes()

    def close(self):
        series = getattr(self, "series", None)
        if series is not None:
            series.close()
            self.series = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def _iteration(self, iteration_index: int):
        if self.series is None:
            raise RuntimeError("The openPMD reader is closed.")
        return self.series.iterations[int(iteration_index)]

    def _validate_meshes(self):
        iteration = self._iteration(self.iterations[0])
        available = set(iteration.meshes)
        missing = [name for name in REQUIRED_MESHES if name not in available]
        if missing:
            self.close()
            raise KeyError(
                "The field dump is missing required mesh record(s): "
                f"{', '.join(missing)}. Set `plot_openpmd_fields = true` and "
                "`plotvelocities = true` in the demo configuration, then rerun it."
            )
        for name in REQUIRED_MESHES:
            mesh = iteration.meshes[name]
            components = set(mesh)
            missing_components = [item for item in VECTOR_COMPONENTS if item not in components]
            if missing_components:
                self.close()
                raise KeyError(
                    f"Mesh {name!r} is missing vector component(s): "
                    f"{', '.join(missing_components)}."
                )

    def read(self, iteration_index: int) -> Frame:
        """Load one complete frame, then release openPMD's pending buffers."""

        iteration = self._iteration(iteration_index)
        pending = []
        descriptors = []

        for mesh_name in REQUIRED_MESHES:
            mesh = iteration.meshes[mesh_name]
            labels = tuple(_decode_axis_label(label) for label in mesh.axis_labels)
            for component_name in VECTOR_COMPONENTS:
                component = mesh[component_name]
                pending.append(component.load_chunk())
                descriptors.append((mesh_name, component_name, labels, float(component.unit_SI)))

        self.series.flush()
        vectors: dict[str, list[np.ndarray]] = {name: [] for name in REQUIRED_MESHES}
        for raw, (mesh_name, _component_name, labels, unit_si) in zip(pending, descriptors):
            vectors[mesh_name].append(to_xz(np.array(raw, copy=True) * unit_si, labels))

        shape = vectors["E"][0].shape
        for mesh_name, components in vectors.items():
            for component_name, values in zip(VECTOR_COMPONENTS, components):
                if values.shape != shape:
                    raise ValueError(
                        f"Mesh {mesh_name}/{component_name} has x-z shape {values.shape}; "
                        f"expected {shape}."
                    )

        reference_mesh = iteration.meshes["E"]
        reference_component = reference_mesh["x"]
        labels = tuple(_decode_axis_label(label) for label in reference_mesh.axis_labels)
        grid = grid_from_metadata(
            reference_component.shape,
            labels,
            reference_mesh.grid_spacing,
            reference_mesh.grid_global_offset,
            getattr(reference_mesh, "grid_unit_SI", 1.0),
        )
        time = float(iteration.time) * float(getattr(iteration, "time_unit_SI", 1.0))
        return Frame(
            iteration=int(iteration_index),
            time=time,
            grid=grid,
            electric=tuple(vectors["E"]),
            magnetic=tuple(vectors["B"]),
            current=tuple(vectors["J"]),
            velocity=tuple(vectors["fluid_velocity"]),
        )


def collocate_magnetic_for_field_lines(
    bx: np.ndarray, bz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Move Bx and Bz to periodic-x cell centers used by ``streamplot``."""

    bx = np.asarray(bx)
    bz = np.asarray(bz)
    if bx.shape != bz.shape or bx.ndim != 2 or min(bx.shape) < 2:
        raise ValueError(f"Bx and Bz must share a 2-D shape of at least 2x2; got {bx.shape} and {bz.shape}.")
    bx_center = 0.5 * (bx + np.roll(bx, -1, axis=0))
    bz_center = 0.5 * (bz[:, :-1] + bz[:, 1:])
    return bx_center[:, :-1], bz_center


def smooth_field_line_component(
    values: np.ndarray,
    sigma_cells: float = FIELD_LINE_SMOOTHING_SIGMA_CELLS,
) -> np.ndarray:
    """Gaussian-smooth one displayed field component without SciPy.

    The x direction wraps across the simulation's periodic seam.  The z
    direction instead extends its nearest edge value, matching the
    nonperiodic conducting boundaries.  This filter is intentionally used
    only by :func:`field_line_quantities`; quantitative diagnostics continue
    to operate on the raw field dump.
    """

    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError(
            "A field-line component must be a non-empty 2-D array; "
            f"received shape {values.shape}."
        )
    if not np.isfinite(sigma_cells) or sigma_cells < 0.0:
        raise ValueError(
            "The field-line smoothing width must be finite and non-negative; "
            f"received {sigma_cells}."
        )
    if sigma_cells == 0.0:
        return values.copy()

    radius = max(
        1, int(np.ceil(GAUSSIAN_KERNEL_TRUNCATION * float(sigma_cells)))
    )
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma_cells) ** 2)
    kernel /= np.sum(kernel)

    # Convolve x using periodic samples.  The kernel is symmetric, so the
    # sign convention of np.roll does not affect the result.
    filtered_x = np.zeros_like(values)
    for offset, weight in zip(offsets, kernel):
        filtered_x += weight * np.roll(values, int(offset), axis=0)

    # Extend the nearest z-edge value instead of wrapping through a
    # conducting boundary.
    padded_z = np.pad(filtered_x, ((0, 0), (radius, radius)), mode="edge")
    filtered = np.zeros_like(filtered_x)
    for index, weight in enumerate(kernel):
        filtered += weight * padded_z[:, index : index + values.shape[1]]
    return filtered


def collocate_magnetic_to_e_grid(
    bx: np.ndarray, bz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Move Bx and Bz to the Ey/Jy/u grid, omitting the unsupported z edge."""

    bx = np.asarray(bx)
    bz = np.asarray(bz)
    if bx.shape != bz.shape or bx.ndim != 2 or min(bx.shape) < 2:
        raise ValueError(f"Bx and Bz must share a 2-D shape of at least 2x2; got {bx.shape} and {bz.shape}.")
    bx_on_e = 0.5 * (bx[:, :-1] + bx[:, 1:])
    bz_on_e = 0.5 * (np.roll(bz, 1, axis=0) + bz)
    return bx_on_e, bz_on_e[:, 1:]


def field_line_quantities(frame: Frame):
    """Return smoothed display fields and magnitude at cell centers.

    Two-cell Gaussian smoothing suppresses nonpersistent particle-noise null
    pairs in the streamline movie.  It does not modify ``frame`` or feed the
    effective-resistivity and velocity diagnostics.
    """

    bx, by, bz = frame.magnetic
    bx_center, bz_center = collocate_magnetic_for_field_lines(bx, bz)
    by_center = np.asarray(by)[:, :-1]
    if by_center.shape != bx_center.shape:
        raise ValueError(
            f"Centered By has shape {by_center.shape}; expected {bx_center.shape}."
        )
    x = frame.grid.x + 0.5 * frame.grid.dx
    z = frame.grid.z[:-1] + 0.5 * frame.grid.dz
    bx_normalized = smooth_field_line_component(bx_center) / B0
    by_normalized = smooth_field_line_component(by_center) / B0
    bz_normalized = smooth_field_line_component(bz_center) / B0
    magnitude = np.sqrt(
        bx_normalized**2 + by_normalized**2 + bz_normalized**2
    )
    return x, z, bx_normalized, bz_normalized, magnitude


def effective_resistivity(frame: Frame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(x, z, eta_eff)`` where eta is the nonideal Ohm-law proxy.

    ``eta_eff = (E + u x B)_y / J_y`` is meaningful only where the current is
    resolved, so low-current and non-finite cells are represented by NaN.
    """

    _, ey, _ = frame.electric
    _, jy, _ = frame.current
    ux, _, uz = frame.velocity
    bx, _, bz = frame.magnetic
    bx_on_e, bz_on_e = collocate_magnetic_to_e_grid(bx, bz)

    ey = ey[:, 1:]
    jy = jy[:, 1:]
    ux = ux[:, 1:]
    uz = uz[:, 1:]
    nonideal_y = ey + uz * bx_on_e - ux * bz_on_e
    valid = (
        np.isfinite(nonideal_y)
        & np.isfinite(jy)
        & (np.abs(jy) >= CURRENT_MASK_FRACTION * CURRENT_DENSITY_SCALE)
    )
    eta = np.full(nonideal_y.shape, np.nan, dtype=float)
    np.divide(nonideal_y, jy, out=eta, where=valid)
    return frame.grid.x, frame.grid.z[1:], eta


def nearest_axis_index(axis: np.ndarray, coordinate: float = 0.0) -> int:
    axis = np.asarray(axis)
    if axis.ndim != 1 or axis.size == 0 or not np.all(np.isfinite(axis)):
        raise ValueError("Coordinate axis must be a non-empty finite 1-D array.")
    return int(np.argmin(np.abs(axis - coordinate)))


def velocity_cuts(frame: Frame):
    """Return the x cut at z=0 and z cut at x=0, normalized by upstream VA."""

    x_index = nearest_axis_index(frame.grid.x)
    z_index = nearest_axis_index(frame.grid.z)
    x_cut = tuple(component[:, z_index] / ALFVEN_SPEED for component in frame.velocity)
    z_cut = tuple(component[x_index, :] / ALFVEN_SPEED for component in frame.velocity)
    return x_cut, z_cut


def central_x_mask(x: np.ndarray, fraction: float = DISPLAY_X_FRACTION) -> np.ndarray:
    """Select a central window that excludes the periodic seam and its O-point."""

    x = np.asarray(x)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("The x coordinate must contain at least two points.")
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"x crop fraction must lie in (0, 1], received {fraction}.")
    dx = float(np.median(np.diff(x)))
    domain_width = abs(dx) * x.size
    mask = np.abs(x) <= 0.5 * fraction * domain_width
    if np.count_nonzero(mask) < 2:
        raise ValueError("The central x crop contains fewer than two grid points.")
    return mask


def find_central_x_point(
    x: np.ndarray,
    z: np.ndarray,
    bx: np.ndarray,
    bz: np.ndarray,
) -> tuple[float, float]:
    """Locate the single X-point in the displayed central x window.

    ``bx`` and ``bz`` must already be collocated and display-smoothed.  Each
    cell that brackets zero in both components is searched for a root of the
    bilinear field interpolant.  The root is retained only when its local
    Jacobian has negative determinant (a magnetic saddle), then duplicate
    roots on cell boundaries are merged.
    """

    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    bx = np.asarray(bx, dtype=float)
    bz = np.asarray(bz, dtype=float)
    if (
        x.ndim != 1
        or z.ndim != 1
        or x.size < 2
        or z.size < 2
        or bx.shape != (x.size, z.size)
        or bz.shape != bx.shape
    ):
        raise ValueError(
            "X-point coordinates must be 1-D and Bx/Bz must share shape "
            f"(len(x), len(z)); received {x.shape}, {z.shape}, "
            f"{bx.shape}, and {bz.shape}."
        )
    if (
        not np.all(np.isfinite(x))
        or not np.all(np.isfinite(z))
        or not np.all(np.isfinite(bx))
        or not np.all(np.isfinite(bz))
        or not np.all(np.diff(x) > 0.0)
        or not np.all(np.diff(z) > 0.0)
    ):
        raise ValueError("X-point coordinates and fields must be finite on increasing axes.")

    crop = central_x_mask(x)
    x_search = x[crop]
    bx_search = bx[crop]
    bz_search = bz[crop]
    bx_corners = (
        bx_search[:-1, :-1],
        bx_search[1:, :-1],
        bx_search[:-1, 1:],
        bx_search[1:, 1:],
    )
    bz_corners = (
        bz_search[:-1, :-1],
        bz_search[1:, :-1],
        bz_search[:-1, 1:],
        bz_search[1:, 1:],
    )
    brackets_bx = np.minimum.reduce(bx_corners) <= 0.0
    brackets_bx &= np.maximum.reduce(bx_corners) >= 0.0
    brackets_bz = np.minimum.reduce(bz_corners) <= 0.0
    brackets_bz &= np.maximum.reduce(bz_corners) >= 0.0

    roots: list[tuple[float, float]] = []
    for i, j in np.argwhere(brackets_bx & brackets_bz):
        # F(u,v) = a + b*u + c*v + d*u*v on the unit cell.
        f00, f10, f01, f11 = (component[i, j] for component in bx_corners)
        g00, g10, g01, g11 = (component[i, j] for component in bz_corners)
        a, b, c = f00, f10 - f00, f01 - f00
        d = f11 - f10 - f01 + f00
        e, f, g = g00, g10 - g00, g01 - g00
        h = g11 - g10 - g01 + g00

        # Eliminating v gives A*u**2 + B*u + C = 0.
        coefficients = np.array(
            [f * d - h * b, e * d + f * c - g * b - h * a, e * c - g * a]
        )
        coefficient_scale = float(np.max(np.abs(coefficients)))
        if coefficient_scale == 0.0:
            continue
        polynomial_tolerance = 256.0 * np.finfo(float).eps * coefficient_scale
        if abs(coefficients[0]) <= polynomial_tolerance:
            if abs(coefficients[1]) <= polynomial_tolerance:
                continue
            u_values = (-coefficients[2] / coefficients[1],)
        else:
            u_values = tuple(np.roots(coefficients[:]))

        field_scale = max(abs(a), abs(b), abs(c), abs(d), abs(e), abs(f), abs(g), abs(h))
        for u_value in u_values:
            if abs(np.imag(u_value)) > 1.0e-10:
                continue
            u = float(np.real(u_value))
            if not -1.0e-9 <= u <= 1.0 + 1.0e-9:
                continue
            denominator_f = c + d * u
            denominator_g = g + h * u
            denominator = max(abs(denominator_f), abs(denominator_g))
            if denominator <= 256.0 * np.finfo(float).eps * max(
                field_scale, np.finfo(float).tiny
            ):
                continue
            if abs(denominator_f) >= abs(denominator_g):
                v = -(a + b * u) / denominator_f
            else:
                v = -(e + f * u) / denominator_g
            if not -1.0e-9 <= v <= 1.0 + 1.0e-9:
                continue
            u = float(np.clip(u, 0.0, 1.0))
            v = float(np.clip(v, 0.0, 1.0))
            residual = max(
                abs(a + b * u + c * v + d * u * v),
                abs(e + f * u + g * v + h * u * v),
            )
            if residual > 1.0e-9 * max(field_scale, np.finfo(float).tiny):
                continue

            df_du, df_dv = b + d * v, c + d * u
            dg_du, dg_dv = f + h * v, g + h * u
            determinant = df_du * dg_dv - df_dv * dg_du
            jacobian_scale = max(abs(df_du), abs(df_dv), abs(dg_du), abs(dg_dv))
            if determinant >= -1.0e-10 * jacobian_scale**2:
                continue

            root = (
                float(x_search[i] + u * (x_search[i + 1] - x_search[i])),
                float(z[j] + v * (z[j + 1] - z[j])),
            )
            duplicate = any(
                abs(root[0] - other[0]) <= 1.0e-6 * (x_search[i + 1] - x_search[i])
                and abs(root[1] - other[1]) <= 1.0e-6 * (z[j + 1] - z[j])
                for other in roots
            )
            if not duplicate:
                roots.append(root)

    roots.sort(key=lambda root: (root[0], root[1]))
    if not roots:
        raise ValueError("No central X-point was found in the displayed magnetic field.")
    if len(roots) != 1:
        coordinates = ", ".join(f"({xp:.6g}, {zp:.6g})" for xp, zp in roots)
        raise ValueError(
            f"Expected one central X-point in the displayed magnetic field; "
            f"found {len(roots)} at {coordinates}."
        )
    return roots[0]


def _finite_absolute_max(values: np.ndarray) -> float:
    finite = np.asarray(values)[np.isfinite(values)]
    return float(np.max(np.abs(finite))) if finite.size else 0.0


def validate_frame(frame: Frame) -> None:
    """Reject corrupt field dumps before fixed plot limits are calculated."""

    records = {
        "E": frame.electric,
        "B": frame.magnetic,
        "J": frame.current,
        "fluid_velocity": frame.velocity,
    }
    for record_name, components in records.items():
        for component_name, values in zip(VECTOR_COMPONENTS, components):
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"Iteration {frame.iteration} contains non-finite values in "
                    f"{record_name}/{component_name}."
                )

    speed = np.sqrt(sum(np.asarray(component) ** 2 for component in frame.velocity))
    maximum_speed = float(np.max(speed))
    if maximum_speed > 1.001 * LIGHT_SPEED:
        raise ValueError(
            f"Iteration {frame.iteration} contains a nonphysical bulk velocity "
            f"of {maximum_speed / LIGHT_SPEED:.6g} c."
        )


def scan_plot_scales(reader: OpenPMDFieldReader) -> PlotScales:
    """Scan frames one at a time to establish non-flickering plot scales."""

    magnetic_max = 0.0
    eta_max = 0.0
    velocity_max = 0.0
    eta_frames = 0

    for iteration_index in reader.iterations:
        frame = reader.read(iteration_index)
        validate_frame(frame)
        x, _z, _bx, _bz, magnitude = field_line_quantities(frame)
        crop = central_x_mask(x)
        magnetic_max = max(
            magnetic_max,
            _finite_absolute_max(magnitude[crop]),
        )

        eta_x, _eta_z, eta = effective_resistivity(frame)
        eta_crop = central_x_mask(eta_x)
        displayed_eta = eta[eta_crop]
        finite_eta = np.abs(displayed_eta[np.isfinite(displayed_eta)])
        if finite_eta.size:
            eta_max = max(eta_max, float(np.percentile(finite_eta, ETA_PERCENTILE)))
            eta_frames += 1

        x_cut, z_cut = velocity_cuts(frame)
        for component in (*x_cut, *z_cut):
            velocity_max = max(velocity_max, _finite_absolute_max(component))

    if magnetic_max <= 0.0:
        raise ValueError("No finite nonzero magnetic field was found in the field dump.")
    if eta_frames == 0:
        raise ValueError(
            "No valid effective-resistivity cells were found. The current may be "
            "zero in every frame or below the configured 0.02 J0 mask."
        )
    return PlotScales(
        magnetic_max=1.02 * magnetic_max,
        eta_max=max(1.02 * eta_max, np.finfo(float).eps),
        velocity_max=max(1.05 * velocity_max, 1.0e-6),
    )


def _frame_title(frame: Frame, diagnostic: str) -> str:
    return f"{diagnostic}    $\\Omega_c t$ = {CYCLOTRON_FREQUENCY * frame.time:.3f}"


@contextmanager
def _movie_writer(path: Path, fps: int, dpi: int, figsize=(8.0, 6.0)):
    plt, animation, _colors = _matplotlib_modules()
    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError(
            "Matplotlib cannot find FFmpeg. Install FFmpeg or make its executable "
            "available on PATH before running this analysis."
        )
    figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
    writer = animation.FFMpegWriter(
        fps=fps,
        metadata={"artist": "PyPIC3D", "title": path.stem},
        codec="h264",
        bitrate=2400,
    )
    try:
        with writer.saving(figure, str(path), dpi=dpi):
            yield plt, figure, axis, writer
    finally:
        plt.close(figure)


def write_field_line_movie(
    reader: OpenPMDFieldReader, output_path: Path, scales: PlotScales, fps: int, dpi: int
):
    """Write normalized reconnecting field lines, centered on the seeded X-point."""

    _plt, _animation, colors = _matplotlib_modules()
    norm = colors.Normalize(vmin=0.0, vmax=scales.magnetic_max)
    with _movie_writer(output_path, fps, dpi) as (plt, figure, axis, writer):
        colorbar = figure.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=axis
        )
        colorbar.set_label(r"$|B|/B_0$")
        for iteration_index in reader.iterations:
            frame = reader.read(iteration_index)
            x, z, bx, bz, magnitude = field_line_quantities(frame)
            crop = central_x_mask(x)
            x_plot = x[crop] / ELECTRON_SKIN_DEPTH
            z_plot = z / ELECTRON_SKIN_DEPTH
            bx_plot = bx[crop]
            bz_plot = bz[crop]
            magnitude_plot = magnitude[crop]
            x_point, z_point = find_central_x_point(x, z, bx, bz)

            axis.clear()
            axis.set_facecolor("black")
            axis.streamplot(
                x_plot,
                z_plot,
                bx_plot.T,
                bz_plot.T,
                color=magnitude_plot.T,
                cmap="viridis",
                norm=norm,
                density=(1.5, 1.2),
                linewidth=0.9,
                arrowsize=0.7,
                broken_streamlines=False,
            )
            axis.plot(
                x_point / ELECTRON_SKIN_DEPTH,
                z_point / ELECTRON_SKIN_DEPTH,
                marker="x",
                color="red",
                markersize=8,
                mew=1.8,
            )
            smoothing_width = (
                FIELD_LINE_SMOOTHING_SIGMA_CELLS
                * frame.grid.dx
                / ELECTRON_SKIN_DEPTH
            )
            axis.text(
                0.99,
                0.02,
                rf"display smoothing: $\sigma={smoothing_width:.2f}\,d_e$",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                color="white",
                fontsize="small",
                bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none"},
            )
            axis.set_xlabel(r"$x/d_e$")
            axis.set_ylabel(r"$z/d_e$")
            axis.set_aspect("equal", adjustable="box")
            axis.set_title(_frame_title(frame, "Reconnecting magnetic field"))
            writer.grab_frame()


def write_effective_resistivity_movie(
    reader: OpenPMDFieldReader, output_path: Path, scales: PlotScales, fps: int, dpi: int
):
    """Write the masked nonideal Ohm-law proxy in SI resistivity units."""

    _plt, _animation, colors = _matplotlib_modules()
    norm = colors.Normalize(vmin=-scales.eta_max, vmax=scales.eta_max)
    with _movie_writer(output_path, fps, dpi) as (plt, figure, axis, writer):
        colorbar = figure.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r"), ax=axis
        )
        colorbar.set_label(r"$\eta_{\mathrm{eff}}$ ($\Omega$ m)")
        for iteration_index in reader.iterations:
            frame = reader.read(iteration_index)
            x, z, eta = effective_resistivity(frame)
            crop = central_x_mask(x)
            axis.clear()
            axis.pcolormesh(
                x[crop] / ELECTRON_SKIN_DEPTH,
                z / ELECTRON_SKIN_DEPTH,
                eta[crop].T,
                cmap="RdBu_r",
                norm=norm,
                shading="auto",
            )
            message = None
            if frame.iteration == 0:
                message = "Iteration zero: J has not yet been deposited"
            elif not np.any(np.isfinite(eta[crop])):
                message = r"No cells satisfy $|J_y| \geq 0.02 J_0$"
            if message is not None:
                axis.text(
                    0.5,
                    0.5,
                    message,
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
                )
            axis.plot(0.0, 0.0, marker="x", color="black", markersize=7, mew=1.5)
            axis.set_xlabel(r"$x/d_e$")
            axis.set_ylabel(r"$z/d_e$")
            axis.set_aspect("equal", adjustable="box")
            axis.set_title(_frame_title(frame, "Effective nonideal resistivity"))
            writer.grab_frame()


def _write_velocity_cut_movie(
    reader: OpenPMDFieldReader,
    output_path: Path,
    scales: PlotScales,
    fps: int,
    dpi: int,
    direction: str,
):
    if direction not in {"x", "z"}:
        raise ValueError(f"Velocity cut direction must be x or z, received {direction!r}.")

    with _movie_writer(output_path, fps, dpi, figsize=(8.0, 5.0)) as (
        _plt,
        _figure,
        axis,
        writer,
    ):
        for iteration_index in reader.iterations:
            frame = reader.read(iteration_index)
            x_cut, z_cut = velocity_cuts(frame)
            if direction == "x":
                coordinate = frame.grid.x / ELECTRON_SKIN_DEPTH
                values = x_cut
                fixed_coordinate = frame.grid.z[nearest_axis_index(frame.grid.z)]
                subtitle = f"z = {fixed_coordinate / ELECTRON_SKIN_DEPTH:.3f} d_e"
            else:
                coordinate = frame.grid.z / ELECTRON_SKIN_DEPTH
                values = z_cut
                fixed_coordinate = frame.grid.x[nearest_axis_index(frame.grid.x)]
                subtitle = f"x = {fixed_coordinate / ELECTRON_SKIN_DEPTH:.3f} d_e"

            axis.clear()
            for component, label, color in zip(
                values,
                (r"$u_x/V_A$", r"$u_y/V_A$", r"$u_z/V_A$"),
                ("tab:blue", "tab:orange", "tab:green"),
            ):
                axis.plot(coordinate, component, label=label, color=color, lw=1.5)
            axis.axhline(0.0, color="0.35", lw=0.7)
            axis.set_xlim(float(coordinate[0]), float(coordinate[-1]))
            axis.set_ylim(-scales.velocity_max, scales.velocity_max)
            axis.set_xlabel(rf"${direction}/d_e$")
            axis.set_ylabel(r"Flow velocity / $V_A$")
            axis.legend(loc="upper right", ncol=3, fontsize="small")
            axis.grid(alpha=0.2)
            axis.set_title(_frame_title(frame, f"Flow-velocity {direction} cut ({subtitle})"))
            writer.grab_frame()


def write_all_movies(
    reader: OpenPMDFieldReader, output_dir: str | Path, scales: PlotScales, fps: int, dpi: int
) -> tuple[Path, Path, Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        output_dir / "field_lines.mp4",
        output_dir / "effective_resistivity.mp4",
        output_dir / "flow_velocity_x_cut.mp4",
        output_dir / "flow_velocity_z_cut.mp4",
    )
    write_field_line_movie(reader, paths[0], scales, fps, dpi)
    write_effective_resistivity_movie(reader, paths[1], scales, fps, dpi)
    _write_velocity_cut_movie(reader, paths[2], scales, fps, dpi, "x")
    _write_velocity_cut_movie(reader, paths[3], scales, fps, dpi, "z")
    return paths


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create normalized movies from the 2-D Harris-sheet field dump."
    )
    parser.add_argument(
        "--fields",
        type=Path,
        default=Path("data/fields.pmd"),
        help="openPMD series or PyPIC3D .pmd pointer (default: data/fields.pmd)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis"),
        help="directory for generated MP4 files (default: analysis)",
    )
    parser.add_argument("--fps", type=int, default=10, help="movie frame rate (default: 10)")
    parser.add_argument("--dpi", type=int, default=150, help="movie resolution (default: 150)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.fps <= 0 or args.dpi <= 0:
        raise ValueError(f"--fps and --dpi must be positive; received {args.fps} and {args.dpi}.")

    with OpenPMDFieldReader(args.fields) as reader:
        print(
            f"Reading {len(reader.iterations)} frame(s) from {reader.path}\n"
            f"B0={B0:.8g} T, d_e={ELECTRON_SKIN_DEPTH:.8g} m, "
            f"V_A={ALFVEN_SPEED:.8g} m/s ({ALFVEN_SPEED / LIGHT_SPEED:.6f} c)"
        )
        scales = scan_plot_scales(reader)
        paths = write_all_movies(reader, args.output_dir, scales, args.fps, args.dpi)

    for path in paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
