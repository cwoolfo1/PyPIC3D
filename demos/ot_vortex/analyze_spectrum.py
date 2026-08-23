"""Measure the late-time magnetic-energy spectrum of an OT run."""

from __future__ import annotations

import argparse
from collections import deque
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MU0 = 1.25663706212e-6


def magnetic_shell_spectrum(B, dx, dy, mu0=MU0):
    """Return annularly integrated 2-D magnetic energy ``E_B(k)``."""

    components = []
    for component in B:
        array = np.asarray(component, dtype=np.float64)
        array = np.squeeze(array)
        if array.ndim != 2:
            raise ValueError(f"Expected a 2-D magnetic component, got shape {array.shape}.")
        components.append(array - np.mean(array))

    nx, ny = components[0].shape
    if any(component.shape != (nx, ny) for component in components):
        raise ValueError("Magnetic components do not share one mesh shape.")

    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kmag = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    dk = min(2.0 * np.pi / (nx * dx), 2.0 * np.pi / (ny * dy))
    shell_index = np.floor(kmag / dk).astype(np.int64)

    normalization = float((nx * ny) ** 2)
    mode_energy = np.zeros((nx, ny), dtype=np.float64)
    for component in components:
        transform = np.fft.fft2(component)
        mode_energy += np.abs(transform) ** 2 / (2.0 * mu0 * normalization)

    flat_shell = shell_index.ravel()
    shell_energy = np.bincount(flat_shell, weights=mode_energy.ravel())
    shell_k_sum = np.bincount(flat_shell, weights=kmag.ravel())
    shell_count = np.bincount(flat_shell)
    valid = (shell_count > 0) & (np.arange(shell_count.size) > 0)
    k = shell_k_sum[valid] / shell_count[valid]
    spectrum = shell_energy[valid] / dk
    return k, spectrum


def fit_power_law(k, spectrum, kdi_min, kdi_max, di, minimum_shells=5):
    """Fit a fixed log-log interval and return explicit quality metadata."""

    k = np.asarray(k)
    spectrum = np.asarray(spectrum)
    kdi = k * di
    mask = (
        (kdi >= kdi_min)
        & (kdi <= kdi_max)
        & np.isfinite(spectrum)
        & (spectrum > 0.0)
    )
    count = int(np.count_nonzero(mask))
    result = {
        "shell_count": count,
        "kdi_min": float(kdi_min),
        "kdi_max": float(kdi_max),
        "slope": None,
        "intercept": None,
        "slope_standard_error": None,
        "r_squared": None,
        "bandwidth_sufficient": count >= minimum_shells,
        "mhd_like": False,
    }
    if count < minimum_shells:
        result["status"] = "insufficient inertial bandwidth"
        return result, mask

    x = np.log(k[mask])
    y = np.log(spectrum[mask])
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual = y - fitted
    sum_squared_error = float(np.sum(residual**2))
    sum_squared_total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - sum_squared_error / sum_squared_total if sum_squared_total else 1.0
    sxx = float(np.sum((x - np.mean(x)) ** 2))
    slope_error = np.sqrt(sum_squared_error / (count - 2) / sxx) if count > 2 and sxx else 0.0
    mhd_like = r_squared >= 0.8 and abs(slope + 5.0 / 3.0) <= 0.2
    result.update(
        {
            "slope": float(slope),
            "intercept": float(intercept),
            "slope_standard_error": float(slope_error),
            "r_squared": float(r_squared),
            "mhd_like": bool(mhd_like),
            "status": "MHD-like" if mhd_like else "fit outside acceptance bounds",
        }
    )
    return result, mask


def _load_component(series, iteration, component_name):
    record = iteration.meshes["B"][component_name]
    data = record.load_chunk()
    series.flush()
    return np.array(data, dtype=np.float64, copy=True)


def _iteration_time(iteration):
    return float(iteration.time) * float(iteration.time_unit_SI)


def averaged_openpmd_spectrum(fields_path, start_time, end_time, dx, dy):
    """Stream snapshots and average spectra of three-frame-smoothed fields."""

    import openpmd_api as io

    series = io.Series(str(fields_path), io.Access.read_only)
    spectra = []
    analyzed_times = []
    window = deque(maxlen=3)
    k_reference = None
    try:
        for iteration_index in sorted(series.iterations):
            iteration = series.iterations[iteration_index]
            time = _iteration_time(iteration)
            B = tuple(
                _load_component(series, iteration, component)
                for component in ("x", "y", "z")
            )
            window.append((time, B))
            if len(window) < 3:
                continue

            center_time = window[1][0]
            if center_time < start_time or center_time > end_time:
                continue
            averaged_B = tuple(
                sum(frame[1][component] for frame in window) / 3.0
                for component in range(3)
            )
            k, spectrum = magnetic_shell_spectrum(averaged_B, dx, dy)
            if k_reference is None:
                k_reference = k
            elif not np.allclose(k, k_reference):
                raise RuntimeError("The magnetic mesh changed between openPMD iterations.")
            spectra.append(spectrum)
            analyzed_times.append(center_time)
    finally:
        series.close()

    if not spectra:
        raise RuntimeError(
            f"No complete three-frame windows were found between {start_time} and {end_time} s."
        )
    return k_reference, np.mean(np.stack(spectra), axis=0), np.asarray(analyzed_times)


def load_conservation_summary(path, start_time, end_time):
    if not path.exists():
        return {"available": False}
    data = np.genfromtxt(path, delimiter=",", names=True)
    data = np.atleast_1d(data)
    mask = (data["time_s"] >= start_time) & (data["time_s"] <= end_time)
    selected = data[mask]
    if selected.size == 0:
        return {"available": True, "samples": 0}
    gauss = np.asarray(selected["gauss_residual"])
    energy_drift = np.asarray(selected["relative_energy_drift"])
    slope = float(np.polyfit(np.asarray(selected["time_s"]), gauss, 1)[0]) if selected.size > 1 else 0.0
    return {
        "available": True,
        "samples": int(selected.size),
        "median_gauss_residual": float(np.median(gauss)),
        "maximum_gauss_residual": float(np.max(gauss)),
        "gauss_residual_time_slope": slope,
        "maximum_relative_energy_drift": float(np.max(energy_drift)),
        "pilot_quality_pass": bool(
            np.median(gauss) < 1.0e-2
            and slope <= 0.0
            and np.max(energy_drift) < 0.05
        ),
    }


def write_spectrum_csv(path, k, spectrum, di, de, fit_mask):
    with open(path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("k_rad_m", "k_di", "k_de", "magnetic_energy_spectrum", "fit_shell"))
        for values in zip(k, k * di, k * de, spectrum, fit_mask):
            writer.writerow((*values[:-1], bool(values[-1])))


def plot_spectrum(path, k, spectrum, di, de, fit, fit_mask, nyquist):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].loglog(k, spectrum, marker=".", linewidth=1.0, label="magnetic energy")
    positive = np.flatnonzero(np.isfinite(spectrum) & (spectrum > 0.0))
    if positive.size:
        target_kdi = np.sqrt(fit["kdi_min"] * fit["kdi_max"])
        anchor = positive[np.argmin(np.abs(k[positive] * di - target_kdi))]
        reference = spectrum[anchor] * (k / k[anchor]) ** (-5.0 / 3.0)
        axes[0].loglog(k, reference, "--", linewidth=1.0, label=r"$k^{-5/3}$")
    if fit["slope"] is not None:
        fit_curve = np.exp(fit["intercept"]) * k[fit_mask] ** fit["slope"]
        axes[0].loglog(k[fit_mask], fit_curve, linewidth=2.0, label=f"fit: {fit['slope']:.3f}")
    axes[0].axvspan(
        fit["kdi_min"] / di,
        fit["kdi_max"] / di,
        alpha=0.12,
        color="tab:green",
    )
    axes[0].axvline(1.0 / di, color="tab:orange", linestyle=":", label=r"$kd_i=1$")
    axes[0].axvline(1.0 / de, color="tab:red", linestyle=":", label=r"$kd_e=1$")
    axes[0].axvline(nyquist, color="0.35", linestyle="-.", label="axis Nyquist")
    axes[0].set(xlabel=r"$k$ [rad/m]", ylabel=r"$E_B(k)$", title="Magnetic-energy spectrum")
    axes[0].legend(fontsize="small")
    axes[0].grid(True, which="both", alpha=0.25)

    compensated = spectrum * k ** (5.0 / 3.0)
    axes[1].semilogx(k * di, compensated, marker=".", linewidth=1.0)
    axes[1].axvspan(fit["kdi_min"], fit["kdi_max"], alpha=0.15, color="tab:green")
    axes[1].axvline(1.0, color="tab:orange", linestyle=":")
    axes[1].axvline(di / de, color="tab:red", linestyle=":")
    axes[1].axvline(nyquist * di, color="0.35", linestyle="-.")
    axes[1].set(
        xlabel=r"$kd_i$",
        ylabel=r"$k^{5/3}E_B(k)$",
        title="Compensated spectrum",
    )
    axes[1].grid(True, which="both", alpha=0.25)
    fig.suptitle(f"{fit['status']} ({fit['shell_count']} fit shells)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze(config_path, fields_path=None, output_dir=None):
    import toml

    config_path = Path(config_path).resolve()
    config = toml.load(config_path)
    simulation = config["simulation_parameters"]
    physics = config["ot_vortex"]
    base = config_path.parent
    if fields_path is None:
        fields_path = base / simulation["output_dir"] / "data" / "fields.h5"
    else:
        fields_path = Path(fields_path).resolve()
    if output_dir is None:
        output_dir = base / simulation["output_dir"] / "analysis"
    else:
        output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tau = float(physics["turnover_time"])
    start_time, end_time = 4.0 * tau, 6.0 * tau
    dx = float(simulation["x_wind"]) / int(simulation["Nx"])
    dy = float(simulation["y_wind"]) / int(simulation["Ny"])
    de = float(physics["electron_inertial_length"])
    di = float(physics["ion_inertial_length"])
    k, spectrum, analyzed_times = averaged_openpmd_spectrum(
        fields_path,
        start_time,
        end_time,
        dx,
        dy,
    )
    fit, fit_mask = fit_power_law(
        k,
        spectrum,
        float(physics["fit_kdi_min"]),
        float(physics["fit_kdi_max"]),
        di,
    )
    conservation_path = base / simulation["output_dir"] / "data" / "conservation.csv"
    conservation = load_conservation_summary(conservation_path, start_time, end_time)
    result = {
        "case": physics["case"],
        "fields_path": str(fields_path),
        "snapshot_count": int(analyzed_times.size),
        "analysis_time_min_s": float(np.min(analyzed_times)),
        "analysis_time_max_s": float(np.max(analyzed_times)),
        "L_over_de": float(simulation["x_wind"]) / de,
        "L_over_di": float(simulation["x_wind"]) / di,
        "axis_nyquist_rad_m": float(np.pi / min(dx, dy)),
        "axis_nyquist_kdi": float(np.pi / min(dx, dy) * di),
        "fit": fit,
        "conservation": conservation,
    }
    write_spectrum_csv(output_dir / "magnetic_spectrum.csv", k, spectrum, di, de, fit_mask)
    plot_spectrum(
        output_dir / "magnetic_spectrum.png",
        k,
        spectrum,
        di,
        de,
        fit,
        fit_mask,
        np.pi / min(dx, dy),
    )
    with open(output_dir / "magnetic_spectrum.json", "w") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--fields", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    analyze(args.config, args.fields, args.output_dir)


if __name__ == "__main__":
    main()
