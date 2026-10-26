"""Plot saved diagnostics and RN comparisons; never advance a simulation."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_run(directory):
    directory = Path(directory)
    metadata = json.loads((directory/"metadata.json").read_text())
    mass, charge = (metadata["config"][k] for k in ("mass", "charge"))
    with (directory/"diagnostics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    data = {k: np.array([float(row[k]) for row in rows]) for k in rows[0]}
    fig, panels = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, name in zip(panels.flat, ("hamiltonian", "momentum", "gauss", "divB")):
        for region in ("exterior", "sponge"):
            kinds = ("raw_", "filtered_") if name in ("hamiltonian", "momentum") else ("",)
            for kind in kinds:
                values = data[f"{region}_{kind}{name}_l2"]
                ax.semilogy(data["time"], np.where(values > 0, values, np.nan),
                            label=f"{region} {kind.rstrip('_')}")
        ax.set(xlabel="t/M", ylabel=f"{name} RMS")
        ax.legend(fontsize=8)
        ax.grid(alpha=.25)
    fig.savefig(directory/"constraints.png", dpi=160)
    plt.close(fig)

    fig, panels = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for radius in (2, 4, 6):
        panels[0].plot(data["time"], data[f"charge_box_{radius}"], label=f"box half-width {radius}M")
    panels[0].axhline(charge, color="k", linestyle="--", label="input Q")
    panels[0].set(xlabel="t/M", ylabel="electric flux / (4 pi)")
    for name in ("D", "B"):
        values = data[f"{name}_max"]
        panels[1].semilogy(data["time"], np.where(values > 0, values, np.nan), label=name)
        panels[2].plot(data["time"], data[f"{name}_roughness"], label=name)
    panels[1].set(xlabel="t/M", ylabel="max native density")
    panels[2].set(xlabel="t/M", ylabel="normalized second-difference power (r<8M)")
    for ax in panels:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.savefig(directory/"charge_and_oscillations.png", dpi=160)
    plt.close(fig)

    latest = sorted(directory.glob("snapshot_*.npz"))[-1]
    with np.load(latest, allow_pickle=False) as saved:
        axis = saved["axis"]
        n = len(axis)
        def line(q):
            # Interpolate the four neighboring C lines to y=z=0.
            return q[..., :, n//2-1:n//2+1, n//2-1:n//2+1].mean(axis=(-2, -1))
        W, lapse, K = (line(saved[name]) for name in ("conformal_factor", "lapse", "trace_K"))
        gamma = line(saved["conformal_metric"] / saved["conformal_factor"]**2)
        beta = line(saved["shift"])
        electric = line(saved["D"] * saved["conformal_factor"]**3)
        timestamp = float(saved["time"])
    positive = axis > 0
    r = axis[positive]
    gamma = gamma[..., positive]
    beta = beta[:, positive]
    R = r*np.sqrt((gamma[1, 1]+gamma[2, 2])/2)
    dR = np.gradient(R, r, edge_order=2)
    gtt = -lapse[positive]**2 + np.einsum("ij...,i...,j...->...", gamma, beta, beta)
    gtr = np.einsum("j...,j...->...", gamma[0], beta)
    f = 1-2*mass/R+charge**2/R**2
    valid = (dR > 0) & (gtt < 0) & (R > mass+np.sqrt(mass**2-charge**2)) & (r < 8)
    reconstructed = (gamma[0, 0] - gtr**2/np.where(gtt != 0, gtt, np.nan))/dR**2
    fig, panels = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    panels[0, 0].plot(r, W[positive], label="W")
    panels[0, 0].plot(r, lapse[positive], label="lapse")
    panels[0, 0].plot(r, K[positive], label="K")
    panels[0, 0].set(xlabel="coordinate r/M", ylabel="evolved gauge/geometry")
    panels[0, 1].plot(R[valid], (electric[0, positive]*np.sqrt(gamma[0, 0]))[valid], label="FIDO radial E (axis)")
    panels[0, 1].plot(R[valid], charge/R[valid]**2, "--", label="Q/R²")
    panels[0, 1].set(xlabel="areal R/M (spherical approximation)", ylabel="electric field")
    panels[1, 0].plot(R[valid], gtt[valid], label="g_tt")
    panels[1, 0].plot(R[valid], -f[valid], "--", label="RN, fixed M,Q")
    panels[1, 0].set(xlabel="areal R/M", ylabel="g_tt")
    panels[1, 1].semilogy(R[valid], reconstructed[valid], label="stationary reconstruction")
    panels[1, 1].semilogy(R[valid], 1/f[valid], "--", label="RN, fixed M,Q")
    panels[1, 1].set(xlabel="areal R/M", ylabel="g_RR")
    for ax in panels.flat:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"t={timestamp:.3f}M; spherical axis profiles; metric comparison assumes stationarity")
    fig.savefig(directory/"radial_RN_comparison.png", dpi=160)
    plt.close(fig)
    report = dict(final_time=float(data["time"][-1]), input_mass=mass, input_charge=charge,
                  final_snapshot=latest.name, stationarity_verified=False)
    for radius in (2, 4, 6):
        q = data[f"charge_box_{radius}"]
        report[f"charge_box_{radius}"] = dict(initial=float(q[0]), final=float(q[-1]),
                                              max_drift=float(np.max(np.abs(q-q[0]))))
    (directory/"analysis.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    plot_run(parser.parse_args().directory)
