"""Plot unaveraged BSSN cuts from a complete charged-puncture checkpoint.

This is saved-data analysis: no JAX initialization or evolution is performed.
The frozen variables.py supplies the NamedTuple field order used by the
checkpoint PyTree. All nine stored components of both tensors are retained.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np


def plot_cuts(checkpoint, variables_source, output, index, zoom):
    output.mkdir(parents=True, exist_ok=True)
    schema = ast.parse(variables_source.read_text())
    names = {node.name: [item.target.id for item in node.body
                        if isinstance(item, ast.AnnAssign)]
             for node in schema.body if isinstance(node, ast.ClassDef)}
    state_names = names["BSSNVariables"]
    parameter_names = names["BSSNParameters"]
    with np.load(checkpoint, allow_pickle=False) as saved:
        metadata = json.loads(saved["metadata"].item())
        # checkpoint_tree ends with (state, evolution_parameters, G, time).
        count = len(saved.files) - 1
        first = count - len(state_names) - len(parameter_names) - 2
        mapping = {name: f"q{first+i}" for i, name in enumerate(state_names)}
        state = {name: saved[key] for name, key in mapping.items()}
        parameters = {name: float(saved[f"q{first+len(state_names)+i}"])
                      for i, name in enumerate(parameter_names)}
        time = float(saved[f"q{count-1}"])
    n = metadata["config"]["n"]
    expected = dict(conformal_metric=(3, 3), conformal_factor=(), traceless_K=(3, 3),
                    trace_K=(), conformal_connection=(3,), lapse=(), shift=(3,))
    assert metadata["version"] == 1 and abs(time-metadata["time"]) < 1.e-10
    assert set(state) == set(expected)
    for name, value in state.items():
        assert value.shape == expected[name] + (n, n, n), (name, value.shape)
        assert np.isfinite(value).all(), name
    dx = parameters["dx"]
    axes = [parameters[f"{a}_min"] + dx*np.arange(n) for a in "xyz"]
    point = [float(axis[i]) for axis, i in zip(axes, index)]

    # Components are never spatially averaged, symmetrized, or projected.
    groups = {}
    for name, symbol in (("conformal_metric", r"\tilde{\gamma}"), ("traceless_K", r"\tilde{A}")):
        groups[name] = [(f"{name}_{a}{b}", rf"${symbol}_{{{a}{b}}}$", state[name][i, j])
                        for i, a in enumerate("xyz") for j, b in enumerate("xyz")]
    groups["scalars_and_vectors"] = [
        ("conformal_factor", "$W$", state["conformal_factor"]),
        ("trace_K", "$K$", state["trace_K"]),
        ("lapse", r"$\alpha$", state["lapse"]),
    ] + [(f"{name}_{a}", rf"${symbol}^{{{a}}}$", state[name][i])
         for name, symbol in (("conformal_connection", r"\tilde{\Gamma}"), ("shift", r"\beta"))
         for i, a in enumerate("xyz")]

    cuts = {f"axis_{a}": axis for a, axis in zip("xyz", axes)}
    extrema = {}
    for components in groups.values():
        for name, label, value in components:
            extrema[name] = dict(min=float(value.min()), max=float(value.max()),
                                 at_failure_location=float(value[tuple(index)]))
            for normal in range(3):
                cuts[f"{name}_plane_{'xyz'[normal]}"] = np.take(value, index[normal], axis=normal)
                selection = [index[d] if d != normal else slice(None) for d in range(3)]
                cuts[f"{name}_line_{'xyz'[normal]}"] = value[tuple(selection)]
    np.savez_compressed(output/"cuts.npz", **cuts)

    figures = []
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    with PdfPages(output/"all_bssn_cuts.pdf") as pdf:
        for view, limit in (("full", None), ("inner", zoom)):
            for group, components in groups.items():
                for normal in range(3):
                    tangent = [d for d in range(3) if d != normal]
                    u, v = tangent
                    masks = [np.ones(n, dtype=bool) if limit is None else np.abs(axis) <= limit
                             for axis in axes]
                    fig, panels = plt.subplots(3, 3, figsize=(14, 12), layout="constrained")
                    for ax, (name, label, value) in zip(panels.flat, components):
                        plane = cuts[f"{name}_plane_{'xyz'[normal]}"]
                        shown = plane[np.ix_(masks[u], masks[v])]
                        lo, hi = float(shown.min()), float(shown.max())
                        if lo < 0 < hi:
                            bound = max(abs(lo), abs(hi))
                            norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
                            cmap = "RdBu_r"
                        else:
                            span = max(abs(lo), abs(hi), 1.)*1.e-12 if lo == hi else 0.
                            norm = Normalize(vmin=lo-span, vmax=hi+span)
                            cmap = "viridis"
                        # Pixel edges surround actual C nodes; nearest display
                        # preserves the grid-scale structure of the saved cut.
                        U, V = axes[u][masks[u]], axes[v][masks[v]]
                        im = ax.imshow(shown.T, origin="lower", interpolation="nearest",
                                       extent=(U[0]-dx/2, U[-1]+dx/2, V[0]-dx/2, V[-1]+dx/2),
                                       cmap=cmap, norm=norm, aspect="equal")
                        ax.plot(point[u], point[v], marker="x", color="magenta", ms=5)
                        ax.set(title=label, xlabel=f"{'xyz'[u]}/M", ylabel=f"{'xyz'[v]}/M")
                        fig.colorbar(im, ax=ax, shrink=.85, format="%.3g")
                    plane_name = ''.join('xyz'[d] for d in tangent)
                    fig.suptitle(f"t={time:.2f}M, last valid step {metadata['step']} | {group}\n"
                                 f"Raw {plane_name} cut at {'xyz'[normal]}={point[normal]:+.3f}M; "
                                 "magenta x: location of first nonfinite value on next step")
                    filename = f"{group}_{plane_name}_{view}.png"
                    fig.savefig(output/filename, dpi=150)
                    pdf.savefig(fig)
                    plt.close(fig)
                    figures.append(filename)

                fig, panels = plt.subplots(3, 3, figsize=(14, 11), layout="constrained")
                for ax, (name, label, value) in zip(panels.flat, components):
                    for direction, letter in enumerate("xyz"):
                        mask = np.ones(n, dtype=bool) if limit is None else np.abs(axes[direction]) <= limit
                        ax.plot(axes[direction][mask], cuts[f"{name}_line_{letter}"][mask],
                                ".-", ms=3, lw=1, label=f"{letter} direction")
                    ax.set(title=label, xlabel="varying coordinate / M")
                    ax.grid(alpha=.25)
                    ax.legend(fontsize=8)
                fig.suptitle(f"t={time:.2f}M | {group} | unaveraged lines through "
                             f"({point[0]:+.3f}, {point[1]:+.3f}, {point[2]:+.3f})M")
                filename = f"{group}_lines_{view}.png"
                fig.savefig(output/filename, dpi=150)
                pdf.savefig(fig)
                plt.close(fig)
                figures.append(filename)

    report = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  variables_source=str(variables_source), variables_source_sha256=hashlib.sha256(variables_source.read_bytes()).hexdigest(),
                  step=metadata["step"], time=time, dt=metadata["config"]["dt"], dx=dx,
                  index=list(index), cut_point=point, inner_half_width=zoom,
                  field_mapping=mapping, component_count=len(extrema), component_extrema=extrema,
                  figures=figures, pdf_pages=len(figures),
                  sampling="Raw native BSSN C nodes; no spatial averaging or field projection; linear color scales per panel/view")
    (output/"cuts_metadata.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("component_extrema", "figures")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--variables-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=int, nargs=3, default=(47, 47, 48))
    parser.add_argument("--zoom", type=float, default=2.)
    args = parser.parse_args()
    plot_cuts(args.checkpoint, args.variables_source, args.output, args.index, args.zoom)
