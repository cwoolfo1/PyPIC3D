"""Compare saved thermal-plasma histories; neither simulator runs here."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def history(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    def number(value):
        return float(value == "True") if value in ("True", "False") else float(value)
    return {name: [number(row[name]) for row in rows] for name in rows[0]}


def metrics(values, times):
    error = [(value - values[0]) / abs(values[0]) for value in values]
    upstream = [value for value, time in zip(error, times)
                if abs(time/20.0 - round(time/20.0)) < 1e-8]
    return dict(initial_total=values[0], final_signed=error[-1],
                maximum=max(abs(value) for value in error),
                rms=math.sqrt(sum(value*value for value in error)/len(error)),
                upstream_cadence_maximum=max(abs(value) for value in upstream),
                upstream_cadence_samples=len(upstream),
                samples=len(error)), error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    runs, curves, comparisons = [], {}, []
    for code in ("warpx", "pypic3d"):
        filename = "energy.csv" if code == "warpx" else "history.csv"
        for path in sorted((root/code).glob("*/" + filename)):
            directory = path.parent
            if directory.name.startswith(("export_", "native")):
                continue
            metadata = json.loads((directory/("metadata.json" if code == "warpx" else "manifest.json")).read_text())
            if code == "pypic3d":
                status = json.loads((directory/"status.json").read_text())
                if status["status"] != "complete":
                    raise RuntimeError(f"Incomplete run: {directory}")
            data = history(path)
            total = data["total" if code == "warpx" else "total_sync"]
            metric, error = metrics(total, data["wpe_t"])
            dim = metadata["dimension"]
            solver = "es" if metadata["solver"] == "es" else "em"
            shape = metadata.get("particle_shape", metadata.get("shape_factor"))
            if code == "warpx":
                gather = "galerkin" if metadata["galerkin_scheme"] else "uniform"
                if metadata["field_gathering"] == "momentum-conserving":
                    gather = "momentum"
            else:
                gather = metadata["field_gather"]
            dt = metadata["dt_wpe"]
            run = dict(code=code, case=directory.name, dimension=dim, solver=solver,
                       shape=shape, gather=gather, dt_wpe=dt, **metric,
                       final_wpe_t=data["wpe_t"][-1], path=str(directory),
                       initial_sha256=metadata.get("initial_array_sha256", metadata.get("initial_sha256")))
            if abs(run["final_wpe_t"] - 100.0) > 1e-8:
                raise RuntimeError(f"Run does not reach omega_pe*t=100: {directory}")
            if code == "pypic3d":
                run.update(max_poisson_residual=status["max_poisson_residual_normalized"],
                           poisson_maxiter_hits=status["poisson_maxiter_hits"])
                for diagnostic in ("total_raw", "total_raw_stable", "total_modified", "total_poisson"):
                    alternative, _ = metrics(data[diagnostic], data["wpe_t"])
                    run[diagnostic+"_initial_total"] = alternative["initial_total"]
                    run[diagnostic+"_maximum"] = alternative["maximum"]
                    run[diagnostic+"_final_signed"] = alternative["final_signed"]
                    run[diagnostic+"_rms"] = alternative["rms"]
                    run[diagnostic+"_upstream_cadence_maximum"] = alternative["upstream_cadence_maximum"]
            runs.append(run)
            curves[(code, dim, solver, shape, gather, dt)] = (data["wpe_t"], total, error)

    for run in runs:
        if run["code"] != "pypic3d" or run["gather"] != "galerkin":
            continue
        key = (run["dimension"], run["solver"], run["shape"], "galerkin", run["dt_wpe"])
        tp, hp, _ = curves[("pypic3d",) + key]
        tw, hw, _ = curves[("warpx",) + key]
        reference = next(candidate for candidate in runs if candidate["code"] == "warpx"
                         and all(candidate[name] == run[name] for name in
                                 ("dimension", "solver", "shape", "gather", "dt_wpe")))
        if reference["initial_sha256"] != run["initial_sha256"]:
            raise ValueError(f"Unmatched phase-space input: {run['case']}")
        if len(tp) != len(tw) or max(abs(a-b) for a,b in zip(tp,tw)) > 1e-9:
            raise ValueError(f"Unmatched sample times: {run['case']}")
        mismatch = max(abs(a-b) for a,b in zip(hp,hw)) / abs(hw[0])
        limit = 1e-8 if run["solver"] == "es" else 1e-10
        comparisons.append(dict(case=run["case"], max_history_difference_over_H0=mismatch,
                                threshold=limit, pass_history=mismatch < limit))

    refinement = []
    for run in runs:
        if run["gather"] != "galerkin" or run["dt_wpe"] not in (.2, .1):
            continue
        if run["shape"] == 2 and run["dt_wpe"] != .2:
            continue
        fine = next(candidate for candidate in runs if all(candidate[name] == run[name]
                    for name in ("code", "dimension", "solver", "shape", "gather"))
                    and candidate["dt_wpe"] == run["dt_wpe"]/2)
        ratio = fine["maximum"] / run["maximum"]
        refinement.append(dict(case=run["case"], ratio_half_dt=ratio,
                               coarse_dt_wpe=run["dt_wpe"], fine_dt_wpe=fine["dt_wpe"],
                               observed_order=-math.log2(ratio),
                               quadratic_bound_pass=run["maximum"] < .003 if run["shape"] == 2 else None,
                               quadratic_refinement_pass=ratio < .30 if run["shape"] == 2 else None))

    controls = []
    if args.legacy_root:
        for run in runs:
            if run["code"] != "pypic3d" or run["gather"] != "uniform":
                continue
            solver = "es" if run["solver"] == "es" else "esirkepov"
            old = args.legacy_root/"pypic3d"/f"{run['dimension']}d_{solver}_dt{run['dt_wpe']:g}"/"history.csv"
            before = history(old)["total_sync"]
            after = curves[("pypic3d",run["dimension"],run["solver"],run["shape"],"uniform",run["dt_wpe"])][1]
            if len(before) != len(after):
                raise ValueError(f"Unmatched legacy sample count: {run['case']}")
            difference = max(abs(a-b) for a,b in zip(before,after))/abs(before[0])
            controls.append(dict(case=run["case"], archived_history=str(old.resolve()),
                                 archived_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
                                 max_history_difference_over_H0=difference,
                                 threshold=1e-10, pass_history=difference < 1e-10))

    checks = dict(history_comparisons=comparisons, timestep_refinement=refinement,
                  legacy_controls=controls,
                  note="History differences use actual H0. Quadratic thresholds do not apply to linear shapes.")
    (root/"checks.json").write_text(json.dumps(checks, indent=2)+"\n")
    (root/"summary.json").write_text(json.dumps(runs, indent=2)+"\n")
    columns = sorted(set().union(*(run.keys() for run in runs)))
    with (root/"summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(runs)

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .2})
    colors = {0.2: "#2b6f9b", 0.1: "#cb6335", 0.05: "#63843e"}
    for shape in (1, 2):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        for row, dim in enumerate((1, 2)):
            for col, solver in enumerate(("es", "em")):
                ax = axes[row, col]
                for dt in ((.2, .1, .05) if shape == 1 else (.2, .1)):
                    for code in ("warpx", "pypic3d"):
                        key = (code, dim, solver, shape, "galerkin", dt)
                        if key not in curves:
                            continue
                        t, _, error = curves[key]
                        ax.plot(t, [100*value for value in error], color=colors[dt],
                                linestyle="-" if code == "pypic3d" else "--",
                                linewidth=1.0, label=f"{code}, dt wpe={dt:g}")
                ax.set_title(f"{dim}D {solver.upper()}, shape {shape}")
                ax.set_ylabel("(H(t)-H(0))/H(0) [%]")
                ax.set_xlabel("omega_pe t")
                ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(root/f"energy_shape{shape}.png", dpi=180)
        fig.savefig(root/f"energy_shape{shape}.pdf")
        plt.close(fig)

    if any(run["gather"] == "uniform" for run in runs):
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = []
        for index, (dim, solver) in enumerate(((1, "es"), (1, "em"), (2, "es"), (2, "em"))):
            labels.append(f"{dim}D {solver.upper()}")
            for offset, gather, color in ((-.18, "uniform", "#a76853"), (.18, "galerkin", "#2b6f9b")):
                run = next(candidate for candidate in runs if candidate["code"] == "pypic3d"
                           and candidate["dimension"] == dim and candidate["solver"] == solver
                           and candidate["shape"] == 2 and candidate["gather"] == gather
                           and candidate["dt_wpe"] == .2)
                value = 100*run["maximum"]
                ax.bar(index+offset, value, width=.34, color=color,
                       label=gather if index == 0 else None)
                ax.text(index+offset, value*1.15, f"{value:.4g}%", ha="center", fontsize=9)
        ax.set_xticks(range(4), labels)
        ax.set_yscale("log")
        ax.set_ylim(.005, 1500)
        ax.set_ylabel("Maximum absolute energy error [%]")
        ax.set_title("Quadratic gathering, omega_pe dt=0.2, omega_pe t=100")
        ax.legend()
        fig.tight_layout()
        fig.savefig(root/"uniform_vs_galerkin.png", dpi=180)
        fig.savefig(root/"uniform_vs_galerkin.pdf")
        plt.close(fig)

    failures = [check for check in comparisons + controls if not check["pass_history"]]
    failures += [check for check in refinement if check["quadratic_bound_pass"] is False
                 or check["quadratic_refinement_pass"] is False]
    report = ["PyPIC3D Galerkin / WarpX energy-conserving comparison", "",
              "Periodic thermal plasma; identical seed-1 phase space, 100 eV, eight cells per active axis.",
              "Duration omega_pe*t=100. Every-step diagnostics, CPU float64, one OpenMP thread.",
              "Primary H = electric + magnetic + synchronized stable relativistic kinetic energy.",
              "All relative errors divide by the actual initial total H0; no denominator floor.", "",
              f"Completed runs: {len(runs)}. Matched history comparisons: {len(comparisons)}.",
              f"Failed acceptance checks: {len(failures)}.", "",
              "PyPIC3D errors (percent); WarpX agrees within the per-history limits below.",
              "dimension solver shape omega_pe_dt maximum_abs final_signed RMS upstream_cadence_maximum"]
    for run in runs:
        if run["code"] == "pypic3d" and run["gather"] == "galerkin":
            report.append(f"{run['dimension']}D {run['solver']:2} {run['shape']} {run['dt_wpe']:.3g} "
                          f"{100*run['maximum']:.10g} {100*run['final_signed']:.10g} {100*run['rms']:.10g} "
                          f"{100*run['upstream_cadence_maximum']:.10g}")
    report += ["", "Cross-code maximum absolute total-history difference / WarpX H0:"]
    for solver in ("es", "esirkepov"):
        subset = [item for item in comparisons if f"_{solver}_" in item["case"]]
        if subset:
            report.append(f"{solver}: {max(item['max_history_difference_over_H0'] for item in subset):.12g}")
    report += ["", "Quadratic dt-halving maximum-error ratios (PyPIC3D):"]
    for item in refinement:
        if "_p2_galerkin_" in item["case"]:
            report.append(f"{item['case']}: {item['ratio_half_dt']:.10g}")
    report += ["", "Linear measured refinement (PyPIC3D; same behavior in WarpX):"]
    for item in refinement:
        if "_p1_galerkin_" in item["case"]:
            report.append(f"{item['case']}: dt {item['coarse_dt_wpe']:g} -> {item['fine_dt_wpe']:g}, "
                          f"max-error ratio {item['ratio_half_dt']:.10g}, observed order {item['observed_order']:.10g}")
    report += ["", "Explicit uniform quadratic controls at omega_pe*dt=0.2:"]
    for run in runs:
        if run["code"] == "pypic3d" and run["gather"] == "uniform":
            report.append(f"{run['case']}: max absolute energy error {100*run['maximum']:.10g}%")
    report += ["", f"Uniform controls compared against the original archive: {len(controls)}; details in checks.json.",
               "Uniform EM uses native staggered fields, not WarpX's nodally centered momentum-conserving gather.",
               "Linear shape results are empirical: native WarpX also shows non-monotonic refinement in some cases.",
               "The 0.3 percent bound and <0.30 half-step ratio apply only to quadratic cases.",
               "These results validate the spatial gather/ES gradient change for this benchmark; they do not prove exact conservation.",
               "Diagnostic alternatives remain in each PyPIC3D history.csv and do not feed the trajectory."]
    report += ["Upstream cadence means physical times omega_pe*t=0,20,...100 for every timestep, not every 100 fine steps.",
               "summary.csv also includes raw, stable raw, modified, and Poisson diagnostic H0, maxima, final errors, RMS and upstream-cadence maxima."]
    (root/"REPORT.txt").write_text("\n".join(report)+"\n")
    print(json.dumps(dict(runs=len(runs), comparisons=len(comparisons),
                         legacy_controls=len(controls), failed_checks=failures), indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
