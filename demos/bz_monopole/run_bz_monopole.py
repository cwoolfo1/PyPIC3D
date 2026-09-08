"""Fresh BZ monopole runner. Default mode validates only; --mode run is explicit.

Examples (from the repository root)::

    python -m demos.bz_monopole.run_bz_monopole --mode validate --backend cpu --devices 1
    CUDA_VISIBLE_DEVICES=0,1 JAX_PLATFORMS=cuda python -m demos.bz_monopole.run_bz_monopole --mode smoke
"""
import argparse
from dataclasses import asdict
import json
import hashlib
import math
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from PyPIC3D.boundary_conditions.ghost_cells import update_tiled_vector_ghost_cells
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.relativity.core import B_FIELD_LOCATIONS, D_FIELD_LOCATIONS
from PyPIC3D.solvers.gr_static.static_metric import (
    compute_covariant_E, compute_covariant_H, update_B_relativity,
    update_D_relativity, _location_interpolate)
from PyPIC3D.solvers.gr_static.time_loop import time_loop_static_metric
from demos.bz_monopole.simulation_parameters import SimulationParameters, build_runtime, shard_array
from demos.bz_monopole.magnetization import measure_magnetization, collocate_magnetic_field
from demos.bz_monopole.plasma_injector import empty_particles, inject_pairs, check_species
from demos.bz_monopole.source_validation import check_source_orientation


def monopole_field(p, metric, dynamic):
    """Discrete curl of A_phi on D_phi=(C,C,V), yielding B_r=(C,V,V)."""
    theta = dynamic.grids.tiled_center_grid[1][..., None, :, None]
    flux = (-p.B0*jnp.cos(theta+dynamic.dy)+p.B0*jnp.cos(theta))/dynamic.dy
    if metric.geometry is not None:
        from PyPIC3D.boundary_conditions.polar import divide
        Br=divide(jnp.broadcast_to(flux,metric.B[0].sqrt_gamma.shape)*dynamic.dy*dynamic.dz,metric.geometry.B_area[0])
    else:
        Br = jnp.broadcast_to(flux, metric.B[0].sqrt_gamma.shape)/metric.B[0].sqrt_gamma
    return (Br, jnp.zeros_like(Br), jnp.zeros_like(Br))


def refresh_fields(vector, locations, metric, static):
    """Extrapolate densitized fields radially, exchange periodic/internal halos."""
    if metric.geometry is not None:
        from PyPIC3D.boundary_conditions.polar import refresh_vector
        return refresh_vector(vector,static,locations,'B' if locations==B_FIELD_LOCATIONS else 'D')
    factors = metric.B if locations == B_FIELD_LOCATIONS else metric.D
    weighted = tuple(vector[i]*factors[i].sqrt_gamma for i in range(3))
    weighted = update_tiled_vector_ghost_cells(weighted, static, static.guard_cells)
    return tuple(weighted[i]/factors[i].sqrt_gamma for i in range(3))


def initialize_fields(p, static, dynamic, metric):
    B0 = tuple(shard_array(x, static) for x in monopole_field(p, metric, dynamic))
    B0=refresh_fields(B0,B_FIELD_LOCATIONS,metric,static)
    zero = jnp.zeros_like(B0[0])
    D0 = (zero,)*3
    def rhs(D, B):
        E, H = compute_covariant_E(D, B, metric), compute_covariant_H(D, B, metric)
        db = update_B_relativity(E, (zero,)*3, metric, static, dynamic, 1.)
        dd = update_D_relativity((zero,)*3, H, (zero,)*3, metric, static, dynamic, 1.)
        return (refresh_fields(dd, D_FIELD_LOCATIONS, metric, static),
                refresh_fields(db, B_FIELD_LOCATIONS, metric, static))
    dD, dB = rhs(D0, B0)
    ddD, ddB = rhs(dD, dB)
    def at(initial, first, second, t):
        return tuple(x+t*v+0.5*t*t*a for x, v, a in zip(initial, first, second))
    dt = dynamic.dt
    Bhalf = at(B0, dB, ddB, -0.5*dt)
    previous = (at(D0, dD, ddD, -dt), at(B0, dB, ddB, -1.5*dt))
    # add_external_fields expects a pair of vectors (D_external, B_external).
    external = (D0, D0)
    return (D0, Bhalf, D0, zero, zero, external, metric, previous, jnp.asarray(False)), B0


def apply_sponge(fields, background, p, static, dynamic):
    """Damp deviations from the monopole; report constraint changes separately.

    This is an absorbing numerical layer, not a charge-conserving physical source.
    Constraint diagnostics therefore exclude it and its adjacent stencil cells.
    """
    D, B = fields[:2]
    metric = fields[6]
    def damp(vector, baseline, locations):
        result = []
        for i, loc in enumerate(locations):
            grid = dynamic.grids.tiled_center_grid if loc[0] == "C" else dynamic.grids.tiled_vertex_grid
            r = grid[0][..., :, None, None]
            ramp = jnp.clip((r-p.sponge_start)/(p.r_max-p.sponge_start), 0., 1.)**4
            factor = jnp.exp(-p.sponge_rate*dynamic.dt*ramp)
            result.append(baseline[i]+factor*(vector[i]-baseline[i]))
        return refresh_fields(tuple(result), locations, metric, static)
    zero = tuple(jnp.zeros_like(x) for x in D)
    return (damp(D, zero, D_FIELD_LOCATIONS), damp(B, background, B_FIELD_LOCATIONS))+fields[2:]


def make_step(p, species, static, dynamic, background):
    injection_steps = max(1, math.ceil(p.injection_interval/float(dynamic.dt)))
    def step(particles, fields, key, index):
        metric = fields[6]
        def inject(args):
            particles, key = args
            mag = measure_magnetization(particles, species, fields[1], metric, static, dynamic)
            new, key, report = inject_pairs(particles, species, mag, fields[0], fields[1],
                                            metric, static, dynamic, p, key, index)
            return new, key, report.requested, report.inserted, report.rejected
        def no_inject(args):
            particles, key = args
            zero = jnp.zeros(p.devices, jnp.int64)
            return particles, key, zero, zero, zero
        particles, key, requested, inserted, rejected = jax.lax.cond(
            index % injection_steps == 0, inject, no_inject, (particles, key))
        particles, fields, boundary = time_loop_static_metric(particles, species, fields, static, dynamic, return_diagnostics=True)
        fields = apply_sponge(fields, background, p, static, dynamic)
        return particles, fields, key, (requested, inserted, rejected, boundary)
    return jax.jit(step)


def assemble(array, g=3, theta_base=True):
    """Host-only output boundary; production stepping never assembles the mesh."""
    host = np.asarray(jax.device_get(array))
    return np.concatenate([host[i, 0, 0, g:-g, g:(-g+1 if theta_base else -g), g] for i in range(host.shape[0])], axis=0)


def diagnostics(particles, species, fields, p, static, dynamic):
    D, B, _, _, _, _, metric, previous, _ = fields
    # Reconstruct B at the integer D/position time using the production field stage.
    Dhalf = tuple((D[i]+previous[0][i])/2 for i in range(3))
    Bminus = tuple((B[i]+previous[1][i])/2 for i in range(3))
    B = update_B_relativity(compute_covariant_E(Dhalf, B, metric), Bminus,
                            metric, static, dynamic, dynamic.dt)
    B = refresh_fields(B, B_FIELD_LOCATIONS, metric, static)
    E, H = compute_covariant_E(D, B, metric), compute_covariant_H(D, B, metric)
    E = tuple(_location_interpolate(E[i], D_FIELD_LOCATIONS[i], ("C",)*3) for i in range(3))
    H = tuple(_location_interpolate(H[i], B_FIELD_LOCATIONS[i], ("C",)*3) for i in range(3))
    bc = collocate_magnetic_field(B, metric)
    mag = measure_magnetization(particles, species, B, metric, static, dynamic)
    rho = compute_rho(particles, species, fields[3], static, dynamic)
    divD = jnp.zeros_like(rho)
    divB = jnp.zeros_like(rho)
    for i, spacing in enumerate((dynamic.dx, dynamic.dy, dynamic.dz)):
        wd = metric.D[i].sqrt_gamma*D[i]
        wb = metric.B[i].sqrt_gamma*B[i]
        divD += (wd-jnp.roll(wd, 1, axis=i+3))/spacing
        divB += (jnp.roll(wb, -1, axis=i+3)-wb)/spacing
    determinant = metric.center.sqrt_gamma
    denominator = determinant*bc[..., 0]
    omega = jnp.where(jnp.abs(denominator)>1e-12, -E[1]/denominator, jnp.nan)
    r = np.asarray(dynamic.grids.center[0][1:-1])
    theta = np.asarray(dynamic.grids.center[1][1:])
    # Uniform quadrature over one physical meridian, never both theta copies.
    sector = (theta>=0)&(theta<=np.pi)
    flux = assemble((E[1]*H[2]-E[2]*H[1])/(4*jnp.pi))
    luminosity = 2*np.pi*np.trapezoid(flux[:,sector],theta[sector],axis=1)
    safe_r = (r>p.r_min+2*p.dr)&(r<p.sponge_start-2*p.dr)
    if metric.geometry is not None:
        from PyPIC3D.boundary_conditions.polar import divergence
        divD=divergence(D,metric.geometry,static)
        divB=divergence(B,metric.geometry,static,True)
        rho=rho*dynamic.dx*dynamic.dy*dynamic.dz
    constraints = assemble(divD-4*jnp.pi*rho)
    scale = max(float(np.max(np.abs(assemble(divD)))), float(np.max(np.abs(assemble(4*jnp.pi*rho)))), 1.)
    return dict(r=r, theta=theta, Hphi=assemble(H[2]), omega=assemble(omega),
                Br=assemble(bc[..., 0]), Btheta=assemble(bc[..., 1]),
                radial_flux=assemble(determinant*bc[..., 0]),
                sigma=assemble(mag.sigma), density=assemble(jnp.sum(mag.number_density, axis=0)),
                luminosity=luminosity, gauss=constraints, divB=assemble(divB,theta_base=False),
                gauss_relative=np.asarray(np.max(np.abs(constraints[safe_r]))/scale if safe_r.any() else np.nan),
                active_per_tile=np.asarray(jnp.sum(particles.active, axis=(-1, -2))).reshape(-1))


def save_checkpoint(path, particles, fields, key, step, p):
    """Portable array-only checkpoint; no pickle or executable serialized objects."""
    arrays = dict(checkpoint_version=np.asarray(2),geometry_id=np.asarray("polar-cap-v1"),x=particles.x, u=particles.u, active=particles.active, key=key,
                  step=np.asarray(step), parameters=np.asarray(json.dumps(asdict(p))))
    for label, vector in (("D", fields[0]), ("B", fields[1]), ("J", fields[2]),
                           ("previous_D", fields[7][0]), ("previous_B", fields[7][1])):
        arrays.update({f"{label}_{i}": value for i, value in enumerate(vector)})
    arrays.update(rho=fields[3], phi=fields[4], overflow=fields[8])
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **{k: np.asarray(jax.device_get(v)) for k, v in arrays.items()})
    temporary.replace(path)


def load_checkpoint(path, particles, fields, p, static):
    from PyPIC3D.particles.particle_tile_communication import shard_tiled_particles
    with np.load(path, allow_pickle=False) as saved:
        if 'checkpoint_version' not in saved or int(saved['checkpoint_version']) != 2 or str(saved['geometry_id']) != 'polar-cap-v1':
            raise ValueError('Checkpoint requires polar-cap-v1 geometry; full-theta restarts are incompatible')
        if json.loads(str(saved["parameters"])) != asdict(p):
            raise ValueError("Checkpoint parameters do not match this run")
        particles = shard_tiled_particles(particles._replace(**{k:jnp.asarray(saved[k]) for k in ("x", "u", "active")}), static)
        def vector(label):
            return tuple(shard_array(saved[f"{label}_{i}"], static) for i in range(3))
        fields = (vector("D"), vector("B"), vector("J"), shard_array(saved["rho"], static),
                  shard_array(saved["phi"], static), fields[5], fields[6],
                  (vector("previous_D"), vector("previous_B")), jnp.asarray(saved["overflow"]))
        return particles, fields, jnp.asarray(saved["key"]), int(saved["step"])


def plot_diagnostics(snapshot, p, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    r, theta = snapshot["r"], snapshot["theta"]
    sector = (theta>=0)&(theta<=np.pi)
    t = theta[sector]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5), constrained_layout=True)
    rr, tt = np.meshgrid(r, t, indexing="ij")
    signal = -snapshot["Hphi"][:, sector]/p.B0
    # Supply explicit curvilinear corners; Matplotlib cannot infer these reliably
    # from non-monotone Cartesian coordinates of polar cell centers.
    redges = np.r_[r-p.dr/2, r[-1]+p.dr/2]
    tedges = np.r_[t-p.dtheta/2, t[-1]+p.dtheta/2]
    redges[0], redges[-1] = p.r_min, p.r_max
    tedges[0], tedges[-1] = 0., np.pi
    re, te = np.meshgrid(redges,tedges,indexing="ij")
    mesh = axes[0].pcolormesh(re*np.sin(te), re*np.cos(te), np.ma.masked_less_equal(signal,0),
                               shading="flat", norm=LogNorm(1e-4,max(.05,float(np.nanmax(signal)))))
    fig.colorbar(mesh, ax=axes[0])
    # Flux contours obtained from the measured radial flux, not imposed radial lines.
    flux = np.cumsum(snapshot["radial_flux"][:, sector], axis=1)*p.dtheta
    if np.ptp(flux)>0:
        axes[0].contour(rr*np.sin(tt), rr*np.cos(tt), flux, levels=12, colors="white", linewidths=.5)
    axes[0].set(aspect="equal", xlabel="x / rg", ylabel="z / rg", title="-Hphi / B0")
    for radius in (2, 3, 4, 5):
        index = int(np.argmin(abs(r-radius)))
        axes[1].plot(t, snapshot["Hphi"][index, sector]/p.B0, label=f"r={r[index]:.2f}")
        axes[2].plot(t, snapshot["omega"][index, sector]/p.omega_h if p.omega_h else np.full_like(t,np.nan))
    axes[1].plot(t, -p.spin*np.sin(t)**2/8, "k--", label="BZ leading order")
    axes[1].legend(fontsize=7)
    axes[1].set(xlabel="theta", title="Hphi / B0")
    axes[2].axhline(.5, color="k", linestyle="--")
    axes[2].set(xlabel="theta", title="Omega / OmegaH", ylim=(-.1, 1.1))
    lbz = p.B0**2*p.omega_h**2/6
    axes[3].plot(r, snapshot["luminosity"]/lbz if lbz else np.full_like(r,np.nan))
    axes[3].axhline(1., color="k", linestyle="--")
    axes[3].set(xlabel="r / rg", title="LEM / LBZ")
    fig.suptitle(f"BZ diagnostics, t={float(snapshot.get('time', 0)):.6g} rg/c (not a convergence claim)")
    fig.savefig(output, dpi=150)
    plt.close(fig)


def comparison_errors(snapshot, p):
    """Report numerical comparisons without declaring an early snapshot converged."""
    theta = snapshot["theta"]
    angular = (theta >= np.pi/12) & (theta <= 11*np.pi/12)
    result = {"polar_exclusion_degrees": 15, "time": float(snapshot["time"]), "profiles": []}
    analytic = -p.spin*np.sin(theta[angular])**2/8
    scale = float(np.max(np.abs(analytic))) if p.spin else 0.
    for radius in (2, 3, 4, 5):
        i = int(np.argmin(abs(snapshot["r"]-radius)))
        result["profiles"].append(dict(radius=float(snapshot["r"][i]),
            H_relative_l2=float(np.linalg.norm(snapshot["Hphi"][i,angular]/p.B0-analytic)/np.linalg.norm(analytic)) if scale else None,
            omega_relative_rms=float(np.sqrt(np.mean((snapshot["omega"][i,angular]/p.omega_h/.5-1)**2))) if p.omega_h else None))
    radial = (snapshot["r"]>=2) & (snapshot["r"]<p.sponge_start-2*p.dr)
    lbz = p.B0**2*p.omega_h**2/6
    result["luminosity_relative_rms"] = float(np.sqrt(np.mean((snapshot["luminosity"][radial]/lbz-1)**2))) if lbz and radial.any() else None
    result["late_time_comparison"] = float(snapshot["time"]) >= 150.
    return result


def run(args):
    p = SimulationParameters(nr=args.nr, ntheta=args.ntheta, devices=args.devices)
    if args.backend == "gpu":
        if p.devices != 2 or len(jax.devices("gpu")) != 2:
            raise RuntimeError("GPU mode requires exactly two visible CUDA GPUs")
    elif jax.default_backend() != "cpu":
        raise RuntimeError("CPU test mode requires JAX_PLATFORMS=cpu before launch")
    s, d, m, report = build_runtime(p)
    orientation = check_source_orientation(p,s,d,m,include_orbit=args.mode in ("validate","smoke","run"))
    report["physical_source_orientation"] = orientation
    print(json.dumps(report, indent=2), flush=True)
    if args.mode in ("run", "smoke", "validate") and not orientation["passed"]:
        raise RuntimeError("Polar validation failed. BZ execution is blocked; "
                           "see source_validation.py and the verification report. Complete the polar geometry, source and orbit tests first.")
    if args.mode == "validate":
        assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(m))
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(report, mode=args.mode, status="initializing")
    manifest["revision"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    manifest["working_tree_status"] = subprocess.run(["git", "status", "--short"], capture_output=True, text=True).stdout
    manifest["demo_source_sha256"] = {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                                      for file in sorted(Path(__file__).parent.glob("*.py"))}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2))
    particles, species = empty_particles(p, s)
    check_species(species)
    fields, background = initialize_fields(p, s, d, m)
    key = jax.random.PRNGKey(p.seed)
    index = 0
    if args.restart:
        particles, fields, key, index = load_checkpoint(Path(args.restart), particles, fields, p, s)
    step = make_step(p, species, s, d, background)
    count = 0 if args.mode == "initialize" else (32 if args.mode == "smoke" else math.ceil(p.end_time/float(d.dt))-index)
    if args.steps is not None:
        if args.mode not in ("smoke", "run") or args.steps <= 0:
            raise ValueError("--steps must be positive and requires smoke or run mode")
        count = args.steps
    records = []
    started = time.perf_counter()
    try:
        for iteration in range(count):
            before = time.perf_counter()
            particles, fields, key, injection = step(particles, fields, key, jnp.asarray(index))
            jax.block_until_ready(particles.x)
            if len(particles.x.sharding.device_set) != p.devices:
                raise RuntimeError("Particle state lost its requested device sharding")
            if bool(fields[-1]) or int(jnp.sum(injection[2])) or bool(injection[3].invalid_push):
                raise RuntimeError(f"Particle displacement/capacity/communication failure at step {index}")
            if not all(bool(jnp.all(jnp.isfinite(x))) for x in (*fields[0], *fields[1], particles.x, particles.u)):
                raise FloatingPointError(f"Nonfinite state at step {index}")
            index += 1
            record = dict(step=index, time=index*float(d.dt), seconds=time.perf_counter()-before,
                          inserted=int(jnp.sum(injection[1])), requested=int(jnp.sum(injection[0])),
                          absorbed_count=np.asarray(injection[3].absorbed_count).tolist(),
                          absorbed_charge=np.asarray(injection[3].absorbed_charge).tolist(),
                          removed_grid_charge=float(injection[3].removed_grid_charge.sum()),
                          radial_current_outflow=np.asarray(injection[3].radial_current_outflow).tolist(),
                          active_per_tile=np.asarray(jnp.sum(particles.active, axis=(-1,-2))).reshape(-1).tolist())
            records.append(record)
            if iteration == 0 or index%8 == 0:
                print(json.dumps(record), flush=True)
            if args.mode == "run" and index % max(1, round(p.output_interval/float(d.dt))) == 0:
                snap = diagnostics(particles, species, fields, p, s, d)
                snap["time"] = index*float(d.dt)
                np.savez(output/f"snapshot_{index:08d}.npz", **snap)
                save_checkpoint(output/"checkpoint.npz", particles, fields, key, index, p)
        snap = diagnostics(particles, species, fields, p, s, d)
        snap["time"] = index*float(d.dt)
        np.savez(output/"diagnostics.npz", **snap)
        (output/"comparison.json").write_text(json.dumps(comparison_errors(snap,p),indent=2))
        plot_diagnostics(snap, p, output/"figure6_diagnostics.png")
        save_checkpoint(output/"checkpoint.npz", particles, fields, key, index, p)
        manifest.update(status="execution_passed" if args.mode == "smoke" else "completed", steps=index,
                        physics_validated=False, component_validation=orientation,
                        elapsed_seconds=time.perf_counter()-started,
                        gauss_relative=float(snap["gauss_relative"]),
                        first_step_compile_and_execution_seconds=records[0]["seconds"] if records else None,
                        median_subsequent_step_seconds=float(np.median([r["seconds"] for r in records[1:]])) if len(records)>1 else None,
                        particle_sharding=str(particles.x.sharding),
                        device_memory=[device.memory_stats() for device in s.field_mesh.devices.flat])
    except Exception as exc:
        manifest.update(status="failed", error=str(exc), step=index)
        save_checkpoint(output/"failed_checkpoint.npz", particles, fields, key, index, p)
        raise
    finally:
        (output/"timings.json").write_text(json.dumps(records, indent=2))
        (output/"manifest.json").write_text(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "initialize", "smoke", "run", "analyze"), default="validate")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--nr", type=int, default=64)
    parser.add_argument("--ntheta", type=int, default=128)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--restart")
    parser.add_argument("--output", default="demos/bz_monopole/runs/smoke")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    if args.self_test:
        import unittest
        from tests.code_tests.bz_monopole_test import TestRunner
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TestRunner))
        if not result.wasSuccessful():
            raise SystemExit(1)
    elif args.mode == "analyze":
        output = Path(args.output)
        p = SimulationParameters(**json.loads((output/"manifest.json").read_text())["parameters"])
        with np.load(output/"diagnostics.npz", allow_pickle=False) as snap:
            plot_diagnostics(dict(snap), p, output/"figure6_diagnostics.png")
    else:
        run(args)


if __name__ == "__main__":
    main()
