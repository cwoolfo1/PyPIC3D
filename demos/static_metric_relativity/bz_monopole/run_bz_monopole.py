"""Run a fresh BZ monopole using settings in simulation_parameters.py.

Run from this directory with:
    python run_bz_monopole.py

The fixed-step explicit simulation runs from zero to SimulationParameters.end_time.
Diagnostic snapshots and final particle/field arrays are saved as NumPy files.
Runtime metric, finite-state, displacement, capacity, and exterior field-constraint
checks remain enabled. Interruptions leave completed snapshots in place.
"""
from dataclasses import asdict
import math
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from PyPIC3D.boundary_conditions.polar import divergence, divide, refresh_vector
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.relativity.core import B_FIELD_LOCATIONS, D_FIELD_LOCATIONS
from PyPIC3D.solvers.gr_static.static_metric import (
    compute_covariant_E, compute_covariant_H, update_B_relativity,
    update_D_relativity, _location_interpolate)
from PyPIC3D.solvers.gr_static.time_loop import time_loop_static_metric
from simulation_parameters import SimulationParameters, build_runtime, shard_array
from magnetization import measure_magnetization, collocate_magnetic_field
from plasma_injector import empty_particles, inject_pairs

def monopole_field(p, metric, dynamic):
    """Discrete curl of A_phi on D_phi=(C,C,V), yielding B_r=(C,V,V)."""
    theta = dynamic.grids.tiled_center_grid[1][..., None, :, None]
    flux = (-p.B0*jnp.cos(theta+dynamic.dy)+p.B0*jnp.cos(theta))/dynamic.dy
    Br = divide(jnp.broadcast_to(flux, metric.B[0].sqrt_gamma.shape)*dynamic.dy*dynamic.dz,
                metric.geometry.B_area[0])
    return (Br, jnp.zeros_like(Br), jnp.zeros_like(Br))


def refresh_fields(vector, locations, static):
    """Refresh polar D or B halos, including the frozen horizon layers."""
    return refresh_vector(vector, static, locations, 'B' if locations == B_FIELD_LOCATIONS else 'D')


def initialize_fields(p, static, dynamic, metric):
    B0 = tuple(shard_array(x, static) for x in monopole_field(p, metric, dynamic))
    B0=refresh_fields(B0, B_FIELD_LOCATIONS, static)
    zero = jnp.zeros_like(B0[0])
    D0 = (zero,)*3
    def rhs(D, B):
        E = compute_covariant_E(D, B, metric)
        H = compute_covariant_H(D, B, metric)
        db = update_B_relativity(E, (zero,)*3, metric, static, dynamic, 1.)
        dd = update_D_relativity((zero,)*3, H, (zero,)*3, metric, static, dynamic, 1.)
        return (refresh_fields(dd, D_FIELD_LOCATIONS, static),
                refresh_fields(db, B_FIELD_LOCATIONS, static))
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
    def damp(vector, baseline, locations):
        result = []
        for i, loc in enumerate(locations):
            grid = dynamic.grids.tiled_center_grid if loc[0] == "C" else dynamic.grids.tiled_vertex_grid
            r = grid[0][..., :, None, None]
            ramp = jnp.clip((r-p.sponge_start)/(p.r_max-p.sponge_start), 0., 1.)**4
            factor = jnp.exp(-p.sponge_rate*dynamic.dt*ramp)
            result.append(baseline[i]+factor*(vector[i]-baseline[i]))
        return refresh_fields(tuple(result), locations, static)
    zero = tuple(jnp.zeros_like(x) for x in D)
    return (damp(D, zero, D_FIELD_LOCATIONS), damp(B, background, B_FIELD_LOCATIONS))+fields[2:]


def make_step(p, species, static, dynamic, background, *, sponge=True,
              current_filter_passes=0):
    """Check deterministic kernels, leaving rejection sampling outside checkify.

    Errors cross the host boundary before any failed state can be consumed.
    """
    injection_steps = max(1, math.ceil(p.injection_interval/float(dynamic.dt)))
    @jax.jit
    def inject(particles, fields, key, index):
        metric = fields[6]
        mag = measure_magnetization(particles, species, fields[1], metric, static, dynamic)
        return inject_pairs(particles, species, mag, fields[0], fields[1], metric,
                            static, dynamic, p, key, index)
    def evolve(particles, fields):
        transform = None
        if current_filter_passes:
            from current_filter import filter_current
            transform = lambda current: filter_current(current, fields[6].geometry,
                                                       static, current_filter_passes)
        errors, (particles, fields, boundary) = time_loop_static_metric(
            particles, species, fields, static, dynamic, return_diagnostics=True,
            return_errors=True, current_transform=transform)
        if sponge:
            fields = apply_sponge(fields, background, p, static, dynamic)
        finite = finite_state(particles, fields)
        return errors, (particles, fields, boundary), finite
    checked = jax.jit(evolve)
    def execute(particles, fields, key, index):
        if int(index) % injection_steps == 0:
            # Key by injection event, not timestep: matched physical injection
            # schedules must draw the same candidates during dt refinement.
            event = index // injection_steps
            particles, key, report = inject(particles, fields, key, event)
            for error in report.errors:
                error.throw()
            requested, inserted, rejected = report.requested, report.inserted, report.rejected
            if bool(jnp.any(rejected)):
                raise RuntimeError(f"Injection capacity failure at step {index}: "
                                   f"requested={requested}, inserted={inserted}, rejected={rejected}")
        else:
            requested = inserted = rejected = jnp.zeros(p.devices, jnp.int64)
        errors, (particles, fields, boundary), finite = checked(particles, fields)
        errors.throw()
        if not bool(finite):
            raise FloatingPointError(f'Nonfinite particle or field state at step {index}')
        if bool(boundary.invalid_push):
            raise RuntimeError(f'Unsupported particle displacement at step {index}')
        if bool(fields[-1]):
            raise RuntimeError(f'Particle migration/capacity failure at step {index}')
        check_sharding(particles, fields, static)
        return particles, fields, key, (requested, inserted, rejected, boundary)
    return execute


def finite_state(particles, fields):
    valid = jnp.all(~particles.active | (jnp.isfinite(particles.x).all(-1)
                                        & jnp.isfinite(particles.u).all(-1)))
    for value in (*fields[0], *fields[1], *fields[2], fields[3], fields[4],
                  *fields[7][0], *fields[7][1]):
        valid &= jnp.isfinite(value).all()
    return valid


def check_sharding(particles, fields, static):
    from jax.sharding import NamedSharding, PartitionSpec
    layout = NamedSharding(static.field_mesh, PartitionSpec('tile_x', 'tile_y', 'tile_z'))
    for value in (*jax.tree.leaves(particles), *fields[0], *fields[1], *fields[2],
                  *fields[7][0], *fields[7][1]):
        if not value.sharding.is_equivalent_to(layout, value.ndim):
            raise RuntimeError('Particle or field state lost radial device sharding')


def exterior_mask(geometry, static, dynamic, p, *, magnetic=False):
    """Owned exterior cells checked for Gauss/div B; tile seams stay included.

    Excludes the horizon interior, two cells at each physical boundary, and the
    sponge plus two cells.
    """
    g = static.guard_cells
    if magnetic:
        owned = jnp.zeros_like(geometry.charge_owned).at[
            (slice(None),)*3+(slice(g,-g),slice(g,-g),slice(g,g+1))].set(True)
        grids = dynamic.grids.tiled_vertex_grid
    else:
        owned = geometry.charge_owned
        grids = dynamic.grids.tiled_center_grid
    r = grids[0][..., :, None, None]
    theta = grids[1][..., None, :, None]
    interior = owned & (r >= p.r_min+2*p.dr) & (r <= p.r_max-2*p.dr)
    interior &= r >= p.horizon
    interior &= (r < p.sponge_start-2*p.dr)
    interior &= (theta >= 2*p.dtheta) & (theta <= jnp.pi-2*p.dtheta)
    return interior


def constraint_residuals(particles, species, fields, static, dynamic, p, *, current_filter_passes=0):
    """Exterior integrated-flux Gauss and div B residuals, normalized by field scale.

    Cell counts distinguish an empty exterior from a valid zero residual.
    """
    geometry = fields[6].geometry
    charge = compute_rho(particles, species, fields[3], static, dynamic)
    charge *= 4*jnp.pi*dynamic.dx*dynamic.dy*dynamic.dz
    if current_filter_passes:
        from current_filter import smooth_integrated
        charge = smooth_integrated(charge, static, current_filter_passes)
    divD = divergence(fields[0], geometry, static)
    divB = divergence(fields[1], geometry, static, True)
    dm = exterior_mask(geometry, static, dynamic, p)
    bm = exterior_mask(geometry, static, dynamic, p, magnetic=True)
    def maximum(value, mask):
        return jnp.max(jnp.where(mask, jnp.abs(value), 0.))
    dscale = jnp.maximum(1., jnp.maximum(maximum(divD, dm), maximum(charge, dm)))
    bscale = jnp.maximum(1., jnp.max(jnp.array([
        maximum(b*a, bm) for b, a in zip(fields[1], geometry.B_area)])))
    return dict(gauss=jnp.where(jnp.any(dm), maximum(divD-charge, dm)/dscale, jnp.nan),
                magnetic_divergence=jnp.where(jnp.any(bm), maximum(divB, bm)/bscale, jnp.nan),
                gauss_cells=jnp.sum(dm), magnetic_divergence_cells=jnp.sum(bm))


def validate_constraint_settings(gauss_tolerance, magnetic_divergence_tolerance,
                                 constraint_check_interval=100):
    for name, value in (('gauss_tolerance', gauss_tolerance),
                        ('magnetic_divergence_tolerance', magnetic_divergence_tolerance)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if (isinstance(constraint_check_interval, bool)
            or not isinstance(constraint_check_interval, (int, np.integer))
            or constraint_check_interval <= 0):
        raise ValueError('constraint_check_interval must be a positive integer')


def check_constraints(residuals, allow_divergence_errors=False, *,
                      gauss_tolerance=1e-10, magnetic_divergence_tolerance=1e-10):
    """Only exterior residuals participate in constraint acceptance."""
    validate_constraint_settings(gauss_tolerance, magnetic_divergence_tolerance)
    exceeded = []
    for name, tolerance in (('gauss', gauss_tolerance),
                            ('magnetic_divergence', magnetic_divergence_tolerance)):
        value = float(residuals[name])
        if residuals.get(name+'_cells', 1) <= 0 or not math.isfinite(value):
            raise FloatingPointError(f'Nonfinite exterior {name} or empty exterior diagnostic region')
        if value >= tolerance:
            exceeded.append(f'{name}={value:.17g} >= {tolerance:.17g}')
    if exceeded and not allow_divergence_errors:
        raise RuntimeError('Exterior divergence acceptance failed: '+', '.join(exceeded))


def assemble(array, g=3, theta_base=True):
    """Host-only output boundary; production stepping never assembles the mesh."""
    host = np.asarray(jax.device_get(array))
    return np.concatenate([host[i, 0, 0, g:-g, g:(-g+1 if theta_base else -g), g] for i in range(host.shape[0])], axis=0)


def diagnostics(particles, species, fields, p, static, dynamic, *, current_filter_passes=0):
    D, B, _, _, _, _, metric, previous, _ = fields
    # Reconstruct B at the integer D/position time using the production field stage.
    Dhalf = tuple((D[i]+previous[0][i])/2 for i in range(3))
    Bminus = tuple((B[i]+previous[1][i])/2 for i in range(3))
    B = update_B_relativity(compute_covariant_E(Dhalf, B, metric), Bminus,
                            metric, static, dynamic, dynamic.dt)
    B = refresh_fields(B, B_FIELD_LOCATIONS, static)
    E = compute_covariant_E(D, B, metric)
    H = compute_covariant_H(D, B, metric)
    E = tuple(_location_interpolate(E[i], D_FIELD_LOCATIONS[i], ("C",)*3) for i in range(3))
    H = tuple(_location_interpolate(H[i], B_FIELD_LOCATIONS[i], ("C",)*3) for i in range(3))
    bc = collocate_magnetic_field(B)
    dc = jnp.stack([_location_interpolate(D[i], D_FIELD_LOCATIONS[i], ("C",)*3)
                    for i in range(3)], axis=-1)
    d2 = jnp.einsum('...i,...ij,...j->...', dc, metric.center.gamma, dc)
    db = jnp.einsum('...i,...ij,...j->...', dc, metric.center.gamma, bc)
    mag = measure_magnetization(particles, species, B, metric, static, dynamic)
    b2_safe = jnp.maximum(mag.magnetic_squared, jnp.finfo(d2.dtype).tiny)
    determinant = metric.center.sqrt_gamma
    denominator = determinant*bc[..., 0]
    omega = jnp.where(jnp.abs(denominator)>1e-12, -E[1]/denominator, jnp.nan)
    r = np.asarray(dynamic.grids.center[0][1:-1])
    theta = np.asarray(dynamic.grids.center[1][1:])
    # Uniform quadrature over one physical meridian, never both theta copies.
    sector = (theta>=0)&(theta<=np.pi)
    flux = assemble((E[1]*H[2]-E[2]*H[1])/(4*jnp.pi))
    luminosity = 2*np.pi*np.trapezoid(flux[:,sector],theta[sector],axis=1)
    divD = divergence(D, metric.geometry, static)
    divB = divergence(B, metric.geometry, static, True)
    rho = compute_rho(particles, species, fields[3], static, dynamic)*dynamic.dx*dynamic.dy*dynamic.dz
    if current_filter_passes:
        from current_filter import smooth_integrated
        rho = smooth_integrated(rho, static, current_filter_passes)
    constraints = assemble(divD-4*jnp.pi*rho)
    residuals = constraint_residuals(particles, species, fields, static, dynamic, p,
                                     current_filter_passes=current_filter_passes)
    return dict(r=r, theta=theta, Hphi=assemble(H[2]), omega=assemble(omega),
                Br=assemble(bc[..., 0]), Btheta=assemble(bc[..., 1]),
                radial_flux=assemble(determinant*bc[..., 0]),
                sigma=assemble(mag.sigma), density=assemble(jnp.sum(mag.number_density, axis=0)),
                D2_over_B2=assemble(d2/b2_safe), DdotB_over_B2=assemble(db/b2_safe),
                luminosity=luminosity, gauss=constraints, divB=assemble(divB,theta_base=False),
                gauss_relative=np.asarray(residuals['gauss']),
                divB_relative=np.asarray(residuals['magnetic_divergence']),
                active_per_tile=np.asarray(jnp.sum(particles.active, axis=(-1, -2))).reshape(-1))


def plot_diagnostics(snapshot, p, output):
    from plot_entity_bz import Normalization, make_figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    radii = tuple(r for r in (2., 3., 4., 5.) if snapshot['r'][0] <= r <= snapshot['r'][-1])
    figure, _ = make_figure(snapshot, Normalization.from_snapshot(snapshot), radii=radii)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def output_metadata(p, static, dynamic):
    """Flat scalar arrays shared by snapshots and the final analysis dump.

    NaN for maximum_timestep means there is no explicit cap on the CFL step.
    Stored particle positions are spherical and momenta are covariant and
    leapfrog-staggered.
    Field arrays retain their tiled Yee layout and guards.
    """
    data = {name: np.asarray(np.nan if value is None else value)
            for name, value in asdict(p).items()}
    data.update(dt=np.asarray(dynamic.dt), B0=np.asarray(p.B0),
                omega_h=np.asarray(p.omega_h), horizon=np.asarray(p.horizon),
                n0_total=np.asarray(p.n0),
                particle_batch_size=np.asarray(static.particle_batch_size),
                horizon_field_cells=np.asarray(static.horizon_field_cells))
    return data


def save_final_state(path, particles, species, fields, step, metadata, dynamic, budget):
    """Write analysis arrays without recovery history or executable objects."""
    arrays = dict(metadata, x=particles.x, u=particles.u, active=particles.active,
                  step=np.asarray(step), time=np.asarray(step*float(dynamic.dt)),
                  rho=fields[3], phi=fields[4], boundary_budget=budget)
    for label, vector in (("D", fields[0]), ("B", fields[1]), ("J", fields[2])):
        arrays.update({f"{label}_{i}": value for i, value in enumerate(vector)})
    for name in ('charge', 'mass', 'weight', 'update_x'):
        arrays['species_'+name] = getattr(species, name)
    for label, grids in (('center', dynamic.grids.tiled_center_grid),
                         ('vertex', dynamic.grids.tiled_vertex_grid)):
        arrays.update({f'grid_{label}_{axis}': value for axis, value in zip('xyz', grids)})
    np.savez(path, **{name: np.asarray(jax.device_get(value)) for name, value in arrays.items()})


def prepare_output_directory(output):
    output = Path(output).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Output directory must be empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    return output


def evolve(particles, species, fields, key, p, static, dynamic, background, output):
    """Evolve a fresh initial state to p.end_time, saving checked snapshots."""
    output = prepare_output_directory(output)
    dt = float(dynamic.dt)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    target = math.ceil(p.end_time/dt)
    metadata = output_metadata(p, static, dynamic)
    execute = make_step(p, species, static, dynamic, background,
                        current_filter_passes=p.current_filter_passes)
    measure = jax.jit(lambda pts, fs: constraint_residuals(
        pts, species, fs, static, dynamic, p, current_filter_passes=p.current_filter_passes))
    budget = np.zeros(11)
    next_snapshot = p.output_interval

    def check_state(pts, fs, step):
        try:
            check_constraints(measure(pts, fs), p.allow_divergence_errors,
                              gauss_tolerance=p.gauss_tolerance,
                              magnetic_divergence_tolerance=p.magnetic_divergence_tolerance)
        except (RuntimeError, FloatingPointError) as error:
            raise type(error)(f'Step {step}, t={step*dt:g}: {error}') from error

    def snapshot(step):
        data = diagnostics(particles, species, fields, p, static, dynamic,
                           current_filter_passes=p.current_filter_passes)
        data.update(metadata, step=np.asarray(step), time=np.asarray(step*dt),
                    boundary_budget=budget.copy())
        np.savez(output/f'snapshot_{step:012d}.npz', **data)
        return data

    check_sharding(particles, fields, static)
    if not bool(finite_state(particles, fields)):
        raise FloatingPointError('Nonfinite initial state')
    check_state(particles, fields, 0)
    snapshot(0)
    last_refresh = time.perf_counter()
    with tqdm(total=target, desc='BZ monopole', unit='step', mininterval=1.,
              dynamic_ncols=True) as progress:
        progress.set_postfix(time='0 M', active=int(particles.active.sum()), refresh=False)
        for step in range(1, target+1):
            new, newfields, newkey, report = execute(particles, fields, key, step-1)
            if step % p.constraint_check_interval == 0 or step == target:
                check_state(new, newfields, step)
            particles, fields, key = new, newfields, newkey
            _, _, _, boundary = report
            budget += np.concatenate((np.asarray(boundary.absorbed_count).reshape(-1),
                np.asarray(boundary.absorbed_charge).reshape(-1),
                [float(jnp.sum(boundary.removed_grid_charge))],
                np.asarray(boundary.radial_current_outflow)*dt))
            now = time.perf_counter()
            if now-last_refresh >= 1. or step == target:
                progress.set_postfix(time=f'{step*dt:g} M', active=int(particles.active.sum()),
                                     refresh=False)
                last_refresh = now
            progress.update(1)
            if step*dt >= next_snapshot or step == target:
                data = snapshot(step)
                next_snapshot = (math.floor(step*dt/p.output_interval)+1)*p.output_interval
    np.savez(output/'diagnostics.npz', **data)
    save_final_state(output/'final_state.npz', particles, species, fields, target,
                     metadata, dynamic, budget)
    plot_diagnostics(data, p, output/'figure6_diagnostics.png')
    return particles, fields, key, target


def run(parameters=None):
    p = parameters if parameters is not None else SimulationParameters()
    output = prepare_output_directory(p.output_directory)
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_platforms', 'cpu' if p.backend == 'cpu' else 'cuda')
    if jax.default_backend() not in (('cpu',) if p.backend == 'cpu' else ('gpu', 'cuda')):
        raise RuntimeError(f'The initialized JAX backend differs from backend={p.backend!r}')
    if len(jax.devices()) < p.devices:
        raise RuntimeError(f'Requested {p.devices} devices, but only {len(jax.devices())} are visible')
    if p.current_filter_passes and p.r_min+(p.current_filter_passes+2)*p.dr >= p.horizon:
        raise ValueError('Source filter and absorption stencil must fit inside the horizon; increase nr')
    static, dynamic, metric, _ = build_runtime(p)
    particles, species = empty_particles(p, static)
    fields, background = initialize_fields(p, static, dynamic, metric)
    return evolve(particles, species, fields, jax.random.PRNGKey(p.seed), p,
                  static, dynamic, background, output)


def main():
    import sys
    if len(sys.argv) != 1:
        raise SystemExit('This runner takes no arguments; edit simulation_parameters.py.')
    run(SimulationParameters())


if __name__ == '__main__':
    main()
