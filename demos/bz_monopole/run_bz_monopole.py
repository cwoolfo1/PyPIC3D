"""Fresh BZ monopole runner. Default mode validates only; --mode run is explicit.

Examples (from the repository root)::

    JAX_PLATFORMS=cpu python -m demos.bz_monopole.run_bz_monopole --mode validate --backend cpu --devices 1 --nr 32 --ntheta 32
    CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false python -m demos.bz_monopole.run_bz_monopole --mode run --output data

The runner owns initialization, evolve(), periodic diagnostics and checkpoints.
Defaults use the grid from SimulationParameters, one GPU, and a CFL
timestep. Use CUDA_VISIBLE_DEVICES to select the GPU, or --devices 2 to use
two visible GPUs. Validation checks
the initialized state.
Runtime metric, finite-state, displacement, capacity and sharding checks remain
fatal. Divergence/Gauss acceptance uses the exterior r>=r_H, excluding two
physical boundary cells and the sponge plus two cells. Horizon-interior and
outer-boundary residuals are recorded separately. Constraints are checked every
100 steps by default, with independently configurable tolerances (1e-10 each).
Output defaults to ./data in the caller's cwd.
SIGINT/SIGTERM request a checkpointed stop after the current checked step.
"""
import argparse
from dataclasses import asdict
import json
import hashlib
import math
import os
import signal
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from PyPIC3D.boundary_conditions.ghost_cells import update_tiled_vector_ghost_cells
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.relativity.core import B_FIELD_LOCATIONS, D_FIELD_LOCATIONS
from PyPIC3D.solvers.gr_static.static_metric import (
    compute_covariant_E, compute_covariant_H, update_B_relativity,
    update_D_relativity, _location_interpolate)
from PyPIC3D.solvers.gr_static.time_loop import time_loop_static_metric
from .simulation_parameters import PARTICLE_INTEGRATOR, SimulationParameters, build_runtime, shard_array
from .magnetization import measure_magnetization, collocate_magnetic_field
from .plasma_injector import empty_particles, inject_pairs, check_species

CONSTRAINT_REGION = ('Exterior r>=r_H: two-cell physical boundary buffers; '
                     'exclude sponge plus two cells; retain tile seams')

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
        E = compute_covariant_E(D, B, metric, static.polar_field_interpolation)
        H = compute_covariant_H(D, B, metric, static.polar_field_interpolation)
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


def make_step(p, species, static, dynamic, background, *, sponge=True,
              injection_rng_policy='event_ordinal_v1', inject_plasma=True,
              current_filter_passes=0):
    """Check deterministic kernels, leaving rejection sampling outside checkify.

    Errors cross the host boundary before any failed state can be consumed.
    """
    injection_steps = max(1, math.ceil(p.injection_interval/float(dynamic.dt)))
    if injection_rng_policy not in ('event_ordinal_v1', 'step_v0'):
        raise ValueError('Unknown injection RNG policy')
    @jax.jit
    def inject(particles, fields, key, index):
        metric = fields[6]
        mag = measure_magnetization(particles, species, fields[1], metric, static, dynamic)
        return inject_pairs(particles, species, mag, fields[0], fields[1], metric,
                            static, dynamic, p, key, index)
    def evolve(particles, fields):
        transform = None
        if current_filter_passes:
            from .current_filter import filter_current
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
        if inject_plasma and int(index) % injection_steps == 0:
            # Key by injection event, not timestep: matched physical injection
            # schedules must draw the same candidates during dt refinement.
            event = index // injection_steps if injection_rng_policy == 'event_ordinal_v1' else index
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


def _constraint_regions(geometry, static, dynamic, p, *, magnetic=False):
    """Owned diagnostic regions on the appropriate divergence-node grid."""
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
    return {'': interior, 'horizon_': owned & (r < p.horizon),
            'outer_boundary_': owned & (r >= p.horizon)
                & ((r >= p.sponge_start-2*p.dr) | (r > p.r_max-2*p.dr)),
            'boundary_': owned & ~interior, 'full_': owned}


def constraint_masks(geometry, static, dynamic, p, *, magnetic=False):
    """Full, exterior acceptance, and excluded masks; tile seams stay owned."""
    regions = _constraint_regions(geometry, static, dynamic, p, magnetic=magnetic)
    return regions['full_'], regions[''], regions['boundary_']


def constraint_residuals(particles, species, fields, static, dynamic, p, *, current_filter_passes=0):
    """Integrated-flux residuals normalized independently within each region.

    gauss/magnetic_divergence are exterior acceptance values. Other regions
    are measurements only. Counts distinguish empty masks from valid zeros.
    """
    from PyPIC3D.boundary_conditions.polar import divergence
    geometry = fields[6].geometry
    charge = compute_rho(particles, species, fields[3], static, dynamic)
    charge *= 4*jnp.pi*dynamic.dx*dynamic.dy*dynamic.dz
    raw_charge = charge
    if current_filter_passes:
        from .current_filter import smooth_integrated
        charge = smooth_integrated(charge, static, current_filter_passes)
    divD = divergence(fields[0], geometry, static)
    divB = divergence(fields[1], geometry, static, True)
    d_masks = _constraint_regions(geometry, static, dynamic, p)
    b_masks = _constraint_regions(geometry, static, dynamic, p, magnetic=True)
    def maximum(value, mask):
        return jnp.max(jnp.where(mask, jnp.abs(value), 0.))
    result = {}
    for prefix in d_masks:
        dm, bm = d_masks[prefix], b_masks[prefix]
        dscale = jnp.maximum(1., jnp.maximum(maximum(divD,dm), maximum(charge,dm)))
        bscale = jnp.maximum(1., jnp.max(jnp.array([
            maximum(b*a,bm) for b,a in zip(fields[1],geometry.B_area)])))
        result[prefix+'gauss'] = jnp.where(jnp.any(dm),maximum(divD-charge,dm)/dscale,jnp.nan)
        result[prefix+'magnetic_divergence'] = jnp.where(jnp.any(bm),maximum(divB,bm)/bscale,jnp.nan)
        result[prefix+'gauss_cells'] = jnp.sum(dm)
        result[prefix+'magnetic_divergence_cells'] = jnp.sum(bm)
        if current_filter_passes:
            raw_scale = jnp.maximum(1., jnp.maximum(maximum(divD, dm), maximum(raw_charge, dm)))
            result[prefix+'raw_charge_gauss'] = jnp.where(
                jnp.any(dm), maximum(divD-raw_charge, dm)/raw_scale, jnp.nan)
            result[prefix+'raw_charge_gauss_cells'] = jnp.sum(dm)
    return result


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


def constraint_payload(residuals):
    """JSON-safe measurements, retaining validity for empty/overflowed regions."""
    values = {k: (int(v) if k.endswith('_cells') else
                  float(v) if math.isfinite(float(v)) else None)
              for k, v in residuals.items()}
    valid = {k: v is not None and values.get(k+'_cells', 1) > 0
             for k, v in values.items() if not k.endswith('_cells')}
    return dict(constraints=values, constraint_validity=valid)


def constraint_policy(p, gauss_tolerance, magnetic_divergence_tolerance,
                      constraint_check_interval):
    return dict(constraint_policy_version=2, constraint_region=CONSTRAINT_REGION,
                constraint_horizon_radius=p.horizon,
                constraint_outer_buffer_start=p.sponge_start-2*p.dr,
                gauss_tolerance=gauss_tolerance,
                magnetic_divergence_tolerance=magnetic_divergence_tolerance,
                constraint_check_interval=constraint_check_interval)


def write_manifest(path, data):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def rolling_checkpoint(output, particles, fields, key, step, p, manifest):
    pending = output/'checkpoint_pending.npz'
    save_checkpoint(pending, particles, fields, key, step, p, run_metadata=manifest)
    current = output/'checkpoint.npz'
    if current.exists():
        current.replace(output/'checkpoint_previous.npz')
    pending.replace(current)
    manifest['checkpoint_step'] = step
    write_manifest(output/'manifest.json', manifest)


def assemble(array, g=3, theta_base=True):
    """Host-only output boundary; production stepping never assembles the mesh."""
    host = np.asarray(jax.device_get(array))
    return np.concatenate([host[i, 0, 0, g:-g, g:(-g+1 if theta_base else -g), g] for i in range(host.shape[0])], axis=0)


def diagnostics(particles, species, fields, p, static, dynamic, *, current_filter_passes=0):
    D, B, _, _, _, _, metric, previous, _ = fields
    # Reconstruct B at the integer D/position time using the production field stage.
    Dhalf = tuple((D[i]+previous[0][i])/2 for i in range(3))
    Bminus = tuple((B[i]+previous[1][i])/2 for i in range(3))
    B = update_B_relativity(compute_covariant_E(Dhalf, B, metric, static.polar_field_interpolation), Bminus,
                            metric, static, dynamic, dynamic.dt)
    B = refresh_fields(B, B_FIELD_LOCATIONS, metric, static)
    E = compute_covariant_E(D, B, metric, static.polar_field_interpolation)
    H = compute_covariant_H(D, B, metric, static.polar_field_interpolation)
    E = tuple(_location_interpolate(E[i], D_FIELD_LOCATIONS[i], ("C",)*3) for i in range(3))
    H = tuple(_location_interpolate(H[i], B_FIELD_LOCATIONS[i], ("C",)*3) for i in range(3))
    bc = collocate_magnetic_field(B, metric)
    dc = jnp.stack([_location_interpolate(D[i], D_FIELD_LOCATIONS[i], ("C",)*3)
                    for i in range(3)], axis=-1)
    d2 = jnp.einsum('...i,...ij,...j->...', dc, metric.center.gamma, dc)
    db = jnp.einsum('...i,...ij,...j->...', dc, metric.center.gamma, bc)
    mag = measure_magnetization(particles, species, B, metric, static, dynamic)
    b2_safe = jnp.maximum(mag.magnetic_squared, jnp.finfo(d2.dtype).tiny)
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
    if metric.geometry is not None:
        from PyPIC3D.boundary_conditions.polar import divergence
        divD=divergence(D,metric.geometry,static)
        divB=divergence(B,metric.geometry,static,True)
        rho=rho*dynamic.dx*dynamic.dy*dynamic.dz
    if current_filter_passes:
        from .current_filter import smooth_integrated
        rho = smooth_integrated(rho, static, current_filter_passes)
    constraints = assemble(divD-4*jnp.pi*rho)
    residuals = {k:np.asarray(v) for k,v in constraint_residuals(
        particles, species, fields, static, dynamic, p,
        current_filter_passes=current_filter_passes).items()}
    regional = {}
    for prefix in ('', 'horizon_', 'outer_boundary_', 'boundary_', 'full_'):
        for name, label in (('gauss', 'gauss'), ('magnetic_divergence', 'divB')):
            key = prefix+name
            regional[prefix+label+'_relative'] = residuals[key]
            regional[prefix+label+'_cells'] = residuals[key+'_cells']
            regional[prefix+label+'_valid'] = (np.isfinite(residuals[key])
                                              & (residuals[key+'_cells'] > 0))
    return dict(r=r, theta=theta, Hphi=assemble(H[2]), omega=assemble(omega),
                Br=assemble(bc[..., 0]), Btheta=assemble(bc[..., 1]),
                radial_flux=assemble(determinant*bc[..., 0]),
                sigma=assemble(mag.sigma), density=assemble(jnp.sum(mag.number_density, axis=0)),
                D2_over_B2=assemble(d2/b2_safe), DdotB_over_B2=assemble(db/b2_safe),
                luminosity=luminosity, gauss=constraints, divB=assemble(divB,theta_base=False),
                **regional, constraint_policy_version=np.asarray(2),
                constraint_horizon_radius=np.asarray(p.horizon),
                active_per_tile=np.asarray(jnp.sum(particles.active, axis=(-1, -2))).reshape(-1))


def check_checkpoint_integrator(metadata):
    """Accept explicit checkpoints without changing their staggered integrator."""
    if metadata.get('particle_integrator', PARTICLE_INTEGRATOR) != PARTICLE_INTEGRATOR:
        raise ValueError('Unknown checkpoint particle integrator; explicit midpoint Strang splitting is required')
    # Legacy v3 checkpoints describe the integrator using these two fields.
    if 'geodesic_iterations' in metadata:
        iterations = metadata['geodesic_iterations']
        if type(iterations) is not int or iterations != 0:
            raise ValueError('Checkpoint particle integrator is not explicit; implicit restarts are unsupported')
    if 'geodesic_nonlinear_solver' in metadata:
        expected = ('cartesian_explicit_midpoint_v1'
                    if metadata.get('particle_coordinates', 'native') == 'cartesian'
                    else 'explicit_midpoint')
        if metadata['geodesic_nonlinear_solver'] != expected:
            raise ValueError('Checkpoint particle integrator is unknown, implicit, or conflicts with its coordinate chart')


def save_checkpoint(path, particles, fields, key, step, p, run_metadata=None):
    """Portable array-only checkpoint; no pickle or executable serialized objects."""
    metadata = dict(run_metadata or {})
    check_checkpoint_integrator(metadata)
    metadata.pop('geodesic_iterations', None)
    metadata.pop('geodesic_nonlinear_solver', None)
    metadata['particle_integrator'] = PARTICLE_INTEGRATOR
    arrays = dict(checkpoint_version=np.asarray(3),
                  reconstruction_id=np.asarray(metadata.get(
                      'metric_reconstruction', 'cardinal_cubic_hermite_consistent_v1')),
                  configuration_sha256=np.asarray(hashlib.sha256(json.dumps(asdict(p),sort_keys=True).encode()).hexdigest()),geometry_id=np.asarray("polar-cap-v1"),x=particles.x, u=particles.u, active=particles.active, key=key,
                  step=np.asarray(step), parameters=np.asarray(json.dumps(asdict(p))))
    for label, vector in (("D", fields[0]), ("B", fields[1]), ("J", fields[2]),
                           ("previous_D", fields[7][0]), ("previous_B", fields[7][1])):
        arrays.update({f"{label}_{i}": value for i, value in enumerate(vector)})
    arrays.update(rho=fields[3], phi=fields[4], overflow=fields[8],
                  run_metadata=np.asarray(json.dumps(metadata)))
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **{k: np.asarray(jax.device_get(v)) for k, v in arrays.items()})
    temporary.replace(path)


def load_checkpoint(path, particles, fields, p, static, expected_dt=None):
    from PyPIC3D.particles.particle_tile_communication import shard_tiled_particles
    with np.load(path, allow_pickle=False) as saved:
        metadata=json.loads(str(saved.get('run_metadata','{}')))
        coordinates = getattr(static, 'particle_coordinates', 'native')
        if metadata.get('particle_coordinates', 'native') != coordinates:
            raise ValueError('Checkpoint particle coordinates differ; momentum bases cannot be mixed')
        check_checkpoint_integrator(metadata)
        if metadata.get('field_interpolation', 'physical') != static.polar_field_interpolation:
            raise ValueError('Checkpoint auxiliary field interpolation differs')
        if metadata.get('horizon_field_cells', 0) != static.horizon_field_cells:
            raise ValueError('Checkpoint horizon field boundary differs')
        if expected_dt is not None and 'dt' in metadata and float(metadata['dt']) != float(expected_dt):
            raise ValueError('Checkpoint timestep differs; leapfrog history cannot be reused at a different timestep')
        if expected_dt is None and metadata.get('timestep_policy') == 'cfl-plus-gyro':
            raise ValueError('CFL benchmark checkpoint requires an explicit matching expected_dt')
        if 'checkpoint_version' not in saved or int(saved['checkpoint_version']) != 3 or str(saved['geometry_id']) != 'polar-cap-v1':
            raise ValueError('Checkpoint requires polar-cap-v1 geometry; full-theta restarts are incompatible')
        reconstruction = ('orthonormal_spherical_hermite_v1' if coordinates == 'cartesian'
                          else 'cardinal_cubic_hermite_consistent_v1')
        if str(saved.get('reconstruction_id','')) != reconstruction:
            raise ValueError("Checkpoint particle metric reconstruction is incompatible")
        saved_parameters = json.loads(str(saved['parameters']))
        expected=hashlib.sha256(json.dumps(saved_parameters,sort_keys=True).encode()).hexdigest()
        if str(saved.get('configuration_sha256','')) != expected:
            raise ValueError("Checkpoint configuration identifier differs")
        # Output cadence and stopping time do not enter any evolution kernel.
        # Permit extending a validated short run without changing its physics,
        # timestep, RNG, particle storage, or staggered field history.
        operational = {'end_time', 'output_interval'}
        if ({k:v for k,v in saved_parameters.items() if k not in operational}
                != {k:v for k,v in asdict(p).items() if k not in operational}):
            raise ValueError("Checkpoint parameters do not match this run")
        particles = shard_tiled_particles(particles._replace(**{k:jnp.asarray(saved[k]) for k in ("x", "u", "active")}), static)
        def vector(label):
            return tuple(shard_array(saved[f"{label}_{i}"], static) for i in range(3))
        fields = (vector("D"), vector("B"), vector("J"), shard_array(saved["rho"], static),
                  shard_array(saved["phi"], static), fields[5], fields[6],
                  (vector("previous_D"), vector("previous_B")), jnp.asarray(saved["overflow"]))
        return particles, fields, jnp.asarray(saved["key"]), int(saved["step"])


def plot_diagnostics(snapshot, p, output):
    """Use the same measured-field plotter for live and saved diagnostics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .plot_entity_bz import Normalization, make_figure
    norm = Normalization.from_manifest(dict(parameters=asdict(p)))
    radii = tuple(r for r in (2., 3., 4., 5.) if snapshot['r'][0] <= r <= snapshot['r'][-1])
    figure, _ = make_figure(snapshot, norm, radii=radii)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def comparison_errors(snapshot, p):
    """Measured BZ profiles, power, and screening; no run certification policy."""
    from .plot_entity_bz import radial_profile
    theta = snapshot['theta']
    angular = (theta >= np.pi/12) & (theta <= 11*np.pi/12)
    radial = (snapshot['r'] >= 2.2) & (snapshot['r'] <= 7.5)
    full = (snapshot['r'] >= p.horizon) & (snapshot['r'] < p.sponge_start-2*p.dr)
    target = -p.spin*np.sin(theta[angular])**2/8
    result = dict(time=float(snapshot['time']), polar_exclusion_degrees=15,
                  profiles=[], unavailable_profile_radii=[])
    for radius in (2., 3., 4., 5.):
        if not snapshot['r'][0] <= radius <= snapshot['r'][-1]:
            result['unavailable_profile_radii'].append(radius)
            continue
        h = radial_profile(snapshot['r'], snapshot['Hphi']/p.B0, radius)[angular]
        omega = radial_profile(snapshot['r'], snapshot['omega'], radius)[angular]
        result['profiles'].append(dict(radius=radius,
            H_relative_l2=float(np.linalg.norm(h-target)/np.linalg.norm(target)) if p.spin else None,
            omega_relative_rms=float(np.sqrt(np.mean((omega/p.omega_h/.5-1)**2))) if p.omega_h else None))
    power = snapshot['luminosity'][radial]/(p.B0**2*p.omega_h**2/6) if p.omega_h else np.array([])
    result.update(luminosity_mean=float(power.mean()) if power.size else None,
                  luminosity_relative_rms=float(np.sqrt(np.mean((power-1)**2))) if power.size else None)
    for label, mask in [('bulk', np.ix_(radial, angular)), ('full_exterior', np.ix_(full, np.ones(theta.shape, bool)))]:
        if all(key in snapshot for key in ('D2_over_B2', 'DdotB_over_B2')):
            d2, parallel = snapshot['D2_over_B2'][mask], snapshot['DdotB_over_B2'][mask]
            result[label] = dict(D2_over_B2_maximum=float(d2.max()) if d2.size else None,
                                 DdotB_over_B2_rms=float(np.sqrt(np.mean(parallel**2))) if parallel.size else None)
    return result


def evolve(particles, species, fields, key, p, static, dynamic, background, output,
           *, start_step=0, steps=None, allow_divergence_errors=False,
           manifest=None, checkpoint_seconds=300., checkpoint_interval=None, constraint_check_interval=100,
           gauss_tolerance=1e-10, magnetic_divergence_tolerance=1e-10):
    """Advance the BZ state with transactional steps and bounded streamed output.

    Injection precedes the push. The caller's state is only replaced after every
    scheduled fatal check passes, so failures save the last committed state.
    ``steps`` limits this invocation; otherwise evolution ends at p.end_time.
    """
    import threading
    if steps is not None and steps < 0:
        raise ValueError('steps must be nonnegative')
    if not math.isfinite(checkpoint_seconds) or checkpoint_seconds < 0:
        raise ValueError('checkpoint_seconds must be finite and nonnegative')
    if checkpoint_interval is not None and (not math.isfinite(checkpoint_interval) or checkpoint_interval <= 0):
        raise ValueError('checkpoint_interval must be finite and positive')
    validate_constraint_settings(gauss_tolerance, magnetic_divergence_tolerance,
                                 constraint_check_interval)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    dt = float(dynamic.dt); step = int(start_step)
    checkpoint_stride = (None if checkpoint_interval is None else
                         max(1, math.ceil(checkpoint_interval/dt-1e-12)))
    target = math.ceil(p.end_time/dt)
    if steps is not None:
        target = min(target, step+int(steps))
    manifest = dict(manifest or {}, particle_integrator=PARTICLE_INTEGRATOR,
                    dt=dt, parameters=asdict(p), step=step,
                    time=step*dt, target_step=target, status='running',
                    checkpoint_seconds=checkpoint_seconds,
                    checkpoint_interval=checkpoint_interval, checkpoint_stride_steps=checkpoint_stride,
                    divergence_checks_waived=allow_divergence_errors,
                    output_directory=str(output.resolve()))
    manifest.update(constraint_policy(p, gauss_tolerance, magnetic_divergence_tolerance,
                                      constraint_check_interval))
    budget = np.asarray(manifest.get('boundary_budget', np.zeros(11)), dtype=float)
    filter_passes = manifest.get('current_filter_passes', 0)
    options = {}
    if manifest.get('vacuum', False): options['inject_plasma'] = False
    if manifest.get('injection_rng_policy') == 'step_v0': options['injection_rng_policy'] = 'step_v0'
    if filter_passes: options['current_filter_passes'] = filter_passes
    execute = make_step(p, species, static, dynamic, background, **options)
    measure = jax.jit(lambda pts, fs: constraint_residuals(pts, species, fs, static, dynamic, p,
                         **({'current_filter_passes': filter_passes} if filter_passes else {})))
    started = time.perf_counter(); last_save = last_log = started
    last_checked_step = None
    next_snapshot = (math.floor(step*dt/p.output_interval)+1)*p.output_interval
    warmup = []; stop = [False]; old_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, lambda *_: stop.__setitem__(0, True))

    def snapshot(final=False):
        data = diagnostics(particles, species, fields, p, static, dynamic,
                           **({'current_filter_passes': filter_passes} if filter_passes else {}))
        data['time'] = step*dt
        np.savez(output/('diagnostics.npz' if final else f'snapshot_{step:012d}.npz'), **data)
        if final:
            (output/'comparison.json').write_text(json.dumps(comparison_errors(data, p), indent=2))
            plot_diagnostics(data, p, output/'figure6_diagnostics.png')

    with (output/'progress.jsonl').open('a', buffering=1) as stream, tqdm(
            total=max(target, step), initial=step, desc='BZ monopole',
            unit='step', dynamic_ncols=True, disable=step >= target) as progress:
        def record(row):
            message = json.dumps(row, allow_nan=False)
            stream.write(message+'\n'); tqdm.write(message)

        def check_state(pts, fs, checked_step):
            nonlocal last_checked_step
            residuals = measure(pts, fs)
            payload = constraint_payload(residuals)
            event = dict(event='constraints', step=checked_step, time=checked_step*dt,
                         gauss_tolerance=gauss_tolerance,
                         magnetic_divergence_tolerance=magnetic_divergence_tolerance,
                         **payload)
            try:
                check_constraints(residuals, allow_divergence_errors,
                                  gauss_tolerance=gauss_tolerance,
                                  magnetic_divergence_tolerance=magnetic_divergence_tolerance)
            except (RuntimeError, FloatingPointError) as error:
                manifest.update(failed_candidate_step=checked_step,
                                failed_candidate_time=checked_step*dt,
                                failure_constraints=payload['constraints'],
                                failure_constraint_validity=payload['constraint_validity'])
                record(dict(event, status='failed', error=str(error)))
                raise
            waived = (float(residuals['gauss']) >= gauss_tolerance
                      or float(residuals['magnetic_divergence']) >= magnetic_divergence_tolerance)
            record(dict(event, status='waived' if waived else 'passed'))
            manifest.update(payload, constraints_step=checked_step, constraints_time=checked_step*dt)
            last_checked_step = checked_step
            return payload

        try:
            check_species(species); check_sharding(particles, fields, static)
            if not bool(finite_state(particles, fields)):
                raise FloatingPointError('Nonfinite initial state')
            initial = check_state(particles, fields, step)
            manifest['initial_constraints'] = initial['constraints']
            manifest['initial_constraint_validity'] = initial['constraint_validity']
            rolling_checkpoint(output, particles, fields, key, step, p, manifest)
            snapshot()
            while step < target and not stop[0]:
                before = time.perf_counter()
                new, newfields, newkey, report = execute(particles, fields, key, step)
                now = time.perf_counter()
                if (step+1) % constraint_check_interval == 0 or step+1 == target:
                    check_state(new, newfields, step+1)
                # All acceptance checks completed; commit this step and its RNG.
                particles, fields, key = new, newfields, newkey
                step += 1
                progress.update(1)
                requested, inserted, rejected, boundary = report
                budget += np.concatenate((np.asarray(boundary.absorbed_count).reshape(-1),
                    np.asarray(boundary.absorbed_charge).reshape(-1),
                    [float(jnp.sum(boundary.removed_grid_charge))],
                    np.asarray(boundary.radial_current_outflow)*dt))
                elapsed = now-before
                if bool(jnp.any(requested)):
                    record(dict(event='injection', step=step-1, requested=np.asarray(requested).tolist(),
                                inserted=np.asarray(inserted).tolist(), rejected=np.asarray(rejected).tolist()))
                if len(warmup) < 32:
                    warmup.append(elapsed)
                    if len(warmup) == 32:
                        median = float(np.median(warmup[1:]))
                        manifest.update(first_step_compile_injection_execution_seconds=warmup[0],
                            steady_step_seconds=median, estimated_remaining_seconds=(target-step)*median,
                            device_memory=[d.memory_stats() for d in static.field_mesh.devices.flat])
                        record(dict(event='32_step_timing', step=step, seconds_per_step=median,
                                    estimated_remaining_seconds=(target-step)*median))
                manifest.update(step=step, time=step*dt, elapsed_seconds=now-started,
                                boundary_budget=budget.tolist(), last_update_unix=time.time())
                if now-last_log >= 30 or step == target:
                    record(dict(event='progress', step=step, time=step*dt, seconds=elapsed,
                                constraints=manifest['constraints'],
                                constraint_validity=manifest['constraint_validity'],
                                constraints_step=manifest['constraints_step'],
                                boundary_budget=budget.tolist(),
                                active_per_tile=np.asarray(particles.active.sum(axis=(-1,-2))).reshape(-1).tolist()))
                    write_manifest(output/'manifest.json', manifest); last_log = now
                if step*dt >= next_snapshot:
                    snapshot(); next_snapshot += p.output_interval
                if (now-last_save >= checkpoint_seconds
                        or (checkpoint_stride is not None and step % checkpoint_stride == 0)):
                    rolling_checkpoint(output, particles, fields, key, step, p, manifest)
                    last_save = time.perf_counter()
            if last_checked_step != step:
                check_state(particles, fields, step)
            manifest.update(status='finalizing', step=step, time=step*dt,
                            device_memory=[d.memory_stats() for d in static.field_mesh.devices.flat])
            rolling_checkpoint(output, particles, fields, key, step, p, manifest)
            snapshot(final=True)
            manifest['status'] = 'stopped' if stop[0] else 'completed'
        except Exception as error:
            manifest.update(status='failed', error=str(error), step=step, time=step*dt)
            write_manifest(output/'manifest.json', manifest)
            try:
                save_checkpoint(output/'failed_checkpoint.npz', particles, fields, key, step, p,
                                run_metadata=manifest)
            except Exception as save_error:
                manifest['failure_checkpoint_error'] = str(save_error)
            raise
        finally:
            write_manifest(output/'manifest.json', manifest)
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
    return particles, fields, key, step


def run(args):
    if args.steps is not None and args.steps <= 0:
        raise ValueError('--steps must be positive')
    validate_constraint_settings(args.gauss_tolerance, args.magnetic_divergence_tolerance,
                                 args.constraint_check_interval)
    if args.backend == 'gpu':
        available = len(jax.devices('gpu'))
        if available < args.devices:
            raise RuntimeError(f'GPU execution requested {args.devices} device(s), '
                               f'but only {available} CUDA GPU(s) are visible')
    elif jax.default_backend() != 'cpu':
        raise RuntimeError('CPU execution requires JAX_PLATFORMS=cpu')
    p = SimulationParameters(nr=args.nr, ntheta=args.ntheta, devices=args.devices,
                             maximum_timestep=args.maximum_timestep,
                             end_time=args.end_time, output_interval=args.output_interval,
                             courant=args.courant, pairs_per_cell=args.pairs_per_cell,
                             capacity_factor=args.capacity_factor, seed=args.seed,
                             skin_depth=args.skin_depth, injection_interval=args.injection_interval)
    if args.current_filter_passes < 0:
        raise ValueError('current_filter_passes must be nonnegative')
    if args.current_filter_passes and p.r_min+(args.current_filter_passes+2)*p.dr >= p.horizon:
        raise ValueError('Source filter and absorption stencil must fit inside the horizon; increase nr')
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    if (output/'manifest.json').exists():
        raise FileExistsError('Choose a fresh output directory, including when restarting')
    static, dynamic, metric, report = build_runtime(p, timestep_policy=args.timestep_policy,
                                                   particle_batch_size=args.particle_batch_size,
                                                   horizon_field_cells=args.horizon_field_cells,
                                                   field_interpolation=args.field_interpolation,
                                                   particle_coordinates=getattr(args, 'particle_coordinates', 'native'))
    report.update(constraint_policy(p, args.gauss_tolerance, args.magnetic_divergence_tolerance,
                                    args.constraint_check_interval))
    particles, species = empty_particles(p, static)
    fields, background = initialize_fields(p, static, dynamic, metric)
    key = jax.random.PRNGKey(p.seed); step = 0; previous_metadata = {}
    if args.restart:
        particles, fields, key, step = load_checkpoint(Path(args.restart), particles, fields, p,
                                                       static, expected_dt=dynamic.dt)
        with np.load(args.restart, allow_pickle=False) as saved:
            previous_metadata = json.loads(str(saved.get('run_metadata', '{}')))
        if bool(previous_metadata.get('vacuum', False)) != args.vacuum:
            raise ValueError('Checkpoint vacuum/plasma mode differs')
        if previous_metadata.get('current_filter_passes', 0) != args.current_filter_passes:
            raise ValueError('Checkpoint current filter differs')
        if previous_metadata.get('horizon_field_cells', 0) != args.horizon_field_cells:
            raise ValueError('Checkpoint horizon field boundary differs')
    manifest = dict(report, seed=p.seed, pid=os.getpid(), started_unix=time.time(),
        vacuum=args.vacuum,
        current_filter_passes=args.current_filter_passes,
        charge_constraint_source=('binomial_filtered_integrated_charge' if args.current_filter_passes
                                  else 'raw_integrated_charge'),
        injection_rng_policy=(previous_metadata.get('injection_rng_policy', 'step_v0')
                              if args.restart else 'event_ordinal_v1'),
        boundary_budget=previous_metadata.get('boundary_budget', [0.]*11),
        resumed_from=str(Path(args.restart).resolve()) if args.restart else None,
        device_information=[dict(id=d.id, kind=d.device_kind, platform=d.platform)
                            for d in static.field_mesh.devices.flat])
    print(json.dumps(report), flush=True)
    if args.mode == 'validate':
        check_species(species); check_sharding(particles, fields, static)
        if not bool(finite_state(particles, fields)):
            raise FloatingPointError('Nonfinite initialized state')
        residuals = constraint_residuals(particles,species,fields,static,dynamic,p,
                                         current_filter_passes=args.current_filter_passes)
        payload = constraint_payload(residuals)
        manifest.update(payload, status='validated', step=step, time=step*float(dynamic.dt),
                        constraints_step=step, constraints_time=step*float(dynamic.dt),
                        divergence_checks_waived=args.allow_divergence_errors,
                        output_directory=str(output))
        event = dict(event='constraints', step=step, time=step*float(dynamic.dt),
                     gauss_tolerance=args.gauss_tolerance,
                     magnetic_divergence_tolerance=args.magnetic_divergence_tolerance, **payload)
        try:
            check_constraints(residuals, args.allow_divergence_errors,
                              gauss_tolerance=args.gauss_tolerance,
                              magnetic_divergence_tolerance=args.magnetic_divergence_tolerance)
            waived = (float(residuals['gauss']) >= args.gauss_tolerance
                      or float(residuals['magnetic_divergence']) >= args.magnetic_divergence_tolerance)
            event['status'] = 'waived' if waived else 'passed'
        except (RuntimeError, FloatingPointError) as error:
            event.update(status='failed', error=str(error))
            manifest.update(status='failed', error=str(error), failed_candidate_step=step,
                            failed_candidate_time=step*float(dynamic.dt),
                            failure_constraints=payload['constraints'],
                            failure_constraint_validity=payload['constraint_validity'])
            raise
        finally:
            write_manifest(output/'manifest.json', manifest)
            with (output/'progress.jsonl').open('a') as stream:
                stream.write(json.dumps(event, allow_nan=False)+'\n')
        return
    steps = 0 if args.mode == 'initialize' else (32 if args.mode == 'smoke' else None)
    if args.steps is not None:
        steps = args.steps if steps is None else min(steps, args.steps)
    return evolve(particles, species, fields, key, p, static, dynamic, background, output,
                  start_step=step, steps=steps, allow_divergence_errors=args.allow_divergence_errors,
                  gauss_tolerance=args.gauss_tolerance,
                  magnetic_divergence_tolerance=args.magnetic_divergence_tolerance,
                  constraint_check_interval=args.constraint_check_interval,
                  checkpoint_seconds=getattr(args, 'checkpoint_seconds', 300.),
                  checkpoint_interval=getattr(args, 'checkpoint_interval', None),
                  manifest=manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "initialize", "smoke", "run", "analyze"), default="validate")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--devices", type=int, choices=(1, 2), default=1)
    parser.add_argument('--field-interpolation', choices=('physical', 'entity'), default='entity',
                        help='Use Entity II metric-weighted auxiliary fields or the original interpolation')
    parser.add_argument('--particle-coordinates', choices=('native', 'cartesian'), default='cartesian',
                        help='Particle coordinate chart; restart must use the same chart')
    parser.add_argument("--maximum-timestep", type=float, default=SimulationParameters().maximum_timestep)
    parser.add_argument("--nr", type=int, default=SimulationParameters().nr)
    parser.add_argument("--ntheta", type=int, default=SimulationParameters().ntheta)
    parser.add_argument("--particle-batch-size", type=int, default=8192,
                        help="Active pusher batch size; smaller batches suit lightweight runs")
    parser.add_argument("--vacuum", action="store_true",
                        help="Disable pair injection for a field-solver stability control")
    parser.add_argument("--current-filter-passes", type=int, default=4,
                        help="Conservative binomial source passes; Gauss uses the same filtered charge")
    parser.add_argument("--horizon-field-cells", type=int, default=5,
                        help="Freeze this many inner physical field planes; Entity uses filter passes + 1")
    for name in ("end_time", "output_interval", "courant", "skin_depth", "injection_interval"):
        parser.add_argument("--"+name.replace("_", "-"), type=float,
                            default=getattr(SimulationParameters(), name))
    for name in ("pairs_per_cell", "capacity_factor", "seed"):
        parser.add_argument("--"+name.replace("_", "-"), type=int,
                            default=getattr(SimulationParameters(), name))
    parser.add_argument("--timestep-policy", choices=("cfl",), default="cfl")
    parser.add_argument("--steps", type=int)
    parser.add_argument('--checkpoint-seconds', type=float, default=300.,
                        help='Wall-clock interval for recovery checkpoints')
    parser.add_argument('--checkpoint-interval', type=float,
                        help='Also checkpoint every interval in M, rounded up to a whole number of steps')
    parser.add_argument("--gauss-tolerance", type=float, default=1e-10,
                        help="Exterior normalized Gauss residual stopping threshold (default: 1e-10)")
    parser.add_argument("--magnetic-divergence-tolerance", type=float, default=1e-10,
                        help="Exterior normalized magnetic divergence stopping threshold (default: 1e-10)")
    parser.add_argument("--constraint-check-interval", type=int, default=100,
                        help="Check constraints every N absolute steps, plus initial/final states (default: 100)")
    parser.add_argument("--allow-divergence-errors", action="store_true",
                        help="Explicitly waive exterior constraint tolerances; nonfinite state remains fatal")
    parser.add_argument("--restart")
    parser.add_argument("--output", default="data", help="Output directory (default: ./data in the current working directory)")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    if args.mode == "analyze":
        output = Path(args.output)
        p = SimulationParameters(**json.loads((output/"manifest.json").read_text())["parameters"])
        with np.load(output/"diagnostics.npz", allow_pickle=False) as snap:
            plot_diagnostics(dict(snap), p, output/"figure6_diagnostics.png")
    else:
        run(args)


if __name__ == "__main__":
    main()
