"""Exterior constraints, charge flux and coordinate profiles for the RN demo."""

from functools import partial

import jax
import jax.numpy as jnp

from JAX_BSSN.bssn.constraints import compute_all_constraints_with_matter
from JAX_BSSN.bssn.tensor_algebra import determinant_3x3_metric
from PyPIC3D.diagnostics.static_metric import divergence
from PyPIC3D.relativity.core import D_FIELD_LOCATIONS, B_FIELD_LOCATIONS
from PyPIC3D.relativity.dynamical_relativity import tiles_to_bssn
from PyPIC3D.relativity.energy_momentum_tensor import electromagnetic_moments, compute_energy_momentum_tensor
from PyPIC3D.relativity.field_interpolation import location_interpolate


def centered_fields(fields, static):
    """Common-time densities at C nodes, for output only (not source products)."""
    return tuple(jnp.stack(tuple(tiles_to_bssn(location_interpolate(q, loc, ("C",)*3), static)
                                 for q, loc in zip(vector, locations)))
                 for vector, locations in ((fields[0], D_FIELD_LOCATIONS), (fields[7][2], B_FIELD_LOCATIONS)))


@jax.jit
def valid_state(fields, bssn):
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(q))
                               for q in jax.tree.leaves((fields, bssn))]))
    g = bssn.state.conformal_metric
    positive = (g[0, 0] > 0) & (g[0, 0]*g[1, 1] - g[0, 1]**2 > 0)
    positive = positive & (determinant_3x3_metric(g) > 0)
    return finite & jnp.all(positive) & jnp.all(bssn.state.conformal_factor > 0)


def failure_description(fields, bssn):
    for path, q in jax.tree_util.tree_flatten_with_path(dict(fields=fields, bssn=bssn))[0]:
        if not bool(jnp.all(jnp.isfinite(q))):
            index = jnp.argwhere(~jnp.isfinite(q), size=1)[0].tolist()
            return f"nonfinite {jax.tree_util.keystr(path)} at {index}"
    if bool(jnp.any(bssn.state.conformal_factor <= 0)):
        return "nonpositive conformal_factor W"
    return "conformal metric is not positive definite"


def masked_norms(value, mask):
    count = jnp.sum(jnp.broadcast_to(mask, value.shape))
    rms = jnp.sqrt(jnp.sum(jnp.where(mask, value*value, 0))/jnp.maximum(count, 1))
    maximum = jnp.max(jnp.where(mask, jnp.abs(value), 0))
    return rms, maximum


@partial(jax.jit, static_argnames="static")
def diagnostic_values(particles, species, fields, static, dynamic):
    bssn = dynamic.bssn_parameters
    state = bssn.state
    axis = dynamic.grids.center[0][1:-1]
    x, y, z = jnp.meshgrid(axis, axis, axis, indexing="ij")
    r = jnp.sqrt(x*x + y*y + z*z)
    outer = jnp.maximum(jnp.maximum(jnp.abs(x), jnp.abs(y)), jnp.abs(z))
    half_width = dynamic.x_wind/2
    # Fixed coordinate masks: neither is a measured apparent horizon.
    masks = dict(exterior=(r >= 2.) & (r <= 8.), sponge=outer >= half_width - 3.)
    moments = electromagnetic_moments(fields[0], fields[7][2], static)
    raw = compute_energy_momentum_tensor(particles, species, moments, state, static, dynamic)
    smooth = compute_energy_momentum_tensor(particles, species, moments, state, static, dynamic,
                                           source_filter="binomial")
    constraints = compute_all_constraints_with_matter(state, bssn.evolution_parameters,
                                                      bssn.gravity_G*raw[0], bssn.gravity_G*raw[1])
    quantities = dict(
        raw_hamiltonian=constraints.hamiltonian, raw_momentum=constraints.momentum,
        filtered_hamiltonian=constraints.hamiltonian + 16*jnp.pi*bssn.gravity_G*(raw[0]-smooth[0]),
        filtered_momentum=constraints.momentum + 8*jnp.pi*bssn.gravity_G*(raw[1]-smooth[1]),
        det_gamma=constraints.det_gamma, trace_A=constraints.trace_A,
        connection=constraints.gamma_condition,
        gauss=tiles_to_bssn(divergence(fields[0], dynamic), static))
    # Magnetic divergence lives on V,V,V, so construct its mask there.
    divB = tiles_to_bssn(divergence(fields[7][2], dynamic, forward=True), static)
    vertex = dynamic.grids.vertex[0][1:-1]
    vx, vy, vz = jnp.meshgrid(vertex, vertex, vertex, indexing="ij")
    vr = jnp.sqrt(vx*vx + vy*vy + vz*vz)
    vmax = jnp.maximum(jnp.maximum(jnp.abs(vx), jnp.abs(vy)), jnp.abs(vz))
    vmasks = dict(exterior=(vr >= 2.) & (vr <= 8.), sponge=vmax >= half_width - 3.)
    result = dict(time=bssn.time, lapse_min=jnp.min(state.lapse), lapse_max=jnp.max(state.lapse),
                  W_min=jnp.min(state.conformal_factor), W_max=jnp.max(state.conformal_factor))
    for region, mask in masks.items():
        for name, q in quantities.items():
            result[f"{region}_{name}_l2"], result[f"{region}_{name}_linf"] = masked_norms(q, mask)
        result[f"{region}_divB_l2"], result[f"{region}_divB_linf"] = masked_norms(divB, vmasks[region])

    # The backward native divergence telescopes to a cubical electric flux.
    # Include the unresolved puncture cells here; exclude them in Gauss norms.
    for radius in (2, 4, 6):
        box = outer < radius
        result[f"charge_box_{radius}"] = jnp.sum(jnp.where(box, quantities["gauss"], 0))*dynamic.dx**3/(4*jnp.pi)
    for name, vector in (("D", fields[0]), ("B", fields[7][2])):
        q = jnp.stack(tuple(tiles_to_bssn(a, static) for a in vector))
        result[f"{name}_max"] = jnp.max(jnp.abs(q))
        # Squared second differences resolve short wavelengths without hiding
        # checkerboard energy through component averaging.
        rough = sum((jnp.roll(q, 1, a)+jnp.roll(q, -1, a)-2*q)**2 for a in (1, 2, 3))
        interior = r < 8.
        result[f"{name}_roughness"] = jnp.sum(jnp.where(interior, rough, 0))/jnp.maximum(
            jnp.sum(jnp.where(interior, q*q, 0)), jnp.finfo(q.dtype).tiny)
    return result
