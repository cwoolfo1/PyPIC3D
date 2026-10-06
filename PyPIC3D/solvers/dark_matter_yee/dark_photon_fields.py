"""Tile-native Proca fields: E/phi at n and A at n - 1/2.

A and E use electric Yee edges; phi uses (C, C, C) nodes. The forward
scalar gradient and backward vector divergence form a compatible pair.
``dark_mu`` is an inverse length, not electromagnetic permeability.
"""

import jax.numpy as jnp

from PyPIC3D.boundary_conditions.ghost_cells import (
    update_tiled_ghost_cells,
    update_tiled_vector_ghost_cells,
)
from PyPIC3D.relativity.core import D_FIELD_LOCATIONS, B_FIELD_LOCATIONS
from PyPIC3D.solvers.yee.first_order_yee import yee_curl_e_to_b, yee_curl_b_to_e


def _interior(static_parameters):
    g = static_parameters.guard_cells
    return (slice(None),) * 3 + (slice(g, -g),) * 3


def dark_gradient(phi, static_parameters, dynamic_parameters):
    """Return +grad(phi) on electric edges, including refreshed halos."""

    phi = update_tiled_ghost_cells(
        phi, static_parameters, static_parameters.guard_cells, location=("C", "C", "C"),
    )
    # update phi to full ghost-celled tiles, including the electric Yee edges
    
    interior = _interior(static_parameters)
    g = static_parameters.guard_cells
    result = []
    for axis, spacing in enumerate((dynamic_parameters.dx, dynamic_parameters.dy, dynamic_parameters.dz)):
        forward = list(interior)
        forward[axis + 3] = slice(g + 1, None if g == 1 else -g + 1)
        derivative = (phi[tuple(forward)] - phi[interior]) / spacing
        result.append(jnp.zeros_like(phi).at[interior].set(derivative))

    gradient = tuple(result)
    gradient = update_tiled_vector_ghost_cells(
        gradient, static_parameters, static_parameters.guard_cells, locations=D_FIELD_LOCATIONS,
    )

    return gradient

def dark_divergence(A, static_parameters, dynamic_parameters):
    """Return the backward edge-to-node divergence, including halos."""

    A = update_tiled_vector_ghost_cells(
        A, static_parameters, static_parameters.guard_cells, locations=D_FIELD_LOCATIONS,
    )
    # update A to full ghost-celled tiles, including the electric Yee edges

    interior = _interior(static_parameters)
    g = static_parameters.guard_cells
    divergence = jnp.zeros_like(A[0][interior])
    for axis, spacing in enumerate((dynamic_parameters.dx, dynamic_parameters.dy, dynamic_parameters.dz)):
        backward = list(interior)
        backward[axis + 3] = slice(g - 1, -g - 1)
        divergence = divergence + (A[axis][interior] - A[axis][tuple(backward)]) / spacing

    divergence = jnp.zeros_like(A[0]).at[interior].set(divergence)
    # add divergence to an interior-only array before applying ghost cell functions

    divergence = update_tiled_ghost_cells(
        divergence, static_parameters, static_parameters.guard_cells, location=("C", "C", "C"),
    )

    return divergence

def compute_dark_B(A_n, static_parameters, dynamic_parameters):
    """Compute curl(A) on magnetic Yee faces as full ghost-celled tiles."""
    curl = yee_curl_e_to_b(A_n, static_parameters, dynamic_parameters)
    interior = _interior(static_parameters)
    B = tuple(jnp.zeros_like(a).at[interior].set(c) for a, c in zip(A_n, curl))

    B = update_tiled_vector_ghost_cells(
        B, static_parameters, static_parameters.guard_cells, locations=B_FIELD_LOCATIONS,
    )
    # update B to full ghost-celled tiles, including the magnetic Yee faces

    return B

def update_dark_E(E_n, B_half, A_half, J_half, static_parameters, dynamic_parameters, dt):
    """Kick E from n to n+1 with A, B and deposited J at n+1/2."""
    curl = yee_curl_b_to_e(B_half, static_parameters, dynamic_parameters)
    interior = _interior(static_parameters)
    c2 = dynamic_parameters.C**2
    mass2 = static_parameters.dark_mu**2
    source = static_parameters.sin_chi / dynamic_parameters.eps
    E = tuple(e.at[interior].add(dt * (c2 * (b + mass2 * a[interior]) + source * j[interior]))
              for e, b, a, j in zip(E_n, curl, A_half, J_half))

    E = update_tiled_vector_ghost_cells(
        E, static_parameters, static_parameters.guard_cells, locations=D_FIELD_LOCATIONS,
    )
    # update E to full ghost-celled tiles, including the electric Yee edges

    return E

def update_dark_A(E_n, A_half, phi_n, J_n, static_parameters, dynamic_parameters, dt):
    """Drift A with E/phi at the intervening integer time.

    J_n is unused, retained for compatibility with the original kernel API.
    """
    gradient = dark_gradient(phi_n, static_parameters, dynamic_parameters)
    interior = _interior(static_parameters)
    A = tuple(a.at[interior].add(-dt * (e[interior] + grad[interior]))
              for a, e, grad in zip(A_half, E_n, gradient))
    A = update_tiled_vector_ghost_cells(
        A, static_parameters, static_parameters.guard_cells, locations=D_FIELD_LOCATIONS,
    )
    # update A to full ghost-celled tiles, including the electric Yee edges

    return A


def update_dark_phi(E_n, A_half, phi_n, J_n, static_parameters, dynamic_parameters, dt):
    """Kick phi from n to n+1 with A at n+1/2 (E_n/J_n are unused)."""
    divergence = dark_divergence(A_half, static_parameters, dynamic_parameters)
    interior = _interior(static_parameters)
    phi = phi_n.at[interior].add(-dt * dynamic_parameters.C**2 * divergence[interior])
    # update phi to interior-only tiles, then refresh halos
    phi = update_tiled_ghost_cells(
        phi, static_parameters, static_parameters.guard_cells, location=("C", "C", "C")
    )
    # update phi to full ghost-celled tiles, including the electric Yee edges

    return phi

def advance_dark_photon_fields(dark_fields, J_half, static_parameters, dynamic_parameters):
    """Advance (E^n, A^{n-1/2}, phi^n) by one complete leapfrog step."""
    E, A, phi = dark_fields
    dt = dynamic_parameters.dt
    A = update_dark_A(E, A, phi, J_half, static_parameters, dynamic_parameters, dt)
    B = compute_dark_B(A, static_parameters, dynamic_parameters)
    phi_new = update_dark_phi(E, A, phi, J_half, static_parameters, dynamic_parameters, dt)
    E = update_dark_E(E, B, A, J_half, static_parameters, dynamic_parameters, dt)
    return E, A, phi_new


def synchronized_dark_fields(dark_fields, static_parameters, dynamic_parameters):
    """Return (E, A, phi, B) at integer time for gathering and diagnostics.

    The half drift equals the average of the adjacent staggered A levels.
    It reconstructs A to second order without modifying the live state.
    """
    E, A, phi = dark_fields
    A = update_dark_A(E, A, phi, None, static_parameters, dynamic_parameters, dynamic_parameters.dt / 2)
    return E, A, phi, compute_dark_B(A, static_parameters, dynamic_parameters)


def initialize_dark_photon_fields(static_parameters, dynamic_parameters, E=None, A=None, phi=None):
    """Initialize physical tiled data at t=0 and seed A onto t=-dt/2.

    Missing components default to zero. Nonzero data must satisfy
    div(E) + dark_mu**2 * phi = -sin_chi * rho / eps for a constraint-compatible
    initial state; this initializer does not solve that constraint.
    """
    widths = static_parameters.tile_shape
    shape = tuple(int(n) // width for n, width in zip(
        (dynamic_parameters.Nx, dynamic_parameters.Ny, dynamic_parameters.Nz), widths,
    )) + tuple(width + 2 * static_parameters.guard_cells for width in widths)
    zero = jnp.zeros(shape, dtype=jnp.float64)
    if E is None and A is None and phi is None:
        # Zero already satisfies every periodic halo and the half-step seed.
        return (zero, zero, zero), (zero, zero, zero), zero

    def scalar(value):
        value = zero if value is None else jnp.asarray(value)
        if value.shape != shape or not jnp.issubdtype(value.dtype, jnp.floating):
            raise ValueError(f"Dark fields require real floating tiled arrays with shape {shape}")
        return value

    def vector(value):
        if value is None:
            value = (zero, zero, zero)
        if len(value) != 3:
            raise ValueError("Dark vectors require three components")
        return update_tiled_vector_ghost_cells(tuple(scalar(v) for v in value), static_parameters, static_parameters.guard_cells, locations=D_FIELD_LOCATIONS)

    E, A = vector(E), vector(A)
    # tile and refresh E and A to full ghost-celled tiles, including the electric Yee edges
    phi = update_tiled_ghost_cells(scalar(phi), static_parameters, static_parameters.guard_cells, location=("C", "C", "C"))
    # tile and refresh phi to full ghost-celled tiles, including the electric Yee edges
    A = update_dark_A(E, A, phi, None, static_parameters, dynamic_parameters, -dynamic_parameters.dt / 2)
    # seed A to t=-dt/2 with a half drift, then refresh to full ghost-celled tiles, including the electric Yee edges
    return E, A, phi
