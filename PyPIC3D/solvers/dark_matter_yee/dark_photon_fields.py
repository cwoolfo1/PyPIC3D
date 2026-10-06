"""Tile-native Proca fields: E/phi at n and A at n - 1/2.

A and E use electric Yee edges; phi uses (C, C, C) nodes. The forward
scalar gradient and backward vector divergence form a compatible pair.
``dark_mu`` is an inverse length, not electromagnetic permeability.

The live dark state (E, A, phi) always carries refreshed halos, so the
operators below read their inputs directly. Gradient, divergence and B are
interior-only intermediates with zero halos; ``synchronized_dark_fields``
refreshes B because particle gathers and output read its guard cells.
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
    """Return +grad(phi) on electric edges with zero halos; phi halos must be fresh."""

    interior = _interior(static_parameters)
    g = static_parameters.guard_cells
    result = []
    for axis, spacing in enumerate((dynamic_parameters.dx, dynamic_parameters.dy, dynamic_parameters.dz)):
        forward = list(interior)
        forward[axis + 3] = slice(g + 1, None if g == 1 else -g + 1)
        derivative = (phi[tuple(forward)] - phi[interior]) / spacing
        result.append(jnp.zeros_like(phi).at[interior].set(derivative))

    return tuple(result)

def dark_divergence(A, static_parameters, dynamic_parameters):
    """Return the backward edge-to-node divergence with zero halos; A halos must be fresh."""

    interior = _interior(static_parameters)
    g = static_parameters.guard_cells
    divergence = jnp.zeros_like(A[0][interior])
    for axis, spacing in enumerate((dynamic_parameters.dx, dynamic_parameters.dy, dynamic_parameters.dz)):
        backward = list(interior)
        backward[axis + 3] = slice(g - 1, -g - 1)
        divergence = divergence + (A[axis][interior] - A[axis][tuple(backward)]) / spacing

    return jnp.zeros_like(A[0]).at[interior].set(divergence)

def compute_dark_B(A_n, static_parameters, dynamic_parameters):
    """Compute curl(A) on magnetic Yee faces with zero halos."""
    curl = yee_curl_e_to_b(A_n, static_parameters, dynamic_parameters)
    interior = _interior(static_parameters)
    return tuple(jnp.zeros_like(a).at[interior].set(c) for a, c in zip(A_n, curl))

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
    phi = update_tiled_ghost_cells(
        phi, static_parameters, static_parameters.guard_cells, location=("C", "C", "C")
    )
    # update phi to full ghost-celled tiles on the (C, C, C) nodes

    return phi


def synchronized_dark_fields(dark_fields, static_parameters, dynamic_parameters):
    """Return (E, A, phi, B) at integer time for gathering and diagnostics.

    The half drift equals the average of the adjacent staggered A levels.
    It reconstructs A to second order without modifying the live state.
    """
    E, A, phi = dark_fields
    A = update_dark_A(E, A, phi, None, static_parameters, dynamic_parameters, dynamic_parameters.dt / 2)
    B = update_tiled_vector_ghost_cells(
        compute_dark_B(A, static_parameters, dynamic_parameters),
        static_parameters, static_parameters.guard_cells, locations=B_FIELD_LOCATIONS,
    )
    # refresh B halos for the particle gather and field output
    return E, A, phi, B


def initialize_dark_photon_fields(static_parameters, dynamic_parameters, E=None, A=None, phi=None):
    """Initialize physical tiled data at t=0 and seed A onto t=-dt/2.

    Missing components default to zero. Nonzero data must satisfy
    div(E) + dark_mu**2 * phi = -sin_chi * rho / eps for a constraint-compatible
    initial state; this initializer does not solve that constraint.
    """
    from PyPIC3D.initialization import build_tiled_array

    zero = build_tiled_array(static_parameters, dynamic_parameters)
    if E is None and A is None and phi is None:
        # Zero already satisfies every periodic halo and the half-step seed.
        return (zero, zero, zero), (zero, zero, zero), zero

    def scalar(value):
        value = zero if value is None else jnp.asarray(value)
        if value.shape != zero.shape or not jnp.issubdtype(value.dtype, jnp.floating):
            raise ValueError(f"Dark fields require real floating tiled arrays with shape {zero.shape}")
        return value

    def vector(value):
        if value is None:
            value = (zero, zero, zero)
        if len(value) != 3:
            raise ValueError("Dark vectors require three components")
        return tuple(scalar(v) for v in value)

    E = update_tiled_vector_ghost_cells(
        vector(E), static_parameters, static_parameters.guard_cells, locations=D_FIELD_LOCATIONS,
    )
    # refresh E to full ghost-celled tiles, including the electric Yee edges
    phi = update_tiled_ghost_cells(
        scalar(phi), static_parameters, static_parameters.guard_cells, location=("C", "C", "C"),
    )
    # refresh phi to full ghost-celled tiles on the (C, C, C) nodes
    A = update_dark_A(E, vector(A), phi, None, static_parameters, dynamic_parameters, -dynamic_parameters.dt / 2)
    # seed A to t=-dt/2 with a half drift, which also refreshes its halos
    return E, A, phi
