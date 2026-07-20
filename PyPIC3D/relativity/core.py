from typing import NamedTuple

import jax.numpy as jnp


D_FIELD_LOCATIONS = (("V", "C", "C"), ("C", "V", "C"), ("C", "C", "V"))
B_FIELD_LOCATIONS = (("C", "V", "V"), ("V", "C", "V"), ("V", "V", "C"))


class Metric(NamedTuple):
    """
    3+1 metric data sampled on one Yee-grid location.

    Particles store covariant spatial four-velocity components ``u_i`` in the
    existing three-component ``particles.u`` slot.  The geodesic and current
    routines use ``gamma_inv`` to convert those covariant components into
    contravariant spatial velocities.
    """

    lapse: object
    shift: object
    gamma: object
    gamma_inv: object
    sqrt_gamma: object
    christoffel: object
    grad_lapse: object
    grad_shift: object


class YeeMetric(NamedTuple):
    """
    Metric state on the grid locations used by the static-metric update.

    ``D`` and ``B`` are tuples with one metric per component location.
    ``center`` is used by the particle pusher and current deposition.
    ``vertex`` is kept for interpolation and diagnostics that need a shared
    nodal metric.
    """

    D: tuple
    B: tuple
    center: Metric
    vertex: Metric


def covariant_lorentz_factor(u_cov, gamma_inv):
    """
    Compute Gamma = sqrt(1 + gamma^ij u_i u_j).
    """

    u_sq = jnp.einsum("...i,...ij,...j->...", u_cov, gamma_inv, u_cov)
    return jnp.sqrt(1.0 + u_sq)


def contravariant_three_velocity(u_cov, gamma_inv):
    """
    Convert covariant spatial velocity/momentum to FIDO three-velocity v^i.
    """

    u_con = jnp.einsum("...ij,...j->...i", gamma_inv, u_cov)
    Gamma = covariant_lorentz_factor(u_cov, gamma_inv)
    return u_con / Gamma[..., jnp.newaxis]


def lower_vector(vector_con, gamma):
    """
    Lower a contravariant spatial vector with gamma_ij.
    """

    return jnp.einsum("...ij,...j->...i", gamma, vector_con)


def centered_metric_gradient(field, dx, dy, dz):
    """
    Central differences on tile-local metric arrays including guard cells.
    """

    spacings = (dx, dy, dz)
    gradients = []
    for axis, spacing in enumerate(spacings):
        array_axis = axis + 3
        gradients.append(
            (jnp.roll(field, -1, axis=array_axis) - jnp.roll(field, 1, axis=array_axis))
            / (2.0 * spacing)
        )
    return jnp.stack(gradients, axis=-1)


def fill_metric_derivatives(metric, dx, dy, dz):
    """
    Fill lapse, shift, and Christoffel derivative data for a spatial metric.
    """

    grad_lapse = centered_metric_gradient(metric.lapse, dx, dy, dz)
    grad_shift = jnp.stack(
        [centered_metric_gradient(metric.shift[..., i], dx, dy, dz) for i in range(3)],
        axis=-2,
    )

    grad_gamma = jnp.stack(
        [
            [
                centered_metric_gradient(metric.gamma[..., i, j], dx, dy, dz)
                for j in range(3)
            ]
            for i in range(3)
        ],
        axis=-2,
    )
    grad_gamma = jnp.moveaxis(grad_gamma, 0, -3)

    christoffel = jnp.zeros_like(metric.christoffel)
    for k in range(3):
        for i in range(3):
            for j in range(3):
                term = 0.0
                for l in range(3):
                    term = term + metric.gamma_inv[..., k, l] * (
                        grad_gamma[..., l, j, i]
                        + grad_gamma[..., l, i, j]
                        - grad_gamma[..., i, j, l]
                    )
                christoffel = christoffel.at[..., k, i, j].set(0.5 * term)

    return metric._replace(
        christoffel=christoffel,
        grad_lapse=grad_lapse,
        grad_shift=grad_shift,
    )


def metric_for_location(center_grid, vertex_grid, location):
    """
    Select the tuple of grid axes associated with a C/V location triplet.
    """

    return tuple(
        center_grid[axis] if location[axis] == "C" else vertex_grid[axis]
        for axis in range(3)
    )
