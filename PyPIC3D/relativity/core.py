from typing import NamedTuple

import jax
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
    center_grad_gamma_inv: object


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


@jax.jit
def _christoffel_from_metric_gradient(gamma_inv, grad_gamma):
    """
    Compute ``Gamma^k_ij`` from ``gamma^kl`` and ``partial_i gamma_lj``.

    ``grad_gamma`` stores the metric component indices first and the
    derivative index last, ``[..., l, j, i]``.  The returned Christoffel
    symbols use the layout ``[..., k, i, j]``.
    """

    metric_derivative = (
        jnp.swapaxes(grad_gamma, -1, -2)
        + grad_gamma
        - jnp.moveaxis(grad_gamma, -1, -3)
    )
    return 0.5 * jnp.einsum(
        "...kl,...lij->...kij",
        gamma_inv,
        metric_derivative,
    )


def centered_metric_gradient(field, dx, dy, dz):
    """
    Central differences on tile-local metric arrays including guard cells.

    Spatial tile axes are stored at positions 3, 4, and 5.  Any trailing
    vector or tensor component axes are preserved, and the derivative index
    is appended as the final axis.
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


def centered_inverse_metric_gradient(gamma_inv, dx, dy, dz):
    """
    Return ``partial_i gamma^{jk}`` on a metric grid.

    The derivative index is stored first in the trailing tensor dimensions,
    giving the layout ``[..., i, j, k]`` used by the Hamiltonian geodesic
    force.
    """

    grad_gamma_inv_jki = centered_metric_gradient(gamma_inv, dx, dy, dz)
    return jnp.moveaxis(grad_gamma_inv_jki, -1, -3)


def analytic_metric_on_grid(grid, metric_at_position):
    """
    Evaluate an analytic spatial metric and its derivatives on a tiled grid.

    ``metric_at_position`` returns ``(lapse, shift, gamma, gamma_inv,
    sqrt_gamma)`` for one coordinate position.  Forward-mode automatic
    differentiation supplies derivatives with respect to those coordinates.
    """

    x_grid, y_grid, z_grid = grid
    X = x_grid[..., :, jnp.newaxis, jnp.newaxis]
    Y = y_grid[..., jnp.newaxis, :, jnp.newaxis]
    Z = z_grid[..., jnp.newaxis, jnp.newaxis, :]
    X, Y, Z = jnp.broadcast_arrays(X, Y, Z)

    grid_shape = X.shape
    positions = jnp.stack((X, Y, Z), axis=-1).reshape((-1, 3))

    def differentiable_metric(position):
        metric_values = metric_at_position(position)
        return metric_values[:4], metric_values

    metric_jacobian, metric_values = jax.vmap(
        jax.jacfwd(differentiable_metric, has_aux=True)
    )(positions)
    lapse, shift, gamma, gamma_inv, sqrt_gamma = metric_values
    grad_lapse, grad_shift, grad_gamma, grad_gamma_inv = metric_jacobian

    lapse = lapse.reshape(grid_shape)
    shift = shift.reshape(grid_shape + (3,))
    gamma = gamma.reshape(grid_shape + (3, 3))
    gamma_inv = gamma_inv.reshape(grid_shape + (3, 3))
    sqrt_gamma = sqrt_gamma.reshape(grid_shape)
    grad_lapse = grad_lapse.reshape(grid_shape + (3,))
    grad_shift = grad_shift.reshape(grid_shape + (3, 3))
    grad_gamma = grad_gamma.reshape(grid_shape + (3, 3, 3))
    grad_gamma_inv = grad_gamma_inv.reshape(grid_shape + (3, 3, 3))

    christoffel = _christoffel_from_metric_gradient(gamma_inv, grad_gamma)

    metric = Metric(
        lapse=lapse,
        shift=shift,
        gamma=gamma,
        gamma_inv=gamma_inv,
        sqrt_gamma=sqrt_gamma,
        christoffel=christoffel,
        grad_lapse=grad_lapse,
        grad_shift=grad_shift,
    )
    # jacfwd returns ``[..., component_j, component_k, derivative_i]``.
    grad_gamma_inv = jnp.moveaxis(grad_gamma_inv, -1, -3)
    return metric, grad_gamma_inv


def fill_metric_derivatives(metric, dx, dy, dz):
    """
    Fill lapse, shift, and Christoffel derivative data for a spatial metric.
    """

    grad_lapse = centered_metric_gradient(metric.lapse, dx, dy, dz)
    grad_shift = centered_metric_gradient(metric.shift, dx, dy, dz)
    grad_gamma = centered_metric_gradient(metric.gamma, dx, dy, dz)

    christoffel = _christoffel_from_metric_gradient(metric.gamma_inv, grad_gamma)

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
