"""Algebraically consistent geometry sampled exclusively from supplied grids.

Only covariant gamma, lapse and shift are interpolated. Derivatives are those
of that same Hermite interpolant, not interpolated nodal derivative arrays.
Three guard cells cover midpoint sampling before tile migration. Supplied
grids must be uniform and stationary with valid primitive-metric halos.
Invalid interpolated metrics are reported through checkify, never clamped
or repaired. PIC shape factors still select field gathering and deposition,
independently of this metric reconstruction.
"""

import jax
import jax.numpy as jnp
from jax.experimental import checkify

RECONSTRUCTION = "cardinal_cubic_hermite_consistent_v1"
from PyPIC3D.relativity.hermite_metric import interpolate_hermite
from PyPIC3D.relativity.core import Metric, _christoffel_from_metric_gradient


def sample_particle_metric(
    metric,
    position,
    grid,
    shape_factor,
    metric_name,
    active_axes,
    inactive_axis_indices,
    *,
    derivatives=True,
    stage="particle metric",
    tile=None,
    check=True,
    regularize_spherical=False,
):
    if regularize_spherical:
        from PyPIC3D.relativity.cartesian_particle_metric import sample_regularized_metric
        return sample_regularized_metric(
            metric, position, grid, metric_name, active_axes, inactive_axis_indices,
            derivatives=derivatives, check=check, stage=stage, tile=tile)
    # shape_factor is retained for private diagnostic callers; it never selects
    # the metric interpolant. Only field gathering and deposition use PIC shapes.
    del shape_factor
    shape = position.shape[:-1]
    packed = jnp.concatenate(
        (
            metric.lapse[..., None],
            metric.shift,
            metric.gamma.reshape(metric.gamma.shape[:-2] + (9,)),
        ),
        axis=-1,
    )

    def interpolate(q):
        return interpolate_hermite(
            packed, q, grid, active_axes, inactive_axis_indices
        ).reshape(shape + (13,))

    values = interpolate(position)
    lapse, shift = values[..., 0], values[..., 1:4]
    gamma = values[..., 4:].reshape(shape + (3, 3))
    inverse = jnp.linalg.inv(gamma)
    root = jnp.sqrt(jnp.linalg.det(gamma))
    orientation = (
        position[..., 0]
        if metric_name == "flat_cylindrical"
        else (
            jnp.sin(position[..., 1])
            if metric_name in ("flat_spherical", "kerr_schild_spherical")
            else jnp.ones(shape)
        )
    )
    root = jnp.copysign(root, orientation)
    # Three directional JVPs move each particle independently: no N x N Jacobian.
    if derivatives:
        gradients = []
        for axis in range(3):
            if active_axes[axis]:
                tangent = jnp.broadcast_to(
                    jnp.eye(3, dtype=position.dtype)[axis], position.shape
                )
                gradients.append(jax.jvp(interpolate, (position,), (tangent,))[1])
            else:
                gradients.append(jnp.zeros_like(values))
        gradient = jnp.stack(gradients, axis=-1)  # component, derivative
        grad_lapse = gradient[..., 0, :]
        grad_shift = gradient[..., 1:4, :]
        grad_gamma = gradient[..., 4:, :].reshape(shape + (3, 3, 3))
        grad_inverse = -jnp.einsum(
            "...ij,...jlk,...lm->...kim", inverse, grad_gamma, inverse
        )
        christoffel = _christoffel_from_metric_gradient(inverse, grad_gamma)
    else:
        grad_lapse = jnp.zeros(shape + (3,), dtype=values.dtype)
        grad_shift = jnp.zeros(shape + (3, 3), dtype=values.dtype)
        christoffel = jnp.zeros(shape + (3, 3, 3), dtype=values.dtype)
        grad_inverse = jnp.zeros_like(christoffel)
    result = Metric(
        lapse, shift, gamma, inverse, root, christoffel, grad_lapse, grad_shift
    )
    if check:
        valid = particle_metric_valid(result, position, metric_name)
        valid &= jnp.all(jnp.isfinite(grad_inverse), axis=(-3, -2, -1))
        check_particle_samples(valid, position, stage, tile)
    return result, grad_inverse


def safe_inactive_positions(position, active, grid, active_axes, guard_cells):
    """Evaluate unused slots strictly inside the tile; preserve every live point."""
    interior = jnp.stack(
        tuple(
            (
                axis[guard_cells] + (axis[1] - axis[0]) / 2
                if resolved
                else axis[guard_cells]
            )
            for axis, resolved in zip(grid, active_axes)
        )
    )
    return jnp.where(active[..., None], position, interior)


def positive_definite_3x3(gamma):
    """Sylvester criterion for the symmetric 3x3 tensor, no eigensolver workspace.

    This only validates the tensor: no metric value, inverse or derivative is
    modified. Symmetrization matches eigvalsh's default interpretation; the
    independent symmetry check below still rejects asymmetric supplied tensors.
    """
    a=gamma[...,0,0];d=gamma[...,1,1];f=gamma[...,2,2]
    b=(gamma[...,0,1]+gamma[...,1,0])/2
    c=(gamma[...,0,2]+gamma[...,2,0])/2
    e=(gamma[...,1,2]+gamma[...,2,1])/2
    minor=a*d-b*b
    determinant=a*(d*f-e*e)-b*(b*f-c*e)+c*(b*e-c*d)
    return (a>0)&(minor>0)&(determinant>0)&jnp.isfinite(minor)&jnp.isfinite(determinant)


def particle_metric_valid(metric, position, metric_name):
    """Diagnostic validity only; no regularization of failed active samples."""
    valid = (
        jnp.all(jnp.isfinite(metric.gamma), axis=(-2, -1))
        & jnp.all(jnp.isfinite(metric.gamma_inv), axis=(-2, -1))
        & jnp.isfinite(metric.lapse)
        & (metric.lapse > 0)
        & jnp.all(jnp.isfinite(metric.grad_lapse), axis=-1)
        & jnp.all(jnp.isfinite(metric.grad_shift), axis=(-2,-1))
        & jnp.all(jnp.isfinite(metric.shift), axis=-1)
        & positive_definite_3x3(metric.gamma)
        & jnp.all(
            jnp.abs(metric.gamma - jnp.swapaxes(metric.gamma, -1, -2))
            <= 32
            * jnp.finfo(metric.gamma.dtype).eps
            * jnp.maximum(1.0, jnp.abs(metric.gamma)),
            axis=(-2, -1),
        )
    )
    if metric_name in ("flat_spherical", "kerr_schild_spherical"):
        from PyPIC3D.boundary_conditions.polar import axes

        valid = valid & ~axes(position) & (position[..., 0] != 0)
    elif metric_name == "flat_cylindrical":
        valid = valid & (position[..., 0] != 0)
    return valid


def check_particle_samples(valid, position, stage, tile=None):
    """Checks activate under checkify; no host callbacks inside device kernels."""
    if valid.size == 0:
        return
    first = jnp.argmax((~valid).reshape(-1))
    if tile is None and valid.ndim == 5:
        index = jnp.unravel_index(first, valid.shape)
        tile = jnp.stack(index[:3])
        slot = index[3]*valid.shape[4] + index[4]
    else:
        tile = jnp.asarray((-1, -1, -1)) if tile is None else jnp.asarray(tile)
        slot = first
    checkify.debug_check(
        jnp.all(valid),
        stage
        + ": invalid particle sample; tile={tile}, flattened species/slot={slot}, position={position}",
        tile=tile,
        slot=slot,
        position=position.reshape(-1, 3)[first],
    )


def validate_particle_metric_grids(metric, grids, active_axes, guard_cells):
    """Host validation for supplied uniform static grids, including their halos."""
    import numpy as np

    if guard_cells < 3:
        raise ValueError("Hybrid Hermite particle metrics require guard_cells >= 3")
    expected = metric.center.gamma.shape[:6]
    for k, values in enumerate(grids):
        values = np.asarray(values)
        if values.shape[:3] != expected[:3] or values.shape[-1] != expected[3 + k]:
            raise ValueError("Metric tile and coordinate grid shapes differ")
        if not np.isfinite(values).all():
            raise ValueError("Metric coordinates must be finite")
        if active_axes[k]:
            spacing = np.diff(values, axis=-1)
            if values.shape[-1] < 4 or not (spacing > 0).all():
                raise ValueError("Hermite grids require four ordered nodes")
            tolerance = (
                32 * np.finfo(values.dtype).eps * max(1.0, np.max(np.abs(values)))
            )
            if not np.allclose(spacing, spacing[..., :1], rtol=1e-12, atol=tolerance):
                raise ValueError("Hermite particle metrics require uniform grids")
    for name, tail in (("gamma", (3, 3)), ("lapse", ()), ("shift", (3,))):
        if getattr(metric.center, name).shape != expected + tail:
            raise ValueError("Metric primitive arrays have inconsistent shapes")
