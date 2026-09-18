"""Supplied spherical metric grids sampled in a regular particle basis.

Interpolate background-orthonormal spherical components, then transform to
the requested chart. No analytic spacetime metric is evaluated here. This is
a different between-node reconstruction from coordinate-component Hermite
interpolation; its flat-space limit is a constant Cartesian metric.
"""
import jax
import jax.numpy as jnp

from PyPIC3D.relativity.core import Metric, _christoffel_from_metric_gradient
from PyPIC3D.relativity.hermite_metric import interpolate_hermite
from PyPIC3D.relativity.particle_metric import particle_metric_valid, check_particle_samples

RECONSTRUCTION = 'orthonormal_spherical_hermite_v1'


def validate_regularized_axes(metric, grids):
    """Validate the supplied nodal limits before removing coordinate factors."""
    import numpy as np
    gamma = np.asarray(metric.center.gamma)
    shift = np.asarray(metric.center.shift)
    theta = np.asarray(grids[1])
    axis = np.broadcast_to((np.abs(np.sin(theta)) < 1e-14)[..., None, :, None], gamma.shape[:-2])
    tolerance = 64*np.finfo(gamma.dtype).eps*np.maximum(1., np.max(np.abs(gamma), axis=(-2,-1)))
    regular = np.isfinite(gamma).all(axis=(-2,-1)) & np.isfinite(shift).all(axis=-1)
    for i, j in ((0,1), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2)):
        regular &= np.abs(gamma[..., i,j]) <= tolerance
    regular &= (gamma[..., 0,0] > 0) & (gamma[..., 1,1] > 0)
    regular &= np.abs(shift[..., 1]) <= 64*np.finfo(shift.dtype).eps
    if not np.all(regular[axis]):
        raise ValueError('Cartesian particle chart requires supplied metric data with a regular axis')


def spherical_scale(q):
    r, theta = q[..., 0], q[..., 1]
    return jnp.stack((jnp.ones_like(r), r, r*jnp.sin(theta)), axis=-1)


def spherical_basis(q):
    theta, phi = q[..., 1], q[..., 2]
    st, ct, sf, cf = jnp.sin(theta), jnp.cos(theta), jnp.sin(phi), jnp.cos(phi)
    return jnp.stack((jnp.stack((st*cf, st*sf, ct), axis=-1),
                      jnp.stack((ct*cf, ct*sf, -st), axis=-1),
                      jnp.stack((-sf, cf, jnp.zeros_like(sf)), axis=-1)), axis=-1)


def spherical_to_cartesian(q):
    return q[..., :1]*spherical_basis(q)[..., :, 0]


def cartesian_to_spherical(position, reference_phi=None):
    radius = jnp.linalg.norm(position, axis=-1)
    theta = jnp.arctan2(jnp.hypot(position[..., 0], position[..., 1]), position[..., 2])
    phi = jnp.arctan2(position[..., 1], position[..., 0])
    if reference_phi is not None:
        difference = phi-reference_phi
        phi = reference_phi+jnp.arctan2(jnp.sin(difference), jnp.cos(difference))
    return jnp.stack((radius, theta, phi), axis=-1)


def covariant_to_cartesian(q, momentum):
    return jnp.einsum('...ij,...j->...i', spherical_basis(q), momentum/spherical_scale(q))


def cartesian_to_covariant(q, momentum):
    return spherical_scale(q)*jnp.einsum('...ji,...j->...i', spherical_basis(q), momentum)


def vector_to_cartesian(q, vector):
    return jnp.einsum('...ij,...j->...i', spherical_basis(q), spherical_scale(q)*vector)


def regularized_grid(metric, grid):
    r, theta, phi = jnp.meshgrid(*grid, indexing='ij')
    q = jnp.stack((r, theta, phi), axis=-1)
    scale = spherical_scale(q)
    axis = jnp.abs(jnp.sin(theta)) < 1e-14
    safe = scale.at[..., 2].set(jnp.where(axis, 1., scale[..., 2]))
    gamma = metric.gamma/(safe[..., :, None]*safe[..., None, :])
    # Axisymmetry and a regular axis determine these removable limits. The
    # supplied coordinate tensor has zero phi row/column at the axis itself.
    gamma = gamma.at[..., 2, 2].set(jnp.where(axis, gamma[..., 1, 1], gamma[..., 2, 2]))
    for component in (0, 1):
        gamma = gamma.at[..., component, 2].set(jnp.where(axis, 0., gamma[..., component, 2]))
        gamma = gamma.at[..., 2, component].set(jnp.where(axis, 0., gamma[..., 2, component]))
    return jnp.concatenate((metric.lapse[..., None], metric.shift*scale,
                            gamma.reshape(gamma.shape[:-2]+(9,))), axis=-1)


def sample_regularized_metric(metric, position, grid, metric_name, active_axes,
                              inactive_axis_indices, *, cartesian=False,
                              derivatives=True, check=True, stage='particle metric', tile=None):
    packed = regularized_grid(metric, grid)
    shape = position.shape[:-1]

    def values(point):
        q = cartesian_to_spherical(point) if cartesian else point
        data = interpolate_hermite(packed, q, grid, active_axes, inactive_axis_indices)
        gamma_hat = data[..., 4:].reshape(shape+(3, 3))
        if cartesian:
            basis = spherical_basis(q)
            shift = jnp.einsum('...ij,...j->...i', basis, data[..., 1:4])
            gamma = jnp.einsum('...ia,...ab,...jb->...ij', basis, gamma_hat, basis)
        else:
            scale = spherical_scale(q)
            shift = data[..., 1:4]/scale
            gamma = gamma_hat*scale[..., :, None]*scale[..., None, :]
        return jnp.concatenate((data[..., :1], shift, gamma.reshape(shape+(9,))), axis=-1)

    sampled = values(position)
    gamma = sampled[..., 4:].reshape(shape+(3, 3))
    inverse = jnp.linalg.inv(gamma)
    if derivatives:
        directions = []
        for axis in range(3):
            if cartesian or active_axes[axis]:
                tangent = jnp.broadcast_to(jnp.eye(3, dtype=position.dtype)[axis], position.shape)
                directions.append(jax.jvp(values, (position,), (tangent,))[1])
            else:
                directions.append(jnp.zeros_like(sampled))
        gradient = jnp.stack(directions, axis=-1)
        grad_gamma = gradient[..., 4:, :].reshape(shape+(3, 3, 3))
        grad_inverse = -jnp.einsum('...ij,...jlk,...lm->...kim', inverse, grad_gamma, inverse)
        christoffel = _christoffel_from_metric_gradient(inverse, grad_gamma)
    else:
        gradient = jnp.zeros(shape+(13, 3), dtype=position.dtype)
        grad_inverse = jnp.zeros(shape+(3, 3, 3), dtype=position.dtype)
        christoffel = jnp.zeros_like(grad_inverse)
    root = jnp.sqrt(jnp.linalg.det(gamma))
    if not cartesian:
        root = jnp.copysign(root, jnp.sin(position[..., 1]))
    result = Metric(sampled[..., 0], sampled[..., 1:4], gamma, inverse, root,
                    christoffel, gradient[..., 0, :], gradient[..., 1:4, :])
    if check:
        valid = particle_metric_valid(result, position, 'flat_cartesian' if cartesian else metric_name)
        valid &= jnp.all(jnp.isfinite(grad_inverse), axis=(-3, -2, -1))
        check_particle_samples(valid, position, stage, tile)
    return result, grad_inverse
