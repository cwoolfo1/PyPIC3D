"""Positive proper number density and metric-aware staggered-grid magnetization."""
from typing import NamedTuple
import jax
import jax.numpy as jnp

from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.relativity.core import B_FIELD_LOCATIONS
from PyPIC3D.solvers.gr_static.static_metric import _location_interpolate


class Magnetization(NamedTuple):
    number_density: jax.Array  # (species, tile_x, tile_y, tile_z, r, theta, phi)
    magnetic_squared: jax.Array
    sigma: jax.Array
    valid: jax.Array


def proper_volume(metric, dynamic):
    """Positive physical measure, independent of the oriented Maxwell Jacobian."""
    if metric.geometry is not None:
        return metric.geometry.volume
    return jnp.abs(metric.center.sqrt_gamma) * dynamic.dx * dynamic.dy * dynamic.dz


def deposit_number_density(particles, species, template, metric, static, dynamic):
    """Reuse production shape deposition/halo folding with unit species charge."""
    result = []
    for s in range(species.charge.shape[0]):
        config = species._replace(charge=jnp.arange(species.charge.shape[0]) == s)
        conformal = compute_rho(particles, config, template, static, dynamic)
        if metric.geometry is not None:
            from PyPIC3D.boundary_conditions.polar import divide
            result.append(divide(conformal*dynamic.dx*dynamic.dy*dynamic.dz,metric.geometry.volume))
        else:
            result.append(conformal / jnp.abs(metric.center.sqrt_gamma))
    return jnp.stack(result)


def collocate_magnetic_field(B, metric):
    """Collocate physical components in the polar chart, including finite caps.

    The legacy point-metric path retains densitized interpolation. Never contract
    components sampled at different Yee positions.
    """
    if metric.geometry is not None:
        return jnp.stack([_location_interpolate(B[i],loc,("C",)*3)
                          for i,loc in enumerate(B_FIELD_LOCATIONS)],axis=-1)
    return jnp.stack([
        _location_interpolate(metric.B[i].sqrt_gamma*B[i], location, ("C",)*3)
        / metric.center.sqrt_gamma
        for i, location in enumerate(B_FIELD_LOCATIONS)
    ], axis=-1)


def magnetization_from_density(B, number_density, masses, metric):
    field = collocate_magnetic_field(B, metric)
    b2 = jnp.einsum("...i,...ij,...j->...", field, metric.center.gamma, field)
    mass_density = jnp.einsum("s,s...->...", masses, number_density)
    valid = (jnp.all(jnp.isfinite(number_density) & (number_density >= 0), axis=0)
             & jnp.isfinite(b2) & (b2 >= 0))
    sigma = jnp.where(mass_density > 0, b2/(4*jnp.pi*jnp.where(mass_density>0, mass_density, 1)),
                      jnp.where(b2 > 0, jnp.inf, 0.))
    return Magnetization(number_density, b2, jnp.where(valid, sigma, jnp.nan), valid)


def measure_magnetization(particles, species, B, metric, static, dynamic):
    n = deposit_number_density(particles, species, jnp.zeros_like(B[0]), metric, static, dynamic)
    return magnetization_from_density(B, n, species.mass, metric)


def self_test():
    from demos.bz_monopole.simulation_parameters import SimulationParameters, build_runtime
    p = SimulationParameters(devices=1, nr=16, ntheta=16, r_max=4., sponge_start=3.)
    _, _, metric, _ = build_runtime(p)
    # A constant off-diagonal metric is an independent, exactly known contraction.
    tensor = jnp.array([[2., .3, 0.], [.3, 1.5, .2], [0., .2, 1.]])
    shape = metric.center.sqrt_gamma.shape
    def constant(m):
        return m._replace(gamma=jnp.broadcast_to(tensor, shape+(3, 3)),
                          gamma_inv=jnp.broadcast_to(jnp.linalg.inv(tensor), shape+(3, 3)),
                          sqrt_gamma=jnp.full(shape, jnp.sqrt(jnp.linalg.det(tensor))))
    metric = metric._replace(center=constant(metric.center), B=tuple(constant(m) for m in metric.B))
    B = tuple(jnp.full(shape, x) for x in (2., 3., 4.))
    n = jnp.full((2,)+shape, 45.9/(8*jnp.pi*100))
    result = magnetization_from_density(B, n, jnp.ones(2), metric)
    assert bool(jnp.allclose(result.magnetic_squared, 45.9, rtol=1e-13))
    assert bool(jnp.allclose(result.sigma, 100., rtol=1e-13))
    print("magnetization: PASS (B^2=45.9, sigma=100, off-diagonal metric)")


if __name__ == "__main__":
    self_test()
