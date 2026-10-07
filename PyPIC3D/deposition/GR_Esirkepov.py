from functools import partial

import jax
import jax.numpy as jnp

from PyPIC3D.deposition.Esirkepov import _deposit_esirkepov_tiles
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles


__all__ = ["GR_Esirkepov_current"]


@partial(jax.jit, static_argnames="static_parameters")
def GR_Esirkepov_current(
    particles_old: TiledParticles,
    particles_new: TiledParticles,
    species_config: SpeciesConfig,
    J,
    metric,
    static_parameters,
    dynamic_parameters,
):
    """
    Charge-conserving Esirkepov current deposition for a fixed 3+1 metric.

    The scheme deposits the conformal current ``sqrt(gamma) J^i`` and returns it
    directly for the densitized Maxwell update.

    Unlike the direct deposit, this one satisfies the discrete continuity
    equation exactly.  The conformal charge density carries no metric,

        sqrt(gamma) rho = q S(x) / (dx dy dz),

    so the conformal continuity equation

        d_t( sqrt(gamma) rho ) + d_i( sqrt(gamma) J^i ) = 0

    is the flat Esirkepov identity verbatim and the ordinary density
    decomposition applies unchanged.  Because the backward-difference divergence
    of the backward-difference curl in ``update_D`` vanishes
    identically, satisfying that equation preserves

        d_i( sqrt(gamma) D^i ) = 4 pi sqrt(gamma) rho

    to round-off, with no divergence cleaning.

    ``particles_old`` holds ``x^n`` and ``particles_new`` holds ``x^{n+1}``.
    Both must be supplied before the full-step retile, so that they share one
    tile frame and neither has been wrapped by the periodic boundary -- the
    deposition differences the two positions directly and a wrap would appear as
    a domain-sized displacement.  Passing the positions explicitly is required:
    ``particles.u`` stores covariant ``u_i``, so the flat shortcut
    ``x^n = x^{n+1} - dt u^{n+1/2}`` does not hold in a curved chart.

    No current filter is applied here. Filtering only J destroys continuity
    with the raw deposited charge, so the configuration layer rejects it.
    A caller may instead transform integrated current and charge with compatible
    operators that commute with the discrete divergence.
    """

    conformal_current = _deposit_esirkepov_tiles(
        particles_old.x, particles_new.x, particles_new.active, species_config, J,
        static_parameters, dynamic_parameters, trajectory=_gr_trajectory,
    )
    return conformal_current


def _gr_trajectory(old_position, endpoint, update_axes, dt):
    """Use actual pre-retile endpoints and their coordinate displacement.

    Covariant particle momentum cannot predict these endpoints. Frozen axes
    stay fixed; unresolved axes deposit the velocity produced by the position
    update without an additional particle-metric sample.
    """
    new_position = tuple(
        jnp.where(update, new, old)
        for old, new, update in zip(old_position, endpoint, update_axes)
    )
    velocity = tuple((new - old) / dt for new, old in zip(new_position, old_position))
    return new_position, velocity
