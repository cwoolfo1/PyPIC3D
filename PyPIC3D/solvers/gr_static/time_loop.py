from PyPIC3D.deposition.GR_direct_deposition import GR_direct_deposition
from PyPIC3D.particles.particle_tile_communication import refresh_tiled_particle_tiles
from PyPIC3D.pusher.hybrid_boris_geodesic import hybrid_boris_geodesic_push
from PyPIC3D.utilities.field_helpers import add_external_fields

from .static_metric import (
    compute_covariant_E,
    compute_covariant_H,
    update_B_relativity,
    update_D_relativity,
)


__all__ = ["time_loop_static_metric"]


def time_loop_static_metric(
    particles,
    species_config,
    fields,
    static_parameters,
    dynamic_parameters,
):
    """
    Advance a tiled PIC system in a prescribed 3+1 metric.

    The first field slot is the contravariant displacement field ``D^i``.  The
    particle velocity slot stores covariant spatial components ``u_i``.
    """

    D_n, B_n_minushalf, J_n_minushalf, rho, phi, external_fields, metric, previous_fields, overflow_previous = fields
    # unpack the fixed-metric field state

    D_n_minusone, B_n_minusthreehalves = previous_fields
    # unpack the previous fixed-metric field state

    D_n_minushalf = tuple( 0.5 * (D_n[i] + D_n_minusone[i]) for i in range(3) )
    B_n_minusone = tuple( 0.5 * (B_n_minushalf[i] + B_n_minusthreehalves[i]) for i in range(3) )
    # compute the centered fields for the current time step

    E_n_minusonehalf = compute_covariant_E(D_n_minushalf, B_n_minushalf, metric)
    # compute the covariant electric field from the centered displacement and magnetic fields

    B_n = update_B_relativity(E_n_minusonehalf, B_n_minusone, metric, static_parameters, dynamic_parameters, dynamic_parameters.dt)
    # update the contravariant magnetic field using the centered displacement field

    push_D, push_B = add_external_fields(D_n, B_n, external_fields)
    # particles see evolved fields plus prescribed external fields

    particles, centered_particles = hybrid_boris_geodesic_push(
        particles,
        species_config,
        push_D,
        push_B,
        metric,
        static_parameters,
        dynamic_parameters,
    )
    # advance full-step particles and keep the intermediate particles (x_n_plushalf, v_n_plushalf) for the centered current deposition

    centered_particles, centered_overflow = refresh_tiled_particle_tiles(
        centered_particles,
        static_parameters,
        dynamic_parameters,
    )
    # apply particle boundaries and move midpoint particles into the tiles that own the current-deposition positions

    J_n_plushalf = GR_direct_deposition(
        centered_particles,
        species_config,
        J_n_minushalf,
        metric,
        static_parameters,
        dynamic_parameters,
    )
    # deposit contravariant current density from the centered particles

    particles, fullstep_overflow = refresh_tiled_particle_tiles(
        particles,
        static_parameters,
        dynamic_parameters,
    )
    overflow = overflow_previous | centered_overflow | fullstep_overflow
    # apply boundaries and restore full-step tile ownership for the next push while preserving all overflow events


    E_n = compute_covariant_E(D_n, B_n, metric)
    # compute the covariant electric field from the updated displacement and magnetic fields
    H_n = compute_covariant_H(D_n, B_n, metric)
    # compute the covariant magnetic field from the updated displacement and magnetic fields

    B_n_plushalf = update_B_relativity(E_n, B_n_minushalf, metric, static_parameters, dynamic_parameters, dynamic_parameters.dt)
    # update the contravariant magnetic field using the updated displacement field

    J_n = tuple( 0.5 * (J_n_plushalf[i] + J_n_minushalf[i]) for i in range(3))
    # compute the centered current for the current time step

    D_n_plushalf = update_D_relativity(D_n_minushalf, H_n, J_n, metric, static_parameters, dynamic_parameters, dynamic_parameters.dt)
    # update the contravariant displacement field using the updated magnetic field and current

    H_n_plushalf = compute_covariant_H(D_n_plushalf, B_n_plushalf, metric)
    # compute the covariant magnetic field from the updated displacement and magnetic fields

    D_n_plusone = update_D_relativity(D_n, H_n_plushalf, J_n_plushalf, metric, static_parameters, dynamic_parameters, dynamic_parameters.dt)
    # update the contravariant displacement field using the updated magnetic field and current


    previous_fields = (D_n, B_n_minushalf)
    # store the current fields for the next time step

    fields = (
        D_n_plusone,
        B_n_plushalf,
        J_n_plushalf,
        rho,
        phi,
        external_fields,
        metric,
        previous_fields,
        overflow,
    )
    # pack the fixed-metric field state

    return particles, fields
