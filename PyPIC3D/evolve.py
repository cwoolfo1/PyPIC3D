from PyPIC3D.deposition.Esirkepov import Esirkepov_current
from PyPIC3D.deposition.GR_direct_deposition import GR_direct_deposition
from PyPIC3D.deposition.J_from_rhov import J_from_rhov
from PyPIC3D.particles.particle_tile_communication import (
    refresh_tiled_particle_tiles,
    update_tiled_particle_positions,
)
from PyPIC3D.pusher.particle_push import particle_push
from PyPIC3D.pusher.hybrid_boris_geodesic import hybrid_boris_geodesic_push
from PyPIC3D.solvers.electrostatic_yee import calculate_tiled_electrostatic_fields
from PyPIC3D.solvers.first_order_yee import update_B, update_E
from PyPIC3D.solvers.static_metric import update_B_relativity, update_D_relativity, compute_covariant_E, compute_covariant_H
from PyPIC3D.utils import add_external_fields


__all__ = ["time_loop_electrodynamic", "time_loop_electrostatic", "time_loop_static_metric"]


def time_loop_electrodynamic(
    particles,
    species_config,
    fields,
    static_parameters,
    dynamic_parameters,
):
    """
    Advance a tiled electrodynamic PIC system by one time step.
    """

    E, B, J, rho, phi, external_fields, pml_state, overflow_previous = fields
    # unpack the tiled field state

    dt = dynamic_parameters.dt
    # get the dynamic timestep used by the tiled push/deposition sequence

    push_E, push_B = add_external_fields(E, B, external_fields)
    # particles see evolved fields plus external-only fields

    particles = particle_push(
        particles,
        species_config,
        push_E,
        push_B,
        static_parameters,
        dynamic_parameters,
    )
    # use the selected tiled pusher for particle velocities

    def direct_deposition_step(state):
        particles, J_tiles, overflow_previous = state
        particles = update_tiled_particle_positions(particles, species_config, dt / 2)
        # update particle positions to the centered direct-current deposition time
        particles, overflow = refresh_tiled_particle_tiles(particles, static_parameters, dynamic_parameters)
        # wrap particles and move them into their owning tiles.
        overflow = overflow_previous | overflow
        # keep fixed-capacity tile overflow visible to the Python driver
        J_tiles = J_from_rhov(
            particles,
            species_config,
            J_tiles,
            static_parameters,
            dynamic_parameters,
        )
        # deposit current directly into tile-local Yee current arrays
        particles = update_tiled_particle_positions(particles, species_config, dt / 2)
        # complete the full particle position update
        particles, overflow = refresh_tiled_particle_tiles(particles, static_parameters, dynamic_parameters)
        # refresh tile ownership after the full position update.
        overflow = overflow_previous | overflow
        return particles, J_tiles, overflow
    # if the direct deposition method is selected, first refresh the particle tiles, then deposit current directly into the tiled J arrays

    def esirkepov_deposition_step(state):
        particles, J_tiles, overflow_previous = state
        J_tiles = Esirkepov_current(particles, species_config, J_tiles, static_parameters, dynamic_parameters)
        # deposit current into the tiled J arrays using the Esirkepov method, which requires old and new particle positions
        particles = update_tiled_particle_positions(particles, species_config, dt)
        # update particle positions to the new time step
        particles, overflow = refresh_tiled_particle_tiles(particles, static_parameters, dynamic_parameters)
        # refresh tile ownership after the full position update
        overflow = overflow_previous | overflow
        return particles, J_tiles, overflow
    # if the Esirkepov deposition method is selected, first deposit current into the tiled J arrays, then refresh the particle tiles

    if static_parameters.current_deposition == "esirkepov":
        particles, J, overflow = esirkepov_deposition_step((particles, J, overflow_previous))
    else:
        particles, J, overflow = direct_deposition_step((particles, J, overflow_previous))
    # deposit current into the tiled J arrays using the selected deposition method

    B, pml_state = update_B(E, B, static_parameters, dynamic_parameters, pml_state, do_filter=False)
    # update magnetic field from the previous electric field by half a timestep
    # for no pml, the pml_state is None, and the update_B function returns None for the pml_state

    E, pml_state = update_E(E, B, J, static_parameters, dynamic_parameters, pml_state)
    # update electric field from B and the supplied current
    # for no pml, the pml_state is None, and the update_E function returns None for the pml_state

    B, pml_state = update_B(E, B, static_parameters, dynamic_parameters, pml_state, do_filter=True)
    # update magnetic field from the newly updated electric field by half a timestep
    # for no pml, the pml_state is None, and the update_B function returns None for the pml_state

    fields = (E, B, J, rho, phi, external_fields, pml_state, overflow)
    # pack the tiled field state

    return particles, fields


def time_loop_electrostatic(
    particles,
    species_config,
    fields,
    static_parameters,
    dynamic_parameters,
):
    """
    Advance a tiled electrostatic PIC system by one time step.

    The particle push and retile use tile-local fields. Charge density is
    deposited into tiled scalar storage, assembled for the existing global
    Poisson solve, then the solved potential is tiled again before computing
    tile-local electrostatic E.
    """

    E_tiles, B_tiles, J_tiles, rho_tiles, phi_tiles, external_fields, pml_state, overflow_previous = fields
    # unpack the tiled field state

    dt = dynamic_parameters.dt
    # get the dynamic timestep used by the tiled electrostatic step

    push_E_tiles, push_B_tiles = add_external_fields(E_tiles, B_tiles, external_fields)
    # particles see evolved fields plus prescribed external fields

    particles = particle_push(
        particles,
        species_config,
        push_E_tiles,
        push_B_tiles,
        static_parameters,
        dynamic_parameters,
    )
    # push velocities using the selected tiled particle pusher

    particles = update_tiled_particle_positions(particles, species_config, dt)
    # update particle forward positions before depositing rho

    particles, overflow = refresh_tiled_particle_tiles(particles, static_parameters, dynamic_parameters)
    overflow = overflow_previous | overflow
    # keep fixed-capacity tile overflow visible to the Python driver

    E_tiles, phi_tiles, rho_tiles = calculate_tiled_electrostatic_fields(
        static_parameters,
        dynamic_parameters,
        particles,
        species_config,
        rho_tiles,
        phi_tiles
    )
    # solve electrostatic fields from tiled charge density

    fields = (E_tiles, B_tiles, J_tiles, rho_tiles, phi_tiles, external_fields, pml_state, overflow)
    # pack the tiled field state

    return particles, fields


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

    J_n_plushalf = GR_direct_deposition(
        centered_particles,
        species_config,
        J_n_minushalf,
        metric,
        static_parameters,
        dynamic_parameters,
    )
    # deposit contravariant current density from the centered particles


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
        overflow_previous,
    )
    # pack the fixed-metric field state

    return particles, fields