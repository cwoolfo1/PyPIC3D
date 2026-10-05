# Christopher Woolford October 2026

# This script contains my time evolution loop for 
# the coupled Maxwell-Proca equations for the dark photon 
# field. I am dynamically evolving the dark photon E and A fields
# while computing the dark photon B field on the fly from A for the
# particle update.

import jax
from PyPIC3D.deposition.Esirkepov import Esirkepov_current
from PyPIC3D.deposition.J_from_rhov import J_from_rhov
from PyPIC3D.particles.particle_tile_communication import (
    refresh_tiled_particle_tiles,
    update_tiled_particle_positions,
)
from PyPIC3D.pusher.particle_push import particle_push
from PyPIC3D.utilities.field_helpers import add_external_fields
from PyPIC3D.solvers.yee.first_order_yee import update_B, update_E

from PyPIC3D.solvers.dark_matter_yee.dark_photon_fields import (
    update_dark_E,
    compute_dark_B,
    update_dark_A,
    update_dark_phi
)

__all__ = ["time_loop_dark_photon"]


def time_loop_dark_photon(
    particles,
    species_config,
    fields,
    static_parameters,
    dynamic_parameters,
):
    """
    Advance a tiled dark photon PIC system by one time step.
    """

    E, B, J, rho, phi, dark_photon_fields, external_fields, pml_state, overflow_previous = fields
    # unpack the tiled field state

    dt = dynamic_parameters.dt
    # get the dynamic timestep used by the tiled push/deposition sequence


    Eprime, Aprime, phiprime = dark_photon_fields
    # unpack the dark photon field state

    push_E, push_B = add_external_fields(E, B, external_fields)
    # prescribed external fields and the magnetic gather remain unfiltered

    Bprime = compute_dark_B(Aprime, static_parameters, dynamic_parameters)
    # compute the dark photon magnetic field from the vector potential

    dark_photon_push_fields = (Eprime, Bprime)
    # create a tuple of the dark photon fields for use in the particle push

    push_E, push_B = add_external_fields(push_E, push_B, dark_photon_push_fields)
    # add the dark photon fields to the prescribed external fields for the particle push

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

    B, pml_state = update_B(E, B, static_parameters, dynamic_parameters, pml_state)
    # update magnetic field from the previous electric field by half a timestep
    # for no pml, the pml_state is None, and the update_B function returns None for the pml_state

    E, pml_state = update_E(E, B, J, static_parameters, dynamic_parameters, pml_state)
    # update electric field from B and the supplied current
    # for no pml, the pml_state is None, and the update_E function returns None for the pml_state

    B, pml_state = update_B(E, B, static_parameters, dynamic_parameters, pml_state)
    # update magnetic field from the newly updated electric field by half a timestep
    # for no pml, the pml_state is None, and the update_B function returns None for the pml_state



    ######################### DARK PHOTON FIELD UPDATES #########################

    Aprime = update_dark_A(Eprime, Aprime, phiprime, J, static_parameters, dynamic_parameters, dt)
    # update the dark photon vector potential A using the Proca equations

    phiprime = update_dark_phi(Eprime, Aprime, phiprime, J, static_parameters, dynamic_parameters, dt)
    # update the dark photon scalar potential phi using the Proca equations

    Bprime = compute_dark_B(Aprime, static_parameters, dynamic_parameters)
    # compute the dark photon magnetic field from the vector potential

    Eprime = update_dark_E(Eprime, Bprime, Aprime, J, static_parameters, dynamic_parameters, dt)
    # update the dark photon electric field E using the Proca equations

    Aprime = update_dark_A(Eprime, Aprime, phiprime, J, static_parameters, dynamic_parameters, dt)
    # update the dark photon vector potential A using the Proca equations

    phiprime = update_dark_phi(Eprime, Aprime, phiprime, J, static_parameters, dynamic_parameters, dt)
    # update the dark photon scalar potential phi using the Proca equations

    # Leapfrog the dark photon fields to the next time step using the Proca equations


    dark_photon_fields = (Eprime, Aprime, phiprime)
    # pack the updated dark photon field state

    fields = (E, B, J, rho, phi, dark_photon_fields, external_fields, pml_state, overflow)
    # pack the tiled field state

    return particles, fields
