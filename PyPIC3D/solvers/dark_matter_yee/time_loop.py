"""Coupled Maxwell-Proca PIC with E'/phi' at n and A' at n-1/2."""

from PyPIC3D.deposition.Esirkepov import Esirkepov_current
from PyPIC3D.deposition.J_from_rhov import J_from_rhov
from PyPIC3D.particles.particle_tile_communication import (
    refresh_tiled_particle_tiles,
    update_tiled_particle_positions,
)
from PyPIC3D.pusher.particle_push import particle_push
from PyPIC3D.utilities.field_helpers import add_external_fields
from PyPIC3D.solvers.yee.first_order_yee import update_B, update_E
from PyPIC3D.solvers.yee.time_loop import _filter_electric_field_for_particles
from .dark_photon_fields import advance_dark_photon_fields, synchronized_dark_fields


__all__ = ["time_loop_dark_photon"]


def dark_photon_push_fields(E, B, dark_fields, external_fields, static_parameters, dynamic_parameters):
    """Build the integer-time force fields, including adjoint current filtering."""
    dark_E, _A, _phi, dark_B = synchronized_dark_fields(dark_fields, static_parameters, dynamic_parameters)
    mixing = static_parameters.sin_chi
    E = tuple(e - mixing * dark_e for e, dark_e in zip(E, dark_E))
    B = tuple(b - mixing * dark_b for b, dark_b in zip(B, dark_B))
    E = _filter_electric_field_for_particles(E, static_parameters, dynamic_parameters)
    return add_external_fields(E, B, external_fields)


def time_loop_dark_photon(particles, species_config, fields, static_parameters, dynamic_parameters):
    """Advance particles and both field sectors by one timestep."""
    E, B, J, rho, phi, external_fields, pml_state, dark_fields, overflow = fields
    dt = dynamic_parameters.dt
    push_E, push_B = dark_photon_push_fields(
        E, B, dark_fields, external_fields, static_parameters, dynamic_parameters,
    )
    particles = particle_push(particles, species_config, push_E, push_B, static_parameters, dynamic_parameters)

    if static_parameters.current_deposition == "esirkepov":
        J = Esirkepov_current(particles, species_config, J, static_parameters, dynamic_parameters)
        particles = update_tiled_particle_positions(particles, species_config, dt)
    else:
        particles = update_tiled_particle_positions(particles, species_config, dt / 2)
        particles, new_overflow = refresh_tiled_particle_tiles(particles, static_parameters, dynamic_parameters)
        overflow = overflow | new_overflow
        J = J_from_rhov(particles, species_config, J, static_parameters, dynamic_parameters)
        particles = update_tiled_particle_positions(particles, species_config, dt / 2)
    particles, new_overflow = refresh_tiled_particle_tiles(particles, static_parameters, dynamic_parameters)
    overflow = overflow | new_overflow

    B, pml_state = update_B(E, B, static_parameters, dynamic_parameters, pml_state)
    E, pml_state = update_E(E, B, J, static_parameters, dynamic_parameters, pml_state)
    B, pml_state = update_B(E, B, static_parameters, dynamic_parameters, pml_state)
    dark_fields = advance_dark_photon_fields(dark_fields, J, static_parameters, dynamic_parameters)

    return particles, (E, B, J, rho, phi, external_fields, pml_state, dark_fields, overflow)
