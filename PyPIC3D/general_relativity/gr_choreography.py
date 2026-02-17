from PyPIC3D.general_relativity.GR_fields import (
    GR_Update_B,
    GR_Update_D,
    GR_Update_E,
    GR_Update_H,
    GR_TimeAverageJ,
)
from PyPIC3D.general_relativity.gr_current import GR_deposit_current
from PyPIC3D.general_relativity.gr_particle_pusher import particle_push_relativistic_metric


def _avg_vec(a, b):
    return tuple(0.5 * (a[i] + b[i]) for i in range(3))


def _regular_step(
    particles,
    E,
    B,
    D,
    H,
    D0,
    B0,
    J,
    J0,
    world,
    constants,
):
    # 1) TimeAverageDB
    D0_half = _avg_vec(D0, D)
    B0_half = _avg_vec(B0, B)

    # 2) ComputeAuxE(D0, B), 3) Faraday(aux)
    auxE_nmh = GR_Update_E(B, D0_half, world, constants)
    B0_n = GR_Update_B(B0_half, auxE_nmh, world, constants)

    # 4) ComputeAuxH(D, B0)
    auxH_n = GR_Update_H(B0_n, D, world, constants)

    # 5) Particle push with (D, B0)
    for i in range(len(particles)):
        particles[i] = particle_push_relativistic_metric(particles[i], D, B0_n, world, constants)

    # 6) Current deposit (cur0 at n+1/2), 7) TimeAverageJ (cur at n)
    J_dep = GR_deposit_current(particles, J, constants, world)
    J_avg = GR_TimeAverageJ(J_dep, J)

    # 8) ComputeAuxE(D, B0), 9) Faraday(main): B0 <- B - curl(auxE)
    auxE_n = GR_Update_E(B0_n, D, world, constants)
    B0_np = GR_Update_B(B, auxE_n, world, constants)

    # 10) Ampere(aux) + AmpereCurrents(aux): D0 <- D0 + curl(auxH) - J_avg
    D0_nph = GR_Update_D(D0_half, auxH_n, J_avg, world, constants)

    # 11) ComputeAuxH(D0, B0)
    auxH_nph = GR_Update_H(B0_np, D0_nph, world, constants)

    # 12) Ampere(main) + AmpereCurrents(main): D0 <- D + curl(auxH) - J_dep
    D0_np1 = GR_Update_D(D, auxH_nph, J_dep, world, constants)

    # 13) SwapFields / Swap currents
    D_new = D0_np1
    B_new = B0_np
    D0_new = tuple(D)
    B0_new = tuple(B)
    J_new = J_dep
    J0_new = J_avg

    # 14) Recover constitutive fields from updated main EM state
    H_new = GR_Update_H(B_new, D_new, world, constants)
    E_new = GR_Update_E(B_new, D_new, world, constants)

    return particles, E_new, B_new, D_new, H_new, D0_new, B0_new, J_new, J0_new


def gr_entity_choreography_step(
    particles,
    E,
    B,
    D,
    H,
    D0,
    B0,
    J,
    J0,
    world,
    constants,
):
    """
    GR field/particle sequencing modeled after Entity's choreography.

    Stored state mapping:
    - (D, B): main EM state
    - (D0, B0): stagger/history EM state
    - J: half-step current-like history
    - J0: centered/current-average history
    """

    return _regular_step(
        particles, E, B, D, H, D0, B0, J, J0, world, constants
    )
