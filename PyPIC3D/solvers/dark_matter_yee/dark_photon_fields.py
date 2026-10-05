# Christopher Woolford October 2026

# This script contains update methods for the Proca equations for 
# a special relativity formulation of the dark photon field.
# I am dynamically evolving E and A, while computing B on the fly from A.

import jax
import jax.numpy as jnp


from PyPIC3D.solvers.yee.first_order_yee import (
    yee_curl_e_to_b,
    yee_curl_b_to_e,
)

from PyPIC3D.solvers.electrostatic.electrostatic_yee import (
    _centered_tiled_electrostatic_gradient
)

def update_dark_E(E_nplushalf, B_n, A_n, J_n, static_parameters, dynamic_parameters, dt):
    """Update the electric field E using the Proca equations."""

    curl_B = yee_curl_b_to_e(B_n, static_parameters, dynamic_parameters)
    # compute the curl of B

    sin_chi = static_parameters.sin_chi
    # get the mixing angle parameter
    dark_mu = static_parameters.dark_mu
    # get the dark photon mass parameter

    E_new = E_nplushalf + dt * (curl_B - dark_mu**2 * A_n - sin_chi * J_n)
    # update E using the Proca equation

    return E_new



def compute_dark_B(A_n, static_parameters, dynamic_parameters):
    """Compute the magnetic field B from the vector potential A."""

    B_n = yee_curl_e_to_b(A_n, static_parameters, dynamic_parameters)
    # compute the curl of A to get B

    return B_n

def update_dark_A(E_n, A_n, phi_n, J_n, static_parameters, dynamic_parameters, dt):
    """Update the vector potential A using the Proca equations."""

    guard_cells = static_parameters.g
    # the number of guard cells in the simulation

    grad_phi = _centered_tiled_electrostatic_gradient(phi_n, static_parameters, dynamic_parameters, guard_cells)
    # compute the gradient of the scalar potential phi

    A_n_plusone = A_n - dt * (E_n + grad_phi)
    # update A using the Proca equation

    return A_n_plusone



def update_dark_phi(E_n, A_n, phi_n, J_n, static_parameters, dynamic_parameters, dt):
    """Update the scalar potential phi using the Proca equations."""

    Ax, Ay, Az = A_n
    # unpack the vector potential A into its components

    grad_Ax = _centered_tiled_electrostatic_gradient(Ax, static_parameters, dynamic_parameters, static_parameters.g)
    grad_Ay = _centered_tiled_electrostatic_gradient(Ay, static_parameters, dynamic_parameters, static_parameters.g)
    grad_Az = _centered_tiled_electrostatic_gradient(Az, static_parameters, dynamic_parameters, static_parameters.g)
    # compute the gradients of the components of A

    div_A = grad_Ax[0] + grad_Ay[1] + grad_Az[2]
    # compute the divergence of A

    phi_n_plusone = phi_n - dt * (div_A)
    # update phi using the Proca equation

    return phi_n_plusone