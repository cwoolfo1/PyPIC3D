import jax
import jax.numpy as jnp
from jax import jit
from jax import lax
import functools
from functools import partial
# import external libraries

from PyPIC3D.utils import digital_filter
# import internal libraries


@partial(jit, static_argnames=())
def update_E_noether(E_prev, B, J, world, constants):

    # dEdt^t = C^2 curl(B)^{t} - J^{t} / eps
    # use a time centered update for E:
    # E^{t+dt} = E^{t-dt} + 2*dt * dEdt^{t}

    Ex_prev, Ey_prev, Ez_prev = E_prev
    # E at previous time step
    Bx, By, Bz = B
    # B at current time step
    Jx, Jy, Jz = J
    # J at current time step

    dt = world['dt']
    dx, dy, dz = world['dx'], world['dy'], world['dz']
    # get the time resolution and grid spacings
    C = constants['C']
    eps = constants['eps']
    # get the time resolution and necessary constants

    Bx = jnp.pad(Bx, ((1,1), (1,1), (1,1)), mode="wrap")
    By = jnp.pad(By, ((1,1), (1,1), (1,1)), mode="wrap")
    Bz = jnp.pad(Bz, ((1,1), (1,1), (1,1)), mode="wrap")
    # pad the magnetic field components for periodic boundary conditions

    dBz_dy = (jnp.roll(Bz, shift=-1, axis=1) - Bz) / dy
    dBx_dy = (jnp.roll(Bx, shift=-1, axis=1) - Bx) / dy
    dBy_dz = (jnp.roll(By, shift=-1, axis=2) - By) / dz
    dBx_dz = (jnp.roll(Bx, shift=-1, axis=2) - Bx) / dz
    dBz_dx = (jnp.roll(Bz, shift=-1, axis=0) - Bz) / dx
    dBy_dx = (jnp.roll(By, shift=-1, axis=0) - By) / dx
    # calculate the spatial derivatives of B


    curl_x = (dBz_dy - dBy_dz)[1:-1,1:-1,1:-1]
    curl_y = (dBx_dz - dBz_dx)[1:-1,1:-1,1:-1]
    curl_z = (dBy_dx - dBx_dy)[1:-1,1:-1,1:-1]
    # calculate the curl of the magnetic field

    Ex = Ex_prev + ( C**2 * curl_x - Jx / eps ) * (2*dt)
    Ey = Ey_prev + ( C**2 * curl_y - Jy / eps ) * (2*dt)
    Ez = Ez_prev + ( C**2 * curl_z - Jz / eps ) * (2*dt)
    # use a time centered update for the electric field from Maxwell's equations

    E_next = (Ex, Ey, Ez)
    return E_next



@partial(jit, static_argnames=())
def update_B_noether(B_prev, E, world, constants):

    # dBdt^t = - curl(E)^{t}
    # use a time centered update for B:
    # B^{t+dt} = B^{t-dt} + 2*dt * dBdt^{t}

    Bx_prev, By_prev, Bz_prev = B_prev
    # B at previous time step

    Ex, Ey, Ez = E
    # E at current time step

    dt = world['dt']
    # get the time resolution
    dx, dy, dz = world['dx'], world['dy'], world['dz']
    # get the grid spacings

    Ex = jnp.pad(Ex, ((1,1), (1,1), (1,1)), mode="wrap")
    Ey = jnp.pad(Ey, ((1,1), (1,1), (1,1)), mode="wrap")
    Ez = jnp.pad(Ez, ((1,1), (1,1), (1,1)), mode="wrap")
    # pad the electric field components for periodic boundary conditions

    dEz_dy = (Ez - jnp.roll(Ez, shift=1, axis=1)) / dy
    dEx_dy = (Ex - jnp.roll(Ex, shift=1, axis=1)) / dy
    dEy_dz = (Ey - jnp.roll(Ey, shift=1, axis=2)) / dz
    dEx_dz = (Ex - jnp.roll(Ex, shift=1, axis=2)) / dz
    dEz_dx = (Ez - jnp.roll(Ez, shift=1, axis=0)) / dx
    dEy_dx = (Ey - jnp.roll(Ey, shift=1, axis=0)) / dx

    curl_x = (dEz_dy - dEy_dz)[1:-1,1:-1,1:-1]
    curl_y = (dEx_dz - dEz_dx)[1:-1,1:-1,1:-1]
    curl_z = (dEy_dx - dEx_dy)[1:-1,1:-1,1:-1]
    # calculate the curl of the electric field

    Bx_next = Bx_prev - 2*dt*curl_x
    By_next = By_prev - 2*dt*curl_y
    Bz_next = Bz_prev - 2*dt*curl_z
    # update the magnetic field from Maxwell's equations

    B_next = (Bx_next, By_next, Bz_next)
    return B_next