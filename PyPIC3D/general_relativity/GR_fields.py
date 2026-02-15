import jax.numpy as jnp
from jax import jit, lax
from functools import partial



def _curl_forward(Fx, Fy, Fz, dx, dy, dz):
    Fx = jnp.pad(Fx, ((1, 1), (1, 1), (1, 1)), mode="wrap")
    Fy = jnp.pad(Fy, ((1, 1), (1, 1), (1, 1)), mode="wrap")
    Fz = jnp.pad(Fz, ((1, 1), (1, 1), (1, 1)), mode="wrap")

    dFz_dy = (jnp.roll(Fz, shift=-1, axis=1) - Fz) / dy
    dFx_dy = (jnp.roll(Fx, shift=-1, axis=1) - Fx) / dy
    dFy_dz = (jnp.roll(Fy, shift=-1, axis=2) - Fy) / dz
    dFx_dz = (jnp.roll(Fx, shift=-1, axis=2) - Fx) / dz
    dFz_dx = (jnp.roll(Fz, shift=-1, axis=0) - Fz) / dx
    dFy_dx = (jnp.roll(Fy, shift=-1, axis=0) - Fy) / dx

    curl_x = (dFz_dy - dFy_dz)[1:-1, 1:-1, 1:-1]
    curl_y = (dFx_dz - dFz_dx)[1:-1, 1:-1, 1:-1]
    curl_z = (dFy_dx - dFx_dy)[1:-1, 1:-1, 1:-1]
    return curl_x, curl_y, curl_z


def _curl_backward(Fx, Fy, Fz, dx, dy, dz):
    Fx = jnp.pad(Fx, ((1, 1), (1, 1), (1, 1)), mode="wrap")
    Fy = jnp.pad(Fy, ((1, 1), (1, 1), (1, 1)), mode="wrap")
    Fz = jnp.pad(Fz, ((1, 1), (1, 1), (1, 1)), mode="wrap")

    dFz_dy = (Fz - jnp.roll(Fz, shift=1, axis=1)) / dy
    dFx_dy = (Fx - jnp.roll(Fx, shift=1, axis=1)) / dy
    dFy_dz = (Fy - jnp.roll(Fy, shift=1, axis=2)) / dz
    dFx_dz = (Fx - jnp.roll(Fx, shift=1, axis=2)) / dz
    dFz_dx = (Fz - jnp.roll(Fz, shift=1, axis=0)) / dx
    dFy_dx = (Fy - jnp.roll(Fy, shift=1, axis=0)) / dx

    curl_x = (dFz_dy - dFy_dz)[1:-1, 1:-1, 1:-1]
    curl_y = (dFx_dz - dFz_dx)[1:-1, 1:-1, 1:-1]
    curl_z = (dFy_dx - dFx_dy)[1:-1, 1:-1, 1:-1]
    return curl_x, curl_y, curl_z


def GR_Update_D(D, H, J, world, constants):
    dx, dy, dz = world["dx"], world["dy"], world["dz"]
    dt = world["dt"]
    mu0 = constants["mu0"]
    C   = constants["C"]
    # unpack constants and world parameters

    metric = world["metric"]
    # (3, 3, Nx, Ny, Nz) for relativity
    h = metric[1:, 1:, ...]
    # get the space terms of the metric
    h = h.transpose((2, 3, 4, 0, 1))  # (Nx, Ny, Nz, 3, 3)
    # flip the last two dimensions to get the correct order for matrix determinant
    det_h = jnp.linalg.det(h)
    # compute the determinant of the spatial metric

    Dx, Dy, Dz = D
    Hx, Hy, Hz = H
    # unpack the D and H fields

    curl_H = _curl_forward(Hx, Hy, Hz, dx, dy, dz)
    # compute the curl of H

    dDx_dt = C / jnp.sqrt(det_h) * curl_H[0] - mu0 * J[0]
    dDy_dt = C / jnp.sqrt(det_h) * curl_H[1] - mu0 * J[1]
    dDz_dt = C / jnp.sqrt(det_h) * curl_H[2] - mu0 * J[2]
    # compute the time derivatives of D using the curl of H and the current density J

    Dx = Dx + dt * dDx_dt
    Dy = Dy + dt * dDy_dt
    Dz = Dz + dt * dDz_dt
    # update D using the computed time derivatives

    return Dx, Dy, Dz


def GR_Update_B(B, E, world, constants):
    dx, dy, dz = world["dx"], world["dy"], world["dz"]
    dt = world["dt"]
    C   = constants["C"]
    # unpack constants and world parameters

    metric = world["metric"]
    # (3, 3, Nx, Ny, Nz) for relativity
    h = metric[1:, 1:, ...]
    # get the space terms of the metric
    h = h.transpose((2, 3, 4, 0, 1))  # (Nx, Ny, Nz, 3, 3)
    # flip the last two dimensions to get the correct order for matrix determinant
    det_h = jnp.linalg.det(h)
    # compute the determinant of the spatial metric

    Ex, Ey, Ez = E
    # unpack the E field

    curl_E = _curl_backward(Ex, Ey, Ez, dx, dy, dz)
    # compute the curl of E

    dBx_dt = -C / jnp.sqrt(det_h) * curl_E[0]
    dBy_dt = -C / jnp.sqrt(det_h) * curl_E[1]
    dBz_dt = -C / jnp.sqrt(det_h) * curl_E[2]
    # compute the time derivatives of B using the curl of E

    Bx = B[0] + dt * dBx_dt
    By = B[1] + dt * dBy_dt
    Bz = B[2] + dt * dBz_dt
    # update B using the computed time derivatives

    return Bx, By, Bz

def GR_Update_H(B, D, world, constants):
    
    metric = world["metric"]
    # (3, 3, Nx, Ny, Nz) for relativity

    h = metric[1:, 1:, ...]
    # get the space terms of the metric
    beta = metric[0, 1:, ...]
    # get the shift vector components
    alpha = metric[0, 0, ...]
    # get the lapse function

    h_ = h.transpose((2, 3, 4, 0, 1))  # (Nx, Ny, Nz, 3, 3)
    # flip the last two dimensions to get the correct order for matrix determinant
    det_h = jnp.linalg.det(h_)
    # compute the determinant of the spatial metric

    D = jnp.stack(D, axis=0)  # (3, Nx, Ny, Nz)
    B = jnp.stack(B, axis=0)  # (3, Nx, Ny, Nz)
    # stack D and B for easier manipulation

    H = alpha * jnp.einsum("ij...,j...->i...", h, D)  # (3, Nx, Ny, Nz)
    H = H - 1/jnp.sqrt(det_h) * jnp.cross(beta, D, axis=0)  # (3, Nx, Ny, Nz)
    # compute H using the metric constitutive relation

    return H[0], H[1], H[2]

def GR_Update_E(B, D, world, constants):
    
    metric = world["metric"]
    # (3, 3, Nx, Ny, Nz) for relativity

    h = metric[1:, 1:, ...]
    # get the space terms of the metric
    beta = metric[0, 1:, ...]
    # get the shift vector components
    alpha = metric[0, 0, ...]
    # get the lapse function

    h_ = h.transpose((2, 3, 4, 0, 1))  # (Nx, Ny, Nz, 3, 3)
    # flip the last two dimensions to get the correct order for matrix determinant
    det_h = jnp.linalg.det(h_)
    # compute the determinant of the spatial metric

    D = jnp.stack(D, axis=0)  # (3, Nx, Ny, Nz)
    B = jnp.stack(B, axis=0)  # (3, Nx, Ny, Nz)
    # stack D and B for easier manipulation

    E = alpha * jnp.einsum("ij...,j...->i...", h, D)  # (3, Nx, Ny, Nz)
    E = E + 1/jnp.sqrt(det_h) * jnp.cross(beta, B, axis=0)  # (3, Nx, Ny, Nz)
    # compute E using the metric constitutive relation

    return E[0], E[1], E[2]