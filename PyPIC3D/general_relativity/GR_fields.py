import jax.numpy as jnp
from jax import jit, lax
from functools import partial


def _inv_sqrt_det_h_faces(metric):
    """Metric-volume factors at staggered locations for each component."""
    h = metric[1:, 1:, ...]
    h = h.transpose((2, 3, 4, 0, 1))  # (Nx, Ny, Nz, 3, 3)
    det_h = jnp.linalg.det(h)
    inv_sqrt_center = 1.0 / jnp.sqrt(jnp.maximum(det_h, 1e-30))
    inv_sqrt_x = 0.5 * (inv_sqrt_center + jnp.roll(inv_sqrt_center, shift=-1, axis=0))
    inv_sqrt_y = 0.5 * (inv_sqrt_center + jnp.roll(inv_sqrt_center, shift=-1, axis=1))
    inv_sqrt_z = 0.5 * (inv_sqrt_center + jnp.roll(inv_sqrt_center, shift=-1, axis=2))
    return inv_sqrt_x, inv_sqrt_y, inv_sqrt_z, inv_sqrt_center


def _adm_lapse_shift(metric, constants):
    """
    Recover ADM lapse (dimensionless) and contravariant shift from covariant metric.
    """
    h = metric[1:, 1:, ...]  # g_ij
    beta_cov = metric[0, 1:, ...]  # g_0i

    h_ = h.transpose((2, 3, 4, 0, 1))
    h_inv_ = jnp.linalg.inv(h_)
    h_inv = h_inv_.transpose((3, 4, 0, 1, 2))

    beta_contra = jnp.einsum("ij...,j...->i...", h_inv, beta_cov)
    beta_sq = jnp.einsum("i...,i...->...", beta_cov, beta_contra)

    # g00 = -alpha^2 c^2 + beta_i beta^i  -> alpha = sqrt(beta_i beta^i - g00) / c
    c = constants["C"]
    alpha = jnp.sqrt(jnp.maximum(beta_sq - metric[0, 0, ...], 1e-30)) / c
    return alpha, beta_contra, h


def _stagger_scalar_face(field, axis):
    return 0.5 * (field + jnp.roll(field, shift=-1, axis=axis))


def _stagger_vec_face(field, axis):
    return jnp.stack([_stagger_scalar_face(field[i], axis) for i in range(3)], axis=0)


def _stagger_mat_face(field, axis):
    return jnp.stack(
        [
            jnp.stack([_stagger_scalar_face(field[i, j], axis) for j in range(3)], axis=0)
            for i in range(3)
        ],
        axis=0,
    )



def _curl_forward(Fx, Fy, Fz, dx, dy, dz):
    Fx = jnp.pad(Fx, ((1, 1), (1, 1), (1, 1)), mode="wrap")
    Fy = jnp.pad(Fy, ((1, 1), (1, 1), (1, 1)), mode="wrap")
    Fz = jnp.pad(Fz, ((1, 1), (1, 1), (1, 1)), mode="wrap")

    # Match Entity GR kernels: finite differences are in index space and metric
    # factors account for geometry/scale at update sites.
    dFz_dy = jnp.roll(Fz, shift=-1, axis=1) - Fz
    dFx_dy = jnp.roll(Fx, shift=-1, axis=1) - Fx
    dFy_dz = jnp.roll(Fy, shift=-1, axis=2) - Fy
    dFx_dz = jnp.roll(Fx, shift=-1, axis=2) - Fx
    dFz_dx = jnp.roll(Fz, shift=-1, axis=0) - Fz
    dFy_dx = jnp.roll(Fy, shift=-1, axis=0) - Fy

    curl_x = (dFz_dy - dFy_dz)[1:-1, 1:-1, 1:-1]
    curl_y = (dFx_dz - dFz_dx)[1:-1, 1:-1, 1:-1]
    curl_z = (dFy_dx - dFx_dy)[1:-1, 1:-1, 1:-1]
    return curl_x, curl_y, curl_z


def _curl_backward(Fx, Fy, Fz, dx, dy, dz):
    Fx = jnp.pad(Fx, ((1, 1), (1, 1), (1, 1)), mode="wrap")
    Fy = jnp.pad(Fy, ((1, 1), (1, 1), (1, 1)), mode="wrap")
    Fz = jnp.pad(Fz, ((1, 1), (1, 1), (1, 1)), mode="wrap")

    # Match Entity GR kernels: finite differences are in index space and metric
    # factors account for geometry/scale at update sites.
    dFz_dy = Fz - jnp.roll(Fz, shift=1, axis=1)
    dFx_dy = Fx - jnp.roll(Fx, shift=1, axis=1)
    dFy_dz = Fy - jnp.roll(Fy, shift=1, axis=2)
    dFx_dz = Fx - jnp.roll(Fx, shift=1, axis=2)
    dFz_dx = Fz - jnp.roll(Fz, shift=1, axis=0)
    dFy_dx = Fy - jnp.roll(Fy, shift=1, axis=0)

    curl_x = (dFz_dy - dFy_dz)[1:-1, 1:-1, 1:-1]
    curl_y = (dFx_dz - dFz_dx)[1:-1, 1:-1, 1:-1]
    curl_z = (dFy_dx - dFx_dy)[1:-1, 1:-1, 1:-1]
    return curl_x, curl_y, curl_z


def GR_Update_D(D, H, J, world, constants):
    dx, dy, dz = world["dx"], world["dy"], world["dz"]
    dt = world["dt"]
    mu0 = constants["mu"]
    C   = constants["C"]
    # unpack constants and world parameters

    metric = world["metric"]
    inv_sqrt_x, inv_sqrt_y, inv_sqrt_z, inv_sqrt_center = _inv_sqrt_det_h_faces(metric)

    Dx, Dy, Dz = D
    Hx, Hy, Hz = H
    # unpack the D and H fields

    def branch_generic(_):
        curl_H = _curl_forward(Hx, Hy, Hz, dx, dy, dz)
        Jx = J[0] * inv_sqrt_x
        Jy = J[1] * inv_sqrt_y
        Jz = J[2] * inv_sqrt_z
        return (
            C * inv_sqrt_x * curl_H[0] - mu0 * Jx,
            C * inv_sqrt_y * curl_H[1] - mu0 * Jy,
            C * inv_sqrt_z * curl_H[2] - mu0 * Jz,
        )

    dDx_dt, dDy_dt, dDz_dt = branch_generic(None)

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
    inv_sqrt_x, inv_sqrt_y, inv_sqrt_z, inv_sqrt_center = _inv_sqrt_det_h_faces(metric)

    Ex, Ey, Ez = E
    # unpack the E field

    def branch_generic(_):
        curl_E = _curl_backward(Ex, Ey, Ez, dx, dy, dz)
        return -C * inv_sqrt_x * curl_E[0], -C * inv_sqrt_y * curl_E[1], -C * inv_sqrt_z * curl_E[2]

    dBx_dt, dBy_dt, dBz_dt = branch_generic(None)

    Bx = B[0] + dt * dBx_dt
    By = B[1] + dt * dBy_dt
    Bz = B[2] + dt * dBz_dt
    # update B using the computed time derivatives

    return Bx, By, Bz

def GR_Update_H(B, D, world, constants):
    
    metric = world["metric"]
    alpha, beta, h = _adm_lapse_shift(metric, constants)
    inv_sqrt_x, inv_sqrt_y, inv_sqrt_z, _ = _inv_sqrt_det_h_faces(metric)

    D = jnp.stack(D, axis=0)  # (3, Nx, Ny, Nz)
    B = jnp.stack(B, axis=0)  # (3, Nx, Ny, Nz)
    # stack D and B for easier manipulation

    alpha_x = _stagger_scalar_face(alpha, axis=0)
    alpha_y = _stagger_scalar_face(alpha, axis=1)
    alpha_z = _stagger_scalar_face(alpha, axis=2)
    beta_x = _stagger_vec_face(beta, axis=0)
    beta_y = _stagger_vec_face(beta, axis=1)
    beta_z = _stagger_vec_face(beta, axis=2)
    h_x = _stagger_mat_face(h, axis=0)
    h_y = _stagger_mat_face(h, axis=1)
    h_z = _stagger_mat_face(h, axis=2)

    cross_x = jnp.cross(beta_x, D, axis=0)
    cross_y = jnp.cross(beta_y, D, axis=0)
    cross_z = jnp.cross(beta_z, D, axis=0)

    Hx = alpha_x * jnp.einsum("j...,j...->...", h_x[0], B) - inv_sqrt_x * cross_x[0]
    Hy = alpha_y * jnp.einsum("j...,j...->...", h_y[1], B) - inv_sqrt_y * cross_y[1]
    Hz = alpha_z * jnp.einsum("j...,j...->...", h_z[2], B) - inv_sqrt_z * cross_z[2]
    return Hx, Hy, Hz

def GR_Update_E(B, D, world, constants):
    
    metric = world["metric"]
    alpha, beta, h = _adm_lapse_shift(metric, constants)
    inv_sqrt_x, inv_sqrt_y, inv_sqrt_z, _ = _inv_sqrt_det_h_faces(metric)

    D = jnp.stack(D, axis=0)  # (3, Nx, Ny, Nz)
    B = jnp.stack(B, axis=0)  # (3, Nx, Ny, Nz)
    # stack D and B for easier manipulation

    alpha_x = _stagger_scalar_face(alpha, axis=0)
    alpha_y = _stagger_scalar_face(alpha, axis=1)
    alpha_z = _stagger_scalar_face(alpha, axis=2)
    beta_x = _stagger_vec_face(beta, axis=0)
    beta_y = _stagger_vec_face(beta, axis=1)
    beta_z = _stagger_vec_face(beta, axis=2)
    h_x = _stagger_mat_face(h, axis=0)
    h_y = _stagger_mat_face(h, axis=1)
    h_z = _stagger_mat_face(h, axis=2)

    cross_x = jnp.cross(beta_x, B, axis=0)
    cross_y = jnp.cross(beta_y, B, axis=0)
    cross_z = jnp.cross(beta_z, B, axis=0)

    Ex = alpha_x * jnp.einsum("j...,j...->...", h_x[0], D) + inv_sqrt_x * cross_x[0]
    Ey = alpha_y * jnp.einsum("j...,j...->...", h_y[1], D) + inv_sqrt_y * cross_y[1]
    Ez = alpha_z * jnp.einsum("j...,j...->...", h_z[2], D) + inv_sqrt_z * cross_z[2]
    return Ex, Ey, Ez


def GR_TimeAverageJ(J_new, J_old):
    """Entity-like current time-centering: J(n) = 0.5 * (J(n+1/2) + J(n-1/2))."""
    return tuple(0.5 * (J_new[i] + J_old[i]) for i in range(3))
