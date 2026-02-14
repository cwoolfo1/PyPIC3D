import jax.numpy as jnp
from jax import jit, lax
from functools import partial

from PyPIC3D.metric import METRIC_CYLINDRICAL
from PyPIC3D.utils import digital_filter


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


@jit
def recover_E_H_from_metric(D, B, world):
    """
    Recover E and H from D and B via metric constitutive relations:
      E = alpha * gamma^{-1} D + beta x B
      H = alpha * gamma^{-1} B - beta x D
    """
    metric = world["metric"]
    metric_type = metric["metric_type"]
    alpha = metric["lapse"]

    Dx, Dy, Dz = D
    Bx, By, Bz = B
    x_grid = world["grids"]["vertex"][0] if "grids" in world else (jnp.arange(Dx.shape[0], dtype=Dx.dtype) + 1.0)

    def constant_metric(_):
        ginv = metric["spatial_contra"]
        beta = metric["shift_contra"]

        D_stack = jnp.stack([Dx, Dy, Dz], axis=0)
        B_stack = jnp.stack([Bx, By, Bz], axis=0)
        gD = jnp.einsum("ij,jxyz->ixyz", ginv, D_stack)
        gB = jnp.einsum("ij,jxyz->ixyz", ginv, B_stack)

        beta_x_B = jnp.stack(
            [
                beta[1] * Bz - beta[2] * By,
                beta[2] * Bx - beta[0] * Bz,
                beta[0] * By - beta[1] * Bx,
            ],
            axis=0,
        )
        beta_x_D = jnp.stack(
            [
                beta[1] * Dz - beta[2] * Dy,
                beta[2] * Dx - beta[0] * Dz,
                beta[0] * Dy - beta[1] * Dx,
            ],
            axis=0,
        )

        E_stack = alpha * gD + beta_x_B
        H_stack = alpha * gB - beta_x_D
        return (E_stack[0], E_stack[1], E_stack[2]), (H_stack[0], H_stack[1], H_stack[2])

    def cylindrical_metric(_):
        reg = metric["regularization"]
        r = jnp.maximum(jnp.abs(x_grid), reg).reshape((-1, 1, 1))
        inv_r2 = 1.0 / (r * r)

        Ex = Dx
        Ey = inv_r2 * Dy
        Ez = Dz
        Hx = Bx
        Hy = inv_r2 * By
        Hz = Bz
        return (Ex, Ey, Ez), (Hx, Hy, Hz)

    return lax.cond(
        metric_type == METRIC_CYLINDRICAL,
        cylindrical_metric,
        constant_metric,
        operand=None,
    )


@partial(jit, static_argnames=("curl_func",))
def update_DB_and_recover_EH(D, B, J, world, constants, curl_func):
    """
    One GR electrodynamic step:
      dB/dt = -curl(E)
      dD/dt =  curl(H) - J
      then recover E,H from metric constitutive relations.
    """
    _ = curl_func
    dt = world["dt"]
    dx, dy, dz = world["dx"], world["dy"], world["dz"]
    alpha = constants["alpha"]

    E, H = recover_E_H_from_metric(D, B, world)

    curl_H = _curl_forward(H[0], H[1], H[2], dx, dy, dz)
    curl_E = _curl_backward(E[0], E[1], E[2], dx, dy, dz)

    D = (
        D[0] + dt * (curl_H[0] - J[0]),
        D[1] + dt * (curl_H[1] - J[1]),
        D[2] + dt * (curl_H[2] - J[2]),
    )
    B = (
        B[0] - dt * curl_E[0],
        B[1] - dt * curl_E[1],
        B[2] - dt * curl_E[2],
    )

    D = (digital_filter(D[0], alpha), digital_filter(D[1], alpha), digital_filter(D[2], alpha))
    B = (digital_filter(B[0], alpha), digital_filter(B[1], alpha), digital_filter(B[2], alpha))

    E, H = recover_E_H_from_metric(D, B, world)
    return E, B, D, H
