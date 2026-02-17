import jax
import jax.numpy as jnp

from PyPIC3D.boris import interpolate_field_to_particles
from PyPIC3D.utils import wrap_around


GR_EPS = 1e-30


def _sqrt_det_h_faces(metric):
    h = metric[1:, 1:, ...]
    h_t = h.transpose((2, 3, 4, 0, 1))
    det_h = jnp.linalg.det(h_t)
    sqrt_center = jnp.sqrt(jnp.maximum(det_h, GR_EPS))
    sqrt_x = 0.5 * (sqrt_center + jnp.roll(sqrt_center, shift=-1, axis=0))
    sqrt_y = 0.5 * (sqrt_center + jnp.roll(sqrt_center, shift=-1, axis=1))
    sqrt_z = 0.5 * (sqrt_center + jnp.roll(sqrt_center, shift=-1, axis=2))
    return sqrt_x, sqrt_y, sqrt_z, sqrt_center


def _adm_lapse_shift(metric, constants):
    h = metric[1:, 1:, ...]
    beta_cov = metric[0, 1:, ...]
    h_t = h.transpose((2, 3, 4, 0, 1))
    h_inv_t = jnp.linalg.inv(h_t)
    h_inv = h_inv_t.transpose((3, 4, 0, 1, 2))
    beta_contra = jnp.einsum("ij...,j...->i...", h_inv, beta_cov)
    beta_sq = jnp.einsum("i...,i...->...", beta_cov, beta_contra)
    alpha = jnp.sqrt(jnp.maximum(beta_sq - metric[0, 0, ...], GR_EPS)) / constants["C"]
    return alpha, beta_contra, h_inv


def _metric_transport_velocity(species, world, constants):
    x, y, z = species.get_forward_position()
    vx, vy, vz = species.get_velocity()
    shape_factor = species.get_shape()

    metric = world["metric"]
    alpha, _, h_inv = _adm_lapse_shift(metric, constants)
    grid = world["grids"]["center"]
    c = constants["C"]

    alpha_p = interpolate_field_to_particles(alpha, x, y, z, grid, shape_factor)
    h_inv_p = jnp.stack(
        [
            jnp.stack(
                [
                    interpolate_field_to_particles(h_inv[i, j], x, y, z, grid, shape_factor)
                    for j in range(3)
                ],
                axis=1,
            )
            for i in range(3)
        ],
        axis=1,
    )

    v_cov = jnp.stack([vx, vy, vz], axis=1)
    v_contra = jnp.einsum("nij,nj->ni", h_inv_p, v_cov)
    v2_over_c2 = jnp.einsum("ni,ni->n", v_cov, v_contra) / (c * c)
    v2_over_c2 = jnp.clip(v2_over_c2, 0.0, 1.0 - 1e-12)
    gamma = 1.0 / jnp.sqrt(1.0 - v2_over_c2)

    u_cov = gamma[:, None] * v_cov / c
    u_contra = jnp.einsum("nij,nj->ni", h_inv_p, u_cov)
    inv_energy = alpha_p / jnp.sqrt(jnp.maximum(1.0 + jnp.einsum("ni,ni->n", u_cov, u_contra), GR_EPS))
    vp = u_contra * inv_energy[:, None] * c
    return vp[:, 0], vp[:, 1], vp[:, 2]


def _shape_order_nonstaggered(order, i, di):
    if order == 1:
        i_min = i
        s0 = 1.0 - di
        s1 = di
        s = jnp.stack([s0, s1], axis=0)
        return i_min, s

    cond = di < 0.5
    i_min = jnp.where(cond, i - 1, i)

    s0_lo = 0.5 * (0.5 - di) ** 2
    s1_lo = 0.75 - di**2
    s2_lo = 1.0 - s0_lo - s1_lo

    s0_hi = 0.5 * (1.5 - di) ** 2
    s1_hi = 0.75 - (1.0 - di) ** 2
    s2_hi = 1.0 - s0_hi - s1_hi

    s0 = jnp.where(cond, s0_lo, s0_hi)
    s1 = jnp.where(cond, s1_lo, s1_hi)
    s2 = jnp.where(cond, s2_lo, s2_hi)
    s = jnp.stack([s0, s1, s2], axis=0)
    return i_min, s


def _for_deposit_1d(order, i_init, di_init, i_fin, di_fin):
    i_init_min, i_s = _shape_order_nonstaggered(order, i_init, di_init)
    i_fin_min, f_s = _shape_order_nonstaggered(order, i_fin, di_fin)

    zeros = jnp.zeros_like(i_s[0:1, :])
    i_s_lt = jnp.concatenate([i_s, zeros], axis=0)
    f_s_lt = jnp.concatenate([zeros, f_s], axis=0)
    i_s_gt = jnp.concatenate([zeros, i_s], axis=0)
    f_s_gt = jnp.concatenate([f_s, zeros], axis=0)
    i_s_eq = jnp.concatenate([i_s, zeros], axis=0)
    f_s_eq = jnp.concatenate([f_s, zeros], axis=0)

    cond_lt = i_init_min < i_fin_min
    cond_gt = i_init_min > i_fin_min
    cond_shift = cond_lt | cond_gt

    i_min = jnp.where(cond_lt, i_init_min, jnp.where(cond_gt, i_fin_min, i_init_min))
    i_max = jnp.where(cond_shift, i_min + order + 1, i_min + order)

    i_s_out = jnp.where(cond_lt[None, :], i_s_lt, jnp.where(cond_gt[None, :], i_s_gt, i_s_eq))
    f_s_out = jnp.where(cond_lt[None, :], f_s_lt, jnp.where(cond_gt[None, :], f_s_gt, f_s_eq))
    return i_min, i_max, i_s_out, f_s_out


def _compose_indices_2d(idx_a, idx_b, idx_c, a_axis, b_axis):
    idx_by_dim = [None, None, None]
    idx_by_dim[a_axis] = idx_a[:, None, :]
    idx_by_dim[b_axis] = idx_b[None, :, :]
    c_axis = 3 - a_axis - b_axis
    idx_by_dim[c_axis] = idx_c[None, None, :]
    return idx_by_dim[0], idx_by_dim[1], idx_by_dim[2]


def _compose_indices_1d(idx_a, fixed_other, a_axis):
    s, npart = idx_a.shape
    idx_by_dim = [None, None, None]
    idx_by_dim[a_axis] = idx_a
    for dim in range(3):
        if dim != a_axis:
            idx_by_dim[dim] = jnp.broadcast_to(fixed_other[dim][None, :], (s, npart))
    return idx_by_dim[0], idx_by_dim[1], idx_by_dim[2]


def GR_esirkepov_metric_current(particles, J, constants, world, apply_metric_scaling=True):
    Jx, Jy, Jz = J
    Nx, Ny, Nz = Jx.shape
    dx, dy, dz, dt = world["dx"], world["dy"], world["dz"], world["dt"]
    grid = world["grids"]["center"]
    xmin, ymin, zmin = grid[0][0], grid[1][0], grid[2][0]

    Jx = Jx.at[:, :, :].set(0)
    Jy = Jy.at[:, :, :].set(0)
    Jz = Jz.at[:, :, :].set(0)

    dxyz = [dx, dy, dz]
    x_active = Nx != 1
    y_active = Ny != 1
    z_active = Nz != 1
    active_dims = [x_active, y_active, z_active]

    for species in particles:
        q = species.get_charge()
        shape_factor = int(species.get_shape())
        order = 1 if shape_factor == 1 else 2
        s = order + 2

        x, y, z = species.get_forward_position()
        old_x, old_y, old_z = species.get_previous_forward_position()
        vx, vy, vz = _metric_transport_velocity(species, world, constants)

        xn = (x - xmin) / dx
        yn = (y - ymin) / dy
        zn = (z - zmin) / dz
        xo = (old_x - xmin) / dx
        yo = (old_y - ymin) / dy
        zo = (old_z - zmin) / dz

        i1 = jnp.floor(xn).astype(int)
        i2 = jnp.floor(yn).astype(int)
        i3 = jnp.floor(zn).astype(int)
        i1_prev = jnp.floor(xo).astype(int)
        i2_prev = jnp.floor(yo).astype(int)
        i3_prev = jnp.floor(zo).astype(int)

        dx1 = xn - i1
        dx2 = yn - i2
        dx3 = zn - i3
        dx1_prev = xo - i1_prev
        dx2_prev = yo - i2_prev
        dx3_prev = zo - i3_prev

        i1_min, i1_max, iSx, fSx = _for_deposit_1d(order, i1_prev, dx1_prev, i1, dx1)
        i2_min, i2_max, iSy, fSy = _for_deposit_1d(order, i2_prev, dx2_prev, i2, dx2)
        i3_min, i3_max, iSz, fSz = _for_deposit_1d(order, i3_prev, dx3_prev, i3, dx3)

        di_x1 = i1_max - i1_min
        di_x2 = i2_max - i2_min
        di_x3 = i3_max - i3_min

        rng = jnp.arange(s, dtype=i1.dtype)[:, None]
        x_idx = wrap_around(i1_min[None, :] + rng, Nx)
        y_idx = wrap_around(i2_min[None, :] + rng, Ny)
        z_idx = wrap_around(i3_min[None, :] + rng, Nz)

        q_over_dt = q / dt

        if x_active and y_active and z_active:
            iSx4 = iSx[:, None, None, :]
            iSy4 = iSy[None, :, None, :]
            iSz4 = iSz[None, None, :, :]
            fSx4 = fSx[:, None, None, :]
            fSy4 = fSy[None, :, None, :]
            fSz4 = fSz[None, None, :, :]

            Wx = (1.0 / 3.0) * (fSx4 - iSx4) * (
                iSy4 * iSz4 + fSy4 * fSz4 + 0.5 * (iSz4 * fSy4 + iSy4 * fSz4)
            )
            Wy = (1.0 / 3.0) * (fSy4 - iSy4) * (
                iSx4 * iSz4 + fSx4 * fSz4 + 0.5 * (iSz4 * fSx4 + iSx4 * fSz4)
            )
            Wz = (1.0 / 3.0) * (fSz4 - iSz4) * (
                iSx4 * iSy4 + fSx4 * fSy4 + 0.5 * (iSx4 * fSy4 + iSy4 * fSx4)
            )

            jx1 = jnp.cumsum((-q_over_dt) * Wx, axis=0)
            jx2 = jnp.cumsum((-q_over_dt) * Wy, axis=1)
            jx3 = jnp.cumsum((-q_over_dt) * Wz, axis=2)

            I = jnp.arange(s)[:, None, None, None]
            Jj = jnp.arange(s)[None, :, None, None]
            K = jnp.arange(s)[None, None, :, None]

            m1 = (I < di_x1[None, None, None, :]) & (Jj <= di_x2[None, None, None, :]) & (K <= di_x3[None, None, None, :])
            m2 = (I <= di_x1[None, None, None, :]) & (Jj < di_x2[None, None, None, :]) & (K <= di_x3[None, None, None, :])
            m3 = (I <= di_x1[None, None, None, :]) & (Jj <= di_x2[None, None, None, :]) & (K < di_x3[None, None, None, :])

            ix = x_idx[:, None, None, :]
            iy = y_idx[None, :, None, :]
            iz = z_idx[None, None, :, :]

            Jx = Jx.at[(ix, iy, iz)].add(jnp.where(m1, jx1, 0.0), mode="drop")
            Jy = Jy.at[(ix, iy, iz)].add(jnp.where(m2, jx2, 0.0), mode="drop")
            Jz = Jz.at[(ix, iy, iz)].add(jnp.where(m3, jx3, 0.0), mode="drop")
        else:
            active_idx = [i for i, is_active in enumerate(active_dims) if is_active]
            if len(active_idx) == 2:
                a, b = active_idx[0], active_idx[1]
                c = 3 - a - b

                iS = [iSx, iSy, iSz]
                fS = [fSx, fSy, fSz]
                i_min = [i1_min, i2_min, i3_min]
                i_max = [i1_max, i2_max, i3_max]
                idx = [x_idx, y_idx, z_idx]
                vel = [vx, vy, vz]

                di_a = i_max[a] - i_min[a]
                di_b = i_max[b] - i_min[b]

                iSa = iS[a][:, None, :]
                iSb = iS[b][None, :, :]
                fSa = fS[a][:, None, :]
                fSb = fS[b][None, :, :]

                Wa = 0.5 * (fSa - iSa) * (fSb + iSb)
                Wb = 0.5 * (fSa + iSa) * (fSb - iSb)
                Wc = (1.0 / 3.0) * (fSb * (0.5 * iSa + fSa) + iSb * (0.5 * fSa + iSa))

                ja = jnp.cumsum((-q_over_dt) * Wa, axis=0)
                jb = jnp.cumsum((-q_over_dt) * Wb, axis=1)
                jc = (q * vel[c])[None, None, :] * Wc

                Ia = jnp.arange(s)[:, None, None]
                Jb = jnp.arange(s)[None, :, None]
                ma = (Ia < di_a[None, None, :]) & (Jb <= di_b[None, None, :])
                mb = (Ia <= di_a[None, None, :]) & (Jb < di_b[None, None, :])
                mc = (Ia <= di_a[None, None, :]) & (Jb <= di_b[None, None, :])

                ix, iy, iz = _compose_indices_2d(idx[a], idx[b], idx[c][0, :], a, b)

                comp_vals = [None, None, None]
                comp_vals[a] = jnp.where(ma, ja, 0.0)
                comp_vals[b] = jnp.where(mb, jb, 0.0)
                comp_vals[c] = jnp.where(mc, jc, 0.0)

                Jx = Jx.at[(ix, iy, iz)].add(comp_vals[0], mode="drop")
                Jy = Jy.at[(ix, iy, iz)].add(comp_vals[1], mode="drop")
                Jz = Jz.at[(ix, iy, iz)].add(comp_vals[2], mode="drop")
            elif len(active_idx) == 1:
                a = active_idx[0]
                iS = [iSx, iSy, iSz]
                fS = [fSx, fSy, fSz]
                i_min = [i1_min, i2_min, i3_min]
                i_max = [i1_max, i2_max, i3_max]
                idx = [x_idx, y_idx, z_idx]
                vel = [vx, vy, vz]

                di_a = i_max[a] - i_min[a]
                iSa = iS[a]
                fSa = fS[a]
                W1 = fSa - iSa
                W23 = 0.5 * (fSa + iSa)

                j_active = jnp.cumsum((-q_over_dt) * W1, axis=0)

                pref = [q * vel[0], q * vel[1], q * vel[2]]
                comp_vals = [pref[0][None, :] * W23, pref[1][None, :] * W23, pref[2][None, :] * W23]
                comp_vals[a] = j_active

                I = jnp.arange(s)[:, None]
                m_active = I < di_a[None, :]
                m_other = I <= di_a[None, :]
                for comp in range(3):
                    comp_vals[comp] = jnp.where(m_active if comp == a else m_other, comp_vals[comp], 0.0)

                fixed_other = [idx[0][0, :], idx[1][0, :], idx[2][0, :]]
                ix, iy, iz = _compose_indices_1d(idx[a], fixed_other, a)
                Jx = Jx.at[(ix, iy, iz)].add(comp_vals[0], mode="drop")
                Jy = Jy.at[(ix, iy, iz)].add(comp_vals[1], mode="drop")
                Jz = Jz.at[(ix, iy, iz)].add(comp_vals[2], mode="drop")
            else:
                raise ValueError("At least one active dimension is required for current deposition.")

    J_base = (Jx, Jy, Jz)

    metric = world.get("metric", None)
    if metric is None or (not apply_metric_scaling):
        return J_base

    sqrt_x, sqrt_y, sqrt_z, _ = _sqrt_det_h_faces(metric)
    return (
        J_base[0] * sqrt_x,
        J_base[1] * sqrt_y,
        J_base[2] * sqrt_z,
    )


def GR_deposit_current(particles, J, constants, world):
    return GR_esirkepov_metric_current(particles, J, constants, world)
