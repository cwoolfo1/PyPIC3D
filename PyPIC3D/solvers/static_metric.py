import jax.numpy as jnp

from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.boundary_conditions.supergaussian import apply_tiled_supergaussian_absorber
from PyPIC3D.relativity.core import B_FIELD_LOCATIONS, D_FIELD_LOCATIONS


def _location_interpolate_axis(field, source_location, target_location, axis):
    array_axis = axis + 3
    if source_location[axis] == target_location[axis]:
        return field
    if source_location[axis] == "C":
        return 0.5 * (field + jnp.roll(field, -1, axis=array_axis))
    return 0.5 * (field + jnp.roll(field, 1, axis=array_axis))


def _location_interpolate(field, source_location, target_location):
    interpolated = field
    for axis in range(3):
        interpolated = _location_interpolate_axis(interpolated, source_location, target_location, axis)
    return interpolated


def _metric_weighted_interpolate(field, source_metric, target_metric, source_location, target_location):
    weighted = source_metric.sqrt_gamma * field
    weighted = _location_interpolate(weighted, source_location, target_location)
    return weighted / target_metric.sqrt_gamma


def _shift_cross_component(beta, vector_components, component):
    beta_x = beta[..., 0]
    beta_y = beta[..., 1]
    beta_z = beta[..., 2]
    vector_x, vector_y, vector_z = vector_components

    if component == 0:
        return beta_y * vector_z - beta_z * vector_y
    if component == 1:
        return beta_z * vector_x - beta_x * vector_z
    return beta_x * vector_y - beta_y * vector_x


def compute_covariant_E(D_tiles, B_tiles, metric):
    """
    Compute covariant E_i on the D component locations using FPIC Eq. (10).
    """

    E_cov = []
    for i, target_location in enumerate(D_FIELD_LOCATIONS):
        D_on_target = []
        B_on_target = []
        for j, source_location in enumerate(D_FIELD_LOCATIONS):
            D_on_target.append(
                _metric_weighted_interpolate(
                    D_tiles[j],
                    metric.D[j],
                    metric.D[i],
                    source_location,
                    target_location,
                )
            )
        for j, source_location in enumerate(B_FIELD_LOCATIONS):
            B_on_target.append(
                _metric_weighted_interpolate(
                    B_tiles[j],
                    metric.B[j],
                    metric.D[i],
                    source_location,
                    target_location,
                )
            )

        D_lower_i = 0.0
        for j in range(3):
            D_lower_i = D_lower_i + metric.D[i].gamma[..., i, j] * D_on_target[j]

        shift_cross = _shift_cross_component(metric.D[i].shift, tuple(B_on_target), i)
        E_cov.append(
            metric.D[i].lapse * D_lower_i
            + metric.D[i].sqrt_gamma * shift_cross
        )
    return tuple(E_cov)


def compute_covariant_H(D_tiles, B_tiles, metric):
    """
    Compute covariant H_i on the B component locations using FPIC Eq. (9).
    """

    H_cov = []
    for i, target_location in enumerate(B_FIELD_LOCATIONS):
        B_on_target = []
        D_on_target = []
        for j, source_location in enumerate(B_FIELD_LOCATIONS):
            B_on_target.append(
                _metric_weighted_interpolate(
                    B_tiles[j],
                    metric.B[j],
                    metric.B[i],
                    source_location,
                    target_location,
                )
            )
        for j, source_location in enumerate(D_FIELD_LOCATIONS):
            D_on_target.append(
                _metric_weighted_interpolate(
                    D_tiles[j],
                    metric.D[j],
                    metric.B[i],
                    source_location,
                    target_location,
                )
            )

        B_lower_i = 0.0
        for j in range(3):
            B_lower_i = B_lower_i + metric.B[i].gamma[..., i, j] * B_on_target[j]

        shift_cross = _shift_cross_component(metric.B[i].shift, tuple(D_on_target), i)
        H_cov.append(
            metric.B[i].lapse * B_lower_i
            - metric.B[i].sqrt_gamma * shift_cross
        )
    return tuple(H_cov)


def update_D_relativity(D_tiles, H_tiles, J_tiles, metric, static_parameters, dynamic_parameters, dt):
    """
    Update contravariant displacement field D^i in a fixed 3+1 metric.
    """

    Dx, Dy, Dz = D_tiles
    Jx, Jy, Jz = J_tiles
    Hx, Hy, Hz = H_tiles

    g = int(static_parameters.guard_cells)
    active = slice(g, -g)
    backward = slice(g - 1, -g - 1)
    dx, dy, dz = dynamic_parameters.dx, dynamic_parameters.dy, dynamic_parameters.dz

    dHz_dy = (Hz[:, :, :, active, active, active] - Hz[:, :, :, active, backward, active]) / dy
    dHy_dz = (Hy[:, :, :, active, active, active] - Hy[:, :, :, active, active, backward]) / dz
    dHx_dz = (Hx[:, :, :, active, active, active] - Hx[:, :, :, active, active, backward]) / dz
    dHz_dx = (Hz[:, :, :, active, active, active] - Hz[:, :, :, backward, active, active]) / dx
    dHy_dx = (Hy[:, :, :, active, active, active] - Hy[:, :, :, backward, active, active]) / dx
    dHx_dy = (Hx[:, :, :, active, active, active] - Hx[:, :, :, active, backward, active]) / dy

    sqrt_Dx = metric.D[0].sqrt_gamma[:, :, :, active, active, active]
    sqrt_Dy = metric.D[1].sqrt_gamma[:, :, :, active, active, active]
    sqrt_Dz = metric.D[2].sqrt_gamma[:, :, :, active, active, active]
    current = slice(g, -g)

    Dx = Dx.at[:, :, :, active, active, active].set(
        Dx[:, :, :, active, active, active]
        + dt * ((dHz_dy - dHy_dz) / sqrt_Dx - 4.0 * jnp.pi * Jx[:, :, :, current, current, current])
    )
    Dy = Dy.at[:, :, :, active, active, active].set(
        Dy[:, :, :, active, active, active]
        + dt * ((dHx_dz - dHz_dx) / sqrt_Dy - 4.0 * jnp.pi * Jy[:, :, :, current, current, current])
    )
    Dz = Dz.at[:, :, :, active, active, active].set(
        Dz[:, :, :, active, active, active]
        + dt * ((dHy_dx - dHx_dy) / sqrt_Dz - 4.0 * jnp.pi * Jz[:, :, :, current, current, current])
    )

    D_tiles = (Dx, Dy, Dz)
    if static_parameters.supergaussian_active:
        return apply_tiled_supergaussian_absorber(
            D_tiles,
            static_parameters,
            dynamic_parameters,
            dt,
        )

    return ghost_cells.update_tiled_vector_ghost_cells(D_tiles, static_parameters, g)


def update_B_relativity(E_tiles, B_tiles, metric, static_parameters, dynamic_parameters, dt):
    """
    Update contravariant magnetic field B^i in a fixed 3+1 metric.
    """

    Bx, By, Bz = B_tiles
    Ex, Ey, Ez = E_tiles

    g = int(static_parameters.guard_cells)
    active = slice(g, -g)
    forward = slice(g + 1, None if g == 1 else -g + 1)
    dx, dy, dz = dynamic_parameters.dx, dynamic_parameters.dy, dynamic_parameters.dz

    dEz_dy = (Ez[:, :, :, active, forward, active] - Ez[:, :, :, active, active, active]) / dy
    dEy_dz = (Ey[:, :, :, active, active, forward] - Ey[:, :, :, active, active, active]) / dz
    dEx_dz = (Ex[:, :, :, active, active, forward] - Ex[:, :, :, active, active, active]) / dz
    dEz_dx = (Ez[:, :, :, forward, active, active] - Ez[:, :, :, active, active, active]) / dx
    dEy_dx = (Ey[:, :, :, forward, active, active] - Ey[:, :, :, active, active, active]) / dx
    dEx_dy = (Ex[:, :, :, active, forward, active] - Ex[:, :, :, active, active, active]) / dy

    sqrt_Bx = metric.B[0].sqrt_gamma[:, :, :, active, active, active]
    sqrt_By = metric.B[1].sqrt_gamma[:, :, :, active, active, active]
    sqrt_Bz = metric.B[2].sqrt_gamma[:, :, :, active, active, active]

    Bx = Bx.at[:, :, :, active, active, active].set(
        Bx[:, :, :, active, active, active] - dt * (dEz_dy - dEy_dz) / sqrt_Bx
    )
    By = By.at[:, :, :, active, active, active].set(
        By[:, :, :, active, active, active] - dt * (dEx_dz - dEz_dx) / sqrt_By
    )
    Bz = Bz.at[:, :, :, active, active, active].set(
        Bz[:, :, :, active, active, active] - dt * (dEy_dx - dEx_dy) / sqrt_Bz
    )

    B_tiles = (Bx, By, Bz)
    if static_parameters.supergaussian_active:
        return apply_tiled_supergaussian_absorber(
            B_tiles,
            static_parameters,
            dynamic_parameters,
            dt,
        )

    return ghost_cells.update_tiled_vector_ghost_cells(B_tiles, static_parameters, g)
