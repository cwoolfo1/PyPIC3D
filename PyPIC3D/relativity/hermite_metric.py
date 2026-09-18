"""C1 tensor-product Hermite interpolation of supplied nodal data.

Each interval shares centered-difference endpoint slopes with its neighbors.
This is the cardinal cubic Hermite (Catmull-Rom) reconstruction, not a PIC
B-spline gather. It exactly reproduces constants, linear and quadratic data.
Three guard cells cover midpoint sampling before one-cell tile migration. No slope limiting, axis floor,
analytic metric evaluation, or positive-definiteness repair is performed.
"""

import jax.numpy as jnp


def weights(t):
    return jnp.stack(
        (
            -0.5 * t + t * t - 0.5 * t**3,
            1 - 2.5 * t * t + 1.5 * t**3,
            0.5 * t + 2 * t * t - 1.5 * t**3,
            -0.5 * t * t + 0.5 * t**3,
        ),
        axis=0,
    )


def interpolate_hermite(field, position, grid, active_axes, inactive_axis_indices):
    if len(grid) != 3 or position.shape[-1] != 3:
        raise ValueError("Hermite sampling requires three coordinates and three grids")
    for k in range(3):
        if field.shape[k] != len(grid[k]):
            raise ValueError("Metric array and coordinate grid shapes differ")
        if active_axes[k] and len(grid[k]) < 4:
            raise ValueError(
                "Hermite sampling requires four nodes in each resolved direction"
            )
        if not active_axes[k] and not 0 <= inactive_axis_indices[k] < len(grid[k]):
            raise ValueError("Invalid unresolved-axis sample index")
    shape = position.shape[:-1]
    p = position.reshape((-1, 3))
    inds = []
    ws = []
    valid = jnp.ones(p.shape[0], bool)
    for k in range(3):
        if active_axes[k]:
            h = grid[k][1] - grid[k][0]
            i = jnp.floor((p[:, k] - grid[k][0]) / h).astype(jnp.int32)
            valid = valid & (i >= 1) & (i + 2 < len(grid[k]))
            # Clipping makes the gather defined; out-of-stencil evaluations
            # remain explicit NaNs and are rejected by diagnostic gates.
            safe_i = jnp.clip(i, 1, len(grid[k]) - 3)
            t = (p[:, k] - grid[k][safe_i]) / h
            inds.append(safe_i[None, :] + jnp.arange(-1, 3)[:, None])
            ws.append(weights(t))
        else:
            inds.append(
                jnp.full((1, p.shape[0]), inactive_axis_indices[k], dtype=jnp.int32)
            )
            ws.append(jnp.ones((1, p.shape[0]), dtype=p.dtype))
    values = field[
        inds[0][:, None, None, :], inds[1][None, :, None, :], inds[2][None, None, :, :]
    ]
    result = jnp.einsum("ip,jp,kp,ijkp...->p...", *ws, values)
    result = jnp.where(valid.reshape((-1,) + (1,) * (result.ndim - 1)), result, jnp.nan)
    return result.reshape(shape + field.shape[3:])
