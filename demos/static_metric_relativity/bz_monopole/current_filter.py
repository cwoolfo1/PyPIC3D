"""Conservative binomial source smoothing for the axisymmetric BZ experiment.

Filter integrated charge and current face fluxes with commuting discrete
operators. Polar charge caps have half width: normalize their integrated
values before even reflection and restore their half weights afterwards.
The theta face flux instead has odd reflection. This preserves div(S J)=S div J
including the axes and radial tile seams, away from physical radial boundaries.
"""
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.ghost_cells import update_tiled_ghost_cells
from PyPIC3D.boundary_conditions.polar import divide, plane, refresh


def smooth_integrated(value, static, passes, *, theta_face=False):
    g, n = static.guard_cells, static.tile_shape[1]
    cap = jnp.ones_like(value)
    if not theta_face:
        cap = cap.at[plane(cap, g)].set(.5).at[plane(cap, g+n)].set(.5)
    for _ in range(passes):
        # Generic exchange handles radial seams, constant radial boundaries,
        # and periodic phi; it deliberately leaves polar theta unchanged.
        value = update_tiled_ghost_cells(value, static, g)
        regular = refresh(value/cap, g, n, theta_face, -1 if theta_face else 1)
        regular = (.5*regular + .25*(jnp.roll(regular, 1, axis=4)
                                    + jnp.roll(regular, -1, axis=4)))
        value = regular*cap
        value = .5*value + .25*(jnp.roll(value, 1, axis=3)+jnp.roll(value, -1, axis=3))
    return value


def filter_current(current, geometry, static, passes):
    if passes == 0:
        return current
    result = []
    for i, (value, area) in enumerate(zip(current, geometry.D_area)):
        flux = smooth_integrated(value*area, static, passes, theta_face=i == 1)
        value = divide(flux, area)
        if i == 2:
            g, n = static.guard_cells, static.tile_shape[1]
            value = value.at[plane(value, g)].set(0).at[plane(value, g+n)].set(0)
        result.append(value)
    return tuple(result)
