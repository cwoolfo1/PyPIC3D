import math

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from PyPIC3D.boundary_conditions.grid_and_stencil import BC_ABSORBING, BC_CONDUCTING, BC_CONSTANT, BC_PERIODIC, BC_POLAR


MESH_AXES = ("tile_x", "tile_y", "tile_z")
SCALAR_TILE_SPEC = P("tile_x", "tile_y", "tile_z", None, None, None)
VECTOR_TILE_SPEC = P(None, "tile_x", "tile_y", "tile_z", None, None, None)
BC_TYPE_FIELD = 0
BC_TYPE_PARTICLE = 1
_PARTICLE_SCALAR_REFLECTING_PARITY = (1, 1, 1)


def particle_vector_reflecting_parity(component):
    """Return specular-reflection parity for a deposited Cartesian component.

    The normal component is odd and the two tangential components are even.
    The returned tuple is ordered by wall normal as ``(x, y, z)``.
    """

    return tuple(-1 if axis == int(component) else 1 for axis in range(3))


def _particle_reflecting_parity(bc_type, reflecting_parity, vector):
    """Default particle-deposit parity: even scalars, odd normal vector components."""

    if int(bc_type) == BC_TYPE_FIELD:
        if reflecting_parity is not None:
            raise ValueError("reflecting_parity is only valid for particle boundary conditions.")
        return None
    if reflecting_parity is not None:
        return reflecting_parity
    if vector:
        return tuple(particle_vector_reflecting_parity(component) for component in range(3))
    return _PARTICLE_SCALAR_REFLECTING_PARITY


def _as_python_int(value):
    return int(jax.device_get(value))


def _boundary_tuple(boundary_conditions):
    if isinstance(boundary_conditions, tuple):
        return tuple(_as_python_int(value) for value in boundary_conditions)
    return (
        _as_python_int(boundary_conditions["x"]),
        _as_python_int(boundary_conditions["y"]),
        _as_python_int(boundary_conditions["z"]),
    )


def _boundary_conditions_for_type(static_parameters, bc_type):
    bc_type = int(bc_type)
    if bc_type == BC_TYPE_FIELD:
        return _boundary_tuple(static_parameters.boundary_conditions)
    if bc_type == BC_TYPE_PARTICLE:
        return _boundary_tuple(static_parameters.particle_boundary_conditions)
    raise ValueError("bc_type must be 0 for field boundaries or 1 for particle boundaries.")


def _reduced_axes_from_tile_shape(tile_shape, mesh_shape):
    tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]
    ntx, nty, ntz = [int(width) for width in mesh_shape]
    return (
        tile_nx == 1 and ntx == 1,
        tile_ny == 1 and nty == 1,
        tile_nz == 1 and ntz == 1,
    )


def _is_stacked_tiled_vector_field(field_tiles):
    return hasattr(field_tiles, "ndim") and field_tiles.ndim == 7


def _stack_tiled_vector_field(field_tiles):
    if _is_stacked_tiled_vector_field(field_tiles):
        return field_tiles
    return jnp.stack(field_tiles, axis=0)


def _unstack_tiled_vector_field(field_tiles):
    return tuple(field_tiles[i] for i in range(int(field_tiles.shape[0])))


def _restore_tiled_vector_layout(stacked_tiles, original_tiles):
    if _is_stacked_tiled_vector_field(original_tiles):
        return stacked_tiles
    return _unstack_tiled_vector_field(stacked_tiles)


def _send_positive_permutation(axis_size, boundary_condition):
    axis_size = int(axis_size)
    if boundary_condition == BC_PERIODIC:
        return tuple((i, (i + 1) % axis_size) for i in range(axis_size))
    return tuple((i, i + 1) for i in range(axis_size - 1))


def _send_negative_permutation(axis_size, boundary_condition):
    axis_size = int(axis_size)
    if boundary_condition == BC_PERIODIC:
        return tuple((i, (i - 1) % axis_size) for i in range(axis_size))
    return tuple((i, i - 1) for i in range(1, axis_size))


def _axis_permutations(mesh_shape, boundary_conditions):
    send_positive = tuple(
        _send_positive_permutation(axis_size, bc)
        for axis_size, bc in zip(mesh_shape, boundary_conditions)
    )
    send_negative = tuple(
        _send_negative_permutation(axis_size, bc)
        for axis_size, bc in zip(mesh_shape, boundary_conditions)
    )
    return send_positive, send_negative


def _default_mesh_for_tile_shape(tile_grid_shape):
    tile_grid_shape = tuple(int(width) for width in tile_grid_shape)
    n_devices = math.prod(tile_grid_shape)
    devices = jax.devices()
    if len(devices) < n_devices:
        raise ValueError(
            "Tiled field communication requires one logical tile per device: "
            f"tile topology {tile_grid_shape} needs {n_devices} devices, "
            f"but JAX exposes {len(devices)}."
        )
    return jax.make_mesh(
        tile_grid_shape,
        MESH_AXES,
        devices=devices[:n_devices],
        axis_types=(jax.sharding.AxisType.Auto,) * len(MESH_AXES),
    )


def make_field_mesh(tile_grid_shape):
    """
    Build the JAX device mesh for one logical field tile per device.
    """

    return _default_mesh_for_tile_shape(tile_grid_shape)


def _validate_scalar_tile_topology(field_tiles, mesh):
    tile_grid_shape = tuple(int(width) for width in field_tiles.shape[:3])
    mesh_shape = tuple(int(width) for width in mesh.devices.shape)
    if tile_grid_shape != mesh_shape:
        raise ValueError(
            "Tiled field communication requires one logical tile per device: "
            f"field tile topology {tile_grid_shape} does not match device mesh {mesh_shape}."
        )


def _validate_vector_tile_topology(field_tiles, mesh):
    if _is_stacked_tiled_vector_field(field_tiles):
        tile_grid_shape = tuple(int(width) for width in field_tiles.shape[1:4])
    else:
        tile_grid_shape = tuple(int(width) for width in field_tiles[0].shape[:3])
    mesh_shape = tuple(int(width) for width in mesh.devices.shape)
    if tile_grid_shape != mesh_shape:
        raise ValueError(
            "Tiled vector communication requires one logical tile per device: "
            f"field tile topology {tile_grid_shape} does not match device mesh {mesh_shape}."
        )


def _local_refresh_reduced_axis(
    tile,
    axis,
    g,
    boundary_condition,
    reflecting_parity=None,
):
    interior_slice = [slice(None), slice(None), slice(None)]
    interior_slice[axis] = slice(g, g + 1)
    interior = tile[tuple(interior_slice)]

    lower_slice = [slice(None), slice(None), slice(None)]
    lower_slice[axis] = slice(0, g)
    upper_slice = [slice(None), slice(None), slice(None)]
    upper_slice[axis] = slice(-g, None)

    if boundary_condition == BC_PERIODIC:
        tile = tile.at[tuple(lower_slice)].set(jnp.broadcast_to(interior, tile[tuple(lower_slice)].shape))
        tile = tile.at[tuple(upper_slice)].set(jnp.broadcast_to(interior, tile[tuple(upper_slice)].shape))
    elif boundary_condition == BC_CONSTANT:
        tile = tile.at[tuple(lower_slice)].set(jnp.broadcast_to(interior, tile[tuple(lower_slice)].shape))
        tile = tile.at[tuple(upper_slice)].set(jnp.broadcast_to(interior, tile[tuple(upper_slice)].shape))
    elif boundary_condition == BC_CONDUCTING and reflecting_parity is not None:
        reflected = reflecting_parity * interior
        tile = tile.at[tuple(lower_slice)].set(
            jnp.broadcast_to(reflected, tile[tuple(lower_slice)].shape)
        )
        tile = tile.at[tuple(upper_slice)].set(
            jnp.broadcast_to(reflected, tile[tuple(upper_slice)].shape)
        )
    else:
        tile = tile.at[tuple(lower_slice)].set(0.0)
        tile = tile.at[tuple(upper_slice)].set(0.0)

    return tile


def _axis_slices(axis, g):
    lower_ghost = [slice(None), slice(None), slice(None)]
    upper_ghost = [slice(None), slice(None), slice(None)]
    lower_interior = [slice(None), slice(None), slice(None)]
    upper_interior = [slice(None), slice(None), slice(None)]

    lower_ghost[axis] = slice(0, g)
    upper_ghost[axis] = slice(-g, None)
    lower_interior[axis] = slice(g, 2 * g)
    upper_interior[axis] = slice(-2 * g, -g)

    return (
        tuple(lower_ghost),
        tuple(upper_ghost),
        tuple(lower_interior),
        tuple(upper_interior),
    )


def _refresh_axis(tile, axis, g, axis_name, send_positive, send_negative):
    lower_ghost, upper_ghost, lower_interior, upper_interior = _axis_slices(axis, g)

    # Source i sends its upper interior to destination i + 1, filling the
    # receiver's lower ghost.  With a one-device periodic axis the permutation
    # is ((0, 0),), so this is the same self-exchange used on larger meshes.
    lower_values = jax.lax.ppermute(tile[upper_interior], axis_name, send_positive)
    # Source i sends its lower interior to destination i - 1, filling the
    # receiver's upper ghost.  Empty nonperiodic permutations naturally produce
    # zeros where no neighbor exists.
    upper_values = jax.lax.ppermute(tile[lower_interior], axis_name, send_negative)

    tile = tile.at[lower_ghost].set(lower_values)
    tile = tile.at[upper_ghost].set(upper_values)

    return tile


def _apply_local_reflecting_boundary_axis(
    tile,
    axis,
    g,
    parity,
    axis_name,
    axis_size,
):
    """Mirror owner interiors into halos on true global reflecting walls."""

    lower_ghost, upper_ghost, lower_interior, upper_interior = _axis_slices(axis, g)
    tile_index = jax.lax.axis_index(axis_name)

    tile = jax.lax.cond(
        tile_index == 0,
        lambda local_tile: local_tile.at[lower_ghost].set(
            parity * jnp.flip(local_tile[lower_interior], axis=axis)
        ),
        lambda local_tile: local_tile,
        tile,
    )
    tile = jax.lax.cond(
        tile_index == axis_size - 1,
        lambda local_tile: local_tile.at[upper_ghost].set(
            parity * jnp.flip(local_tile[upper_interior], axis=axis)
        ),
        lambda local_tile: local_tile,
        tile,
    )
    return tile


def staggered_mirror(node, width, location):
    """Return the owner and ordered wall encounters for a C/V image node."""
    vertex = location == 'V'
    last = width-1 if vertex else width
    walls = []
    while node < 0 or node > last:
        wall = 0 if node < 0 else width
        walls.append(wall)
        node = 2*wall-node-int(vertex)
    return node, tuple(walls)


def _staggered_conducting_axis(tile, axis, g, axis_name, axis_size,
                                positive, negative, location, parity, *, fold):
    """Reflect about actual endpoint nodes, preserving the owned upper C plane.

    Folding creates the even/odd image source. An even source on a wall node
    receives its coincident image, so its point density is doubled; integrating
    nodal densities uses endpoint trapezoid weights. No metric volumes enter.
    """
    n = tile.shape[axis] - 2*g
    staggered = location == 'V'
    upper = g+n-1 if staggered else g+n
    index = jax.lax.axis_index(axis_name)
    parity = 1 if parity is None else parity
    def plane(i):
        result = [slice(None)] * 3
        result[axis] = i
        return tuple(result)
    def wall(a, low):
        outside = range(g) if low else range(upper+1, a.shape[axis])
        original = a
        for i in outside:
            j = i-g
            owner = (-j-1 if staggered else -j) if low else (2*n-1-j if staggered else 2*n-j)
            sign = parity
            # A narrow single-tile direction can fit fewer cells than halos.
            # Continue reflecting until the owner is inside the domain.
            if axis_size == 1:
                owner, walls = staggered_mirror(owner, n, location)
                sign *= parity**len(walls)
            if fold:
                a = a.at[plane(g+owner)].add(sign*original[plane(i)])
                a = a.at[plane(i)].set(0.)
            else:
                a = a.at[plane(i)].set(sign*original[plane(g+owner)])
        return a
    if fold:
        tile = jax.lax.cond(index == 0, lambda a: wall(a, True), lambda a: a, tile)
        tile = jax.lax.cond(index == axis_size-1, lambda a: wall(a, False), lambda a: a, tile)
        if not staggered:
            tile = jax.lax.cond(index == 0,
                lambda a: a.at[plane(g)].multiply(1+parity), lambda a: a, tile)
            tile = jax.lax.cond(index == axis_size-1,
                lambda a: a.at[plane(g+n)].multiply(1+parity), lambda a: a, tile)
        endpoint = tile[plane(g+n)]
        # Reflection has already folded the exterior sources. Exchange only
        # inter-tile deposits, using the ordinary nonperiodic communication.
        tile = _fold_axis(tile, axis, g, axis_name, axis_size, BC_ABSORBING, positive, negative)
    else:
        endpoint = tile[plane(g+n)]
        tile = _refresh_axis(tile, axis, g, axis_name, positive, negative)
    if not staggered:
        tile = jax.lax.cond(index == axis_size-1,
                            lambda a: a.at[plane(g+n)].set(endpoint), lambda a: a, tile)
    if not fold:
        tile = jax.lax.cond(index == 0, lambda a: wall(a, True), lambda a: a, tile)
        tile = jax.lax.cond(index == axis_size-1, lambda a: wall(a, False), lambda a: a, tile)
    return tile


def _local_refresh_scalar_tile(
    tile,
    g,
    boundary_conditions,
    reduced_axes,
    mesh_shape,
    send_positive,
    send_negative,
    reflecting_parity=None,
    location=None,
    preserve_exterior=False,
):
    axis_parities = (None, None, None) if reflecting_parity is None else reflecting_parity
    for axis, axis_name, boundary_condition, reduced_axis, positive, negative, parity in zip(
        range(3),
        MESH_AXES,
        boundary_conditions,
        reduced_axes,
        send_positive,
        send_negative,
        axis_parities,
    ):
        if boundary_condition == BC_POLAR:
            continue  # polar theta is handled with explicit C/V ownership
        if boundary_condition == BC_CONDUCTING and location is not None and parity is not None:
            tile = _staggered_conducting_axis(
                tile, axis, g, axis_name, mesh_shape[axis], positive, negative,
                location[axis], parity, fold=False)
            continue
        keep_exterior = (preserve_exterior[axis] if isinstance(preserve_exterior, tuple)
                         else preserve_exterior)
        if (keep_exterior and boundary_condition != BC_PERIODIC or
                boundary_condition == BC_CONDUCTING and location is not None):
            # C endpoints are owned by the final tile. Restore them before
            # transverse exchange, along with computed exterior values when
            # requested by the metric constitutive/projector operators.
            lower, upper, _, _ = _axis_slices(axis, g)
            before = tile
            tile = _refresh_axis(tile, axis, g, axis_name, positive, negative)
            index = jax.lax.axis_index(axis_name)
            if keep_exterior:
                tile = jax.lax.cond(index == 0,
                    lambda a: a.at[lower].set(before[lower]), lambda a: a, tile)
                tile = jax.lax.cond(index == mesh_shape[axis]-1,
                    lambda a: a.at[upper].set(before[upper]), lambda a: a, tile)
            elif location[axis] == 'C':
                face = _axis_boundary_plane(axis, -g)
                tile = jax.lax.cond(index == mesh_shape[axis]-1,
                    lambda a: a.at[face].set(before[face]), lambda a: a, tile)
            continue
        # A normal staggered source owns the lower absorbing face even though
        # it is stored in a halo. Preserve it before the transverse directions
        # are processed, so corner sources receive their transverse folding.
        keep_face = (boundary_condition == BC_ABSORBING and location is not None
                     and parity is not None and location[axis] == 'V')
        if keep_face:
            face = _axis_boundary_plane(axis, g-1)
            lower_flux = tile[face]
        if reduced_axis:
            tile = _local_refresh_reduced_axis(
                tile,
                axis,
                g,
                boundary_condition,
                reflecting_parity=parity,
            )
        else:
            tile = _refresh_axis(tile, axis, g, axis_name, positive, negative)
            if boundary_condition == BC_CONSTANT:
                tile = _apply_local_constant_boundary_axis(tile, axis, g, axis_name, mesh_shape[axis])
            elif boundary_condition == BC_CONDUCTING and parity is not None:
                tile = _apply_local_reflecting_boundary_axis(
                    tile,
                    axis,
                    g,
                    parity,
                    axis_name,
                    mesh_shape[axis],
                )
        if keep_face:
            tile = jax.lax.cond(jax.lax.axis_index(axis_name) == 0,
                lambda a: a.at[face].set(lower_flux), lambda a: a, tile)

    return tile


def _local_fold_reduced_axis(
    tile,
    axis,
    g,
    boundary_condition,
    reflecting_parity=None,
):
    lower_slice = [slice(None), slice(None), slice(None)]
    lower_slice[axis] = slice(0, g)
    upper_slice = [slice(None), slice(None), slice(None)]
    upper_slice[axis] = slice(-g, None)
    interior_slice = [slice(None), slice(None), slice(None)]
    interior_slice[axis] = slice(g, g + 1)

    ghost_sum = jnp.sum(tile[tuple(lower_slice)], axis=axis, keepdims=True)
    ghost_sum = ghost_sum + jnp.sum(tile[tuple(upper_slice)], axis=axis, keepdims=True)
    if boundary_condition == BC_PERIODIC:
        tile = tile.at[tuple(interior_slice)].add(ghost_sum)
    elif boundary_condition == BC_CONDUCTING:
        contribution = (
            -ghost_sum
            if reflecting_parity is None
            else reflecting_parity * ghost_sum
        )
        tile = tile.at[tuple(interior_slice)].add(contribution)
    tile = tile.at[tuple(lower_slice)].set(0.0)
    tile = tile.at[tuple(upper_slice)].set(0.0)

    return tile


def _add_exterior_boundary_fold(
    tile,
    axis,
    g,
    lower_ghost,
    upper_ghost,
    axis_name,
    axis_size,
    reflecting_parity=None,
):
    lower_index = jax.lax.axis_index(axis_name) == 0
    upper_index = jax.lax.axis_index(axis_name) == axis_size - 1

    lower_target = [slice(None), slice(None), slice(None)]
    lower_target[axis] = slice(g, 2 * g)
    upper_target = [slice(None), slice(None), slice(None)]
    upper_target[axis] = slice(-2 * g, -g)

    lower_contribution = (
        -lower_ghost
        if reflecting_parity is None
        else reflecting_parity * jnp.flip(lower_ghost, axis=axis)
    )
    upper_contribution = (
        -upper_ghost
        if reflecting_parity is None
        else reflecting_parity * jnp.flip(upper_ghost, axis=axis)
    )

    tile = jax.lax.cond(
        lower_index,
        lambda local_tile: local_tile.at[tuple(lower_target)].add(lower_contribution),
        lambda local_tile: local_tile,
        tile,
    )
    tile = jax.lax.cond(
        upper_index,
        lambda local_tile: local_tile.at[tuple(upper_target)].add(upper_contribution),
        lambda local_tile: local_tile,
        tile,
    )

    return tile


def _fold_axis(
    tile,
    axis,
    g,
    axis_name,
    axis_size,
    boundary_condition,
    send_positive,
    send_negative,
    reflecting_parity=None,
):
    lower_ghost, upper_ghost, lower_interior, upper_interior = _axis_slices(axis, g)

    lower_values = tile[lower_ghost]
    upper_values = tile[upper_ghost]
    # Folding reverses halo refresh ownership: a lower ghost belongs to the
    # negative neighbor's upper interior, and an upper ghost belongs to the
    # positive neighbor's lower interior.
    from_positive_neighbor = jax.lax.ppermute(lower_values, axis_name, send_negative)
    from_negative_neighbor = jax.lax.ppermute(upper_values, axis_name, send_positive)

    tile = tile.at[upper_interior].add(from_positive_neighbor)
    tile = tile.at[lower_interior].add(from_negative_neighbor)
    if boundary_condition == BC_CONDUCTING:
        tile = _add_exterior_boundary_fold(
            tile,
            axis,
            g,
            lower_values,
            upper_values,
            axis_name,
            axis_size,
            reflecting_parity=reflecting_parity,
        )
    tile = tile.at[lower_ghost].set(0.0)
    tile = tile.at[upper_ghost].set(0.0)

    return tile


def _local_fold_scalar_tile(
    tile,
    g,
    boundary_conditions,
    reduced_axes,
    mesh_shape,
    send_positive,
    send_negative,
    reflecting_parity=None,
    location=None,
):
    axis_parities = (None, None, None) if reflecting_parity is None else reflecting_parity
    for axis, axis_name, axis_size, boundary_condition, reduced_axis, positive, negative, parity in zip(
        range(3),
        MESH_AXES,
        mesh_shape,
        boundary_conditions,
        reduced_axes,
        send_positive,
        send_negative,
        axis_parities,
    ):
        if boundary_condition == BC_POLAR:
            continue  # polar theta is handled with explicit C/V ownership
        if boundary_condition == BC_CONDUCTING and location is not None:
            tile = _staggered_conducting_axis(
                tile, axis, g, axis_name, mesh_shape[axis], positive, negative,
                location[axis], parity, fold=True)
            continue
        keep_face = (boundary_condition == BC_ABSORBING and location is not None
                     and parity is not None and location[axis] == 'V')
        if keep_face:
            face = _axis_boundary_plane(axis, g-1)
            lower_flux = tile[face]
        if reduced_axis:
            tile = _local_fold_reduced_axis(
                tile,
                axis,
                g,
                boundary_condition,
                reflecting_parity=parity,
            )
        else:
            tile = _fold_axis(
                tile,
                axis,
                g,
                axis_name,
                axis_size,
                boundary_condition,
                positive,
                negative,
                reflecting_parity=parity,
            )
        if keep_face:
            tile = jax.lax.cond(jax.lax.axis_index(axis_name) == 0,
                lambda a: a.at[face].set(lower_flux), lambda a: a, tile)

    return tile


def _axis_boundary_plane(axis, index):
    plane = [slice(None), slice(None), slice(None)]
    plane[axis] = index
    return tuple(plane)


def _axis_constant_boundary_slices(axis, g):
    lower_ghost = [slice(None), slice(None), slice(None)]
    upper_ghost = [slice(None), slice(None), slice(None)]
    lower_interior = [slice(None), slice(None), slice(None)]
    upper_interior = [slice(None), slice(None), slice(None)]

    lower_ghost[axis] = slice(0, g)
    upper_ghost[axis] = slice(-g, None)
    lower_interior[axis] = slice(g, g + 1)
    upper_interior[axis] = slice(-g - 1, -g)

    return (
        tuple(lower_ghost),
        tuple(upper_ghost),
        tuple(lower_interior),
        tuple(upper_interior),
    )


def _apply_local_zero_boundary_axis(tile, axis, g, axis_name, axis_size):
    lower_plane = _axis_boundary_plane(axis, g)
    upper_plane = _axis_boundary_plane(axis, -g)
    tile_index = jax.lax.axis_index(axis_name)

    tile = jax.lax.cond(
        tile_index == 0,
        lambda local_tile: local_tile.at[lower_plane].set(0.0),
        lambda local_tile: local_tile,
        tile,
    )
    tile = jax.lax.cond(
        tile_index == axis_size - 1,
        lambda local_tile: local_tile.at[upper_plane].set(0.0),
        lambda local_tile: local_tile,
        tile,
    )

    return tile


def _apply_local_constant_boundary_axis(tile, axis, g, axis_name, axis_size):
    lower_ghost, upper_ghost, lower_interior, upper_interior = _axis_constant_boundary_slices(axis, g)
    tile_index = jax.lax.axis_index(axis_name)

    tile = jax.lax.cond(
        tile_index == 0,
        lambda local_tile: local_tile.at[lower_ghost].set(
            jnp.broadcast_to(local_tile[lower_interior], local_tile[lower_ghost].shape)
        ),
        lambda local_tile: local_tile,
        tile,
    )
    tile = jax.lax.cond(
        tile_index == axis_size - 1,
        lambda local_tile: local_tile.at[upper_ghost].set(
            jnp.broadcast_to(local_tile[upper_interior], local_tile[upper_ghost].shape)
        ),
        lambda local_tile: local_tile,
        tile,
    )

    return tile


def make_distributed_ghost_updater(
    mesh,
    tile_shape,
    boundary_conditions,
    num_guard_cells,
    *,
    reflecting_parity=None,
    location=None,
    preserve_exterior=False,
):
    """
    Build a shard-mapped scalar halo refresher.

    Timestepping code should construct this once during simulation setup when
    possible, then reuse the returned updater instead of rebuilding it every
    step.
    """

    g = int(num_guard_cells)
    tile_shape = tuple(int(width) for width in tile_shape)
    boundary_conditions = tuple(int(bc) for bc in boundary_conditions)
    if reflecting_parity is not None:
        reflecting_parity = tuple(int(value) for value in reflecting_parity)
    mesh_shape = tuple(int(width) for width in mesh.devices.shape)
    reduced_axes = _reduced_axes_from_tile_shape(tile_shape, mesh_shape)
    send_positive, send_negative = _axis_permutations(mesh_shape, boundary_conditions)

    def local_update(local_tiles):
        tile = local_tiles[0, 0, 0]
        tile = _local_refresh_scalar_tile(
            tile,
            g,
            boundary_conditions,
            reduced_axes,
            mesh_shape,
            send_positive,
            send_negative,
            reflecting_parity=reflecting_parity,
            location=location,
            preserve_exterior=preserve_exterior,
        )
        return tile[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :, :]

    mapped_update = jax.shard_map(
        local_update,
        mesh=mesh,
        in_specs=SCALAR_TILE_SPEC,
        out_specs=SCALAR_TILE_SPEC,
        check_vma=False,
    )

    def update(field_tiles):
        _validate_scalar_tile_topology(field_tiles, mesh)
        return mapped_update(field_tiles)

    return update


def make_distributed_vector_ghost_updater(
    mesh,
    tile_shape,
    boundary_conditions,
    num_guard_cells,
    *,
    reflecting_parity=None,
):
    """
    Build a shard-mapped vector halo refresher.

    Timestepping code should construct this once during simulation setup when
    possible, then reuse the returned updater instead of rebuilding it every
    step.
    """

    g = int(num_guard_cells)
    tile_shape = tuple(int(width) for width in tile_shape)
    boundary_conditions = tuple(int(bc) for bc in boundary_conditions)
    if reflecting_parity is not None:
        reflecting_parity = tuple(tuple(int(value) for value in component)
                                  for component in reflecting_parity)
    mesh_shape = tuple(int(width) for width in mesh.devices.shape)
    reduced_axes = _reduced_axes_from_tile_shape(tile_shape, mesh_shape)
    send_positive, send_negative = _axis_permutations(mesh_shape, boundary_conditions)

    if reflecting_parity is None:
        component_parity = None
    else:
        component_parity = jnp.asarray(reflecting_parity)

    def local_update(local_tiles):
        def update_component(local_component, parity):
            tile = local_component[0, 0, 0]
            tile = _local_refresh_scalar_tile(
                tile,
                g,
                boundary_conditions,
                reduced_axes,
                mesh_shape,
                send_positive,
                send_negative,
                reflecting_parity=parity,
            )
            return tile[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :, :]

        if component_parity is None:
            return jax.vmap(
                lambda component: update_component(component, None),
                in_axes=0,
                out_axes=0,
            )(local_tiles)
        return jax.vmap(update_component, in_axes=(0, 0), out_axes=0)(
            local_tiles,
            component_parity,
        )

    mapped_update = jax.shard_map(
        local_update,
        mesh=mesh,
        in_specs=VECTOR_TILE_SPEC,
        out_specs=VECTOR_TILE_SPEC,
        check_vma=False,
    )

    def update(field_tiles):
        _validate_vector_tile_topology(field_tiles, mesh)
        stacked_tiles = _stack_tiled_vector_field(field_tiles)
        refreshed = mapped_update(stacked_tiles)
        return _restore_tiled_vector_layout(refreshed, field_tiles)

    return update


def make_distributed_ghost_folder(
    mesh,
    tile_shape,
    boundary_conditions,
    num_guard_cells,
    *,
    reflecting_parity=None,
    location=None,
):
    """
    Build a shard-mapped scalar ghost-deposit folder.

    Timestepping code should construct this once during simulation setup when
    possible, then reuse the returned folder instead of rebuilding it every
    step.
    """

    g = int(num_guard_cells)
    tile_shape = tuple(int(width) for width in tile_shape)
    boundary_conditions = tuple(int(bc) for bc in boundary_conditions)
    if reflecting_parity is not None:
        reflecting_parity = tuple(int(value) for value in reflecting_parity)
    mesh_shape = tuple(int(width) for width in mesh.devices.shape)
    reduced_axes = _reduced_axes_from_tile_shape(tile_shape, mesh_shape)
    send_positive, send_negative = _axis_permutations(mesh_shape, boundary_conditions)

    def local_fold(local_tiles):
        tile = local_tiles[0, 0, 0]
        tile = _local_fold_scalar_tile(
            tile,
            g,
            boundary_conditions,
            reduced_axes,
            mesh_shape,
            send_positive,
            send_negative,
            reflecting_parity=reflecting_parity,
            location=location,
        )
        return tile[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :, :]

    mapped_fold = jax.shard_map(
        local_fold,
        mesh=mesh,
        in_specs=SCALAR_TILE_SPEC,
        out_specs=SCALAR_TILE_SPEC,
        check_vma=False,
    )

    def fold(field_tiles):
        _validate_scalar_tile_topology(field_tiles, mesh)
        return mapped_fold(field_tiles)

    return fold


def make_distributed_vector_ghost_folder(
    mesh,
    tile_shape,
    boundary_conditions,
    num_guard_cells,
    *,
    reflecting_parity=None,
):
    """
    Build a shard-mapped vector ghost-deposit folder.

    Timestepping code should construct this once during simulation setup when
    possible, then reuse the returned folder instead of rebuilding it every
    step.
    """

    g = int(num_guard_cells)
    tile_shape = tuple(int(width) for width in tile_shape)
    boundary_conditions = tuple(int(bc) for bc in boundary_conditions)
    if reflecting_parity is not None:
        reflecting_parity = tuple(tuple(int(value) for value in component)
                                  for component in reflecting_parity)
    mesh_shape = tuple(int(width) for width in mesh.devices.shape)
    reduced_axes = _reduced_axes_from_tile_shape(tile_shape, mesh_shape)
    send_positive, send_negative = _axis_permutations(mesh_shape, boundary_conditions)

    if reflecting_parity is None:
        component_parity = None
    else:
        component_parity = jnp.asarray(reflecting_parity)

    def local_fold(local_tiles):
        def fold_component(local_component, parity):
            tile = local_component[0, 0, 0]
            tile = _local_fold_scalar_tile(
                tile,
                g,
                boundary_conditions,
                reduced_axes,
                mesh_shape,
                send_positive,
                send_negative,
                reflecting_parity=parity,
            )
            return tile[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :, :]

        if component_parity is None:
            return jax.vmap(
                lambda component: fold_component(component, None),
                in_axes=0,
                out_axes=0,
            )(local_tiles)
        return jax.vmap(fold_component, in_axes=(0, 0), out_axes=0)(
            local_tiles,
            component_parity,
        )

    mapped_fold = jax.shard_map(
        local_fold,
        mesh=mesh,
        in_specs=VECTOR_TILE_SPEC,
        out_specs=VECTOR_TILE_SPEC,
        check_vma=False,
    )

    def fold(field_tiles):
        _validate_vector_tile_topology(field_tiles, mesh)
        stacked_tiles = _stack_tiled_vector_field(field_tiles)
        folded = mapped_fold(stacked_tiles)
        return _restore_tiled_vector_layout(folded, field_tiles)

    return fold


def make_distributed_zero_boundary(mesh, tile_shape, axis, num_guard_cells):
    g = int(num_guard_cells)
    del tile_shape
    axis = int(axis)
    mesh_shape = tuple(int(width) for width in mesh.devices.shape)
    axis_name = MESH_AXES[axis]
    axis_size = mesh_shape[axis]

    def local_apply(local_tiles):
        tile = local_tiles[0, 0, 0]
        tile = _apply_local_zero_boundary_axis(tile, axis, g, axis_name, axis_size)
        return tile[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :, :]

    mapped_apply = jax.shard_map(
        local_apply,
        mesh=mesh,
        in_specs=SCALAR_TILE_SPEC,
        out_specs=SCALAR_TILE_SPEC,
        check_vma=False,
    )

    def apply(field_tiles):
        _validate_scalar_tile_topology(field_tiles, mesh)
        return mapped_apply(field_tiles)

    return apply


def make_distributed_constant_boundary(mesh, tile_shape, axis, num_guard_cells):
    g = int(num_guard_cells)
    del tile_shape
    axis = int(axis)
    mesh_shape = tuple(int(width) for width in mesh.devices.shape)
    axis_name = MESH_AXES[axis]
    axis_size = mesh_shape[axis]

    def local_apply(local_tiles):
        tile = local_tiles[0, 0, 0]
        tile = _apply_local_constant_boundary_axis(tile, axis, g, axis_name, axis_size)
        return tile[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :, :]

    mapped_apply = jax.shard_map(
        local_apply,
        mesh=mesh,
        in_specs=SCALAR_TILE_SPEC,
        out_specs=SCALAR_TILE_SPEC,
        check_vma=False,
    )

    def apply(field_tiles):
        _validate_scalar_tile_topology(field_tiles, mesh)
        return mapped_apply(field_tiles)

    return apply


def update_tiled_ghost_cells(
    field_tiles,
    static_parameters,
    num_guard_cells=2,
    bc_type=BC_TYPE_FIELD,
    *,
    reflecting_parity=None,
    location=None,
    preserve_exterior=False,
):
    """
    Refresh scalar tile halos with one logical tile per JAX device.

    Scalar tiled fields have logical shape
    ``(ntx, nty, ntz, tile_nx + 2*g, tile_ny + 2*g, tile_nz + 2*g)``.
    Cross-tile communication uses ``jax.lax.ppermute`` inside
    ``jax.shard_map`` over the named mesh axes ``tile_x``, ``tile_y``, and
    ``tile_z``. The leading tile topology must match the device mesh. Particle
    boundaries mirror exterior halos using ``reflecting_parity``; scalar
    particle deposits are even by default. ``location`` preserves owned upper
    C endpoints at conducting walls. ``preserve_exterior`` keeps computed
    nonperiodic exterior slabs (a bool, or an x/y/z tuple of bools).
    """

    if (bc_type == BC_TYPE_PARTICLE and static_parameters.solver == 'static_metric'
            and BC_POLAR not in static_parameters.particle_boundary_conditions):
        from .staggered import source_boundaries
        return source_boundaries(field_tiles, static_parameters._replace(guard_cells=int(num_guard_cells)),
                                 fold=False, vector=False, reflecting_parity=reflecting_parity)

    tile_shape = tuple(int(width) for width in static_parameters.tile_shape)
    mesh = static_parameters.field_mesh
    reflecting_parity = _particle_reflecting_parity(bc_type, reflecting_parity, vector=False)
    updater = make_distributed_ghost_updater(
        mesh,
        tile_shape,
        _boundary_conditions_for_type(static_parameters, bc_type),
        num_guard_cells,
        reflecting_parity=reflecting_parity,
        location=location,
        preserve_exterior=preserve_exterior,
    )
    result = updater(field_tiles)
    if bc_type == BC_TYPE_PARTICLE and static_parameters.particle_boundary_conditions[1] == BC_POLAR:
        from .polar import refresh
        return refresh(result, num_guard_cells, tile_shape[1])
    return result


def update_tiled_vector_ghost_cells(
    field_tiles,
    static_parameters,
    num_guard_cells=2,
    bc_type=BC_TYPE_FIELD,
    *,
    reflecting_parity=None,
    locations=None,
    preserve_exterior=False,
):
    """
    Refresh tiled multi-component field halos, preserving stacked or tuple layout.

    Production vector fields use three components. The component axis remains
    general so related spatial operators can batch several derivative channels
    through the same distributed halo exchange. Particle vectors default to
    odd normal and even tangential reflection parity.
    """

    if (bc_type == BC_TYPE_PARTICLE and static_parameters.solver == 'static_metric'
            and BC_POLAR not in static_parameters.particle_boundary_conditions):
        from .staggered import source_boundaries
        return source_boundaries(field_tiles, static_parameters._replace(guard_cells=int(num_guard_cells)),
                                 fold=False, vector=True, reflecting_parity=reflecting_parity)

    preserve_any = any(preserve_exterior) if isinstance(preserve_exterior, tuple) else preserve_exterior
    if (preserve_any or locations is not None and
            BC_CONDUCTING in _boundary_conditions_for_type(static_parameters, bc_type)):
        locations = (None,) * len(field_tiles) if locations is None else locations
        result = tuple(update_tiled_ghost_cells(
            value, static_parameters, num_guard_cells, bc_type,
            reflecting_parity=None if reflecting_parity is None else reflecting_parity[i],
            location=location, preserve_exterior=preserve_exterior)
            for i, (value, location) in enumerate(zip(field_tiles, locations)))
        return jnp.stack(result) if _is_stacked_tiled_vector_field(field_tiles) else result

    tile_shape = tuple(int(width) for width in static_parameters.tile_shape)
    mesh = static_parameters.field_mesh
    reflecting_parity = _particle_reflecting_parity(bc_type, reflecting_parity, vector=True)
    updater = make_distributed_vector_ghost_updater(
        mesh,
        tile_shape,
        _boundary_conditions_for_type(static_parameters, bc_type),
        num_guard_cells,
        reflecting_parity=reflecting_parity,
    )
    result = updater(field_tiles)
    if bc_type == BC_TYPE_PARTICLE and static_parameters.particle_boundary_conditions[1] == BC_POLAR:
        from .polar import refresh
        arrays = tuple(refresh(result[i], num_guard_cells, tile_shape[1], i == 1, -1 if i == 1 else 1) for i in range(3))
        return jnp.stack(arrays) if hasattr(result, 'ndim') and result.ndim == 7 else arrays
    return result


def apply_tiled_zero_boundary(field_tiles, static_parameters, axis, num_guard_cells=2, *, location=None):
    """
    Zero scalar values on the global conducting wall for one spatial axis.
    """

    axis = int(axis)
    boundary_conditions = _boundary_tuple(static_parameters.boundary_conditions)
    if boundary_conditions[axis] != BC_CONDUCTING:
        return update_tiled_ghost_cells(field_tiles, static_parameters, num_guard_cells, location=location)

    tile_shape = tuple(int(width) for width in static_parameters.tile_shape)
    mesh = static_parameters.field_mesh
    apply_bc = make_distributed_zero_boundary(
        mesh,
        tile_shape,
        axis,
        num_guard_cells,
    )
    field_tiles = apply_bc(field_tiles)
    return update_tiled_ghost_cells(field_tiles, static_parameters, num_guard_cells, location=location)


def apply_tiled_pec_boundary(fields, static_parameters):
    """Yee tangential electric clamp at the physical endpoints, g and g+n."""
    from PyPIC3D.relativity.core import D_FIELD_LOCATIONS
    result = list(fields)
    for axis, bc in enumerate(static_parameters.boundary_conditions):
        if bc == BC_CONDUCTING:
            for i in range(3):
                if i != axis:
                    result[i] = apply_tiled_zero_boundary(
                        result[i], static_parameters, axis, static_parameters.guard_cells,
                        location=D_FIELD_LOCATIONS[i])
    return tuple(result)


def apply_tiled_constant_boundary(field_tiles, static_parameters, axis, num_guard_cells=2):
    """
    Fill exterior ghost cells from the adjacent interior plane.

    This is used by the electrostatic scalar potential on conducting walls and
    by explicit constant field boundaries.  Internal tile halos are still
    refreshed through the distributed ppermute path before the exterior ghosts
    are overwritten.
    """

    axis = int(axis)
    field_tiles = update_tiled_ghost_cells(field_tiles, static_parameters, num_guard_cells)

    boundary_conditions = _boundary_tuple(static_parameters.boundary_conditions)
    if boundary_conditions[axis] not in (BC_CONDUCTING, BC_CONSTANT):
        return field_tiles

    tile_shape = tuple(int(width) for width in static_parameters.tile_shape)
    mesh = static_parameters.field_mesh
    apply_bc = make_distributed_constant_boundary(
        mesh,
        tile_shape,
        axis,
        num_guard_cells,
    )
    return apply_bc(field_tiles)


def fold_tiled_ghost_cells(
    field_tiles,
    static_parameters,
    num_guard_cells=2,
    bc_type=BC_TYPE_FIELD,
    *,
    reflecting_parity=None,
):
    """
    Add tile-ghost deposits into owning interiors, then clear ghost cells.

    Folding uses the same x -> y -> z order as halo refresh. A ghost deposit
    is sent back to the neighboring interior that owns it; conducting exterior
    deposits reflect only on devices touching the true global walls. Particle
    deposits use parity-aware nearest-to-nearest reflection.
    """

    if (bc_type == BC_TYPE_PARTICLE and static_parameters.solver == 'static_metric'
            and BC_POLAR not in static_parameters.particle_boundary_conditions):
        from .staggered import source_boundaries
        return source_boundaries(field_tiles, static_parameters._replace(guard_cells=int(num_guard_cells)),
                                 fold=True, vector=False, reflecting_parity=reflecting_parity)

    tile_shape = tuple(int(width) for width in static_parameters.tile_shape)
    mesh = static_parameters.field_mesh
    reflecting_parity = _particle_reflecting_parity(bc_type, reflecting_parity, vector=False)
    folder = make_distributed_ghost_folder(
        mesh,
        tile_shape,
        _boundary_conditions_for_type(static_parameters, bc_type),
        num_guard_cells,
        reflecting_parity=reflecting_parity,
    )
    result = folder(field_tiles)
    if bc_type == BC_TYPE_PARTICLE and static_parameters.particle_boundary_conditions[1] == BC_POLAR:
        from .polar import fold
        return fold(result, num_guard_cells, tile_shape[1])
    return result

def fold_tiled_vector_ghost_cells(
    field_tiles,
    static_parameters,
    num_guard_cells=2,
    bc_type=BC_TYPE_FIELD,
    *,
    reflecting_parity=None,
):
    """
    Fold tile-ghost deposits for a tiled vector field.
    """

    if (bc_type == BC_TYPE_PARTICLE and static_parameters.solver == 'static_metric'
            and BC_POLAR not in static_parameters.particle_boundary_conditions):
        from .staggered import source_boundaries
        return source_boundaries(field_tiles, static_parameters._replace(guard_cells=int(num_guard_cells)),
                                 fold=True, vector=True, reflecting_parity=reflecting_parity)

    tile_shape = tuple(int(width) for width in static_parameters.tile_shape)
    mesh = static_parameters.field_mesh
    reflecting_parity = _particle_reflecting_parity(bc_type, reflecting_parity, vector=True)
    folder = make_distributed_vector_ghost_folder(
        mesh,
        tile_shape,
        _boundary_conditions_for_type(static_parameters, bc_type),
        num_guard_cells,
        reflecting_parity=reflecting_parity,
    )
    result = folder(field_tiles)
    if bc_type == BC_TYPE_PARTICLE and static_parameters.particle_boundary_conditions[1] == BC_POLAR:
        from .polar import fold
        arrays = tuple(fold(result[i], num_guard_cells, tile_shape[1], i == 1, -1 if i == 1 else 1) for i in range(3))
        return jnp.stack(arrays) if hasattr(result, 'ndim') and result.ndim == 7 else arrays
    return result
