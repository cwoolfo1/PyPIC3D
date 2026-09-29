"""Local metric projections for coordinate-aligned, stationary FIDO walls.

The algebra acts on co-located contravariant vectors. Writing individual rows
back to a Yee grid is an explicit interpolation, not a coupled boundary solve.
"""
from itertools import combinations

import jax.numpy as jnp

from .grid_and_stencil import BC_CONDUCTING
from .ghost_cells import update_tiled_vector_ghost_cells, staggered_mirror
from PyPIC3D.relativity.field_interpolation import metric_at_location, metric_weighted_interpolate


def normal_projector(gamma_inv, covector):
    """Return G=n^i n_j for an arbitrary (not necessarily unit) covector."""
    raised = jnp.einsum('...ij,...j->...i', gamma_inv, covector)
    norm2 = jnp.einsum('...i,...i->...', covector, raised)
    return raised[..., :, None] * covector[..., None, :] / norm2[..., None, None]


def tangent_intersection_projector(gamma_inv, covectors):
    """Project onto the common tangent space of independent surface normals.

    ``covectors`` has shape (..., number_of_faces, 3). The Gram matrix uses
    the inverse spatial metric; normalization of its rows is unnecessary.
    """
    raised = jnp.einsum('...ij,...aj->...ia', gamma_inv, covectors)
    gram = jnp.einsum('...ai,...ib->...ab', covectors, raised)
    return jnp.eye(3) - raised @ jnp.linalg.solve(gram, covectors)


def _face_mask(shape, axis, n, g):
    tile = jnp.arange(shape[axis])
    node = jnp.arange(shape[axis+3])
    mask = ((tile[:, None] == 0) & (node[None, :] == g)
            | (tile[:, None] == shape[axis]-1) & (node[None, :] == g+n))
    dims = [1]*6
    dims[axis], dims[axis+3] = shape[axis], shape[axis+3]
    return mask.reshape(dims)


def owned_nodes(shape, location, static):
    """Active nodes, including conducting upper C endpoints owned by a tile."""
    mask = True
    g = static.guard_cells
    for axis, n in enumerate(static.tile_shape):
        nodes = jnp.arange(shape[axis+3])
        dims = [1]*6
        dims[axis+3] = len(nodes)
        active = ((nodes >= g) & (nodes < g+n)).reshape(dims)
        if static.boundary_conditions[axis] == BC_CONDUCTING and location[axis] == 'C':
            active = active | _face_mask(shape, axis, n, g)
        mask = mask & active
    return mask


def _reconstruct(vector, locations, metric, target):
    target_metric = metric_at_location(metric, target)
    return tuple(value if source == target else metric_weighted_interpolate(value, metric_at_location(metric, source),
                                            target_metric, source, target)
                 for value, source in zip(vector, locations))


def _reflect_vector(vector, metric, location, axis, static, field_kind):
    """Reflect a co-located vector about one pair of physical faces."""
    shape = vector[0].shape
    g, n = static.guard_cells, static.tile_shape[axis]
    total = n*shape[axis]
    surface = list(location)
    surface[axis] = 'C'
    inverse = metric_at_location(metric, tuple(surface)).gamma_inv
    result = list(vector)
    for low in (True, False):
        tile = 0 if low else shape[axis]-1
        offset = tile*n
        last = g+n-1 if location[axis] == 'V' else g+n
        outside = range(g) if low else range(last+1, shape[axis+3])
        def plane(node):
            index = [slice(None)]*6
            index[axis], index[axis+3] = tile, node
            return tuple(index)
        for node in outside:
            owner, walls = staggered_mirror(node-g+offset, total, location[axis])
            components = [v[plane(g+owner-offset)] for v in vector]
            for wall in reversed(walls):
                wall_inverse = inverse[plane(g+wall-offset)]
                normal = [wall_inverse[..., j, axis]/wall_inverse[..., axis, axis]*components[axis]
                          for j in range(3)]
                components = [2*normal[j]-components[j] if field_kind == 'D'
                              else components[j]-2*normal[j] for j in range(3)]
            for j in range(3):
                result[j] = result[j].at[plane(node)].set(components[j])
    return tuple(result)


def project_fields(vector, static, locations, field_kind, metric):
    """Project native wall nodes and reflect exterior ghosts in x/y/z order.

    Each face reads a common snapshot. Intersections use the simultaneous
    constraints (zero D for two independent normals; common tangent B).
    Internal boundaries are communicated, never projected. No global field
    assembly or metric-specific formula is used.
    """
    axes = tuple(a for a, bc in enumerate(static.boundary_conditions) if bc == BC_CONDUCTING)
    g = static.guard_cells
    shape = vector[0].shape

    rounds = max([1] + [(g+n-1)//n for a, n in enumerate(static.tile_shape) if shape[a] > 1])

    def exchange(fields):
        # Only narrow distributed axes need multiple nearest-neighbor passes;
        # a reduced periodic direction is filled in one pass.
        for _ in range(rounds):
            fields = update_tiled_vector_ghost_cells(
                fields, static, g, locations=locations,
                preserve_exterior=tuple(bc == BC_CONDUCTING for bc in static.boundary_conditions))
        return fields

    snapshot = exchange(vector)
    if field_kind == 'D':
        # R_D has an identity normal row for a coordinate face, even for a
        # full metric. Prepare this row before reconstructing wall D so the
        # result uses the current owner values, not stale input exterior data.
        prepared = list(snapshot)
        for axis in axes:
            n = static.tile_shape[axis]
            for low in (True, False):
                tile = 0 if low else shape[axis]-1
                offset = tile*n
                outside = range(g) if low else range(g+n, shape[axis+3])
                def plane(node):
                    index = [slice(None)]*6
                    index[axis], index[axis+3] = tile, node
                    return tuple(index)
                for node in outside:
                    owner, _ = staggered_mirror(node-g+offset, n*shape[axis], 'V')
                    prepared[axis] = prepared[axis].at[plane(node)].set(
                        snapshot[axis][plane(g+owner-offset)])
        snapshot = exchange(tuple(prepared))
    projected = []
    for i, location in enumerate(locations):
        value = snapshot[i]
        at_target = _reconstruct(snapshot, locations, metric, location)
        inverse = metric_at_location(metric, location).gamma_inv
        incident = tuple(a for a in axes if location[a] == 'C')
        for count in range(1, len(incident)+1):
            for faces in combinations(incident, count):
                mask = True
                for a in faces:
                    mask = mask & _face_mask(shape, a, static.tile_shape[a], g)
                if count == 1:
                    a = faces[0]
                    normal = inverse[..., i, a] / inverse[..., a, a] * at_target[a]
                    component = normal if field_kind == 'D' else at_target[i] - normal
                elif field_kind == 'D':
                    component = 0.
                else:
                    projector = tangent_intersection_projector(inverse, jnp.eye(3)[jnp.array(faces)])
                    component = sum(projector[..., i, j]*at_target[j] for j in range(3))
                value = jnp.where(mask, component, value)
        projected.append(value)
    result = exchange(tuple(projected))

    for position, axis in enumerate(axes):
        snapshot = result
        result = []
        for i, location in enumerate(locations):
            reconstructed = _reconstruct(snapshot, locations, metric, location)
            # At an exterior corner, a native component transfer can reach
            # beyond the allocated halo. Reconstruct at the reflected owner
            # first, then compose the co-located reflections in axis order.
            # This avoids local-array wraparound and is independent of tiling.
            for previous in axes[:position]:
                reconstructed = _reflect_vector(reconstructed, metric, location,
                                                previous, static, field_kind)
            reflected = _reflect_vector(reconstructed, metric, location, axis, static, field_kind)
            result.append(reflected[i])
        result = exchange(tuple(result))
    return result
