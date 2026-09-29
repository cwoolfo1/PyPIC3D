"""Local metric projections for coordinate-aligned, stationary FIDO walls.

The algebra acts on co-located contravariant vectors. Writing individual rows
back to a Yee grid is an explicit interpolation, not a coupled boundary solve.
"""
from functools import partial
from itertools import combinations

import jax.numpy as jnp

from .grid_and_stencil import BC_CONDUCTING
from .ghost_cells import update_tiled_vector_ghost_cells
from .ownership import boundary_plane, face_mask, staggered_mirror
from PyPIC3D.relativity.field_interpolation import metric_at_location, reconstruct_vector


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


def _normal_component(inverse, vector, axis, component):
    """One row of the coordinate-face projector G applied to a vector."""
    return inverse[..., component, axis] / inverse[..., axis, axis] * vector[axis]


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
        plane = partial(boundary_plane, axis, tile)
        for node in outside:
            owner, walls = staggered_mirror(node-g+offset, total, location[axis])
            components = [v[plane(g+owner-offset)] for v in vector]
            for wall in reversed(walls):
                wall_inverse = inverse[plane(g+wall-offset)]
                normal = [_normal_component(wall_inverse, components, axis, j) for j in range(3)]
                components = [2*normal[j]-components[j] if field_kind == 'D'
                              else components[j]-2*normal[j] for j in range(3)]
            for j in range(3):
                result[j] = result[j].at[plane(node)].set(components[j])
    return tuple(result)


def _exchange_preserving_walls(fields, static, locations, rounds):
    """Communicate owner values without replacing conducting exterior slabs."""
    for _ in range(rounds):
        fields = update_tiled_vector_ghost_cells(
            fields, static, static.guard_cells, locations=locations,
            preserve_exterior=tuple(bc == BC_CONDUCTING for bc in static.boundary_conditions),
        )
    return fields


def _prepare_normal_D(snapshot, static, axes):
    """Supply the identity normal reflection row before reconstructing wall D.

    This makes the wall trace depend on current owners rather than stale
    exterior input. All faces read the same exchanged snapshot.
    """
    shape = snapshot[0].shape
    guard = static.guard_cells
    prepared = list(snapshot)
    for axis in axes:
        width = static.tile_shape[axis]
        for low in (True, False):
            tile = 0 if low else shape[axis] - 1
            offset = tile * width
            outside = range(guard) if low else range(guard + width, shape[axis + 3])
            plane = partial(boundary_plane, axis, tile)
            for node in outside:
                owner, _ = staggered_mirror(node - guard + offset, width * shape[axis], 'V')
                prepared[axis] = prepared[axis].at[plane(node)].set(
                    snapshot[axis][plane(guard + owner - offset)]
                )
    return tuple(prepared)


def _project_native_nodes(snapshot, static, locations, field_kind, metric, axes):
    """Project faces first, then overwrite intersections with joint constraints."""
    shape = snapshot[0].shape
    projected = []
    for component, location in enumerate(locations):
        value = snapshot[component]
        at_target = reconstruct_vector(snapshot, locations, metric, location)
        inverse = metric_at_location(metric, location).gamma_inv
        incident = tuple(axis for axis in axes if location[axis] == 'C')
        for count in range(1, len(incident) + 1):
            for faces in combinations(incident, count):
                mask = True
                for axis in faces:
                    mask = mask & face_mask(shape, axis, static.tile_shape[axis], static.guard_cells)
                if count == 1:
                    normal = _normal_component(inverse, at_target, faces[0], component)
                    constrained = normal if field_kind == 'D' else at_target[component] - normal
                elif field_kind == 'D':
                    constrained = 0.
                else:
                    projector = tangent_intersection_projector(inverse, jnp.eye(3)[jnp.array(faces)])
                    constrained = sum(projector[..., component, j] * at_target[j] for j in range(3))
                value = jnp.where(mask, constrained, value)
        projected.append(value)
    return tuple(projected)


def _reflect_exterior_axis(snapshot, static, locations, field_kind, metric, axis, previous_axes):
    """Reflect one exterior axis, composing earlier reflections at corners."""
    result = []
    for component, location in enumerate(locations):
        reconstructed = reconstruct_vector(snapshot, locations, metric, location)
        # Native transfers at corners can leave the allocated halo. Reconstruct
        # at the reflected owner first, then compose co-located reflections.
        for previous in previous_axes:
            reconstructed = _reflect_vector(reconstructed, metric, location, previous, static, field_kind)
        reflected = _reflect_vector(reconstructed, metric, location, axis, static, field_kind)
        result.append(reflected[component])
    return tuple(result)


def project_fields(vector, static, locations, field_kind, metric):
    """Exchange, project native walls, then reflect exterior ghosts in x/y/z order.

    Each face reads a common snapshot. Intersections use simultaneous
    constraints (zero D for independent normals; common tangent B). Internal
    boundaries are communicated, never projected.
    """
    axes = tuple(axis for axis, bc in enumerate(static.boundary_conditions) if bc == BC_CONDUCTING)
    shape = vector[0].shape
    guard = static.guard_cells
    # Narrow distributed axes need multiple neighbor passes; reduced periodic
    # directions are filled in one pass.
    rounds = max([1] + [(guard + width - 1) // width
                       for axis, width in enumerate(static.tile_shape) if shape[axis] > 1])
    exchange = partial(_exchange_preserving_walls, static=static, locations=locations, rounds=rounds)

    snapshot = exchange(vector)
    if field_kind == 'D':
        snapshot = exchange(_prepare_normal_D(snapshot, static, axes))
    result = exchange(_project_native_nodes(snapshot, static, locations, field_kind, metric, axes))
    for position, axis in enumerate(axes):
        result = exchange(_reflect_exterior_axis(
            result, static, locations, field_kind, metric, axis, axes[:position]
        ))
    return result
