"""Regression contracts for persistent Maxwell densities and physical adapters."""
import unittest
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

from PyPIC3D.boundary_conditions.ownership import owned_nodes
from PyPIC3D.boundary_conditions.staggered import refresh_fields
from PyPIC3D.boundary_conditions.supergaussian import apply_tiled_supergaussian_absorber
from PyPIC3D.diagnostics.output_adapters import build_field_output_map, fields_for_output
from PyPIC3D.diagnostics.static_metric import densitized_divergence
from PyPIC3D.deposition.GR_Esirkepov import GR_Esirkepov_densitized_current, GR_Esirkepov_current
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.relativity.core import D_FIELD_LOCATIONS, B_FIELD_LOCATIONS
from PyPIC3D.relativity.field_interpolation import reconstruct_vector
from PyPIC3D.relativity.field_state import (
    densitize_vector, physical_vector, densitize_fields, physical_fields,
)
from PyPIC3D.solvers.gr_static.static_metric import (
    compute_covariant_E_densitized, compute_covariant_H_densitized,
    update_D_densitized, update_B_densitized, refresh_densitized_fields,
)
from PyPIC3D.solvers.gr_static.time_loop import time_loop_static_metric
from tests.code_tests.pec_projector_test import coupled_setup


def random_vectors(metric):
    rng = np.random.default_rng(2026)
    return [tuple(jnp.asarray(rng.normal(size=metric.center.lapse.shape))
                  for _ in range(3)) for _ in range(3)]


def curl(vector, dynamic, forward):
    """Independent cyclic finite-difference reference."""
    spacing = (dynamic.dx, dynamic.dy, dynamic.dz)
    def derivative(value, axis):
        shift = -1 if forward else 1
        return (jnp.roll(value, shift, axis+3)-value)*(1 if forward else -1)/spacing[axis]
    return tuple(derivative(vector[(i+2)%3], (i+1)%3)
                 - derivative(vector[(i+1)%3], (i+2)%3) for i in range(3))


class TestDensitizedMaxwell(unittest.TestCase):
    def assert_vectors_close(self, actual, expected, tolerance=3e-13):
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, rtol=tolerance, atol=tolerance)

    def test_state_and_output_use_each_native_volume(self):
        s, d, m = coupled_setup((8, 6, 1), (8, 6, 1), (0, 0, 0))
        D, B, J = random_vectors(m)
        scalar = jnp.zeros_like(D[0])
        original = (D, B, J, scalar, scalar, (D, B), m, (B, D), False)
        state = densitize_fields(original)
        for slot, samples in ((0, m.D), (1, m.B), (2, m.D)):
            self.assert_vectors_close(state[slot], tuple(
                value*sample.sqrt_gamma for value, sample in zip(original[slot], samples)))
        restored = physical_fields(state)
        for slot in (0, 1, 2):
            self.assert_vectors_close(restored[slot], original[slot])
        for a, b in zip(restored[7], original[7]):
            self.assert_vectors_close(a, b)
        self.assertIs(state[5], original[5])
        output = build_field_output_map(state, None, None, s, d)
        for label, expected in zip(('E', 'B', 'J'), (D, B, J)):
            self.assert_vectors_close(output[label], expected)
        # The assembled output also converts before gathering the tiled mesh.
        assembled = fields_for_output(state, s)
        for output_vector, expected in zip(assembled[:3], (D, B, J)):
            for a, b in zip(output_vector, expected):
                np.testing.assert_allclose(a, b[0, 0, 0, 2:-2, 2:-2, 2:-2], atol=3e-13)

    def test_constitutive_fields_match_physical_reconstruction(self):
        _, _, m = coupled_setup((8, 6, 1), (8, 6, 1), (0, 0, 0))
        D, B, _ = random_vectors(m)
        for magnetic, locations, samples, compute in (
                (False, D_FIELD_LOCATIONS, m.D, compute_covariant_E_densitized),
                (True, B_FIELD_LOCATIONS, m.B, compute_covariant_H_densitized)):
            expected = []
            for i, (location, sample) in enumerate(zip(locations, samples)):
                dd = jnp.stack(reconstruct_vector(D, D_FIELD_LOCATIONS, m, location,
                                                 preserve_native=False), axis=-1)
                bb = jnp.stack(reconstruct_vector(B, B_FIELD_LOCATIONS, m, location,
                                                 preserve_native=False), axis=-1)
                lower = jnp.sum(sample.gamma[..., i, :]*(bb if magnetic else dd), axis=-1)
                cross = jnp.cross(sample.shift, dd if magnetic else bb)[..., i]
                expected.append(sample.lapse*lower + (-1 if magnetic else 1)*sample.sqrt_gamma*cross)
            self.assert_vectors_close(compute(densitize_vector(D, m.D), densitize_vector(B, m.B), m), expected)

    def test_updates_match_original_physical_equations_and_boundaries(self):
        # Non-orthogonal conducting edges and an internal tile seam.
        s, d, m = coupled_setup((16, 8, 1), (8, 8, 1), (1, 1, 0))
        D, B, J = random_vectors(m)
        dt = .007
        for magnetic, locations, samples, initial, auxiliary in (
                (False, D_FIELD_LOCATIONS, m.D, D, B),
                (True, B_FIELD_LOCATIONS, m.B, B, D)):
            derivatives = curl(auxiliary, d, magnetic)
            expected = tuple(jnp.where(owned_nodes(value.shape, locations[i], s),
                value + dt*((-1 if magnetic else 1)*derivatives[i]/samples[i].sqrt_gamma
                            - (0 if magnetic else 4*jnp.pi*J[i])), value)
                for i, value in enumerate(initial))
            expected = refresh_fields(expected, s, locations, 'B' if magnetic else 'D', m)
            density = densitize_vector(initial, samples)
            actual = (update_B_densitized(auxiliary, density, m, s, d, dt) if magnetic else
                      update_D_densitized(density, auxiliary, densitize_vector(J, m.D), m, s, d, dt))
            self.assert_vectors_close(physical_vector(actual, samples), expected, 2e-12)

    def test_vacuum_curls_preserve_density_divergences(self):
        s, d, m = coupled_setup((16, 8, 1), (8, 8, 1), (0, 0, 0))
        D, B, _ = random_vectors(m)
        D = refresh_densitized_fields(densitize_vector(D, m.D), s, D_FIELD_LOCATIONS, 'D', m)
        B = refresh_densitized_fields(densitize_vector(B, m.B), s, B_FIELD_LOCATIONS, 'B', m)
        # Auxiliary fields must also have periodic, communicated halos.
        auxiliary = refresh_fields(random_vectors(m)[2], s, D_FIELD_LOCATIONS)
        zero = tuple(jnp.zeros_like(a) for a in D)
        next_D = update_D_densitized(D, auxiliary, zero, m, s, d, .001)
        auxiliary_B = refresh_fields(random_vectors(m)[2], s, B_FIELD_LOCATIONS)
        next_B = update_B_densitized(auxiliary_B, B, m, s, d, .001)
        interior = (slice(None),)*3+(slice(3, -3),)*3
        for before, after, forward in ((D, next_D, False), (B, next_B, True)):
            error = densitized_divergence(after, d, forward=forward)-densitized_divergence(before, d, forward=forward)
            np.testing.assert_allclose(error[interior], 0., rtol=0, atol=1e-11)

    def test_horizon_and_absorber_keep_physical_boundary_policy(self):
        s, d, m = coupled_setup((16, 8, 1), (8, 8, 1), (3, 1, 0))
        s = s._replace(horizon_field_cells=2, supergaussian_active=True,
                       supergaussian_layers=((0, 1, 3, 4., 10.),))
        D, B, _ = random_vectors(m)
        zero = tuple(jnp.zeros_like(a) for a in D)
        for magnetic, locations, samples, initial in (
                (False, D_FIELD_LOCATIONS, m.D, D),
                (True, B_FIELD_LOCATIONS, m.B, B)):
            expected = apply_tiled_supergaussian_absorber(initial, s, d, .03, locations=locations)
            expected = refresh_fields(expected, s, locations, 'B' if magnetic else 'D', m)
            density = densitize_vector(initial, samples)
            actual = (update_B_densitized(zero, density, m, s, d, .03) if magnetic else
                      update_D_densitized(density, zero, zero, m, s, d, .03))
            self.assert_vectors_close(physical_vector(actual, samples), expected)

    def test_new_volume_does_not_rescale_stored_density_in_zero_rhs(self):
        s, d, m = coupled_setup((8, 6, 1), (8, 6, 1), (0, 0, 0))
        D, B, _ = random_vectors(m)
        density_D, density_B = densitize_vector(D, m.D), densitize_vector(B, m.B)
        # A spatial metric scaled by four has an eightfold volume factor.
        def scale(sample):
            return sample._replace(gamma=sample.gamma*4, gamma_inv=sample.gamma_inv/4,
                                   sqrt_gamma=sample.sqrt_gamma*8)
        changed = m._replace(D=tuple(map(scale, m.D)), B=tuple(map(scale, m.B)),
                             center=scale(m.center), vertex=scale(m.vertex))
        zero = tuple(jnp.zeros_like(a) for a in D)
        next_D = update_D_densitized(density_D, zero, zero, changed, s, d, .01)
        next_B = update_B_densitized(zero, density_B, changed, s, d, .01)
        interior = (slice(None),)*3+(slice(3, -3),)*3
        for actual, original in ((next_D, density_D), (next_B, density_B)):
            self.assert_vectors_close(tuple(a[interior] for a in actual),
                                      tuple(a[interior] for a in original))
        self.assert_vectors_close(tuple(a[interior] for a in physical_vector(next_D, changed.D)),
                                  tuple(a[interior]/8 for a in D))

    def test_esirkepov_source_preserves_gauss_in_curved_metric(self):
        s, d, m = coupled_setup((8, 6, 1), (8, 6, 1), (0, 0, 0))
        s = s._replace(current_deposition='GR_esirkepov', current_filter='none')
        d = d._replace(dt=.02)
        x = jnp.array([.43, .47, .5]).reshape(1, 1, 1, 1, 1, 3)
        old = TiledParticles(x, jnp.zeros_like(x), jnp.ones(x.shape[:-1], dtype=bool))
        new = old._replace(x=x+jnp.array([.01, -.02, 0.]))
        species = SpeciesConfig(jnp.ones(1), jnp.ones(1), jnp.ones(1),
                                jnp.array([[True, True, False]]))
        scalar = jnp.zeros_like(m.center.sqrt_gamma)
        zero = (scalar,)*3
        current = GR_Esirkepov_densitized_current(old, new, species, zero, m, s, d)
        physical = GR_Esirkepov_current(old, new, species, zero, m, s, d)
        self.assert_vectors_close(physical_vector(current, m.D), physical)
        updated = update_D_densitized(zero, zero, current, m, s, d, d.dt)
        rho_old = compute_rho(old, species, scalar, s, d)
        rho_new = compute_rho(new, species, scalar, s, d)
        drift = densitized_divergence(updated, d)-4*jnp.pi*(rho_new-rho_old)
        interior = (slice(None),)*3+(slice(3, -3),)*3
        np.testing.assert_allclose(drift[interior], 0., rtol=0., atol=1e-11)

    def test_time_loop_centers_densitized_current_and_pushes_physical_fields(self):
        s, d, m = coupled_setup((8, 6, 1), (8, 6, 1), (0, 0, 0))
        D, B, old_J = random_vectors(m)
        scalar = jnp.zeros_like(D[0])
        new_J = tuple(a*.25 for a in old_J)
        state = densitize_fields((D, B, old_J, scalar, scalar, (D, B), m, (D, B), False))
        captured = []
        def push(particles, species, DD, BB, *args):
            self.assert_vectors_close(DD, tuple(2*a for a in D))
            self.assert_vectors_close(BB, tuple(2*a for a in B))
            return particles, particles
        def update_D(DD, H, JJ, *args):
            captured.append(JJ)
            return DD
        def transform(JJ):
            self.assert_vectors_close(JJ, new_J)
            return tuple(2*a for a in JJ)
        module = 'PyPIC3D.solvers.gr_static.time_loop.'
        with patch(module+'hybrid_boris_geodesic_push', side_effect=push), \
             patch(module+'GR_direct_deposition', return_value=new_J), \
             patch(module+'refresh_tiled_particle_tiles', side_effect=lambda p,*a:(p,False)), \
             patch(module+'update_B_densitized', side_effect=lambda E,B,*a:B), \
             patch(module+'update_D_densitized', side_effect=update_D):
            _, result = time_loop_static_metric(None, None, state, s, d, current_transform=transform)
        new_density = densitize_vector(tuple(2*a for a in new_J), m.D)
        self.assert_vectors_close(captured[0], tuple((a+b)/2 for a,b in zip(state[2], new_density)))
        self.assert_vectors_close(captured[1], new_density)
        self.assert_vectors_close(result[2], new_density)
        self.assertIs(result[7][0], state[0])
        self.assertIs(result[7][1], state[1])


if __name__ == '__main__':
    unittest.main()
