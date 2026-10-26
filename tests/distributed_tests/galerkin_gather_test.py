"""Production mixed-order gathers agree across tile seams and periodic walls."""
import unittest

import jax
import jax.numpy as jnp

from tests.kernel_fixtures import particle_species
from tests.support.pusher_fixtures import TiledParticlePusherFixtures


@unittest.skipUnless(jax.device_count() >= 8, "Requires eight CPU or accelerator devices")
class DistributedGalerkinGatherTests(TiledParticlePusherFixtures, unittest.TestCase):
    def test_both_pushers_match_one_tile_at_seams_and_periodic_walls(self):
        for order in (1, 2):
            values = self._build_parameter_values(shape_factor=order)
            values.update(current_deposition="esirkepov", field_gather="galerkin",
                          particle_batch_size=2)
            tiled = self._with_tiled_grids(values, (4, 3, 2), g=2)
            whole = self._copy_parameters_for_tile_shape(tiled, (8, 6, 4), 2)
            E = self._deterministic_vector_field(values, 1.)
            B = self._deterministic_vector_field(values, .3)
            species = particle_species(
                name="seam particles", charge=-1., mass=2., weight=.5,
                x1=jnp.asarray([-1.99, -.01, .01, 1.99]),
                x2=jnp.asarray([-1.49, .01, -.01, 1.49]),
                x3=jnp.asarray([-.99, -.01, .01, .99]),
                v1=jnp.asarray([.2, -.1, .05, .3]),
                v2=jnp.asarray([.1, .15, -.2, .1]),
                v3=jnp.asarray([-.05, .25, .1, -.15]))
            for pusher in ("boris", "higuera_cary"):
                with self.subTest(order=order, pusher=pusher):
                    actual = self._push_tiled_species(
                        species, tiled, (4, 3, 2), E, B, {"C": 10.}, particle_pusher=pusher)
                    expected = self._push_tiled_species(
                        species, whole, (8, 6, 4), E, B, {"C": 10.}, particle_pusher=pusher)
                    actual_x, actual_u = self._flatten_active_by_position(actual)
                    expected_x, expected_u = self._flatten_active_by_position(expected)
                    # Compare both decompositions on one device after the distributed push.
                    actual_x = jax.device_put(actual_x, jax.devices()[0])
                    actual_u = jax.device_put(actual_u, jax.devices()[0])
                    self.assertTrue(bool(jnp.array_equal(actual_x, expected_x)))
                    self.assertTrue(bool(jnp.allclose(actual_u, expected_u, rtol=2e-13, atol=2e-13)))


if __name__ == "__main__":
    unittest.main()
