import unittest
import jax.numpy as jnp

from PyPIC3D.particles.species_class import particle_species
from PyPIC3D.particles.flat_particles import to_flat_particles, check_flat_compat


class TestFlatParticleBoundaryConditions(unittest.TestCase):
    def _make_species(self, x1, v1, x_bc="reflecting", y_bc="periodic", z_bc="periodic", dt=1):
        return particle_species(
            name="test",
            N_particles=len(x1),
            charge=1.0,
            mass=1.0,
            weight=1,
            T=100.0,
            v1=jnp.array(v1),
            v2=jnp.array([0.0] * len(x1)),
            v3=jnp.array([0.0] * len(x1)),
            x1=jnp.array(x1),
            x2=jnp.array([0.0] * len(x1)),
            x3=jnp.array([0.0] * len(x1)),
            dx=1.0,
            dy=1.0,
            dz=1.0,
            xwind=10.0,
            ywind=10.0,
            zwind=10.0,
            x_bc=x_bc,
            y_bc=y_bc,
            z_bc=z_bc,
            dt=dt,
        )

    def test_reflecting_boundary_flips_velocity_in_flat_backend(self):
        species = self._make_species(x1=[6.0, -6.0, 0.0], v1=[1.0, -2.0, 3.0], x_bc="reflecting")
        self.assertTrue(check_flat_compat([species]))

        flat = to_flat_particles([species])[0]
        flat.boundary_conditions()

        self.assertTrue(jnp.allclose(flat.x1, jnp.array([6.0, -6.0, 0.0])))
        self.assertTrue(jnp.allclose(flat.v1, jnp.array([-1.0, 2.0, 3.0])))

    def test_get_position_does_not_wrap_for_non_periodic_bc(self):
        species = self._make_species(x1=[7.0], v1=[1.0], x_bc="reflecting", dt=2)
        flat = to_flat_particles([species])[0]

        x_back, _, _ = flat.get_position()
        self.assertTrue(jnp.allclose(x_back, jnp.array([6.0])))


if __name__ == "__main__":
    unittest.main()
