import unittest
import jax
import jax.numpy as jnp

from PyPIC3D.metric import build_metric_from_parameters
from PyPIC3D.solvers.GR_fields import recover_E_H_from_metric, update_DB_and_recover_EH
from PyPIC3D.utils import build_yee_grid


jax.config.update("jax_enable_x64", True)


class TestGRFieldsMethods(unittest.TestCase):
    def setUp(self):
        self.world = {
            'Nx': 48,
            'Ny': 24,
            'Nz': 16,
            'dx': 0.02,
            'dy': 0.02,
            'dz': 0.02,
            'dt': 0.01,
            'x_wind': 0.96,
            'y_wind': 0.48,
            'z_wind': 0.32,
            'boundary_conditions': {'x': 0, 'y': 0, 'z': 0},
        }
        B_grid, E_grid = build_yee_grid(self.world)
        self.world['grids'] = {'center': B_grid, 'vertex': E_grid}
        self.world['metric'] = build_metric_from_parameters({"metric": "minkowski"})
        self.constants = {
            'eps': 1.0,
            'mu': 1.0,
            'C': 1.0,
            'alpha': 1.0,
        }

    def test_recover_EH_minkowski(self):
        Nx, Ny, Nz = self.world["Nx"], self.world["Ny"], self.world["Nz"]
        D = (
            jnp.ones((Nx, Ny, Nz)),
            2.0 * jnp.ones((Nx, Ny, Nz)),
            3.0 * jnp.ones((Nx, Ny, Nz)),
        )
        B = (
            4.0 * jnp.ones((Nx, Ny, Nz)),
            5.0 * jnp.ones((Nx, Ny, Nz)),
            6.0 * jnp.ones((Nx, Ny, Nz)),
        )
        E, H = recover_E_H_from_metric(D, B, self.world)
        self.assertTrue(jnp.allclose(E[0], D[0]))
        self.assertTrue(jnp.allclose(E[1], D[1]))
        self.assertTrue(jnp.allclose(E[2], D[2]))
        self.assertTrue(jnp.allclose(H[0], B[0]))
        self.assertTrue(jnp.allclose(H[1], B[1]))
        self.assertTrue(jnp.allclose(H[2], B[2]))

    def test_update_DB_no_source_still_zero(self):
        Nx, Ny, Nz = self.world["Nx"], self.world["Ny"], self.world["Nz"]
        zeros = jnp.zeros((Nx, Ny, Nz))
        D = (zeros, zeros, zeros)
        B = (zeros, zeros, zeros)
        J = (zeros, zeros, zeros)
        E, Bout, Dout, H = update_DB_and_recover_EH(
            D, B, J, self.world, self.constants, curl_func=None
        )
        for comp in E + Bout + Dout + H:
            self.assertTrue(jnp.allclose(comp, zeros))


if __name__ == "__main__":
    unittest.main()
