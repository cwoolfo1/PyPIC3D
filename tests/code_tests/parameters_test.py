import unittest

import jax

from PyPIC3D.boundary_conditions.grid_and_stencil import (
    BC_ABSORBING,
    BC_CONDUCTING,
    BC_PERIODIC,
)
from PyPIC3D.deposition.Esirkepov import Esirkepov_current
from PyPIC3D.deposition.J_from_rhov import J_from_rhov
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.utilities.parameters import (
    DynamicParameters,
    GridParameters,
    StaticParameters,
    build_dynamic_parameters,
    build_static_parameters,
    dynamic_parameters_for_output,
    static_parameters_for_output,
)
from tests.kernel_fixtures import kernel_parameters


class TestKernelParameters(unittest.TestCase):
    def test_static_and_dynamic_parameters_split_kernel_contract(self):
        static_parameters, dynamic_parameters = kernel_parameters(
            dt=0.1,
            dx=0.25,
            dy=0.5,
            dz=1.0,
            Nx=4,
            Ny=2,
            Nz=1,
            x_wind=1.0,
            y_wind=1.0,
            z_wind=1.0,
            shape_factor=1,
            guard_cells=2,
            tile_shape=(4, 2, 1),
            current_deposition="direct",
            current_filter="none",
            particle_boundary_conditions=(BC_PERIODIC, BC_CONDUCTING, BC_ABSORBING),
            relativistic=False,
            C=1.0,
            eps=2.0,
            mu=3.0,
            kb=4.0,
            alpha=0.5,
        )

        self.assertIsInstance(static_parameters, StaticParameters)
        self.assertEqual(static_parameters.current_deposition, "direct")
        self.assertEqual(static_parameters.current_filter, "none")
        self.assertEqual(static_parameters.particle_pusher, "boris")
        self.assertEqual(static_parameters.tile_shape, (4, 2, 1))
        self.assertEqual(static_parameters.boundary_conditions, (0, 0, 0))
        self.assertEqual(
            static_parameters.particle_boundary_conditions,
            (BC_PERIODIC, BC_CONDUCTING, BC_ABSORBING),
        )
        self.assertNotIn("particle_species_names", static_parameters._asdict())
        self.assertNotIn("particle_species_metadata", static_parameters._asdict())
        self.assertFalse(static_parameters.fmr_enabled)
        self.assertEqual(static_parameters.fmr_levels, ())
        self.assertIsInstance(hash(static_parameters), int)
        with self.assertRaises(TypeError):
            static_parameters["current_deposition"]

        self.assertIsInstance(dynamic_parameters, DynamicParameters)
        self.assertIsInstance(dynamic_parameters.grids, GridParameters)
        self.assertNotIn("current_deposition", dynamic_parameters._asdict())
        self.assertNotIn("current_filter", dynamic_parameters._asdict())
        self.assertNotIn("field_mesh", dynamic_parameters._asdict())
        self.assertIsNone(dynamic_parameters.fmr)
        self.assertAlmostEqual(float(dynamic_parameters.dt), 0.1)
        self.assertAlmostEqual(float(dynamic_parameters.C), 1.0)
        with self.assertRaises(TypeError):
            dynamic_parameters["dt"]

        flattened, _ = jax.tree_util.tree_flatten(dynamic_parameters)
        self.assertTrue(all(hasattr(leaf, "shape") for leaf in flattened))

    def test_fmr_parameters_are_built_but_large_metadata_is_not_serialized(self):
        static_parameters, dynamic_parameters = kernel_parameters(
            Nx=4,
            Ny=4,
            Nz=4,
            tile_shape=(4, 4, 4),
        )
        fmr_levels = ((0, 2, (1, 1, 1), (3, 3, 3)),)
        fmr_data = {"weights": jax.numpy.ones((4, 8))}

        static_config = static_parameters._asdict()
        static_config.update(
            fmr_enabled=True,
            fmr_levels=fmr_levels,
        )
        dynamic_config = dynamic_parameters._asdict()
        dynamic_config["fmr"] = fmr_data

        static_parameters = build_static_parameters(static_config)
        dynamic_parameters = build_dynamic_parameters(dynamic_config)

        self.assertTrue(static_parameters.fmr_enabled)
        self.assertEqual(static_parameters.fmr_levels, fmr_levels)
        self.assertNotIn("fmr_interpolation_order", static_parameters._fields)
        self.assertIsInstance(hash(static_parameters), int)
        self.assertIs(dynamic_parameters.fmr, fmr_data)

        static_output = static_parameters_for_output(static_parameters)
        dynamic_output = dynamic_parameters_for_output(dynamic_parameters)
        self.assertTrue(static_output["fmr_enabled"])
        self.assertNotIn("fmr_levels", static_output)
        self.assertNotIn("fmr", dynamic_output)

    def test_public_deposition_methods_are_jitted_static_parameter_boundaries(self):
        self.assertTrue(hasattr(J_from_rhov, "lower"))
        self.assertTrue(hasattr(Esirkepov_current, "lower"))
        self.assertTrue(hasattr(compute_rho, "lower"))


if __name__ == "__main__":
    unittest.main()
