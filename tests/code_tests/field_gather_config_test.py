"""Static gather selection and output coordinates follow the production grids."""
import unittest
from types import SimpleNamespace

import jax.numpy as jnp

from PyPIC3D.diagnostics.openPMD import _set_mesh_position
from PyPIC3D.pusher.boris import field_component_orders, interpolate_field_to_particles
from PyPIC3D.relativity.core import B_FIELD_LOCATIONS, D_FIELD_LOCATIONS, location_grid
from PyPIC3D.utilities.parameters import (
    build_static_parameters,
    resolve_field_gather,
    static_parameters_for_output,
)
from tests.kernel_fixtures import kernel_parameters, kernel_parameters_from_values


class FieldGatherConfigurationTests(unittest.TestCase):
    def test_constant_fields_for_all_component_orders_and_dimensions(self):
        center = tuple(jnp.arange(7, dtype=float) * spacing + origin
                       for spacing, origin in ((.25, -1.3), (.5, 2.1), (.125, -.7)))
        vertex = tuple(axis + (axis[1] - axis[0]) / 2 for axis in center)
        position = jnp.stack(tuple(axis[jnp.asarray([2, 3])] + .037 for axis in center), axis=-1)
        field = jnp.full((7, 7, 7), 2.3)
        for p in (1, 2):
            electric, magnetic = field_component_orders(p, "galerkin")
            for location, orders in zip(D_FIELD_LOCATIONS + B_FIELD_LOCATIONS, electric + magnetic):
                for active in ((True, False, False), (True, False, True), (True, True, True)):
                    with self.subTest(p=p, location=location, active=active):
                        result = interpolate_field_to_particles(
                            field, *position.T, location_grid(center, vertex, location), p,
                            active_axes=active, inactive_axis_indices=(2, 2, 2), axis_orders=orders)
                        self.assertTrue(bool(jnp.allclose(result, 2.3, rtol=2e-14, atol=2e-14)))

    def test_selection_and_recorded_configuration(self):
        for solver in ("electrostatic", "electrodynamic_yee", "dark_matter_yee",
                       "static_metric", "dynamic_metric"):
            for deposition in ("direct", "GR_direct", "esirkepov"):
                compatible = solver == "electrostatic" or deposition == "esirkepov"
                expected = "galerkin" if compatible else "uniform"
                with self.subTest(solver=solver, deposition=deposition):
                    self.assertEqual(resolve_field_gather(solver, deposition), expected)
                    self.assertEqual(resolve_field_gather(solver, deposition, "uniform"), "uniform")
                    static, _ = kernel_parameters(solver=solver, current_deposition=deposition,
                                                  guard_cells=3)
                    self.assertEqual(static_parameters_for_output(static)["field_gather"], expected)
                    built = build_static_parameters(static._asdict())
                    self.assertEqual(built.field_gather, expected)
                    if compatible:
                        self.assertEqual(resolve_field_gather(solver, deposition, "galerkin"), "galerkin")
                    else:
                        with self.assertRaisesRegex(ValueError, "Esirkepov"):
                            resolve_field_gather(solver, deposition, "galerkin")

    def test_invalid_gather_and_insufficient_electrostatic_halos(self):
        with self.assertRaisesRegex(ValueError, "field_gather"):
            resolve_field_gather("electrostatic", "direct", "invalid")
        static, _ = kernel_parameters(solver="electrostatic", guard_cells=1)
        with self.assertRaisesRegex(ValueError, "guard_cells >= 2"):
            build_static_parameters(static._asdict())
        legacy = build_static_parameters(static._replace(field_gather="uniform")._asdict())
        self.assertEqual(legacy.guard_cells, 1)

    def test_native_component_output_positions_with_shifted_origin(self):
        for gather in ("auto", "uniform", "galerkin"):
            for cells in ((8, 1, 6), (1, 1, 8), (4, 3, 2)):
                with self.subTest(gather=gather, cells=cells):
                    _, dynamic = kernel_parameters_from_values(dict(
                        Nx=cells[0], Ny=cells[1], Nz=cells[2],
                        x_min=3.25, y_min=-2.5, z_min=7.125,
                        dx=0.25, dy=0.5, dz=0.125,
                        solver="electrostatic", field_gather=gather))
                    active = tuple(n > 1 for n in cells)
                    for name in ("E", "J", "B"):
                        for component in range(3):
                            record = SimpleNamespace()
                            _set_mesh_position(record, name, component, active, dynamic)
                            expected = [0.5 * ((axis != component) if name == "B" else
                                               (axis == component))
                                        if gather != "uniform" else 0.0
                                        for axis in range(3) if active[axis]]
                            self.assertTrue(bool(jnp.allclose(jnp.asarray(record.position),
                                                              jnp.asarray(expected), atol=1e-14)))


if __name__ == "__main__":
    unittest.main()
