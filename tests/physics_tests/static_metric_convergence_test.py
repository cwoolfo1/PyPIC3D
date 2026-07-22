import math
import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.pusher.hybrid_boris_geodesic import hybrid_boris_geodesic_push
from PyPIC3D.relativity.flat import initialize_flat_cartesian_metric
from PyPIC3D.solvers.static_metric import update_B_relativity
from tests.kernel_fixtures import empty_tiled_vector, kernel_parameters


jax.config.update("jax_enable_x64", True)


class TestStaticMetricConvergence(unittest.TestCase):
    def test_flat_vacuum_update_B_converges_for_smooth_mode(self):
        errors = []
        for Nx in (32, 64, 128):
            static_parameters, dynamic_parameters = kernel_parameters(
                Nx=Nx,
                Ny=1,
                Nz=1,
                x_wind=2.0 * math.pi,
                y_wind=1.0,
                z_wind=1.0,
                dx=2.0 * math.pi / Nx,
                dy=1.0,
                dz=1.0,
                dt=1.0e-3,
                tile_shape=(Nx, 1, 1),
                solver="static_metric",
                current_deposition="GR_direct",
                particle_pusher="hybrid_boris_geodesic",
            )
            metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
            D = empty_tiled_vector(static_parameters, dynamic_parameters)
            B = empty_tiled_vector(static_parameters, dynamic_parameters)
            Dx, Dy, Dz = D

            g = int(static_parameters.guard_cells)
            active = slice(g, -g)
            x_center = dynamic_parameters.grids.tiled_center_grid[0][:, :, :, active]
            Dz_values = jnp.sin(x_center)[:, :, :, :, jnp.newaxis, jnp.newaxis]
            Dz = Dz.at[:, :, :, active, active, active].set(Dz_values)

            _Bx, By, _Bz = update_B_relativity((Dx, Dy, Dz), B, metric, static_parameters, dynamic_parameters, dynamic_parameters.dt)

            x_vertex = dynamic_parameters.grids.tiled_vertex_grid[0][:, :, :, active]
            exact_By = dynamic_parameters.dt * jnp.cos(x_vertex)
            diff = By[:, :, :, active, active, active] - exact_By[:, :, :, :, jnp.newaxis, jnp.newaxis]
            errors.append(float(jnp.sqrt(jnp.mean(diff**2))))

        first_order = math.log(errors[0] / errors[1], 2.0)
        second_order = math.log(errors[1] / errors[2], 2.0)

        self.assertGreater(first_order, 1.8)
        self.assertGreater(second_order, 1.8)

    def test_fixed_field_hybrid_particle_pusher_self_refines(self):
        def run_with_dt(dt):
            static_parameters, dynamic_parameters = kernel_parameters(
                Nx=8,
                Ny=8,
                Nz=8,
                x_wind=8.0,
                y_wind=8.0,
                z_wind=8.0,
                dt=dt,
                solver="static_metric",
                current_deposition="GR_direct",
                particle_pusher="hybrid_boris_geodesic",
            )
            metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
            D = empty_tiled_vector(static_parameters, dynamic_parameters)
            B = empty_tiled_vector(static_parameters, dynamic_parameters)
            D = (D[0].at[:, :, :, :, :, :].set(0.05), D[1], D[2])
            B = (B[0], B[1], B[2].at[:, :, :, :, :, :].set(0.25))

            particles = TiledParticles(
                x=jnp.zeros((1, 1, 1, 1, 1, 3)),
                u=jnp.asarray((0.1, 0.03, 0.0), dtype=float).reshape((1, 1, 1, 1, 1, 3)),
                active=jnp.ones((1, 1, 1, 1, 1), dtype=bool),
            )
            species = SpeciesConfig(
                charge=jnp.asarray([1.0]),
                mass=jnp.asarray([1.0]),
                weight=jnp.asarray([1.0]),
                update_x=jnp.asarray([[True, True, True]]),
                update_u=jnp.asarray([[True, True, True]]),
            )

            n_steps = int(round(0.2 / dt))
            for _ in range(n_steps):
                particles, _centered = hybrid_boris_geodesic_push(
                    particles,
                    species,
                    D,
                    B,
                    metric,
                    static_parameters,
                    dynamic_parameters,
                )
            return particles.x[0, 0, 0, 0, 0], particles.u[0, 0, 0, 0, 0]

        x_1, u_1 = run_with_dt(0.05)
        x_2, u_2 = run_with_dt(0.025)
        x_4, u_4 = run_with_dt(0.0125)
        x_8, u_8 = run_with_dt(0.00625)

        error_1 = jnp.linalg.norm(x_1 - x_8) + jnp.linalg.norm(u_1 - u_8)
        error_2 = jnp.linalg.norm(x_2 - x_8) + jnp.linalg.norm(u_2 - u_8)
        error_4 = jnp.linalg.norm(x_4 - x_8) + jnp.linalg.norm(u_4 - u_8)

        first_order = math.log(float(error_1 / error_2), 2.0)
        second_order = math.log(float(error_2 / error_4), 2.0)

        self.assertGreater(first_order, 1.5)
        self.assertGreater(second_order, 1.5)


if __name__ == "__main__":
    unittest.main()
