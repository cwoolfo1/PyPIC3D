import math
import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.pusher.hybrid_boris_geodesic import hybrid_boris_geodesic_push
from PyPIC3D.relativity.flat import initialize_flat_cartesian_metric
from PyPIC3D.relativity.kerr_schild import initialize_kerr_schild_cartesian_metric
from PyPIC3D.solvers.static_metric import update_B_relativity
from tests.kernel_fixtures import empty_tiled_vector, kernel_parameters


jax.config.update("jax_enable_x64", True)


PUSHER_DT_LEVELS = (0.04, 0.02, 0.01, 0.005)
PUSHER_FINAL_TIME = 0.2


def _hybrid_pusher_parameters(dt, Nx=24, x_wind=24.0):
    return kernel_parameters(
        Nx=Nx,
        Ny=Nx,
        Nz=Nx,
        x_wind=x_wind,
        y_wind=x_wind,
        z_wind=x_wind,
        dt=dt,
        tile_shape=(Nx, Nx, Nx),
        solver="static_metric",
        current_deposition="GR_direct",
        particle_pusher="hybrid_boris_geodesic",
    )


def _single_particle_species(charge):
    return SpeciesConfig(
        charge=jnp.asarray([charge]),
        mass=jnp.asarray([1.0]),
        weight=jnp.asarray([1.0]),
        update_x=jnp.asarray([[True, True, True]]),
    )


def _single_particle(x, u):
    return TiledParticles(
        x=x.reshape((1, 1, 1, 1, 1, 3)),
        u=u.reshape((1, 1, 1, 1, 1, 3)),
        active=jnp.ones((1, 1, 1, 1, 1), dtype=bool),
    )


def _constant_tiled_vector(static_parameters, dynamic_parameters, values):
    vector = empty_tiled_vector(static_parameters, dynamic_parameters)
    return tuple(vector[i].at[:, :, :, :, :, :].set(values[i]) for i in range(3))


def _velocity_update_with_production_pusher(
    position,
    u_cov,
    charge,
    D,
    B,
    metric,
    static_parameters,
    dynamic_parameters,
    dt,
):
    particles = _single_particle(position, u_cov)
    velocity_update_species = SpeciesConfig(
        charge=jnp.asarray([charge]),
        mass=jnp.asarray([1.0]),
        weight=jnp.asarray([1.0]),
        update_x=jnp.asarray([[True, True, True]]),
    )
    velocity_only_dynamic_parameters = dynamic_parameters._replace(dt=jnp.asarray(dt))

    particles, _centered = hybrid_boris_geodesic_push(
        particles,
        velocity_update_species,
        D,
        B,
        metric,
        static_parameters,
        velocity_only_dynamic_parameters,
    )
    return particles.u[0, 0, 0, 0, 0]


def _run_hybrid_pusher_trajectory(
    dt,
    metric_initializer,
    charge,
    x_0,
    u_0,
    D_values=(0.0, 0.0, 0.0),
    B_values=(0.0, 0.0, 0.0),
    Nx=24,
    x_wind=24.0,
):
    static_parameters, dynamic_parameters = _hybrid_pusher_parameters(
        dt,
        Nx=Nx,
        x_wind=x_wind,
    )
    metric = metric_initializer(static_parameters, dynamic_parameters)
    D = _constant_tiled_vector(static_parameters, dynamic_parameters, D_values)
    B = _constant_tiled_vector(static_parameters, dynamic_parameters, B_values)

    x_0 = jnp.asarray(x_0, dtype=float)
    u_0 = jnp.asarray(u_0, dtype=float)

    u_n_minushalf = _velocity_update_with_production_pusher(
        x_0,
        u_0,
        charge,
        D,
        B,
        metric,
        static_parameters,
        dynamic_parameters,
        -0.5 * dt,
    )
    # The pusher stores u at half time steps.  Initialize every dt from the
    # same physical u(t=0), shifted back to u^{-1/2} by the same split map.
    particles = _single_particle(x_0, u_n_minushalf)
    species = _single_particle_species(charge)

    n_steps = int(round(PUSHER_FINAL_TIME / dt))
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

    x_final = particles.x[0, 0, 0, 0, 0]
    u_final = particles.u[0, 0, 0, 0, 0]
    u_at_T = _velocity_update_with_production_pusher(
        x_final,
        u_final,
        charge,
        D,
        B,
        metric,
        static_parameters,
        dynamic_parameters,
        0.5 * dt,
    )
    # Recenter the final u^{N-1/2} to the common physical time T before
    # comparing across refinements.
    return x_final, u_at_T


def _hybrid_pusher_self_convergence_orders(run_with_dt):
    x_1, u_1 = run_with_dt(PUSHER_DT_LEVELS[0])
    x_2, u_2 = run_with_dt(PUSHER_DT_LEVELS[1])
    x_4, u_4 = run_with_dt(PUSHER_DT_LEVELS[2])
    x_8, u_8 = run_with_dt(PUSHER_DT_LEVELS[3])

    error_1 = jnp.linalg.norm(x_1 - x_8) + jnp.linalg.norm(u_1 - u_8)
    error_2 = jnp.linalg.norm(x_2 - x_8) + jnp.linalg.norm(u_2 - u_8)
    error_4 = jnp.linalg.norm(x_4 - x_8) + jnp.linalg.norm(u_4 - u_8)

    first_order = math.log(float(error_1 / error_2), 2.0)
    second_order = math.log(float(error_2 / error_4), 2.0)
    return first_order, second_order, (float(error_1), float(error_2), float(error_4))


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

    def test_flat_cartesian_hybrid_particle_pusher_self_refines(self):
        def run_with_dt(dt):
            return _run_hybrid_pusher_trajectory(
                dt=dt,
                metric_initializer=initialize_flat_cartesian_metric,
                charge=1.0,
                x_0=(0.0, 0.0, 0.0),
                u_0=(0.1, 0.03, 0.0),
                D_values=(0.05, 0.0, 0.0),
                B_values=(0.0, 0.0, 0.25),
                Nx=8,
                x_wind=8.0,
            )

        first_order, second_order, errors = _hybrid_pusher_self_convergence_orders(run_with_dt)
        message = f"errors={errors}, orders=({first_order}, {second_order})"
        self.assertGreater(first_order, 1.8, message)
        self.assertGreater(second_order, 1.8, message)

    def test_schwarzschild_no_field_hybrid_particle_pusher_self_refines(self):
        def schwarzschild_metric(static_parameters, dynamic_parameters):
            return initialize_kerr_schild_cartesian_metric(
                static_parameters,
                dynamic_parameters,
                mass=1.0,
                spin=0.0,
            )

        def run_with_dt(dt):
            return _run_hybrid_pusher_trajectory(
                dt=dt,
                metric_initializer=schwarzschild_metric,
                charge=0.0,
                x_0=(6.0, 1.0, 0.0),
                u_0=(0.02, 0.01, 0.005),
            )

        first_order, second_order, errors = _hybrid_pusher_self_convergence_orders(run_with_dt)
        message = f"errors={errors}, orders=({first_order}, {second_order})"
        self.assertGreater(first_order, 1.8, message)
        self.assertGreater(second_order, 1.8, message)

    def test_kerr_schild_uniform_B_hybrid_particle_pusher_self_refines(self):
        def kerr_schild_metric(static_parameters, dynamic_parameters):
            return initialize_kerr_schild_cartesian_metric(
                static_parameters,
                dynamic_parameters,
                mass=1.0,
                spin=0.5,
            )

        def run_with_dt(dt):
            return _run_hybrid_pusher_trajectory(
                dt=dt,
                metric_initializer=kerr_schild_metric,
                charge=1.0,
                x_0=(6.0, 1.2, 0.0),
                u_0=(0.02, 0.01, 0.04),
                B_values=(0.0, 0.015, 0.01),
            )

        first_order, second_order, errors = _hybrid_pusher_self_convergence_orders(run_with_dt)
        message = f"errors={errors}, orders=({first_order}, {second_order})"
        self.assertGreater(first_order, 1.8, message)
        self.assertGreater(second_order, 1.8, message)


if __name__ == "__main__":
    unittest.main()
