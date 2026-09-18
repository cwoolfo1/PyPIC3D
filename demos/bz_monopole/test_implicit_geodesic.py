"""Independent straight-line orbit near a spherical coordinate pole."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import checkify

from demos.bz_monopole.simulation_parameters import SimulationParameters, build_runtime
from PyPIC3D.relativity.flat import initialize_flat_spherical_metric
from PyPIC3D.pusher.hybrid_boris_geodesic import (
    hybrid_boris_geodesic_push, _implicit_velocity_newton, geodesic_velocity,
    _implicit_position_newton, GR_position_update)
from PyPIC3D.relativity.core import Metric
from PyPIC3D.pusher.particle_push import seed_leapfrog_velocity
from tests.physics_tests.static_metric_validation_test import one_particle, one_species

jax.config.update('jax_enable_x64', True)


def flat_spherical_sample(q, derivatives):
    r, t, _ = q
    gamma = jnp.diag(jnp.array([1., r*r, (r*jnp.sin(t))**2]))
    inverse = jnp.linalg.inv(gamma)
    grad = jnp.zeros((3,3,3)).at[0,1,1].set(-2/r**3)
    grad = grad.at[0,2,2].set(-2/(r**3*jnp.sin(t)**2))
    grad = grad.at[1,2,2].set(-2*jnp.cos(t)/(r*r*jnp.sin(t)**3))
    metric = Metric(jnp.array(1.), jnp.zeros(3), gamma, inverse,
                    jnp.sqrt(jnp.linalg.det(gamma)), jnp.zeros((3,3,3)),
                    jnp.zeros(3), jnp.zeros((3,3)))
    return metric, grad, jnp.isfinite(inverse).all() & (r > 0) & (jnp.abs(t) > 1e-14)


class TestImplicitGeodesic(unittest.TestCase):
    def test_cyclic_chart_retry_matches_bracketed_root_and_preserves_inactive_lanes(self):
        from scipy.optimize import brentq
        old = jnp.array([5.69456, 4.77676e-5, .50043])
        momentum = jnp.array([.62288, -7.93411, -2.78398e-5])
        dt = .001
        spacing = jnp.array([.14, jnp.pi/64, 2*jnp.pi])
        def sample(q, derivatives):
            metric, gradient, valid = flat_spherical_sample(q, derivatives)
            return metric, gradient, valid & (q[1] > 0) & (q[1] < jnp.pi)
        def residual(q):
            mid = .5*(old+q)
            return (q-old-dt*GR_position_update(mid, momentum, sample(mid, False)[0]))/spacing
        error = jax.jit(residual)
        # Nested scalar brackets do not share the production Newton method.
        def radius(theta):
            return brentq(lambda r: float(error(jnp.array([r, theta, old[2]]))[0]),
                          float(old[0]-.01), float(old[0]+.01), xtol=1e-13)
        theta = brentq(lambda t: float(error(jnp.array([radius(t), t, old[2]]))[1]),
                       -float(old[1])+1e-10, float(old[1]), xtol=1e-15)
        expected = jnp.array([radius(theta), theta, old[2]])
        mid = .5*(old+expected)
        expected = expected.at[2].set((old+dt*GR_position_update(
            mid, momentum, sample(mid, False)[0]))[2])
        south = old.at[1].set(jnp.pi-old[1])
        positions = jnp.stack([old, south, old])
        momenta = jnp.stack([momentum, momentum.at[1].multiply(-1), momentum])
        for dynamic in (False, True):
            initial = jnp.stack([old, south, old+jnp.array([.01, .001, 1.])])
            active = jnp.array([True, True, False])
            actual = jax.jit(lambda guess: _implicit_position_newton(
                positions, guess, momenta,
                lambda q, derivatives: jax.vmap(lambda x: sample(x, derivatives))(q),
                dt, 20, active, jnp.ones((3,3), bool), spacing,
                dynamic, cyclic_coordinate=2))(initial)
            self.assertLess(float(jnp.max(jnp.abs(error(actual[0])))), 1e-8)
            np.testing.assert_allclose(actual[0], expected, rtol=1e-7, atol=1e-8)
            np.testing.assert_allclose(actual[1], expected.at[1].set(jnp.pi-expected[1]),
                                       rtol=1e-7, atol=1e-8)
            np.testing.assert_array_equal(actual[2], initial[2])

    def test_close_axis_orbit_has_separate_space_and_time_convergence(self):
        # Exact Cartesian straight line, passing 3e-4 M from the coordinate
        # axis. Measure momentum at the returned half-time position, not at
        # the staggered final position. Fixed-grid temporal errors have a
        # spatial floor, so use differences between refinements for the order.
        def cartesian(q):
            r, t, f = q
            return jnp.array([r*jnp.sin(t)*jnp.cos(f), r*jnp.sin(t)*jnp.sin(f), r*jnp.cos(t)])
        start = jnp.array([.01, .0003, 6.])
        velocity = jnp.array([-.2, 0., 0.])
        expected_gamma = 1/jnp.sqrt(1-jnp.dot(velocity, velocity))
        radius = jnp.linalg.norm(start)
        q = jnp.array([radius, jnp.arccos(start[2]/radius), jnp.arctan2(start[1], start[0])])
        u = jax.jacfwd(cartesian)(q).T @ (expected_gamma*velocity)
        duration = .1
        errors = {}
        for resolution in (64, 128, 256):
            p = SimulationParameters(nr=resolution, ntheta=resolution, devices=1)
            s, d, _, _ = build_runtime(p, particle_batch_size=1, geodesic_iterations=20)
            s = s._replace(metric='flat_spherical', metric_mass=0., metric_spin=0.)
            metric = initialize_flat_spherical_metric(s, d)
            zero = (jnp.zeros_like(metric.center.sqrt_gamma),)*3
            species = one_species(0.)
            for dt in ((.00025, .000125, .0000625) if resolution == 64 else (.0000625,)):
                dynamic = d._replace(dt=jnp.asarray(dt))
                particles = seed_leapfrog_velocity(one_particle(q, u), species, zero, zero,
                                                   s, dynamic, metric=metric)
                def evolve(pts):
                    def step(state, _):
                        new, half = hybrid_boris_geodesic_push(
                            state, species, zero, zero, metric, s, dynamic)
                        return new, half.x
                    return jax.lax.scan(step, pts, None, length=round(duration/dt))
                checked, (final, midpoints) = jax.jit(checkify.checkify(evolve))(particles)
                checked.throw()
                position = final.x.reshape(-1, 3)[0]
                midpoint = midpoints[-1].reshape(-1, 3)[0]
                physical_u = jnp.linalg.solve(jax.jacfwd(cartesian)(midpoint).T,
                                              final.u.reshape(-1, 3)[0])
                gamma = jnp.sqrt(1+jnp.dot(physical_u, physical_u))
                self.assertLess(abs(float(gamma-expected_gamma)), 1e-8)
                errors[resolution, dt] = float(jnp.linalg.norm(
                    cartesian(position)-(start+velocity*duration)))
        temporal = [errors[64, dt] for dt in (.00025, .000125, .0000625)]
        spatial = [errors[n, .0000625] for n in (64, 128, 256)]
        for sequence in (temporal, spatial):
            self.assertGreater((sequence[0]-sequence[1])/(sequence[1]-sequence[2]), 3.5, sequence)
        self.assertLess(spatial[-1], 1.2e-5, spatial)

    def test_stiff_position_solve_matches_independent_root(self):
        from scipy.optimize import root
        sample = flat_spherical_sample
        old = jnp.array([3.8856, .0002644, 1.9487])
        u = jnp.array([.8583, -11.355, -.0007403]); dt = .002
        spacing = jnp.array([.14, jnp.pi/64, 2*jnp.pi])
        def residual(q):
            mid = .5*(old+q)
            return (q-old-dt*GR_position_update(mid, u, sample(mid, False)[0]))/spacing
        error = jax.jit(residual); jacobian = jax.jit(jax.jacfwd(residual))
        expected = root(lambda q: np.asarray(error(q)), np.asarray(old),
                        jac=lambda q: np.asarray(jacobian(q)), tol=1e-10)
        self.assertTrue(expected.success)
        actual = jax.jit(lambda guess: _implicit_position_newton(
            old, guess, u, sample, dt, 20, jnp.array(True), jnp.ones(3,bool), spacing))(old)
        np.testing.assert_allclose(actual, expected.x, rtol=1e-8, atol=1e-9)
        self.assertLess(float(jnp.max(jnp.abs(error(actual)))), 1e-8)

    def test_stiff_midpoint_solve_matches_independent_root(self):
        from scipy.optimize import root
        # A nearly polar flat-space state for which Picard does not contract.
        r, theta = 6., 7e-5
        gamma = jnp.diag(jnp.array([1., r*r, (r*jnp.sin(theta))**2]))
        inverse = jnp.linalg.inv(gamma)
        grad = jnp.zeros((3, 3, 3)).at[0, 1, 1].set(-2/r**3)
        grad = grad.at[0, 2, 2].set(-2/(r**3*jnp.sin(theta)**2))
        grad = grad.at[1, 2, 2].set(-2*jnp.cos(theta)/(r*r*jnp.sin(theta)**3))
        metric = Metric(jnp.array(1.), jnp.zeros(3), gamma, inverse,
                        jnp.sqrt(jnp.linalg.det(gamma)), jnp.zeros((3, 3, 3)),
                        jnp.zeros(3), jnp.zeros((3, 3)))
        old = jnp.array([-.12, -10.4, -.00065]); dt = .004
        def residual(u):
            return u-old-dt*geodesic_velocity(None, .5*(old+u), metric, grad)
        expected = root(lambda u: np.asarray(residual(u)), np.asarray(old), tol=1e-11)
        self.assertTrue(expected.success)
        guess = old
        for _ in range(20):
            guess = old+dt*geodesic_velocity(None, .5*(old+guess), metric, grad)
        self.assertGreater(float(jnp.max(jnp.abs(residual(guess)))), 1e-5)
        actual = jax.jit(lambda value: _implicit_velocity_newton(
            old, value, metric, grad, dt, 20, jnp.array(True)))(guess)
        np.testing.assert_allclose(actual, expected.x, rtol=1e-8, atol=1e-9)
        self.assertEqual(float(actual[2]), float(old[2]))
        self.assertLess(float(jnp.max(jnp.abs(residual(actual)))) /
                        max(1., float(jnp.max(jnp.abs(actual)))), 1e-8)

    def test_near_pole_straight_line_refines_in_time(self):
        # Resolve the angular metric so the comparison measures time error,
        # rather than the Hermite representation error close to the axis.
        p = SimulationParameters(nr=32, ntheta=128, devices=1, r_max=4., sponge_start=3.5)
        s, d, _, _ = build_runtime(p, particle_batch_size=1, geodesic_iterations=10)
        s = s._replace(metric='flat_spherical', metric_mass=0., metric_spin=0.)
        metric = initialize_flat_spherical_metric(s, d)
        zero = (jnp.zeros_like(metric.center.sqrt_gamma),)*3
        species = one_species(0.)
        def cartesian(q):
            r, t, f = q
            return jnp.array([r*jnp.sin(t)*jnp.cos(f), r*jnp.sin(t)*jnp.sin(f), r*jnp.cos(t)])
        start = jnp.array([.3, .1, 3.])
        velocity = jnp.array([-.2, 0., 0.])
        gamma = 1/jnp.sqrt(1-jnp.dot(velocity, velocity))
        radius = jnp.linalg.norm(start)
        q = jnp.array([radius, jnp.arccos(start[2]/radius), jnp.arctan2(start[1], start[0])])
        u = jax.jacfwd(cartesian)(q).T @ (gamma*velocity)
        duration = 1.2
        errors = []
        for dt in (.08, .04, .02):
            dynamic = d._replace(dt=jnp.asarray(dt))
            particles = seed_leapfrog_velocity(one_particle(q, u), species, zero, zero,
                                               s, dynamic, metric=metric)
            def evolve(pts):
                def step(state, _):
                    new, _ = hybrid_boris_geodesic_push(state, species, zero, zero, metric, s, dynamic)
                    return new, None
                return jax.lax.scan(step, pts, None, length=round(duration/dt))[0]
            checked, final = jax.jit(checkify.checkify(evolve))(particles)
            checked.throw()
            position = final.x.reshape(-1, 3)[0]
            errors.append(float(jnp.linalg.norm(cartesian(position)-(start+velocity*duration))))
        self.assertLess(errors[-1], 2e-5, errors)
        self.assertGreater(errors[0]/errors[1], 3., errors)
        self.assertGreater(errors[1]/errors[2], 3., errors)
        # One Picard/Newton iteration at a much closer pole approach is
        # insufficient: the implicit option
        # must diagnose this instead of silently behaving like an explicit step.
        coarse = d._replace(dt=jnp.asarray(.002))
        checked, _ = jax.jit(checkify.checkify(lambda pts: hybrid_boris_geodesic_push(
            pts, species, zero, zero, metric, s._replace(geodesic_iterations=1), coarse)))(
                one_particle(jnp.array([3., 1e-4, .2]), jnp.array([.01, -3., .0003])))
        self.assertIn('implicit', checked.get())


if __name__ == '__main__':
    unittest.main()
