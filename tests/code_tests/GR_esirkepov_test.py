"""
Tests for the charge-conserving GR Esirkepov current deposition.

The property that matters is the discrete continuity equation in conformal
variables,

    ( sqrt(gamma) rho^{n+1} - sqrt(gamma) rho^n ) / dt
        + d-_i( sqrt(gamma) J^i ) = 0,

where ``d-`` is the backward difference that ``update_D_relativity`` uses for its
curl.  When that holds, and because the backward-difference divergence of a
backward-difference curl vanishes identically, the Gauss constraint

    d-_i( sqrt(gamma) D^i ) - 4 pi sqrt(gamma) rho

is preserved exactly in time with no divergence cleaning.
"""

import unittest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from PyPIC3D.boundary_conditions.ghost_cells import update_tiled_vector_ghost_cells
from PyPIC3D.deposition.Esirkepov import Esirkepov_current
from PyPIC3D.deposition.GR_direct_deposition import GR_direct_deposition
from PyPIC3D.deposition.GR_Esirkepov import GR_Esirkepov_current
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.initialization import (
    _encode_current_calculation,
    _validate_current_filter_contract,
    _validate_tiled_yee_configuration,
)
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.relativity.flat import (
    initialize_flat_cartesian_metric,
    initialize_flat_cylindrical_metric,
    initialize_flat_spherical_metric,
)
from PyPIC3D.relativity.kerr_schild import (
    initialize_kerr_schild_cartesian_metric,
    initialize_kerr_schild_spherical_metric,
)
from PyPIC3D.solvers.gr_static.static_metric import update_D_relativity
from PyPIC3D.solvers.gr_static.time_loop import time_loop_static_metric
from tests.kernel_fixtures import (
    empty_tiled_scalar,
    empty_tiled_vector,
    kernel_parameters,
)


METRIC_BUILDERS = {
    "flat_cartesian": initialize_flat_cartesian_metric,
    "flat_cylindrical": initialize_flat_cylindrical_metric,
    "flat_spherical": initialize_flat_spherical_metric,
    "kerr_schild_cartesian": initialize_kerr_schild_cartesian_metric,
    "kerr_schild_spherical": initialize_kerr_schild_spherical_metric,
}


def _unit_species():
    return SpeciesConfig(
        charge=jnp.asarray([1.0]),
        mass=jnp.asarray([1.0]),
        weight=jnp.asarray([1.0]),
        update_x=jnp.asarray([[True, True, True]]),
    )


def _interior(static_parameters):
    g = int(static_parameters.guard_cells)
    return slice(g, -g)


def _backward_divergence(vector_tiles, static_parameters, dynamic_parameters):
    """
    ``d-_i V^i`` at the cell centre, matching ``update_D_relativity``.

    The D components live on the faces, so the backward difference of a
    face-centred density lands on the centre -- exactly where ``compute_rho``
    deposits.  The slice idiom is the one used in ``static_metric.py``.
    """

    g = int(static_parameters.guard_cells)
    active = slice(g, -g)
    backward = slice(g - 1, -g - 1)
    Vx, Vy, Vz = vector_tiles
    return (
        (Vx[:, :, :, active, active, active] - Vx[:, :, :, backward, active, active])
        / dynamic_parameters.dx
        + (Vy[:, :, :, active, active, active] - Vy[:, :, :, active, backward, active])
        / dynamic_parameters.dy
        + (Vz[:, :, :, active, active, active] - Vz[:, :, :, active, active, backward])
        / dynamic_parameters.dz
    )


def _conformal(J, metric):
    """Undo the physical-current divide to recover ``sqrt(gamma) J^i``."""

    return tuple(metric.D[i].sqrt_gamma * J[i] for i in range(3))


def _tiled_single_particle(x, tile_grid_shape):
    """One particle owned by tile (0, 0, 0), broadcast across the tile grid."""

    position = jnp.asarray(x, dtype=float).reshape((1, 1, 1, 1, 1, 3))
    position = jnp.broadcast_to(position, tile_grid_shape + (1, 1, 3))
    active = jnp.zeros(tile_grid_shape + (1, 1), dtype=bool).at[0, 0, 0].set(True)
    return position, active


def _continuity_residual(
    metric_name,
    x_old,
    x_new,
    *,
    shape_factor=1,
    N=(8, 8, 8),
    wind=(8.0, 8.0, 8.0),
    mins=(None, None, None),
    tile_shape=None,
):
    """Return (max |residual|, max |div J|) for one displacement."""

    Nx, Ny, Nz = N
    tile_shape = tile_shape or (Nx, Ny, Nz)
    static_parameters, dynamic_parameters = kernel_parameters(
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        x_wind=wind[0],
        y_wind=wind[1],
        z_wind=wind[2],
        x_min=mins[0],
        y_min=mins[1],
        z_min=mins[2],
        dt=0.1,
        tile_shape=tile_shape,
        shape_factor=shape_factor,
        solver="static_metric",
        metric=metric_name,
        current_deposition="GR_esirkepov",
        current_filter="none",
        particle_pusher="hybrid_boris_geodesic",
    )
    metric = METRIC_BUILDERS[metric_name](static_parameters, dynamic_parameters)
    tile_grid_shape = (
        Nx // tile_shape[0],
        Ny // tile_shape[1],
        Nz // tile_shape[2],
    )

    old_position, active = _tiled_single_particle(x_old, tile_grid_shape)
    new_position, _ = _tiled_single_particle(x_new, tile_grid_shape)
    velocity = jnp.zeros_like(old_position)
    species = _unit_species()
    particles_old = TiledParticles(x=old_position, u=velocity, active=active)
    particles_new = TiledParticles(x=new_position, u=velocity, active=active)

    J = GR_Esirkepov_current(
        particles_old,
        particles_new,
        species,
        empty_tiled_vector(static_parameters, dynamic_parameters),
        metric,
        static_parameters,
        dynamic_parameters,
    )

    rho_template = empty_tiled_scalar(static_parameters, dynamic_parameters)
    rho_old = compute_rho(particles_old, species, rho_template, static_parameters, dynamic_parameters)
    rho_new = compute_rho(particles_new, species, rho_template, static_parameters, dynamic_parameters)

    interior = _interior(static_parameters)
    d_rho_dt = (
        rho_new[:, :, :, interior, interior, interior]
        - rho_old[:, :, :, interior, interior, interior]
    ) / dynamic_parameters.dt
    div_J = _backward_divergence(
        _conformal(J, metric), static_parameters, dynamic_parameters
    )

    residual = d_rho_dt + div_J
    return float(jnp.max(jnp.abs(residual))), float(jnp.max(jnp.abs(div_J)))


class TestGRESirkepovContinuity(unittest.TestCase):
    """The deposition satisfies discrete charge continuity exactly."""

    # Each particle sits at least three cells clear of every domain edge.  The
    # continuity identity is local, so a stencil that straddles the boundary
    # folds charge around the periodic wrap and the residual stops telescoping
    # -- a property of the probe, not of the deposit.  The curvilinear charts
    # also keep the domain away from r = 0 and sin(theta) = 0, where sqrt(gamma)
    # degenerates.
    CASES = (
        ("flat_cartesian", (0.30, -0.70, 1.10), (0.62, -0.41, 1.33), (8.0, 8.0, 8.0), (None, None, None)),
        ("flat_cylindrical", (5.30, 4.70, 4.10), (5.62, 4.91, 4.33), (8.0, 8.0, 8.0), (2.0, 0.3, 0.2)),
        ("flat_spherical", (5.30, 1.10, 1.00), (5.62, 1.18, 1.06), (8.0, 1.6, 1.6), (2.0, 0.4, 0.3)),
        ("kerr_schild_cartesian", (5.30, 5.70, 6.10), (5.62, 5.99, 6.33), (8.0, 8.0, 8.0), (2.0, 2.0, 2.0)),
        ("kerr_schild_spherical", (7.30, 1.10, 1.00), (7.62, 1.18, 1.06), (8.0, 1.6, 1.6), (4.0, 0.4, 0.3)),
    )

    def test_continuity_is_exact_in_every_chart(self):
        for metric_name, x_old, x_new, wind, mins in self.CASES:
            for shape_factor in (1, 2):
                with self.subTest(metric=metric_name, shape_factor=shape_factor):
                    residual, scale = _continuity_residual(
                        metric_name,
                        x_old,
                        x_new,
                        shape_factor=shape_factor,
                        wind=wind,
                        mins=mins,
                    )
                    self.assertGreater(scale, 0.0, "the probe deposited no current")
                    self.assertLess(residual / scale, 1.0e-12)

    def test_continuity_is_exact_across_a_tile_boundary(self):
        # the domain spans [-4, 4) with two tiles in x, so the particle crosses
        # the interior tile seam at x = 0
        residual, scale = _continuity_residual(
            "flat_cartesian",
            (-0.25, -0.70, 1.10),
            (0.20, -0.41, 1.33),
            tile_shape=(4, 8, 8),
        )
        self.assertGreater(scale, 0.0)
        self.assertLess(residual / scale, 1.0e-12)

    def test_continuity_is_exact_on_a_reduced_axis(self):
        # Ny = 1 collapses the polar axis, exercising the out-of-plane
        # displacement branch and collapse_redundant_axis
        residual, scale = _continuity_residual(
            "flat_spherical",
            (5.30, 0.70, 4.50),
            (5.62, 0.93, 4.63),
            N=(8, 1, 8),
            wind=(8.0, 1.0, 8.0),
            mins=(2.0, 0.4, 0.3),
            tile_shape=(8, 1, 8),
        )
        self.assertGreater(scale, 0.0)
        self.assertLess(residual / scale, 1.0e-12)

    def test_stationary_particle_deposits_no_current(self):
        residual, scale = _continuity_residual(
            "flat_spherical",
            (5.30, 1.10, 1.00),
            (5.30, 1.10, 1.00),
            wind=(8.0, 1.6, 1.6),
            mins=(2.0, 0.4, 0.3),
        )
        self.assertEqual(scale, 0.0)
        self.assertEqual(residual, 0.0)


class TestGRESirkepovConventions(unittest.TestCase):
    """Agreement with the flat kernel, and the sqrt(gamma) convention."""

    def _flat_setup(self):
        static_parameters, dynamic_parameters = kernel_parameters(
            Nx=8,
            Ny=8,
            Nz=8,
            x_wind=8.0,
            y_wind=8.0,
            z_wind=8.0,
            dt=0.1,
            tile_shape=(8, 8, 8),
            shape_factor=1,
            metric="flat_cartesian",
            current_deposition="esirkepov",
            current_filter="none",
        )
        metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
        return static_parameters, dynamic_parameters, metric

    def test_matches_the_flat_kernel_in_flat_cartesian(self):
        """
        With sqrt(gamma) == 1 and the new position supplied as x + u dt, the GR
        kernel must reproduce the flat kernel exactly.  This pins both the
        shared-decomposition refactor and the displacement-velocity
        substitution, which coincide with the flat forms in this limit.
        """

        static_parameters, dynamic_parameters, metric = self._flat_setup()
        position = jnp.asarray([(0.30, -0.70, 1.10)]).reshape((1, 1, 1, 1, 1, 3))
        velocity = jnp.asarray([(1.7, 0.9, -1.3)]).reshape((1, 1, 1, 1, 1, 3))
        active = jnp.ones((1, 1, 1, 1, 1), dtype=bool)
        species = _unit_species()
        particles = TiledParticles(x=position, u=velocity, active=active)
        particles_next = TiledParticles(
            x=position + velocity * dynamic_parameters.dt, u=velocity, active=active
        )
        template = empty_tiled_vector(static_parameters, dynamic_parameters)

        flat_J = Esirkepov_current(
            particles, species, template, static_parameters, dynamic_parameters
        )
        gr_J = GR_Esirkepov_current(
            particles,
            particles_next,
            species,
            template,
            metric,
            static_parameters,
            dynamic_parameters,
        )

        self.assertGreater(float(jnp.max(jnp.abs(flat_J[0]))), 0.0)
        for component in range(3):
            self.assertTrue(bool(jnp.array_equal(flat_J[component], gr_J[component])))

    def test_frozen_axes_deposit_no_current(self):
        static_parameters, dynamic_parameters, metric = self._flat_setup()
        position = jnp.asarray([(0.30, -0.70, 1.10)]).reshape((1, 1, 1, 1, 1, 3))
        displaced = position + jnp.asarray([(0.31, 0.22, -0.18)]).reshape((1, 1, 1, 1, 1, 3))
        velocity = jnp.zeros_like(position)
        active = jnp.ones((1, 1, 1, 1, 1), dtype=bool)
        particles_old = TiledParticles(x=position, u=velocity, active=active)
        particles_new = TiledParticles(x=displaced, u=velocity, active=active)
        template = empty_tiled_vector(static_parameters, dynamic_parameters)

        def deposit(update_x):
            return GR_Esirkepov_current(
                particles_old,
                particles_new,
                _unit_species()._replace(update_x=jnp.asarray([update_x])),
                template,
                metric,
                static_parameters,
                dynamic_parameters,
            )

        full = deposit([True, True, True])
        masked = deposit([False, True, False])
        disabled = deposit([False, False, False])

        self.assertGreater(float(jnp.max(jnp.abs(full[1]))), 0.0)
        self.assertTrue(bool(jnp.allclose(masked[0], 0.0)))
        self.assertTrue(bool(jnp.allclose(masked[2], 0.0)))
        for component in disabled:
            self.assertTrue(bool(jnp.allclose(component, 0.0)))

    def test_returns_physical_contravariant_current_in_a_spherical_chart(self):
        """
        The returned current is the physical J^i, so the conformal flux
        sqrt(gamma) J^i d^3x is radius-independent while the physical current
        itself is not.  Mirrors the equivalent GR_direct_deposition test.
        """

        static_parameters, dynamic_parameters = kernel_parameters(
            Nx=8,
            Ny=1,
            Nz=1,
            x_wind=8.0,
            y_wind=1.0,
            z_wind=1.0,
            x_min=2.0,
            y_min=0.4,
            z_min=0.2,
            dt=0.1,
            tile_shape=(8, 1, 1),
            shape_factor=1,
            solver="static_metric",
            metric="flat_spherical",
            current_deposition="GR_esirkepov",
            current_filter="none",
            particle_pusher="hybrid_boris_geodesic",
        )
        metric = initialize_flat_spherical_metric(static_parameters, dynamic_parameters)
        template = empty_tiled_vector(static_parameters, dynamic_parameters)
        species = _unit_species()
        interior = _interior(static_parameters)
        window = (slice(None), slice(None), slice(None), interior, interior, interior)
        radial_step = 0.05

        def deposit_radial_particle(radius):
            active = jnp.ones((1, 1, 1, 1, 1), dtype=bool)
            velocity = jnp.zeros((1, 1, 1, 1, 1, 3))
            old = jnp.asarray((radius, 0.4, 0.2)).reshape((1, 1, 1, 1, 1, 3))
            new = jnp.asarray((radius + radial_step, 0.4, 0.2)).reshape((1, 1, 1, 1, 1, 3))
            J = GR_Esirkepov_current(
                TiledParticles(x=old, u=velocity, active=active),
                TiledParticles(x=new, u=velocity, active=active),
                species,
                template,
                metric,
                static_parameters,
                dynamic_parameters,
            )
            physical = jnp.sum(J[0][window])
            conformal_flux = jnp.sum(
                metric.D[0].sqrt_gamma[window]
                * J[0][window]
                * dynamic_parameters.dx
                * dynamic_parameters.dy
                * dynamic_parameters.dz
            )
            return physical, conformal_flux

        inner_current, inner_flux = deposit_radial_particle(2.5)
        outer_current, outer_flux = deposit_radial_particle(6.5)

        # the swept conformal charge flux is q * dr / dt regardless of radius
        expected_flux = radial_step / dynamic_parameters.dt
        self.assertTrue(bool(jnp.allclose(inner_flux, expected_flux, rtol=1.0e-10)))
        self.assertTrue(bool(jnp.allclose(outer_flux, expected_flux, rtol=1.0e-10)))
        self.assertGreater(float(inner_current), float(outer_current))


class TestGaussConstraintPreservation(unittest.TestCase):
    """The Gauss constraint is conserved by the field update and the deposit."""

    def _loop_setup(self, current_deposition):
        static_parameters, dynamic_parameters = kernel_parameters(
            Nx=8,
            Ny=8,
            Nz=8,
            x_wind=8.0,
            y_wind=8.0,
            z_wind=8.0,
            dt=0.02,
            tile_shape=(8, 8, 8),
            shape_factor=1,
            solver="static_metric",
            metric="flat_cartesian",
            current_deposition=current_deposition,
            current_filter="none",
            particle_pusher="hybrid_boris_geodesic",
        )
        metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
        D = empty_tiled_vector(static_parameters, dynamic_parameters)
        B = empty_tiled_vector(static_parameters, dynamic_parameters)
        J = empty_tiled_vector(static_parameters, dynamic_parameters)
        rho = jnp.zeros_like(J[0])
        phi = jnp.zeros_like(J[0])
        fields = (D, B, J, rho, phi, (D, B), metric, (D, B), jnp.asarray(False))

        position = jnp.asarray([(0.30, -0.70, 1.10)]).reshape((1, 1, 1, 1, 1, 3))
        velocity = jnp.asarray([(0.45, -0.30, 0.20)]).reshape((1, 1, 1, 1, 1, 3))
        active = jnp.ones((1, 1, 1, 1, 1), dtype=bool)
        particles = TiledParticles(x=position, u=velocity, active=active)
        return static_parameters, dynamic_parameters, metric, fields, particles

    def _gauss_residual(self, D, particles, species, metric, static_parameters, dynamic_parameters):
        rho = compute_rho(
            particles,
            species,
            empty_tiled_scalar(static_parameters, dynamic_parameters),
            static_parameters,
            dynamic_parameters,
        )
        interior = _interior(static_parameters)
        conformal_rho = rho[:, :, :, interior, interior, interior]
        return (
            _backward_divergence(_conformal(D, metric), static_parameters, dynamic_parameters)
            - 4.0 * jnp.pi * conformal_rho
        )

    def _constraint_drift(self, current_deposition, steps=4):
        (
            static_parameters,
            dynamic_parameters,
            metric,
            fields,
            particles,
        ) = self._loop_setup(current_deposition)
        species = _unit_species()

        # jit the step, closing over static_parameters the way
        # __main__.run_PyPIC3D does.  Without this the lax.cond deposition
        # selector re-traces both kernels on every iteration, which dominates
        # the runtime of this test.
        step = jax.jit(
            lambda particles, species, fields, dynamic: time_loop_static_metric(
                particles, species, fields, static_parameters, dynamic
            )
        )

        initial = self._gauss_residual(
            fields[0], particles, species, metric, static_parameters, dynamic_parameters
        )
        scale = float(jnp.max(jnp.abs(initial)))

        drift = 0.0
        for _ in range(steps):
            particles, fields = step(particles, species, fields, dynamic_parameters)
            residual = self._gauss_residual(
                fields[0], particles, species, metric, static_parameters, dynamic_parameters
            )
            drift = max(drift, float(jnp.max(jnp.abs(residual - initial))))
        return drift, scale

    def test_gr_esirkepov_conserves_the_gauss_constraint(self):
        """
        d-(sqrt(g) D) - 4 pi sqrt(g) rho is a discrete invariant of the coupled
        update: the curl contributes nothing to it, and the deposit cancels the
        charge change exactly.  It is not zero here (D starts at zero while rho
        does not) -- it is *constant*, which is what a charge-conserving deposit
        buys and what direct deposition destroys.
        """

        drift, scale = self._constraint_drift("GR_esirkepov")
        self.assertGreater(scale, 0.0)
        self.assertLess(drift / scale, 1.0e-12)

    def test_direct_deposition_does_not_conserve_the_gauss_constraint(self):
        """The contrast case: this is audit finding F3."""

        drift, scale = self._constraint_drift("GR_direct")
        self.assertGreater(drift / scale, 1.0e-6)

    def test_backward_divergence_annihilates_the_ampere_curl(self):
        """
        With no current, update_D_relativity cannot change the weighted
        divergence of D at all.  This is why a charge-conserving deposit is
        sufficient on its own, and why the averaged-current auxiliary chain in
        the time loop cannot leak into the constraint: it only ever reaches the
        stored field through a curl.
        """

        static_parameters, dynamic_parameters, metric, _, _ = self._loop_setup("GR_esirkepov")
        key = jax.random.PRNGKey(0)
        shape = empty_tiled_vector(static_parameters, dynamic_parameters)[0].shape
        keys = jax.random.split(key, 6)
        D = update_tiled_vector_ghost_cells(
            tuple(jax.random.normal(keys[i], shape) for i in range(3)),
            static_parameters,
            int(static_parameters.guard_cells),
        )
        H = update_tiled_vector_ghost_cells(
            tuple(jax.random.normal(keys[3 + i], shape) for i in range(3)),
            static_parameters,
            int(static_parameters.guard_cells),
        )
        zero_current = empty_tiled_vector(static_parameters, dynamic_parameters)

        before = _backward_divergence(
            _conformal(D, metric), static_parameters, dynamic_parameters
        )
        D_next = update_D_relativity(
            D, H, zero_current, metric, static_parameters, dynamic_parameters,
            dynamic_parameters.dt,
        )
        after = _backward_divergence(
            _conformal(D_next, metric), static_parameters, dynamic_parameters
        )

        curl_scale = float(jnp.max(jnp.abs(after)))
        self.assertGreater(curl_scale, 0.0)
        self.assertLess(
            float(jnp.max(jnp.abs(after - before))) / curl_scale, 1.0e-12
        )


class TestDepositionDispatch(unittest.TestCase):
    """The lax.cond selector works under jit, the way the driver runs it."""

    def _one_jitted_step(self, current_deposition):
        static_parameters, dynamic_parameters = kernel_parameters(
            Nx=8,
            Ny=8,
            Nz=8,
            x_wind=8.0,
            y_wind=8.0,
            z_wind=8.0,
            dt=0.02,
            tile_shape=(8, 8, 8),
            shape_factor=1,
            solver="static_metric",
            metric="flat_cartesian",
            current_deposition=current_deposition,
            current_filter="none",
            particle_pusher="hybrid_boris_geodesic",
        )
        metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
        D = empty_tiled_vector(static_parameters, dynamic_parameters)
        B = empty_tiled_vector(static_parameters, dynamic_parameters)
        J = empty_tiled_vector(static_parameters, dynamic_parameters)
        rho = jnp.zeros_like(J[0])
        fields = (D, B, J, rho, jnp.zeros_like(J[0]), (D, B), metric, (D, B), jnp.asarray(False))
        particles = TiledParticles(
            x=jnp.asarray([(0.30, -0.70, 1.10)]).reshape((1, 1, 1, 1, 1, 3)),
            u=jnp.asarray([(0.45, -0.30, 0.20)]).reshape((1, 1, 1, 1, 1, 3)),
            active=jnp.ones((1, 1, 1, 1, 1), dtype=bool),
        )

        # close over static_parameters exactly as __main__.run_PyPIC3D does
        step = jax.jit(
            lambda particles, species, fields, dynamic: time_loop_static_metric(
                particles, species, fields, static_parameters, dynamic
            )
        )
        particles, fields = step(particles, _unit_species(), fields, dynamic_parameters)
        return fields

    def test_both_schemes_jit_and_stay_finite(self):
        for current_deposition in ("GR_esirkepov", "GR_direct"):
            with self.subTest(current_deposition=current_deposition):
                fields = self._one_jitted_step(current_deposition)
                for component in fields[0]:
                    self.assertTrue(bool(jnp.all(jnp.isfinite(component))))
                for component in fields[2]:
                    self.assertTrue(bool(jnp.all(jnp.isfinite(component))))

    def test_the_selector_actually_switches_kernels(self):
        """
        Guards against the branch being silently ignored: the two schemes
        deposit genuinely different currents for the same particle state.
        """

        esirkepov = self._one_jitted_step("GR_esirkepov")[2]
        direct = self._one_jitted_step("GR_direct")[2]
        self.assertGreater(float(jnp.max(jnp.abs(esirkepov[0]))), 0.0)
        self.assertGreater(float(jnp.max(jnp.abs(direct[0]))), 0.0)
        differs = any(
            not bool(jnp.allclose(esirkepov[i], direct[i])) for i in range(3)
        )
        self.assertTrue(differs)


class TestGRESirkepovConfiguration(unittest.TestCase):
    """The scheme is reachable from configuration and correctly constrained."""

    def _static_config(self, **overrides):
        config = {
            "solver": "static_metric",
            "current_calculation": "GR_esirkepov",
            "particle_pusher": "hybrid_boris_geodesic",
            "filter_j": "none",
            "particle_tile_nx": 4,
            "particle_tile_ny": 4,
            "particle_tile_nz": 4,
        }
        config.update(overrides)
        return config

    def test_encoder_accepts_the_new_spellings(self):
        for spelling in ("GR_esirkepov", "gr_esirkepov", "GR_Esirkepov"):
            self.assertEqual(_encode_current_calculation(spelling), "GR_esirkepov")

    def test_encoder_still_maps_the_existing_schemes(self):
        self.assertEqual(_encode_current_calculation("GR_direct_deposition"), "GR_direct")
        self.assertEqual(_encode_current_calculation("esirkepov"), "esirkepov")
        self.assertEqual(_encode_current_calculation("j_from_rhov"), "direct")

    def test_encoder_rejects_an_unknown_scheme(self):
        with self.assertRaises(ValueError):
            _encode_current_calculation("not_a_scheme")

    def test_static_metric_accepts_the_new_scheme(self):
        dynamic_config = {"Nx": 8, "Ny": 8, "Nz": 8}
        _validate_tiled_yee_configuration(self._static_config(), dynamic_config)

    def test_static_metric_still_accepts_direct_deposition(self):
        dynamic_config = {"Nx": 8, "Ny": 8, "Nz": 8}
        _validate_tiled_yee_configuration(
            self._static_config(current_calculation="GR_direct_deposition"),
            dynamic_config,
        )

    def test_the_yee_runtime_rejects_the_gr_scheme(self):
        dynamic_config = {"Nx": 8, "Ny": 8, "Nz": 8}
        with self.assertRaises(ValueError):
            _validate_tiled_yee_configuration(
                self._static_config(solver="electrodynamic_yee", particle_pusher="boris"),
                dynamic_config,
            )

    def test_current_filtering_is_rejected(self):
        """Filtering destroys the exact continuity the scheme exists to give."""

        with self.assertRaises(ValueError):
            _validate_current_filter_contract(self._static_config(filter_j="digital"))
        with self.assertRaises(ValueError):
            _validate_current_filter_contract(self._static_config(filter_j="bilinear"))

    def test_unfiltered_configuration_is_accepted(self):
        _validate_current_filter_contract(self._static_config())

    def test_direct_deposition_may_still_be_filtered(self):
        _validate_current_filter_contract(
            self._static_config(current_calculation="GR_direct_deposition", filter_j="digital")
        )


if __name__ == "__main__":
    unittest.main()
