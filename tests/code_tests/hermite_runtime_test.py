"""Checked production sampling, halo contracts, and numerical-grid validation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import checkify
from PyPIC3D.relativity.interpolate_metric import interpolate_metric
from PyPIC3D.pusher.hybrid_boris_geodesic import hybrid_boris_geodesic_push
from PyPIC3D.pusher.particle_push import seed_leapfrog_velocity
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from tests.code_tests.particle_metric_consistency_test import manufactured
from tests.support.particle_metric_fixtures import make_runtime
from tests.kernel_fixtures import kernel_parameters
from PyPIC3D.utilities.parameters import (
    build_static_parameters,
    static_parameters_for_output,
)


def test_checked_sampler_reports_stencil_and_tensor_errors():
    grid, m = manufactured()
    get = jax.jit(
        checkify.checkify(
            lambda q, metric: interpolate_metric(
                metric,
                q,
                grid,
                "numerical",
                (True, True, False),
                (3, 3, 3),
                stage="test metric",
                tile=(1, 0, 0),
            )
        )
    )
    q = jnp.array([[0.23, 0.35, 3.0]])
    errors, _ = get(q, m)
    errors.throw()
    for pos, metric in [
        (q.at[0, 0].set(-5.0), m),
        (q, m._replace(gamma=m.gamma.at[..., 0, 0].set(-1.0))),
        (q, m._replace(lapse=-m.lapse)),
    ]:
        errors, _ = get(pos, metric)
        with pytest.raises(
            Exception, match="test metric.*tile=.*flattened species/slot"
        ):
            errors.throw()


def test_checked_pusher_and_seed_preserve_inactive_slots():
    s, d, m, D, B = make_runtime("spherical", 16, 32)
    x = jnp.array([[[[[[2.0, 0.7, 0.0], [0.0, 0.0, 0.0]]]]]])
    p = TiledParticles(x, jnp.zeros_like(x), jnp.array([[[[[True, False]]]]]))
    sp = SpeciesConfig(jnp.ones(1), jnp.ones(1), jnp.ones(1), jnp.ones((1, 3), bool))
    seed = jax.jit(
        checkify.checkify(lambda p: seed_leapfrog_velocity(p, sp, D, B, s, d, m))
    )
    errors, seeded = seed(p)
    errors.throw()
    push = jax.jit(
        checkify.checkify(lambda p: hybrid_boris_geodesic_push(p, sp, D, B, m, s, d))
    )
    errors, (new, mid) = push(seeded)
    errors.throw()
    np.testing.assert_array_equal(new.x[..., 1, :], p.x[..., 1, :])
    # Invalid active metric samples must fail the functionalized pusher too.
    errors, _ = push(p._replace(x=p.x.at[..., 0, 1].set(0.0)))
    with pytest.raises(Exception, match="invalid particle sample"):
        errors.throw()
    far = d._replace(dt=jnp.asarray(100.0))
    errors, _ = checkify.checkify(
        lambda p: hybrid_boris_geodesic_push(p, sp, D, B, m, s, far)
    )(p._replace(u=p.u.at[..., 0, 0].set(10.0)))
    assert errors.get() is not None


def test_configuration_defaults_and_metadata():
    s, _ = kernel_parameters(
        solver="static_metric", particle_pusher="hybrid_boris_geodesic"
    )
    config = s._asdict()
    config.pop("guard_cells")
    new = build_static_parameters(config)
    assert new.guard_cells == 3
    assert (
        static_parameters_for_output(new)["particle_metric_reconstruction"]
        == "cardinal_cubic_hermite_consistent_v1"
    )
    config["guard_cells"] = 2
    with pytest.raises(ValueError, match="guard_cells >= 3"):
        build_static_parameters(config)
    config.update(solver="electrodynamic_yee", particle_pusher="boris")
    config.pop("guard_cells")
    assert build_static_parameters(config).guard_cells == 2


def test_checked_loop_keeps_migration_contract():
    from unittest.mock import patch
    from tests.code_tests import static_metric_test as existing
    from PyPIC3D.solvers.gr_static.time_loop import time_loop_static_metric
    def checked(p, species, fields, static, dynamic):
        errors, (p, fields) = jax.jit(
            lambda p, fields: time_loop_static_metric(
                p, species, fields, static, dynamic, return_errors=True))(p, fields)
        errors.throw()
        return p, fields
    with patch.object(existing, "time_loop_static_metric", checked):
        existing.test_static_metric_time_loop_retiles_midpoint_and_fullstep_particles()
