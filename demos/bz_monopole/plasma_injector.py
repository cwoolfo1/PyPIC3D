"""Neutral thermal pair injection into fixed-capacity tiled particle storage."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from PyPIC3D.particles.particle_class import TiledParticles, SpeciesConfig
from PyPIC3D.particles.particle_tile_communication import shard_tiled_particles
from PyPIC3D.pusher.particle_push import seed_leapfrog_velocity
from PyPIC3D.pusher.hybrid_boris_geodesic import _sample_scalar
from demos.bz_monopole.magnetization import proper_volume


class InjectionReport(NamedTuple):
    requested: jax.Array
    inserted: jax.Array
    rejected: jax.Array
    newborn: jax.Array


def empty_particles(p, static):
    shape = (p.devices, 1, 1, 2, p.slots_per_species)
    # Inactive positions must also have a finite metric: JAX evaluates masked lanes.
    x = jnp.broadcast_to(jnp.array([p.r_min+0.5*p.dr, 0.5*jnp.pi, 0.]), shape+(3,))
    particles = TiledParticles(x=x, u=jnp.zeros_like(x), active=jnp.zeros(shape, bool))
    species = SpeciesConfig(charge=jnp.array([-1., 1.]), mass=jnp.ones(2),
                            weight=jnp.full(2, p.weight), update_x=jnp.ones((2, 3), bool))
    return shard_tiled_particles(particles, static), species


def thermal_momentum(key, temperature):
    """Exact isotropic Maxwell--Juttner draw, density proportional to exp(-gamma/T).

    Propose |u| from Gamma(3,T): p^2 exp(-p/T). Acceptance exp((p-gamma)/T)
    leaves the desired relativistic radial density, without a finite tail cutoff.
    """
    key, direction_key = jax.random.split(key)
    direction = jax.random.normal(direction_key, (3,), dtype=jnp.float64)
    direction /= jnp.linalg.norm(direction)
    def draw(state):
        key, _, _ = state
        key, k1, k2 = jax.random.split(key, 3)
        # Three exponential variates avoid gamma sampler compilation overhead.
        radius = -temperature*jnp.log(jax.random.uniform(k1, (3,), minval=1e-15, dtype=jnp.float64)).sum()
        accept = jax.random.uniform(k2, dtype=jnp.float64) < jnp.exp((radius-jnp.sqrt(1+radius**2))/temperature)
        return key, radius, accept
    _, radius, _ = jax.lax.while_loop(lambda s: ~s[2], draw, (key, jnp.array(0.), jnp.array(False)))
    return radius*direction


def orthonormal_to_covariant(momentum, gamma):
    """Transform with the supplied grid-interpolated covariant spatial metric."""
    return jnp.linalg.cholesky(gamma) @ momentum


def sample_position(key, rlo, rhi, tlo, thi, spin):
    """Sample n(r)*proper-volume with n proportional to r^-2, inside a cell."""
    bound=(1+spin**2/rlo**2)*jnp.sqrt(1+2/rlo)
    def draw(state):
        key,_,_=state;key,k=jax.random.split(key)
        u=jax.random.uniform(k,(4,),dtype=jnp.float64)
        r=rlo+(rhi-rlo)*u[0]
        mu=jnp.cos(tlo)+(jnp.cos(thi)-jnp.cos(tlo))*u[1]
        edge=jnp.nextafter(jnp.float64(1),jnp.float64(0))
        theta=jnp.arccos(jnp.clip(mu,-edge,edge))
        sig=r*r+spin*spin*mu*mu
        accept=u[3]*bound <= sig*jnp.sqrt(1+2*r/sig)/(r*r)
        return key,jnp.array([r,theta,2*jnp.pi*u[2]]),accept
    _,position,_=jax.lax.while_loop(lambda s:~s[2],draw,(key,jnp.array([rlo,(tlo+thi)/2,0.]),jnp.array(False)))
    return position


def inject_pairs(particles, species, magnetization, D, B, metric, static, dynamic,
                 parameters, key, step):
    """Insert complete pairs; call inside a JIT closing over static/parameters.

    Each event proposes n0/r^2 particles per proper volume (both species combined).
    Stochastic rounding preserves this expectation with fixed macroparticle weights.
    Capacity is per species, not per pair. The caller must fail if rejected != 0.
    """
    p = parameters
    if species.charge.shape != (2,):
        raise ValueError("Pair injection requires exactly two species")
    g = static.guard_cells
    nr, nt, _ = static.tile_shape
    nt += 1  # both polar charge-control volumes are owned
    slots = particles.active.shape[-1]
    candidate_capacity = min(slots, nr*nt*(p.pairs_per_cell+2))
    interior = (slice(None),)*3 + (slice(g,-g),slice(g,-g+1),slice(g,-g))
    volume = proper_volume(metric, dynamic)[interior]
    rgrid, tgrid, _ = dynamic.grids.tiled_center_grid
    sigma = magnetization.sigma[interior]
    valid = magnetization.valid[interior]
    next_key, event_key = jax.random.split(key)
    event_key = jax.random.fold_in(event_key, step)

    def one_tile(x, u, active, sig, valid, vol, rline, tline, pline, gamma_grid, tile_id):
        rr, tt = jnp.meshgrid(rline[g:-g], tline[g:-g+1], indexing="ij")
        rr, tt = rr.reshape(-1), tt.reshape(-1)
        cells = jnp.arange(nr*nt)
        global_cells = tile_id*nr*nt+cells
        cell_keys = jax.vmap(lambda i: jax.random.fold_in(event_key, i))(global_cells)
        # Injection samples clipped charge-control volumes. Radial boundary
        # truncation is integrated with the same 8-point geometry quadrature.
        rlo=jnp.maximum(rr-p.dr/2,p.r_min); rhi=jnp.minimum(rr+p.dr/2,p.sponge_start)
        tlo=jnp.maximum(tt-p.dtheta/2,0.); thi=jnp.minimum(tt+p.dtheta/2,jnp.pi)
        from numpy.polynomial.legendre import leggauss
        nodes,weights=leggauss(8)
        integral=jnp.zeros_like(rr)
        for a,wa in zip(nodes,weights):
            radii=(rlo+rhi)/2+(rhi-rlo)*a/2
            for b,wb in zip(nodes,weights):
                angles=(tlo+thi)/2+(thi-tlo)*b/2
                geom_sig=radii**2+p.spin**2*jnp.cos(angles)**2
                integral+=wa*wb*geom_sig*jnp.sqrt(1+2*radii/geom_sig)*jnp.sin(angles)/radii**2
        expected = p.n0*integral*jnp.maximum(rhi-rlo,0)*(thi-tlo)*2*jnp.pi/4/(2*species.weight[0])
        eligible = valid.reshape(-1) & (sig.reshape(-1)>p.sigma_threshold) & (rr<p.sponge_start)
        rounding = jax.vmap(lambda k: jax.random.uniform(k, dtype=jnp.float64))(cell_keys)
        counts = jnp.where(eligible, jnp.floor(expected+rounding).astype(jnp.int32), 0)
        cumulative = jnp.cumsum(counts)
        requested = jnp.sum(counts)
        free = jnp.sum(~active, axis=-1)
        inserted = jnp.minimum(jnp.minimum(free[0], free[1]), jnp.minimum(requested, candidate_capacity))
        ranks = jnp.arange(candidate_capacity)
        cell = jnp.minimum(jnp.searchsorted(cumulative, ranks, side="right"), nr*nt-1)
        offset = ranks-jnp.where(cell>0, cumulative[jnp.maximum(cell-1, 0)], 0)
        keys = jax.vmap(jax.random.fold_in)(cell_keys[cell], offset)
        # Cell and ordinal keys make samples independent of the tile decomposition.
        positions=jax.vmap(lambda k,rl,rh,tl,th:sample_position(jax.random.fold_in(k,1),rl,rh,tl,th,p.spin))(
            keys,rlo[cell],rhi[cell],tlo[cell],thi[cell])
        momenta = jax.vmap(lambda k: thermal_momentum(jax.random.fold_in(k, 2), p.temperature))(keys)
        gamma = _sample_scalar(gamma_grid,positions[:,0],positions[:,1],positions[:,2],
                               (rline,tline,pline),static.shape_factor,(True,True,False),(g,g,g))
        covariant = jax.vmap(orthonormal_to_covariant)(momenta, gamma)
        newborn = jnp.zeros_like(active)
        for s in range(2):
            indices = jnp.nonzero(~active[s], size=slots, fill_value=slots)[0][:candidate_capacity]
            indices = jnp.where(ranks<inserted, indices, slots)
            x = x.at[s, indices].set(positions, mode="drop")
            u = u.at[s, indices].set(covariant, mode="drop")
            active = active.at[s, indices].set(True, mode="drop")
            newborn = newborn.at[s, indices].set(True, mode="drop")
        return x, u, active, newborn, requested, inserted

    x, u, active, newborn, requested, inserted = jax.vmap(one_tile)(
        particles.x[:, 0, 0], particles.u[:, 0, 0], particles.active[:, 0, 0],
        sigma[:, 0, 0], valid[:, 0, 0], volume[:, 0, 0], rgrid[:, 0, 0],
        tgrid[:, 0, 0], dynamic.grids.tiled_center_grid[2][:,0,0],
        metric.center.gamma[:,0,0], jnp.arange(p.devices))
    result = TiledParticles(x[:, None, None], u[:, None, None], active[:, None, None])
    newborn = newborn[:, None, None]
    seeded = seed_leapfrog_velocity(result._replace(active=newborn), species, D, B, static, dynamic, metric)
    result = result._replace(u=jnp.where(newborn[..., None], seeded.u, result.u))
    return result, next_key, InjectionReport(requested, inserted, requested-inserted, newborn)


def check_species(species):
    """Host-side contract check, outside the compiled injector."""
    if not (bool(jnp.allclose(species.charge, jnp.array([-1., 1.])))
            and bool(jnp.all(species.mass==1)) and bool(jnp.all(species.weight>0))
            and bool(species.weight[0]==species.weight[1]) and bool(jnp.all(species.update_x))):
        raise ValueError("Expected equal-weight unit-mass electron/positron species with all components enabled")


def self_test():
    from demos.bz_monopole.simulation_parameters import SimulationParameters, build_runtime
    from demos.bz_monopole.magnetization import Magnetization, deposit_number_density
    from PyPIC3D.deposition.rho import compute_rho
    p = SimulationParameters(nr=16, ntheta=16, devices=1, pairs_per_cell=1, capacity_factor=2,
                             r_max=4., sponge_start=3.)
    s, d, m, _ = build_runtime(p)
    particles, species = empty_particles(p, s)
    shape = m.center.sqrt_gamma.shape
    zero = jnp.zeros(shape)
    measure = Magnetization(jnp.zeros((2,)+shape), jnp.ones(shape), jnp.full(shape, jnp.inf), jnp.ones(shape, bool))
    call = jax.jit(lambda particles, key: inject_pairs(particles, species, measure, (zero,)*3, (zero,)*3, m, s, d, p, key, 0))
    result, _, report = call(particles, jax.random.PRNGKey(p.seed))
    assert int(jnp.sum(report.inserted)) > 0 and int(jnp.sum(report.rejected)) == 0
    assert bool(jnp.array_equal(result.active[..., 0, :], result.active[..., 1, :]))
    rho = compute_rho(result, species, zero, s, d)
    assert float(jnp.max(jnp.abs(rho))) < 1e-12
    density = deposit_number_density(result, species, zero, m, s, d)
    assert bool(jnp.all(density >= 0))
    print(f"plasma_injector: PASS ({int(jnp.sum(report.inserted))} neutral pairs)")


if __name__ == "__main__":
    self_test()
