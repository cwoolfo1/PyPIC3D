"""Run configuration in Gaussian geometrized units: G=M=c=m=|q|=1.

Run independently with ``python -m demos.bz_monopole.simulation_parameters``.
The density normalization n0 is the TOTAL electron plus positron density.
"""
from dataclasses import asdict, dataclass
import json
import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from PyPIC3D.relativity.kerr_schild import initialize_kerr_schild_spherical_metric
from PyPIC3D.utilities.grids import build_yee_grid, build_tiled_yee_grids
from PyPIC3D.utilities.parameters import build_static_parameters, build_dynamic_parameters


@dataclass(frozen=True)
class SimulationParameters:
    nr: int = 64
    ntheta: int = 64
    devices: int = 1
    r_min: float = 1.0
    r_max: float = 10.0
    sponge_start: float = 9.0
    spin: float = 0.2
    sigma0: float = 2500.0
    sigma_threshold: float = 2000.0
    skin_depth: float = 0.02  # Scaled demo; Entity uses 0.0025.
    temperature: float = 0.5
    pairs_per_cell: int = 16
    capacity_factor: int = 8
    seed: int = 20260908
    courant: float = 0.2
    gyro_angle: float = 0.1  # Legacy checkpoint parameter; unused by CFL timestepping.
    injection_interval: float = 0.1
    sponge_rate: float = 10.0
    end_time: float = 200.0
    output_interval: float = 5.0
    maximum_timestep: float | None = 0.004
    guard_cells: int = 3

    def validate(self):
        for name in ("nr", "ntheta", "devices", "pairs_per_cell", "capacity_factor", "guard_cells"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.maximum_timestep is not None and (not math.isfinite(self.maximum_timestep) or self.maximum_timestep <= 0):
            raise ValueError("maximum_timestep must be finite and positive")
        if self.devices not in (1, 2) or self.nr % self.devices:
            raise ValueError("Use one or two devices, with nr divisible by devices")
        if self.nr // self.devices < 4 or self.ntheta < 8 or self.ntheta % 2:
            raise ValueError("Need >=4 radial cells per tile and an even ntheta >=8")
        if self.guard_cells != 3:
            raise ValueError("Polar endpoint ownership requires three guard cells")
        for name, value in asdict(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.spin < 1 or not 0 < self.r_min < self.horizon:
            raise ValueError("Require 0<=spin<1 and an inner boundary inside the horizon")
        if not self.r_min < self.sponge_start < self.r_max:
            raise ValueError("Require r_min < sponge_start < r_max")
        if self.r_min - self.guard_cells*self.dr <= 0:
            raise ValueError("Radial guard cells must stay above r=0; increase nr")
        for name in ("sigma0", "sigma_threshold", "skin_depth", "temperature", "courant",
                     "gyro_angle", "injection_interval", "sponge_rate", "end_time", "output_interval"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.courant > 0.5 or self.gyro_angle > 0.2:
            raise ValueError("Require courant<=0.5 and gyro_angle<=0.2")
        return self

    @property
    def dr(self):
        return (self.r_max - self.r_min) / self.nr

    @property
    def dtheta(self):
        return math.pi / self.ntheta

    @property
    def n0(self):
        return 1 / (4 * math.pi * self.skin_depth**2)

    @property
    def larmor_radius(self):
        return self.skin_depth / math.sqrt(self.sigma0)

    @property
    def B0(self):
        return 1 / self.larmor_radius

    @property
    def horizon(self):
        return 1 + math.sqrt(1 - self.spin**2)

    @property
    def omega_h(self):
        # -g_tphi/g_phiphi on r=r_h in the metric used by PyPIC3D.
        return self.spin / (2 * self.horizon)

    @property
    def slots_per_species(self):
        return self.nr // self.devices * (self.ntheta+1) * self.pairs_per_cell * self.capacity_factor

    @property
    def weight(self):
        # Fixed species weights; number of injected pairs varies with proper volume.
        # Reference cell at r=1, theta=pi/2, with the full axisymmetric 2pi measure.
        volume = math.sqrt(3) * self.dr * self.dtheta * (2 * math.pi)
        return self.n0 * volume / (2 * self.pairs_per_cell)


def shard_array(array, static):
    return jax.device_put(array, NamedSharding(static.field_mesh, P("tile_x", "tile_y", "tile_z")))


def build_runtime(parameters=SimulationParameters(), *, timestep_policy="cfl", particle_batch_size=65536,
                  horizon_field_cells=0, field_interpolation='physical',
                  particle_coordinates='native'):
    """Build a runtime with explicitly selected numerical methods.

    This low-level builder retains native/physical defaults for component
    tests and callers; the demo CLI supplies the accepted Cartesian preset.
    """
    p = parameters.validate()
    if isinstance(particle_batch_size, bool) or not isinstance(particle_batch_size, int) or particle_batch_size <= 0:
        raise ValueError('particle_batch_size must be a positive integer')
    if horizon_field_cells and (p.r_min+(horizon_field_cells+1)*p.dr >= p.horizon
                                or horizon_field_cells >= p.nr//p.devices):
        raise ValueError('Horizon field layer and its reference plane must lie inside the horizon and first tile')
    jax.config.update("jax_enable_x64", True)
    config = dict(Nx=p.nr, Ny=p.ntheta, Nz=1, x_wind=p.r_max-p.r_min,
                  y_wind=math.pi, z_wind=2*math.pi, x_min=p.r_min,
                  y_min=0., z_min=0.0, dx=p.dr, dy=p.dtheta,
                  dz=2*math.pi, dt=1.0)
    static = build_static_parameters(dict(
        **config, name="bz_monopole", solver="static_metric",
        metric="kerr_schild_spherical", metric_mass=1., metric_spin=p.spin,
        particle_pusher="hybrid_boris_geodesic", current_deposition="GR_esirkepov",
        current_filter="none", shape_factor=1, guard_cells=p.guard_cells,
        tile_shape=(p.nr//p.devices, p.ntheta, 1), boundary_conditions=(3, 4, 0),
        # Two-GPU timing of the same checkpoint favors larger active batches;
        # 256-particle batches spend more time in loop/kernel overhead.
        particle_boundary_conditions=(2, 4, 0), particle_batch_size=particle_batch_size,
        horizon_field_cells=horizon_field_cells, polar_field_interpolation=field_interpolation,
        particle_coordinates=particle_coordinates))
    center, vertex = build_yee_grid(SimpleNamespace(**config))
    theta=jnp.arange(-1,p.ntheta+1,dtype=jnp.float64)*p.dtheta
    center=(center[0],theta,center[2])
    vertex=(vertex[0],theta+p.dtheta/2,vertex[2])
    tc, tv = build_tiled_yee_grids(static, SimpleNamespace(**config, grids=SimpleNamespace(center=center, vertex=vertex)))
    config["grids"] = dict(center=center, vertex=vertex, tiled_center_grid=tc, tiled_vertex_grid=tv)
    dynamic = build_dynamic_parameters(config)
    metric = initialize_kerr_schild_spherical_metric(static, dynamic, mass=1., spin=p.spin)
    metric = jax.tree.map(lambda a: shard_array(a, static), metric)
    from PyPIC3D.relativity.particle_metric import validate_particle_metric_grids
    validate_particle_metric_grids(metric, dynamic.grids.tiled_center_grid, (True,True,False), p.guard_cells)
    if particle_coordinates == 'cartesian':
        from PyPIC3D.relativity.cartesian_particle_metric import validate_regularized_axes
        validate_regularized_axes(metric, dynamic.grids.tiled_center_grid)
    interior = (slice(None),)*3 + (slice(p.guard_cells, -p.guard_cells),)*3
    m = metric.center
    # Directional characteristic bounds use regular analytic inverse components,
    # including the axes where the unused full inverse tensor is masked.
    rr=dynamic.grids.tiled_center_grid[0][..., :,None,None]
    tt=dynamic.grids.tiled_center_grid[1][...,None,:,None]
    sig=rr**2+p.spin**2*jnp.cos(tt)**2
    xi=1+2*rr/sig
    inv_rr=1/xi+p.spin**2*jnp.sin(tt)**2/sig
    sr=jnp.abs(m.shift[...,0])+m.lapse*jnp.sqrt(inv_rr)
    st=m.lapse/jnp.sqrt(sig)
    speed=jnp.stack((sr,st,jnp.zeros_like(sr)),axis=-1)
    cfl_dt = p.courant / float(jnp.max((speed[..., 0]/p.dr + speed[..., 1]/p.dtheta)[interior]))
    if timestep_policy != "cfl":
        raise ValueError('Unknown timestep policy: use cfl')
    limits=[cfl_dt]
    if p.maximum_timestep is not None:limits.append(p.maximum_timestep)
    maximum_dt = p.maximum_timestep
    dynamic = dynamic._replace(dt=jnp.asarray(min(limits)))
    lengths = jnp.sqrt(jnp.diagonal(m.gamma, axis1=-2, axis2=-1))[interior]
    report = dict(parameters=asdict(p), dt=float(dynamic.dt), cfl_dt=cfl_dt,
                  field_interpolation=field_interpolation,
                  particle_coordinates=particle_coordinates,
                  horizon_field_cells=horizon_field_cells,
                  particle_batch_size=static.particle_batch_size,
                  timestep_policy=timestep_policy,
                  maximum_timestep=maximum_dt,
                  metric_reconstruction=("orthonormal_spherical_hermite_v1" if particle_coordinates == 'cartesian'
                                         else "cardinal_cubic_hermite_consistent_v1"),
                  n0_total=p.n0, B0=p.B0, rho0=p.larmor_radius,
                  species_weight=p.weight, slots_per_species_per_tile=p.slots_per_species,
                  particle_storage_bytes=p.devices*2*p.slots_per_species*(6*8+1),
                  metric_storage_bytes=sum(a.size*a.dtype.itemsize for a in jax.tree.leaves(metric)),
                  radial_cell_size=[float(jnp.min(lengths[..., 0])*p.dr), float(jnp.max(lengths[..., 0])*p.dr)],
                  theta_cell_size=[float(jnp.min(lengths[..., 1])*p.dtheta), float(jnp.max(lengths[..., 1])*p.dtheta)],
                  d0_over_max_radial_cell=p.skin_depth/float(jnp.max(lengths[..., 0])*p.dr),
                  d0_over_max_theta_cell=p.skin_depth/float(jnp.max(lengths[...,1])*p.dtheta),
                  rho0_over_max_radial_cell=p.larmor_radius/float(jnp.max(lengths[...,0])*p.dr),
                  rho0_over_max_theta_cell=p.larmor_radius/float(jnp.max(lengths[...,1])*p.dtheta),
                  injection_template=f"total proper number density per event = n0/r^2, r<{p.sponge_start:g}",
                  geometry_id="polar-cap-v1", owned_theta_charge_planes=p.ntheta+1,
                  field_history_storage_bytes=int(metric.center.sqrt_gamma.size*8*18),
                  omega_h=p.omega_h, devices=[str(d) for d in static.field_mesh.devices.flat])
    return static, dynamic, metric, report


def self_test():
    p = SimulationParameters(devices=1, nr=16, ntheta=16, r_max=4., sponge_start=3.).validate()
    static, dynamic, metric, report = build_runtime(p)
    assert float(dynamic.grids.center[1][1]) == 0.
    assert math.isclose(float(dynamic.grids.center[1][-1]),math.pi)
    assert bool(jnp.all(metric.geometry.volume[metric.geometry.charge_owned]>0))
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(metric))
    assert math.isclose(p.B0**2/(4*math.pi*p.n0), p.sigma0)
    assert report["dt"] > 0
    print(json.dumps(report, indent=2))
    print("simulation_parameters: PASS")


if __name__ == "__main__":
    self_test()
