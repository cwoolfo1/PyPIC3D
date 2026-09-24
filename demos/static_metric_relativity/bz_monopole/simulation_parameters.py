"""Run configuration in Gaussian geometrized units: G=M=c=m=|q|=1.

The density normalization n0 is the TOTAL electron plus positron density.
"""
from dataclasses import dataclass
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
    injection_interval: float = 0.1
    sponge_rate: float = 10.0
    end_time: float = 200.0
    output_interval: float = 5.0
    maximum_timestep: float | None = 0.004
    guard_cells: int = 3
    output_directory: str = "data"
    backend: str = "gpu"
    particle_batch_size: int = 8192
    current_filter_passes: int = 4
    horizon_field_cells: int = 5
    gauss_tolerance: float = 1e-10
    magnetic_divergence_tolerance: float = 1e-10
    constraint_check_interval: int = 100
    allow_divergence_errors: bool = False


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


def build_runtime(parameters=SimulationParameters()):
    """Build static/dynamic parameters and the sharded metric; dt is the capped CFL step.

    Returns ``(static, dynamic, metric, cfl_dt)``.
    """
    p = parameters
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
        # Two-GPU timing favors larger active batches;
        # 256-particle batches spend more time in loop/kernel overhead.
        particle_boundary_conditions=(2, 4, 0), particle_batch_size=p.particle_batch_size,
        horizon_field_cells=p.horizon_field_cells))
    center, vertex = build_yee_grid(SimpleNamespace(**config))
    theta=jnp.arange(-1,p.ntheta+1,dtype=jnp.float64)*p.dtheta
    center=(center[0],theta,center[2])
    vertex=(vertex[0],theta+p.dtheta/2,vertex[2])
    tc, tv = build_tiled_yee_grids(static, SimpleNamespace(**config, grids=SimpleNamespace(center=center, vertex=vertex)))
    config["grids"] = dict(center=center, vertex=vertex, tiled_center_grid=tc, tiled_vertex_grid=tv)
    dynamic = build_dynamic_parameters(config)
    metric = initialize_kerr_schild_spherical_metric(static, dynamic, mass=1., spin=p.spin)
    metric = jax.tree.map(lambda a: shard_array(a, static), metric)
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
    limits=[cfl_dt]
    if p.maximum_timestep is not None:limits.append(p.maximum_timestep)
    dynamic = dynamic._replace(dt=jnp.asarray(min(limits)))
    return static, dynamic, metric, cfl_dt
