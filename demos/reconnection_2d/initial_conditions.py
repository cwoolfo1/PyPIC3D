"""Generate a quiet-start pair-plasma Harris current sheet.

Run this script from ``demos/reconnection_2d`` before starting PyPIC3D.  The
saved magnetic components use PyPIC3D's Yee staggering and the particle files
contain the sheet and uniform background populations combined by species.
"""

from pathlib import Path

import numpy as np


# Physical constants (SI)
KB = 1.380649e-23
MU0 = 1.25663706e-6
EPS0 = 8.85418782e-12
ELECTRON_MASS = 9.1093837e-31
ELEMENTARY_CHARGE = 1.602e-19
LIGHT_SPEED = 2.99792458e8

# Harris-sheet plasma and numerical parameters.  Keep these synchronized with
# harris_current.toml and analyze_data.py.
SEED = 20260906
SHEET_DENSITY = 1.0e22
BACKGROUND_DENSITY = 0.3 * SHEET_DENSITY
THERMAL_SPEED = 0.05 * LIGHT_SPEED
PLASMA_FREQUENCY = ELEMENTARY_CHARGE * np.sqrt(
    SHEET_DENSITY / (ELECTRON_MASS * EPS0)
)
SKIN_DEPTH = LIGHT_SPEED / PLASMA_FREQUENCY
SHEET_HALF_WIDTH = 0.5 * SKIN_DEPTH
LX = 15.0 * SKIN_DEPTH
LY = 1.0
LZ = 15.0 * SKIN_DEPTH
NX = 300
NY = 1
NZ = 300

# The two species each supply n_0 k_B T to the sheet pressure.
TEMPERATURE = ELECTRON_MASS * THERMAL_SPEED**2 / KB
B0 = np.sqrt(
    4.0 * MU0 * SHEET_DENSITY * ELECTRON_MASS * THERMAL_SPEED**2
)
DRIFT_SPEED = B0 / (
    2.0
    * MU0
    * ELEMENTARY_CHARGE
    * SHEET_DENSITY
    * SHEET_HALF_WIDTH
)
PERTURBATION_FRACTION = 0.05

N_SHEET_PARTICLES = 100_000
N_BACKGROUND_PARTICLES = 450_000
N_PARTICLES_PER_SPECIES = N_SHEET_PARTICLES + N_BACKGROUND_PARTICLES
SHEET_COLUMN_DENSITY = (
    2.0
    * SHEET_DENSITY
    * SHEET_HALF_WIDTH
    * np.tanh(LZ / (2.0 * SHEET_HALF_WIDTH))
)
MACROPARTICLE_WEIGHT = SHEET_COLUMN_DENSITY * LX * LY / N_SHEET_PARTICLES


def sample_harris_z(
    rng: np.random.Generator,
    count: int,
    *,
    half_width: float = SHEET_HALF_WIDTH,
    z_wind: float = LZ,
) -> np.ndarray:
    """Sample the truncated ``sech^2(z / half_width)`` sheet profile."""

    if count < 0:
        raise ValueError("count must be non-negative")
    if half_width <= 0.0 or z_wind <= 0.0:
        raise ValueError("half_width and z_wind must be positive")

    cdf_limit = np.tanh(z_wind / (2.0 * half_width))
    transformed = rng.uniform(-cdf_limit, cdf_limit, count)
    return half_width * np.arctanh(transformed)


def harris_vector_potential(
    x_vertices: np.ndarray,
    z_vertices: np.ndarray,
    *,
    b0: float = B0,
    half_width: float = SHEET_HALF_WIDTH,
    lx: float = LX,
    lz: float = LZ,
    perturbation_fraction: float = PERTURBATION_FRACTION,
) -> np.ndarray:
    """Return ``A_y`` on x-z vertices with a central hyperbolic null."""

    x_vertices = np.asarray(x_vertices, dtype=float)
    z_vertices = np.asarray(z_vertices, dtype=float)
    if x_vertices.ndim != 1 or z_vertices.ndim != 1:
        raise ValueError("x_vertices and z_vertices must be one-dimensional")
    if b0 <= 0.0 or half_width <= 0.0 or lx <= 0.0 or lz <= 0.0:
        raise ValueError("field, sheet, and domain scales must be positive")

    x = x_vertices[:, np.newaxis]
    z = z_vertices[np.newaxis, :]
    zbar = z / half_width
    log_cosh = np.logaddexp(zbar, -zbar) - np.log(2.0)

    kx = 2.0 * np.pi / lx
    kz = np.pi / lz
    seed_amplitude = perturbation_fraction * b0 / kx
    equilibrium = -b0 * half_width * log_cosh
    perturbation = -seed_amplitude * np.cos(kx * x) * np.cos(kz * z)
    return equilibrium + perturbation


def build_magnetic_fields(
    *,
    nx: int = NX,
    nz: int = NZ,
    lx: float = LX,
    lz: float = LZ,
    b0: float = B0,
    half_width: float = SHEET_HALF_WIDTH,
    perturbation_fraction: float = PERTURBATION_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Build divergence-free ``Bx`` and ``Bz`` at PyPIC3D Yee locations.

    ``Bx`` is evaluated at (x vertex, z center), while ``Bz`` is evaluated at
    (x center, z vertex).  Both returned arrays include the singleton y axis
    and therefore have shape ``(nx, 1, nz)``.
    """

    if nx <= 0 or nz <= 0:
        raise ValueError("nx and nz must be positive")

    dx = lx / nx
    dz = lz / nz
    x_vertices = np.linspace(-0.5 * lx, 0.5 * lx, nx + 1)
    z_vertices = np.linspace(-0.5 * lz, 0.5 * lz, nz + 1)
    ay = harris_vector_potential(
        x_vertices,
        z_vertices,
        b0=b0,
        half_width=half_width,
        lx=lx,
        lz=lz,
        perturbation_fraction=perturbation_fraction,
    )

    # B = curl(A_y e_y): Bx = -dAy/dz and Bz = dAy/dx.  Taking both
    # differences from one vertex potential makes the matching Yee divergence
    # cancel to roundoff, including across the periodic x seam.
    bx = -(ay[:-1, 1:] - ay[:-1, :-1]) / dz
    bz = (ay[1:, :-1] - ay[:-1, :-1]) / dx
    return bx[:, np.newaxis, :], bz[:, np.newaxis, :]


def _quiet_population(
    rng: np.random.Generator,
    count: int,
    *,
    sheet: bool,
    lx: float,
    lz: float,
    thermal_speed: float,
    half_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create duplicated positions with exactly opposite thermal velocities."""

    if count < 0 or count % 2:
        raise ValueError("quiet-start particle counts must be non-negative and even")

    pair_count = count // 2
    positions = np.empty((pair_count, 3), dtype=float)
    positions[:, 0] = rng.uniform(-0.5 * lx, 0.5 * lx, pair_count)
    positions[:, 1] = 0.0
    if sheet:
        positions[:, 2] = sample_harris_z(
            rng,
            pair_count,
            half_width=half_width,
            z_wind=lz,
        )
    else:
        positions[:, 2] = rng.uniform(-0.5 * lz, 0.5 * lz, pair_count)

    thermal = rng.normal(0.0, thermal_speed, size=(pair_count, 3))
    return (
        np.concatenate((positions, positions), axis=0),
        np.concatenate((thermal, -thermal), axis=0),
    )


def build_particle_arrays(
    *,
    n_sheet: int = N_SHEET_PARTICLES,
    n_background: int = N_BACKGROUND_PARTICLES,
    seed: int = SEED,
    shuffle: bool = True,
    lx: float = LX,
    lz: float = LZ,
    thermal_speed: float = THERMAL_SPEED,
    half_width: float = SHEET_HALF_WIDTH,
    drift_speed: float = DRIFT_SPEED,
) -> dict[str, np.ndarray]:
    """Return matched electron/positron quiet-start particle arrays."""

    rng = np.random.default_rng(seed)
    sheet_positions, sheet_thermal = _quiet_population(
        rng,
        n_sheet,
        sheet=True,
        lx=lx,
        lz=lz,
        thermal_speed=thermal_speed,
        half_width=half_width,
    )
    background_positions, background_thermal = _quiet_population(
        rng,
        n_background,
        sheet=False,
        lx=lx,
        lz=lz,
        thermal_speed=thermal_speed,
        half_width=half_width,
    )

    positions = np.concatenate((sheet_positions, background_positions), axis=0)
    electron_velocity = np.concatenate(
        (sheet_thermal + np.array((0.0, -drift_speed, 0.0)), background_thermal),
        axis=0,
    )
    positron_velocity = np.concatenate(
        (sheet_thermal + np.array((0.0, drift_speed, 0.0)), background_thermal),
        axis=0,
    )

    if shuffle and positions.shape[0]:
        permutation = rng.permutation(positions.shape[0])
        positions = positions[permutation]
        electron_velocity = electron_velocity[permutation]
        positron_velocity = positron_velocity[permutation]

    arrays: dict[str, np.ndarray] = {}
    for species, velocity in (
        ("electron", electron_velocity),
        ("positron", positron_velocity),
    ):
        arrays[f"{species}_x"] = positions[:, 0]
        arrays[f"{species}_y"] = positions[:, 1]
        arrays[f"{species}_z"] = positions[:, 2]
        arrays[f"{species}_vx"] = velocity[:, 0]
        arrays[f"{species}_vy"] = velocity[:, 1]
        arrays[f"{species}_vz"] = velocity[:, 2]
    return arrays


def generate_initial_conditions(
    output_dir: str | Path = ".",
    *,
    nx: int = NX,
    nz: int = NZ,
    n_sheet: int = N_SHEET_PARTICLES,
    n_background: int = N_BACKGROUND_PARTICLES,
    seed: int = SEED,
) -> dict[str, np.ndarray]:
    """Generate and save every array consumed by ``harris_current.toml``."""

    bx, bz = build_magnetic_fields(nx=nx, nz=nz)
    arrays = {
        "Bx": bx,
        "Bz": bz,
        **build_particle_arrays(
            n_sheet=n_sheet,
            n_background=n_background,
            seed=seed,
        ),
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        np.save(destination / f"{name}.npy", array)
    return arrays


def main() -> None:
    """Generate the production-quality reconnection inputs in the CWD."""

    generate_initial_conditions()
    print("Harris-sheet initial conditions generated.")
    print(f"Grid: {NX} x {NY} x {NZ}")
    print(f"Particles: {N_PARTICLES_PER_SPECIES:,} per species")
    print(f"B0: {B0:.9g} T")
    print(f"Sheet half-width: {SHEET_HALF_WIDTH:.9g} m")
    print(f"Thermal speed: {THERMAL_SPEED / LIGHT_SPEED:.3f} c")
    print(f"Counter-drift: {DRIFT_SPEED / LIGHT_SPEED:.3f} c")
    print(f"Macroparticle weight: {MACROPARTICLE_WEIGHT:.9g}")


if __name__ == "__main__":
    main()
