"""Generate quiet-start Orszag--Tang initial data for the GPU cases.

The checked-in TOML files refer to ``generated/<case>/*.npy``.  Particle
arrays are written as NumPy-format memory maps so the production case can be
created without holding the full particle population in host memory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil

import numpy as np


MU0 = 1.25663706212e-6
EPS0 = 8.8541878128e-12
ELEMENTARY_CHARGE = 1.602176634e-19
ELECTRON_MASS = 9.1093837015e-31
BOLTZMANN = 1.380649e-23
C = 1.0 / np.sqrt(MU0 * EPS0)
NUMBER_DENSITY = 1.0e14
MASS_RATIO = 4.0
ION_MASS = MASS_RATIO * ELECTRON_MASS
PPC = 16
SUBCELLS = 4
ALFVEN_SPEED = 0.1 * C
BETA_SPECIES = 0.08
GUIDE_FIELD_RATIO = 5.0
MAX_PARTICLE_SPEED = 0.95 * C


@dataclass(frozen=True)
class Case:
    name: str
    nx: int
    ny: int
    output_interval: int

    @property
    def de(self):
        return np.sqrt(ELECTRON_MASS / (MU0 * NUMBER_DENSITY * ELEMENTARY_CHARGE**2))

    @property
    def di(self):
        return np.sqrt(ION_MASS / ELECTRON_MASS) * self.de

    @property
    def dx(self):
        return 0.1 * self.de

    @property
    def lx(self):
        return self.nx * self.dx

    @property
    def ly(self):
        return self.ny * self.dx

    @property
    def particle_count(self):
        return PPC * self.nx * self.ny

    @property
    def turnover_time(self):
        return self.lx / (2.0 * np.pi * ALFVEN_SPEED)

    @property
    def run_time(self):
        return 6.0 * self.turnover_time


CASES = {
    "smoke": Case("smoke", 64, 64, 5),
    "pilot": Case("pilot", 512, 512, 82),
    "spectrum": Case("spectrum", 1024, 1024, 165),
}


def magnetic_amplitudes():
    fluctuation = ALFVEN_SPEED * np.sqrt(MU0 * NUMBER_DENSITY * ION_MASS)
    return fluctuation, GUIDE_FIELD_RATIO * fluctuation


def species_temperature():
    _, guide = magnetic_amplitudes()
    return BETA_SPECIES * guide**2 / (2.0 * MU0 * NUMBER_DENSITY * BOLTZMANN)


def discrete_wavenumber(wavenumber, spacing):
    """Symbol of the backward Yee derivative for one Fourier mode."""

    return 2.0 * np.sin(0.5 * wavenumber * spacing) / spacing


def density_perturbation_amplitude(case):
    """Amplitude per cosine needed by the discrete divergence of ideal E."""

    _, guide = magnetic_amplitudes()
    k = 2.0 * np.pi / case.lx
    return EPS0 * ALFVEN_SPEED * guide * discrete_wavenumber(k, case.dx) / (
        ELEMENTARY_CHARGE * NUMBER_DENSITY
    )


def build_fields(case):
    """Return all six evolved Yee components with shape ``(Nx, Ny, 1)``."""

    fluctuation, guide = magnetic_amplitudes()
    kx = 2.0 * np.pi / case.lx
    ky = 2.0 * np.pi / case.ly
    xmin = -0.5 * case.lx
    ymin = -0.5 * case.ly
    center_x = xmin + np.arange(case.nx) * case.dx
    center_y = ymin + np.arange(case.ny) * case.dx
    vertex_x = center_x + 0.5 * case.dx
    vertex_y = center_y + 0.5 * case.dx

    Ex = -ALFVEN_SPEED * guide * np.sin(kx * vertex_x)[:, None, None]
    Ex = np.broadcast_to(Ex, (case.nx, case.ny, 1)).copy()
    Ey = -ALFVEN_SPEED * guide * np.sin(ky * vertex_y)[None, :, None]
    Ey = np.broadcast_to(Ey, (case.nx, case.ny, 1)).copy()
    Ez = np.zeros((case.nx, case.ny, 1), dtype=np.float64)

    Bx = -fluctuation * np.sin(ky * vertex_y)[None, :, None]
    Bx = np.broadcast_to(Bx, (case.nx, case.ny, 1)).copy()
    By = fluctuation * np.sin(2.0 * kx * vertex_x)[:, None, None]
    By = np.broadcast_to(By, (case.nx, case.ny, 1)).copy()
    Bz = np.full((case.nx, case.ny, 1), guide, dtype=np.float64)
    return {"Ex": Ex, "Ey": Ey, "Ez": Ez, "Bx": Bx, "By": By, "Bz": Bz}


def _open_particle_arrays(output_dir, count):
    arrays = {}
    for species in ("electron", "ion"):
        for component in ("x", "y", "z", "vx", "vy", "vz"):
            path = output_dir / f"{species}_{component}.npy"
            arrays[f"{species}_{component}"] = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=np.float64,
                shape=(count,),
            )
    return arrays


def _quiet_cell_positions(case, cell_indices):
    cell_x = cell_indices // case.ny
    cell_y = cell_indices % case.ny
    subparticle = np.arange(PPC)
    sub_x = subparticle // SUBCELLS
    sub_y = subparticle % SUBCELLS
    xmin = -0.5 * case.lx
    ymin = -0.5 * case.ly
    qx = xmin + (
        cell_x[:, None] + (sub_x[None, :] + 0.5) / SUBCELLS
    ) * case.dx
    qy = ymin + (
        cell_y[:, None] + (sub_y[None, :] + 0.5) / SUBCELLS
    ) * case.dx
    return qx, qy


def _thermal_velocities(rng, cell_count, sigma):
    half = rng.normal(size=(cell_count, PPC // 2, 3)) * sigma
    return np.concatenate((half, -half), axis=1).reshape((-1, 3))


def _resample_superluminal(rng, thermal, bulk, sigma):
    total = thermal + bulk
    invalid = np.linalg.norm(total, axis=1) >= MAX_PARTICLE_SPEED
    attempts = 0
    while np.any(invalid):
        thermal[invalid] = rng.normal(size=(int(np.count_nonzero(invalid)), 3)) * sigma
        total[invalid] = thermal[invalid] + bulk[invalid]
        invalid = np.linalg.norm(total, axis=1) >= MAX_PARTICLE_SPEED
        attempts += 1
        if attempts > 100:
            raise RuntimeError("Unable to sample a subluminal thermal population after 100 attempts.")
    return total


def write_particles(case, output_dir, seed=42, cells_per_chunk=32768):
    arrays = _open_particle_arrays(output_dir, case.particle_count)
    rng = np.random.default_rng(seed)
    temperature = species_temperature()
    electron_sigma = np.sqrt(BOLTZMANN * temperature / ELECTRON_MASS)
    ion_sigma = np.sqrt(BOLTZMANN * temperature / ION_MASS)
    fluctuation, _ = magnetic_amplitudes()
    k = 2.0 * np.pi / case.lx
    k_discrete = discrete_wavenumber(k, case.dx)
    k2_discrete = discrete_wavenumber(2.0 * k, case.dx)
    density_amplitude = density_perturbation_amplitude(case)

    total_cells = case.nx * case.ny
    for first_cell in range(0, total_cells, cells_per_chunk):
        last_cell = min(first_cell + cells_per_chunk, total_cells)
        cells = np.arange(first_cell, last_cell)
        qx, qy = _quiet_cell_positions(case, cells)
        qx = qx.reshape(-1)
        qy = qy.reshape(-1)

        electron_x = qx - density_amplitude * np.sin(k * qx) / k
        electron_y = qy - density_amplitude * np.sin(k * qy) / k
        ion_x = qx
        ion_y = qy
        zeros = np.zeros_like(qx)

        electron_bulk = np.empty((qx.size, 3), dtype=np.float64)
        electron_bulk[:, 0] = -ALFVEN_SPEED * np.sin(k * electron_y)
        electron_bulk[:, 1] = ALFVEN_SPEED * np.sin(k * electron_x)
        local_density = NUMBER_DENSITY * (
            1.0
            + density_amplitude * (np.cos(k * electron_x) + np.cos(k * electron_y))
        )
        target_jz = fluctuation / MU0 * (
            k2_discrete * np.cos(2.0 * k * electron_x)
            + k_discrete * np.cos(k * electron_y)
        )
        electron_bulk[:, 2] = -target_jz / (ELEMENTARY_CHARGE * local_density)

        ion_bulk = np.empty_like(electron_bulk)
        ion_bulk[:, 0] = -ALFVEN_SPEED * np.sin(k * ion_y)
        ion_bulk[:, 1] = ALFVEN_SPEED * np.sin(k * ion_x)
        ion_bulk[:, 2] = 0.0

        cell_count = last_cell - first_cell
        electron_velocity = _resample_superluminal(
            rng,
            _thermal_velocities(rng, cell_count, electron_sigma),
            electron_bulk,
            electron_sigma,
        )
        ion_velocity = _resample_superluminal(
            rng,
            _thermal_velocities(rng, cell_count, ion_sigma),
            ion_bulk,
            ion_sigma,
        )

        start = first_cell * PPC
        stop = last_cell * PPC
        for species, positions, velocity in (
            ("electron", (electron_x, electron_y, zeros), electron_velocity),
            ("ion", (ion_x, ion_y, zeros), ion_velocity),
        ):
            arrays[f"{species}_x"][start:stop] = positions[0]
            arrays[f"{species}_y"][start:stop] = positions[1]
            arrays[f"{species}_z"][start:stop] = positions[2]
            arrays[f"{species}_vx"][start:stop] = velocity[:, 0]
            arrays[f"{species}_vy"][start:stop] = velocity[:, 1]
            arrays[f"{species}_vz"][start:stop] = velocity[:, 2]

    for array in arrays.values():
        array.flush()


def write_fields(case, output_dir):
    for name, field in build_fields(case).items():
        np.save(output_dir / f"{name}.npy", field)


def report_case(case):
    fluctuation, guide = magnetic_amplitudes()
    temperature = species_temperature()
    dt = 0.99 / (C * (1.0 / case.dx + 1.0 / case.dx))
    debye = np.sqrt(
        EPS0 * BOLTZMANN * temperature
        / (NUMBER_DENSITY * ELEMENTARY_CHARGE**2)
    )
    weight = NUMBER_DENSITY * case.dx**3 / PPC
    input_gib = (12 * case.particle_count + 6 * case.nx * case.ny) * 8 / 2**30
    print(f"\n{case.name}: {case.nx} x {case.ny} x 1")
    print(f"  L/de={case.lx / case.de:.3f}, L/di={case.lx / case.di:.3f}")
    print(f"  dx/de={case.dx / case.de:.3f}, dx/lambda_D={case.dx / debye:.3f}")
    print(f"  particles/species={case.particle_count:,}, PPC={PPC}, weight={weight:.9e}")
    print(f"  B0={fluctuation:.9e} T, Bg={guide:.9e} T, V0/c={ALFVEN_SPEED / C:.3f}")
    print(f"  T={temperature:.9e} K ({BOLTZMANN * temperature / ELEMENTARY_CHARGE:.3f} eV)")
    print(f"  tau={case.turnover_time:.9e} s, t_end={case.run_time:.9e} s")
    print(f"  CFL dt={dt:.9e} s, expected Nt={int(case.run_time / dt):,}")
    print(f"  density perturbation max={2 * density_perturbation_amplitude(case):.3%}")
    print(f"  generated input size~{input_gib:.2f} GiB")


def generate_case(case, output_root, seed=42, force=False):
    output_dir = Path(output_root) / case.name
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"{output_dir} exists; pass --force to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    write_fields(case, output_dir)
    write_particles(case, output_dir, seed=seed)
    report_case(case)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("smoke", "pilot", "spectrum", "all"), default="pilot")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "generated",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    names = tuple(CASES) if args.case == "all" else (args.case,)
    for offset, name in enumerate(names):
        generate_case(
            CASES[name],
            args.output_root,
            seed=args.seed + offset,
            force=args.force,
        )


if __name__ == "__main__":
    main()
