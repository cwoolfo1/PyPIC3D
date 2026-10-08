"""Generate physical t=0 dark Harris inputs beside this script."""

from pathlib import Path
import sys

import numpy as np
import toml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from demos.standard_yee.reconnection_2d.initial_data import build_particle_arrays


def harris_scales(config):
    """Return d_e, force-field B0, sheet width, drift and thermal speed in SI."""
    simulation, loading = config["simulation_parameters"], config["initial_data"]
    electron = config["particle1"]
    n0, mass, charge = loading["sheet_density"], electron["mass"], abs(electron["charge"])
    vth = electron["vth"]
    omega_pe = charge * np.sqrt(n0 / (mass * simulation["eps"]))
    skin_depth = simulation["C"] / omega_pe
    half_width = loading["sheet_half_width_skin_depth"] * skin_depth
    b0 = np.sqrt(4 * simulation["mu"] * n0 * mass * vth**2)
    drift = b0 / (2 * simulation["mu"] * charge * n0 * half_width)
    return skin_depth, b0, half_width, drift, vth


def dark_vector_potential(x, z, config):
    """A'_y on (center_x, vertex_y, center_z), independent of y.

    The absolute offset enforces A'_y=0 at the conducting z walls. It is
    physical initial data for a massive field, not an arbitrary gauge shift.
    """
    simulation, loading = config["simulation_parameters"], config["initial_data"]
    _, b0, half_width, _, _ = harris_scales(config)
    mixing = simulation["sin_chi"]
    if mixing == 0:
        raise ValueError("Force-matched dark Harris loading requires nonzero sin_chi")
    lx, lz = simulation["x_wind"], simulation["z_wind"]
    x, z = np.asarray(x)[:, None], np.asarray(z)[None, :]
    log_cosh = np.logaddexp(z / half_width, -z / half_width) - np.log(2.)
    wall = lz / (2 * half_width)
    log_cosh_wall = np.logaddexp(wall, -wall) - np.log(2.)
    kx = 2 * np.pi / lx
    a_harris = b0 * half_width * (log_cosh_wall - log_cosh)
    a_harris = a_harris - loading["perturbation_fraction"] * b0 / kx * np.cos(kx*x) * np.cos(np.pi*z/lz)
    # -sin_chi * curl(A') is the desired Harris field seen by particles.
    return (-a_harris / mixing)[:, None, :]


def generate_initial_data(config):
    """Write physical t=0 arrays beside dark_harris.toml, replacing old inputs."""
    simulation, loading = config["simulation_parameters"], config["initial_data"]
    electrons, positrons = config["particle1"], config["particle2"]
    skin_depth, b0, half_width, drift, vth = harris_scales(config)
    count = loading["n_sheet"] + loading["n_background"]
    if simulation["Ny"] != 1:
        raise ValueError("This Harris loader requires an x-z grid with Ny=1")
    if any(p["N_particles"] != count for p in (electrons, positrons)):
        raise ValueError("Each species must contain n_sheet + n_background particles")
    if (electrons["charge"] != -positrons["charge"] or electrons["mass"] != positrons["mass"]
            or electrons["weight"] != positrons["weight"] or electrons["vth"] != positrons["vth"]):
        raise ValueError("Neutral pair loading requires equal masses, weights and temperatures, and opposite charges")
    lx, ly, lz = (simulation[f"{axis}_wind"] for axis in "xyz")
    sheet_number = 2 * loading["sheet_density"] * half_width * np.tanh(lz/(2*half_width)) * lx * ly
    if not np.isclose(electrons["weight"] * loading["n_sheet"], sheet_number, rtol=1e-10, atol=0.):
        raise ValueError("Particle weight and sheet count must reproduce the configured sheet density")

    x = -lx/2 + np.arange(simulation["Nx"]) * lx/simulation["Nx"]
    z = -lz/2 + np.arange(simulation["Nz"]) * lz/simulation["Nz"]
    ay = dark_vector_potential(x, z, config)
    arrays = build_particle_arrays(
        n_sheet=loading["n_sheet"], n_background=loading["n_background"],
        seed=loading["seed"], lx=lx, lz=lz, thermal_speed=vth,
        half_width=half_width, drift_speed=drift,
    )
    destination = Path(__file__).resolve().parent
    for name in ("electron", "positron"):
        velocity = np.column_stack([arrays[f"{name}_v{axis}"] for axis in "xyz"])
        if np.any(np.sum(velocity**2, axis=1) >= simulation["C"]**2):
            raise ValueError("The Gaussian Harris loading produced a superluminal particle")
        for component in ("x", "y", "z", "vx", "vy", "vz"):
            np.save(destination / f"{name}_{component}.npy", arrays[f"{name}_{component}"])
    np.save(destination / "dark_Ay.npy", ay)
    print(f"{count:,} particles/species; d_e={skin_depth:.9g} m; B0={b0:.9g} T")
    print(f"drift/c={drift/simulation['C']:.6g}; dark_mu*d_e={simulation['dark_mu']*skin_depth:.6g}")
    return {"dark_Ay": ay, **arrays}


def main():
    config = toml.load(Path(__file__).resolve().with_name("dark_harris.toml"))
    generate_initial_data(config)
    print("Initial conditions generated. Run from this directory: PyPIC3D --config dark_harris.toml")


if __name__ == "__main__":
    main()
