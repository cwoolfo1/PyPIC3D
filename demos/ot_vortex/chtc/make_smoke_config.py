"""Create a five-step, single-GPU smoke config from the checked-in pilot."""

from pathlib import Path
import sys

import toml


def build_config(demo_dir):
    sys.path.insert(0, str(demo_dir))
    from initial_conditions import CASES

    case = CASES["smoke"]
    config = toml.load(demo_dir / "orszag_tang.toml")
    simulation = config["simulation_parameters"]
    simulation.update(
        {
            "name": "Orszag-Tang local GPU smoke test",
            "output_dir": "runs/smoke",
            "Nt": 5,
            "Nx": case.nx,
            "Ny": case.ny,
            "Nz": 1,
            "x_wind": float(case.lx),
            "y_wind": float(case.ly),
            "t_wind": float(case.run_time),
            "particle_tile_nx": case.nx,
            "particle_tile_ny": case.ny,
            "particle_tile_nz": 1,
        }
    )
    config["plotting"]["plotting_interval"] = 5
    config["ot_vortex"]["case"] = "smoke"

    for section_name, section in config.items():
        if not section_name.startswith(("field", "particle")):
            continue
        for key, value in section.items():
            if isinstance(value, str):
                section[key] = value.replace("generated/pilot/", "generated/smoke/")

    return config


def main():
    demo_dir = Path(__file__).resolve().parents[1]
    config = build_config(demo_dir)
    destination = demo_dir / "orszag_tang_smoke.toml"
    with open(destination, "w") as stream:
        toml.dump(config, stream)
    print(destination)


if __name__ == "__main__":
    main()
