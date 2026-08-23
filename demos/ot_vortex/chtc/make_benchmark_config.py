"""Create a 200-step benchmark config from the checked-in GPU pilot."""

from pathlib import Path
import toml


def main():
    demo_dir = Path(__file__).resolve().parents[1]
    config = toml.load(demo_dir / "orszag_tang.toml")
    simulation = config["simulation_parameters"]
    simulation["name"] = "Orszag-Tang GPU benchmark"
    simulation["output_dir"] = "runs/benchmark"
    simulation["Nt"] = 200
    config["plotting"]["plotting_interval"] = 100
    config["ot_vortex"]["case"] = "benchmark"
    destination = demo_dir / "orszag_tang_benchmark.toml"
    with open(destination, "w") as stream:
        toml.dump(config, stream)
    print(destination)


if __name__ == "__main__":
    main()
