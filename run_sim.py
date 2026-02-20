import argparse
import yaml
from pathlib import Path

from src.simulation.platelet_sim import run_simulation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file"
    )

    args = parser.parse_args()

    config_path = Path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_simulation(config)


if __name__ == "__main__":
    main()