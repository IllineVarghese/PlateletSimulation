import yaml
from src.simulation.platelet_sim import run_simulation

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    run_simulation(config)

if __name__ == "__main__":
    main()
