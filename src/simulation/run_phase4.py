from __future__ import annotations

from pathlib import Path
import yaml
import numpy as np

from src.simulation.flow_fields import poiseuille_velocity_radial, radial_distance_yz


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def initialize_agents(num_agents: int, radius: float, length: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)

    positions = np.zeros((num_agents, 3), dtype=np.float32)

    # x = flow direction, uniformly distributed along vessel length
    positions[:, 0] = rng.uniform(0.0, length, size=num_agents)

    # sample points uniformly inside circular cross-section
    theta = rng.uniform(0.0, 2.0 * np.pi, size=num_agents)
    r = radius * np.sqrt(rng.uniform(0.0, 1.0, size=num_agents))

    positions[:, 1] = r * np.cos(theta)
    positions[:, 2] = r * np.sin(theta)

    return positions


def compute_flow_velocity(positions: np.ndarray, radius: float, vmax: float) -> np.ndarray:
    velocities = np.zeros_like(positions)

    for i, (_, y, z) in enumerate(positions):
        r = radial_distance_yz(float(y), float(z))
        ux = poiseuille_velocity_radial(r, radius, vmax)
        velocities[i, 0] = ux

    return velocities


def run_phase4(config_path: str | Path = "configs/phase4_tube_flow.yaml") -> None:
    config = load_config(config_path)

    sim = config["simulation"]
    geom = config["geometry"]
    flow = config["flow"]
    output = config["output"]

    num_agents = int(sim["num_agents"])
    steps = int(sim["steps"])
    dt = float(sim["dt"])
    seed = int(sim["seed"])

    radius = float(geom["radius"])
    length = float(geom["length"])
    vmax = float(flow["vmax"])

    base_dir = Path(output["base_dir"])
    base_dir.mkdir(parents=True, exist_ok=True)
    save_every = int(output["save_every"])

    positions = initialize_agents(num_agents, radius, length, seed)

    saved_positions = []
    saved_velocities = []

    for step in range(steps):
        velocities = compute_flow_velocity(positions, radius, vmax)

        positions += velocities * dt

        # outlet-to-inlet periodic respawn
        positions[:, 0] = np.where(positions[:, 0] > length, positions[:, 0] - length, positions[:, 0])

        if step % save_every == 0:
            saved_positions.append(positions.copy())
            saved_velocities.append(velocities.copy())

    saved_positions = np.asarray(saved_positions, dtype=np.float32)
    saved_velocities = np.asarray(saved_velocities, dtype=np.float32)

    np.save(base_dir / "positions.npy", saved_positions)
    np.save(base_dir / "velocities.npy", saved_velocities)

    print(f"Saved positions: {saved_positions.shape}")
    print(f"Saved velocities: {saved_velocities.shape}")
    print(f"Output folder: {base_dir}")


if __name__ == "__main__":
    run_phase4()