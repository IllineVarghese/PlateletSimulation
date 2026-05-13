from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from src.simulation.flow_fields import poiseuille_velocity_radial
from src.simulation.shear_stress import (
    normalize_shear_stress,
    poiseuille_shear_rate_radial,
    poiseuille_shear_stress_radial,
    radial_distance_yz,
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def initialize_positions(
    n_agents: int,
    radius: float,
    length: float,
) -> np.ndarray:
    positions = np.zeros((n_agents, 3), dtype=np.float32)

    for i in range(n_agents):
        while True:
            y = np.random.uniform(-radius, radius)
            z = np.random.uniform(-radius, radius)

            if y * y + z * z <= radius * radius:
                break

        x = np.random.uniform(0.0, length)

        positions[i] = [x, y, z]

    return positions


def compute_flow_velocity(
    positions: np.ndarray,
    radius: float,
    vmax: float,
) -> np.ndarray:
    velocities = np.zeros_like(positions)

    for i, (_, y, z) in enumerate(positions):
        r = radial_distance_yz(float(y), float(z))

        ux = poiseuille_velocity_radial(r, radius, vmax)

        velocities[i, 0] = ux

    return velocities


def compute_shear_fields(
    positions: np.ndarray,
    radius: float,
    vmax: float,
    viscosity: float,
    reference_shear_stress: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shear_rates = np.zeros(len(positions), dtype=np.float32)
    shear_stresses = np.zeros(len(positions), dtype=np.float32)
    normalized_shear = np.zeros(len(positions), dtype=np.float32)

    for i, (_, y, z) in enumerate(positions):
        r = radial_distance_yz(float(y), float(z))

        shear_rate = poiseuille_shear_rate_radial(r, radius, vmax)

        shear_stress = poiseuille_shear_stress_radial(
            r,
            radius,
            vmax,
            viscosity,
        )

        shear_input = normalize_shear_stress(
            shear_stress,
            reference_shear_stress,
        )

        shear_rates[i] = shear_rate
        shear_stresses[i] = shear_stress
        normalized_shear[i] = shear_input

    return shear_rates, shear_stresses, normalized_shear


def advance_positions(
    positions: np.ndarray,
    velocities: np.ndarray,
    dt: float,
    length: float,
) -> np.ndarray:
    new_positions = positions.copy()

    new_positions[:, 0] += velocities[:, 0] * dt

    # periodic wrap in x direction
    new_positions[:, 0] %= length

    return new_positions


def run_phase4(config_path: str = "configs/phase4_tube_flow.yaml") -> None:
    config = load_config(config_path)

    simulation = config["simulation"]
    geometry = config["geometry"]
    flow = config["flow"]
    output = config["output"]

    n_agents = int(simulation["num_agents"])
    dt = float(simulation["dt"])
    steps = int(simulation["steps"])

    radius = float(geometry["radius"])
    length = float(geometry["length"])

    vmax = float(flow["vmax"])
    viscosity = float(flow["viscosity"])
    reference_shear_stress = float(flow["reference_shear_stress"])

    save_every = int(output["save_every"])

    base_dir = Path(output["base_dir"])
    base_dir.mkdir(parents=True, exist_ok=True)

    positions = initialize_positions(
        n_agents=n_agents,
        radius=radius,
        length=length,
    )

    saved_positions = []
    saved_velocities = []

    saved_shear_rates = []
    saved_shear_stresses = []
    saved_normalized_shear = []

    for step in range(steps):
        velocities = compute_flow_velocity(
            positions,
            radius,
            vmax,
        )

        shear_rates, shear_stresses, normalized_shear = compute_shear_fields(
            positions,
            radius,
            vmax,
            viscosity,
            reference_shear_stress,
        )

        positions = advance_positions(
            positions,
            velocities,
            dt,
            length,
        )

        if step % save_every == 0:
            saved_positions.append(positions.copy())
            saved_velocities.append(velocities.copy())

            saved_shear_rates.append(shear_rates.copy())
            saved_shear_stresses.append(shear_stresses.copy())
            saved_normalized_shear.append(normalized_shear.copy())

    saved_positions = np.asarray(saved_positions, dtype=np.float32)
    saved_velocities = np.asarray(saved_velocities, dtype=np.float32)

    saved_shear_rates = np.asarray(saved_shear_rates, dtype=np.float32)
    saved_shear_stresses = np.asarray(saved_shear_stresses, dtype=np.float32)
    saved_normalized_shear = np.asarray(saved_normalized_shear, dtype=np.float32)

    np.save(base_dir / "positions.npy", saved_positions)
    np.save(base_dir / "velocities.npy", saved_velocities)

    np.save(base_dir / "shear_rates.npy", saved_shear_rates)
    np.save(base_dir / "shear_stresses.npy", saved_shear_stresses)
    np.save(base_dir / "normalized_shear.npy", saved_normalized_shear)

    print(f"Saved positions: {saved_positions.shape}")
    print(f"Saved velocities: {saved_velocities.shape}")

    print(f"Saved shear rates: {saved_shear_rates.shape}")
    print(f"Saved shear stresses: {saved_shear_stresses.shape}")
    print(f"Saved normalized shear: {saved_normalized_shear.shape}")

    print(f"Output folder: {base_dir}")


if __name__ == "__main__":
    run_phase4()