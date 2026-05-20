from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from src.simulation.cone_geometry import cone_radius_at_position
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
    radius_start: float,
    length: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    positions = np.zeros((n_agents, 3), dtype=np.float32)

    for i in range(n_agents):
        while True:
            y = rng.uniform(-radius_start, radius_start)
            z = rng.uniform(-radius_start, radius_start)
            if y * y + z * z <= radius_start * radius_start:
                break

        x = rng.uniform(0.0, length)
        positions[i] = [x, y, z]

    return positions


def project_inside_local_radius(
    positions: np.ndarray,
    length: float,
    radius_start: float,
    radius_end: float,
) -> np.ndarray:
    corrected = positions.copy()

    for i, (x, y, z) in enumerate(corrected):
        local_radius = cone_radius_at_position(
            float(x),
            length,
            radius_start,
            radius_end,
        )

        r = radial_distance_yz(float(y), float(z))

        if r > local_radius:
            scale = local_radius / r
            corrected[i, 1] = y * scale
            corrected[i, 2] = z * scale

    return corrected


def compute_cone_flow_and_shear(
    positions: np.ndarray,
    length: float,
    radius_start: float,
    radius_end: float,
    vmax: float,
    viscosity: float,
    reference_shear_stress: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    velocities = np.zeros_like(positions)

    shear_rates = np.zeros(len(positions), dtype=np.float32)
    shear_stresses = np.zeros(len(positions), dtype=np.float32)
    normalized_shear = np.zeros(len(positions), dtype=np.float32)

    for i, (x, y, z) in enumerate(positions):
        local_radius = cone_radius_at_position(
            float(x),
            length,
            radius_start,
            radius_end,
        )

        r = radial_distance_yz(float(y), float(z))

        ux = poiseuille_velocity_radial(r, local_radius, vmax)
        velocities[i, 0] = ux

        shear_rate = poiseuille_shear_rate_radial(r, local_radius, vmax)
        shear_stress = poiseuille_shear_stress_radial(
            r,
            local_radius,
            vmax,
            viscosity,
        )

        shear_rates[i] = shear_rate
        shear_stresses[i] = shear_stress
        normalized_shear[i] = normalize_shear_stress(
            shear_stress,
            reference_shear_stress,
        )

    return velocities, shear_rates, shear_stresses, normalized_shear


def run_phase4_cone(config_path: str = "configs/phase4_cone_flow.yaml") -> None:
    config = load_config(config_path)

    sim = config["simulation"]
    geom = config["geometry"]
    flow = config["flow"]
    output = config["output"]

    n_agents = int(sim["num_agents"])
    steps = int(sim["steps"])
    dt = float(sim["dt"])
    seed = int(sim["seed"])

    radius_start = float(geom["radius_start"])
    radius_end = float(geom["radius_end"])
    length = float(geom["length"])

    vmax = float(flow["vmax"])
    viscosity = float(flow["viscosity"])
    reference_shear_stress = float(flow["reference_shear_stress"])

    save_every = int(output["save_every"])
    base_dir = Path(output["base_dir"])
    base_dir.mkdir(parents=True, exist_ok=True)

    positions = initialize_positions(
        n_agents=n_agents,
        radius_start=radius_start,
        length=length,
        seed=seed,
    )

    positions = project_inside_local_radius(
        positions,
        length,
        radius_start,
        radius_end,
    )

    saved_positions = []
    saved_velocities = []
    saved_shear_rates = []
    saved_shear_stresses = []
    saved_normalized_shear = []
    saved_local_radii = []

    for step in range(steps):
        positions = project_inside_local_radius(
            positions,
            length,
            radius_start,
            radius_end,
        )

        velocities, shear_rates, shear_stresses, normalized_shear = compute_cone_flow_and_shear(
            positions,
            length,
            radius_start,
            radius_end,
            vmax,
            viscosity,
            reference_shear_stress,
        )

        positions[:, 0] += velocities[:, 0] * dt
        positions[:, 0] %= length

        local_radii = np.array(
            [
                cone_radius_at_position(float(x), length, radius_start, radius_end)
                for x in positions[:, 0]
            ],
            dtype=np.float32,
        )

        if step % save_every == 0:
            saved_positions.append(positions.copy())
            saved_velocities.append(velocities.copy())
            saved_shear_rates.append(shear_rates.copy())
            saved_shear_stresses.append(shear_stresses.copy())
            saved_normalized_shear.append(normalized_shear.copy())
            saved_local_radii.append(local_radii.copy())

    saved_positions = np.asarray(saved_positions, dtype=np.float32)
    saved_velocities = np.asarray(saved_velocities, dtype=np.float32)
    saved_shear_rates = np.asarray(saved_shear_rates, dtype=np.float32)
    saved_shear_stresses = np.asarray(saved_shear_stresses, dtype=np.float32)
    saved_normalized_shear = np.asarray(saved_normalized_shear, dtype=np.float32)
    saved_local_radii = np.asarray(saved_local_radii, dtype=np.float32)

    np.save(base_dir / "cone_positions.npy", saved_positions)
    np.save(base_dir / "cone_velocities.npy", saved_velocities)
    np.save(base_dir / "cone_shear_rates.npy", saved_shear_rates)
    np.save(base_dir / "cone_shear_stresses.npy", saved_shear_stresses)
    np.save(base_dir / "cone_normalized_shear.npy", saved_normalized_shear)
    np.save(base_dir / "cone_local_radii.npy", saved_local_radii)

    print(f"Saved cone positions: {saved_positions.shape}")
    print(f"Saved cone velocities: {saved_velocities.shape}")
    print(f"Saved cone shear rates: {saved_shear_rates.shape}")
    print(f"Saved cone shear stresses: {saved_shear_stresses.shape}")
    print(f"Saved cone normalized shear: {saved_normalized_shear.shape}")
    print(f"Saved cone local radii: {saved_local_radii.shape}")
    print(f"Output folder: {base_dir}")


if __name__ == "__main__":
    run_phase4_cone()