from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    base_dir = Path("results/phase4/week1_flow_validation")

    positions_path = base_dir / "positions.npy"
    velocities_path = base_dir / "velocities.npy"

    positions = np.load(positions_path)
    velocities = np.load(velocities_path)

    # Use first saved frame
    pos0 = positions[0]
    vel0 = velocities[0]

    y = pos0[:, 1]
    z = pos0[:, 2]
    radial_distance = np.sqrt(y**2 + z**2)

    axial_velocity = vel0[:, 0]

    # Sort by radial distance for cleaner line plot
    order = np.argsort(radial_distance)
    r_sorted = radial_distance[order]
    u_sorted = axial_velocity[order]

    # Plot 1: radial Poiseuille profile
    plt.figure(figsize=(7, 5))
    plt.scatter(radial_distance, axial_velocity, s=12, alpha=0.5, label="Agents")
    plt.plot(r_sorted, u_sorted, linewidth=2, label="Measured profile")
    plt.xlabel("Radial distance from vessel center")
    plt.ylabel("Axial velocity")
    plt.title("Poiseuille flow validation: velocity decreases toward wall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(base_dir / "poiseuille_radial_profile.png", dpi=300)
    plt.close()

    # Plot 2: velocity distribution histogram
    plt.figure(figsize=(7, 5))
    plt.hist(axial_velocity, bins=30)
    plt.xlabel("Axial velocity")
    plt.ylabel("Number of agents")
    plt.title("Velocity distribution under Poiseuille flow")
    plt.tight_layout()
    plt.savefig(base_dir / "poiseuille_velocity_histogram.png", dpi=300)
    plt.close()

    # Plot 3: cross-section colored by velocity
    plt.figure(figsize=(6, 6))
    sc = plt.scatter(y, z, c=axial_velocity, s=18)
    plt.colorbar(sc, label="Axial velocity")
    plt.xlabel("y position")
    plt.ylabel("z position")
    plt.title("Cross-section velocity map")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(base_dir / "poiseuille_cross_section_velocity.png", dpi=300)
    plt.close()

    print("Saved:")
    print(base_dir / "poiseuille_radial_profile.png")
    print(base_dir / "poiseuille_velocity_histogram.png")
    print(base_dir / "poiseuille_cross_section_velocity.png")


if __name__ == "__main__":
    main()