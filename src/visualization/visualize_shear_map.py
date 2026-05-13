from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    base_dir = Path("results/phase4/week1_flow_validation")

    positions = np.load(base_dir / "positions.npy")
    shear_rates = np.load(base_dir / "shear_rates.npy")
    shear_stresses = np.load(base_dir / "shear_stresses.npy")
    normalized_shear = np.load(base_dir / "normalized_shear.npy")

    pos0 = positions[0]
    shear_rate0 = shear_rates[0]
    shear_stress0 = shear_stresses[0]
    normalized0 = normalized_shear[0]

    y = pos0[:, 1]
    z = pos0[:, 2]
    radial_distance = np.sqrt(y**2 + z**2)

    order = np.argsort(radial_distance)

    # Plot 1: shear rate vs radius
    plt.figure(figsize=(7, 5))
    plt.scatter(radial_distance, shear_rate0, s=12, alpha=0.5, label="Agents")
    plt.plot(radial_distance[order], shear_rate0[order], linewidth=2, label="Measured shear rate")
    plt.xlabel("Radial distance from vessel center")
    plt.ylabel("Shear rate")
    plt.title("Poiseuille shear validation: shear increases toward wall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(base_dir / "shear_rate_vs_radius.png", dpi=300)
    plt.close()

    # Plot 2: shear stress vs radius
    plt.figure(figsize=(7, 5))
    plt.scatter(radial_distance, shear_stress0, s=12, alpha=0.5, label="Agents")
    plt.plot(radial_distance[order], shear_stress0[order], linewidth=2, label="Measured shear stress")
    plt.xlabel("Radial distance from vessel center")
    plt.ylabel("Shear stress")
    plt.title("Poiseuille shear stress increases toward vessel wall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(base_dir / "shear_stress_vs_radius.png", dpi=300)
    plt.close()

    # Plot 3: cross-section colored by shear stress
    plt.figure(figsize=(6, 6))
    sc = plt.scatter(y, z, c=shear_stress0, s=18)
    plt.colorbar(sc, label="Shear stress")
    plt.xlabel("y position")
    plt.ylabel("z position")
    plt.title("Cross-section shear-stress map")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(base_dir / "shear_cross_section_map.png", dpi=300)
    plt.close()

    # Plot 4: normalized GRN shear input
    plt.figure(figsize=(7, 5))
    plt.scatter(radial_distance, normalized0, s=12, alpha=0.5, label="Agents")
    plt.plot(radial_distance[order], normalized0[order], linewidth=2, label="Normalized shear input")
    plt.xlabel("Radial distance from vessel center")
    plt.ylabel("Normalized shear input to GRN")
    plt.title("Normalized InShearStress input increases toward wall")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(base_dir / "normalized_shear_vs_radius.png", dpi=300)
    plt.close()

    print("Saved shear validation plots:")
    print(base_dir / "shear_rate_vs_radius.png")
    print(base_dir / "shear_stress_vs_radius.png")
    print(base_dir / "shear_cross_section_map.png")
    print(base_dir / "normalized_shear_vs_radius.png")


if __name__ == "__main__":
    main()