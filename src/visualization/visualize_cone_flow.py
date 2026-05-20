from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    base_dir = Path("results/phase4/week3_cone_geometry")

    positions = np.load(base_dir / "cone_positions.npy")
    velocities = np.load(base_dir / "cone_velocities.npy")
    shear_stresses = np.load(base_dir / "cone_shear_stresses.npy")
    normalized_shear = np.load(base_dir / "cone_normalized_shear.npy")
    local_radii = np.load(base_dir / "cone_local_radii.npy")

    pos0 = positions[0]
    vel0 = velocities[0]
    shear0 = shear_stresses[0]
    norm0 = normalized_shear[0]
    radii0 = local_radii[0]

    x = pos0[:, 0]
    y = pos0[:, 1]
    z = pos0[:, 2]

    radial_distance = np.sqrt(y**2 + z**2)
    velocity = vel0[:, 0]

    order = np.argsort(x)

    # Plot 1: cone radius along vessel length
    plt.figure(figsize=(8, 5))
    plt.scatter(x, radii0, s=12, alpha=0.5, label="Agent local radius")
    plt.plot(x[order], radii0[order], linewidth=2, label="Cone radius profile")
    plt.xlabel("Axial position x")
    plt.ylabel("Local vessel radius")
    plt.title("Cone vessel geometry: radius decreases along flow direction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(base_dir / "cone_radius_profile.png", dpi=300)
    plt.close()

    # Plot 2: particle positions inside cone, colored by velocity
    plt.figure(figsize=(8, 5))
    sc = plt.scatter(x, radial_distance, c=velocity, s=16)
    plt.colorbar(sc, label="Axial velocity")
    plt.xlabel("Axial position x")
    plt.ylabel("Radial distance from centerline")
    plt.title("Cone flow: particle distribution colored by velocity")
    plt.tight_layout()
    plt.savefig(base_dir / "cone_particles_velocity.png", dpi=300)
    plt.close()

    # Plot 3: shear stress along cone
    plt.figure(figsize=(8, 5))
    sc = plt.scatter(x, shear0, c=radial_distance, s=16)
    plt.colorbar(sc, label="Radial distance")
    plt.xlabel("Axial position x")
    plt.ylabel("Shear stress")
    plt.title("Cone geometry creates spatially varying shear stress")
    plt.tight_layout()
    plt.savefig(base_dir / "cone_shear_stress_along_x.png", dpi=300)
    plt.close()

    # Plot 4: normalized GRN shear input along cone
    plt.figure(figsize=(8, 5))
    sc = plt.scatter(x, norm0, c=radial_distance, s=16)
    plt.colorbar(sc, label="Radial distance")
    plt.xlabel("Axial position x")
    plt.ylabel("Normalized InShearStress")
    plt.title("Cone flow: GRN shear input varies along vessel")
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(base_dir / "cone_normalized_shear_along_x.png", dpi=300)
    plt.close()

    print("Saved cone flow visualizations:")
    print(base_dir / "cone_radius_profile.png")
    print(base_dir / "cone_particles_velocity.png")
    print(base_dir / "cone_shear_stress_along_x.png")
    print(base_dir / "cone_normalized_shear_along_x.png")


if __name__ == "__main__":
    main()