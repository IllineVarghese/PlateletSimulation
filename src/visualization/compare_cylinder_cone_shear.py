from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CYLINDER_DIR = Path("results/phase4/week1_flow_validation")
CONE_DIR = Path("results/phase4/week3_cone_geometry")


def main() -> None:
    cyl_shear = np.load(CYLINDER_DIR / "normalized_shear.npy")
    cone_shear = np.load(CONE_DIR / "cone_normalized_shear.npy")

    cyl_stress = np.load(CYLINDER_DIR / "shear_stresses.npy")
    cone_stress = np.load(CONE_DIR / "cone_shear_stresses.npy")

    cyl_mean = cyl_shear.mean(axis=1)
    cone_mean = cone_shear.mean(axis=1)

    cyl_max = cyl_shear.max(axis=1)
    cone_max = cone_shear.max(axis=1)

    frames = np.arange(len(cyl_mean))

    summary = pd.DataFrame(
        {
            "frame": frames,
            "cylinder_mean_normalized_shear": cyl_mean,
            "cone_mean_normalized_shear": cone_mean,
            "cylinder_max_normalized_shear": cyl_max,
            "cone_max_normalized_shear": cone_max,
            "cylinder_mean_shear_stress": cyl_stress.mean(axis=1),
            "cone_mean_shear_stress": cone_stress.mean(axis=1),
            "cylinder_max_shear_stress": cyl_stress.max(axis=1),
            "cone_max_shear_stress": cone_stress.max(axis=1),
        }
    )

    out_csv = CONE_DIR / "week3_cylinder_vs_cone_shear_summary.csv"
    summary.to_csv(out_csv, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(frames, cyl_mean, label="Cylinder mean InShearStress")
    plt.plot(frames, cone_mean, label="Cone mean InShearStress")
    plt.plot(frames, cyl_max, "--", label="Cylinder max InShearStress")
    plt.plot(frames, cone_max, "--", label="Cone max InShearStress")
    plt.xlabel("Saved frame")
    plt.ylabel("Normalized shear input")
    plt.title("Cylinder vs cone: vessel narrowing changes shear input")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CONE_DIR / "week3_cylinder_vs_cone_normalized_shear.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.boxplot(
        [cyl_shear.flatten(), cone_shear.flatten()],
        labels=["Cylinder", "Cone"],
    )
    plt.ylabel("Normalized InShearStress")
    plt.title("Distribution of GRN shear input: cylinder vs cone")
    plt.tight_layout()
    plt.savefig(CONE_DIR / "week3_shear_distribution_boxplot.png", dpi=300)
    plt.close()

    print(f"Saved CSV: {out_csv}")
    print(f"Saved plot: {CONE_DIR / 'week3_cylinder_vs_cone_normalized_shear.png'}")
    print(f"Saved plot: {CONE_DIR / 'week3_shear_distribution_boxplot.png'}")
    print()
    print("Cylinder mean normalized shear:", float(cyl_shear.mean()))
    print("Cone mean normalized shear:", float(cone_shear.mean()))
    print("Cylinder max normalized shear:", float(cyl_shear.max()))
    print("Cone max normalized shear:", float(cone_shear.max()))


if __name__ == "__main__":
    main()