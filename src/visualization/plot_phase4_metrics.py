from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    base_dir = Path("results/phase4/week1_flow_validation")

    positions = np.load(base_dir / "positions.npy")
    velocities = np.load(base_dir / "velocities.npy")
    shear_rates = np.load(base_dir / "shear_rates.npy")
    shear_stresses = np.load(base_dir / "shear_stresses.npy")
    normalized_shear = np.load(base_dir / "normalized_shear.npy")

    n_frames = positions.shape[0]

    rows = []

    for frame in range(n_frames):
        speed = np.linalg.norm(velocities[frame], axis=1)

        rows.append(
            {
                "frame": frame,
                "mean_velocity": float(np.mean(speed)),
                "max_velocity": float(np.max(speed)),
                "mean_shear_rate": float(np.mean(shear_rates[frame])),
                "max_shear_rate": float(np.max(shear_rates[frame])),
                "mean_shear_stress": float(np.mean(shear_stresses[frame])),
                "max_shear_stress": float(np.max(shear_stresses[frame])),
                "mean_normalized_shear": float(np.mean(normalized_shear[frame])),
                "max_normalized_shear": float(np.max(normalized_shear[frame])),
            }
        )

    df = pd.DataFrame(rows)
    csv_path = base_dir / "phase4_week2_shear_summary.csv"
    df.to_csv(csv_path, index=False)

    # Plot velocity and normalized shear over saved frames
    plt.figure(figsize=(8, 5))
    plt.plot(df["frame"], df["mean_velocity"], label="Mean velocity")
    plt.plot(df["frame"], df["mean_normalized_shear"], label="Mean normalized shear")
    plt.xlabel("Saved frame")
    plt.ylabel("Value")
    plt.title("Phase 4 shear-input summary over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(base_dir / "phase4_week2_velocity_shear_summary.png", dpi=300)
    plt.close()

    # Plot max shear stress over time
    plt.figure(figsize=(8, 5))
    plt.plot(df["frame"], df["max_shear_stress"], label="Maximum shear stress")
    plt.plot(df["frame"], df["mean_shear_stress"], label="Mean shear stress")
    plt.xlabel("Saved frame")
    plt.ylabel("Shear stress")
    plt.title("Shear stress remains stable in cylindrical Poiseuille flow")
    plt.legend()
    plt.tight_layout()
    plt.savefig(base_dir / "phase4_week2_shear_stress_timeseries.png", dpi=300)
    plt.close()

    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {base_dir / 'phase4_week2_velocity_shear_summary.png'}")
    print(f"Saved plot: {base_dir / 'phase4_week2_shear_stress_timeseries.png'}")


if __name__ == "__main__":
    main()