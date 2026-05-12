import numpy as np
from pathlib import Path


def main():
    print("\n--- Week 4 Day 3: Adhesion Analysis ---\n")

    # load data
    pos_path = Path("results/positions_steps.npy")
    adh_path = Path("results/adhesion_steps.npy")

    positions = np.load(pos_path)      # shape: (steps, N, 3)
    adhesion = np.load(adh_path)       # shape: (steps, N)

    steps, N, _ = positions.shape

    near_wall_vals = []
    center_vals = []

    for t in range(steps):
        for i in range(N):
            y = positions[t, i, 1]
            a = adhesion[t, i]

            # same rule as simulation
            if (y < 0.3) or (y > 0.7):
                near_wall_vals.append(a)
            else:
                center_vals.append(a)

    near_wall_mean = np.mean(near_wall_vals) if near_wall_vals else 0.0
    center_mean = np.mean(center_vals) if center_vals else 0.0

    print(f"Near-wall adhesion mean : {near_wall_mean:.5f}")
    print(f"Center adhesion mean    : {center_mean:.5f}")

    if near_wall_mean > center_mean:
        print("\n✔ SUCCESS: Near-wall adhesion is higher")
    else:
        print("\n⚠ WARNING: No clear wall effect detected")


if __name__ == "__main__":
    main()