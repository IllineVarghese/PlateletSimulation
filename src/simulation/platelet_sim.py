import argparse
from pathlib import Path
import numpy as np
import warp as wp

from src.simulation.platelet_step import run_step


def main(steps: int, device: str):
    print(f"Running on device: {device}")
    print(f"Running simulation for {steps} steps...")

    all_positions = []
    all_adhesion = []

    for i in range(steps):
        print(f"\n--- Step {i + 1}/{steps} ---")

        # Week 4 Day 1:
        # run_step can now return either:
        # 1) positions
        # 2) (positions, adhesion_strength)
        result = run_step(device)

        if isinstance(result, tuple) and len(result) == 2:
            positions, adhesion_strength = result
        else:
            positions = result
            adhesion_strength = None

        # positions must be wp.array (N, 3)
        pos_np = positions.numpy()
        all_positions.append(pos_np)

        print("Updated positions:")
        print(pos_np)

        if adhesion_strength is not None:
            if isinstance(adhesion_strength, wp.array):
                adhesion_np = adhesion_strength.numpy()
            else:
                adhesion_np = np.asarray(adhesion_strength)

            all_adhesion.append(adhesion_np)

            print("Adhesion strengths:")
            print(adhesion_np)

    # ---- save results ----
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_positions = np.stack(all_positions)  # shape: (steps, N, 3)
    pos_path = out_dir / "positions_steps.npy"
    np.save(pos_path, all_positions)

    print(f"\nSaved positions to {pos_path}")

    if len(all_adhesion) > 0:
        all_adhesion = np.stack(all_adhesion)  # shape: (steps, N)
        adh_path = out_dir / "adhesion_steps.npy"
        np.save(adh_path, all_adhesion)
        print(f"Saved adhesion strengths to {adh_path}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    main(args.steps, args.device)