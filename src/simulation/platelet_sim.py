import argparse
from pathlib import Path
import numpy as np
import warp as wp

from platelet_step import run_step


def main(steps: int, device: str):
    print(f"Running on device: {device}")
    print(f"Running simulation for {steps} steps...")

    all_positions = []

    for i in range(steps):
        print(f"\n--- Step {i + 1}/{steps} ---")

        # run_step MUST return positions as a wp.array (N, 3)
        positions = run_step(device)

        # convert to numpy and store
        pos_np = positions.numpy()
        all_positions.append(pos_np)

        print("Updated positions:")
        print(pos_np)

    # ---- save results ----
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_positions = np.stack(all_positions)  # shape: (steps, N, 3)
    out_path = out_dir / "positions_steps.npy"

    np.save(out_path, all_positions)

    print(f"\nSaved results to {out_path}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    main(args.steps, args.device)

