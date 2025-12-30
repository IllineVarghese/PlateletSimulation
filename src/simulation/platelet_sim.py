import argparse

from platelet_step import run_step


def main(steps: int, device: str):
    print(f"Running on device: {device}")
    print(f"Running simulation for {steps} steps...")
    for i in range(steps):
        print(f"\n--- Step {i+1}/{steps} ---")
        run_step(device)
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    main(args.steps, args.device)
