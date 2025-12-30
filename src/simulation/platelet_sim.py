import argparse

from platelet_step import run_step


def main(steps: int):
    print(f"Running simulation for {steps} steps...")
    for i in range(steps):
        print(f"\n--- Step {i+1}/{steps} ---")
        run_step()
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    main(args.steps)
