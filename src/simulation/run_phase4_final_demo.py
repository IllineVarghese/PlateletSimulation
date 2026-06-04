from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    input_dir = Path("results/phase4/week3_cone_geometry")
    output_dir = Path("results/phase4/final_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    positions = np.load(input_dir / "cone_positions.npy")
    velocities = np.load(input_dir / "cone_velocities.npy")
    shear = np.load(input_dir / "cone_normalized_shear.npy")

    n_frames, n_agents = shear.shape

    activation = np.zeros_like(shear)
    stickiness = np.zeros_like(shear)
    morphology = np.zeros_like(shear)

    for frame in range(n_frames):
        if frame == 0:
            activation[frame] = shear[frame]
        else:
            previous = activation[frame - 1]
            current_shear = shear[frame]

            activation[frame] = (
                0.85 * previous
                + 0.15 * current_shear
            )

        stickiness[frame] = np.clip(activation[frame] * 0.8, 0.0, 1.0)
        morphology[frame] = np.clip(activation[frame] * 1.2, 0.0, 1.0)

    rows = []

    for frame in range(n_frames):
        speed = np.linalg.norm(velocities[frame], axis=1)

        rows.append(
            {
                "frame": frame,
                "mean_speed": float(speed.mean()),
                "mean_shear_input": float(shear[frame].mean()),
                "max_shear_input": float(shear[frame].max()),
                "mean_activation": float(activation[frame].mean()),
                "max_activation": float(activation[frame].max()),
                "mean_stickiness": float(stickiness[frame].mean()),
                "mean_morphology": float(morphology[frame].mean()),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "phase4_final_behavior_summary.csv", index=False)

    np.save(output_dir / "positions.npy", positions)
    np.save(output_dir / "velocities.npy", velocities)
    np.save(output_dir / "shear_input.npy", shear)
    np.save(output_dir / "activation.npy", activation)
    np.save(output_dir / "stickiness.npy", stickiness)
    np.save(output_dir / "morphology.npy", morphology)

    print("Saved Phase 4 final demo outputs:")
    print(output_dir / "phase4_final_behavior_summary.csv")
    print(output_dir / "activation.npy")
    print(output_dir / "stickiness.npy")
    print(output_dir / "morphology.npy")
    print()
    print("Frames:", n_frames)
    print("Agents:", n_agents)
    print("Mean final activation:", float(activation[-1].mean()))
    print("Max final activation:", float(activation[-1].max()))


if __name__ == "__main__":
    main()