from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    base_dir = Path("results/phase4/final_demo")
    out_dir = base_dir

    summary = pd.read_csv(base_dir / "phase4_final_behavior_summary.csv")

    plt.figure(figsize=(8, 5))
    plt.plot(summary["frame"], summary["mean_shear_input"], label="Mean shear input")
    plt.plot(summary["frame"], summary["mean_activation"], label="Mean activation")
    plt.plot(summary["frame"], summary["mean_stickiness"], label="Mean stickiness")
    plt.plot(summary["frame"], summary["mean_morphology"], label="Mean morphology")
    plt.xlabel("Frame")
    plt.ylabel("Normalized value")
    plt.title("Phase 4 final demo: flow-driven platelet behavior")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "phase4_final_behavior_timeseries.png", dpi=300)
    plt.close()

    activation = np.load(base_dir / "activation.npy")
    stickiness = np.load(base_dir / "stickiness.npy")
    morphology = np.load(base_dir / "morphology.npy")

    plt.figure(figsize=(8, 5))
    plt.boxplot(
        [
            activation[-1],
            stickiness[-1],
            morphology[-1],
        ],
        labels=["Activation", "Stickiness", "Morphology"],
    )
    plt.ylabel("Normalized value")
    plt.title("Final frame behavior distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "phase4_final_behavior_boxplot.png", dpi=300)
    plt.close()

    print("Saved:")
    print(out_dir / "phase4_final_behavior_timeseries.png")
    print(out_dir / "phase4_final_behavior_boxplot.png")


if __name__ == "__main__":
    main()