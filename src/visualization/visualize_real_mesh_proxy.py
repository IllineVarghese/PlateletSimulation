from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    base_dir = Path("results/phase4/real_mesh_proxy")
    out_dir = base_dir

    positions = np.load(base_dir / "real_mesh_proxy_positions.npy")
    shear = np.load(base_dir / "real_mesh_proxy_shear_input.npy")
    activation = np.load(base_dir / "real_mesh_proxy_activation.npy")
    stickiness = np.load(base_dir / "real_mesh_proxy_stickiness.npy")
    morphology = np.load(base_dir / "real_mesh_proxy_morphology.npy")
    summary = pd.read_csv(base_dir / "real_mesh_proxy_summary.csv")

    frame = -1
    final_pos = positions[frame]
    final_shear = shear[frame]

    # 1) Mesh-derived vessel proxy: agents colored by shear
    plt.figure(figsize=(9, 5))
    scatter = plt.scatter(
        final_pos[:, 0],
        final_pos[:, 1],
        c=final_shear,
        s=10,
        alpha=0.8,
    )
    plt.colorbar(scatter, label="Normalized shear input")
    plt.xlabel("Mesh-derived axial position")
    plt.ylabel("Lateral position")
    plt.title("Agents flowing through real mesh-derived vessel proxy")
    plt.tight_layout()
    plt.savefig(out_dir / "real_mesh_proxy_agents_shear.png", dpi=300)
    plt.close()

    # 2) Behavior response over time
    frames = np.arange(activation.shape[0])

    plt.figure(figsize=(9, 5))
    plt.plot(frames, shear.mean(axis=1), label="Mean shear input")
    plt.plot(frames, activation.mean(axis=1), label="Mean activation")
    plt.plot(frames, stickiness.mean(axis=1), label="Mean stickiness")
    plt.plot(frames, morphology.mean(axis=1), label="Mean morphology")
    plt.xlabel("Frame")
    plt.ylabel("Normalized value")
    plt.title("Real mesh proxy: shear-driven platelet behavior")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "real_mesh_proxy_behavior_timeseries.png", dpi=300)
    plt.close()

    # 3) Final behavior distribution
    plt.figure(figsize=(8, 5))
    plt.boxplot(
        [activation[-1], stickiness[-1], morphology[-1]],
        tick_labels=["Activation", "Stickiness", "Morphology"],
    )
    plt.ylabel("Normalized value")
    plt.title("Final behavior distribution in real mesh proxy")
    plt.tight_layout()
    plt.savefig(out_dir / "real_mesh_proxy_behavior_boxplot.png", dpi=300)
    plt.close()

    # 4) Summary figure with mesh metadata
    labels = ["Vertices", "Indices", "Agents", "Frames"]
    values = [
        int(summary["vertex_count"].iloc[0]),
        int(summary["index_count"].iloc[0]),
        int(summary["agents"].iloc[0]),
        int(summary["frames"].iloc[0]),
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.ylabel("Count")
    plt.title("Real mesh proxy simulation metadata")
    plt.tight_layout()
    plt.savefig(out_dir / "real_mesh_proxy_metadata.png", dpi=300)
    plt.close()

    print("Saved real mesh proxy visualizations:")
    print(out_dir / "real_mesh_proxy_agents_shear.png")
    print(out_dir / "real_mesh_proxy_behavior_timeseries.png")
    print(out_dir / "real_mesh_proxy_behavior_boxplot.png")
    print(out_dir / "real_mesh_proxy_metadata.png")


if __name__ == "__main__":
    main()