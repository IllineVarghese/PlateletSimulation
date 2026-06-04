from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def main() -> None:
    base_dir = Path("results/phase4/real_mesh_proxy")
    positions = np.load(base_dir / "real_mesh_proxy_positions.npy")
    shear = np.load(base_dir / "real_mesh_proxy_shear_input.npy")

    out_png = base_dir / "real_mesh_proxy_3d_snapshot.png"
    out_mp4 = base_dir / "real_mesh_proxy_3d_flow.mp4"

    n_frames, n_agents, _ = positions.shape

    # Estimate proxy cylinder from particle cloud
    x_min = positions[:, :, 0].min()
    x_max = positions[:, :, 0].max()
    y_center = positions[:, :, 1].mean()
    z_center = positions[:, :, 2].mean()

    radius = max(
        np.abs(positions[:, :, 1] - y_center).max(),
        np.abs(positions[:, :, 2] - z_center).max(),
    )

    theta = np.linspace(0, 2 * np.pi, 40)
    x_line = np.linspace(x_min, x_max, 40)
    theta_grid, x_grid = np.meshgrid(theta, x_line)

    y_grid = y_center + radius * np.cos(theta_grid)
    z_grid = z_center + radius * np.sin(theta_grid)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    def draw_frame(frame: int):
        ax.clear()

        ax.plot_surface(
            x_grid,
            y_grid,
            z_grid,
            alpha=0.12,
            linewidth=0,
        )

        points = ax.scatter(
            positions[frame, :, 0],
            positions[frame, :, 1],
            positions[frame, :, 2],
            c=shear[frame],
            s=12,
            vmin=0,
            vmax=1,
        )

        ax.set_title(
            f"Real mesh-derived 3D vessel proxy | Frame {frame}"
        )
        ax.set_xlabel("Mesh-derived axial position")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_center - radius * 1.3, y_center + radius * 1.3)
        ax.set_zlim(z_center - radius * 1.3, z_center + radius * 1.3)

        ax.view_init(elev=18, azim=-65)
        return points,

    draw_frame(n_frames - 1)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)

    animation = FuncAnimation(
        fig,
        draw_frame,
        frames=n_frames,
        interval=120,
        blit=False,
    )

    animation.save(out_mp4, fps=10)
    plt.close()

    print("Saved:")
    print(out_png)
    print(out_mp4)


if __name__ == "__main__":
    main()