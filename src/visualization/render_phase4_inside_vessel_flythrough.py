from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def main() -> None:
    base_dir = Path("results/phase4/real_mesh_proxy")
    out_dir = Path("results/phase4/final_thesis_view")
    out_dir.mkdir(parents=True, exist_ok=True)

    positions = np.load(base_dir / "real_mesh_proxy_positions.npy")
    activation = np.load(base_dir / "real_mesh_proxy_activation.npy")
    shear = np.load(base_dir / "real_mesh_proxy_shear_input.npy")

    n_frames, n_agents, _ = positions.shape

    x_min = positions[:, :, 0].min()
    x_max = positions[:, :, 0].max()
    y_center = positions[:, :, 1].mean()
    z_center = positions[:, :, 2].mean()

    radius = max(
        np.abs(positions[:, :, 1] - y_center).max(),
        np.abs(positions[:, :, 2] - z_center).max(),
    )

    out_mp4 = out_dir / "phase4_inside_vessel_platelet_flythrough.mp4"
    out_png = out_dir / "phase4_inside_vessel_platelet_snapshot.png"

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")

    def draw(frame: int):
        ax.clear()

        pos = positions[frame]
        act = activation[frame]

        # Moving camera window
        camera_x = x_min + (x_max - x_min) * frame / max(n_frames - 1, 1)
        window = (x_max - x_min) * 0.35

        visible = (
            (pos[:, 0] > camera_x - window)
            & (pos[:, 0] < camera_x + window)
        )

        if visible.sum() < 20:
            visible = np.ones(n_agents, dtype=bool)

        # Transparent vessel wall close view
        theta = np.linspace(0, 2 * np.pi, 50)
        x = np.linspace(camera_x - window, camera_x + window, 50)
        theta_grid, x_grid = np.meshgrid(theta, x)

        y_grid = y_center + radius * np.cos(theta_grid)
        z_grid = z_center + radius * np.sin(theta_grid)

        ax.plot_surface(
            x_grid,
            y_grid,
            z_grid,
            alpha=0.10,
            linewidth=0,
        )

        sc = ax.scatter(
            pos[visible, 0],
            pos[visible, 1],
            pos[visible, 2],
            c=act[visible],
            s=35,
            vmin=0,
            vmax=1,
        )

        # Flow direction arrow
        ax.quiver(
            camera_x - window * 0.8,
            y_center,
            z_center,
            window * 0.6,
            0,
            0,
            normalize=False,
        )

        ax.set_xlim(camera_x - window, camera_x + window)
        ax.set_ylim(y_center - radius * 1.2, y_center + radius * 1.2)
        ax.set_zlim(z_center - radius * 1.2, z_center + radius * 1.2)

        ax.set_title(
            f"Inside real mesh-derived vessel: platelet flow and activation | Frame {frame}"
        )
        ax.set_xlabel("Flow direction")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # close camera angle
        ax.view_init(elev=8, azim=-78)

        return sc,

    draw(n_frames - 1)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)

    anim = FuncAnimation(
        fig,
        draw,
        frames=n_frames,
        interval=100,
        blit=False,
    )

    anim.save(out_mp4, fps=12)
    plt.close()

    print("Saved:")
    print(out_png)
    print(out_mp4)


if __name__ == "__main__":
    main()