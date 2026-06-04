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

    n_frames, n_agents, _ = positions.shape

    x_min = float(positions[:, :, 0].min())
    x_max = float(positions[:, :, 0].max())
    y_center = float(positions[:, :, 1].mean())
    z_center = float(positions[:, :, 2].mean())

    radius = float(
        max(
            np.abs(positions[:, :, 1] - y_center).max(),
            np.abs(positions[:, :, 2] - z_center).max(),
        )
    )

    out_png = out_dir / "phase4_inside_vessel_unity_style_snapshot.png"
    out_mp4 = out_dir / "phase4_inside_vessel_unity_style_flow.mp4"

    fig = plt.figure(figsize=(12, 7), facecolor="black")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")

    theta = np.linspace(0, 2 * np.pi, 70)

    def draw(frame: int):
        ax.clear()
        ax.set_facecolor("black")

        pos = positions[frame]
        act = activation[frame]

        camera_x = x_min + (x_max - x_min) * frame / max(n_frames - 1, 1)
        window = (x_max - x_min) * 0.28

        visible = (pos[:, 0] > camera_x - window) & (pos[:, 0] < camera_x + window)

        if visible.sum() < 30:
            visible = np.ones(n_agents, dtype=bool)

        x = np.linspace(camera_x - window, camera_x + window, 70)
        theta_grid, x_grid = np.meshgrid(theta, x)

        # slightly organic vessel wall
        wave = 1.0 + 0.08 * np.sin(4 * np.pi * (x_grid - x_min) / (x_max - x_min))
        local_radius = radius * wave

        y_grid = y_center + local_radius * np.cos(theta_grid)
        z_grid = z_center + local_radius * np.sin(theta_grid)

        ax.plot_surface(
            x_grid,
            y_grid,
            z_grid,
            color="darkred",
            alpha=0.18,
            linewidth=0,
            shade=True,
        )

        # inner glow/wall guide rings
        for ring_x in np.linspace(camera_x - window, camera_x + window, 5):
            ring_y = y_center + radius * np.cos(theta)
            ring_z = z_center + radius * np.sin(theta)
            ax.plot(
                np.full_like(theta, ring_x),
                ring_y,
                ring_z,
                color="red",
                alpha=0.22,
                linewidth=1,
            )

        sc = ax.scatter(
            pos[visible, 0],
            pos[visible, 1],
            pos[visible, 2],
            c=act[visible],
            cmap="plasma",
            s=45,
            vmin=0,
            vmax=1,
            alpha=0.95,
            edgecolors="none",
        )

        # flow direction streaks
        for yy in np.linspace(-0.55 * radius, 0.55 * radius, 5):
            ax.plot(
                [camera_x - window * 0.8, camera_x + window * 0.8],
                [y_center + yy, y_center + yy],
                [z_center, z_center],
                color="white",
                alpha=0.18,
                linewidth=1,
            )

        ax.set_xlim(camera_x - window, camera_x + window)
        ax.set_ylim(y_center - radius * 1.05, y_center + radius * 1.05)
        ax.set_zlim(z_center - radius * 1.05, z_center + radius * 1.05)

        ax.view_init(elev=6, azim=-82)

        ax.set_title(
            "Phase 4: inside vessel platelet flow with shear-driven activation",
            color="white",
            fontsize=13,
        )

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)

        # remove panes for cleaner Unity-like view
        ax.xaxis.pane.set_alpha(0.0)
        ax.yaxis.pane.set_alpha(0.0)
        ax.zaxis.pane.set_alpha(0.0)

        return sc,

    draw(n_frames - 1)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, facecolor="black")

    anim = FuncAnimation(
        fig,
        draw,
        frames=n_frames,
        interval=90,
        blit=False,
    )

    anim.save(out_mp4, fps=14, dpi=180)
    plt.close()

    print("Saved:")
    print(out_png)
    print(out_mp4)


if __name__ == "__main__":
    main()