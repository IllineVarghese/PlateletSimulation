from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import imageio.v2 as imageio


def make_tube_mesh(radius: float, length: float) -> pv.PolyData:
    tube = pv.Cylinder(
        center=(length / 2.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0),
        radius=radius,
        height=length,
        resolution=96,
    )
    return tube


def render_snapshot(base_dir: Path, radius: float = 1.0, length: float = 8.0) -> None:
    positions = np.load(base_dir / "positions.npy")
    velocities = np.load(base_dir / "velocities.npy")

    pos = positions[0]
    vel_mag = np.linalg.norm(velocities[0], axis=1)

    tube = make_tube_mesh(radius, length)

    plotter = pv.Plotter(off_screen=True, window_size=(1400, 900))
    plotter.set_background("white")

    plotter.add_mesh(
        tube,
        color="lightblue",
        opacity=0.18,
        smooth_shading=True,
        show_edges=False,
    )

    cloud = pv.PolyData(pos)
    cloud["velocity"] = vel_mag

    plotter.add_mesh(
        cloud,
        scalars="velocity",
        cmap="viridis",
        point_size=10,
        render_points_as_spheres=True,
        clim=[0.0, 2.0],
        scalar_bar_args={"title": "Velocity"},
    )

    plotter.add_title("Phase 4 Week 1: Poiseuille Flow in Cylindrical Vessel", font_size=14)
    plotter.camera_position = [
        (6.5, -5.5, 3.0),
        (4.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ]

    out_path = base_dir / "poiseuille_3d_snapshot.png"
    plotter.screenshot(out_path)
    plotter.close()

    print(f"Saved 3D snapshot: {out_path}")


def render_video(base_dir: Path, radius: float = 1.0, length: float = 8.0) -> None:
    positions = np.load(base_dir / "positions.npy")
    velocities = np.load(base_dir / "velocities.npy")

    frames = []
    tube = make_tube_mesh(radius, length)

    for frame_id in range(len(positions)):
        pos = positions[frame_id]
        vel_mag = np.linalg.norm(velocities[frame_id], axis=1)

        plotter = pv.Plotter(off_screen=True, window_size=(1200, 800))
        plotter.set_background("white")

        plotter.add_mesh(
            tube,
            color="lightblue",
            opacity=0.16,
            smooth_shading=True,
            show_edges=False,
        )

        cloud = pv.PolyData(pos)
        cloud["velocity"] = vel_mag

        plotter.add_mesh(
            cloud,
            scalars="velocity",
            cmap="viridis",
            point_size=9,
            render_points_as_spheres=True,
            clim=[0.0, 2.0],
            scalar_bar_args={"title": "Velocity"},
        )

        plotter.add_text(
            f"Poiseuille flow | frame {frame_id + 1}/{len(positions)}",
            position="upper_left",
            font_size=11,
            color="black",
        )

        plotter.camera_position = [
            (6.5, -5.5, 3.0),
            (4.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ]

        image = plotter.screenshot(return_img=True)
        frames.append(image)
        plotter.close()

    out_path = base_dir / "poiseuille_3d_velocity_video.mp4"
    imageio.mimsave(out_path, frames, fps=8)

    print(f"Saved 3D video: {out_path}")


def main() -> None:
    base_dir = Path("results/phase4/week1_flow_validation")
    render_snapshot(base_dir)
    render_video(base_dir)


if __name__ == "__main__":
    main()