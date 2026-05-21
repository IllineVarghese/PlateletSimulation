from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv


BASE_DIR = Path("results/phase4/week3_cone_geometry")


def build_cone_mesh(
    length: float,
    radius_start: float,
    radius_end: float,
    resolution: int = 80,
):
    x = np.linspace(0.0, length, resolution)

    points = []

    for xi in x:
        t = xi / length
        radius = radius_start + t * (radius_end - radius_start)

        theta = np.linspace(0.0, 2.0 * np.pi, 50)

        for th in theta:
            y = radius * np.cos(th)
            z = radius * np.sin(th)
            points.append([xi, y, z])

    points = np.asarray(points)

    cloud = pv.PolyData(points)

    return cloud.delaunay_3d(alpha=2.0).extract_geometry()


def main() -> None:
    positions = np.load(BASE_DIR / "cone_positions.npy")
    normalized_shear = np.load(BASE_DIR / "cone_normalized_shear.npy")

    radius_start = 1.0
    radius_end = 0.5
    length = 8.0

    cone_mesh = build_cone_mesh(
        length=length,
        radius_start=radius_start,
        radius_end=radius_end,
    )

    plotter = pv.Plotter(off_screen=True)

    plotter.set_background("white")

    plotter.add_mesh(
        cone_mesh,
        color="lightgray",
        opacity=0.15,
        smooth_shading=True,
    )

    first_frame = positions[0]

    pdata = pv.PolyData(first_frame)

    pdata["shear"] = normalized_shear[0]

    actor = plotter.add_mesh(
        pdata,
        scalars="shear",
        cmap="plasma",
        point_size=8,
        render_points_as_spheres=True,
        clim=[0.0, 1.0],
    )

    plotter.camera_position = "xz"

    image_path = BASE_DIR / "cone_3d_snapshot.png"

    plotter.screenshot(str(image_path))

    print(f"Saved snapshot: {image_path}")

    video_path = BASE_DIR / "cone_3d_flow.mp4"

    plotter.open_movie(str(video_path), framerate=10)

    for frame_id in range(len(positions)):
        frame = positions[frame_id]

        pdata = pv.PolyData(frame)
        pdata["shear"] = normalized_shear[frame_id]

        actor.mapper.dataset.points = pdata.points
        actor.mapper.dataset["shear"] = pdata["shear"]

        plotter.write_frame()

    plotter.close()

    print(f"Saved video: {video_path}")


if __name__ == "__main__":
    main()