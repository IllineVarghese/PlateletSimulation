import numpy as np
from pathlib import Path
import pyvista as pv


def main():
    print("\n--- Week 4 Day 4: 3D Cylindrical Visualization ---\n")

    pos_path = Path("results/positions_steps.npy")
    adh_path = Path("results/adhesion_steps.npy")

    positions = np.load(pos_path)   # shape: (steps, N, 3)
    adhesion = np.load(adh_path)    # shape: (steps, N)

    # use the last simulation step for visualization
    pts = positions[-1]
    adh = adhesion[-1]

    # --------------------------------------------------
    # Build a cylinder that encloses the platelet cloud
    # Axis direction = x-direction
    # --------------------------------------------------
    x_min = pts[:, 0].min()
    x_max = pts[:, 0].max()
    x_center = 0.5 * (x_min + x_max)

    y_center = pts[:, 1].mean()
    z_center = pts[:, 2].mean()

    yz = pts[:, 1:3] - np.array([y_center, z_center])
    radial_dist = np.sqrt((yz ** 2).sum(axis=1))

    radius = radial_dist.max() + 0.08
    height = (x_max - x_min) + 0.2

    cylinder = pv.Cylinder(
        center=(x_center, y_center, z_center),
        direction=(1, 0, 0),
        radius=radius,
        height=height,
        resolution=100,
    )

    # --------------------------------------------------
    # Create platelet point cloud
    # --------------------------------------------------
    cloud = pv.PolyData(pts)
    cloud["adhesion"] = adh

    near_wall_mean = adhesion.mean()  # fallback
    center_mean = adhesion.mean()     # fallback

    # same rule as your current measurement script
    near_wall_vals = []
    center_vals = []

    for i in range(len(pts)):
        y = pts[i, 1]
        a = adh[i]

        if (y < 0.4) or (y > 0.6):
            near_wall_vals.append(a)
        else:
            center_vals.append(a)

    if len(near_wall_vals) > 0:
        near_wall_mean = np.mean(near_wall_vals)

    if len(center_vals) > 0:
        center_mean = np.mean(center_vals)

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    plotter = pv.Plotter(window_size=(1400, 900), off_screen=True)

    # translucent cylinder
    plotter.add_mesh(
        cylinder,
        color="lightblue",
        opacity=0.15,
        smooth_shading=True,
    )

    # cylinder outline
    plotter.add_mesh(
        cylinder.extract_feature_edges(),
        color="black",
        line_width=1,
    )

    # platelet points
    plotter.add_points(
        cloud,
        scalars="adhesion",
        cmap="Reds",
        render_points_as_spheres=True,
        point_size=24,
        clim=[0.0, max(0.2, float(adh.max()) + 1e-6)],
        scalar_bar_args={"title": "Adhesion"},
    )

    # add title and metrics
    plotter.add_text(
        "Week 4 Day 4: 3D Cylindrical Platelet Adhesion",
        position="upper_edge",
        font_size=10,
        color="black",
    )

    plotter.add_text(
        f"Near-wall mean adhesion: {near_wall_mean:.5f}\n"
        f"Center mean adhesion: {center_mean:.5f}",
        position="upper_left",
        font_size=10,
        color="black",
    )

    # nice camera angle
    plotter.camera_position = "iso"

    # save screenshot
    out_path = "results/week4_cylinder_adhesion.png"
    plotter.screenshot(out_path)

    print(f"Saved screenshot to: {out_path}")

    # show interactive window
    #plotter.show()


if __name__ == "__main__":
    main()