import numpy as np
from pathlib import Path
import pyvista as pv


def main():
    print("\n--- Month 2 Final: Realistic 3D Cylindrical GRN Platelet Video ---\n")

    pos_path = Path("results/positions_steps.npy")
    adh_path = Path("results/adhesion_steps.npy")

    positions = np.load(pos_path)   # shape: (steps, N, 3)
    adhesion = np.load(adh_path)    # shape: (steps, N)

    steps, n_platelets, _ = positions.shape

    # --------------------------------------------------
    # REMAP AXES FOR VISUALIZATION
    # simulation x -> visualization z  (vertical vessel axis)
    # simulation y -> visualization x
    # simulation z -> visualization y
    # --------------------------------------------------
    viz_positions = np.zeros_like(positions)
    viz_positions[:, :, 0] = positions[:, :, 1]   # x_vis
    viz_positions[:, :, 1] = positions[:, :, 2]   # y_vis
    viz_positions[:, :, 2] = positions[:, :, 0]   # z_vis

    all_pts = viz_positions.reshape(-1, 3)

    # --------------------------------------------------
    # Cylinder geometry
    # --------------------------------------------------
    z_min = float(all_pts[:, 2].min())
    z_max = float(all_pts[:, 2].max())
    z_center = 0.5 * (z_min + z_max)

    x_center = float(all_pts[:, 0].mean())
    y_center = float(all_pts[:, 1].mean())

    xy = all_pts[:, 0:2] - np.array([x_center, y_center])
    radial_dist = np.sqrt((xy ** 2).sum(axis=1))

    radius = float(radial_dist.max() + 0.12)
    height = float((z_max - z_min) + 0.45)

    # open cylindrical vessel wall
    cylinder = pv.Cylinder(
        center=(x_center, y_center, z_center),
        direction=(0, 0, 1),
        radius=radius,
        height=height,
        resolution=160,
        capping=False,
    )

    out_path = "results/month2_final_cylinder_grn_adhesion.mp4"

    # use a video-friendly size divisible by 16
    plotter = pv.Plotter(window_size=(1600, 960), off_screen=True)
    plotter.set_background("white")
    plotter.open_movie(out_path, framerate=5)

    # camera more like your pre-practical cylinder view
    plotter.camera_position = [
        (x_center + 1.4, y_center - 2.6, z_center + 1.1),   # camera
        (x_center, y_center, z_center),                     # look-at
        (0, 0, 1),                                          # up direction
    ]

    # lighting
    plotter.enable_lightkit()

    max_adh = max(0.25, float(adhesion.max()) + 1e-6)

    for t in range(steps):
        pts = viz_positions[t]
        adh = adhesion[t]

        cloud = pv.PolyData(pts)
        cloud["adhesion"] = adh

        # cylindrical radial distance
        x_rel = pts[:, 0] - x_center
        y_rel = pts[:, 1] - y_center
        r = np.sqrt(x_rel**2 + y_rel**2)
        r_norm = np.clip(r / radius, 0.0, 1.0)

        near_wall_mask = r_norm > 0.7
        center_mask = r_norm <= 0.7

        near_wall_vals = adh[near_wall_mask]
        center_vals = adh[center_mask]

        near_wall_mean = float(np.mean(near_wall_vals)) if len(near_wall_vals) > 0 else 0.0
        center_mean = float(np.mean(center_vals)) if len(center_vals) > 0 else 0.0
        high_adh_count = int(np.sum(adh > 0.15))

        plotter.clear()

        # --------------------------------------------------
        # Vessel wall: transparent + wireframe feel
        # --------------------------------------------------
        plotter.add_mesh(
            cylinder,
            color="lightgray",
            opacity=0.10,
            smooth_shading=True,
        )

        plotter.add_mesh(
            cylinder,
            style="wireframe",
            color="black",
            line_width=0.5,
            opacity=0.18,
        )

        # --------------------------------------------------
        # Platelets
        # --------------------------------------------------
        plotter.add_points(
            cloud,
            scalars="adhesion",
            cmap="Reds",
            render_points_as_spheres=True,
            point_size=20,
            clim=[0.0, max_adh],
            scalar_bar_args={
                "title": "GRN Adhesion",
                "vertical": True,
                "position_x": 0.88,
                "position_y": 0.18,
                "height": 0.50,
                "width": 0.05,
                "title_font_size": 16,
                "label_font_size": 12,
                "color": "black",
            },
        )

        # --------------------------------------------------
        # Flow arrow
        # --------------------------------------------------
        flow_arrow = pv.Arrow(
            start=(x_center - radius * 1.35, y_center - radius * 1.45, z_center - height * 0.30),
            direction=(0, 0, 1),
            scale=0.35,
        )
        plotter.add_mesh(flow_arrow, color="black")

        # --------------------------------------------------
        # Text
        # --------------------------------------------------
        plotter.add_text(
            "Month 2: GRN-Controlled Platelet Behavior in 3D Cylindrical Flow",
            position=(210, 915),
            font_size=20,
            color="black",
        )

        plotter.add_text(
            "GraphML network -> collision input -> OutStickiness -> adhesion -> slower motion",
            position=(210, 885),
            font_size=13,
            color="black",
        )

        plotter.add_text(
            f"Near-wall mean adhesion: {near_wall_mean:.5f}\n"
            f"Center mean adhesion:    {center_mean:.5f}\n"
            f"High-adhesion platelets: {high_adh_count}",
            position=(40, 825),
            font_size=8,
            color="black",
        )

        plotter.add_text(
            "Darker red = higher GRN-driven adhesion",
            position=(40, 780),
            font_size=12,
            color="black",
        )

        plotter.add_text(
            f"Step {t + 1}/{steps}",
            position=(40, 45),
            font_size=13,
            color="black",
        )

        plotter.add_text(
            "Flow direction",
            position=(65, 105),
            font_size=11,
            color="black",
        )

        plotter.write_frame()

    plotter.close()

    print(f"Saved video to: {out_path}")


if __name__ == "__main__":
    main()