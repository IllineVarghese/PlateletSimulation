from pathlib import Path
import numpy as np
import pyvista as pv


def main():
    data_path = Path("results") / "positions_steps_warp_cyl.npy"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing file: {data_path}. Run warp_particles_box_collision_cpu.py first.")


    data = np.load(data_path)  # shape: (steps, N, 3)
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Off-screen plotter: NO GUI window at all ---
    plotter = pv.Plotter(off_screen=True, window_size=(1280, 720))
    plotter.set_background("white")
    plotter.add_axes()
    # draw cylinder boundary (must match sim)
    R = 0.2
    ZMIN = 0.0
    ZMAX = 0.8

    cyl = pv.Cylinder(center=(0, 0, 0.5*(ZMIN+ZMAX)),
                      direction=(0, 0, 1),
                      radius=R,
                      height=(ZMAX - ZMIN),
                      resolution=64)
    plotter.add_mesh(cyl, style="wireframe", line_width=2, color="black")

    

    # Start with first frame
    points = pv.PolyData(data[0])
    plotter.add_points(points, render_points_as_spheres=True, point_size=18, color="steelblue")
    plotter.add_text("Platelets (trajectory)", position="upper_edge", font_size=24, color="black")

    # Save ONE image of the last step (and optionally frames)
    last_idx = data.shape[0] - 1
    points.points = data[last_idx]
    plotter.render()

    out_png = out_dir / "pyvista_step_last.png"
    plotter.screenshot(str(out_png))
    plotter.close()

    print(f"Saved screenshot to {out_png}")


if __name__ == "__main__":
    main()


