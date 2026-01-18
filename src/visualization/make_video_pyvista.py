import numpy as np
import pyvista as pv
from pathlib import Path
import imageio

# ---------- CONFIG ----------
DATA_FILE = Path("results/positions_steps_warp_cyl_poiseuille.npy")
OUT_VIDEO = Path("results/cylinder_poiseuille_gpu_colored.mp4")

FPS = 30
POINT_SIZE = 10

# MUST match your simulation parameters
ZMIN = 0.0
ZMAX = 0.8
R = 0.2
VMAX = 0.6
# ---------------------------


def make_cylinder_surface(zmin: float, zmax: float, r: float, n: int = 180) -> pv.PolyData:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

    xb = r * np.cos(theta)
    yb = r * np.sin(theta)
    zb = np.full_like(theta, zmin)

    xt = r * np.cos(theta)
    yt = r * np.sin(theta)
    zt = np.full_like(theta, zmax)

    pts = np.vstack([
        np.stack([xb, yb, zb], axis=1),
        np.stack([xt, yt, zt], axis=1)
    ]).astype(np.float32)

    faces = []
    for i in range(n):
        i0 = i
        i1 = (i + 1) % n
        j0 = n + i
        j1 = n + (i + 1) % n
        faces.extend([4, i0, i1, j1, j0])

    return pv.PolyData(pts, faces=np.array(faces))


def poiseuille_speed_from_xy(xy: np.ndarray, r0: float, vmax: float) -> np.ndarray:
    """Compute Poiseuille speed magnitude from x,y positions."""
    r = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
    v = vmax * (1.0 - (r / r0) ** 2)
    return np.clip(v, 0.0, vmax)


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing file: {DATA_FILE}")

    data = np.load(DATA_FILE)  # (steps, N, 3)

    plotter = pv.Plotter(off_screen=True, window_size=(1280, 720))
    plotter.set_background("white")
    plotter.add_axes()

    # Vessel wireframe
    cyl = make_cylinder_surface(ZMIN, ZMAX, R, n=180)
    plotter.add_mesh(cyl, style="wireframe", color="black", line_width=2)

    # Initial frame particles
    pts0 = data[0]
    speed0 = poiseuille_speed_from_xy(pts0[:, :2], R, VMAX)

    points = pv.PolyData(pts0)
    points["speed"] = speed0

    plotter.add_points(
        points,
        render_points_as_spheres=True,
        point_size=POINT_SIZE,
        scalars="speed",
        cmap="viridis",
        clim=(0.0, VMAX),  # keeps colors consistent across frames
    )

    plotter.add_scalar_bar(title="Velocity magnitude", vertical=True)

    plotter.add_text(
        "Platelets (cylinder Poiseuille, GPU) — velocity-colored",
        position="upper_edge",
        font_size=18,
        color="black",
    )

    plotter.camera_position = "iso"
    plotter.camera.zoom(1.15)

    # Write video frames
    OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(OUT_VIDEO, fps=FPS) as writer:
        for step in range(data.shape[0]):
            pts = data[step]
            points.points = pts
            points["speed"] = poiseuille_speed_from_xy(pts[:, :2], R, VMAX)

            plotter.render()
            frame = plotter.screenshot(return_img=True)
            writer.append_data(frame)

    plotter.close()
    print(f"Saved video to {OUT_VIDEO}")


if __name__ == "__main__":
    main()
