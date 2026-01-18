import numpy as np
import pyvista as pv
from pathlib import Path


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


def make_ring(z: float, r: float, n: int = 200) -> pv.PolyData:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=True)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    zz = np.full_like(theta, z)
    pts = np.stack([x, y, zz], axis=1).astype(np.float32)
    return pv.Spline(pts, n_points=len(pts))


def main():
    data_path = Path("results") / "positions_steps_warp_cyl_poiseuille.npy"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing file: {data_path}. Run the CYLINDER Poiseuille sim first.")

    data = np.load(data_path)  # (steps, N, 3)
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f"Expected data shape (steps, N, 3), got {data.shape}")

    # MUST match your sim params
    ZMIN = 0.0
    ZMAX = 0.8
    R = 0.2       # cylinder radius
    VMAX = 0.6    # Poiseuille vmax (flow-field)

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    plotter = pv.Plotter(off_screen=True, window_size=(1280, 720))
    plotter.set_background("white")
    plotter.add_axes()

    # Cylinder boundary
    cyl = make_cylinder_surface(ZMIN, ZMAX, R, n=180)
    plotter.add_mesh(cyl, style="wireframe", line_width=2, color="black")

    # Inlet/outlet rings
    plotter.add_mesh(make_ring(ZMIN, R), color="black", line_width=3)
    plotter.add_mesh(make_ring(ZMAX, R), color="black", line_width=3)

    # Last frame particles
    last = data[-1]

    # Flow-field speed at particle location (Poiseuille profile)
    radial = np.sqrt(last[:, 0] ** 2 + last[:, 1] ** 2)
    speed = VMAX * (1.0 - (radial / R) ** 2)
    speed = np.clip(speed, 0.0, VMAX)

    points = pv.PolyData(last)
    points["speed"] = speed

    plotter.add_points(
        points,
        render_points_as_spheres=True,
        point_size=12,
        scalars="speed",
        cmap="viridis",
        clim=(0.0, VMAX),
    )

    plotter.add_scalar_bar(title="Flow-field speed", vertical=True)

    plotter.add_text(
        "Platelets (cylinder Poiseuille - last step)",
        position="upper_edge",
        font_size=22,
        color="black",
    )

    plotter.camera_position = "iso"
    plotter.camera.zoom(1.15)

    out_png = out_dir / "baseline_cylinder_poiseuille_gpu.png"
    plotter.screenshot(str(out_png))
    plotter.close()
    print(f"Saved screenshot to {out_png}")


if __name__ == "__main__":
    main()
