import numpy as np
import pyvista as pv
from pathlib import Path


def make_cylinder_surface(zmin: float, zmax: float, r: float, n: int = 180) -> pv.PolyData:
    """Create a cylinder side surface (no caps) as PolyData."""
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

    # bottom ring (z=zmin)
    xb = r * np.cos(theta)
    yb = r * np.sin(theta)
    zb = np.full_like(theta, zmin)

    # top ring (z=zmax)
    xt = r * np.cos(theta)
    yt = r * np.sin(theta)
    zt = np.full_like(theta, zmax)

    pts = np.vstack([
        np.stack([xb, yb, zb], axis=1),
        np.stack([xt, yt, zt], axis=1)
    ]).astype(np.float32)

    # build quad faces between rings
    faces = []
    for i in range(n):
        i0 = i
        i1 = (i + 1) % n
        j0 = n + i
        j1 = n + (i + 1) % n
        # quad: bottom i -> bottom i+1 -> top i+1 -> top i
        faces.extend([4, i0, i1, j1, j0])

    return pv.PolyData(pts, faces=np.array(faces))


def make_ring(z: float, r: float, n: int = 200) -> pv.PolyData:
    """Create a ring (polyline) circle at height z."""
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=True)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    zz = np.full_like(theta, z)
    pts = np.stack([x, y, zz], axis=1).astype(np.float32)
    return pv.Spline(pts, n_points=len(pts))


def main():
    data_path = Path("results") / "positions_steps_warp_cyl_poiseuille.npy"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing file: {data_path}. Run the CYLINDER Poiseuille sim first."
        )

    data = np.load(data_path)  # (steps, N, 3)
    steps, N, dim = data.shape
    if dim != 3:
        raise ValueError(f"Expected data shape (steps, N, 3), got {data.shape}")

    # MUST match your sim params (geometry used by the cylinder simulation)
    ZMIN = 0.0
    ZMAX = 0.8

    # Cylinder radius (use ONE value, same at inlet and outlet)
    R = 0.2  # <-- set this to match your cylinder sim radius (often R = 0.2 in your scripts)

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    plotter = pv.Plotter(off_screen=True, window_size=(1280, 720))
    plotter.set_background("white")
    plotter.add_axes()

    # Cylinder boundary (wireframe)
    cyl = make_cylinder_surface(ZMIN, ZMAX, R, n=180)
    plotter.add_mesh(cyl, style="wireframe", line_width=2, color="black")

    # Inlet/outlet rings
    ring_in = make_ring(ZMIN, R)
    ring_out = make_ring(ZMAX, R)
    plotter.add_mesh(ring_in, color="black", line_width=3)
    plotter.add_mesh(ring_out, color="black", line_width=3)

    # particles (last frame)
    last = data[-1]
    points = pv.PolyData(last)
    plotter.add_points(points, render_points_as_spheres=True, point_size=12, color="steelblue")

    plotter.add_text(
        "Platelets (cylinder Poiseuille - last step)",
        position="upper_edge",
        font_size=22,
        color="black"
    )

    # Camera (nice stable view)
    plotter.camera_position = "iso"
    plotter.camera.zoom(1.15)

    out_png = out_dir / "baseline_cylinder_poiseuille_gpu.png"
    plotter.screenshot(str(out_png))
    plotter.close()
    print(f"Saved screenshot to {out_png}")


if __name__ == "__main__":
    main()

