import numpy as np
import pyvista as pv
from pathlib import Path


def make_frustum(zmin, zmax, r0, r1, n=160):
    """Create a truncated cone (frustum) surface as PolyData."""
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

    # bottom ring (z=zmin)
    xb = r0 * np.cos(theta)
    yb = r0 * np.sin(theta)
    zb = np.full_like(theta, zmin)

    # top ring (z=zmax)
    xt = r1 * np.cos(theta)
    yt = r1 * np.sin(theta)
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

    poly = pv.PolyData(pts, faces=np.array(faces))
    return poly


def make_ring(z, r, n=200):
    """Create a ring (polyline) circle at height z."""
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=True)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.full_like(theta, z)
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    return pv.Spline(pts, n_points=len(pts))


def main():
    data_path = Path("results") / "positions_steps_warp_cone_poiseuille.npy"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing file: {data_path}. Run the cone sim first.")

    data = np.load(data_path)  # (steps, N, 3)
    steps, N, dim = data.shape
    assert dim == 3

    # MUST match your sim params
    ZMIN = 0.0
    ZMAX = 0.8
    R0 = 0.24
    R1 = 0.12

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    plotter = pv.Plotter(off_screen=True, window_size=(1280, 720))
    plotter.set_background("white")
    plotter.add_axes()

    # Exact frustum boundary (wireframe)
    frustum = make_frustum(ZMIN, ZMAX, R0, R1, n=180)
    plotter.add_mesh(frustum, style="wireframe", line_width=2, color="black")

    # Draw inlet/outlet as RINGS (not filled disks)
    ring_in = make_ring(ZMIN, R0)
    ring_out = make_ring(ZMAX, R1)
    plotter.add_mesh(ring_in, color="black", line_width=3)
    plotter.add_mesh(ring_out, color="black", line_width=3)

    # particles (last frame)
    last = data[-1]
    points = pv.PolyData(last)
    plotter.add_points(points, render_points_as_spheres=True, point_size=12, color="steelblue")

    plotter.add_text("Platelets (cone Poiseuille - last step)", position="upper_edge",
                     font_size=22, color="black")

    # Camera (nice stable view)
    plotter.camera_position = "iso"
    plotter.camera.zoom(1.15)

    out_png = out_dir / "pyvista_cone_poiseuille_last.png"
    plotter.screenshot(str(out_png))
    plotter.close()
    print(f"Saved screenshot to {out_png}")


if __name__ == "__main__":
    main()
