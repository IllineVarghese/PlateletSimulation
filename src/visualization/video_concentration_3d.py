import numpy as np
import pyvista as pv
from pathlib import Path
import imageio

# ---------- CONFIG ----------
BASE = Path("results/conc")
RUNS = [
    ("LOW concentration", BASE / "low"),
    ("MEDIUM concentration", BASE / "medium"),
    ("HIGH concentration", BASE / "high"),
]

OUT_VIDEO = BASE / "concentration_3d_comparison.mp4"

FPS = 12
POINT_SIZE = 11

# MUST match your month-1 simulation parameters
ZMIN = 0.0
ZMAX = 10.0
R = 1.0
ACT_MAX = 0.25

# horizontal offsets for the 3 cylinders in one shared scene
X_OFFSETS = [-4.2, 0.0, 4.2]

# display-only scaling to improve the look
DISPLAY_R_SCALE = 1.35   # wider
DISPLAY_Z_SCALE = 0.55   # shorter
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


def make_ring(z: float, r: float, n: int = 200) -> pv.PolyData:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=True)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    zz = np.full_like(theta, z)
    pts = np.stack([x, y, zz], axis=1).astype(np.float32)
    return pv.Spline(pts, n_points=len(pts))


def transform_points_for_display(points_xyz: np.ndarray, dx: float = 0.0) -> np.ndarray:
    pts = points_xyz.copy()
    pts[:, 0] = pts[:, 0] * DISPLAY_R_SCALE + dx
    pts[:, 1] = pts[:, 1] * DISPLAY_R_SCALE
    pts[:, 2] = pts[:, 2] * DISPLAY_Z_SCALE
    return pts


def transform_mesh_for_display(mesh: pv.PolyData, dx: float = 0.0) -> pv.PolyData:
    m = mesh.copy()
    pts = np.asarray(m.points).copy()
    pts[:, 0] = pts[:, 0] * DISPLAY_R_SCALE + dx
    pts[:, 1] = pts[:, 1] * DISPLAY_R_SCALE
    pts[:, 2] = pts[:, 2] * DISPLAY_Z_SCALE
    m.points = pts
    return m


def load_run(folder: Path):
    pos_path = folder / "positions_saved.npy"
    act_path = folder / "activation_saved.npy"

    if not pos_path.exists():
        raise FileNotFoundError(f"Missing file: {pos_path}")
    if not act_path.exists():
        raise FileNotFoundError(f"Missing file: {act_path}")

    P = np.load(pos_path)   # (T, N, 3)
    A = np.load(act_path)   # (T, N)
    return P, A


def add_static_scene(plotter: pv.Plotter):
    plotter.set_background("white")
    plotter.add_axes()

    cyl = make_cylinder_surface(ZMIN, ZMAX, R, n=180)
    ring_bottom = make_ring(ZMIN, R)
    ring_top = make_ring(ZMAX, R)

    for dx in X_OFFSETS:
        plotter.add_mesh(
            transform_mesh_for_display(cyl, dx),
            style="surface",
            color="lightgray",        # softer than black
            line_width=1.0,      # thinner lines
            opacity=0.08,        # much lighter
        )
        plotter.add_mesh(
            transform_mesh_for_display(ring_bottom, dx),
            color="gray",
            line_width=1.5,
            opacity=0.35,
        )
        plotter.add_mesh(
            transform_mesh_for_display(ring_top, dx),
            color="gray",
            line_width=1.5,
            opacity=0.35,
        )

    label_points = np.array([
        [X_OFFSETS[0], -1.95, ZMAX * DISPLAY_Z_SCALE + 0.28],
        [X_OFFSETS[1], -1.95, ZMAX * DISPLAY_Z_SCALE + 0.28],
        [X_OFFSETS[2], -1.95, ZMAX * DISPLAY_Z_SCALE + 0.28],
    ])
    labels = ["LOW", "MEDIUM", "HIGH"]
    plotter.add_point_labels(
        label_points,
        labels,
        font_size=16,
        point_size=0,
        shape=None,
        text_color="black",
        always_visible=True,
    )

    plotter.add_text(
        "Month 1 — Concentration sweep (3D cylindrical model)",
        position="upper_edge",
        font_size=16,
        color="black",
        name="main_title",
    )

    plotter.camera_position = [
        (0.0, -18.0, 8.8),
        (0.0, 0.0, ZMAX * DISPLAY_Z_SCALE * 0.45),
        (0.0, 0.0, 1.0),
    ]
    plotter.camera.zoom(0.82)


def add_frame_text(plotter: pv.Plotter, frame_idx: int, total_frames: int):
    plotter.add_text(
        f"frame {frame_idx + 1}/{total_frames}",
        position=(20, 20),
        font_size=12,
        color="black",
        name="frame_text",
    )


def main():
    loaded = []
    min_frames = None

    for (scenario_name, folder), dx in zip(RUNS, X_OFFSETS):
        P, A = load_run(folder)
        loaded.append((scenario_name, P, A, dx))
        min_frames = P.shape[0] if min_frames is None else min(min_frames, P.shape[0])

    OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    plotter = pv.Plotter(off_screen=True, window_size=(1600, 900))
    add_static_scene(plotter)

    point_clouds = []

    for scenario_name, P, A, dx in loaded:
        poly = pv.PolyData(transform_points_for_display(P[0], dx))
        poly["activation"] = np.clip(A[0], 0.0, ACT_MAX)

        plotter.add_points(
            poly,
            render_points_as_spheres=True,
            point_size=POINT_SIZE,
            scalars="activation",
            cmap="viridis",
            clim=(0.0, ACT_MAX),
        )

        point_clouds.append((poly, P, A, dx))

    plotter.add_scalar_bar(title="Activation", vertical=True)
    add_frame_text(plotter, 0, min_frames)

    with imageio.get_writer(OUT_VIDEO, fps=FPS) as writer:
        for t in range(min_frames):
            for poly, P, A, dx in point_clouds:
                poly.points = transform_points_for_display(P[t], dx)
                poly["activation"] = np.clip(A[t], 0.0, ACT_MAX)

            plotter.remove_actor("frame_text", render=False)
            add_frame_text(plotter, t, min_frames)

            plotter.render()
            frame = plotter.screenshot(return_img=True)
            writer.append_data(frame)

    plotter.close()
    print(f"Saved concentration 3D video to: {OUT_VIDEO}")


if __name__ == "__main__":
    main()