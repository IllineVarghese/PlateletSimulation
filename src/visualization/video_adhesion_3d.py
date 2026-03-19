import csv
from pathlib import Path

import imageio
import numpy as np
import pyvista as pv

# ---------- CONFIG ----------
BASE = Path("results/week4")
RUNS = [
    ("baseline", "BASELINE\n(no adhesion)", BASE / "baseline"),
    ("weak",     "WEAK\n(stick=0.5)",      BASE / "weak"),
    ("medium",   "MEDIUM\n(stick=0.2)",    BASE / "medium"),
    ("strong",   "STRONG\n(stick=0.05)",   BASE / "strong"),
]

OUT_VIDEO = BASE / "adhesion_3d_comparison.mp4"
SUMMARY_CSV = BASE / "summary.csv"

FPS = 12
POINT_SIZE = 9

# Must match simulation geometry
ZMIN = 0.0
ZMAX = 10.0
R = 1.0
LENGTH = ZMAX - ZMIN

# 4 cylinders side by side
X_OFFSETS = [-6.2, -2.1, 2.1, 6.2]

# Display-only scaling
DISPLAY_R_SCALE = 1.35
DISPLAY_Z_SCALE = 0.55
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


def unwrap_dz(z_curr: np.ndarray, z_prev: np.ndarray, length: float) -> np.ndarray:
    """
    Compute absolute z-step while correcting periodic wrap.
    """
    dz = z_curr - z_prev
    dz = np.where(dz > 0.5 * length, dz - length, dz)
    dz = np.where(dz < -0.5 * length, dz + length, dz)
    return np.abs(dz)


def read_summary(summary_csv: Path):
    """
    Reads results/week4/summary.csv and returns:
        metrics[run_name] = {"stick_factor": ..., "dz_near_sel": ...}
    """
    metrics = {}
    if not summary_csv.exists():
        return metrics

    with open(summary_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run = row.get("run", "").strip()
            if not run:
                continue
            metrics[run] = {
                "stick_factor": row.get("stick_factor", ""),
                "dz_near_sel": row.get("dz_near_sel", ""),
            }
    return metrics


def make_label_text(run_key: str, display_name: str, metrics: dict) -> str:
    m = metrics.get(run_key, {})
    dz_txt = m.get("dz_near_sel", "")
    stick_txt = m.get("stick_factor", "")

    if dz_txt != "":
        try:
            dz_txt = f"{float(dz_txt):.4f}"
        except Exception:
            pass

    if run_key == "baseline":
        return f"{display_name}\ndz_near={dz_txt}"
    return f"{display_name}\ndz_near={dz_txt}"


def add_static_scene(plotter: pv.Plotter, label_texts):
    plotter.set_background("white")
    plotter.add_axes()

    cyl = make_cylinder_surface(ZMIN, ZMAX, R, n=180)
    ring_bottom = make_ring(ZMIN, R)
    ring_top = make_ring(ZMAX, R)

    for dx in X_OFFSETS:
        plotter.add_mesh(
            transform_mesh_for_display(cyl, dx),
            style="surface",
            color="lightgray",
            opacity=0.08,
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
        [X_OFFSETS[0], -2.15, ZMAX * DISPLAY_Z_SCALE + 0.38],
        [X_OFFSETS[1], -2.15, ZMAX * DISPLAY_Z_SCALE + 0.38],
        [X_OFFSETS[2], -2.15, ZMAX * DISPLAY_Z_SCALE + 0.38],
        [X_OFFSETS[3], -2.15, ZMAX * DISPLAY_Z_SCALE + 0.38],
    ])

    plotter.add_point_labels(
        label_points,
        label_texts,
        font_size=14,
        point_size=0,
        shape=None,
        text_color="black",
        always_visible=True,
    )

    plotter.add_text(
        "Month 1 — Adhesion / stickiness sweep (3D cylindrical model)",
        position="upper_edge",
        font_size=16,
        color="black",
        name="main_title",
    )

    plotter.add_text(
        "Color = axial movement per saved frame (Δz)\nDarker = more sticking, Brighter = more motion",
        position=(20, 70),
        font_size=11,
        color="black",
        name="explain_text",
    )

    plotter.camera_position = [
        (0.0, -24.0, 9.2),
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
    metrics = read_summary(SUMMARY_CSV)

    loaded = []
    min_frames = None
    all_dz_values = []

    for (run_key, display_name, folder), dx in zip(RUNS, X_OFFSETS):
        P, A = load_run(folder)
        loaded.append((run_key, display_name, P, A, dx))
        min_frames = P.shape[0] if min_frames is None else min(min_frames, P.shape[0])

        # collect dz values for global color scaling
        for t in range(1, P.shape[0]):
            dz = unwrap_dz(P[t, :, 2], P[t - 1, :, 2], LENGTH)
            all_dz_values.append(dz)

    if not all_dz_values:
        raise RuntimeError("No dz values found to visualize.")

    all_dz_concat = np.concatenate(all_dz_values)
    DZ_MAX = float(np.percentile(all_dz_concat, 95))
    if DZ_MAX <= 0.0:
        DZ_MAX = 1e-6

    label_texts = [
        make_label_text(run_key, display_name, metrics)
        for run_key, display_name, _ in RUNS
    ]

    OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    plotter = pv.Plotter(off_screen=True, window_size=(1900, 950))
    add_static_scene(plotter, label_texts)

    point_clouds = []

    # initialize frame 0
    for run_key, display_name, P, A, dx in loaded:
        poly = pv.PolyData(transform_points_for_display(P[0], dx))
        dz0 = np.zeros(P.shape[1], dtype=np.float32)
        poly["dz_step"] = np.clip(dz0, 0.0, DZ_MAX)

        plotter.add_points(
            poly,
            render_points_as_spheres=True,
            point_size=POINT_SIZE,
            scalars="dz_step",
            cmap="viridis",
            clim=(0.0, DZ_MAX),
        )

        point_clouds.append((poly, P, dx))

    plotter.add_scalar_bar(title="Axial movement (Δz)", vertical=True)
    add_frame_text(plotter, 0, min_frames)

    with imageio.get_writer(OUT_VIDEO, fps=FPS) as writer:
        for t in range(min_frames):
            for poly, P, dx in point_clouds:
                poly.points = transform_points_for_display(P[t], dx)

                if t == 0:
                    dz_step = np.zeros(P.shape[1], dtype=np.float32)
                else:
                    dz_step = unwrap_dz(P[t, :, 2], P[t - 1, :, 2], LENGTH)

                poly["dz_step"] = np.clip(dz_step, 0.0, DZ_MAX)

            plotter.remove_actor("frame_text", render=False)
            add_frame_text(plotter, t, min_frames)

            plotter.render()
            frame = plotter.screenshot(return_img=True)
            writer.append_data(frame)

    plotter.close()
    print(f"Saved adhesion 3D video to: {OUT_VIDEO}")


if __name__ == "__main__":
    main()