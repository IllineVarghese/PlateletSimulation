from pathlib import Path
import numpy as np
import pyvista as pv


# ============================================================
# Phase 5: 3D cylindrical platelet flow video
# Real platelet mesh visualization with activation-based
# inactive-blue to activated-red state switching.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = ROOT / "results" / "phase5" / "presentation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_PATH = OUTPUT_DIR / "phase5_cylinder_flow_activation_120_platelets.mp4"
FINAL_FRAME_PATH = OUTPUT_DIR / "phase5_cylinder_flow_activation_120_platelets_final_frame.png"

POSITIONS_PATH = ROOT / "results" / "phase4" / "final_demo" / "positions.npy"
ACTIVATION_PATH = ROOT / "results" / "phase4" / "final_demo" / "activation.npy"
SHEAR_PATH = ROOT / "results" / "phase4" / "final_demo" / "shear_input.npy"

# For 120 platelet video, optimized meshes are safer and faster.
# They are still real platelet meshes, only reduced for visualization performance.
INACTIVE_MESH_CANDIDATES = [
    ROOT / "results" / "phase5" / "week3" / "optimized_meshes" / "inactive_decimated.vtp",
    ROOT / "data" / "meshes" / "platelet" / "inactive.obj",
]

ACTIVATED_MESH_CANDIDATES = [
    ROOT / "results" / "phase5" / "week3" / "optimized_meshes" / "activated_decimated.vtp",
    ROOT / "data" / "meshes" / "platelet" / "activated.obj",
]

STATE_THRESHOLD = 0.50

CYLINDER_LENGTH = 8.0
CYLINDER_RADIUS = 1.05

RENDER_COUNT = 120
VIDEO_FRAMES = 96
FPS = 8

PLATELET_SCALE_BASE = 0.070
FLOW_ARROW_OPACITY = 0.30
FLOW_ARROW_SCALE = 0.32

RANDOM_SEED = 42


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def normalize_mesh(mesh: pv.PolyData) -> pv.PolyData:
    mesh = mesh.extract_surface().triangulate().clean().copy(deep=True)

    center = np.array(mesh.center)
    mesh.translate(-center, inplace=True)

    bounds = np.array(mesh.bounds).reshape(3, 2)
    extents = bounds[:, 1] - bounds[:, 0]
    max_extent = max(float(np.max(extents)), 1e-8)

    mesh.scale(1.0 / max_extent, inplace=True)

    return mesh


def load_meshes():
    inactive_path = first_existing(INACTIVE_MESH_CANDIDATES)
    activated_path = first_existing(ACTIVATED_MESH_CANDIDATES)

    if inactive_path is None:
        raise FileNotFoundError("Could not find inactive platelet mesh.")

    if activated_path is None:
        raise FileNotFoundError("Could not find activated platelet mesh.")

    inactive_mesh = normalize_mesh(pv.read(inactive_path))
    activated_mesh = normalize_mesh(pv.read(activated_path))

    print(f"Inactive mesh loaded : {inactive_path}")
    print(f"Activated mesh loaded: {activated_path}")
    print(f"Inactive mesh cells  : {inactive_mesh.n_cells}")
    print(f"Activated mesh cells : {activated_mesh.n_cells}")

    return inactive_mesh, activated_mesh, inactive_path, activated_path


def load_phase4_arrays():
    require_file(POSITIONS_PATH, "Phase 4 positions")
    require_file(ACTIVATION_PATH, "Phase 4 activation")
    require_file(SHEAR_PATH, "Phase 4 shear")

    positions = np.load(POSITIONS_PATH)
    activation = np.load(ACTIVATION_PATH)
    shear = np.load(SHEAR_PATH)

    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"Expected positions shape (frames, platelets, 3), got {positions.shape}")

    if activation.ndim != 2:
        raise ValueError(f"Expected activation shape (frames, platelets), got {activation.shape}")

    if shear.ndim != 2:
        raise ValueError(f"Expected shear shape (frames, platelets), got {shear.shape}")

    if positions.shape[:2] != activation.shape:
        raise ValueError(f"positions and activation mismatch: {positions.shape[:2]} vs {activation.shape}")

    if positions.shape[:2] != shear.shape:
        raise ValueError(f"positions and shear mismatch: {positions.shape[:2]} vs {shear.shape}")

    activation = np.clip(activation, 0.0, 1.0)
    shear = np.clip(shear, 0.0, None)

    print(f"Positions shape : {positions.shape}")
    print(f"Activation shape: {activation.shape}")
    print(f"Shear shape     : {shear.shape}")

    return positions, activation, shear


def normalize_shear(shear: np.ndarray) -> np.ndarray:
    shear = shear.astype(float)
    min_val = float(np.min(shear))
    max_val = float(np.max(shear))

    if max_val - min_val < 1e-8:
        return np.zeros_like(shear)

    return (shear - min_val) / (max_val - min_val)


def select_120_platelets(activation: np.ndarray, shear: np.ndarray) -> np.ndarray:
    """
    Select a mixed set of platelets:
    - low activation platelets, so blue inactive platelets are visible
    - near-threshold platelets, so switching is visible
    - high activation platelets, so red activated platelets are visible
    - high shear platelets, so flow-sensing context is represented
    """

    final_activation = activation[-1]
    first_activation = activation[0]
    delta_activation = final_activation - first_activation
    final_shear = shear[-1]

    n_platelets = final_activation.size

    low_activation_ids = np.argsort(final_activation)[:35]

    near_threshold_ids = np.argsort(np.abs(final_activation - STATE_THRESHOLD))[:35]

    high_activation_ids = np.argsort(final_activation)[::-1][:35]

    switching_mask = (first_activation < STATE_THRESHOLD) & (final_activation >= STATE_THRESHOLD)
    switching_ids = np.where(switching_mask)[0]
    switching_ids = switching_ids[np.argsort(delta_activation[switching_ids])[::-1]] if len(switching_ids) else np.array([], dtype=int)

    high_shear_ids = np.argsort(final_shear)[::-1][:25]

    selected = np.unique(
        np.concatenate(
            [
                low_activation_ids,
                near_threshold_ids,
                high_activation_ids,
                switching_ids[:35],
                high_shear_ids,
            ]
        )
    )

    if selected.size < RENDER_COUNT:
        remaining = np.setdiff1d(np.arange(n_platelets), selected)
        score = (
            0.40 * final_activation
            + 0.30 * delta_activation
            + 0.30 * final_shear
        )
        remaining_sorted = remaining[np.argsort(score[remaining])[::-1]]
        selected = np.concatenate([selected, remaining_sorted[: RENDER_COUNT - selected.size]])

    if selected.size > RENDER_COUNT:
        score = (
            0.35 * final_activation[selected]
            + 0.30 * delta_activation[selected]
            + 0.20 * final_shear[selected]
            + 0.15 * (1.0 - np.abs(final_activation[selected] - STATE_THRESHOLD))
        )
        selected = selected[np.argsort(score)[::-1][:RENDER_COUNT]]

    selected = selected.astype(int)

    final_selected = final_activation[selected]

    print(f"Selected platelets: {selected.size}")
    print(f"Final inactive in selected set : {np.sum(final_selected < STATE_THRESHOLD)}")
    print(f"Final activated in selected set: {np.sum(final_selected >= STATE_THRESHOLD)}")
    print(f"Switching candidates selected  : {len(np.intersect1d(selected, switching_ids))}")

    return selected


def resample_time_series(arr: np.ndarray, out_frames: int) -> np.ndarray:
    """
    Interpolate arrays along time axis.
    Works for shape (T, N) and (T, N, 3).
    """

    arr = np.asarray(arr)
    in_frames = arr.shape[0]

    old_t = np.linspace(0.0, 1.0, in_frames)
    new_t = np.linspace(0.0, 1.0, out_frames)

    if arr.ndim == 2:
        out = np.empty((out_frames, arr.shape[1]), dtype=float)

        for j in range(arr.shape[1]):
            out[:, j] = np.interp(new_t, old_t, arr[:, j])

        return out

    if arr.ndim == 3:
        out = np.empty((out_frames, arr.shape[1], arr.shape[2]), dtype=float)

        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                out[:, j, k] = np.interp(new_t, old_t, arr[:, j, k])

        return out

    raise ValueError(f"Unsupported shape for resampling: {arr.shape}")


def remap_positions_to_cylinder(positions: np.ndarray) -> np.ndarray:
    """
    Rescale real Phase 4 positions into a clean cylindrical presentation space.
    Keeps the motion pattern while making the view visually readable.
    """

    p = positions.copy().astype(float)

    x = p[:, :, 0]
    y = p[:, :, 1]
    z = p[:, :, 2]

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_mid = 0.5 * (float(np.min(y)) + float(np.max(y)))
    z_mid = 0.5 * (float(np.min(z)) + float(np.max(z)))

    if x_max - x_min > 1e-8:
        p[:, :, 0] = 0.35 + (x - x_min) / (x_max - x_min) * (CYLINDER_LENGTH - 0.70)
    else:
        p[:, :, 0] = CYLINDER_LENGTH / 2.0

    p[:, :, 1] = y - y_mid
    p[:, :, 2] = z - z_mid

    radial = np.sqrt(p[:, :, 1] ** 2 + p[:, :, 2] ** 2)
    max_radial = max(float(np.max(radial)), 1e-8)

    scale = (CYLINDER_RADIUS * 0.72) / max_radial
    p[:, :, 1] *= scale
    p[:, :, 2] *= scale

    return p


def activation_to_color(a: float) -> tuple[float, float, float]:
    """
    Force inactive platelets to be blue and activated platelets to be red.
    """

    a = float(np.clip(a, 0.0, 1.0))

    if a < STATE_THRESHOLD:
        # dark blue to light blue
        t = a / STATE_THRESHOLD
        r = 0.05 + 0.20 * t
        g = 0.18 + 0.28 * t
        b = 0.95
        return (r, g, b)

    # orange-red to dark red
    t = (a - STATE_THRESHOLD) / (1.0 - STATE_THRESHOLD)
    r = 0.95
    g = 0.45 * (1.0 - t) + 0.08 * t
    b = 0.18 * (1.0 - t) + 0.07 * t
    return (r, g, b)


def make_platelet_mesh(
    base_mesh: pv.PolyData,
    position: np.ndarray,
    activation_value: float,
    shear_value: float,
    platelet_id: int,
) -> pv.PolyData:
    mesh = base_mesh.copy(deep=True)

    # deterministic orientation by platelet id
    mesh.rotate_x((platelet_id * 11.0) % 360, inplace=True)
    mesh.rotate_y((platelet_id * 17.0) % 360, inplace=True)
    mesh.rotate_z((platelet_id * 23.0) % 360, inplace=True)

    scale_value = PLATELET_SCALE_BASE * (
        0.90
        + 0.35 * float(activation_value)
        + 0.15 * float(shear_value)
    )

    if activation_value >= STATE_THRESHOLD:
        scale_value *= 1.10

    mesh.scale(scale_value, inplace=True)
    mesh.translate(tuple(position), inplace=True)

    return mesh


def build_static_scene(plotter: pv.Plotter) -> None:
    cylinder = pv.Cylinder(
        center=(CYLINDER_LENGTH / 2.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0),
        radius=CYLINDER_RADIUS,
        height=CYLINDER_LENGTH,
        resolution=160,
        capping=False,
    )

    plotter.add_mesh(
        cylinder,
        color=(0.86, 0.88, 0.90),
        opacity=0.16,
        smooth_shading=True,
        show_edges=False,
        name="vessel_body",
    )

    centerline = pv.Line(
        pointa=(0.0, 0.0, 0.0),
        pointb=(CYLINDER_LENGTH, 0.0, 0.0),
    )

    plotter.add_mesh(
        centerline,
        color=(0.65, 0.65, 0.65),
        line_width=2,
        name="centerline",
    )

    # Poiseuille-style arrows: bigger at center, smaller away from center
    arrows = []
    x_positions = np.linspace(0.85, CYLINDER_LENGTH - 0.95, 5)
    yz_positions = [
        (0.00, 0.00, 1.00),
        (0.32, 0.00, 0.65),
        (-0.32, 0.00, 0.65),
        (0.00, 0.32, 0.65),
        (0.00, -0.32, 0.65),
    ]

    for x in x_positions:
        for y, z, strength in yz_positions:
            arrow = pv.Arrow(
                start=(x, y, z),
                direction=(1.0, 0.0, 0.0),
                scale=FLOW_ARROW_SCALE * strength,
                tip_length=0.22,
                tip_radius=0.045,
                shaft_radius=0.013,
            )
            arrows.append(arrow)

    for i, arrow in enumerate(arrows):
        plotter.add_mesh(
            arrow,
            color=(0.26, 0.52, 0.82),
            opacity=FLOW_ARROW_OPACITY,
            smooth_shading=True,
            name=f"flow_arrow_{i}",
        )

    plotter.add_axes(
        line_width=2,
        x_color="tomato",
        y_color="green",
        z_color="blue",
    )


def add_text_overlay(
    plotter: pv.Plotter,
    frame: int,
    active_count: int,
    inactive_count: int,
    switching_count: int,
    mean_activation: float,
    mean_shear: float,
) -> None:
    text_items = [
        "title",
        "stats",
        "means",
        "legend_title",
        "legend_shape",
        "legend_blue",
        "legend_red",
        "legend_yellow",
        "flow_text",
    ]

    for item in text_items:
        try:
            plotter.remove_actor(item)
        except Exception:
            pass

    plotter.add_text(
        "Phase 5: 3D Cylindrical Platelet Flow with Activation-Based Mesh Switching",
        position=(20, 955),
        font_size=18,
        color="black",
        name="title",
    )

    plotter.add_text(
        f"Frame {frame + 1}/{VIDEO_FRAMES} | rendered={RENDER_COUNT} | inactive={inactive_count} | activated={active_count} | near-threshold={switching_count}",
        position=(20, 925),
        font_size=13,
        color="black",
        name="stats",
    )

    plotter.add_text(
        f"Mean activation={mean_activation:.2f} | mean shear={mean_shear:.2f} | switching threshold={STATE_THRESHOLD:.2f}",
        position=(20, 900),
        font_size=13,
        color="black",
        name="means",
    )

    plotter.add_text(
        "Visual encoding",
        position=(20, 840),
        font_size=13,
        color="dimgray",
        name="legend_title",
    )

    plotter.add_text(
        "Shape: inactive mesh / activated mesh",
        position=(20, 815),
        font_size=12,
        color="gray",
        name="legend_shape",
    )

    plotter.add_text(
        "Blue = inactive / low activation",
        position=(20, 790),
        font_size=12,
        color=(0.05, 0.18, 0.95),
        name="legend_blue",
    )

    plotter.add_text(
        "Red = activated / high activation",
        position=(20, 765),
        font_size=12,
        color=(0.95, 0.08, 0.07),
        name="legend_red",
    )

    plotter.add_text(
        "Gold ring = near switching threshold",
        position=(20, 740),
        font_size=12,
        color="gold",
        name="legend_yellow",
    )

    plotter.add_text(
        "Flow direction",
        position=(1510, 850),
        font_size=13,
        color="gray",
        name="flow_text",
    )


def add_near_threshold_ring(plotter: pv.Plotter, position: np.ndarray, radius: float, name: str) -> None:
    ring = pv.Sphere(
        center=tuple(position),
        radius=radius,
        theta_resolution=24,
        phi_resolution=12,
    )

    plotter.add_mesh(
        ring,
        color="gold",
        style="wireframe",
        opacity=0.70,
        line_width=2,
        name=name,
    )


def main():
    pv.OFF_SCREEN = True

    positions, activation, shear = load_phase4_arrays()
    shear_norm = normalize_shear(shear)

    inactive_mesh, activated_mesh, inactive_path, activated_path = load_meshes()

    selected_ids = select_120_platelets(activation, shear_norm)

    positions_sel = positions[:, selected_ids, :]
    activation_sel = activation[:, selected_ids]
    shear_sel = shear_norm[:, selected_ids]

    positions_sel = remap_positions_to_cylinder(positions_sel)

    positions_video = resample_time_series(positions_sel, VIDEO_FRAMES)
    activation_video = resample_time_series(activation_sel, VIDEO_FRAMES)
    shear_video = resample_time_series(shear_sel, VIDEO_FRAMES)

    plotter = pv.Plotter(off_screen=True, window_size=(1800, 1000))
    plotter.set_background("white")
    plotter.open_movie(str(VIDEO_PATH), framerate=FPS)

    plotter.camera_position = [
        (4.0, -5.8, 2.7),
        (4.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ]

    plotter.camera.zoom(1.18)

    build_static_scene(plotter)

    platelet_actor_names = [f"platelet_{i}" for i in range(RENDER_COUNT)]
    ring_actor_names = [f"ring_{i}" for i in range(RENDER_COUNT)]

    for frame in range(VIDEO_FRAMES):
        for name in platelet_actor_names + ring_actor_names:
            try:
                plotter.remove_actor(name)
            except Exception:
                pass

        frame_positions = positions_video[frame]
        frame_activation = activation_video[frame]
        frame_shear = shear_video[frame]

        inactive_count = int(np.sum(frame_activation < STATE_THRESHOLD))
        active_count = int(np.sum(frame_activation >= STATE_THRESHOLD))
        switching_mask = np.abs(frame_activation - STATE_THRESHOLD) < 0.055
        switching_count = int(np.sum(switching_mask))

        for i, platelet_id in enumerate(selected_ids):
            a = float(frame_activation[i])
            s = float(frame_shear[i])
            p = frame_positions[i]

            base_mesh = inactive_mesh if a < STATE_THRESHOLD else activated_mesh

            platelet = make_platelet_mesh(
                base_mesh=base_mesh,
                position=p,
                activation_value=a,
                shear_value=s,
                platelet_id=int(platelet_id),
            )

            color = activation_to_color(a)

            plotter.add_mesh(
                platelet,
                color=color,
                smooth_shading=True,
                opacity=0.98,
                specular=0.18,
                name=f"platelet_{i}",
            )

            if switching_mask[i]:
                add_near_threshold_ring(
                    plotter,
                    position=p,
                    radius=0.15,
                    name=f"ring_{i}",
                )

        add_text_overlay(
            plotter=plotter,
            frame=frame,
            active_count=active_count,
            inactive_count=inactive_count,
            switching_count=switching_count,
            mean_activation=float(np.mean(frame_activation)),
            mean_shear=float(np.mean(frame_shear)),
        )

        plotter.write_frame()

        if frame == VIDEO_FRAMES - 1:
            plotter.screenshot(str(FINAL_FRAME_PATH))

        if (frame + 1) % 10 == 0 or frame == VIDEO_FRAMES - 1:
            print(
                f"Rendered {frame + 1}/{VIDEO_FRAMES} | "
                f"inactive={inactive_count} | activated={active_count} | near-threshold={switching_count}"
            )

    plotter.close()

    print()
    print("Phase 5 120-platelet activation video created successfully.")
    print(f"Saved video      : {VIDEO_PATH}")
    print(f"Saved final frame: {FINAL_FRAME_PATH}")
    print()
    print("Data sources:")
    print(f"Positions : {POSITIONS_PATH}")
    print(f"Activation: {ACTIVATION_PATH}")
    print(f"Shear     : {SHEAR_PATH}")
    print(f"Inactive mesh : {inactive_path}")
    print(f"Activated mesh: {activated_path}")
    print()
    print("This video shows:")
    print("- 120 real platelet mesh instances")
    print("- inactive platelets in blue")
    print("- activated platelets in red")
    print("- gold rings for near-threshold switching")
    print("- cylindrical flow direction using arrows")
    print("- activation-based state switching over time")


if __name__ == "__main__":
    main()