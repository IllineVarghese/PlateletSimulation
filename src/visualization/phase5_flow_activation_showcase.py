from pathlib import Path
import numpy as np
import pyvista as pv
import imageio.v2 as imageio


# ============================================================
# Phase 5: Real Mesh Activation Switching Showcase
# ------------------------------------------------------------
# Purpose:
# - Uses actual inactive and activated platelet meshes
# - Shows activation-based mesh switching
# - Tracks one switching platelet in close-up
# - Shows local vessel region with real mesh platelets
# - Adds small flow arrows without hiding platelet meshes
# ============================================================


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "phase5" / "presentation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_PATH = OUT_DIR / "phase5_real_mesh_switching_showcase.mp4"
FINAL_FRAME_PATH = OUT_DIR / "phase5_real_mesh_switching_showcase_final_frame.png"

POSITIONS_PATH = RESULTS / "phase4" / "final_demo" / "positions.npy"
ACTIVATION_PATH = RESULTS / "phase4" / "final_demo" / "activation.npy"
SHEAR_PATH = RESULTS / "phase4" / "final_demo" / "shear_input.npy"

INACTIVE_MESH_PATH = RESULTS / "phase5" / "week3" / "optimized_meshes" / "inactive_decimated.vtp"
ACTIVATED_MESH_PATH = RESULTS / "phase5" / "week3" / "optimized_meshes" / "activated_decimated.vtp"

WINDOW_SIZE = (1600, 900)
FPS = 8
SUBSTEPS = 2

ACTIVATION_THRESHOLD = 0.50
N_LOCAL_PLATELETS = 32

LOCAL_PLATELET_SCALE = 0.16
TRACKED_PLATELET_SCALE = 1.05

FLOW_ARROW_SCALE = 0.18
FLOW_ARROW_OPACITY = 0.18

LOW_COLOR = np.array([0.10, 0.32, 0.95])
HIGH_COLOR = np.array([0.95, 0.18, 0.10])


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def normalize_mesh(mesh: pv.PolyData, target_size: float = 1.0) -> pv.PolyData:
    mesh = mesh.triangulate().clean()
    mesh = mesh.copy(deep=True)

    bounds = mesh.bounds
    center = np.array(
        [
            0.5 * (bounds[0] + bounds[1]),
            0.5 * (bounds[2] + bounds[3]),
            0.5 * (bounds[4] + bounds[5]),
        ],
        dtype=float,
    )

    extent = max(
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
    )

    if extent <= 0:
        raise ValueError("Mesh has invalid zero extent.")

    mesh.translate(-center, inplace=True)
    mesh.scale(target_size / extent, inplace=True)

    return mesh


def load_data():
    require_file(POSITIONS_PATH, "Phase 4 positions")
    require_file(ACTIVATION_PATH, "Phase 4 activation")
    require_file(SHEAR_PATH, "Phase 4 shear input")

    positions = np.load(POSITIONS_PATH)
    activation = np.load(ACTIVATION_PATH)
    shear = np.load(SHEAR_PATH)

    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"Unexpected positions shape: {positions.shape}")

    if activation.ndim != 2:
        raise ValueError(f"Unexpected activation shape: {activation.shape}")

    if shear.ndim != 2:
        raise ValueError(f"Unexpected shear shape: {shear.shape}")

    if positions.shape[:2] != activation.shape:
        raise ValueError(f"positions and activation mismatch: {positions.shape[:2]} vs {activation.shape}")

    if positions.shape[:2] != shear.shape:
        raise ValueError(f"positions and shear mismatch: {positions.shape[:2]} vs {shear.shape}")

    activation = np.clip(activation, 0.0, 1.0)
    shear = np.clip(shear, 0.0, None)

    print(f"Using positions:  {POSITIONS_PATH}")
    print(f"Using activation: {ACTIVATION_PATH}")
    print(f"Using shear:      {SHEAR_PATH}")
    print(f"positions shape:  {positions.shape}")
    print(f"activation shape: {activation.shape}")
    print(f"shear shape:      {shear.shape}")

    return positions, activation, shear


def load_meshes():
    require_file(INACTIVE_MESH_PATH, "inactive decimated platelet mesh")
    require_file(ACTIVATED_MESH_PATH, "activated decimated platelet mesh")

    inactive = normalize_mesh(pv.read(INACTIVE_MESH_PATH), target_size=1.0)
    activated = normalize_mesh(pv.read(ACTIVATED_MESH_PATH), target_size=1.0)

    print(f"Inactive mesh:  points={inactive.n_points}, cells={inactive.n_cells}")
    print(f"Activated mesh: points={activated.n_points}, cells={activated.n_cells}")

    return inactive, activated


def activation_to_color(value: float):
    value = float(np.clip(value, 0.0, 1.0))
    color = (1.0 - value) * LOW_COLOR + value * HIGH_COLOR
    return tuple(color.tolist())


def crossing_frame_for_platelet(activation_trace: np.ndarray):
    below = activation_trace[:-1] < ACTIVATION_THRESHOLD
    above = activation_trace[1:] >= ACTIVATION_THRESHOLD
    crossings = np.where(below & above)[0]

    if len(crossings) == 0:
        return None

    return int(crossings[0] + 1)


def choose_tracked_platelet(activation: np.ndarray) -> tuple[int, int]:
    n_platelets = activation.shape[1]

    candidates = []

    for platelet_id in range(n_platelets):
        crossing_frame = crossing_frame_for_platelet(activation[:, platelet_id])

        if crossing_frame is not None:
            delta = float(activation[-1, platelet_id] - activation[0, platelet_id])
            final_value = float(activation[-1, platelet_id])
            candidates.append((platelet_id, crossing_frame, delta, final_value))

    if candidates:
        candidates.sort(key=lambda item: (-item[2], item[1], -item[3]))
        tracked_id, crossing_frame, _, _ = candidates[0]
        return int(tracked_id), int(crossing_frame)

    # fallback: choose platelet with largest activation increase
    delta = activation[-1] - activation[0]
    tracked_id = int(np.argmax(delta))
    crossing_frame = int(np.argmax(np.diff(activation[:, tracked_id])) + 1)

    return tracked_id, crossing_frame


def choose_local_platelets(positions: np.ndarray, activation: np.ndarray, tracked_id: int, crossing_frame: int):
    n_platelets = positions.shape[1]

    reference_position = positions[crossing_frame, tracked_id, :2]
    all_positions = positions[crossing_frame, :, :2]

    distances = np.linalg.norm(all_positions - reference_position, axis=1)
    nearest = np.argsort(distances)

    selected = list(nearest[:N_LOCAL_PLATELETS])

    if tracked_id not in selected:
        selected[-1] = tracked_id

    selected = np.array(selected, dtype=int)

    switch_count = 0
    for idx in selected:
        if crossing_frame_for_platelet(activation[:, idx]) is not None:
            switch_count += 1

    print(f"Tracked platelet ID: {tracked_id}")
    print(f"Tracked platelet crossing frame: {crossing_frame}")
    print(f"Selected local platelets: {len(selected)}")
    print(f"Switching platelets in local view: {switch_count}")

    return selected


def interpolate_frame(positions, activation, shear, frame_float):
    n_frames = positions.shape[0]

    if frame_float >= n_frames - 1:
        return positions[-1], activation[-1], shear[-1]

    i0 = int(np.floor(frame_float))
    i1 = min(i0 + 1, n_frames - 1)
    t = frame_float - i0

    pos = (1.0 - t) * positions[i0] + t * positions[i1]
    act = (1.0 - t) * activation[i0] + t * activation[i1]
    shr = (1.0 - t) * shear[i0] + t * shear[i1]

    return pos, act, shr


def compute_local_bounds(positions: np.ndarray, selected_ids: np.ndarray):
    local_positions = positions[:, selected_ids, :]

    x = local_positions[:, :, 0].ravel()
    y = local_positions[:, :, 1].ravel()

    x_min = float(np.percentile(x, 2))
    x_max = float(np.percentile(x, 98))
    y_min = float(np.percentile(y, 2))
    y_max = float(np.percentile(y, 98))

    x_pad = 0.25 * (x_max - x_min + 1e-6)
    y_pad = 0.45 * (y_max - y_min + 1e-6)

    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    if (x_max - x_min) < 1.5:
        center = 0.5 * (x_min + x_max)
        x_min = center - 0.75
        x_max = center + 0.75

    if (y_max - y_min) < 1.2:
        center = 0.5 * (y_min + y_max)
        y_min = center - 0.60
        y_max = center + 0.60

    return x_min, x_max, y_min, y_max


def make_vessel_segment(bounds):
    x_min, x_max, y_min, y_max = bounds
    center_x = 0.5 * (x_min + x_max)
    length = max(x_max - x_min, 1.0)
    radius = max(abs(y_min), abs(y_max), 0.75)

    vessel = pv.Cylinder(
        center=(center_x, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0),
        radius=radius,
        height=length,
        resolution=96,
    )

    return vessel


def build_flow_arrows(bounds):
    x_min, x_max, y_min, y_max = bounds

    xs = np.linspace(x_min + 0.12 * (x_max - x_min), x_max - 0.12 * (x_max - x_min), 5)
    ys = np.linspace(y_min + 0.30 * (y_max - y_min), y_max - 0.30 * (y_max - y_min), 2)

    arrows = []

    for y in ys:
        for x in xs:
            arrow = pv.Arrow(
                start=(x, y, -0.35),
                direction=(1.0, 0.0, 0.0),
                scale=FLOW_ARROW_SCALE,
                tip_length=0.30,
                tip_radius=0.045,
                shaft_radius=0.012,
            )
            arrows.append(arrow)

    merged = arrows[0]
    for arrow in arrows[1:]:
        merged = merged.merge(arrow)

    return merged


def transform_local_platelet(base_mesh, position, platelet_id: int, activation_value: float):
    mesh = base_mesh.copy(deep=True)

    scale = LOCAL_PLATELET_SCALE * (0.90 + 0.35 * activation_value)

    if activation_value >= ACTIVATION_THRESHOLD:
        scale *= 1.10

    mesh.scale(scale, inplace=True)

    mesh.rotate_x((platelet_id * 11.0) % 360, inplace=True)
    mesh.rotate_y((platelet_id * 17.0) % 360, inplace=True)
    mesh.rotate_z((platelet_id * 23.0) % 360, inplace=True)

    mesh.translate(tuple(position), inplace=True)

    return mesh


def transform_tracked_platelet(base_mesh, activation_value: float):
    mesh = base_mesh.copy(deep=True)

    scale = TRACKED_PLATELET_SCALE * (0.90 + 0.25 * activation_value)

    if activation_value >= ACTIVATION_THRESHOLD:
        scale *= 1.08

    mesh.scale(scale, inplace=True)

    mesh.rotate_x(60.0, inplace=True)
    mesh.rotate_y(0.0, inplace=True)
    mesh.rotate_z(35.0, inplace=True)

    return mesh


def add_switch_highlight(plotter, center, radius):
    sphere = pv.Sphere(
        radius=radius,
        center=tuple(center),
        theta_resolution=32,
        phi_resolution=16,
    )

    plotter.add_mesh(
        sphere,
        color="gold",
        style="wireframe",
        line_width=3,
        opacity=0.90,
    )


def set_left_camera(plotter, bounds):
    x_min, x_max, y_min, y_max = bounds

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)

    x_range = x_max - x_min
    y_range = y_max - y_min

    aspect = (WINDOW_SIZE[0] / 2) / WINDOW_SIZE[1]
    parallel_scale = max(0.60 * y_range, 0.60 * x_range / max(aspect, 0.01))

    plotter.camera_position = [
        (x_mid, y_mid, 6.5),
        (x_mid, y_mid, 0.0),
        (0.0, 1.0, 0.0),
    ]

    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = parallel_scale


def set_right_camera(plotter):
    plotter.camera_position = [
        (2.2, -3.0, 1.8),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ]

    plotter.camera.parallel_projection = False


def render_frame(
    frame_number,
    total_frames,
    frame_float,
    pos_frame,
    act_frame,
    shear_frame,
    selected_ids,
    tracked_id,
    inactive_mesh,
    activated_mesh,
    bounds,
):
    plotter = pv.Plotter(
        off_screen=True,
        shape=(1, 2),
        window_size=WINDOW_SIZE,
        border=False,
    )

    plotter.set_background("white")

    # --------------------------------------------------------
    # Left subplot: local vessel region with real platelet meshes
    # --------------------------------------------------------
    plotter.subplot(0, 0)

    vessel = make_vessel_segment(bounds)
    flow_arrows = build_flow_arrows(bounds)

    plotter.add_mesh(
        vessel,
        color=(0.86, 0.88, 0.90),
        opacity=0.12,
        smooth_shading=True,
        show_edges=False,
    )

    plotter.add_mesh(
        flow_arrows,
        color=(0.25, 0.50, 0.85),
        opacity=FLOW_ARROW_OPACITY,
        smooth_shading=True,
    )

    local_activation = act_frame[selected_ids]
    local_shear = shear_frame[selected_ids]

    inactive_count = 0
    activated_count = 0
    switching_count = 0

    for platelet_id in selected_ids:
        pos = pos_frame[platelet_id]
        act = float(act_frame[platelet_id])

        if act >= ACTIVATION_THRESHOLD:
            base_mesh = activated_mesh
            activated_count += 1
        else:
            base_mesh = inactive_mesh
            inactive_count += 1

        platelet_mesh = transform_local_platelet(
            base_mesh=base_mesh,
            position=pos,
            platelet_id=int(platelet_id),
            activation_value=act,
        )

        plotter.add_mesh(
            platelet_mesh,
            color=activation_to_color(act),
            smooth_shading=True,
            show_edges=False,
            opacity=1.0,
            specular=0.18,
        )

        if abs(act - ACTIVATION_THRESHOLD) < 0.055:
            switching_count += 1
            add_switch_highlight(plotter, pos, radius=0.22)

    plotter.add_text(
        "Local vessel view: real platelet meshes",
        position="upper_left",
        font_size=11,
        color="black",
    )

    plotter.add_text(
        f"Inactive={inactive_count} | Activated={activated_count} | Switching={switching_count}",
        position=(20, 735),
        font_size=10,
        color="black",
    )

    plotter.add_text(
        "Small blue arrows = prescribed Poiseuille flow field",
        position=(20, 705),
        font_size=9,
        color=(0.20, 0.40, 0.85),
    )

    plotter.add_text(
        "Gold ring = platelet near activation threshold",
        position=(20, 675),
        font_size=9,
        color="gold",
    )

    set_left_camera(plotter, bounds)

    # --------------------------------------------------------
    # Right subplot: tracked platelet close-up
    # --------------------------------------------------------
    plotter.subplot(0, 1)

    tracked_act = float(act_frame[tracked_id])
    tracked_shear = float(shear_frame[tracked_id])

    if tracked_act >= ACTIVATION_THRESHOLD:
        tracked_base_mesh = activated_mesh
        tracked_state = "ACTIVATED MESH"
    else:
        tracked_base_mesh = inactive_mesh
        tracked_state = "INACTIVE MESH"

    tracked_mesh = transform_tracked_platelet(
        base_mesh=tracked_base_mesh,
        activation_value=tracked_act,
    )

    plotter.add_mesh(
        tracked_mesh,
        color=activation_to_color(tracked_act),
        smooth_shading=True,
        show_edges=False,
        opacity=1.0,
        specular=0.25,
    )

    if abs(tracked_act - ACTIVATION_THRESHOLD) < 0.060:
        add_switch_highlight(plotter, (0.0, 0.0, 0.0), radius=1.05)

    plotter.add_text(
        "Tracked platelet close-up",
        position="upper_left",
        font_size=12,
        color="black",
    )

    plotter.add_text(
        f"Platelet ID: {tracked_id}",
        position=(20, 735),
        font_size=10,
        color="black",
    )

    plotter.add_text(
        f"Activation: {tracked_act:.2f}",
        position=(20, 705),
        font_size=10,
        color="black",
    )

    plotter.add_text(
        f"Shear input: {tracked_shear:.2f}",
        position=(20, 675),
        font_size=10,
        color="black",
    )

    state_color = "red" if tracked_act >= ACTIVATION_THRESHOLD else "blue"

    plotter.add_text(
        tracked_state,
        position=(20, 635),
        font_size=13,
        color=state_color,
    )

    plotter.add_text(
        "Switching rule: activation < 0.50 = inactive | activation >= 0.50 = activated",
        position=(20, 585),
        font_size=8,
        color="dimgray",
    )

    plotter.add_text(
        f"Video frame {frame_number + 1}/{total_frames}",
        position="upper_right",
        font_size=10,
        color="black",
    )

    set_right_camera(plotter)

    image = plotter.screenshot(return_img=True)
    plotter.close()

    return image


def main():
    print("Creating Phase 5 real-mesh switching showcase video...")

    positions, activation, shear = load_data()
    inactive_mesh, activated_mesh = load_meshes()

    tracked_id, crossing_frame = choose_tracked_platelet(activation)
    selected_ids = choose_local_platelets(
        positions=positions,
        activation=activation,
        tracked_id=tracked_id,
        crossing_frame=crossing_frame,
    )

    bounds = compute_local_bounds(positions, selected_ids)

    print(f"Local bounds: {bounds}")

    n_frames = positions.shape[0]
    frame_times = np.linspace(0, n_frames - 1, (n_frames - 1) * SUBSTEPS + 1)
    total_frames = len(frame_times)

    print(f"Total output frames: {total_frames}")
    print(f"FPS: {FPS}")
    print(f"Output video: {VIDEO_PATH}")

    with imageio.get_writer(
        VIDEO_PATH,
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=16,
    ) as writer:
        final_image = None

        for frame_number, frame_float in enumerate(frame_times):
            pos_frame, act_frame, shear_frame = interpolate_frame(
                positions=positions,
                activation=activation,
                shear=shear,
                frame_float=frame_float,
            )

            image = render_frame(
                frame_number=frame_number,
                total_frames=total_frames,
                frame_float=frame_float,
                pos_frame=pos_frame,
                act_frame=act_frame,
                shear_frame=shear_frame,
                selected_ids=selected_ids,
                tracked_id=tracked_id,
                inactive_mesh=inactive_mesh,
                activated_mesh=activated_mesh,
                bounds=bounds,
            )

            writer.append_data(image)
            final_image = image

            if (frame_number + 1) % 5 == 0 or frame_number == total_frames - 1:
                print(f"Rendered frame {frame_number + 1}/{total_frames}")

    if final_image is not None:
        imageio.imwrite(FINAL_FRAME_PATH, final_image)

    print()
    print("Done.")
    print(f"Saved video:       {VIDEO_PATH}")
    print(f"Saved final frame: {FINAL_FRAME_PATH}")
    print()
    print("This video uses real inactive and activated platelet meshes.")
    print("It shows activation-based mesh switching and a tracked platelet close-up.")


if __name__ == "__main__":
    main()