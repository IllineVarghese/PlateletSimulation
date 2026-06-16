from pathlib import Path
import numpy as np
import pyvista as pv

from mesh_utils import load_mesh, center_and_scale_mesh


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Real Phase 4 final demo output
POSITIONS_PATH = PROJECT_ROOT / "results" / "phase4" / "final_demo" / "positions.npy"
ACTIVATION_PATH = PROJECT_ROOT / "results" / "phase4" / "final_demo" / "activation.npy"
SHEAR_PATH = PROJECT_ROOT / "results" / "phase4" / "final_demo" / "shear_input.npy"

# Platelet mesh files
INACTIVE_MESH_PATH = PROJECT_ROOT / "data" / "meshes" / "platelet" / "inactive.obj"
ACTIVATED_MESH_PATH = PROJECT_ROOT / "data" / "meshes" / "platelet" / "activated.obj"

OUTPUT_DIR = PROJECT_ROOT / "results" / "phase5" / "week2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "phase4_mesh_switching_snapshot.png"

ACTIVATION_THRESHOLD = 0.50

# Rendering all 500 real meshes is possible but slow/heavy.
# For a clean first snapshot, we render a representative subset.
MAX_RENDERED_PLATELETS = 120

# Use last frame because activation is most developed there.
FRAME_INDEX = -1


def load_phase4_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not POSITIONS_PATH.exists():
        raise FileNotFoundError(f"Missing positions file: {POSITIONS_PATH}")

    if not ACTIVATION_PATH.exists():
        raise FileNotFoundError(f"Missing activation file: {ACTIVATION_PATH}")

    if not SHEAR_PATH.exists():
        raise FileNotFoundError(f"Missing shear file: {SHEAR_PATH}")

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
        raise ValueError(
            f"positions and activation do not match: {positions.shape[:2]} vs {activation.shape}"
        )

    if positions.shape[:2] != shear.shape:
        raise ValueError(
            f"positions and shear do not match: {positions.shape[:2]} vs {shear.shape}"
        )

    return positions, activation, shear


def make_vessel_proxy_from_positions(frame_positions: np.ndarray) -> pv.PolyData:
    """
    Build a simple vessel proxy around the real Phase 4 coordinate range.
    """
    x_min = float(np.min(frame_positions[:, 0]))
    x_max = float(np.max(frame_positions[:, 0]))

    length = max(x_max - x_min, 1.0)
    center_x = 0.5 * (x_min + x_max)

    vessel = pv.Cylinder(
        center=(center_x, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0),
        radius=1.05,
        height=length,
        resolution=128,
    )
    return vessel


def transform_platelet_mesh(
    base_mesh: pv.PolyData,
    position: np.ndarray,
    scale: float,
    rotation_x: float,
    rotation_y: float,
    rotation_z: float,
) -> pv.PolyData:
    mesh = base_mesh.copy()

    mesh.scale(scale, inplace=True)
    mesh.rotate_x(rotation_x, inplace=True)
    mesh.rotate_y(rotation_y, inplace=True)
    mesh.rotate_z(rotation_z, inplace=True)
    mesh.translate(tuple(position), inplace=True)

    return mesh


def choose_representative_indices(
    activation_frame: np.ndarray,
    max_count: int,
) -> np.ndarray:
    """
    Choose a balanced subset of inactive and activated platelets.

    This avoids rendering only activated cells if the final frame is mostly activated.
    """
    inactive_indices = np.where(activation_frame < ACTIVATION_THRESHOLD)[0]
    activated_indices = np.where(activation_frame >= ACTIVATION_THRESHOLD)[0]

    half = max_count // 2

    selected = []

    if inactive_indices.size > 0:
        selected.extend(inactive_indices[:half].tolist())

    if activated_indices.size > 0:
        selected.extend(activated_indices[: max_count - len(selected)].tolist())

    selected = np.array(selected, dtype=int)

    if selected.size == 0:
        selected = np.arange(min(max_count, activation_frame.size), dtype=int)

    return selected[:max_count]


def main() -> None:
    print("Phase 5 Week 2 Day 2: Render real Phase 4 data with platelet meshes")

    positions, activation, shear = load_phase4_data()

    frame_positions = positions[FRAME_INDEX]
    frame_activation = activation[FRAME_INDEX]
    frame_shear = shear[FRAME_INDEX]

    print(f"positions shape:  {positions.shape}")
    print(f"activation shape: {activation.shape}")
    print(f"shear shape:      {shear.shape}")
    print(f"selected frame:   {FRAME_INDEX}")
    print(f"frame activation min/max: {frame_activation.min():.3f} / {frame_activation.max():.3f}")
    print(f"frame shear min/max:      {frame_shear.min():.3f} / {frame_shear.max():.3f}")

    inactive_base = center_and_scale_mesh(
        load_mesh(INACTIVE_MESH_PATH),
        target_size=1.0,
    )
    activated_base = center_and_scale_mesh(
        load_mesh(ACTIVATED_MESH_PATH),
        target_size=1.0,
    )

    selected_indices = choose_representative_indices(
        frame_activation,
        max_count=MAX_RENDERED_PLATELETS,
    )

    selected_positions = frame_positions[selected_indices]
    selected_activation = frame_activation[selected_indices]
    selected_shear = frame_shear[selected_indices]

    plotter = pv.Plotter(off_screen=True, window_size=(1800, 1000))
    plotter.set_background("white")

    vessel = make_vessel_proxy_from_positions(frame_positions)
    plotter.add_mesh(
        vessel,
        opacity=0.10,
        smooth_shading=True,
        show_edges=True,
        line_width=0.25,
    )

    inactive_count = 0
    activated_count = 0

    for local_i, idx in enumerate(selected_indices):
        position = frame_positions[idx]
        act = float(frame_activation[idx])
        shr = float(frame_shear[idx])

        if act >= ACTIVATION_THRESHOLD:
            base_mesh = activated_base
            activated_count += 1
        else:
            base_mesh = inactive_base
            inactive_count += 1

        platelet = transform_platelet_mesh(
            base_mesh=base_mesh,
            position=position,
            scale=0.070,
            rotation_x=(idx * 7.0) % 360,
            rotation_y=(idx * 13.0) % 360,
            rotation_z=(idx * 17.0) % 360,
        )

        plotter.add_mesh(
            platelet,
            smooth_shading=True,
            show_edges=False,
            opacity=1.0,
        )

        # Label only a few cells to avoid clutter.
        if local_i in [0, 15, 30, 45, 60, 90, 119]:
            label_position = (position[0], position[1] - 0.12, position[2])
            state = "act" if act >= ACTIVATION_THRESHOLD else "inact"
            plotter.add_point_labels(
                [label_position],
                [f"{state}\nA={act:.2f}\nS={shr:.2f}"],
                font_size=10,
                text_color="black",
                point_color="white",
                point_size=1,
                shape=None,
                always_visible=True,
            )

    title = "Phase 5 Week 2: Real Phase 4 data rendered with platelet meshes"
    subtitle = (
        f"Frame {positions.shape[0] - 1} | rendered={len(selected_indices)} of {positions.shape[1]} platelets | "
        f"inactive={inactive_count}, activated={activated_count} | "
        f"threshold={ACTIVATION_THRESHOLD:.2f}"
    )

    plotter.add_text(
        title,
        position=(390, 940),
        font_size=17,
        color="black",
    )
    plotter.add_text(
        subtitle,
        position=(330, 900),
        font_size=13,
        color="black",
    )

    plotter.add_axes()

    # Camera adjusted for Phase 4 final_demo coordinates.
    plotter.camera_position = [
        (4.0, -6.5, 3.2),
        (4.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ]

    plotter.screenshot(str(OUTPUT_PATH))
    plotter.close()

    print(f"Saved output: {OUTPUT_PATH}")
    print(f"Rendered platelets: {len(selected_indices)}")
    print(f"Inactive rendered:  {inactive_count}")
    print(f"Activated rendered: {activated_count}")
    print("Week 2 Day 2 real Phase 4 mesh snapshot complete.")


if __name__ == "__main__":
    main()