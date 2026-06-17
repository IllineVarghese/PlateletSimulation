from pathlib import Path
import numpy as np
import pyvista as pv

from mesh_utils import load_mesh, center_and_scale_mesh


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Real Phase 4 final demo data
POSITIONS_PATH = PROJECT_ROOT / "results" / "phase4" / "final_demo" / "positions.npy"
ACTIVATION_PATH = PROJECT_ROOT / "results" / "phase4" / "final_demo" / "activation.npy"
SHEAR_PATH = PROJECT_ROOT / "results" / "phase4" / "final_demo" / "shear_input.npy"

# Platelet meshes
INACTIVE_MESH_PATH = PROJECT_ROOT / "data" / "meshes" / "platelet" / "inactive.obj"
ACTIVATED_MESH_PATH = PROJECT_ROOT / "data" / "meshes" / "platelet" / "activated.obj"

OUTPUT_DIR = PROJECT_ROOT / "results" / "phase5" / "week2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "phase5_thesis_activation_snapshot.png"

ACTIVATION_THRESHOLD = 0.50
FRAME_INDEX = -1
MAX_RENDERED_PLATELETS = 160


def load_phase4_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.load(POSITIONS_PATH)
    activation = np.load(ACTIVATION_PATH)
    shear = np.load(SHEAR_PATH)

    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"Expected positions shape (frames, platelets, 3), got {positions.shape}")

    if activation.ndim != 2:
        raise ValueError(f"Expected activation shape (frames, platelets), got {activation.shape}")

    if shear.ndim != 2:
        raise ValueError(f"Expected shear shape (frames, platelets), got {shear.shape}")

    if positions.shape[:2] != activation.shape or positions.shape[:2] != shear.shape:
        raise ValueError("positions, activation, and shear shapes do not match")

    return positions, activation, shear


def activation_to_color(act: float) -> tuple[float, float, float]:
    """
    Simple blue-to-red activation color map.

    Low activation  -> blue
    Mid activation  -> pale/white
    High activation -> red
    """
    a = float(np.clip(act, 0.0, 1.0))

    if a < 0.5:
        t = a / 0.5
        r = 0.15 + 0.70 * t
        g = 0.35 + 0.45 * t
        b = 0.95
    else:
        t = (a - 0.5) / 0.5
        r = 0.95
        g = 0.80 - 0.55 * t
        b = 0.85 - 0.70 * t

    return r, g, b


def choose_thesis_indices(
    frame_activation: np.ndarray,
    frame_shear: np.ndarray,
    max_count: int,
) -> np.ndarray:
    """
    Select platelets that make the figure informative:
    - low activation examples
    - high activation examples
    - high shear examples
    """
    n = frame_activation.size

    low_idx = np.argsort(frame_activation)[: max_count // 4]
    high_idx = np.argsort(frame_activation)[::-1][: max_count // 2]
    shear_idx = np.argsort(frame_shear)[::-1][: max_count // 4]

    selected = np.unique(np.concatenate([low_idx, high_idx, shear_idx]))

    if selected.size < max_count:
        remaining = np.setdiff1d(np.arange(n), selected)
        fill_count = max_count - selected.size
        selected = np.concatenate([selected, remaining[:fill_count]])

    return selected[:max_count]


def make_vessel_proxy(all_positions: np.ndarray) -> pv.PolyData:
    x_min = float(np.min(all_positions[..., 0]))
    x_max = float(np.max(all_positions[..., 0]))
    center_x = 0.5 * (x_min + x_max)
    length = max(x_max - x_min, 1.0)

    vessel = pv.Cylinder(
        center=(center_x, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0),
        radius=1.05,
        height=length,
        resolution=128,
    )
    return vessel


def make_centerline(all_positions: np.ndarray) -> pv.PolyData:
    x_min = float(np.min(all_positions[..., 0]))
    x_max = float(np.max(all_positions[..., 0]))

    return pv.Line(
        pointa=(x_min, 0.0, 0.0),
        pointb=(x_max, 0.0, 0.0),
        resolution=1,
    )


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


def main() -> None:
    print("Phase 5 Week 2 Day 4: Thesis-level visual snapshot")

    positions, activation, shear = load_phase4_data()

    frame_positions = positions[FRAME_INDEX]
    frame_activation = activation[FRAME_INDEX]
    frame_shear = shear[FRAME_INDEX]

    n_frames, n_platelets, _ = positions.shape
    selected_indices = choose_thesis_indices(
        frame_activation=frame_activation,
        frame_shear=frame_shear,
        max_count=MAX_RENDERED_PLATELETS,
    )

    selected_activation = frame_activation[selected_indices]
    selected_shear = frame_shear[selected_indices]

    inactive_count = int(np.sum(selected_activation < ACTIVATION_THRESHOLD))
    activated_count = int(np.sum(selected_activation >= ACTIVATION_THRESHOLD))

    print(f"positions shape:  {positions.shape}")
    print(f"activation shape: {activation.shape}")
    print(f"shear shape:      {shear.shape}")
    print(f"selected frame:   {n_frames - 1 if FRAME_INDEX == -1 else FRAME_INDEX}")
    print(f"rendered count:   {selected_indices.size}")
    print(f"inactive count:   {inactive_count}")
    print(f"activated count:  {activated_count}")
    print(f"mean activation:  {np.mean(selected_activation):.3f}")
    print(f"mean shear:       {np.mean(selected_shear):.3f}")

    inactive_base = center_and_scale_mesh(load_mesh(INACTIVE_MESH_PATH), target_size=1.0)
    activated_base = center_and_scale_mesh(load_mesh(ACTIVATED_MESH_PATH), target_size=1.0)

    vessel = make_vessel_proxy(positions)
    centerline = make_centerline(positions)

    plotter = pv.Plotter(off_screen=True, window_size=(1900, 1100))
    plotter.set_background("white")

    # Clean vessel surface
    plotter.add_mesh(
        vessel,
        color=(0.86, 0.90, 0.92),
        opacity=0.11,
        smooth_shading=True,
        show_edges=False,
    )

    # Subtle flow centerline
    plotter.add_mesh(
        centerline,
        color="gray",
        line_width=3,
        opacity=0.30,
    )

    for local_i, idx in enumerate(selected_indices):
        pos = frame_positions[idx]
        act = float(frame_activation[idx])
        shr = float(frame_shear[idx])

        if act >= ACTIVATION_THRESHOLD:
            base_mesh = activated_base
            state_scale_bonus = 0.018
        else:
            base_mesh = inactive_base
            state_scale_bonus = 0.0

        # Slightly exaggerate scale for readability
        scale = 0.070 + 0.020 * act + 0.010 * shr + state_scale_bonus

        platelet = transform_platelet_mesh(
            base_mesh=base_mesh,
            position=pos,
            scale=scale,
            rotation_x=(idx * 7.0) % 360,
            rotation_y=(idx * 13.0) % 360,
            rotation_z=(idx * 17.0) % 360,
        )

        color = activation_to_color(act)

        plotter.add_mesh(
            platelet,
            color=color,
            smooth_shading=True,
            show_edges=False,
            opacity=0.97,
            specular=0.25,
        )

    # Add a few meaningful labels only
    label_candidates = [
        int(selected_indices[np.argmin(selected_activation)]),
        int(selected_indices[np.argmax(selected_activation)]),
        int(selected_indices[np.argmax(selected_shear)]),
    ]

    label_texts = [
        "lowest activation",
        "highest activation",
        "highest shear",
    ]

    for idx, label in zip(label_candidates, label_texts):
        pos = frame_positions[idx]
        act = float(frame_activation[idx])
        shr = float(frame_shear[idx])

        plotter.add_point_labels(
            [(pos[0], pos[1] - 0.18, pos[2])],
            [f"{label}\nA={act:.2f}, S={shr:.2f}"],
            font_size=12,
            text_color="black",
            point_color="white",
            point_size=1,
            shape=None,
            always_visible=True,
        )

    # Header
    plotter.add_text(
        "Phase 5: Real Platelet Mesh Visualization with Activation-Based State Switching",
        position=(250, 1045),
        font_size=18,
        color="black",
    )

    plotter.add_text(
        f"Real Phase 4 data | frame {n_frames - 1 if FRAME_INDEX == -1 else FRAME_INDEX} | "
        f"rendered={selected_indices.size}/{n_platelets} | inactive={inactive_count}, activated={activated_count}",
        position=(340, 1010),
        font_size=13,
        color="black",
    )

    plotter.add_text(
        f"Mean activation={np.mean(selected_activation):.2f} | mean shear={np.mean(selected_shear):.2f} | "
        f"state threshold={ACTIVATION_THRESHOLD:.2f}",
        position=(470, 982),
        font_size=12,
        color="black",
    )

    # Visual legend
    plotter.add_text(
        "Visual encoding",
        position=(55, 965),
        font_size=13,
        color="black",
    )

    plotter.add_text(
        "Shape: inactive.obj / activated.obj",
        position=(55, 935),
        font_size=11,
        color="dimgray",
    )

    plotter.add_text(
        "Color: activation level",
        position=(55, 910),
        font_size=11,
        color="dimgray",
    )

    plotter.add_text(
        "Blue = low activation",
        position=(55, 880),
        font_size=11,
        color=(0.15, 0.35, 0.95),
    )

    plotter.add_text(
        "Red = high activation",
        position=(55, 855),
        font_size=11,
        color=(0.95, 0.20, 0.15),
    )

    plotter.add_text(
        "Size: activation + shear",
        position=(55, 830),
        font_size=11,
        color="dimgray",
    )

    plotter.add_text(
        "Flow direction →",
        position=(1550, 890),
        font_size=13,
        color="dimgray",
    )

    plotter.add_axes()

    # Better camera for thesis-style view
    plotter.camera_position = [
        (4.1, -6.8, 3.5),
        (4.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ]

    plotter.screenshot(str(OUTPUT_PATH))
    plotter.close()

    print(f"Saved thesis visual snapshot: {OUTPUT_PATH}")
    print("Week 2 Day 4 thesis-level snapshot complete.")


if __name__ == "__main__":
    main()