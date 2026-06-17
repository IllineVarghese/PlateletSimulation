from pathlib import Path
import numpy as np
import pyvista as pv

from mesh_utils import center_and_scale_mesh


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Real Phase 4 final demo data
POSITIONS_PATH = PROJECT_ROOT / "results" / "phase4" / "final_demo" / "positions.npy"
ACTIVATION_PATH = PROJECT_ROOT / "results" / "phase4" / "final_demo" / "activation.npy"
SHEAR_PATH = PROJECT_ROOT / "results" / "phase4" / "final_demo" / "shear_input.npy"

# Decimated meshes created in Week 3 Day 2
INACTIVE_DECIMATED_PATH = (
    PROJECT_ROOT / "results" / "phase5" / "week3" / "optimized_meshes" / "inactive_decimated.vtp"
)
ACTIVATED_DECIMATED_PATH = (
    PROJECT_ROOT / "results" / "phase5" / "week3" / "optimized_meshes" / "activated_decimated.vtp"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "phase5" / "week3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "phase5_decimated_deformed_thesis_video.mp4"

ACTIVATION_THRESHOLD = 0.50
MAX_RENDERED_PLATELETS = 120
FPS = 6

# Keep protrusions limited, otherwise video becomes slow and visually crowded.
ENABLE_PROTRUSIONS = True
MAX_PROTRUSION_PLATELETS_PER_FRAME = 28
PROTRUSION_ACTIVATION_MIN = 0.72


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

    if positions.shape[:2] != activation.shape:
        raise ValueError("positions and activation shapes do not match")

    if positions.shape[:2] != shear.shape:
        raise ValueError("positions and shear shapes do not match")

    return positions, activation, shear


def load_decimated_meshes() -> tuple[pv.PolyData, pv.PolyData]:
    if not INACTIVE_DECIMATED_PATH.exists():
        raise FileNotFoundError(
            f"Missing inactive decimated mesh: {INACTIVE_DECIMATED_PATH}\n"
            "Run Week 3 Day 2 first: python src/visualization/phase5_mesh_decimation_test.py"
        )

    if not ACTIVATED_DECIMATED_PATH.exists():
        raise FileNotFoundError(
            f"Missing activated decimated mesh: {ACTIVATED_DECIMATED_PATH}\n"
            "Run Week 3 Day 2 first: python src/visualization/phase5_mesh_decimation_test.py"
        )

    inactive = center_and_scale_mesh(
        pv.read(INACTIVE_DECIMATED_PATH),
        target_size=1.0,
    )

    activated = center_and_scale_mesh(
        pv.read(ACTIVATED_DECIMATED_PATH),
        target_size=1.0,
    )

    return inactive, activated


def activation_to_color(activation: float) -> tuple[float, float, float]:
    """
    Blue-to-red activation color map.
    Low activation = blue, high activation = red.
    """
    a = float(np.clip(activation, 0.0, 1.0))

    if a < 0.5:
        t = a / 0.5
        return 0.12 + 0.65 * t, 0.34 + 0.45 * t, 0.95

    t = (a - 0.5) / 0.5
    return 0.95, 0.78 - 0.58 * t, 0.82 - 0.70 * t


def choose_informative_indices(
    activation: np.ndarray,
    shear: np.ndarray,
    max_count: int,
) -> np.ndarray:
    """
    Select an informative subset:
    - cells with strongest activation increase
    - cells with highest final activation
    - cells with highest final shear
    - some low activation examples
    """
    n_frames, n_platelets = activation.shape

    final_activation = activation[-1]
    final_shear = shear[-1]
    delta_activation = activation[-1] - activation[0]

    n_dynamic = max_count // 3
    n_high_activation = max_count // 3
    n_high_shear = max_count // 6
    n_low_activation = max_count - n_dynamic - n_high_activation - n_high_shear

    dynamic_idx = np.argsort(delta_activation)[::-1][:n_dynamic]
    high_activation_idx = np.argsort(final_activation)[::-1][:n_high_activation]
    high_shear_idx = np.argsort(final_shear)[::-1][:n_high_shear]
    low_activation_idx = np.argsort(final_activation)[:n_low_activation]

    selected = np.unique(
        np.concatenate(
            [
                dynamic_idx,
                high_activation_idx,
                high_shear_idx,
                low_activation_idx,
            ]
        )
    )

    if selected.size < max_count:
        remaining = np.setdiff1d(np.arange(n_platelets), selected)
        selected = np.concatenate([selected, remaining[: max_count - selected.size]])

    return selected[:max_count].astype(int)


def make_vessel_proxy(all_positions: np.ndarray) -> pv.PolyData:
    x_min = float(np.min(all_positions[..., 0]))
    x_max = float(np.max(all_positions[..., 0]))
    center_x = 0.5 * (x_min + x_max)
    length = max(x_max - x_min, 1.0)

    return pv.Cylinder(
        center=(center_x, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0),
        radius=1.05,
        height=length,
        resolution=128,
    )


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


def protrusion_directions() -> np.ndarray:
    directions = np.array(
        [
            [1.0, 0.30, 0.10],
            [0.65, -0.75, 0.18],
            [-0.45, 0.85, -0.18],
            [-0.75, -0.45, 0.38],
            [0.22, 0.42, 0.88],
            [-0.20, -0.35, -0.92],
            [0.92, -0.25, -0.25],
            [-0.92, 0.22, 0.20],
        ],
        dtype=float,
    )

    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return directions


def protrusion_count_for_activation(activation: float) -> int:
    if activation < 0.72:
        return 0
    if activation < 0.82:
        return 1
    if activation < 0.92:
        return 2
    return 3


def create_protrusion_tubes(
    origin: np.ndarray,
    activation: float,
    color: tuple[float, float, float],
    platelet_scale: float,
    platelet_index: int,
    frame_index: int,
) -> list[pv.PolyData]:
    """
    Create visual-only pseudo-filopodia tubes for highly activated platelets.

    These are not biomechanical structures. They are a visual overlay to make
    highly activated platelets easier to identify in dense scenes.
    """
    count = protrusion_count_for_activation(activation)

    if count == 0:
        return []

    directions = protrusion_directions()

    # Rotate direction selection over platelet index so every platelet does not look identical.
    offset = (platelet_index + frame_index) % len(directions)
    selected_dirs = [directions[(offset + i) % len(directions)] for i in range(count)]

    tubes = []

    base_radius = 0.40 * platelet_scale
    length = (0.16 + 0.18 * activation) * platelet_scale
    tube_radius = (0.010 + 0.005 * activation) * platelet_scale

    for direction in selected_dirs:
        start = origin + direction * base_radius
        end = origin + direction * (base_radius + length)

        line = pv.Line(start, end, resolution=5)
        tube = line.tube(radius=tube_radius, n_sides=8)
        tubes.append(tube)

    return tubes


def count_newly_activated(
    previous_activation: np.ndarray | None,
    current_activation: np.ndarray,
    selected_indices: np.ndarray,
) -> int:
    if previous_activation is None:
        return 0

    previous_selected = previous_activation[selected_indices]
    current_selected = current_activation[selected_indices]

    crossed = (previous_selected < ACTIVATION_THRESHOLD) & (
        current_selected >= ACTIVATION_THRESHOLD
    )

    return int(np.sum(crossed))


def main() -> None:
    print("Phase 5 Week 3 Day 4: Decimated mesh + activation deformation thesis video")

    positions, activation, shear = load_phase4_data()
    inactive_mesh, activated_mesh = load_decimated_meshes()

    n_frames, n_platelets, _ = positions.shape

    selected_indices = choose_informative_indices(
        activation=activation,
        shear=shear,
        max_count=MAX_RENDERED_PLATELETS,
    )

    vessel = make_vessel_proxy(positions)
    centerline = make_centerline(positions)

    print(f"positions shape:  {positions.shape}")
    print(f"activation shape: {activation.shape}")
    print(f"shear shape:      {shear.shape}")
    print(f"selected platelets: {len(selected_indices)} / {n_platelets}")
    print(f"inactive mesh cells:  {inactive_mesh.n_cells}")
    print(f"activated mesh cells: {activated_mesh.n_cells}")
    print(f"output video: {OUTPUT_VIDEO}")

    plotter = pv.Plotter(off_screen=True, window_size=(1700, 950))
    plotter.open_movie(str(OUTPUT_VIDEO), framerate=FPS)

    previous_activation = None

    for frame_idx in range(n_frames):
        frame_positions = positions[frame_idx]
        frame_activation = activation[frame_idx]
        frame_shear = shear[frame_idx]

        selected_activation = frame_activation[selected_indices]
        selected_shear = frame_shear[selected_indices]

        inactive_count = int(np.sum(selected_activation < ACTIVATION_THRESHOLD))
        activated_count = int(np.sum(selected_activation >= ACTIVATION_THRESHOLD))
        highly_activated_count = int(np.sum(selected_activation >= PROTRUSION_ACTIVATION_MIN))
        newly_activated = count_newly_activated(
            previous_activation=previous_activation,
            current_activation=frame_activation,
            selected_indices=selected_indices,
        )

        mean_activation = float(np.mean(selected_activation))
        mean_shear = float(np.mean(selected_shear))

        plotter.clear()
        plotter.set_background("white")

        # Soft vessel body
        plotter.add_mesh(
            vessel,
            color=(0.87, 0.90, 0.92),
            opacity=0.09,
            smooth_shading=True,
            show_edges=False,
        )

        # Centerline for flow direction
        plotter.add_mesh(
            centerline,
            color="gray",
            line_width=3,
            opacity=0.30,
        )

        protrusion_added = 0

        # Render platelet meshes
        for local_i, platelet_idx in enumerate(selected_indices):
            position = frame_positions[platelet_idx]
            act = float(frame_activation[platelet_idx])
            shr = float(frame_shear[platelet_idx])

            if act >= ACTIVATION_THRESHOLD:
                base_mesh = activated_mesh
                state_scale_bonus = 0.012
            else:
                base_mesh = inactive_mesh
                state_scale_bonus = 0.0

            # Slightly exaggerate visual state for readability.
            platelet_scale = 0.063 + 0.014 * act + 0.008 * shr + state_scale_bonus

            platelet = transform_platelet_mesh(
                base_mesh=base_mesh,
                position=position,
                scale=platelet_scale,
                rotation_x=(platelet_idx * 7 + frame_idx * 4) % 360,
                rotation_y=(platelet_idx * 13 + frame_idx * 6) % 360,
                rotation_z=(platelet_idx * 17 + frame_idx * 8) % 360,
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

            # Add limited pseudo-filopodia only for strongest activated platelets.
            if (
                ENABLE_PROTRUSIONS
                and act >= PROTRUSION_ACTIVATION_MIN
                and protrusion_added < MAX_PROTRUSION_PLATELETS_PER_FRAME
            ):
                tubes = create_protrusion_tubes(
                    origin=position,
                    activation=act,
                    color=color,
                    platelet_scale=platelet_scale,
                    platelet_index=int(platelet_idx),
                    frame_index=frame_idx,
                )

                for tube in tubes:
                    plotter.add_mesh(
                        tube,
                        color=color,
                        smooth_shading=True,
                        opacity=1.0,
                    )

                protrusion_added += 1

        # Title and frame statistics
        plotter.add_text(
            "Phase 5 Week 3: Optimized Mesh-Based Platelet Activation Visualization",
            position=(260, 905),
            font_size=18,
            color="black",
        )

        plotter.add_text(
            f"Frame {frame_idx + 1}/{n_frames} | rendered={len(selected_indices)} of {n_platelets} | "
            f"inactive={inactive_count} | activated={activated_count} | newly activated={newly_activated}",
            position=(190, 872),
            font_size=12,
            color="black",
        )

        plotter.add_text(
            f"Mean activation={mean_activation:.2f} | mean shear={mean_shear:.2f} | "
            f"highly activated={highly_activated_count} | visual protrusion overlays={protrusion_added}",
            position=(235, 844),
            font_size=12,
            color="black",
        )

        plotter.add_text(
            "Encoding: mesh shape = activation state | color = activation level | size = activation + shear | protrusions = high activation",
            position=(115, 816),
            font_size=11,
            color="dimgray",
        )

        plotter.add_text(
            "Blue = low activation",
            position=(70, 780),
            font_size=11,
            color=(0.15, 0.35, 0.95),
        )

        plotter.add_text(
            "Red = high activation",
            position=(70, 755),
            font_size=11,
            color=(0.95, 0.20, 0.12),
        )

        plotter.add_text(
            "Flow direction →",
            position=(1420, 785),
            font_size=12,
            color="dimgray",
        )

        plotter.add_text(
            "Note: protrusions are visual overlays, not biomechanical soft-body simulation.",
            position=(1080, 755),
            font_size=10,
            color="dimgray",
        )

        plotter.add_axes()

        plotter.camera_position = [
            (4.0, -6.2, 3.0),
            (4.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ]

        plotter.write_frame()

        print(
            f"Rendered frame {frame_idx + 1:02d}/{n_frames} | "
            f"inactive={inactive_count} | activated={activated_count} | "
            f"newly={newly_activated} | protrusion overlays={protrusion_added}"
        )

        previous_activation = frame_activation.copy()

    plotter.close()

    print(f"\nSaved optimized thesis video: {OUTPUT_VIDEO}")
    print("Week 3 Day 4 optimized thesis video complete.")


if __name__ == "__main__":
    main()