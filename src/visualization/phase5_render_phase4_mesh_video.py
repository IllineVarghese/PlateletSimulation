from pathlib import Path
import numpy as np
import pyvista as pv
from matplotlib import colormaps

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

OUTPUT_VIDEO = OUTPUT_DIR / "phase4_mesh_switching_video_thesis.mp4"

ACTIVATION_THRESHOLD = 0.50
MAX_RENDERED_PLATELETS = 100
FPS = 6

ACTIVATION_CMAP = colormaps["coolwarm"]  # blue -> white -> red


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


def activation_to_rgb(act: float) -> tuple[float, float, float]:
    """
    Map activation [0,1] to RGB color using a blue-to-red colormap.
    """
    act_clipped = float(np.clip(act, 0.0, 1.0))
    rgba = ACTIVATION_CMAP(act_clipped)
    return rgba[0], rgba[1], rgba[2]


def choose_informative_indices(
    activation: np.ndarray,
    max_count: int,
) -> np.ndarray:
    """
    Select platelets that are most informative:
    - many with strongest activation increase over time
    - plus a balanced final-frame inactive/activated subset
    """
    n_frames, n_platelets = activation.shape
    final_act = activation[-1]
    delta_act = activation[-1] - activation[0]

    # Top activation increases (dynamic platelets)
    dynamic_count = max_count // 2
    dynamic_idx = np.argsort(delta_act)[::-1][:dynamic_count]

    selected = list(dynamic_idx)

    remaining = np.setdiff1d(np.arange(n_platelets), np.array(selected, dtype=int))

    inactive_pool = remaining[final_act[remaining] < ACTIVATION_THRESHOLD]
    activated_pool = remaining[final_act[remaining] >= ACTIVATION_THRESHOLD]

    remaining_slots = max_count - len(selected)
    half = remaining_slots // 2

    selected.extend(inactive_pool[:half].tolist())
    selected.extend(activated_pool[: remaining_slots - half].tolist())

    selected = np.array(selected[:max_count], dtype=int)
    return selected


def make_vessel_proxy(all_positions: np.ndarray) -> pv.PolyData:
    """
    Create a simple cylindrical vessel around the whole trajectory range.
    """
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


def count_newly_activated(
    previous_activation: np.ndarray | None,
    current_activation: np.ndarray,
    selected_indices: np.ndarray,
) -> int:
    """
    Count how many selected platelets crossed the activation threshold
    between the previous and current frame.
    """
    if previous_activation is None:
        return 0

    prev = previous_activation[selected_indices]
    curr = current_activation[selected_indices]

    crossed = (prev < ACTIVATION_THRESHOLD) & (curr >= ACTIVATION_THRESHOLD)
    return int(np.sum(crossed))


def main() -> None:
    print("Phase 5 Week 2 Day 3: Thesis-style real Phase 4 mesh-switching video")

    positions, activation, shear = load_phase4_data()
    n_frames, n_platelets, _ = positions.shape

    print(f"positions shape:  {positions.shape}")
    print(f"activation shape: {activation.shape}")
    print(f"shear shape:      {shear.shape}")

    selected_indices = choose_informative_indices(
        activation=activation,
        max_count=MAX_RENDERED_PLATELETS,
    )

    print(f"Selected {len(selected_indices)} informative platelets for rendering.")

    inactive_base = center_and_scale_mesh(
        load_mesh(INACTIVE_MESH_PATH),
        target_size=1.0,
    )
    activated_base = center_and_scale_mesh(
        load_mesh(ACTIVATED_MESH_PATH),
        target_size=1.0,
    )

    vessel = make_vessel_proxy(positions)

    plotter = pv.Plotter(off_screen=True, window_size=(1700, 950))
    plotter.open_movie(str(OUTPUT_VIDEO), framerate=FPS)

    previous_frame_activation = None

    for frame_idx in range(n_frames):
        frame_positions = positions[frame_idx]
        frame_activation = activation[frame_idx]
        frame_shear = shear[frame_idx]

        selected_positions = frame_positions[selected_indices]
        selected_activation = frame_activation[selected_indices]
        selected_shear = frame_shear[selected_indices]

        inactive_count = int(np.sum(selected_activation < ACTIVATION_THRESHOLD))
        activated_count = int(np.sum(selected_activation >= ACTIVATION_THRESHOLD))
        newly_activated_count = count_newly_activated(
            previous_activation=previous_frame_activation,
            current_activation=frame_activation,
            selected_indices=selected_indices,
        )

        mean_activation = float(np.mean(selected_activation))
        mean_shear = float(np.mean(selected_shear))

        plotter.clear()
        plotter.set_background("white")

        # Vessel: smoother and less distracting
        plotter.add_mesh(
            vessel,
            color=(0.88, 0.90, 0.92),
            opacity=0.08,
            smooth_shading=True,
            show_edges=False,
        )

        # Optional centerline to show flow direction
        centerline = pv.Line(
            pointa=(float(np.min(positions[..., 0])), 0.0, 0.0),
            pointb=(float(np.max(positions[..., 0])), 0.0, 0.0),
            resolution=1,
        )
        plotter.add_mesh(
            centerline,
            color="lightgray",
            line_width=2,
            opacity=0.40,
        )

        for platelet_idx in selected_indices:
            pos = frame_positions[platelet_idx]
            act = float(frame_activation[platelet_idx])
            shr = float(frame_shear[platelet_idx])

            if act >= ACTIVATION_THRESHOLD:
                base_mesh = activated_base
                extra_scale = 0.012
            else:
                base_mesh = inactive_base
                extra_scale = 0.0

            # Make changes more visually visible
            scale = 0.060 + 0.015 * act + 0.010 * shr + extra_scale

            platelet = transform_platelet_mesh(
                base_mesh=base_mesh,
                position=pos,
                scale=scale,
                rotation_x=(platelet_idx * 7 + frame_idx * 4) % 360,
                rotation_y=(platelet_idx * 13 + frame_idx * 6) % 360,
                rotation_z=(platelet_idx * 17 + frame_idx * 8) % 360,
            )

            color_rgb = activation_to_rgb(act)

            plotter.add_mesh(
                platelet,
                color=color_rgb,
                smooth_shading=True,
                show_edges=False,
                opacity=0.95 if act < ACTIVATION_THRESHOLD else 1.0,
                specular=0.25,
            )

        # Title and stats
        plotter.add_text(
            "Phase 5 Week 2: Real Phase 4 platelet mesh-switching video",
            position=(290, 900),
            font_size=18,
            color="black",
        )

        plotter.add_text(
            f"Frame {frame_idx + 1}/{n_frames} | rendered={len(selected_indices)} of {n_platelets} | "
            f"inactive={inactive_count} | activated={activated_count} | newly activated={newly_activated_count}",
            position=(180, 868),
            font_size=12,
            color="black",
        )

        plotter.add_text(
            f"Selected mean activation={mean_activation:.2f} | selected mean shear={mean_shear:.2f} | "
            f"activation threshold={ACTIVATION_THRESHOLD:.2f}",
            position=(240, 842),
            font_size=12,
            color="black",
        )

        plotter.add_text(
            "Visual encoding: mesh shape = inactive/activated state | color = activation (blue low → red high) | size = activation + shear",
            position=(120, 816),
            font_size=11,
            color="dimgray",
        )

        plotter.add_text(
            "Flow direction →",
            position=(1420, 785),
            font_size=11,
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
            f"Rendered frame {frame_idx + 1}/{n_frames} | "
            f"inactive={inactive_count} | activated={activated_count} | "
            f"newly activated={newly_activated_count}"
        )

        previous_frame_activation = frame_activation.copy()

    plotter.close()

    print(f"\nSaved thesis-style video: {OUTPUT_VIDEO}")
    print("Week 2 Day 3 thesis-style video rendering complete.")


if __name__ == "__main__":
    main()