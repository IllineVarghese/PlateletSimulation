from pathlib import Path
import time
import csv

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from mesh_utils import load_mesh, center_and_scale_mesh


PROJECT_ROOT = Path(__file__).resolve().parents[2]

POSITIONS_PATH = PROJECT_ROOT / "results" / "phase4" / "final_demo" / "positions.npy"
ACTIVATION_PATH = PROJECT_ROOT / "results" / "phase4" / "final_demo" / "activation.npy"
SHEAR_PATH = PROJECT_ROOT / "results" / "phase4" / "final_demo" / "shear_input.npy"

INACTIVE_MESH_PATH = PROJECT_ROOT / "data" / "meshes" / "platelet" / "inactive.obj"
ACTIVATED_MESH_PATH = PROJECT_ROOT / "data" / "meshes" / "platelet" / "activated.obj"

OUTPUT_DIR = PROJECT_ROOT / "results" / "phase5" / "week2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "mesh_render_performance_week2.csv"
PLOT_PATH = OUTPUT_DIR / "mesh_render_performance_week2.png"

ACTIVATION_THRESHOLD = 0.50
FRAME_INDEX = -1

# Keep this moderate for today. We can test 500 or 1000 later after optimization.
RENDER_COUNTS = [25, 50, 100, 150, 200]


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
    Blue-to-red activation color map without extra dependencies.
    """
    a = float(np.clip(act, 0.0, 1.0))

    if a < 0.5:
        t = a / 0.5
        return 0.15 + 0.70 * t, 0.35 + 0.45 * t, 0.95

    t = (a - 0.5) / 0.5
    return 0.95, 0.80 - 0.55 * t, 0.85 - 0.70 * t


def choose_indices(frame_activation: np.ndarray, frame_shear: np.ndarray, count: int) -> np.ndarray:
    """
    Select a useful mixture of high-activation, high-shear, and remaining platelets.
    """
    n = frame_activation.size
    high_activation = np.argsort(frame_activation)[::-1][: count // 2]
    high_shear = np.argsort(frame_shear)[::-1][: count // 4]

    selected = np.unique(np.concatenate([high_activation, high_shear]))

    if selected.size < count:
        remaining = np.setdiff1d(np.arange(n), selected)
        selected = np.concatenate([selected, remaining[: count - selected.size]])

    return selected[:count]


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
        resolution=96,
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


def render_one_snapshot(
    count: int,
    frame_positions: np.ndarray,
    frame_activation: np.ndarray,
    frame_shear: np.ndarray,
    inactive_base: pv.PolyData,
    activated_base: pv.PolyData,
    vessel: pv.PolyData,
) -> dict:
    selected_indices = choose_indices(frame_activation, frame_shear, count)

    inactive_count = 0
    activated_count = 0

    start_time = time.perf_counter()

    plotter = pv.Plotter(off_screen=True, window_size=(1200, 700))
    plotter.set_background("white")

    plotter.add_mesh(
        vessel,
        color=(0.86, 0.90, 0.92),
        opacity=0.08,
        smooth_shading=True,
        show_edges=False,
    )

    for idx in selected_indices:
        pos = frame_positions[idx]
        act = float(frame_activation[idx])
        shr = float(frame_shear[idx])

        if act >= ACTIVATION_THRESHOLD:
            base_mesh = activated_base
            activated_count += 1
            state_bonus = 0.012
        else:
            base_mesh = inactive_base
            inactive_count += 1
            state_bonus = 0.0

        scale = 0.065 + 0.015 * act + 0.008 * shr + state_bonus

        platelet = transform_platelet_mesh(
            base_mesh=base_mesh,
            position=pos,
            scale=scale,
            rotation_x=(idx * 7.0) % 360,
            rotation_y=(idx * 13.0) % 360,
            rotation_z=(idx * 17.0) % 360,
        )

        plotter.add_mesh(
            platelet,
            color=activation_to_color(act),
            smooth_shading=True,
            show_edges=False,
            opacity=0.97,
        )

    plotter.add_text(
        f"Phase 5 render benchmark | {count} mesh platelets",
        position=(270, 655),
        font_size=14,
        color="black",
    )

    plotter.add_text(
        f"inactive={inactive_count}, activated={activated_count}, threshold={ACTIVATION_THRESHOLD:.2f}",
        position=(330, 625),
        font_size=10,
        color="black",
    )

    plotter.camera_position = [
        (4.0, -6.2, 3.0),
        (4.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ]

    # Screenshot is required to measure realistic render output time.
    tmp_path = OUTPUT_DIR / f"_benchmark_{count}.png"
    plotter.screenshot(str(tmp_path))
    plotter.close()

    elapsed = time.perf_counter() - start_time
    fps_estimate = 1.0 / elapsed if elapsed > 0 else 0.0

    return {
        "rendered_platelets": count,
        "inactive": inactive_count,
        "activated": activated_count,
        "render_time_seconds": elapsed,
        "estimated_fps_if_static_frames": fps_estimate,
    }


def save_csv(rows: list[dict]) -> None:
    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "rendered_platelets",
                "inactive",
                "activated",
                "render_time_seconds",
                "estimated_fps_if_static_frames",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows: list[dict]) -> None:
    counts = [row["rendered_platelets"] for row in rows]
    times = [row["render_time_seconds"] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(counts, times, marker="o")
    plt.xlabel("Rendered platelet mesh count")
    plt.ylabel("Render time per snapshot (seconds)")
    plt.title("Phase 5 Week 2 mesh rendering performance")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    plt.close()


def main() -> None:
    print("Phase 5 Week 2 Day 5: Mesh rendering performance benchmark")

    positions, activation, shear = load_phase4_data()

    frame_positions = positions[FRAME_INDEX]
    frame_activation = activation[FRAME_INDEX]
    frame_shear = shear[FRAME_INDEX]

    inactive_base = center_and_scale_mesh(load_mesh(INACTIVE_MESH_PATH), target_size=1.0)
    activated_base = center_and_scale_mesh(load_mesh(ACTIVATED_MESH_PATH), target_size=1.0)
    vessel = make_vessel_proxy(positions)

    rows = []

    print(f"Using frame: {positions.shape[0] - 1 if FRAME_INDEX == -1 else FRAME_INDEX}")
    print(f"Available platelets: {positions.shape[1]}")

    for count in RENDER_COUNTS:
        print(f"\nBenchmarking {count} rendered mesh platelets...")
        row = render_one_snapshot(
            count=count,
            frame_positions=frame_positions,
            frame_activation=frame_activation,
            frame_shear=frame_shear,
            inactive_base=inactive_base,
            activated_base=activated_base,
            vessel=vessel,
        )
        rows.append(row)

        print(
            f"count={row['rendered_platelets']} | "
            f"time={row['render_time_seconds']:.3f}s | "
            f"estimated fps={row['estimated_fps_if_static_frames']:.2f} | "
            f"inactive={row['inactive']} | activated={row['activated']}"
        )

    save_csv(rows)
    save_plot(rows)

    print("\nSaved benchmark CSV:")
    print(CSV_PATH)
    print("Saved benchmark plot:")
    print(PLOT_PATH)
    print("Week 2 Day 5 performance benchmark complete.")


if __name__ == "__main__":
    main()