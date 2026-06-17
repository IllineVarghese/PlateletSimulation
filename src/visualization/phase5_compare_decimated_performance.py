from pathlib import Path
import csv
import time

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from mesh_utils import load_mesh, center_and_scale_mesh


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Real Phase 4 data
POSITIONS_PATH = PROJECT_ROOT / "results" / "phase4" / "final_demo" / "positions.npy"
ACTIVATION_PATH = PROJECT_ROOT / "results" / "phase4" / "final_demo" / "activation.npy"
SHEAR_PATH = PROJECT_ROOT / "results" / "phase4" / "final_demo" / "shear_input.npy"

# Original meshes
INACTIVE_ORIGINAL_PATH = PROJECT_ROOT / "data" / "meshes" / "platelet" / "inactive.obj"
ACTIVATED_ORIGINAL_PATH = PROJECT_ROOT / "data" / "meshes" / "platelet" / "activated.obj"

# Decimated meshes created in Week 3 Day 2
INACTIVE_DECIMATED_PATH = (
    PROJECT_ROOT / "results" / "phase5" / "week3" / "optimized_meshes" / "inactive_decimated.vtp"
)
ACTIVATED_DECIMATED_PATH = (
    PROJECT_ROOT / "results" / "phase5" / "week3" / "optimized_meshes" / "activated_decimated.vtp"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "phase5" / "week3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "original_vs_decimated_mesh_performance.csv"
PLOT_PATH = OUTPUT_DIR / "original_vs_decimated_mesh_performance.png"

ACTIVATION_THRESHOLD = 0.50
FRAME_INDEX = -1
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

    if positions.shape[:2] != activation.shape:
        raise ValueError("positions and activation shapes do not match")

    if positions.shape[:2] != shear.shape:
        raise ValueError("positions and shear shapes do not match")

    return positions, activation, shear


def activation_to_color(act: float) -> tuple[float, float, float]:
    """
    Blue-to-red activation color map.
    """
    a = float(np.clip(act, 0.0, 1.0))

    if a < 0.5:
        t = a / 0.5
        return 0.15 + 0.70 * t, 0.35 + 0.45 * t, 0.95

    t = (a - 0.5) / 0.5
    return 0.95, 0.80 - 0.55 * t, 0.85 - 0.70 * t


def choose_indices(frame_activation: np.ndarray, frame_shear: np.ndarray, count: int) -> np.ndarray:
    """
    Select the same informative platelet subset for both original and decimated tests.
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


def load_original_meshes() -> tuple[pv.PolyData, pv.PolyData]:
    inactive = center_and_scale_mesh(
        load_mesh(INACTIVE_ORIGINAL_PATH),
        target_size=1.0,
    )

    activated = center_and_scale_mesh(
        load_mesh(ACTIVATED_ORIGINAL_PATH),
        target_size=1.0,
    )

    return inactive, activated


def load_decimated_meshes() -> tuple[pv.PolyData, pv.PolyData]:
    if not INACTIVE_DECIMATED_PATH.exists():
        raise FileNotFoundError(
            f"Missing decimated inactive mesh: {INACTIVE_DECIMATED_PATH}\n"
            "Run Week 3 Day 2 script first: phase5_mesh_decimation_test.py"
        )

    if not ACTIVATED_DECIMATED_PATH.exists():
        raise FileNotFoundError(
            f"Missing decimated activated mesh: {ACTIVATED_DECIMATED_PATH}\n"
            "Run Week 3 Day 2 script first: phase5_mesh_decimation_test.py"
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


def render_snapshot(
    mesh_mode: str,
    count: int,
    frame_positions: np.ndarray,
    frame_activation: np.ndarray,
    frame_shear: np.ndarray,
    inactive_base: pv.PolyData,
    activated_base: pv.PolyData,
    vessel: pv.PolyData,
    save_example: bool,
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
        f"Phase 5 Week 3 performance comparison | {mesh_mode} meshes | {count} platelets",
        position=(180, 655),
        font_size=12,
        color="black",
    )

    plotter.add_text(
        f"inactive={inactive_count}, activated={activated_count}, threshold={ACTIVATION_THRESHOLD:.2f}",
        position=(320, 625),
        font_size=10,
        color="black",
    )

    plotter.camera_position = [
        (4.0, -6.2, 3.0),
        (4.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ]

    if save_example:
        output_path = OUTPUT_DIR / f"_performance_{mesh_mode}_{count}.png"
    else:
        output_path = OUTPUT_DIR / f"_tmp_{mesh_mode}_{count}.png"

    plotter.screenshot(str(output_path))
    plotter.close()

    elapsed = time.perf_counter() - start_time
    fps_estimate = 1.0 / elapsed if elapsed > 0 else 0.0

    return {
        "mesh_mode": mesh_mode,
        "rendered_platelets": count,
        "inactive": inactive_count,
        "activated": activated_count,
        "inactive_mesh_points": inactive_base.n_points,
        "inactive_mesh_cells": inactive_base.n_cells,
        "activated_mesh_points": activated_base.n_points,
        "activated_mesh_cells": activated_base.n_cells,
        "render_time_seconds": elapsed,
        "estimated_fps_if_static_frames": fps_estimate,
    }


def save_csv(rows: list[dict]) -> None:
    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "mesh_mode",
                "rendered_platelets",
                "inactive",
                "activated",
                "inactive_mesh_points",
                "inactive_mesh_cells",
                "activated_mesh_points",
                "activated_mesh_cells",
                "render_time_seconds",
                "estimated_fps_if_static_frames",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows: list[dict]) -> None:
    original_rows = [row for row in rows if row["mesh_mode"] == "original"]
    decimated_rows = [row for row in rows if row["mesh_mode"] == "decimated"]

    counts_original = [row["rendered_platelets"] for row in original_rows]
    times_original = [row["render_time_seconds"] for row in original_rows]

    counts_decimated = [row["rendered_platelets"] for row in decimated_rows]
    times_decimated = [row["render_time_seconds"] for row in decimated_rows]

    plt.figure(figsize=(8, 5))
    plt.plot(counts_original, times_original, marker="o", label="Original meshes")
    plt.plot(counts_decimated, times_decimated, marker="o", label="Decimated meshes")
    plt.xlabel("Rendered platelet mesh count")
    plt.ylabel("Render time per snapshot (seconds)")
    plt.title("Phase 5 Week 3: Original vs decimated mesh rendering time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    plt.close()


def print_mesh_summary(
    mode: str,
    inactive: pv.PolyData,
    activated: pv.PolyData,
) -> None:
    print(f"\n{mode.capitalize()} mesh complexity")
    print("-" * (len(mode) + 16))
    print(f"Inactive:  points={inactive.n_points}, cells={inactive.n_cells}")
    print(f"Activated: points={activated.n_points}, cells={activated.n_cells}")


def main() -> None:
    print("Phase 5 Week 3 Day 3: Original vs decimated mesh performance comparison")

    positions, activation, shear = load_phase4_data()

    frame_positions = positions[FRAME_INDEX]
    frame_activation = activation[FRAME_INDEX]
    frame_shear = shear[FRAME_INDEX]

    vessel = make_vessel_proxy(positions)

    original_inactive, original_activated = load_original_meshes()
    decimated_inactive, decimated_activated = load_decimated_meshes()

    print_mesh_summary("original", original_inactive, original_activated)
    print_mesh_summary("decimated", decimated_inactive, decimated_activated)

    rows = []

    mesh_sets = [
        ("original", original_inactive, original_activated),
        ("decimated", decimated_inactive, decimated_activated),
    ]

    for mesh_mode, inactive_mesh, activated_mesh in mesh_sets:
        print(f"\nBenchmarking {mesh_mode} meshes")
        print("-" * (14 + len(mesh_mode)))

        for count in RENDER_COUNTS:
            save_example = count == 100

            row = render_snapshot(
                mesh_mode=mesh_mode,
                count=count,
                frame_positions=frame_positions,
                frame_activation=frame_activation,
                frame_shear=frame_shear,
                inactive_base=inactive_mesh,
                activated_base=activated_mesh,
                vessel=vessel,
                save_example=save_example,
            )

            rows.append(row)

            print(
                f"{mesh_mode:9s} | count={count:3d} | "
                f"time={row['render_time_seconds']:.3f}s | "
                f"fps={row['estimated_fps_if_static_frames']:.2f} | "
                f"inactive={row['inactive']} | activated={row['activated']}"
            )

    save_csv(rows)
    save_plot(rows)

    print("\nSaved comparison CSV:")
    print(CSV_PATH)

    print("Saved comparison plot:")
    print(PLOT_PATH)

    print("\nSaved example images:")
    print(OUTPUT_DIR / "_performance_original_100.png")
    print(OUTPUT_DIR / "_performance_decimated_100.png")

    print("\nWeek 3 Day 3 original vs decimated performance comparison complete.")


if __name__ == "__main__":
    main()