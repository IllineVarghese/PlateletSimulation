from pathlib import Path
import csv
import numpy as np
import pyvista as pv

from mesh_utils import load_mesh, center_and_scale_mesh, mesh_dimensions


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ACTIVATED_MESH_PATH = PROJECT_ROOT / "data" / "meshes" / "platelet" / "activated.obj"

OUTPUT_DIR = PROJECT_ROOT / "results" / "phase5" / "week3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_IMAGE = OUTPUT_DIR / "advanced_activation_deformation_progression.png"
OUTPUT_CSV = OUTPUT_DIR / "advanced_activation_deformation_metrics.csv"


def activation_to_color(activation: float) -> tuple[float, float, float]:
    """
    Blue-to-red activation color map.
    Low activation = blue, high activation = red.
    """
    a = float(np.clip(activation, 0.0, 1.0))

    if a < 0.5:
        t = a / 0.5
        return 0.10 + 0.65 * t, 0.32 + 0.45 * t, 0.95

    t = (a - 0.5) / 0.5
    return 0.95, 0.78 - 0.58 * t, 0.82 - 0.70 * t


def compute_roughness_index(mesh: pv.PolyData) -> float:
    """
    Geometric roughness proxy:
    coefficient of variation of vertex distance from mesh center.

    This is not a biological measurement.
    It is a reproducible visual morphology descriptor.
    """
    points = mesh.points
    center = points.mean(axis=0)
    distances = np.linalg.norm(points - center, axis=1)

    mean_distance = float(np.mean(distances))
    std_distance = float(np.std(distances))

    if mean_distance <= 1e-8:
        return 0.0

    return std_distance / mean_distance


def mesh_metrics(
    name: str,
    activation: float,
    mesh: pv.PolyData,
    protrusion_count: int,
) -> dict:
    size_x, size_y, size_z = mesh_dimensions(mesh)

    return {
        "name": name,
        "activation": activation,
        "points": mesh.n_points,
        "cells": mesh.n_cells,
        "size_x": size_x,
        "size_y": size_y,
        "size_z": size_z,
        "max_size": max(size_x, size_y, size_z),
        "surface_area": float(mesh.area),
        "roughness_index": compute_roughness_index(mesh),
        "visual_protrusions": protrusion_count,
    }


def deform_with_activation_features(
    mesh: pv.PolyData,
    activation: float,
    elongation_strength: float = 0.15,
    flattening_strength: float = 0.08,
    roughness_strength: float = 0.035,
) -> pv.PolyData:
    """
    Visual-only activated platelet deformation.

    The deformation is intentionally moderate:
    - elongation
    - mild anisotropic flattening
    - deterministic surface roughening

    Strong activation is additionally represented by pseudo-filopodia tubes
    added during rendering.
    """
    activation = float(np.clip(activation, 0.0, 1.0))

    deformed = mesh.copy()
    points = deformed.points.copy()

    center = points.mean(axis=0)
    p = points - center

    x = p[:, 0]
    y = p[:, 1]
    z = p[:, 2]

    radius = np.linalg.norm(p, axis=1)
    radius_max = max(float(np.max(radius)), 1e-8)

    radial_dir = np.divide(
        p,
        radius[:, None],
        out=np.zeros_like(p),
        where=radius[:, None] > 1e-8,
    )

    # Controlled elongation
    p[:, 0] *= 1.0 + elongation_strength * activation

    # Mild activation-associated anisotropy
    p[:, 1] *= 1.0 + 0.06 * activation
    p[:, 2] *= 1.0 - flattening_strength * activation

    # Reproducible roughness pattern
    roughness_pattern = (
        np.sin(9.0 * x)
        * np.cos(11.0 * y)
        * np.sin(13.0 * z)
    )

    surface_weight = radius / radius_max

    p += (
        roughness_strength
        * activation
        * surface_weight[:, None]
        * roughness_pattern[:, None]
        * radial_dir
    )

    deformed.points = p + center
    return deformed


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
    """
    Number of visual pseudo-filopodia shown for each activation level.
    """
    if activation < 0.25:
        return 0
    if activation < 0.50:
        return 1
    if activation < 0.75:
        return 3
    if activation < 1.00:
        return 5
    return 7


def create_visual_protrusions(
    origin: tuple[float, float, float],
    activation: float,
    color: tuple[float, float, float],
    scale: float,
) -> list[tuple[pv.PolyData, tuple[float, float, float]]]:
    """
    Create visual pseudo-filopodia as thin tubes.

    These are not simulated physical structures.
    They are a visualization overlay to make activation morphology readable.
    """
    count = protrusion_count_for_activation(activation)
    dirs = protrusion_directions()[:count]

    tubes = []

    if count == 0:
        return tubes

    origin_arr = np.array(origin, dtype=float)

    base_radius = 0.34 * scale
    length = (0.18 + 0.22 * activation) * scale
    tube_radius = (0.012 + 0.006 * activation) * scale

    for direction in dirs:
        start = origin_arr + direction * base_radius
        end = origin_arr + direction * (base_radius + length)

        line = pv.Line(start, end, resolution=8)
        tube = line.tube(radius=tube_radius, n_sides=12)

        tubes.append((tube, color))

    return tubes


def make_instance(
    mesh: pv.PolyData,
    position: tuple[float, float, float],
    scale: float = 0.56,
) -> pv.PolyData:
    instance = mesh.copy()
    instance.scale(scale, inplace=True)
    instance.translate(position, inplace=True)
    return instance


def save_metrics_csv(rows: list[dict]) -> None:
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "name",
                "activation",
                "points",
                "cells",
                "size_x",
                "size_y",
                "size_z",
                "max_size",
                "surface_area",
                "roughness_index",
                "visual_protrusions",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    print("Phase 5 Week 3 Day 1: Advanced activation deformation progression")

    activated_base = center_and_scale_mesh(
        load_mesh(ACTIVATED_MESH_PATH),
        target_size=1.0,
    )

    activation_levels = [0.00, 0.25, 0.50, 0.75, 1.00]
    x_positions = [-1.9, -0.95, 0.0, 0.95, 1.9]
    visual_scale = 0.56

    meshes = []
    all_protrusions = []
    metrics = []

    for activation, x_pos in zip(activation_levels, x_positions):
        deformed = deform_with_activation_features(
            activated_base,
            activation=activation,
        )

        protrusion_count = protrusion_count_for_activation(activation)

        metrics.append(
            mesh_metrics(
                name=f"activation_{activation:.2f}",
                activation=activation,
                mesh=deformed,
                protrusion_count=protrusion_count,
            )
        )

        position = (x_pos, 0.0, 0.0)
        instance = make_instance(
            deformed,
            position=position,
            scale=visual_scale,
        )

        color = activation_to_color(activation)
        protrusions = create_visual_protrusions(
            origin=position,
            activation=activation,
            color=color,
            scale=visual_scale,
        )

        meshes.append((activation, instance, color))
        all_protrusions.extend(protrusions)

    save_metrics_csv(metrics)

    plotter = pv.Plotter(off_screen=True, window_size=(1900, 1050))
    plotter.set_background("white")

    for activation, mesh, color in meshes:
        plotter.add_mesh(
            mesh,
            color=color,
            smooth_shading=True,
            show_edges=False,
            opacity=1.0,
            specular=0.28,
        )

    for tube, color in all_protrusions:
        plotter.add_mesh(
            tube,
            color=color,
            smooth_shading=True,
            opacity=1.0,
        )

    # Clean readable labels below each object
    for activation, x_pos in zip(activation_levels, x_positions):
        count = protrusion_count_for_activation(activation)
        plotter.add_point_labels(
            [(x_pos, -0.72, -0.10)],
            [f"A = {activation:.2f}\nvisual protrusions = {count}"],
            font_size=13,
            text_color="black",
            point_color="white",
            point_size=1,
            shape=None,
            always_visible=True,
        )

    # Main title
    plotter.add_text(
        "Phase 5 Week 3: Activation-Dependent Platelet Deformation Model",
        position=(370, 985),
        font_size=19,
        color="black",
    )

    plotter.add_text(
        "Continuous visual morphology layer: elongation, roughening, and pseudo-filopodia",
        position=(410, 950),
        font_size=13,
        color="dimgray",
    )

    plotter.add_text(
        "Purpose: improve readability of inactive-to-activated state transitions in dense platelet simulations",
        position=(370, 922),
        font_size=13,
        color="dimgray",
    )

    # Metrics summary
    area_0 = metrics[0]["surface_area"]
    area_1 = metrics[-1]["surface_area"]
    roughness_0 = metrics[0]["roughness_index"]
    roughness_1 = metrics[-1]["roughness_index"]
    max_size_0 = metrics[0]["max_size"]
    max_size_1 = metrics[-1]["max_size"]

    plotter.add_text(
        f"Geometry metrics: surface area {area_0:.2f} to {area_1:.2f}, "
        f"roughness {roughness_0:.3f} to {roughness_1:.3f}, "
        f"max size {max_size_0:.2f} to {max_size_1:.2f}",
        position=(305, 890),
        font_size=12,
        color="black",
    )

    plotter.add_text(
        "Important limitation: this is a reproducible visual deformation, not a biomechanical soft-body model.",
        position=(375, 862),
        font_size=12,
        color="dimgray",
    )

    # Visual legend
    plotter.add_text(
        "Visual encoding: blue = low activation, red = high activation, protrusions = stronger activated morphology",
        position=(275, 832),
        font_size=12,
        color="dimgray",
    )

    plotter.add_axes()

    # Wider, safer camera so edge objects are not clipped
    plotter.camera_position = [
        (0.0, -5.4, 2.2),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ]

    plotter.screenshot(str(OUTPUT_IMAGE))
    plotter.close()

    print(f"Saved improved deformation figure: {OUTPUT_IMAGE}")
    print(f"Saved deformation metrics CSV:     {OUTPUT_CSV}")
    print("\nMetrics:")
    for row in metrics:
        print(
            f"activation={row['activation']:.2f} | "
            f"area={row['surface_area']:.3f} | "
            f"roughness={row['roughness_index']:.4f} | "
            f"max_size={row['max_size']:.3f} | "
            f"visual protrusions={row['visual_protrusions']}"
        )

    print("\nImproved deformation progression complete.")


if __name__ == "__main__":
    main()