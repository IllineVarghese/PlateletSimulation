from pathlib import Path
import csv
import time

import pyvista as pv
import vtk

from mesh_utils import load_mesh, center_and_scale_mesh, mesh_dimensions


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INACTIVE_MESH_PATH = PROJECT_ROOT / "data" / "meshes" / "platelet" / "inactive.obj"
ACTIVATED_MESH_PATH = PROJECT_ROOT / "data" / "meshes" / "platelet" / "activated.obj"

OUTPUT_DIR = PROJECT_ROOT / "results" / "phase5" / "week3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OPTIMIZED_DIR = OUTPUT_DIR / "optimized_meshes"
OPTIMIZED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_IMAGE = OUTPUT_DIR / "mesh_decimation_comparison.png"
OUTPUT_CSV = OUTPUT_DIR / "mesh_decimation_report.csv"

INACTIVE_DECIMATED_PATH = OPTIMIZED_DIR / "inactive_decimated.vtp"
ACTIVATED_DECIMATED_PATH = OPTIMIZED_DIR / "activated_decimated.vtp"

# Aggressive simplification target.
# In the previous test, normal decimation did not reduce inactive.obj.
INACTIVE_TARGET_REDUCTION = 0.97
ACTIVATED_TARGET_REDUCTION = 0.85


def quadric_cluster_decimate(
    mesh: pv.PolyData,
    mesh_name: str,
) -> pv.PolyData:
    """
    Fallback simplification using vtkQuadricClustering.

    This is used when decimate_pro cannot reduce a difficult mesh.
    Lower division numbers produce stronger simplification.
    """
    mesh = mesh.copy().triangulate().clean()
    original_cells = mesh.n_cells

    if original_cells == 0:
        raise ValueError(f"{mesh_name}: mesh has zero cells before quadric clustering.")

    division_candidates = [25, 35, 45, 60, 80, 100]

    best_mesh = mesh
    best_reduction = 0.0

    print(f"\nTrying quadric clustering fallback for {mesh_name}...")

    for divisions in division_candidates:
        try:
            cluster = vtk.vtkQuadricClustering()
            cluster.SetInputData(mesh)
            cluster.SetNumberOfXDivisions(divisions)
            cluster.SetNumberOfYDivisions(divisions)
            cluster.SetNumberOfZDivisions(divisions)
            cluster.Update()

            candidate = pv.wrap(cluster.GetOutput()).triangulate().clean()

            if candidate.n_points == 0 or candidate.n_cells == 0:
                print(f"quadric clustering divisions={divisions} produced empty mesh. Skipping.")
                continue

            reduction_percent = 100.0 * (1.0 - candidate.n_cells / original_cells)

            print(
                f"quadric clustering divisions={divisions}: "
                f"points={candidate.n_points}, cells={candidate.n_cells}, "
                f"reduction={reduction_percent:.1f}%"
            )

            if reduction_percent > best_reduction:
                best_mesh = candidate
                best_reduction = reduction_percent

            # Accept strong but not overly destructive reduction.
            if 60.0 <= reduction_percent <= 95.0:
                print(
                    f"Accepted quadric clustering for {mesh_name}: "
                    f"reduction={reduction_percent:.1f}%"
                )
                return candidate

        except Exception as error:
            print(f"quadric clustering divisions={divisions} failed for {mesh_name}: {error}")

    if best_mesh.n_cells == original_cells:
        print(f"Warning: quadric clustering could not reduce {mesh_name}.")
    else:
        print(
            f"Using best quadric clustering result for {mesh_name}: "
            f"cells={best_mesh.n_cells}, reduction={best_reduction:.1f}%"
        )

    return best_mesh


def safe_decimate(
    mesh: pv.PolyData,
    target_reduction: float,
    mesh_name: str,
) -> pv.PolyData:
    """
    Decimate mesh triangles while keeping a valid PolyData object.

    Strategy:
    1. Try normal PyVista/VTK decimation.
    2. If reduction is too weak, use vtkQuadricClustering fallback.
    """
    mesh = mesh.copy().triangulate().clean()
    original_cells = mesh.n_cells

    if original_cells == 0:
        raise ValueError(f"{mesh_name}: original mesh has zero cells.")

    attempts = [
        {
            "name": "aggressive_decimate_pro",
            "method": "decimate_pro",
            "reduction": target_reduction,
            "preserve_topology": False,
            "splitting": True,
            "boundary_vertex_deletion": True,
            "feature_angle": 60.0,
        },
        {
            "name": "medium_decimate_pro",
            "method": "decimate_pro",
            "reduction": min(target_reduction, 0.90),
            "preserve_topology": False,
            "splitting": False,
            "boundary_vertex_deletion": True,
            "feature_angle": 45.0,
        },
        {
            "name": "safe_decimate_pro",
            "method": "decimate_pro",
            "reduction": min(target_reduction, 0.75),
            "preserve_topology": True,
            "splitting": False,
            "boundary_vertex_deletion": False,
            "feature_angle": 45.0,
        },
        {
            "name": "fallback_decimate",
            "method": "decimate",
            "reduction": min(target_reduction, 0.80),
        },
    ]

    best_mesh = mesh
    best_reduction = 0.0

    print(f"\nDecimating {mesh_name}")
    print("-" * (11 + len(mesh_name)))
    print(f"Original cells: {original_cells}")

    for attempt in attempts:
        try:
            print(f"Trying {attempt['name']}...")

            if attempt["method"] == "decimate_pro":
                candidate = mesh.decimate_pro(
                    reduction=attempt["reduction"],
                    preserve_topology=attempt["preserve_topology"],
                    feature_angle=attempt["feature_angle"],
                    splitting=attempt["splitting"],
                    boundary_vertex_deletion=attempt["boundary_vertex_deletion"],
                )
            else:
                candidate = mesh.decimate(
                    target_reduction=attempt["reduction"],
                    volume_preservation=True,
                )

            candidate = candidate.clean()

            if candidate.n_points == 0 or candidate.n_cells == 0:
                print(f"{attempt['name']} produced an empty mesh. Skipping.")
                continue

            reduction_percent = 100.0 * (1.0 - candidate.n_cells / original_cells)

            print(
                f"{attempt['name']} result: "
                f"points={candidate.n_points}, cells={candidate.n_cells}, "
                f"reduction={reduction_percent:.1f}%"
            )

            if reduction_percent > best_reduction:
                best_mesh = candidate
                best_reduction = reduction_percent

            if reduction_percent >= 60.0:
                print(f"Accepted {attempt['name']} for {mesh_name}.")
                return candidate

        except Exception as error:
            print(f"{attempt['name']} failed for {mesh_name}: {error}")

    if best_reduction < 30.0:
        print(
            f"{mesh_name}: standard decimation reduction was only "
            f"{best_reduction:.1f}%. Switching to quadric clustering fallback."
        )

        clustered = quadric_cluster_decimate(
            mesh=mesh,
            mesh_name=mesh_name,
        )

        clustered_reduction = 100.0 * (1.0 - clustered.n_cells / original_cells)

        if clustered_reduction > best_reduction:
            return clustered

    if best_mesh.n_cells == original_cells:
        print(f"Warning: {mesh_name} could not be reduced. Original mesh will be used.")
    else:
        print(
            f"Using best available result for {mesh_name}: "
            f"cells={best_mesh.n_cells}, reduction={best_reduction:.1f}%"
        )

    return best_mesh


def summarize_mesh(
    name: str,
    mesh: pv.PolyData,
    original_points: int | None = None,
    original_cells: int | None = None,
    elapsed_seconds: float | None = None,
) -> dict:
    size_x, size_y, size_z = mesh_dimensions(mesh)

    row = {
        "name": name,
        "points": mesh.n_points,
        "cells": mesh.n_cells,
        "size_x": size_x,
        "size_y": size_y,
        "size_z": size_z,
        "max_size": max(size_x, size_y, size_z),
        "surface_area": float(mesh.area),
        "elapsed_seconds": elapsed_seconds if elapsed_seconds is not None else 0.0,
        "point_reduction_percent": 0.0,
        "cell_reduction_percent": 0.0,
    }

    if original_points is not None and original_points > 0:
        row["point_reduction_percent"] = 100.0 * (1.0 - mesh.n_points / original_points)

    if original_cells is not None and original_cells > 0:
        row["cell_reduction_percent"] = 100.0 * (1.0 - mesh.n_cells / original_cells)

    return row


def save_report(rows: list[dict]) -> None:
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "name",
                "points",
                "cells",
                "size_x",
                "size_y",
                "size_z",
                "max_size",
                "surface_area",
                "elapsed_seconds",
                "point_reduction_percent",
                "cell_reduction_percent",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def make_instance(
    mesh: pv.PolyData,
    position: tuple[float, float, float],
    scale: float = 0.62,
) -> pv.PolyData:
    instance = mesh.copy()
    instance.scale(scale, inplace=True)
    instance.translate(position, inplace=True)
    return instance


def add_mesh_with_label(
    plotter: pv.Plotter,
    mesh: pv.PolyData,
    position: tuple[float, float, float],
    color: tuple[float, float, float],
    label: str,
) -> None:
    instance = make_instance(mesh, position=position, scale=0.62)

    plotter.add_mesh(
        instance,
        color=color,
        smooth_shading=True,
        show_edges=False,
        opacity=1.0,
        specular=0.22,
    )

    plotter.add_point_labels(
        [(position[0], position[1] - 0.62, position[2])],
        [label],
        font_size=11,
        text_color="black",
        point_color="white",
        point_size=1,
        shape=None,
        always_visible=True,
    )


def main() -> None:
    print("Phase 5 Week 3 Day 2: Mesh decimation / simplification test")

    inactive_original = center_and_scale_mesh(
        load_mesh(INACTIVE_MESH_PATH),
        target_size=1.0,
    )

    activated_original = center_and_scale_mesh(
        load_mesh(ACTIVATED_MESH_PATH),
        target_size=1.0,
    )

    print("\nOriginal mesh sizes")
    print("-------------------")
    print(f"Inactive:  points={inactive_original.n_points}, cells={inactive_original.n_cells}")
    print(f"Activated: points={activated_original.n_points}, cells={activated_original.n_cells}")

    rows = []
    rows.append(summarize_mesh("inactive_original", inactive_original))
    rows.append(summarize_mesh("activated_original", activated_original))

    start = time.perf_counter()
    inactive_decimated = safe_decimate(
        inactive_original,
        target_reduction=INACTIVE_TARGET_REDUCTION,
        mesh_name="inactive",
    )
    inactive_time = time.perf_counter() - start

    start = time.perf_counter()
    activated_decimated = safe_decimate(
        activated_original,
        target_reduction=ACTIVATED_TARGET_REDUCTION,
        mesh_name="activated",
    )
    activated_time = time.perf_counter() - start

    inactive_decimated.save(INACTIVE_DECIMATED_PATH)
    activated_decimated.save(ACTIVATED_DECIMATED_PATH)

    inactive_row = summarize_mesh(
        "inactive_decimated",
        inactive_decimated,
        original_points=inactive_original.n_points,
        original_cells=inactive_original.n_cells,
        elapsed_seconds=inactive_time,
    )

    activated_row = summarize_mesh(
        "activated_decimated",
        activated_decimated,
        original_points=activated_original.n_points,
        original_cells=activated_original.n_cells,
        elapsed_seconds=activated_time,
    )

    rows.append(inactive_row)
    rows.append(activated_row)

    save_report(rows)

    print("\nDecimated mesh sizes")
    print("--------------------")
    print(
        f"Inactive decimated:  points={inactive_decimated.n_points}, "
        f"cells={inactive_decimated.n_cells}, "
        f"cell reduction={inactive_row['cell_reduction_percent']:.1f}%, "
        f"time={inactive_time:.2f}s"
    )
    print(
        f"Activated decimated: points={activated_decimated.n_points}, "
        f"cells={activated_decimated.n_cells}, "
        f"cell reduction={activated_row['cell_reduction_percent']:.1f}%, "
        f"time={activated_time:.2f}s"
    )

    print("\nSaved optimized meshes:")
    print(INACTIVE_DECIMATED_PATH)
    print(ACTIVATED_DECIMATED_PATH)

    print("\nSaved report:")
    print(OUTPUT_CSV)

    plotter = pv.Plotter(off_screen=True, window_size=(1900, 1050))
    plotter.set_background("white")

    add_mesh_with_label(
        plotter,
        inactive_original,
        position=(-1.65, 0.55, 0.0),
        color=(0.20, 0.55, 0.95),
        label=(
            "inactive original\n"
            f"points={inactive_original.n_points}\n"
            f"cells={inactive_original.n_cells}"
        ),
    )

    add_mesh_with_label(
        plotter,
        inactive_decimated,
        position=(0.35, 0.55, 0.0),
        color=(0.20, 0.55, 0.95),
        label=(
            "inactive decimated\n"
            f"points={inactive_decimated.n_points}\n"
            f"cells={inactive_decimated.n_cells}\n"
            f"reduction={inactive_row['cell_reduction_percent']:.1f}%"
        ),
    )

    add_mesh_with_label(
        plotter,
        activated_original,
        position=(-1.65, -0.75, 0.0),
        color=(0.95, 0.25, 0.18),
        label=(
            "activated original\n"
            f"points={activated_original.n_points}\n"
            f"cells={activated_original.n_cells}"
        ),
    )

    add_mesh_with_label(
        plotter,
        activated_decimated,
        position=(0.35, -0.75, 0.0),
        color=(0.95, 0.25, 0.18),
        label=(
            "activated decimated\n"
            f"points={activated_decimated.n_points}\n"
            f"cells={activated_decimated.n_cells}\n"
            f"reduction={activated_row['cell_reduction_percent']:.1f}%"
        ),
    )

    plotter.add_text(
        "Phase 5 Week 3: Platelet Mesh Decimation for Visualization Optimization",
        position=(320, 985),
        font_size=18,
        color="black",
    )

    plotter.add_text(
        "Purpose: reduce mesh complexity while preserving recognizable platelet morphology",
        position=(430, 950),
        font_size=13,
        color="dimgray",
    )

    plotter.add_text(
        f"Target reduction: inactive={INACTIVE_TARGET_REDUCTION:.0%}, "
        f"activated={ACTIVATED_TARGET_REDUCTION:.0%}",
        position=(570, 920),
        font_size=12,
        color="black",
    )

    plotter.add_text(
        f"Measured cell reduction: inactive={inactive_row['cell_reduction_percent']:.1f}% | "
        f"activated={activated_row['cell_reduction_percent']:.1f}%",
        position=(520, 890),
        font_size=12,
        color="black",
    )

    plotter.add_text(
        "Note: optimized VTP meshes are local results for faster visualization and later USD/Omniverse tests.",
        position=(390, 860),
        font_size=12,
        color="dimgray",
    )

    plotter.add_axes()

    plotter.camera_position = [
        (-0.65, -4.7, 2.5),
        (-0.65, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ]

    plotter.screenshot(str(OUTPUT_IMAGE))
    plotter.close()

    print("\nSaved visual comparison:")
    print(OUTPUT_IMAGE)

    print("\nMesh decimation test complete.")


if __name__ == "__main__":
    main()