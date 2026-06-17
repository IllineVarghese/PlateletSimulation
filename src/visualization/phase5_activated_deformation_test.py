from pathlib import Path
import numpy as np
import pyvista as pv

from mesh_utils import load_mesh, center_and_scale_mesh


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INACTIVE_MESH_PATH = PROJECT_ROOT / "data" / "meshes" / "platelet" / "inactive.obj"
ACTIVATED_MESH_PATH = PROJECT_ROOT / "data" / "meshes" / "platelet" / "activated.obj"

OUTPUT_DIR = PROJECT_ROOT / "results" / "phase5" / "week3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "activated_deformation_test.png"


def deform_activated_platelet(
    mesh: pv.PolyData,
    activation: float = 1.0,
    radial_strength: float = 0.28,
    axial_strength: float = 0.18,
    roughness_strength: float = 0.045,
) -> pv.PolyData:
    """
    Create a visual-only deformation of an activated platelet mesh.

    This is not a physical soft-body simulation.
    It is a visual approximation to make the activated state more recognizable.

    Effects:
    - slight elongation
    - radial expansion
    - surface roughening
    """
    deformed = mesh.copy()
    points = deformed.points.copy()

    activation = float(np.clip(activation, 0.0, 1.0))

    # Centered coordinates
    center = points.mean(axis=0)
    p = points - center

    x = p[:, 0]
    y = p[:, 1]
    z = p[:, 2]

    radial = np.sqrt(y**2 + z**2)
    radial_max = max(float(np.max(radial)), 1e-8)

    # Normalize radial direction in YZ plane
    y_dir = np.divide(y, radial, out=np.zeros_like(y), where=radial > 1e-8)
    z_dir = np.divide(z, radial, out=np.zeros_like(z), where=radial > 1e-8)

    # Axial elongation along X
    p[:, 0] *= 1.0 + axial_strength * activation

    # Radial expansion stronger near outer surface
    radial_weight = radial / radial_max
    expansion = radial_strength * activation * radial_weight

    p[:, 1] += y_dir * expansion
    p[:, 2] += z_dir * expansion

    # Deterministic roughness pattern.
    # This avoids random flickering and makes output reproducible.
    roughness = (
        np.sin(11.0 * x)
        * np.cos(13.0 * y)
        * np.sin(17.0 * z)
    )

    p[:, 1] += roughness_strength * activation * roughness * y_dir
    p[:, 2] += roughness_strength * activation * roughness * z_dir

    deformed.points = p + center
    return deformed


def make_instance(
    mesh: pv.PolyData,
    position: tuple[float, float, float],
    scale: float = 0.8,
    rotation_z: float = 0.0,
) -> pv.PolyData:
    instance = mesh.copy()
    instance.scale(scale, inplace=True)
    instance.rotate_z(rotation_z, inplace=True)
    instance.translate(position, inplace=True)
    return instance


def main() -> None:
    print("Phase 5 Week 3 Day 1: Activated platelet deformation test")

    inactive = center_and_scale_mesh(load_mesh(INACTIVE_MESH_PATH), target_size=1.0)
    activated = center_and_scale_mesh(load_mesh(ACTIVATED_MESH_PATH), target_size=1.0)

    activated_deformed_low = deform_activated_platelet(
        activated,
        activation=0.5,
        radial_strength=0.18,
        axial_strength=0.10,
        roughness_strength=0.025,
    )

    activated_deformed_high = deform_activated_platelet(
        activated,
        activation=1.0,
        radial_strength=0.32,
        axial_strength=0.22,
        roughness_strength=0.060,
    )

    inactive_vis = make_instance(inactive, position=(-1.8, 0.0, 0.0), scale=0.75)
    activated_vis = make_instance(activated, position=(-0.6, 0.0, 0.0), scale=0.75)
    deformed_low_vis = make_instance(activated_deformed_low, position=(0.6, 0.0, 0.0), scale=0.75)
    deformed_high_vis = make_instance(activated_deformed_high, position=(1.8, 0.0, 0.0), scale=0.75)

    plotter = pv.Plotter(off_screen=True, window_size=(1800, 1000))
    plotter.set_background("white")

    plotter.add_mesh(
        inactive_vis,
        color=(0.35, 0.65, 0.95),
        smooth_shading=True,
        show_edges=False,
        opacity=1.0,
    )

    plotter.add_mesh(
        activated_vis,
        color=(0.95, 0.55, 0.45),
        smooth_shading=True,
        show_edges=False,
        opacity=1.0,
    )

    plotter.add_mesh(
        deformed_low_vis,
        color=(0.95, 0.35, 0.25),
        smooth_shading=True,
        show_edges=False,
        opacity=1.0,
    )

    plotter.add_mesh(
        deformed_high_vis,
        color=(0.85, 0.10, 0.08),
        smooth_shading=True,
        show_edges=False,
        opacity=1.0,
        specular=0.25,
    )

    labels = [
        (-1.8, -0.70, 0.0, "inactive.obj\nbaseline inactive"),
        (-0.6, -0.70, 0.0, "activated.obj\noriginal activated"),
        (0.6, -0.70, 0.0, "deformed activated\nactivation = 0.5"),
        (1.8, -0.70, 0.0, "deformed activated\nactivation = 1.0"),
    ]

    for x, y, z, text in labels:
        plotter.add_point_labels(
            [(x, y, z)],
            [text],
            font_size=13,
            text_color="black",
            point_color="white",
            point_size=1,
            shape=None,
            always_visible=True,
        )

    plotter.add_text(
        "Phase 5 Week 3: Visual Activated Platelet Deformation Test",
        position=(440, 930),
        font_size=18,
        color="black",
    )

    plotter.add_text(
        "Purpose: make activated platelet morphology more visually distinguishable for thesis visualization",
        position=(390, 895),
        font_size=13,
        color="dimgray",
    )

    plotter.add_text(
        "Note: deformation is visual-only, not a physical soft-body simulation",
        position=(520, 865),
        font_size=12,
        color="dimgray",
    )

    plotter.add_axes()
    plotter.camera_position = [
        (0.0, -4.6, 2.2),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ]

    plotter.screenshot(str(OUTPUT_PATH))
    plotter.close()

    print(f"Saved output: {OUTPUT_PATH}")
    print("Week 3 Day 1 deformation test complete.")


if __name__ == "__main__":
    main()