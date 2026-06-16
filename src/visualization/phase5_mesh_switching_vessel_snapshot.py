from pathlib import Path
import numpy as np
import pyvista as pv

from mesh_utils import load_mesh, center_and_scale_mesh


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INACTIVE_MESH_PATH = PROJECT_ROOT / "data" / "meshes" / "platelet" / "inactive.obj"
ACTIVATED_MESH_PATH = PROJECT_ROOT / "data" / "meshes" / "platelet" / "activated.obj"

OUTPUT_DIR = PROJECT_ROOT / "results" / "phase5" / "week2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "mesh_switching_vessel_snapshot.png"

ACTIVATION_THRESHOLD = 0.50


def make_vessel_proxy(length: float = 5.0, radius: float = 0.75) -> pv.PolyData:
    """
    Create a simple transparent cylindrical vessel proxy.

    This is only for visualization.
    It does not replace the Phase 4 vessel mesh/proxy logic.
    """
    vessel = pv.Cylinder(
        center=(0.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0),
        radius=radius,
        height=length,
        resolution=96,
    )
    return vessel


def transform_platelet_mesh(
    base_mesh: pv.PolyData,
    position: tuple[float, float, float],
    scale: float,
    rotation_x: float,
    rotation_y: float,
    rotation_z: float,
) -> pv.PolyData:
    """
    Create a transformed platelet mesh instance.

    The original base mesh is not modified.
    """
    mesh = base_mesh.copy()

    mesh.scale(scale, inplace=True)
    mesh.rotate_x(rotation_x, inplace=True)
    mesh.rotate_y(rotation_y, inplace=True)
    mesh.rotate_z(rotation_z, inplace=True)
    mesh.translate(position, inplace=True)

    return mesh


def main() -> None:
    print("Phase 5 Week 2 Day 1: Activation-based mesh switching vessel snapshot")

    inactive_base = center_and_scale_mesh(
        load_mesh(INACTIVE_MESH_PATH),
        target_size=1.0,
    )

    activated_base = center_and_scale_mesh(
        load_mesh(ACTIVATED_MESH_PATH),
        target_size=1.0,
    )

    # Small population for safe high-quality rendering.
    # X coordinate follows vessel flow direction.
    positions = np.array(
        [
            [-2.2,  0.20,  0.10],
            [-1.8, -0.25, -0.05],
            [-1.4,  0.35, -0.10],
            [-1.0, -0.10,  0.20],
            [-0.6,  0.15, -0.25],
            [-0.2, -0.35,  0.10],
            [ 0.2,  0.30,  0.05],
            [ 0.6, -0.15, -0.20],
            [ 1.0,  0.25,  0.15],
            [ 1.4, -0.30,  0.00],
            [ 1.8,  0.10, -0.15],
            [ 2.2, -0.05,  0.25],
        ],
        dtype=float,
    )

    # Example activation values.
    # Later this will come from the actual simulation state.
    activation_values = np.array(
        [
            0.05,
            0.12,
            0.22,
            0.35,
            0.44,
            0.49,
            0.52,
            0.61,
            0.70,
            0.78,
            0.88,
            0.96,
        ],
        dtype=float,
    )

    plotter = pv.Plotter(off_screen=True, window_size=(1800, 1000))
    plotter.set_background("white")

    vessel = make_vessel_proxy(length=5.2, radius=0.78)

    plotter.add_mesh(
        vessel,
        opacity=0.12,
        smooth_shading=True,
        show_edges=True,
        line_width=0.4,
    )

    inactive_count = 0
    activated_count = 0

    for i, (position, activation) in enumerate(zip(positions, activation_values)):
        if activation >= ACTIVATION_THRESHOLD:
            base_mesh = activated_base
            state = "activated"
            activated_count += 1
        else:
            base_mesh = inactive_base
            state = "inactive"
            inactive_count += 1

        platelet = transform_platelet_mesh(
            base_mesh=base_mesh,
            position=tuple(position),
            scale=0.18,
            rotation_x=10.0 * i,
            rotation_y=18.0 * i,
            rotation_z=25.0 * i,
        )

        plotter.add_mesh(
            platelet,
            smooth_shading=True,
            show_edges=False,
            opacity=1.0,
        )

        # Label only selected platelets to avoid clutter.
        if i in [0, 3, 5, 6, 9, 11]:
            label_pos = (position[0], position[1] - 0.22, position[2])
            plotter.add_point_labels(
                [label_pos],
                [f"{state}\nact={activation:.2f}"],
                font_size=12,
                text_color="black",
                point_color="white",
                point_size=1,
                shape=None,
                always_visible=True,
            )

    title = "Phase 5 Week 2: Activation-based platelet mesh switching"
    subtitle = (
        f"Rule: activation >= {ACTIVATION_THRESHOLD:.2f} uses activated.obj | "
        f"inactive={inactive_count}, activated={activated_count}"
    )

    plotter.add_text(
        title,
        position=(450, 940),
        font_size=17,
        color="black",
    )

    plotter.add_text(
        subtitle,
        position=(470, 900),
        font_size=13,
        color="black",
    )

    plotter.add_axes()

    # Camera looking slightly from front/top so vessel depth is visible.
    plotter.camera_position = [
        (3.4, -5.2, 3.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ]

    plotter.screenshot(str(OUTPUT_PATH))
    plotter.close()

    print(f"Saved output: {OUTPUT_PATH}")
    print(f"Inactive platelets:  {inactive_count}")
    print(f"Activated platelets: {activated_count}")
    print("Week 2 Day 1 snapshot complete.")


if __name__ == "__main__":
    main()