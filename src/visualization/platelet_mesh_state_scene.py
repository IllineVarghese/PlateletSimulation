from pathlib import Path
import numpy as np
import pyvista as pv

from mesh_utils import load_mesh, center_and_scale_mesh


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INACTIVE_MESH_PATH = PROJECT_ROOT / "data" / "meshes" / "platelet" / "inactive.obj"
ACTIVATED_MESH_PATH = PROJECT_ROOT / "data" / "meshes" / "platelet" / "activated.obj"

OUTPUT_DIR = PROJECT_ROOT / "results" / "phase5" / "week1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ACTIVATION_THRESHOLD = 0.5


def make_platelet_instance(
    base_mesh: pv.PolyData,
    position: tuple[float, float, float],
    scale: float = 0.18,
    rotation_deg: float = 0.0,
) -> pv.PolyData:
    """
    Create a transformed copy of a platelet mesh.

    This is visual-only transformation.
    The original mesh is not modified.
    """
    mesh = base_mesh.copy()
    mesh.scale(scale, inplace=True)
    mesh.rotate_z(rotation_deg, inplace=True)
    mesh.translate(position, inplace=True)
    return mesh


def main() -> None:
    print("Phase 5 Week 1 Day 4: Platelet mesh state scene")

    inactive_base = center_and_scale_mesh(load_mesh(INACTIVE_MESH_PATH), target_size=1.0)
    activated_base = center_and_scale_mesh(load_mesh(ACTIVATED_MESH_PATH), target_size=1.0)

    # Small test population: position + activation value.
    platelet_positions = [
        (-1.8, 0.4, 0.0),
        (-1.3, -0.2, 0.0),
        (-0.8, 0.3, 0.0),
        (-0.2, -0.3, 0.0),
        (0.4, 0.2, 0.0),
        (0.9, -0.1, 0.0),
        (1.4, 0.35, 0.0),
        (1.8, -0.25, 0.0),
    ]

    activation_values = np.array([
        0.05,
        0.18,
        0.30,
        0.45,
        0.55,
        0.68,
        0.82,
        0.95,
    ])

    plotter = pv.Plotter(off_screen=True, window_size=(1600, 900))
    plotter.set_background("white")

    for i, (position, activation) in enumerate(zip(platelet_positions, activation_values)):
        if activation >= ACTIVATION_THRESHOLD:
            selected_mesh = activated_base
            state_name = "activated"
        else:
            selected_mesh = inactive_base
            state_name = "inactive"

        platelet = make_platelet_instance(
            selected_mesh,
            position=position,
            scale=0.35,
            rotation_deg=i * 18.0,
        )

        plotter.add_mesh(
            platelet,
            smooth_shading=True,
            show_edges=False,
            opacity=1.0,
        )

        label_position = (position[0] - 0.13, position[1] - 0.38, position[2])
        plotter.add_point_labels(
            [label_position],
            [f"{state_name}\nact={activation:.2f}"],
            font_size=13,
            text_color="black",
            point_color="white",
            point_size=1,
            shape=None,
            always_visible=True,
        )

    plotter.add_text(
        "Phase 5 Week 1: Activation-based platelet mesh state preview",
        position=(370, 840),
        font_size=15,
        color="black",
    )

    plotter.add_text(
        f"Switching rule: activation >= {ACTIVATION_THRESHOLD} uses activated.obj",
        position=(500, 805),
        font_size=13,
        color="black",
    )

    plotter.add_axes()
    plotter.camera_position = "xy"

    output_path = OUTPUT_DIR / "platelet_mesh_state_scene.png"
    plotter.screenshot(str(output_path))
    plotter.close()

    print(f"Saved output: {output_path}")
    print("Day 4 mesh state scene complete.")


if __name__ == "__main__":
    main()