from pathlib import Path
import numpy as np
import pyvista as pv

from mesh_utils import center_and_scale_mesh


PROJECT_ROOT = Path(__file__).resolve().parents[2]

POSITIONS_PATH = PROJECT_ROOT / "results" / "phase4" / "final_demo" / "positions.npy"
ACTIVATION_PATH = PROJECT_ROOT / "results" / "phase4" / "final_demo" / "activation.npy"
SHEAR_PATH = PROJECT_ROOT / "results" / "phase4" / "final_demo" / "shear_input.npy"

INACTIVE_DECIMATED_PATH = (
    PROJECT_ROOT / "results" / "phase5" / "week3" / "optimized_meshes" / "inactive_decimated.vtp"
)
ACTIVATED_DECIMATED_PATH = (
    PROJECT_ROOT / "results" / "phase5" / "week3" / "optimized_meshes" / "activated_decimated.vtp"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "phase5" / "presentation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_IMAGE = OUTPUT_DIR / "phase5_flow_overlay_snapshot.png"

FRAME_INDEX = -1
MAX_RENDERED_PLATELETS = 100
ACTIVATION_THRESHOLD = 0.50


def load_phase4_data():
    positions = np.load(POSITIONS_PATH)
    activation = np.load(ACTIVATION_PATH)
    shear = np.load(SHEAR_PATH)
    return positions, activation, shear


def load_decimated_meshes():
    inactive = center_and_scale_mesh(
        pv.read(INACTIVE_DECIMATED_PATH),
        target_size=1.0,
    ).triangulate().clean()

    activated = center_and_scale_mesh(
        pv.read(ACTIVATED_DECIMATED_PATH),
        target_size=1.0,
    ).triangulate().clean()

    return inactive, activated


def activation_to_color(a):
    a = float(np.clip(a, 0.0, 1.0))
    if a < 0.5:
        t = a / 0.5
        return 0.12 + 0.65 * t, 0.34 + 0.45 * t, 0.95
    t = (a - 0.5) / 0.5
    return 0.95, 0.78 - 0.58 * t, 0.82 - 0.70 * t


def choose_indices(activation, shear, max_count):
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
            [dynamic_idx, high_activation_idx, high_shear_idx, low_activation_idx]
        )
    )

    if selected.size < max_count:
        remaining = np.setdiff1d(np.arange(final_activation.size), selected)
        selected = np.concatenate([selected, remaining[: max_count - selected.size]])

    return selected[:max_count].astype(int)


def make_vessel_proxy(all_positions):
    x_min = float(np.min(all_positions[..., 0]))
    x_max = float(np.max(all_positions[..., 0]))
    center_x = 0.5 * (x_min + x_max)
    length = max(x_max - x_min, 1.0)

    vessel = pv.Cylinder(
        center=(center_x, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0),
        radius=1.05,
        height=length,
        resolution=96,
    )

    return vessel


def create_flow_arrows(all_positions):
    x_min = float(np.min(all_positions[..., 0]))
    x_max = float(np.max(all_positions[..., 0]))

    xs = np.linspace(x_min + 0.5, x_max - 0.5, 7)
    ys = np.linspace(-0.55, 0.55, 5)

    points = []
    vectors = []
    magnitudes = []

    r_max = 0.75

    for x in xs:
        for y in ys:
            r = abs(y)
            speed = max(0.0, 1.0 - (r / r_max) ** 2)  # simple parabolic profile
            if speed <= 0.02:
                continue

            points.append([x, y, 0.0])
            vectors.append([0.22 + 0.45 * speed, 0.0, 0.0])
            magnitudes.append(speed)

    pdata = pv.PolyData(np.array(points))
    pdata["vectors"] = np.array(vectors)
    pdata["speed"] = np.array(magnitudes)

    arrows = pdata.glyph(orient="vectors", scale="speed", factor=0.9)
    return arrows


def transform_platelet_mesh(base_mesh, position, scale, rx, ry, rz):
    mesh = base_mesh.copy()
    mesh.scale(scale, inplace=True)
    mesh.rotate_x(rx, inplace=True)
    mesh.rotate_y(ry, inplace=True)
    mesh.rotate_z(rz, inplace=True)
    mesh.translate(tuple(position), inplace=True)
    return mesh


def main():
    print("Creating Phase 5 presentation snapshot with visible flow overlay...")

    positions, activation, shear = load_phase4_data()
    inactive_mesh, activated_mesh = load_decimated_meshes()

    frame_positions = positions[FRAME_INDEX]
    frame_activation = activation[FRAME_INDEX]
    frame_shear = shear[FRAME_INDEX]

    selected_indices = choose_indices(activation, shear, MAX_RENDERED_PLATELETS)

    vessel = make_vessel_proxy(positions)
    flow_arrows = create_flow_arrows(positions)

    inactive_count = 0
    activated_count = 0

    plotter = pv.Plotter(off_screen=True, window_size=(1800, 1000))
    plotter.set_background("white")

    plotter.add_mesh(
        vessel,
        color=(0.87, 0.90, 0.92),
        opacity=0.10,
        smooth_shading=True,
        show_edges=False,
    )

    plotter.add_mesh(
        flow_arrows,
        color=(0.20, 0.55, 0.95),
        opacity=0.65,
        smooth_shading=True,
    )

    for idx in selected_indices:
        pos = frame_positions[idx]
        act = float(frame_activation[idx])
        shr = float(frame_shear[idx])

        if act >= ACTIVATION_THRESHOLD:
            base_mesh = activated_mesh
            activated_count += 1
            state_bonus = 0.012
        else:
            base_mesh = inactive_mesh
            inactive_count += 1
            state_bonus = 0.0

        scale = 0.065 + 0.015 * act + 0.008 * shr + state_bonus

        platelet = transform_platelet_mesh(
            base_mesh=base_mesh,
            position=pos,
            scale=scale,
            rx=(int(idx) * 7.0) % 360,
            ry=(int(idx) * 13.0) % 360,
            rz=(int(idx) * 17.0) % 360,
        )

        plotter.add_mesh(
            platelet,
            color=activation_to_color(act),
            smooth_shading=True,
            show_edges=False,
            opacity=0.97,
            specular=0.22,
        )

    plotter.add_text(
        "Phase 5: Flow-Driven Platelet Mesh Visualization",
        position=(430, 950),
        font_size=18,
        color="black",
    )

    plotter.add_text(
        f"Rendered platelets={len(selected_indices)} | inactive={inactive_count} | activated={activated_count}",
        position=(510, 918),
        font_size=12,
        color="black",
    )

    plotter.add_text(
        "Visible arrows represent the prescribed Poiseuille flow field used to advect platelets and compute shear stress",
        position=(200, 888),
        font_size=11,
        color="dimgray",
    )

    plotter.add_text(
        "Blue = low activation",
        position=(70, 850),
        font_size=11,
        color=(0.12, 0.34, 0.95),
    )

    plotter.add_text(
        "Red = high activation",
        position=(70, 825),
        font_size=11,
        color=(0.95, 0.20, 0.12),
    )

    plotter.add_text(
        "Flow direction",
        position=(1450, 840),
        font_size=12,
        color=(0.20, 0.55, 0.95),
    )

    plotter.add_axes()

    plotter.camera_position = [
        (4.0, -6.2, 3.0),
        (4.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ]

    plotter.screenshot(str(OUTPUT_IMAGE))
    plotter.close()

    print(f"Saved: {OUTPUT_IMAGE}")
    print("Done.")


if __name__ == "__main__":
    main()