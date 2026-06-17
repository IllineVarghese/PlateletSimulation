from pathlib import Path
import csv
import numpy as np
import pyvista as pv

from pxr import Usd, UsdGeom, Sdf, Gf, UsdLux

from mesh_utils import center_and_scale_mesh


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Real Phase 4 final demo data
POSITIONS_PATH = PROJECT_ROOT / "results" / "phase4" / "final_demo" / "positions.npy"
ACTIVATION_PATH = PROJECT_ROOT / "results" / "phase4" / "final_demo" / "activation.npy"
SHEAR_PATH = PROJECT_ROOT / "results" / "phase4" / "final_demo" / "shear_input.npy"

# Decimated meshes from Week 3 Day 2
INACTIVE_DECIMATED_PATH = (
    PROJECT_ROOT
    / "results"
    / "phase5"
    / "week3"
    / "optimized_meshes"
    / "inactive_decimated.vtp"
)

ACTIVATED_DECIMATED_PATH = (
    PROJECT_ROOT
    / "results"
    / "phase5"
    / "week3"
    / "optimized_meshes"
    / "activated_decimated.vtp"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "phase5"
    / "week4"
    / "exports"
    / "static_usd_scene"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_USDA = OUTPUT_DIR / "phase5_platelet_static_scene.usda"
OUTPUT_METADATA_CSV = OUTPUT_DIR / "platelet_scene_metadata.csv"
OUTPUT_SUMMARY_MD = OUTPUT_DIR / "usd_static_scene_export_summary.md"
OUTPUT_PREVIEW_PNG = OUTPUT_DIR / "phase5_static_usd_scene_preview.png"

FRAME_INDEX = -1
MAX_EXPORTED_PLATELETS = 80
ACTIVATION_THRESHOLD = 0.50


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

    return inactive.triangulate().clean(), activated.triangulate().clean()


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


def choose_export_indices(
    activation: np.ndarray,
    shear: np.ndarray,
    max_count: int,
) -> np.ndarray:
    """
    Select a useful static export subset:
    - high final activation
    - high final shear
    - strong activation increase
    - some low activation examples
    """
    final_activation = activation[-1]
    final_shear = shear[-1]
    delta_activation = activation[-1] - activation[0]

    n_platelets = final_activation.size

    n_high_activation = max_count // 3
    n_high_shear = max_count // 4
    n_dynamic = max_count // 4
    n_low = max_count - n_high_activation - n_high_shear - n_dynamic

    high_activation_idx = np.argsort(final_activation)[::-1][:n_high_activation]
    high_shear_idx = np.argsort(final_shear)[::-1][:n_high_shear]
    dynamic_idx = np.argsort(delta_activation)[::-1][:n_dynamic]
    low_idx = np.argsort(final_activation)[:n_low]

    selected = np.unique(
        np.concatenate(
            [
                high_activation_idx,
                high_shear_idx,
                dynamic_idx,
                low_idx,
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

    vessel = pv.Cylinder(
        center=(center_x, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0),
        radius=1.05,
        height=length,
        resolution=96,
    )

    return vessel.triangulate().clean()


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
    return mesh.triangulate().clean()


def polydata_faces_to_usd(mesh: pv.PolyData) -> tuple[list[int], list[int]]:
    """
    Convert PyVista face array into USD faceVertexCounts and faceVertexIndices.
    PyVista stores faces as: [n, id0, id1, ..., n, id0, id1, ...]
    """
    faces = mesh.faces
    face_vertex_counts = []
    face_vertex_indices = []

    i = 0
    while i < len(faces):
        n = int(faces[i])
        ids = faces[i + 1 : i + 1 + n]

        face_vertex_counts.append(n)
        face_vertex_indices.extend(int(value) for value in ids)

        i += n + 1

    return face_vertex_counts, face_vertex_indices


def add_usd_mesh(
    stage: Usd.Stage,
    prim_path: str,
    mesh: pv.PolyData,
    color: tuple[float, float, float],
    opacity: float = 1.0,
) -> UsdGeom.Mesh:
    """
    Add a PyVista mesh as a USD Mesh primitive.
    """
    mesh = mesh.triangulate().clean()

    usd_mesh = UsdGeom.Mesh.Define(stage, prim_path)

    points = [
        Gf.Vec3f(float(x), float(y), float(z))
        for x, y, z in mesh.points
    ]

    face_vertex_counts, face_vertex_indices = polydata_faces_to_usd(mesh)

    usd_mesh.CreatePointsAttr(points)
    usd_mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
    usd_mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
    usd_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    usd_mesh.CreateDoubleSidedAttr(True)

    usd_mesh.CreateDisplayColorAttr(
        [Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))]
    )

    usd_mesh.CreateDisplayOpacityAttr([float(opacity)])

    return usd_mesh


def add_scene_lighting(stage: Usd.Stage) -> None:
    light = UsdLux.DistantLight.Define(stage, "/Phase5StaticScene/Lighting/KeyLight")
    light.CreateIntensityAttr(600.0)
    light.CreateAngleAttr(0.35)

    xform = UsdGeom.Xformable(light.GetPrim())
    xform.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 35.0))


def add_scene_camera(stage: Usd.Stage) -> None:
    camera = UsdGeom.Camera.Define(stage, "/Phase5StaticScene/Camera")
    camera.CreateFocalLengthAttr(35.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))

    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(4.0, -6.2, 3.0))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(62.0, 0.0, 35.0))

    stage.SetMetadata("documentation", "Static Phase 5 platelet USD export scene")


def save_metadata_csv(rows: list[dict]) -> None:
    with OUTPUT_METADATA_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "export_id",
                "source_platelet_index",
                "usd_prim_path",
                "frame_index",
                "x",
                "y",
                "z",
                "activation",
                "shear",
                "state",
                "mesh_source",
                "scale",
                "color_r",
                "color_g",
                "color_b",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def save_summary(
    n_frames: int,
    n_platelets: int,
    selected_count: int,
    inactive_count: int,
    activated_count: int,
    inactive_mesh: pv.PolyData,
    activated_mesh: pv.PolyData,
) -> None:
    lines = []

    lines.append("# Phase 5 Week 4 Day 2: Static USD Scene Export")
    lines.append("")
    lines.append("## Export Status")
    lines.append("")
    lines.append("Direct USD export was performed using the Pixar USD Python modules from `usd-core`.")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append(f"- USD scene: `{OUTPUT_USDA}`")
    lines.append(f"- Metadata CSV: `{OUTPUT_METADATA_CSV}`")
    lines.append(f"- Preview image: `{OUTPUT_PREVIEW_PNG}`")
    lines.append("")
    lines.append("## Scene Contents")
    lines.append("")
    lines.append(f"- Source Phase 4 frames: {n_frames}")
    lines.append(f"- Source platelet count: {n_platelets}")
    lines.append(f"- Exported platelet count: {selected_count}")
    lines.append(f"- Inactive exported platelets: {inactive_count}")
    lines.append(f"- Activated exported platelets: {activated_count}")
    lines.append(f"- Activation threshold: {ACTIVATION_THRESHOLD:.2f}")
    lines.append("")
    lines.append("## Mesh Complexity")
    lines.append("")
    lines.append(f"- Inactive decimated mesh points: {inactive_mesh.n_points}")
    lines.append(f"- Inactive decimated mesh cells: {inactive_mesh.n_cells}")
    lines.append(f"- Activated decimated mesh points: {activated_mesh.n_points}")
    lines.append(f"- Activated decimated mesh cells: {activated_mesh.n_cells}")
    lines.append("")
    lines.append("## Visual Encoding")
    lines.append("")
    lines.append("- Mesh state: inactive or activated mesh selected from activation threshold.")
    lines.append("- Color: activation level, blue for lower activation and red for higher activation.")
    lines.append("- Size: scaled slightly by activation and shear for readability.")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "This export is a static USD scene using one selected Phase 4 frame. "
        "It is intended as an Omniverse/USD-compatible visualization asset. "
        "Animation export will be handled separately in the next Week 4 step."
    )

    OUTPUT_SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def render_preview(
    vessel: pv.PolyData,
    platelet_meshes: list[tuple[pv.PolyData, tuple[float, float, float]]],
    inactive_count: int,
    activated_count: int,
) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=(1600, 900))
    plotter.set_background("white")

    plotter.add_mesh(
        vessel,
        color=(0.86, 0.90, 0.92),
        opacity=0.10,
        smooth_shading=True,
        show_edges=False,
    )

    for mesh, color in platelet_meshes:
        plotter.add_mesh(
            mesh,
            color=color,
            smooth_shading=True,
            show_edges=False,
            opacity=0.97,
            specular=0.20,
        )

    plotter.add_text(
        "Phase 5 Week 4: Static USD Export Preview",
        position=(420, 850),
        font_size=17,
        color="black",
    )

    plotter.add_text(
        f"Exported platelets={len(platelet_meshes)} | inactive={inactive_count} | activated={activated_count}",
        position=(465, 820),
        font_size=12,
        color="black",
    )

    plotter.add_text(
        "Scene exported as USDA with activation state, color, shear, and platelet metadata",
        position=(360, 792),
        font_size=11,
        color="dimgray",
    )

    plotter.add_axes()

    plotter.camera_position = [
        (4.0, -6.2, 3.0),
        (4.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ]

    plotter.screenshot(str(OUTPUT_PREVIEW_PNG))
    plotter.close()


def main() -> None:
    print("Phase 5 Week 4 Day 2: Static USD scene export")

    positions, activation, shear = load_phase4_data()
    inactive_mesh, activated_mesh = load_decimated_meshes()

    n_frames, n_platelets, _ = positions.shape
    frame_number = n_frames - 1 if FRAME_INDEX == -1 else FRAME_INDEX

    frame_positions = positions[FRAME_INDEX]
    frame_activation = activation[FRAME_INDEX]
    frame_shear = shear[FRAME_INDEX]

    selected_indices = choose_export_indices(
        activation=activation,
        shear=shear,
        max_count=MAX_EXPORTED_PLATELETS,
    )

    print(f"positions shape:  {positions.shape}")
    print(f"activation shape: {activation.shape}")
    print(f"shear shape:      {shear.shape}")
    print(f"selected frame:   {frame_number}")
    print(f"selected count:   {len(selected_indices)}")
    print(f"inactive mesh:    points={inactive_mesh.n_points}, cells={inactive_mesh.n_cells}")
    print(f"activated mesh:   points={activated_mesh.n_points}, cells={activated_mesh.n_cells}")

    if OUTPUT_USDA.exists():
        OUTPUT_USDA.unlink()

    stage = Usd.Stage.CreateNew(str(OUTPUT_USDA))

    root = UsdGeom.Xform.Define(stage, "/Phase5StaticScene")
    stage.SetDefaultPrim(root.GetPrim())

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    vessel = make_vessel_proxy(positions)

    add_usd_mesh(
        stage=stage,
        prim_path="/Phase5StaticScene/Vessel/VesselProxy",
        mesh=vessel,
        color=(0.86, 0.90, 0.92),
        opacity=0.20,
    )

    add_scene_lighting(stage)
    add_scene_camera(stage)

    metadata_rows = []
    preview_platelets = []

    inactive_count = 0
    activated_count = 0

    for export_id, platelet_idx in enumerate(selected_indices):
        position = frame_positions[platelet_idx]
        act = float(frame_activation[platelet_idx])
        shr = float(frame_shear[platelet_idx])

        if act >= ACTIVATION_THRESHOLD:
            base_mesh = activated_mesh
            state = "activated"
            mesh_source = "activated_decimated.vtp"
            activated_count += 1
            state_scale_bonus = 0.012
        else:
            base_mesh = inactive_mesh
            state = "inactive"
            mesh_source = "inactive_decimated.vtp"
            inactive_count += 1
            state_scale_bonus = 0.0

        scale = 0.065 + 0.015 * act + 0.008 * shr + state_scale_bonus

        transformed = transform_platelet_mesh(
            base_mesh=base_mesh,
            position=position,
            scale=scale,
            rotation_x=(int(platelet_idx) * 7.0) % 360,
            rotation_y=(int(platelet_idx) * 13.0) % 360,
            rotation_z=(int(platelet_idx) * 17.0) % 360,
        )

        color = activation_to_color(act)

        prim_path = f"/Phase5StaticScene/Platelets/Platelet_{export_id:03d}"

        usd_mesh = add_usd_mesh(
            stage=stage,
            prim_path=prim_path,
            mesh=transformed,
            color=color,
            opacity=0.98,
        )

        prim = usd_mesh.GetPrim()
        prim.CreateAttribute("sourcePlateletIndex", Sdf.ValueTypeNames.Int).Set(int(platelet_idx))
        prim.CreateAttribute("frameIndex", Sdf.ValueTypeNames.Int).Set(int(frame_number))
        prim.CreateAttribute("activation", Sdf.ValueTypeNames.Float).Set(float(act))
        prim.CreateAttribute("shear", Sdf.ValueTypeNames.Float).Set(float(shr))
        prim.CreateAttribute("state", Sdf.ValueTypeNames.String).Set(state)
        prim.CreateAttribute("meshSource", Sdf.ValueTypeNames.String).Set(mesh_source)

        metadata_rows.append(
            {
                "export_id": export_id,
                "source_platelet_index": int(platelet_idx),
                "usd_prim_path": prim_path,
                "frame_index": frame_number,
                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2]),
                "activation": act,
                "shear": shr,
                "state": state,
                "mesh_source": mesh_source,
                "scale": scale,
                "color_r": color[0],
                "color_g": color[1],
                "color_b": color[2],
            }
        )

        preview_platelets.append((transformed, color))

    save_metadata_csv(metadata_rows)

    stage.GetRootLayer().Save()

    render_preview(
        vessel=vessel,
        platelet_meshes=preview_platelets,
        inactive_count=inactive_count,
        activated_count=activated_count,
    )

    save_summary(
        n_frames=n_frames,
        n_platelets=n_platelets,
        selected_count=len(selected_indices),
        inactive_count=inactive_count,
        activated_count=activated_count,
        inactive_mesh=inactive_mesh,
        activated_mesh=activated_mesh,
    )

    print("\nExport complete.")
    print(f"USD scene:       {OUTPUT_USDA}")
    print(f"Metadata CSV:    {OUTPUT_METADATA_CSV}")
    print(f"Summary:         {OUTPUT_SUMMARY_MD}")
    print(f"Preview image:   {OUTPUT_PREVIEW_PNG}")
    print(f"Inactive count:  {inactive_count}")
    print(f"Activated count: {activated_count}")
    print("\nWeek 4 Day 2 static USD scene export complete.")


if __name__ == "__main__":
    main()