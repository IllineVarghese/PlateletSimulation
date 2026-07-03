from pathlib import Path
import csv
import numpy as np
import pyvista as pv

try:
    from pxr import Usd, UsdGeom, UsdLux, Sdf, Gf
except ImportError as error:
    raise ImportError(
        "Pixar USD Python tools are missing. Install usd-core first:\n"
        "python -m pip install usd-core"
    ) from error


# ============================================================
# Phase 5: USD export for 120-platelet cylindrical activation
# ------------------------------------------------------------
# Exports the final Phase 5 cylindrical platelet flow scene
# into an Omniverse/USD-ready animated USDA file.
#
# Uses:
# - UsdGeom.PointInstancer
# - inactive platelet prototype
# - activated platelet prototype
# - 120 platelet instances
# - time-sampled positions, scales, and prototype indices
# ============================================================


ROOT = Path(__file__).resolve().parents[2]

POSITIONS_PATH = ROOT / "results" / "phase4" / "final_demo" / "positions.npy"
ACTIVATION_PATH = ROOT / "results" / "phase4" / "final_demo" / "activation.npy"
SHEAR_PATH = ROOT / "results" / "phase4" / "final_demo" / "shear_input.npy"

INACTIVE_MESH_CANDIDATES = [
    ROOT / "results" / "phase5" / "week3" / "optimized_meshes" / "inactive_decimated.vtp",
    ROOT / "data" / "meshes" / "platelet" / "inactive.obj",
]

ACTIVATED_MESH_CANDIDATES = [
    ROOT / "results" / "phase5" / "week3" / "optimized_meshes" / "activated_decimated.vtp",
    ROOT / "data" / "meshes" / "platelet" / "activated.obj",
]

OUTPUT_DIR = ROOT / "results" / "phase5" / "presentation" / "usd"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_USDA = OUTPUT_DIR / "phase5_cylinder_flow_activation_120_platelets.usda"
OUTPUT_METADATA_CSV = OUTPUT_DIR / "phase5_cylinder_flow_activation_120_platelets_metadata.csv"
OUTPUT_SUMMARY_MD = OUTPUT_DIR / "phase5_cylinder_flow_activation_120_platelets_summary.md"

STATE_THRESHOLD = 0.50

CYLINDER_LENGTH = 8.0
CYLINDER_RADIUS = 1.05

RENDER_COUNT = 120
VIDEO_FRAMES = 96
FPS = 8

PLATELET_SCALE_BASE = 0.070


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def normalize_mesh(mesh: pv.PolyData) -> pv.PolyData:
    mesh = mesh.extract_surface().triangulate().clean().copy(deep=True)

    center = np.array(mesh.center)
    mesh.translate(-center, inplace=True)

    bounds = np.array(mesh.bounds).reshape(3, 2)
    extents = bounds[:, 1] - bounds[:, 0]
    max_extent = max(float(np.max(extents)), 1e-8)

    mesh.scale(1.0 / max_extent, inplace=True)

    return mesh


def load_meshes():
    inactive_path = first_existing(INACTIVE_MESH_CANDIDATES)
    activated_path = first_existing(ACTIVATED_MESH_CANDIDATES)

    if inactive_path is None:
        raise FileNotFoundError("Could not find inactive platelet mesh.")

    if activated_path is None:
        raise FileNotFoundError("Could not find activated platelet mesh.")

    inactive_mesh = normalize_mesh(pv.read(inactive_path))
    activated_mesh = normalize_mesh(pv.read(activated_path))

    print(f"Inactive mesh loaded : {inactive_path}")
    print(f"Activated mesh loaded: {activated_path}")
    print(f"Inactive mesh cells  : {inactive_mesh.n_cells}")
    print(f"Activated mesh cells : {activated_mesh.n_cells}")

    return inactive_mesh, activated_mesh, inactive_path, activated_path


def load_phase4_arrays():
    require_file(POSITIONS_PATH, "Phase 4 positions")
    require_file(ACTIVATION_PATH, "Phase 4 activation")
    require_file(SHEAR_PATH, "Phase 4 shear")

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
        raise ValueError(f"positions and activation mismatch: {positions.shape[:2]} vs {activation.shape}")

    if positions.shape[:2] != shear.shape:
        raise ValueError(f"positions and shear mismatch: {positions.shape[:2]} vs {shear.shape}")

    activation = np.clip(activation, 0.0, 1.0)
    shear = np.clip(shear, 0.0, None)

    print(f"Positions shape : {positions.shape}")
    print(f"Activation shape: {activation.shape}")
    print(f"Shear shape     : {shear.shape}")

    return positions, activation, shear


def normalize_shear(shear: np.ndarray) -> np.ndarray:
    shear = shear.astype(float)
    min_val = float(np.min(shear))
    max_val = float(np.max(shear))

    if max_val - min_val < 1e-8:
        return np.zeros_like(shear)

    return (shear - min_val) / (max_val - min_val)


def select_120_platelets(activation: np.ndarray, shear: np.ndarray) -> np.ndarray:
    final_activation = activation[-1]
    first_activation = activation[0]
    delta_activation = final_activation - first_activation
    final_shear = shear[-1]

    n_platelets = final_activation.size

    low_activation_ids = np.argsort(final_activation)[:35]
    near_threshold_ids = np.argsort(np.abs(final_activation - STATE_THRESHOLD))[:35]
    high_activation_ids = np.argsort(final_activation)[::-1][:35]

    switching_mask = (first_activation < STATE_THRESHOLD) & (final_activation >= STATE_THRESHOLD)
    switching_ids = np.where(switching_mask)[0]

    if len(switching_ids) > 0:
        switching_ids = switching_ids[np.argsort(delta_activation[switching_ids])[::-1]]
    else:
        switching_ids = np.array([], dtype=int)

    high_shear_ids = np.argsort(final_shear)[::-1][:25]

    selected = np.unique(
        np.concatenate(
            [
                low_activation_ids,
                near_threshold_ids,
                high_activation_ids,
                switching_ids[:35],
                high_shear_ids,
            ]
        )
    )

    if selected.size < RENDER_COUNT:
        remaining = np.setdiff1d(np.arange(n_platelets), selected)
        score = (
            0.40 * final_activation
            + 0.30 * delta_activation
            + 0.30 * final_shear
        )
        remaining_sorted = remaining[np.argsort(score[remaining])[::-1]]
        selected = np.concatenate([selected, remaining_sorted[: RENDER_COUNT - selected.size]])

    if selected.size > RENDER_COUNT:
        score = (
            0.35 * final_activation[selected]
            + 0.30 * delta_activation[selected]
            + 0.20 * final_shear[selected]
            + 0.15 * (1.0 - np.abs(final_activation[selected] - STATE_THRESHOLD))
        )
        selected = selected[np.argsort(score)[::-1][:RENDER_COUNT]]

    selected = selected.astype(int)

    final_selected = final_activation[selected]

    print(f"Selected platelets: {selected.size}")
    print(f"Final inactive in selected set : {np.sum(final_selected < STATE_THRESHOLD)}")
    print(f"Final activated in selected set: {np.sum(final_selected >= STATE_THRESHOLD)}")
    print(f"Switching candidates selected  : {len(np.intersect1d(selected, switching_ids))}")

    return selected


def resample_time_series(arr: np.ndarray, out_frames: int) -> np.ndarray:
    arr = np.asarray(arr)
    in_frames = arr.shape[0]

    old_t = np.linspace(0.0, 1.0, in_frames)
    new_t = np.linspace(0.0, 1.0, out_frames)

    if arr.ndim == 2:
        out = np.empty((out_frames, arr.shape[1]), dtype=float)

        for j in range(arr.shape[1]):
            out[:, j] = np.interp(new_t, old_t, arr[:, j])

        return out

    if arr.ndim == 3:
        out = np.empty((out_frames, arr.shape[1], arr.shape[2]), dtype=float)

        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                out[:, j, k] = np.interp(new_t, old_t, arr[:, j, k])

        return out

    raise ValueError(f"Unsupported shape for resampling: {arr.shape}")


def remap_positions_to_cylinder(positions: np.ndarray) -> np.ndarray:
    p = positions.copy().astype(float)

    x = p[:, :, 0]
    y = p[:, :, 1]
    z = p[:, :, 2]

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_mid = 0.5 * (float(np.min(y)) + float(np.max(y)))
    z_mid = 0.5 * (float(np.min(z)) + float(np.max(z)))

    if x_max - x_min > 1e-8:
        p[:, :, 0] = 0.35 + (x - x_min) / (x_max - x_min) * (CYLINDER_LENGTH - 0.70)
    else:
        p[:, :, 0] = CYLINDER_LENGTH / 2.0

    p[:, :, 1] = y - y_mid
    p[:, :, 2] = z - z_mid

    radial = np.sqrt(p[:, :, 1] ** 2 + p[:, :, 2] ** 2)
    max_radial = max(float(np.max(radial)), 1e-8)

    scale = (CYLINDER_RADIUS * 0.72) / max_radial
    p[:, :, 1] *= scale
    p[:, :, 2] *= scale

    return p


def polydata_faces_to_usd(mesh: pv.PolyData):
    faces = mesh.faces

    face_vertex_counts = []
    face_vertex_indices = []

    index = 0

    while index < len(faces):
        n_vertices = int(faces[index])
        vertex_ids = faces[index + 1 : index + 1 + n_vertices]

        face_vertex_counts.append(n_vertices)
        face_vertex_indices.extend(int(v) for v in vertex_ids)

        index += n_vertices + 1

    return face_vertex_counts, face_vertex_indices


def add_usd_mesh(
    stage: Usd.Stage,
    prim_path: str,
    mesh: pv.PolyData,
    color: tuple[float, float, float],
    opacity: float = 1.0,
) -> UsdGeom.Mesh:
    mesh = mesh.extract_surface().triangulate().clean()

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


def build_vessel_mesh() -> pv.PolyData:
    vessel = pv.Cylinder(
        center=(CYLINDER_LENGTH / 2.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0),
        radius=CYLINDER_RADIUS,
        height=CYLINDER_LENGTH,
        resolution=160,
        capping=False,
    )

    return vessel.extract_surface().triangulate().clean()


def build_centerline_mesh() -> pv.PolyData:
    return pv.Line(
        pointa=(0.0, 0.0, 0.0),
        pointb=(CYLINDER_LENGTH, 0.0, 0.0),
    )


def build_flow_arrow_meshes():
    arrows = []

    x_positions = np.linspace(0.85, CYLINDER_LENGTH - 0.95, 5)
    yz_positions = [
        (0.00, 0.00, 1.00),
        (0.32, 0.00, 0.65),
        (-0.32, 0.00, 0.65),
        (0.00, 0.32, 0.65),
        (0.00, -0.32, 0.65),
    ]

    for x in x_positions:
        for y, z, strength in yz_positions:
            arrow = pv.Arrow(
                start=(x, y, z),
                direction=(1.0, 0.0, 0.0),
                scale=0.32 * strength,
                tip_length=0.22,
                tip_radius=0.045,
                shaft_radius=0.013,
            )
            arrows.append(arrow.extract_surface().triangulate().clean())

    return arrows


def compute_platelet_scale(activation_value: float, shear_value: float) -> float:
    scale = PLATELET_SCALE_BASE * (
        0.90
        + 0.35 * float(activation_value)
        + 0.15 * float(shear_value)
    )

    if activation_value >= STATE_THRESHOLD:
        scale *= 1.10

    return float(scale)


def add_lighting(stage: Usd.Stage):
    key = UsdLux.DistantLight.Define(stage, "/Phase5CylinderActivation/Lighting/KeyLight")
    key.CreateIntensityAttr(700.0)
    key.CreateAngleAttr(0.35)

    key_xform = UsdGeom.Xformable(key.GetPrim())
    key_xform.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 35.0))

    fill = UsdLux.DistantLight.Define(stage, "/Phase5CylinderActivation/Lighting/FillLight")
    fill.CreateIntensityAttr(120.0)
    fill.CreateAngleAttr(0.70)

    fill_xform = UsdGeom.Xformable(fill.GetPrim())
    fill_xform.AddRotateXYZOp().Set(Gf.Vec3f(25.0, 0.0, -35.0))


def add_camera(stage: Usd.Stage):
    camera = UsdGeom.Camera.Define(stage, "/Phase5CylinderActivation/Camera")

    camera.CreateFocalLengthAttr(35.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))

    camera_xform = UsdGeom.Xformable(camera.GetPrim())
    camera_xform.AddTranslateOp().Set(Gf.Vec3d(4.0, -5.8, 2.7))
    camera_xform.AddRotateXYZOp().Set(Gf.Vec3f(62.0, 0.0, 35.0))


def create_point_instancer(
    stage: Usd.Stage,
    inactive_mesh: pv.PolyData,
    activated_mesh: pv.PolyData,
    selected_ids: np.ndarray,
    positions_video: np.ndarray,
    activation_video: np.ndarray,
    shear_video: np.ndarray,
):
    instancer_path = "/Phase5CylinderActivation/PlateletInstancer"
    instancer = UsdGeom.PointInstancer.Define(stage, instancer_path)

    inactive_prototype_path = f"{instancer_path}/Prototypes/InactivePlatelet"
    activated_prototype_path = f"{instancer_path}/Prototypes/ActivatedPlatelet"

    add_usd_mesh(
        stage=stage,
        prim_path=inactive_prototype_path,
        mesh=inactive_mesh,
        color=(0.05, 0.18, 0.95),
        opacity=1.0,
    )

    add_usd_mesh(
        stage=stage,
        prim_path=activated_prototype_path,
        mesh=activated_mesh,
        color=(0.95, 0.08, 0.07),
        opacity=1.0,
    )

    instancer.GetPrototypesRel().SetTargets(
        [
            Sdf.Path(inactive_prototype_path),
            Sdf.Path(activated_prototype_path),
        ]
    )

    ids_attr = instancer.CreateIdsAttr()
    positions_attr = instancer.CreatePositionsAttr()
    scales_attr = instancer.CreateScalesAttr()
    proto_indices_attr = instancer.CreateProtoIndicesAttr()

    activation_attr = instancer.GetPrim().CreateAttribute(
        "phase5:activation",
        Sdf.ValueTypeNames.FloatArray,
        custom=True,
    )

    shear_attr = instancer.GetPrim().CreateAttribute(
        "phase5:shear",
        Sdf.ValueTypeNames.FloatArray,
        custom=True,
    )

    state_attr = instancer.GetPrim().CreateAttribute(
        "phase5:stateIndex",
        Sdf.ValueTypeNames.IntArray,
        custom=True,
    )

    ids_attr.Set([int(x) for x in selected_ids])

    metadata_rows = []

    for frame in range(VIDEO_FRAMES):
        p_frame = positions_video[frame]
        a_frame = activation_video[frame]
        s_frame = shear_video[frame]

        usd_positions = []
        usd_scales = []
        proto_indices = []
        activation_values = []
        shear_values = []
        state_values = []

        inactive_count = 0
        activated_count = 0
        near_threshold_count = 0

        for local_id, platelet_id in enumerate(selected_ids):
            position = p_frame[local_id]
            activation_value = float(a_frame[local_id])
            shear_value = float(s_frame[local_id])

            state_index = 0 if activation_value < STATE_THRESHOLD else 1

            if state_index == 0:
                inactive_count += 1
            else:
                activated_count += 1

            if abs(activation_value - STATE_THRESHOLD) < 0.055:
                near_threshold_count += 1

            scale = compute_platelet_scale(activation_value, shear_value)

            usd_positions.append(
                Gf.Vec3f(
                    float(position[0]),
                    float(position[1]),
                    float(position[2]),
                )
            )

            usd_scales.append(
                Gf.Vec3f(float(scale), float(scale), float(scale))
            )

            proto_indices.append(int(state_index))
            activation_values.append(float(activation_value))
            shear_values.append(float(shear_value))
            state_values.append(int(state_index))

            metadata_rows.append(
                {
                    "frame": frame,
                    "local_id": local_id,
                    "source_platelet_id": int(platelet_id),
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "z": float(position[2]),
                    "activation": activation_value,
                    "shear": shear_value,
                    "state": "inactive" if state_index == 0 else "activated",
                    "proto_index": state_index,
                    "scale": scale,
                }
            )

        time_code = Usd.TimeCode(frame)

        positions_attr.Set(usd_positions, time_code)
        scales_attr.Set(usd_scales, time_code)
        proto_indices_attr.Set(proto_indices, time_code)

        activation_attr.Set(activation_values, time_code)
        shear_attr.Set(shear_values, time_code)
        state_attr.Set(state_values, time_code)

        if (frame + 1) % 10 == 0 or frame == VIDEO_FRAMES - 1:
            print(
                f"USD frame {frame + 1}/{VIDEO_FRAMES} | "
                f"inactive={inactive_count} | activated={activated_count} | near-threshold={near_threshold_count}"
            )

    return metadata_rows


def write_metadata_csv(rows):
    with OUTPUT_METADATA_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "frame",
                "local_id",
                "source_platelet_id",
                "x",
                "y",
                "z",
                "activation",
                "shear",
                "state",
                "proto_index",
                "scale",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    inactive_path,
    activated_path,
    selected_ids,
    metadata_rows,
):
    final_rows = [row for row in metadata_rows if row["frame"] == VIDEO_FRAMES - 1]
    final_inactive = sum(1 for row in final_rows if row["state"] == "inactive")
    final_activated = sum(1 for row in final_rows if row["state"] == "activated")

    lines = []

    lines.append("# Phase 5 Cylinder Flow Activation USD Export")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This export converts the final Phase 5 120-platelet cylindrical activation video scene "
        "into an Omniverse/USD-ready animated USDA file."
    )
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append(f"- USD scene: `{OUTPUT_USDA}`")
    lines.append(f"- Metadata CSV: `{OUTPUT_METADATA_CSV}`")
    lines.append(f"- Summary: `{OUTPUT_SUMMARY_MD}`")
    lines.append("")
    lines.append("## Scene Contents")
    lines.append("")
    lines.append(f"- Rendered platelet instances: {len(selected_ids)}")
    lines.append(f"- Animation frames: {VIDEO_FRAMES}")
    lines.append(f"- Frames per second metadata: {FPS}")
    lines.append(f"- Activation threshold: {STATE_THRESHOLD}")
    lines.append(f"- Final inactive platelet instances: {final_inactive}")
    lines.append(f"- Final activated platelet instances: {final_activated}")
    lines.append("")
    lines.append("## USD Structure")
    lines.append("")
    lines.append("- Root prim: `/Phase5CylinderActivation`")
    lines.append("- Platelet animation: `/Phase5CylinderActivation/PlateletInstancer`")
    lines.append("- Prototype 0: inactive platelet mesh")
    lines.append("- Prototype 1: activated platelet mesh")
    lines.append("- Time-sampled data: positions, scales, prototype indices, activation, shear")
    lines.append("")
    lines.append("## Mesh Sources")
    lines.append("")
    lines.append(f"- Inactive mesh: `{inactive_path}`")
    lines.append(f"- Activated mesh: `{activated_path}`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Inactive platelets are represented by the blue inactive prototype. Activated platelets are "
        "represented by the red activated prototype. The prototype index changes over time according "
        "to the activation threshold."
    )
    lines.append("")
    lines.append("## Thesis Wording")
    lines.append("")
    lines.append(
        "This scene is USD-ready and Omniverse-compatible. Direct Omniverse rendering should only be "
        "claimed after the USDA file has been opened and visually inspected inside Omniverse."
    )

    OUTPUT_SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    print("Creating Phase 5 cylindrical activation USD export...")

    positions, activation, shear = load_phase4_arrays()
    shear_norm = normalize_shear(shear)

    inactive_mesh, activated_mesh, inactive_path, activated_path = load_meshes()

    selected_ids = select_120_platelets(activation, shear_norm)

    positions_sel = positions[:, selected_ids, :]
    activation_sel = activation[:, selected_ids]
    shear_sel = shear_norm[:, selected_ids]

    positions_sel = remap_positions_to_cylinder(positions_sel)

    positions_video = resample_time_series(positions_sel, VIDEO_FRAMES)
    activation_video = resample_time_series(activation_sel, VIDEO_FRAMES)
    shear_video = resample_time_series(shear_sel, VIDEO_FRAMES)

    if OUTPUT_USDA.exists():
        OUTPUT_USDA.unlink()

    stage = Usd.Stage.CreateNew(str(OUTPUT_USDA))

    root = UsdGeom.Xform.Define(stage, "/Phase5CylinderActivation")
    stage.SetDefaultPrim(root.GetPrim())

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(VIDEO_FRAMES - 1)
    stage.SetFramesPerSecond(FPS)
    stage.SetTimeCodesPerSecond(FPS)

    vessel_mesh = build_vessel_mesh()
    add_usd_mesh(
        stage=stage,
        prim_path="/Phase5CylinderActivation/Vessel/CylinderProxy",
        mesh=vessel_mesh,
        color=(0.86, 0.88, 0.90),
        opacity=0.18,
    )

    centerline_mesh = build_centerline_mesh()
    add_usd_mesh(
        stage=stage,
        prim_path="/Phase5CylinderActivation/Vessel/Centerline",
        mesh=centerline_mesh,
        color=(0.65, 0.65, 0.65),
        opacity=1.0,
    )

    for i, arrow_mesh in enumerate(build_flow_arrow_meshes()):
        add_usd_mesh(
            stage=stage,
            prim_path=f"/Phase5CylinderActivation/FlowArrows/Arrow_{i:02d}",
            mesh=arrow_mesh,
            color=(0.26, 0.52, 0.82),
            opacity=0.45,
        )

    add_lighting(stage)
    add_camera(stage)

    metadata_rows = create_point_instancer(
        stage=stage,
        inactive_mesh=inactive_mesh,
        activated_mesh=activated_mesh,
        selected_ids=selected_ids,
        positions_video=positions_video,
        activation_video=activation_video,
        shear_video=shear_video,
    )

    stage.GetRootLayer().Save()

    write_metadata_csv(metadata_rows)
    write_summary(
        inactive_path=inactive_path,
        activated_path=activated_path,
        selected_ids=selected_ids,
        metadata_rows=metadata_rows,
    )

    print()
    print("Phase 5 cylindrical activation USD export complete.")
    print(f"USD scene:    {OUTPUT_USDA}")
    print(f"Metadata CSV: {OUTPUT_METADATA_CSV}")
    print(f"Summary:      {OUTPUT_SUMMARY_MD}")
    print()
    print("This USDA file is ready for future Omniverse import/inspection.")


if __name__ == "__main__":
    main()