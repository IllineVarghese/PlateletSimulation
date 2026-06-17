from pathlib import Path
import csv
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from pxr import Usd, UsdGeom, Sdf, Gf, UsdLux

from mesh_utils import center_and_scale_mesh


PROJECT_ROOT = Path(__file__).resolve().parents[2]

POSITIONS_PATH = PROJECT_ROOT / "results" / "phase4" / "final_demo" / "positions.npy"
ACTIVATION_PATH = PROJECT_ROOT / "results" / "phase4" / "final_demo" / "activation.npy"
SHEAR_PATH = PROJECT_ROOT / "results" / "phase4" / "final_demo" / "shear_input.npy"

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
    / "animated_usd_scene"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_USDA = OUTPUT_DIR / "phase5_platelet_animated_scene.usda"
OUTPUT_METADATA_CSV = OUTPUT_DIR / "platelet_animation_metadata.csv"
OUTPUT_SUMMARY_MD = OUTPUT_DIR / "animated_usd_export_summary.md"
OUTPUT_PREVIEW_PNG = OUTPUT_DIR / "phase5_animated_usd_final_preview.png"
OUTPUT_COUNTS_PNG = OUTPUT_DIR / "activation_state_counts_over_time.png"

MAX_EXPORTED_PLATELETS = 120
ACTIVATION_THRESHOLD = 0.50
FPS = 6


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
    value = float(np.clip(activation, 0.0, 1.0))

    if value < 0.5:
        t = value / 0.5
        return 0.12 + 0.65 * t, 0.34 + 0.45 * t, 0.95

    t = (value - 0.5) / 0.5
    return 0.95, 0.78 - 0.58 * t, 0.82 - 0.70 * t


def choose_export_indices(
    activation: np.ndarray,
    shear: np.ndarray,
    max_count: int,
) -> np.ndarray:
    final_activation = activation[-1]
    final_shear = shear[-1]
    delta_activation = activation[-1] - activation[0]

    n_platelets = final_activation.size

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
            [
                dynamic_idx,
                high_activation_idx,
                high_shear_idx,
                low_activation_idx,
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


def polydata_faces_to_usd(mesh: pv.PolyData) -> tuple[list[int], list[int]]:
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
    light = UsdLux.DistantLight.Define(stage, "/Phase5AnimatedScene/Lighting/KeyLight")
    light.CreateIntensityAttr(600.0)
    light.CreateAngleAttr(0.35)

    xform = UsdGeom.Xformable(light.GetPrim())
    xform.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 35.0))


def add_scene_camera(stage: Usd.Stage) -> None:
    camera = UsdGeom.Camera.Define(stage, "/Phase5AnimatedScene/Camera")
    camera.CreateFocalLengthAttr(35.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))

    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(4.0, -6.2, 3.0))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(62.0, 0.0, 35.0))


def compute_scale(activation_value: float, shear_value: float) -> float:
    state_bonus = 0.012 if activation_value >= ACTIVATION_THRESHOLD else 0.0
    return 0.065 + 0.015 * activation_value + 0.008 * shear_value + state_bonus


def create_animated_point_instancer(
    stage: Usd.Stage,
    positions: np.ndarray,
    activation: np.ndarray,
    shear: np.ndarray,
    selected_indices: np.ndarray,
) -> tuple[list[dict], list[dict]]:
    instancer_path = "/Phase5AnimatedScene/PlateletInstancer"

    instancer = UsdGeom.PointInstancer.Define(stage, instancer_path)

    inactive_prototype_path = f"{instancer_path}/Prototypes/InactivePlatelet"
    activated_prototype_path = f"{instancer_path}/Prototypes/ActivatedPlatelet"

    inactive_mesh, activated_mesh = load_decimated_meshes()

    add_usd_mesh(
        stage=stage,
        prim_path=inactive_prototype_path,
        mesh=inactive_mesh,
        color=(0.20, 0.45, 0.95),
        opacity=1.0,
    )

    add_usd_mesh(
        stage=stage,
        prim_path=activated_prototype_path,
        mesh=activated_mesh,
        color=(0.95, 0.20, 0.12),
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

    ids = [int(idx) for idx in selected_indices]
    ids_attr.Set(ids)

    metadata_rows = []
    count_rows = []

    n_frames = positions.shape[0]

    for frame_idx in range(n_frames):
        frame_positions = positions[frame_idx]
        frame_activation = activation[frame_idx]
        frame_shear = shear[frame_idx]

        usd_positions = []
        usd_scales = []
        proto_indices = []

        inactive_count = 0
        activated_count = 0

        for local_id, source_idx in enumerate(selected_indices):
            pos = frame_positions[source_idx]
            act = float(frame_activation[source_idx])
            shr = float(frame_shear[source_idx])

            state = "activated" if act >= ACTIVATION_THRESHOLD else "inactive"
            proto_index = 1 if state == "activated" else 0

            if state == "activated":
                activated_count += 1
            else:
                inactive_count += 1

            scale = compute_scale(act, shr)
            color = activation_to_color(act)

            usd_positions.append(
                Gf.Vec3f(float(pos[0]), float(pos[1]), float(pos[2]))
            )

            usd_scales.append(
                Gf.Vec3f(float(scale), float(scale), float(scale))
            )

            proto_indices.append(int(proto_index))

            metadata_rows.append(
                {
                    "frame_index": frame_idx,
                    "local_export_id": local_id,
                    "source_platelet_index": int(source_idx),
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "z": float(pos[2]),
                    "activation": act,
                    "shear": shr,
                    "state": state,
                    "proto_index": proto_index,
                    "scale": scale,
                    "color_r": color[0],
                    "color_g": color[1],
                    "color_b": color[2],
                }
            )

        time_code = Usd.TimeCode(frame_idx)

        positions_attr.Set(usd_positions, time_code)
        scales_attr.Set(usd_scales, time_code)
        proto_indices_attr.Set(proto_indices, time_code)

        count_rows.append(
            {
                "frame_index": frame_idx,
                "inactive_count": inactive_count,
                "activated_count": activated_count,
                "mean_activation": float(np.mean(frame_activation[selected_indices])),
                "mean_shear": float(np.mean(frame_shear[selected_indices])),
            }
        )

        print(
            f"USD time sample frame {frame_idx + 1:02d}/{n_frames} | "
            f"inactive={inactive_count} | activated={activated_count}"
        )

    return metadata_rows, count_rows


def save_metadata_csv(rows: list[dict]) -> None:
    with OUTPUT_METADATA_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "frame_index",
                "local_export_id",
                "source_platelet_index",
                "x",
                "y",
                "z",
                "activation",
                "shear",
                "state",
                "proto_index",
                "scale",
                "color_r",
                "color_g",
                "color_b",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def save_counts_plot(count_rows: list[dict]) -> None:
    frames = [row["frame_index"] for row in count_rows]
    inactive_counts = [row["inactive_count"] for row in count_rows]
    activated_counts = [row["activated_count"] for row in count_rows]

    plt.figure(figsize=(8, 5))
    plt.plot(frames, inactive_counts, marker="o", label="Inactive")
    plt.plot(frames, activated_counts, marker="o", label="Activated")
    plt.xlabel("Frame")
    plt.ylabel("Exported platelet count")
    plt.title("Phase 5 Week 4: Activation state counts in animated USD export")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_COUNTS_PNG, dpi=200)
    plt.close()


def save_summary(
    n_frames: int,
    n_platelets: int,
    selected_count: int,
    inactive_mesh: pv.PolyData,
    activated_mesh: pv.PolyData,
    count_rows: list[dict],
) -> None:
    final_counts = count_rows[-1]
    first_counts = count_rows[0]

    lines = []

    lines.append("# Phase 5 Week 4 Day 3: Animated USD Scene Export")
    lines.append("")
    lines.append("## Export Status")
    lines.append("")
    lines.append("An animation-ready USD scene was exported using `usd-core` and `UsdGeom.PointInstancer`.")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append(f"- Animated USD scene: `{OUTPUT_USDA}`")
    lines.append(f"- Metadata CSV: `{OUTPUT_METADATA_CSV}`")
    lines.append(f"- Final-frame preview: `{OUTPUT_PREVIEW_PNG}`")
    lines.append(f"- Activation count plot: `{OUTPUT_COUNTS_PNG}`")
    lines.append("")
    lines.append("## Scene Contents")
    lines.append("")
    lines.append(f"- Source Phase 4 frames: {n_frames}")
    lines.append(f"- Source platelet count: {n_platelets}")
    lines.append(f"- Exported platelet count per frame: {selected_count}")
    lines.append(f"- Activation threshold: {ACTIVATION_THRESHOLD:.2f}")
    lines.append(f"- Frames per second metadata: {FPS}")
    lines.append("")
    lines.append("## First and Final Frame State Counts")
    lines.append("")
    lines.append(
        f"- First frame: inactive={first_counts['inactive_count']}, "
        f"activated={first_counts['activated_count']}"
    )
    lines.append(
        f"- Final frame: inactive={final_counts['inactive_count']}, "
        f"activated={final_counts['activated_count']}"
    )
    lines.append("")
    lines.append("## Mesh Prototype Complexity")
    lines.append("")
    lines.append(f"- Inactive prototype points: {inactive_mesh.n_points}")
    lines.append(f"- Inactive prototype cells: {inactive_mesh.n_cells}")
    lines.append(f"- Activated prototype points: {activated_mesh.n_points}")
    lines.append(f"- Activated prototype cells: {activated_mesh.n_cells}")
    lines.append("")
    lines.append("## Technical Design")
    lines.append("")
    lines.append(
        "The animated USD scene uses two platelet mesh prototypes, one inactive and one activated. "
        "The `PointInstancer` stores time-sampled positions, scales, and prototype indices for each selected platelet. "
        "The prototype index changes when a platelet crosses the activation threshold."
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "The USD file is animation-ready and can be opened in a USD-compatible viewer or Omniverse environment. "
        "Per-frame activation, shear, state, color, and scale values are also saved in the metadata CSV."
    )

    OUTPUT_SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def render_final_preview(
    positions: np.ndarray,
    activation: np.ndarray,
    shear: np.ndarray,
    selected_indices: np.ndarray,
    inactive_mesh: pv.PolyData,
    activated_mesh: pv.PolyData,
) -> None:
    vessel = make_vessel_proxy(positions)

    final_positions = positions[-1]
    final_activation = activation[-1]
    final_shear = shear[-1]

    inactive_count = 0
    activated_count = 0

    plotter = pv.Plotter(off_screen=True, window_size=(1600, 900))
    plotter.set_background("white")

    plotter.add_mesh(
        vessel,
        color=(0.86, 0.90, 0.92),
        opacity=0.10,
        smooth_shading=True,
        show_edges=False,
    )

    for idx in selected_indices:
        pos = final_positions[idx]
        act = float(final_activation[idx])
        shr = float(final_shear[idx])

        if act >= ACTIVATION_THRESHOLD:
            base_mesh = activated_mesh
            activated_count += 1
        else:
            base_mesh = inactive_mesh
            inactive_count += 1

        mesh = base_mesh.copy()
        scale = compute_scale(act, shr)
        mesh.scale(scale, inplace=True)
        mesh.rotate_x((int(idx) * 7.0) % 360, inplace=True)
        mesh.rotate_y((int(idx) * 13.0) % 360, inplace=True)
        mesh.rotate_z((int(idx) * 17.0) % 360, inplace=True)
        mesh.translate(tuple(pos), inplace=True)

        plotter.add_mesh(
            mesh,
            color=activation_to_color(act),
            smooth_shading=True,
            show_edges=False,
            opacity=0.97,
            specular=0.20,
        )

    plotter.add_text(
        "Phase 5 Week 4: Animated USD Export Final-Frame Preview",
        position=(310, 850),
        font_size=17,
        color="black",
    )

    plotter.add_text(
        f"Exported platelets={len(selected_indices)} | inactive={inactive_count} | activated={activated_count}",
        position=(440, 820),
        font_size=12,
        color="black",
    )

    plotter.add_text(
        "Animated USD uses PointInstancer with time-sampled positions, scales, and inactive/activated prototype indices",
        position=(250, 792),
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
    print("Phase 5 Week 4 Day 3: Animated USD scene export")

    positions, activation, shear = load_phase4_data()
    inactive_mesh, activated_mesh = load_decimated_meshes()

    n_frames, n_platelets, _ = positions.shape

    selected_indices = choose_export_indices(
        activation=activation,
        shear=shear,
        max_count=MAX_EXPORTED_PLATELETS,
    )

    print(f"positions shape:  {positions.shape}")
    print(f"activation shape: {activation.shape}")
    print(f"shear shape:      {shear.shape}")
    print(f"selected count:   {len(selected_indices)}")
    print(f"inactive mesh:    points={inactive_mesh.n_points}, cells={inactive_mesh.n_cells}")
    print(f"activated mesh:   points={activated_mesh.n_points}, cells={activated_mesh.n_cells}")

    if OUTPUT_USDA.exists():
        OUTPUT_USDA.unlink()

    stage = Usd.Stage.CreateNew(str(OUTPUT_USDA))

    root = UsdGeom.Xform.Define(stage, "/Phase5AnimatedScene")
    stage.SetDefaultPrim(root.GetPrim())

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(n_frames - 1)
    stage.SetTimeCodesPerSecond(FPS)
    stage.SetFramesPerSecond(FPS)

    vessel = make_vessel_proxy(positions)

    add_usd_mesh(
        stage=stage,
        prim_path="/Phase5AnimatedScene/Vessel/VesselProxy",
        mesh=vessel,
        color=(0.86, 0.90, 0.92),
        opacity=0.20,
    )

    add_scene_lighting(stage)
    add_scene_camera(stage)

    metadata_rows, count_rows = create_animated_point_instancer(
        stage=stage,
        positions=positions,
        activation=activation,
        shear=shear,
        selected_indices=selected_indices,
    )

    save_metadata_csv(metadata_rows)
    save_counts_plot(count_rows)

    stage.GetRootLayer().Save()

    render_final_preview(
        positions=positions,
        activation=activation,
        shear=shear,
        selected_indices=selected_indices,
        inactive_mesh=inactive_mesh,
        activated_mesh=activated_mesh,
    )

    save_summary(
        n_frames=n_frames,
        n_platelets=n_platelets,
        selected_count=len(selected_indices),
        inactive_mesh=inactive_mesh,
        activated_mesh=activated_mesh,
        count_rows=count_rows,
    )

    print("\nAnimated USD export complete.")
    print(f"USD scene:       {OUTPUT_USDA}")
    print(f"Metadata CSV:    {OUTPUT_METADATA_CSV}")
    print(f"Summary:         {OUTPUT_SUMMARY_MD}")
    print(f"Preview image:   {OUTPUT_PREVIEW_PNG}")
    print(f"Counts plot:     {OUTPUT_COUNTS_PNG}")

    print("\nWeek 4 Day 3 animated USD scene export complete.")


if __name__ == "__main__":
    main()