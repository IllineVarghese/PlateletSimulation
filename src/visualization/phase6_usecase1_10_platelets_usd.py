"""
Phase 6 - Use Case 1
10 original platelet meshes inside a 3D vessel as animated USD scene.

Goal:
- Use real Phase 4 data
- Select 10 representative platelets
- Use original inactive / activated platelet meshes
- Animate platelet movement over time
- Switch mesh visibility based on activation threshold
- Export USD for Omniverse inspection / rendering

Outputs:
results/phase6/usecase1_actual_mesh/
    usecase1_10_platelets_original_mesh.usd
    selected_platelets_usecase1.csv
    usecase1_10_platelets_summary.json
    README_usecase1_actual_mesh.md
"""

from pathlib import Path
import csv
import json
import math
import numpy as np

from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, UsdLux


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(".").resolve()

INACTIVE_MESH_PATH = PROJECT_ROOT / "data/meshes/platelet/inactive.obj"

# Use recovered mesh if present, otherwise fallback
ACTIVATED_MESH_PATH = PROJECT_ROOT / "data/meshes/platelet/activated_usecase1.obj"
if not ACTIVATED_MESH_PATH.exists():
    ACTIVATED_MESH_PATH = PROJECT_ROOT / "data/meshes/platelet/activated.obj"

POSITIONS_PATH = PROJECT_ROOT / "results/phase4/final_demo/positions.npy"
ACTIVATION_PATH = PROJECT_ROOT / "results/phase4/final_demo/activation.npy"
SHEAR_PATH = PROJECT_ROOT / "results/phase4/final_demo/shear_input.npy"

OUTPUT_DIR = PROJECT_ROOT / "results/phase6/usecase1_actual_mesh"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

USD_PATH = OUTPUT_DIR / "usecase1_10_platelets_original_mesh.usd"
CSV_PATH = OUTPUT_DIR / "selected_platelets_usecase1.csv"
JSON_PATH = OUTPUT_DIR / "usecase1_10_platelets_summary.json"
README_PATH = OUTPUT_DIR / "README_usecase1_actual_mesh.md"

N_SELECTED = 10
ACTIVATION_THRESHOLD = 0.5
NEAR_THRESHOLD_BAND = 0.12

# Optional manual override if you want specific platelet indices
# Example:
# MANUAL_SELECTED_INDICES = [42, 137, 403, 255, 78, 274, 19, 325, 326, 327]
MANUAL_SELECTED_INDICES = None

# USD / animation settings
FPS = 6  # playback speed in Omniverse
USD_UP_AXIS = UsdGeom.Tokens.z


# ============================================================
# Utilities
# ============================================================

def load_required_array(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing required {name} file: {path}")
    arr = np.load(path)
    print(f"[OK] Loaded {name}: shape={arr.shape}, dtype={arr.dtype}")
    return arr


def ensure_positions(arr):
    arr = np.asarray(arr)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"positions must have shape (T, N, 3), got {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("positions contains non-finite values.")
    return arr


def ensure_scalar_timeseries(arr, name):
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape (T, N), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def classify_activation(a, threshold=ACTIVATION_THRESHOLD, band=NEAR_THRESHOLD_BAND):
    if a < threshold - band:
        return "inactive"
    elif a > threshold + band:
        return "activated"
    else:
        return "near-threshold"


# ============================================================
# OBJ loader
# ============================================================

def load_obj_mesh(obj_path: Path):
    """
    Minimal OBJ loader for vertices and triangulated faces.
    Supports:
      v x y z
      f i j k
      f i/... j/... k/...
      polygon fan triangulation
    """
    if not obj_path.exists():
        raise FileNotFoundError(f"OBJ not found: {obj_path}")

    vertices = []
    faces = []

    with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

            elif line.startswith("f "):
                raw = line.split()[1:]
                idxs = []
                for token in raw:
                    idx = token.split("/")[0]
                    if idx:
                        idxs.append(int(idx) - 1)

                if len(idxs) >= 3:
                    for i in range(1, len(idxs) - 1):
                        faces.append([idxs[0], idxs[i], idxs[i + 1]])

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)

    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError(f"Could not load valid mesh from {obj_path}")

    print(f"[OK] Mesh loaded: {obj_path.name} | vertices={len(vertices)} | faces={len(faces)}")
    return vertices, faces


def normalize_mesh(vertices):
    """
    Center and normalize mesh to a consistent local size.
    """
    v = vertices.copy()
    center = v.mean(axis=0)
    v -= center

    radius = np.max(np.linalg.norm(v, axis=1))
    if radius > 0:
        v /= radius

    return v


# ============================================================
# Platelet selection
# ============================================================

def select_representative_platelets(activation_ts, n_selected=10, threshold=ACTIVATION_THRESHOLD):
    """
    Select 10 representative platelets based on final-frame activation:
    - 3 low activation
    - 3 near threshold
    - 4 high activation

    To make animation more interesting, prioritizes platelets with larger activation variation over time.
    """
    final_activation = activation_ts[-1]
    n = len(final_activation)

    variability = activation_ts.max(axis=0) - activation_ts.min(axis=0)

    all_indices = np.arange(n)

    low_mask = final_activation < (threshold - NEAR_THRESHOLD_BAND)
    near_mask = np.abs(final_activation - threshold) <= NEAR_THRESHOLD_BAND
    high_mask = final_activation > (threshold + NEAR_THRESHOLD_BAND)

    low_candidates = all_indices[low_mask]
    near_candidates = all_indices[near_mask]
    high_candidates = all_indices[high_mask]

    # Sort each group by variability descending, then by closeness to group condition
    low_candidates = sorted(
        low_candidates,
        key=lambda i: (-variability[i], final_activation[i])
    )
    near_candidates = sorted(
        near_candidates,
        key=lambda i: (-variability[i], abs(final_activation[i] - threshold))
    )
    high_candidates = sorted(
        high_candidates,
        key=lambda i: (-variability[i], -final_activation[i])
    )

    selected = []

    def add_from_group(group, k):
        for idx in group:
            if idx not in selected:
                selected.append(int(idx))
            if len(selected) >= k:
                break

    # target: 3 low, 3 near, 4 high
    add_from_group(low_candidates, 3)
    add_from_group(near_candidates, 6)
    add_from_group(high_candidates, 10)

    # fallback if one group is too small
    if len(selected) < n_selected:
        remaining = [i for i in all_indices if i not in selected]
        remaining = sorted(
            remaining,
            key=lambda i: (-variability[i], abs(final_activation[i] - threshold))
        )
        for idx in remaining:
            selected.append(int(idx))
            if len(selected) == n_selected:
                break

    return np.asarray(selected[:n_selected], dtype=int)


# ============================================================
# Vessel geometry
# ============================================================

def principal_axis(points):
    """
    PCA-based principal axis for a set of 3D points.
    """
    pts = points.reshape(-1, 3)
    center = pts.mean(axis=0)

    centered = pts - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    axis = eigvecs[:, np.argmax(eigvals)]
    axis = axis / np.linalg.norm(axis)

    projections = centered @ axis
    p0 = center + axis * projections.min()
    p1 = center + axis * projections.max()

    return center, axis, p0, p1, projections.min(), projections.max()


def orthonormal_basis(axis):
    axis = axis / np.linalg.norm(axis)

    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(axis, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    n1 = np.cross(axis, ref)
    n1 /= np.linalg.norm(n1)

    n2 = np.cross(axis, n1)
    n2 /= np.linalg.norm(n2)

    return n1, n2


def build_tube_mesh(p0, p1, radius, segments=32):
    """
    Build a hollow cylindrical vessel mesh (tube wall only) between p0 and p1.
    """
    axis = p1 - p0
    length = np.linalg.norm(axis)
    if length <= 1e-8:
        raise ValueError("Tube length too small.")

    axis = axis / length
    n1, n2 = orthonormal_basis(axis)

    points = []
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        circle_dir = math.cos(theta) * n1 + math.sin(theta) * n2
        points.append(p0 + radius * circle_dir)
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        circle_dir = math.cos(theta) * n1 + math.sin(theta) * n2
        points.append(p1 + radius * circle_dir)

    points = np.asarray(points, dtype=np.float32)

    faces = []
    for i in range(segments):
        j = (i + 1) % segments

        a0 = i
        a1 = j
        b0 = i + segments
        b1 = j + segments

        faces.append([a0, b0, a1])
        faces.append([a1, b0, b1])

    faces = np.asarray(faces, dtype=np.int32)
    return points, faces


# ============================================================
# USD helpers
# ============================================================

def define_mesh(stage, path, vertices, faces):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in vertices])
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([int(i) for tri in faces for i in tri])
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    return mesh


def create_preview_material(stage, path, color, opacity=1.0, roughness=0.4, metallic=0.0):
    material = UsdShade.Material.Define(stage, path)

    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
    )
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))

    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def bind_material(prim, material):
    UsdShade.MaterialBindingAPI(prim).Bind(material)


def set_visibility(imageable, visible, time_code):
    token = UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible
    imageable.GetVisibilityAttr().Set(token, time_code)


# ============================================================
# Main exporter
# ============================================================

def main():
    print("\n=== Phase 6 / Use Case 1 / USD export ===\n")

    positions = ensure_positions(load_required_array(POSITIONS_PATH, "positions"))
    activation = ensure_scalar_timeseries(load_required_array(ACTIVATION_PATH, "activation"), "activation")
    shear = ensure_scalar_timeseries(load_required_array(SHEAR_PATH, "shear_input"), "shear_input")

    T = min(len(positions), len(activation), len(shear))
    N = min(positions.shape[1], activation.shape[1], shear.shape[1])

    positions = positions[:T, :N, :]
    activation = activation[:T, :N]
    shear = shear[:T, :N]

    print(f"[OK] Using T={T} frames and N={N} platelets")

    if MANUAL_SELECTED_INDICES is not None:
        selected_indices = np.asarray(MANUAL_SELECTED_INDICES, dtype=int)
        print(f"[OK] Using manual platelet selection: {selected_indices.tolist()}")
    else:
        selected_indices = select_representative_platelets(
            activation,
            n_selected=min(N_SELECTED, N),
            threshold=ACTIVATION_THRESHOLD,
        )
        print(f"[OK] Auto-selected representative platelets: {selected_indices.tolist()}")

    inactive_vertices, inactive_faces = load_obj_mesh(INACTIVE_MESH_PATH)
    activated_vertices, activated_faces = load_obj_mesh(ACTIVATED_MESH_PATH)

    inactive_vertices = normalize_mesh(inactive_vertices)
    activated_vertices = normalize_mesh(activated_vertices)

    selected_positions = positions[:, selected_indices, :]
    selected_activation = activation[:, selected_indices]
    selected_shear = shear[:, selected_indices]

    # Scene scale estimation
    scene_points = selected_positions.reshape(-1, 3)
    center, axis, vessel_start, vessel_end, _, _ = principal_axis(scene_points)

    # Vessel radius estimated from distances to principal axis
    diffs = scene_points - center
    proj = np.outer((diffs @ axis), axis)
    radial = diffs - proj
    radial_dists = np.linalg.norm(radial, axis=1)
    vessel_radius = max(0.15, radial_dists.max() * 1.35)

    vessel_vertices, vessel_faces = build_tube_mesh(
        vessel_start,
        vessel_end,
        radius=vessel_radius,
        segments=40,
    )

    vessel_length = np.linalg.norm(vessel_end - vessel_start)
    platelet_scale = max(0.04 * vessel_radius, 0.02 * vessel_length)

    print(f"[OK] Vessel radius estimate: {vessel_radius:.4f}")
    print(f"[OK] Platelet display scale: {platelet_scale:.4f}")

    # --------------------------------------------------------
    # Save CSV selection table
    # --------------------------------------------------------
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "selected_order",
                "platelet_index",
                "final_activation",
                "final_shear",
                "final_state",
                "mesh_low_state",
                "mesh_high_state",
            ]
        )
        for order, idx in enumerate(selected_indices):
            final_a = float(selected_activation[-1, order])
            final_s = float(selected_shear[-1, order])
            final_state = classify_activation(final_a)
            writer.writerow(
                [
                    order,
                    int(idx),
                    final_a,
                    final_s,
                    final_state,
                    INACTIVE_MESH_PATH.name,
                    ACTIVATED_MESH_PATH.name,
                ]
            )
    print(f"[OK] Saved selection CSV: {CSV_PATH}")

    # --------------------------------------------------------
    # Build USD stage
    # --------------------------------------------------------
    stage = Usd.Stage.CreateNew(str(USD_PATH))
    UsdGeom.SetStageUpAxis(stage, USD_UP_AXIS)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(T - 1)
    stage.SetTimeCodesPerSecond(FPS)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    # --------------------------------------------------------
    # Materials
    # --------------------------------------------------------
    blue_mat = create_preview_material(
        stage,
        "/World/Looks/InactiveBlue",
        color=(0.20, 0.45, 0.95),
        opacity=1.0,
        roughness=0.45,
    )

    red_mat = create_preview_material(
        stage,
        "/World/Looks/ActivatedRed",
        color=(0.88, 0.18, 0.18),
        opacity=1.0,
        roughness=0.40,
    )

    vessel_mat = create_preview_material(
        stage,
        "/World/Looks/VesselTransparent",
        color=(0.78, 0.85, 0.95),
        opacity=0.18,
        roughness=0.20,
    )

    marker_blue = create_preview_material(
        stage,
        "/World/Looks/MarkerBlue",
        color=(0.20, 0.45, 0.95),
        opacity=1.0,
        roughness=0.25,
    )

    marker_orange = create_preview_material(
        stage,
        "/World/Looks/MarkerOrange",
        color=(1.00, 0.62, 0.10),
        opacity=1.0,
        roughness=0.25,
    )

    marker_red = create_preview_material(
        stage,
        "/World/Looks/MarkerRed",
        color=(0.90, 0.15, 0.15),
        opacity=1.0,
        roughness=0.25,
    )

    # --------------------------------------------------------
    # Lights
    # --------------------------------------------------------
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
    dome.CreateIntensityAttr(500.0)

    distant = UsdLux.DistantLight.Define(stage, "/World/Lights/DistantLight")
    distant.CreateIntensityAttr(3000.0)
    distant.CreateAngleAttr(0.5)

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")

    cam_translate = camera.AddTranslateOp()
    # A simple good starting viewpoint; you can refine in Omniverse interactively
    cam_pos = center + np.array([vessel_length * 0.25, -vessel_length * 1.3, vessel_radius * 2.0], dtype=np.float64)
    cam_translate.Set(Gf.Vec3d(*map(float, cam_pos)))

    cam_rotate = camera.AddRotateXYZOp()
    cam_rotate.Set(Gf.Vec3f(70.0, 0.0, 15.0))

    camera.CreateFocalLengthAttr(35.0)

    # --------------------------------------------------------
    # Vessel
    # --------------------------------------------------------
    vessel_mesh = define_mesh(stage, "/World/Vessel/Tube", vessel_vertices, vessel_faces)
    bind_material(vessel_mesh.GetPrim(), vessel_mat)

    # --------------------------------------------------------
    # Platelets
    # --------------------------------------------------------
    platelets_root = UsdGeom.Xform.Define(stage, "/World/Platelets")

    summary = {
        "selected_indices": [],
        "mesh_files": {
            "inactive": str(INACTIVE_MESH_PATH),
            "activated": str(ACTIVATED_MESH_PATH),
        },
        "activation_threshold": ACTIVATION_THRESHOLD,
        "fps": FPS,
        "num_frames": int(T),
    }

    for local_i, platelet_idx in enumerate(selected_indices):
        prim_name = f"P{local_i:02d}_idx{int(platelet_idx)}"
        platelet_root = UsdGeom.Xform.Define(stage, f"/World/Platelets/{prim_name}")

        translate_op = platelet_root.AddTranslateOp()
        # parent translation is animated over frames

        # static child scale xform so platelets are visible
        model_xf = UsdGeom.Xform.Define(stage, f"/World/Platelets/{prim_name}/Model")
        scale_op = model_xf.AddScaleOp()
        scale_op.Set(Gf.Vec3f(float(platelet_scale), float(platelet_scale), float(platelet_scale)))

        # Create inactive / activated meshes
        inactive_mesh = define_mesh(
            stage,
            f"/World/Platelets/{prim_name}/Model/InactiveMesh",
            inactive_vertices,
            inactive_faces,
        )
        bind_material(inactive_mesh.GetPrim(), blue_mat)

        activated_mesh = define_mesh(
            stage,
            f"/World/Platelets/{prim_name}/Model/ActivatedMesh",
            activated_vertices,
            activated_faces,
        )
        bind_material(activated_mesh.GetPrim(), red_mat)

        # Small marker sphere above each platelet to help visibility in scene
        marker = UsdGeom.Sphere.Define(stage, f"/World/Platelets/{prim_name}/Marker")
        marker.CreateRadiusAttr(float(platelet_scale * 0.35))
        marker_xf = UsdGeom.Xformable(marker.GetPrim())
        marker_translate = marker_xf.AddTranslateOp()
        marker_translate.Set(Gf.Vec3d(0.0, 0.0, float(platelet_scale * 1.8)))

        final_state = classify_activation(float(selected_activation[-1, local_i]))
        if final_state == "inactive":
            bind_material(marker.GetPrim(), marker_blue)
        elif final_state == "near-threshold":
            bind_material(marker.GetPrim(), marker_orange)
        else:
            bind_material(marker.GetPrim(), marker_red)

        # Animate position and mesh visibility
        for t in range(T):
            pos = selected_positions[t, local_i]
            a = float(selected_activation[t, local_i])

            translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])), t)

            # Strict switch rule: below threshold = inactive mesh, above/equal = activated mesh
            is_active = a >= ACTIVATION_THRESHOLD
            set_visibility(UsdGeom.Imageable(inactive_mesh.GetPrim()), not is_active, t)
            set_visibility(UsdGeom.Imageable(activated_mesh.GetPrim()), is_active, t)

        summary["selected_indices"].append(
            {
                "selected_order": int(local_i),
                "platelet_index": int(platelet_idx),
                "final_activation": float(selected_activation[-1, local_i]),
                "final_shear": float(selected_shear[-1, local_i]),
                "final_state": final_state,
            }
        )

        print(
            f"[OK] Added platelet {local_i:02d} | idx={int(platelet_idx)} | "
            f"final activation={float(selected_activation[-1, local_i]):.3f} | "
            f"final state={final_state}"
        )

    # Save USD
    stage.GetRootLayer().Save()
    print(f"[OK] Saved USD: {USD_PATH}")

    # --------------------------------------------------------
    # Save JSON summary
    # --------------------------------------------------------
    JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] Saved JSON summary: {JSON_PATH}")

    # --------------------------------------------------------
    # Save README
    # --------------------------------------------------------
    readme = f"""# Phase 6 - Use Case 1: 10 original platelet meshes in vessel (USD)

## Purpose

This use case visualizes a small representative subset of platelets using the **original high-detail platelet meshes** inside a 3D vessel scene.

The purpose is to clearly demonstrate:

- successful import of the original inactive mesh
- successful import of the original activated mesh
- activation-based mesh switching in a live 3D scene
- platelet motion using real Phase 4 output data
- suitability of USD / Omniverse for final scientific visualization

## Input data

### Real simulation data
- `{POSITIONS_PATH.relative_to(PROJECT_ROOT)}`
- `{ACTIVATION_PATH.relative_to(PROJECT_ROOT)}`
- `{SHEAR_PATH.relative_to(PROJECT_ROOT)}`

### Original platelet meshes
- `{INACTIVE_MESH_PATH.relative_to(PROJECT_ROOT)}`
- `{ACTIVATED_MESH_PATH.relative_to(PROJECT_ROOT)}`

## Selection logic

A total of **{len(selected_indices)} representative platelets** are shown.
They are selected to include:

- low activation examples
- near-threshold examples
- activated examples

The activation threshold for mesh switching is:

- `activation >= {ACTIVATION_THRESHOLD}` → activated mesh visible
- `activation < {ACTIVATION_THRESHOLD}` → inactive mesh visible

## Visual interpretation

- **Blue mesh** = inactive platelet morphology
- **Red mesh** = activated platelet morphology
- **Small marker sphere** indicates final-category grouping:
  - blue = inactive
  - orange = near-threshold
  - red = activated

## Important thesis wording

This use case uses the **original high-detail platelet meshes** only for a small platelet subset (10 platelets). This is appropriate for validation and presentation quality.

For larger dense scenes and performance-oriented visualization, decimated meshes or instancing remain more appropriate.

The activated mesh should be described as the **visual representation of the high-activation / increased-adhesion platelet state**. The mesh itself does not store a numerical stickiness value; quantitative activation comes from the simulation output.

## Omniverse usage

Open the exported USD file in Omniverse Composer / View:

- `{USD_PATH.relative_to(PROJECT_ROOT)}`

Then:

1. play the timeline
2. inspect the platelet movement
3. record a presentation video or take screenshots

## Output files

- `usecase1_10_platelets_original_mesh.usd`
- `selected_platelets_usecase1.csv`
- `usecase1_10_platelets_summary.json`
- `README_usecase1_actual_mesh.md`
"""

    README_PATH.write_text(readme, encoding="utf-8")
    print(f"[OK] Saved README: {README_PATH}")

    print("\n[DONE] Use Case 1 USD export completed.")
    print(f"       USD:   {USD_PATH}")
    print(f"       CSV:   {CSV_PATH}")
    print(f"       JSON:  {JSON_PATH}")
    print(f"       README:{README_PATH}\n")


if __name__ == "__main__":
    main()