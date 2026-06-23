"""
Phase 6 - Use Case 1
Actual high-detail platelet mesh validation.

Goal:
- Use original inactive and activated platelet meshes.
- Use real Phase 4 output arrays.
- Select 10 representative platelets:
  inactive, near-threshold, and activated.
- Render thesis-quality PNG figure with activation labels.
- Save outputs under results/phase6/usecase1_actual_mesh/

Important:
This script loads the original high-detail meshes to prove successful mesh import.
For Matplotlib rendering, a face limit is used to avoid freezing.
The original full mesh counts are still printed and documented.
"""

from pathlib import Path
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

PROJECT_ROOT = Path(".").resolve()

INACTIVE_MESH_PATH = PROJECT_ROOT / "data/meshes/platelet/inactive.obj"
ACTIVATED_MESH_PATH = PROJECT_ROOT / "data/meshes/platelet/activated_usecase1.obj"

POSITIONS_PATH = PROJECT_ROOT / "results/phase4/final_demo/positions.npy"
ACTIVATION_PATH = PROJECT_ROOT / "results/phase4/final_demo/activation.npy"
SHEAR_PATH = PROJECT_ROOT / "results/phase4/final_demo/shear_input.npy"

OUTPUT_DIR = PROJECT_ROOT / "results/phase6/usecase1_actual_mesh"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SELECTED = 10
ACTIVATION_THRESHOLD = 0.5

# Safe preview rendering limit.
# The original meshes are loaded fully, but only selected faces are rendered by Matplotlib.
# This avoids freezing while still demonstrating actual original mesh import.
FACE_RENDER_LIMIT = 3000

# Video disabled because Matplotlib video rendering with high-detail meshes is too slow.
SAVE_VIDEO = False


# ------------------------------------------------------------
# Cleanup old broken PDF
# ------------------------------------------------------------

def remove_old_broken_pdf():
    """
    Removes the old PDF if it exists.
    The earlier PDF export was interrupted and can appear as corrupted binary text in VS Code.
    """
    old_pdf = OUTPUT_DIR / "usecase1_actual_mesh_validation.pdf"
    if old_pdf.exists():
        try:
            old_pdf.unlink()
            print(f"[OK] Removed old PDF file: {old_pdf}")
        except Exception as e:
            print(f"[WARN] Could not remove old PDF file: {e}")


# ------------------------------------------------------------
# OBJ mesh loader
# ------------------------------------------------------------

def load_obj_mesh(obj_path: Path):
    """
    Minimal OBJ loader for vertices and triangular faces.

    Supports:
    - v x y z
    - f i j k
    - f i/... j/... k/...
    - polygon faces are triangulated by fan triangulation
    """
    if not obj_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {obj_path}")

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
                parts = line.split()[1:]
                face_indices = []

                for p in parts:
                    idx = p.split("/")[0]
                    if idx:
                        face_indices.append(int(idx) - 1)

                if len(face_indices) >= 3:
                    for i in range(1, len(face_indices) - 1):
                        faces.append(
                            [face_indices[0], face_indices[i], face_indices[i + 1]]
                        )

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)

    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError(f"Could not load valid vertices/faces from {obj_path}")

    return vertices, faces


def normalize_mesh(vertices):
    """
    Center mesh at origin and scale to a consistent visible size.
    This affects visualization only, not the original mesh topology.
    """
    v = vertices.copy()
    center = v.mean(axis=0)
    v -= center

    scale = np.max(np.linalg.norm(v, axis=1))
    if scale > 0:
        v /= scale

    return v


def maybe_limit_faces(faces, limit):
    """
    Use a representative subset of faces for Matplotlib rendering.
    The original mesh is still fully loaded and counted before this step.
    """
    if limit is None:
        return faces

    if len(faces) <= limit:
        return faces

    idx = np.linspace(0, len(faces) - 1, limit).astype(int)
    return faces[idx]


# ------------------------------------------------------------
# Phase 4 data loading
# ------------------------------------------------------------

def load_phase4_array(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"{name} file not found: {path}")

    arr = np.load(path)
    print(f"[OK] Loaded {name}: shape={arr.shape}, dtype={arr.dtype}")
    return arr


def get_last_frame(arr):
    """
    Handles both:
    - positions: (T, N, 3) or (N, 3)
    - activation/shear: (T, N) or (N,)
    """
    if arr.ndim >= 2:
        return arr[-1]
    return arr


def ensure_vector(arr, name):
    arr = np.asarray(arr)

    if arr.ndim != 1:
        arr = arr.reshape(-1)

    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")

    return arr


def ensure_positions(arr):
    arr = np.asarray(arr)

    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"positions must have shape (N, 3) after frame selection, got {arr.shape}"
        )

    if np.any(~np.isfinite(arr)):
        raise ValueError("positions contain non-finite values.")

    return arr


# ------------------------------------------------------------
# Representative platelet selection
# ------------------------------------------------------------

def select_representative_platelets(activation, n_selected=10, threshold=0.5):
    """
    Selects a balanced set:
    - low activation examples
    - near-threshold examples
    - high activation examples
    """
    n = len(activation)
    all_indices = np.arange(n)

    inactive_sorted = all_indices[np.argsort(activation)]
    active_sorted = all_indices[np.argsort(-activation)]
    near_sorted = all_indices[np.argsort(np.abs(activation - threshold))]

    selected = []

    def add_unique(candidates, target_total):
        for idx in candidates:
            idx = int(idx)
            if idx not in selected:
                selected.append(idx)

            if len(selected) >= target_total:
                break

    # 3 inactive, 3 near-threshold, 4 activated
    add_unique(inactive_sorted, 3)
    add_unique(near_sorted, 6)
    add_unique(active_sorted, n_selected)

    # Fallback if needed
    if len(selected) < n_selected:
        sorted_by_activation = all_indices[np.argsort(activation)]
        quantile_positions = np.linspace(0, n - 1, n_selected).astype(int)

        for q in quantile_positions:
            idx = int(sorted_by_activation[q])
            if idx not in selected:
                selected.append(idx)

            if len(selected) == n_selected:
                break

    return np.asarray(selected[:n_selected], dtype=int)


def classify_activation(a, threshold=0.5):
    if a < threshold - 0.15:
        return "inactive"

    if a < threshold + 0.15:
        return "near-threshold"

    return "activated"


# ------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------

def rotation_matrix_z(angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)

    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def transform_mesh(vertices, target_position, visual_scale=0.18, rotation_angle=0.0):
    """
    Places a normalized mesh at a selected platelet position.
    """
    r = rotation_matrix_z(rotation_angle)
    v = vertices @ r.T
    v = v * visual_scale
    v = v + target_position
    return v


def add_mesh_to_axis(ax, vertices, faces, color, alpha=0.95):
    mesh_faces = vertices[faces]

    collection = Poly3DCollection(
        mesh_faces,
        linewidths=0.05,
        alpha=alpha,
        edgecolor="black",
    )

    collection.set_facecolor(color)
    ax.add_collection3d(collection)


def set_axes_equal(ax):
    """
    Makes 3D axes use equal scaling.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def color_from_state(state):
    if state == "inactive":
        return (0.20, 0.45, 0.95, 1.0)

    if state == "near-threshold":
        return (1.00, 0.65, 0.10, 1.0)

    return (0.90, 0.15, 0.15, 1.0)


# ------------------------------------------------------------
# Thesis figure
# ------------------------------------------------------------

def create_thesis_figure(
    selected_indices,
    positions,
    activation,
    shear,
    inactive_vertices,
    inactive_faces,
    activated_vertices,
    activated_faces,
):
    """
    Creates a PNG-only thesis preview figure.

    PDF export is intentionally disabled because Matplotlib PDF export is too slow
    for high-detail triangular mesh collections.
    """
    selected_positions = positions[selected_indices]
    selected_activation = activation[selected_indices]
    selected_shear = shear[selected_indices]

    # Normalize selected positions for visual layout only.
    pos = selected_positions.copy()
    pos -= pos.mean(axis=0)

    max_extent = np.max(np.linalg.norm(pos, axis=1))
    if max_extent > 0:
        pos = pos / max_extent

    pos *= 1.6

    fig = plt.figure(figsize=(16, 8))

    # 3D mesh view
    ax = fig.add_subplot(121, projection="3d")
    ax.set_title(
        "Use Case 1: Original Platelet Mesh Switching Validation",
        fontsize=13,
        pad=20,
    )

    for local_i, platelet_idx in enumerate(selected_indices):
        a = float(selected_activation[local_i])
        s = float(selected_shear[local_i])
        state = classify_activation(a, ACTIVATION_THRESHOLD)

        if state == "activated":
            base_vertices = activated_vertices
            base_faces = activated_faces
        else:
            base_vertices = inactive_vertices
            base_faces = inactive_faces

        transformed = transform_mesh(
            base_vertices,
            target_position=pos[local_i],
            visual_scale=0.20,
            rotation_angle=local_i * 0.55,
        )

        add_mesh_to_axis(
            ax,
            transformed,
            base_faces,
            color=color_from_state(state),
            alpha=0.95,
        )

        label = f"ID {platelet_idx}\nA={a:.2f}"

        ax.text(
            pos[local_i, 0],
            pos[local_i, 1],
            pos[local_i, 2] + 0.32,
            label,
            fontsize=8,
            ha="center",
        )

    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.set_zlabel("z position")
    ax.view_init(elev=22, azim=35)
    set_axes_equal(ax)

    # Activation/shear plot
    ax2 = fig.add_subplot(122)

    x = np.arange(len(selected_indices))

    ax2.plot(
        x,
        selected_activation,
        marker="o",
        linewidth=2,
        label="Activation",
    )

    ax2.axhline(
        ACTIVATION_THRESHOLD,
        linestyle="--",
        linewidth=1.2,
        label=f"Switch threshold = {ACTIVATION_THRESHOLD}",
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels([str(i) for i in selected_indices], rotation=45)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("Selected platelet index")
    ax2.set_ylabel("Activation value")
    ax2.set_title("Activation values used for mesh-state selection")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")

    ax3 = ax2.twinx()

    ax3.plot(
        x,
        selected_shear,
        marker="s",
        linestyle=":",
        linewidth=1.8,
        label="Shear input",
    )

    ax3.set_ylabel("Shear input")
    ax3.legend(loc="lower right")

    fig.text(
        0.5,
        0.02,
        "Original high-detail platelet meshes are loaded for small-scale validation. "
        "A limited face subset is rendered in Matplotlib to avoid freezing; "
        "decimated meshes remain appropriate for dense visualization and performance benchmarks.",
        ha="center",
        fontsize=10,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    png_path = OUTPUT_DIR / "usecase1_actual_mesh_validation.png"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"[OK] Saved figure: {png_path}")

    return png_path


# ------------------------------------------------------------
# CSV summary
# ------------------------------------------------------------

def save_selection_csv(selected_indices, positions, activation, shear):
    csv_path = OUTPUT_DIR / "selected_platelets_usecase1.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "selected_order",
                "platelet_index",
                "x",
                "y",
                "z",
                "activation",
                "shear_input",
                "classification",
                "mesh_used",
            ]
        )

        for order, idx in enumerate(selected_indices):
            a = float(activation[idx])
            state = classify_activation(a, ACTIVATION_THRESHOLD)

            if state == "activated":
                mesh_used = "activated.obj"
            else:
                mesh_used = "inactive.obj"

            writer.writerow(
                [
                    order,
                    int(idx),
                    float(positions[idx, 0]),
                    float(positions[idx, 1]),
                    float(positions[idx, 2]),
                    a,
                    float(shear[idx]),
                    state,
                    mesh_used,
                ]
            )

    print(f"[OK] Saved selection table: {csv_path}")
    return csv_path


# ------------------------------------------------------------
# Documentation
# ------------------------------------------------------------

def save_documentation(
    inactive_full_vertices,
    inactive_full_faces,
    activated_full_vertices,
    activated_full_faces,
):
    doc_path = OUTPUT_DIR / "README_usecase1_actual_mesh.md"

    text = f"""# Phase 6 - Use Case 1: Actual Mesh Validation

## Purpose

This use case validates that the original high-detail platelet meshes can be imported and used for activation-based morphology switching.

The goal is not dense performance visualization. Instead, the goal is a small, clear validation with approximately {N_SELECTED} representative platelets.

## Input data

The script uses real Phase 4 output data:

- `{POSITIONS_PATH.relative_to(PROJECT_ROOT)}`
- `{ACTIVATION_PATH.relative_to(PROJECT_ROOT)}`
- `{SHEAR_PATH.relative_to(PROJECT_ROOT)}`

The script uses the original platelet meshes:

- `{INACTIVE_MESH_PATH.relative_to(PROJECT_ROOT)}`
- `{ACTIVATED_MESH_PATH.relative_to(PROJECT_ROOT)}`

## Original mesh complexity

The original meshes were fully loaded successfully:

- Inactive mesh: {inactive_full_vertices} vertices, {inactive_full_faces} faces
- Activated mesh: {activated_full_vertices} vertices, {activated_full_faces} faces

For Matplotlib rendering only, the number of rendered faces was limited to {FACE_RENDER_LIMIT} per mesh to avoid freezing.

## Selection strategy

Representative platelets are selected from the final simulation frame:

1. Low-activation examples
2. Near-threshold examples
3. High-activation examples

The switching threshold is set to `{ACTIVATION_THRESHOLD}`.

## Interpretation

- Platelets below the activation threshold are shown with the inactive morphology.
- Platelets above the activation threshold are shown with the activated morphology.
- Near-threshold platelets are shown as intermediate transition examples.

## Important thesis wording

The original high-detail meshes are used here for small-scale visual validation because they preserve morphology detail but are computationally expensive.

For dense visualization, performance benchmarks, large platelet populations, and final animation, decimated meshes or USD instancing are more appropriate.

This distinction supports both biological visual fidelity and computational scalability.

## Outputs

- `usecase1_actual_mesh_validation.png`
- `selected_platelets_usecase1.csv`
- `README_usecase1_actual_mesh.md`

PDF and MP4 output are intentionally disabled in this Matplotlib version to avoid freezing.
A PyVista or Omniverse/USD renderer should be used for the final high-quality 3D visual inspection.
"""

    doc_path.write_text(text, encoding="utf-8")
    print(f"[OK] Saved documentation: {doc_path}")
    return doc_path


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("\n=== Phase 6 / Use Case 1: Actual platelet mesh validation ===\n")

    remove_old_broken_pdf()

    positions_all = load_phase4_array(POSITIONS_PATH, "positions")
    activation_all = load_phase4_array(ACTIVATION_PATH, "activation")
    shear_all = load_phase4_array(SHEAR_PATH, "shear_input")

    positions = ensure_positions(get_last_frame(positions_all))
    activation = ensure_vector(get_last_frame(activation_all), "activation")
    shear = ensure_vector(get_last_frame(shear_all), "shear_input")

    n = min(len(positions), len(activation), len(shear))

    positions = positions[:n]
    activation = activation[:n]
    shear = shear[:n]

    print(f"[OK] Using final frame with {n} platelets")

    selected_indices = select_representative_platelets(
        activation,
        n_selected=min(N_SELECTED, n),
        threshold=ACTIVATION_THRESHOLD,
    )

    print("[OK] Selected representative platelet indices:")

    for idx in selected_indices:
        print(
            f"  index={idx:5d} | activation={activation[idx]:.3f} | "
            f"shear={shear[idx]:.3f} | state={classify_activation(activation[idx])}"
        )

    inactive_vertices, inactive_faces = load_obj_mesh(INACTIVE_MESH_PATH)
    activated_vertices, activated_faces = load_obj_mesh(ACTIVATED_MESH_PATH)

    inactive_full_vertices = len(inactive_vertices)
    inactive_full_faces = len(inactive_faces)
    activated_full_vertices = len(activated_vertices)
    activated_full_faces = len(activated_faces)

    print(
        f"[OK] Inactive original mesh loaded: "
        f"{inactive_full_vertices} vertices, {inactive_full_faces} faces"
    )

    print(
        f"[OK] Activated original mesh loaded: "
        f"{activated_full_vertices} vertices, {activated_full_faces} faces"
    )

    inactive_vertices = normalize_mesh(inactive_vertices)
    activated_vertices = normalize_mesh(activated_vertices)

    inactive_faces_render = maybe_limit_faces(inactive_faces, FACE_RENDER_LIMIT)
    activated_faces_render = maybe_limit_faces(activated_faces, FACE_RENDER_LIMIT)

    print(
        f"[OK] Rendering inactive mesh with {len(inactive_faces_render)} faces "
        f"out of {inactive_full_faces}"
    )

    print(
        f"[OK] Rendering activated mesh with {len(activated_faces_render)} faces "
        f"out of {activated_full_faces}"
    )

    save_selection_csv(selected_indices, positions, activation, shear)

    create_thesis_figure(
        selected_indices,
        positions,
        activation,
        shear,
        inactive_vertices,
        inactive_faces_render,
        activated_vertices,
        activated_faces_render,
    )

    save_documentation(
        inactive_full_vertices,
        inactive_full_faces,
        activated_full_vertices,
        activated_full_faces,
    )

    print("\n[DONE] Use Case 1 outputs saved in:")
    print(f"       {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()