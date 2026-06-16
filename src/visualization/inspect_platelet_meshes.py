from pathlib import Path
import pyvista as pv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INACTIVE_MESH = PROJECT_ROOT / "data" / "meshes" / "platelet" / "inactive.obj"
ACTIVATED_MESH = PROJECT_ROOT / "data" / "meshes" / "platelet" / "activated.obj"

OUTPUT_DIR = PROJECT_ROOT / "results" / "phase5" / "week1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_mesh(path: Path) -> pv.PolyData:
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    mesh = pv.read(path)

    if mesh.n_points == 0:
        raise ValueError(f"Mesh has no points: {path}")

    return mesh


def normalize_mesh(mesh: pv.PolyData, target_size: float = 1.0) -> pv.PolyData:
    mesh = mesh.copy()

    center = mesh.center
    mesh.translate((-center[0], -center[1], -center[2]), inplace=True)

    bounds = mesh.bounds
    size_x = bounds[1] - bounds[0]
    size_y = bounds[3] - bounds[2]
    size_z = bounds[5] - bounds[4]
    max_size = max(size_x, size_y, size_z)

    if max_size <= 0:
        raise ValueError("Mesh has invalid size and cannot be normalized.")

    scale_factor = target_size / max_size
    mesh.scale(scale_factor, inplace=True)

    return mesh


def print_mesh_info(name: str, mesh: pv.PolyData) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Number of points: {mesh.n_points}")
    print(f"Number of cells:  {mesh.n_cells}")
    print(f"Bounds:           {mesh.bounds}")
    print(f"Center:           {mesh.center}")


def main() -> None:
    print("Phase 5 Week 1: Platelet Mesh Inspection")

    inactive = load_mesh(INACTIVE_MESH)
    activated = load_mesh(ACTIVATED_MESH)

    print_mesh_info("Inactive platelet mesh - original", inactive)
    print_mesh_info("Activated platelet mesh - original", activated)

    inactive_norm = normalize_mesh(inactive, target_size=1.0)
    activated_norm = normalize_mesh(activated, target_size=1.0)

    inactive_norm.translate((-0.8, 0.0, 0.0), inplace=True)
    activated_norm.translate((0.8, 0.0, 0.0), inplace=True)

    plotter = pv.Plotter(off_screen=True, window_size=(1600, 900))
    plotter.set_background("white")

    plotter.add_mesh(
        inactive_norm,
        smooth_shading=True,
        show_edges=False,
        opacity=1.0,
    )

    plotter.add_mesh(
        activated_norm,
        smooth_shading=True,
        show_edges=False,
        opacity=1.0,
    )

    plotter.add_text(
        "Inactive platelet mesh",
        position=(240, 830),
        font_size=16,
        color="black",
    )

    plotter.add_text(
        "Activated platelet mesh",
        position=(940, 830),
        font_size=16,
        color="black",
    )

    plotter.add_axes()
    plotter.camera_position = "xy"

    output_path = OUTPUT_DIR / "platelet_mesh_comparison.png"
    plotter.screenshot(str(output_path))
    plotter.close()

    print("\nSaved output:")
    print(output_path)


if __name__ == "__main__":
    main()