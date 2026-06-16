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


def center_and_scale(mesh: pv.PolyData, target_size: float = 1.0) -> pv.PolyData:
    """
    Center mesh at origin and scale it to a common size for visual comparison.
    """
    mesh = mesh.copy()

    center = mesh.center
    mesh.translate((-center[0], -center[1], -center[2]), inplace=True)

    bounds = mesh.bounds
    size_x = bounds[1] - bounds[0]
    size_y = bounds[3] - bounds[2]
    size_z = bounds[5] - bounds[4]
    max_size = max(size_x, size_y, size_z)

    if max_size <= 0:
        raise ValueError("Mesh has invalid size.")

    mesh.scale(target_size / max_size, inplace=True)
    return mesh


def mesh_dimensions(mesh: pv.PolyData) -> tuple[float, float, float]:
    bounds = mesh.bounds
    return (
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
    )


def print_report(name: str, mesh: pv.PolyData) -> None:
    size_x, size_y, size_z = mesh_dimensions(mesh)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Points:      {mesh.n_points}")
    print(f"Cells:       {mesh.n_cells}")
    print(f"Bounds:      {mesh.bounds}")
    print(f"Center:      {mesh.center}")
    print(f"Size X:      {size_x:.4f}")
    print(f"Size Y:      {size_y:.4f}")
    print(f"Size Z:      {size_z:.4f}")
    print(f"Max size:    {max(size_x, size_y, size_z):.4f}")


def save_view(mesh_a: pv.PolyData, mesh_b: pv.PolyData, camera: str, filename: str) -> None:
    """
    Save inactive and activated meshes side by side from one camera direction.
    """
    inactive = mesh_a.copy()
    activated = mesh_b.copy()

    inactive.translate((-0.8, 0.0, 0.0), inplace=True)
    activated.translate((0.8, 0.0, 0.0), inplace=True)

    plotter = pv.Plotter(off_screen=True, window_size=(1600, 900))
    plotter.set_background("white")

    plotter.add_mesh(
        inactive,
        smooth_shading=True,
        show_edges=True,
        line_width=0.3,
        opacity=1.0,
    )

    plotter.add_mesh(
        activated,
        smooth_shading=True,
        show_edges=True,
        line_width=0.3,
        opacity=1.0,
    )

    plotter.add_text(
        f"Inactive mesh | Activated mesh | Camera: {camera}",
        position=(420, 840),
        font_size=15,
        color="black",
    )

    plotter.add_axes()

    if camera == "xy":
        plotter.camera_position = "xy"
    elif camera == "xz":
        plotter.camera_position = "xz"
    elif camera == "yz":
        plotter.camera_position = "yz"
    elif camera == "iso":
        plotter.camera_position = "iso"
    else:
        raise ValueError(f"Unknown camera view: {camera}")

    output_path = OUTPUT_DIR / filename
    plotter.screenshot(str(output_path))
    plotter.close()

    print(f"Saved: {output_path}")


def main() -> None:
    print("Phase 5 Week 1 Day 2: Mesh Orientation and Scale Check")

    inactive_raw = load_mesh(INACTIVE_MESH)
    activated_raw = load_mesh(ACTIVATED_MESH)

    print_report("Inactive platelet mesh - original", inactive_raw)
    print_report("Activated platelet mesh - original", activated_raw)

    inactive = center_and_scale(inactive_raw, target_size=1.0)
    activated = center_and_scale(activated_raw, target_size=1.0)

    print_report("Inactive platelet mesh - normalized", inactive)
    print_report("Activated platelet mesh - normalized", activated)

    save_view(inactive, activated, "xy", "platelet_mesh_view_xy.png")
    save_view(inactive, activated, "xz", "platelet_mesh_view_xz.png")
    save_view(inactive, activated, "yz", "platelet_mesh_view_yz.png")
    save_view(inactive, activated, "iso", "platelet_mesh_view_iso.png")

    print("\nDay 2 inspection complete.")
    print("Check the four images in results/phase5/week1/")


if __name__ == "__main__":
    main()