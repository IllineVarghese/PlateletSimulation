from pathlib import Path
import pyvista as pv


def load_mesh(path: Path) -> pv.PolyData:
    """
    Load a mesh file using PyVista and validate that it contains geometry.

    Parameters
    ----------
    path:
        Path to the mesh file.

    Returns
    -------
    pv.PolyData
        Loaded mesh geometry.

    Raises
    ------
    FileNotFoundError
        If the mesh file does not exist.

    ValueError
        If the mesh has no points.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    mesh = pv.read(path)

    if mesh.n_points == 0:
        raise ValueError(f"Mesh has no points: {path}")

    return mesh


def mesh_dimensions(mesh: pv.PolyData) -> tuple[float, float, float]:
    """
    Return the physical dimensions of a mesh along x, y, and z.
    """
    bounds = mesh.bounds

    size_x = bounds[1] - bounds[0]
    size_y = bounds[3] - bounds[2]
    size_z = bounds[5] - bounds[4]

    return size_x, size_y, size_z


def center_mesh(mesh: pv.PolyData) -> pv.PolyData:
    """
    Return a centered copy of the mesh.

    The original mesh is not modified.
    """
    centered = mesh.copy()
    center = centered.center

    centered.translate(
        (-center[0], -center[1], -center[2]),
        inplace=True,
    )

    return centered


def scale_mesh_to_size(mesh: pv.PolyData, target_size: float = 1.0) -> pv.PolyData:
    """
    Return a scaled copy of the mesh.

    The largest axis of the mesh is scaled to target_size.
    The original mesh is not modified.
    """
    scaled = mesh.copy()

    size_x, size_y, size_z = mesh_dimensions(scaled)
    max_size = max(size_x, size_y, size_z)

    if max_size <= 0:
        raise ValueError("Mesh has invalid size and cannot be scaled.")

    scale_factor = target_size / max_size
    scaled.scale(scale_factor, inplace=True)

    return scaled


def center_and_scale_mesh(mesh: pv.PolyData, target_size: float = 1.0) -> pv.PolyData:
    """
    Return a centered and normalized copy of the mesh.

    This is useful for visual comparison of platelet states.
    """
    centered = center_mesh(mesh)
    normalized = scale_mesh_to_size(centered, target_size=target_size)

    return normalized


def print_mesh_report(name: str, mesh: pv.PolyData) -> None:
    """
    Print useful mesh information for debugging and documentation.
    """
    size_x, size_y, size_z = mesh_dimensions(mesh)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Points:   {mesh.n_points}")
    print(f"Cells:    {mesh.n_cells}")
    print(f"Bounds:   {mesh.bounds}")
    print(f"Center:   {mesh.center}")
    print(f"Size X:   {size_x:.4f}")
    print(f"Size Y:   {size_y:.4f}")
    print(f"Size Z:   {size_z:.4f}")
    print(f"Max size: {max(size_x, size_y, size_z):.4f}")