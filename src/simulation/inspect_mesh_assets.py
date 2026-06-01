from pathlib import Path

import trimesh


MESH_FILES = [
    Path("data/meshes/platelet/activated.obj"),
    Path("data/meshes/platelet/inactive.obj"),
]


def inspect_mesh(path: Path) -> None:
    print("\n" + "=" * 80)
    print(f"Inspecting: {path}")

    if not path.exists():
        print("ERROR: file does not exist")
        return

    mesh = trimesh.load(path, force="scene")

    print(f"Loaded type: {type(mesh)}")

    if isinstance(mesh, trimesh.Scene):
        print(f"Scene geometry count: {len(mesh.geometry)}")

        for name, geom in mesh.geometry.items():
            print(f"\nObject: {name}")
            print(f"  Type: {type(geom)}")
            print(f"  Vertices: {len(geom.vertices)}")
            print(f"  Faces: {len(geom.faces)}")
            print(f"  Bounds min: {geom.bounds[0]}")
            print(f"  Bounds max: {geom.bounds[1]}")
            print(f"  Extents: {geom.extents}")
            print(f"  Watertight: {geom.is_watertight}")

    elif isinstance(mesh, trimesh.Trimesh):
        print(f"Vertices: {len(mesh.vertices)}")
        print(f"Faces: {len(mesh.faces)}")
        print(f"Bounds min: {mesh.bounds[0]}")
        print(f"Bounds max: {mesh.bounds[1]}")
        print(f"Extents: {mesh.extents}")
        print(f"Watertight: {mesh.is_watertight}")

    else:
        print("Unknown mesh type")


def main() -> None:
    for path in MESH_FILES:
        inspect_mesh(path)


if __name__ == "__main__":
    main()