from pathlib import Path

from mesh_utils import (
    load_mesh,
    center_and_scale_mesh,
    print_mesh_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INACTIVE_MESH = PROJECT_ROOT / "data" / "meshes" / "platelet" / "inactive.obj"
ACTIVATED_MESH = PROJECT_ROOT / "data" / "meshes" / "platelet" / "activated.obj"


def main() -> None:
    print("Phase 5 Week 1 Day 3: Testing mesh utilities")

    inactive = load_mesh(INACTIVE_MESH)
    activated = load_mesh(ACTIVATED_MESH)

    print_mesh_report("Inactive mesh - original", inactive)
    print_mesh_report("Activated mesh - original", activated)

    inactive_normalized = center_and_scale_mesh(inactive, target_size=1.0)
    activated_normalized = center_and_scale_mesh(activated, target_size=1.0)

    print_mesh_report("Inactive mesh - normalized", inactive_normalized)
    print_mesh_report("Activated mesh - normalized", activated_normalized)

    print("\nMesh utility test completed successfully.")


if __name__ == "__main__":
    main()