from pathlib import Path
import csv
import importlib
import importlib.metadata
import platform
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "results" / "phase5" / "week4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_MD = OUTPUT_DIR / "usd_environment_check.md"
OUTPUT_CSV = OUTPUT_DIR / "usd_environment_check.csv"

DECIMATED_INACTIVE = (
    PROJECT_ROOT
    / "results"
    / "phase5"
    / "week3"
    / "optimized_meshes"
    / "inactive_decimated.vtp"
)

DECIMATED_ACTIVATED = (
    PROJECT_ROOT
    / "results"
    / "phase5"
    / "week3"
    / "optimized_meshes"
    / "activated_decimated.vtp"
)


MODULE_CHECKS = [
    {
        "module": "pxr",
        "package_hint": "usd-core",
        "purpose": "Base Pixar USD Python package",
    },
    {
        "module": "pxr.Usd",
        "package_hint": "usd-core",
        "purpose": "Create and save USD stages",
    },
    {
        "module": "pxr.UsdGeom",
        "package_hint": "usd-core",
        "purpose": "Create USD geometry primitives",
    },
    {
        "module": "pxr.Sdf",
        "package_hint": "usd-core",
        "purpose": "USD layer and path utilities",
    },
    {
        "module": "pxr.Gf",
        "package_hint": "usd-core",
        "purpose": "USD vector and matrix math",
    },
    {
        "module": "omni.usd",
        "package_hint": "Omniverse Kit Python environment",
        "purpose": "Omniverse USD integration",
    },
    {
        "module": "pyvista",
        "package_hint": "pyvista",
        "purpose": "Current mesh visualization and VTP/PLY export fallback",
    },
    {
        "module": "vtk",
        "package_hint": "vtk",
        "purpose": "Mesh processing, decimation, and export backend",
    },
    {
        "module": "numpy",
        "package_hint": "numpy",
        "purpose": "Simulation array loading and processing",
    },
    {
        "module": "meshio",
        "package_hint": "meshio",
        "purpose": "Optional mesh format conversion fallback",
    },
    {
        "module": "trimesh",
        "package_hint": "trimesh",
        "purpose": "Optional OBJ/PLY/GLB mesh conversion support",
    },
]


def get_package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except Exception:
        return "not found"


def check_module(module_name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "available")
        return True, str(version)
    except Exception as error:
        return False, str(error)


def write_csv(rows: list[dict]) -> None:
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "module",
                "available",
                "module_version_or_error",
                "package_hint",
                "package_version",
                "purpose",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict]) -> None:
    pxr_available = any(row["module"] == "pxr.Usd" and row["available"] for row in rows)
    omni_available = any(row["module"] == "omni.usd" and row["available"] for row in rows)
    pyvista_available = any(row["module"] == "pyvista" and row["available"] for row in rows)
    vtk_available = any(row["module"] == "vtk" and row["available"] for row in rows)

    decimated_inactive_exists = DECIMATED_INACTIVE.exists()
    decimated_activated_exists = DECIMATED_ACTIVATED.exists()

    lines = []

    lines.append("# Phase 5 Week 4 Day 1: USD / Omniverse Environment Check")
    lines.append("")
    lines.append("## System")
    lines.append("")
    lines.append(f"- Python executable: `{sys.executable}`")
    lines.append(f"- Python version: `{sys.version.split()[0]}`")
    lines.append(f"- Platform: `{platform.platform()}`")
    lines.append("")
    lines.append("## Module Availability")
    lines.append("")
    lines.append("| Module | Available | Version / Error | Purpose |")
    lines.append("|---|---:|---|---|")

    for row in rows:
        lines.append(
            f"| `{row['module']}` | {row['available']} | "
            f"`{row['module_version_or_error']}` | {row['purpose']} |"
        )

    lines.append("")
    lines.append("## Local Optimized Mesh Assets")
    lines.append("")
    lines.append(f"- Inactive decimated mesh exists: `{decimated_inactive_exists}`")
    lines.append(f"- Activated decimated mesh exists: `{decimated_activated_exists}`")
    lines.append(f"- Inactive path: `{DECIMATED_INACTIVE}`")
    lines.append(f"- Activated path: `{DECIMATED_ACTIVATED}`")
    lines.append("")
    lines.append("## Export Decision")
    lines.append("")

    if pxr_available:
        lines.append(
            "Direct USD export is available because the `pxr` USD Python modules are installed."
        )
        lines.append(
            "Week 4 Day 2 can proceed with direct `.usd` or `.usda` stage generation."
        )
    elif omni_available:
        lines.append(
            "Omniverse USD integration appears available through `omni.usd`, but direct standalone `pxr` export was not confirmed."
        )
        lines.append(
            "Week 4 Day 2 should either run inside an Omniverse Kit Python environment or use fallback exports."
        )
    elif pyvista_available and vtk_available:
        lines.append(
            "Direct USD export is not available in this environment, but PyVista/VTK fallback mesh export is available."
        )
        lines.append(
            "Week 4 Day 2 should create export-ready `.vtp`, `.ply`, or `.obj` assets and document USD as a future Omniverse-side conversion step."
        )
    else:
        lines.append(
            "Neither direct USD export nor PyVista/VTK fallback export is fully available."
        )
        lines.append(
            "The environment needs additional setup before Week 4 export work can continue."
        )

    lines.append("")
    lines.append("## Recommended Next Step")
    lines.append("")
    lines.append(
        "Proceed to Week 4 Day 2 by creating a static export-ready platelet scene. "
        "Use direct USD export if `pxr.Usd` is available; otherwise use VTP/PLY fallback export."
    )

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("Phase 5 Week 4 Day 1: USD / Omniverse environment check")
    print("Checking export-related Python modules...\n")

    rows = []

    for item in MODULE_CHECKS:
        module_name = item["module"]
        package_hint = item["package_hint"]

        available, version_or_error = check_module(module_name)

        package_version = (
            get_package_version(package_hint)
            if package_hint not in ["Omniverse Kit Python environment"]
            else "not checked"
        )

        row = {
            "module": module_name,
            "available": available,
            "module_version_or_error": version_or_error,
            "package_hint": package_hint,
            "package_version": package_version,
            "purpose": item["purpose"],
        }

        rows.append(row)

        status = "OK" if available else "MISSING"
        print(f"{status:8s} | {module_name:14s} | {version_or_error}")

    write_csv(rows)
    write_markdown(rows)

    print("\nChecking local optimized meshes...")
    print(f"inactive_decimated.vtp:  {DECIMATED_INACTIVE.exists()}")
    print(f"activated_decimated.vtp: {DECIMATED_ACTIVATED.exists()}")

    print("\nSaved report:")
    print(OUTPUT_MD)
    print(OUTPUT_CSV)

    print("\nWeek 4 Day 1 environment check complete.")


if __name__ == "__main__":
    main()