from pathlib import Path
import csv
import hashlib
from typing import Any

from pxr import Usd, UsdGeom


PROJECT_ROOT = Path(__file__).resolve().parents[2]

WEEK4_DIR = PROJECT_ROOT / "results" / "phase5" / "week4"

STATIC_DIR = WEEK4_DIR / "exports" / "static_usd_scene"
ANIMATED_DIR = WEEK4_DIR / "exports" / "animated_usd_scene"

STATIC_USDA = STATIC_DIR / "phase5_platelet_static_scene.usda"
STATIC_METADATA_CSV = STATIC_DIR / "platelet_scene_metadata.csv"
STATIC_SUMMARY_MD = STATIC_DIR / "usd_static_scene_export_summary.md"
STATIC_PREVIEW_PNG = STATIC_DIR / "phase5_static_usd_scene_preview.png"

ANIMATED_USDA = ANIMATED_DIR / "phase5_platelet_animated_scene.usda"
ANIMATED_METADATA_CSV = ANIMATED_DIR / "platelet_animation_metadata.csv"
ANIMATED_SUMMARY_MD = ANIMATED_DIR / "animated_usd_export_summary.md"
ANIMATED_PREVIEW_PNG = ANIMATED_DIR / "phase5_animated_usd_final_preview.png"
ANIMATED_COUNTS_PNG = ANIMATED_DIR / "activation_state_counts_over_time.png"

OUTPUT_VALIDATION_MD = WEEK4_DIR / "phase5_usd_export_validation.md"
OUTPUT_VALIDATION_CSV = WEEK4_DIR / "phase5_usd_export_validation.csv"


def file_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def file_sha256_short(path: Path) -> str:
    if not path.exists():
        return "missing"

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()[:12]


def read_first_line(path: Path) -> str:
    if not path.exists():
        return ""

    with path.open("r", encoding="utf-8", errors="replace") as file:
        return file.readline().strip()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def add_result(
    rows: list[dict[str, Any]],
    category: str,
    item: str,
    status: str,
    details: str,
    path: Path | None = None,
) -> None:
    rows.append(
        {
            "category": category,
            "item": item,
            "status": status,
            "details": details,
            "path": str(path) if path is not None else "",
            "size_mb": f"{file_size_mb(path):.3f}" if path is not None else "",
            "sha256_short": file_sha256_short(path) if path is not None and path.exists() else "",
        }
    )


def check_required_file(
    rows: list[dict[str, Any]],
    category: str,
    item: str,
    path: Path,
) -> bool:
    if path.exists():
        add_result(
            rows,
            category,
            item,
            "PASS",
            f"File exists, size={file_size_mb(path):.3f} MB",
            path,
        )
        return True

    add_result(rows, category, item, "FAIL", "Required file is missing.", path)
    return False


def open_usd_stage(
    rows: list[dict[str, Any]],
    category: str,
    path: Path,
) -> Usd.Stage | None:
    if not path.exists():
        add_result(rows, category, "Open USD stage", "FAIL", "USD file is missing.", path)
        return None

    header = read_first_line(path)

    if header == "#usda 1.0":
        add_result(rows, category, "USDA header", "PASS", "File starts with #usda 1.0.", path)
    else:
        add_result(
            rows,
            category,
            "USDA header",
            "WARN",
            f"Unexpected first line: {header}",
            path,
        )

    try:
        stage = Usd.Stage.Open(str(path))
    except Exception as error:
        add_result(rows, category, "Open USD stage", "FAIL", f"Could not open USD file: {error}", path)
        return None

    if stage is None:
        add_result(rows, category, "Open USD stage", "FAIL", "Usd.Stage.Open returned None.", path)
        return None

    add_result(rows, category, "Open USD stage", "PASS", "USD stage opened successfully.", path)
    return stage


def count_mesh_prims_under(stage: Usd.Stage, prefix: str) -> int:
    count = 0

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith(prefix) and prim.GetTypeName() == "Mesh":
            count += 1

    return count


def validate_static_usd(rows: list[dict[str, Any]]) -> dict[str, Any]:
    category = "Static USD export"

    check_required_file(rows, category, "Static USD scene", STATIC_USDA)
    check_required_file(rows, category, "Static metadata CSV", STATIC_METADATA_CSV)
    check_required_file(rows, category, "Static preview image", STATIC_PREVIEW_PNG)
    check_required_file(rows, category, "Static export summary", STATIC_SUMMARY_MD)

    stage = open_usd_stage(rows, category, STATIC_USDA)

    metadata_rows = read_csv_rows(STATIC_METADATA_CSV)
    metadata_count = len(metadata_rows)

    inactive_count = sum(1 for row in metadata_rows if row.get("state") == "inactive")
    activated_count = sum(1 for row in metadata_rows if row.get("state") == "activated")

    if metadata_count > 0:
        add_result(
            rows,
            category,
            "Static metadata rows",
            "PASS",
            f"Metadata contains {metadata_count} exported platelet rows: inactive={inactive_count}, activated={activated_count}.",
            STATIC_METADATA_CSV,
        )
    else:
        add_result(
            rows,
            category,
            "Static metadata rows",
            "FAIL",
            "Metadata CSV is empty or missing.",
            STATIC_METADATA_CSV,
        )

    mesh_count = 0

    if stage is not None:
        default_prim = stage.GetDefaultPrim()
        default_name = default_prim.GetName() if default_prim.IsValid() else "missing"

        if default_name == "Phase5StaticScene":
            add_result(rows, category, "Default prim", "PASS", "Default prim is Phase5StaticScene.", STATIC_USDA)
        else:
            add_result(rows, category, "Default prim", "WARN", f"Default prim is {default_name}.", STATIC_USDA)

        vessel_prim = stage.GetPrimAtPath("/Phase5StaticScene/Vessel/VesselProxy")
        if vessel_prim.IsValid():
            add_result(rows, category, "Vessel proxy prim", "PASS", "VesselProxy mesh prim exists.", STATIC_USDA)
        else:
            add_result(rows, category, "Vessel proxy prim", "FAIL", "VesselProxy mesh prim is missing.", STATIC_USDA)

        mesh_count = count_mesh_prims_under(stage, "/Phase5StaticScene/Platelets/Platelet_")

        if mesh_count == metadata_count and mesh_count > 0:
            add_result(
                rows,
                category,
                "Static platelet mesh prim count",
                "PASS",
                f"USD mesh prim count matches metadata rows: {mesh_count}.",
                STATIC_USDA,
            )
        elif mesh_count > 0:
            add_result(
                rows,
                category,
                "Static platelet mesh prim count",
                "WARN",
                f"USD mesh prim count={mesh_count}, metadata rows={metadata_count}.",
                STATIC_USDA,
            )
        else:
            add_result(
                rows,
                category,
                "Static platelet mesh prim count",
                "FAIL",
                "No exported platelet mesh prims found.",
                STATIC_USDA,
            )

    return {
        "metadata_count": metadata_count,
        "inactive_count": inactive_count,
        "activated_count": activated_count,
        "mesh_prim_count": mesh_count,
    }


def validate_animated_usd(rows: list[dict[str, Any]]) -> dict[str, Any]:
    category = "Animated USD export"

    check_required_file(rows, category, "Animated USD scene", ANIMATED_USDA)
    check_required_file(rows, category, "Animated metadata CSV", ANIMATED_METADATA_CSV)
    check_required_file(rows, category, "Animated final preview image", ANIMATED_PREVIEW_PNG)
    check_required_file(rows, category, "Activation count plot", ANIMATED_COUNTS_PNG)
    check_required_file(rows, category, "Animated export summary", ANIMATED_SUMMARY_MD)

    stage = open_usd_stage(rows, category, ANIMATED_USDA)

    metadata_rows = read_csv_rows(ANIMATED_METADATA_CSV)
    metadata_count = len(metadata_rows)

    if metadata_count > 0:
        add_result(
            rows,
            category,
            "Animated metadata rows",
            "PASS",
            f"Metadata contains {metadata_count} frame-wise platelet rows.",
            ANIMATED_METADATA_CSV,
        )
    else:
        add_result(
            rows,
            category,
            "Animated metadata rows",
            "FAIL",
            "Animated metadata CSV is empty or missing.",
            ANIMATED_METADATA_CSV,
        )

    summary = {
        "metadata_count": metadata_count,
        "start_time": None,
        "end_time": None,
        "fps": None,
        "ids_count": 0,
        "position_time_samples": 0,
        "scale_time_samples": 0,
        "proto_index_time_samples": 0,
        "prototype_count": 0,
    }

    if stage is None:
        return summary

    default_prim = stage.GetDefaultPrim()
    default_name = default_prim.GetName() if default_prim.IsValid() else "missing"

    if default_name == "Phase5AnimatedScene":
        add_result(rows, category, "Default prim", "PASS", "Default prim is Phase5AnimatedScene.", ANIMATED_USDA)
    else:
        add_result(rows, category, "Default prim", "WARN", f"Default prim is {default_name}.", ANIMATED_USDA)

    start_time = stage.GetStartTimeCode()
    end_time = stage.GetEndTimeCode()
    fps = stage.GetFramesPerSecond()
    time_codes_per_second = stage.GetTimeCodesPerSecond()

    summary["start_time"] = start_time
    summary["end_time"] = end_time
    summary["fps"] = fps

    if start_time == 0 and end_time >= 1:
        add_result(
            rows,
            category,
            "Animation time range",
            "PASS",
            f"startTimeCode={start_time}, endTimeCode={end_time}.",
            ANIMATED_USDA,
        )
    else:
        add_result(
            rows,
            category,
            "Animation time range",
            "WARN",
            f"Unexpected time range: start={start_time}, end={end_time}.",
            ANIMATED_USDA,
        )

    if fps > 0 and time_codes_per_second > 0:
        add_result(
            rows,
            category,
            "Animation FPS metadata",
            "PASS",
            f"framesPerSecond={fps}, timeCodesPerSecond={time_codes_per_second}.",
            ANIMATED_USDA,
        )
    else:
        add_result(
            rows,
            category,
            "Animation FPS metadata",
            "FAIL",
            "FPS or timeCodesPerSecond metadata is missing.",
            ANIMATED_USDA,
        )

    instancer_prim = stage.GetPrimAtPath("/Phase5AnimatedScene/PlateletInstancer")

    if not instancer_prim.IsValid():
        add_result(
            rows,
            category,
            "PointInstancer prim",
            "FAIL",
            "PointInstancer prim is missing.",
            ANIMATED_USDA,
        )
        return summary

    if instancer_prim.GetTypeName() == "PointInstancer":
        add_result(
            rows,
            category,
            "PointInstancer prim",
            "PASS",
            "PointInstancer exists and has correct type.",
            ANIMATED_USDA,
        )
    else:
        add_result(
            rows,
            category,
            "PointInstancer prim",
            "WARN",
            f"Prim exists but type is {instancer_prim.GetTypeName()}.",
            ANIMATED_USDA,
        )

    instancer = UsdGeom.PointInstancer(instancer_prim)

    prototype_targets = instancer.GetPrototypesRel().GetTargets()
    prototype_count = len(prototype_targets)
    summary["prototype_count"] = prototype_count

    if prototype_count == 2:
        add_result(
            rows,
            category,
            "Prototype count",
            "PASS",
            f"Two prototypes found: {prototype_targets}.",
            ANIMATED_USDA,
        )
    else:
        add_result(
            rows,
            category,
            "Prototype count",
            "WARN",
            f"Expected 2 prototypes, found {prototype_count}: {prototype_targets}.",
            ANIMATED_USDA,
        )

    ids = instancer.GetIdsAttr().Get()
    ids_count = len(ids) if ids is not None else 0
    summary["ids_count"] = ids_count

    if ids_count > 0:
        add_result(
            rows,
            category,
            "Instancer IDs",
            "PASS",
            f"PointInstancer contains {ids_count} persistent platelet IDs.",
            ANIMATED_USDA,
        )
    else:
        add_result(
            rows,
            category,
            "Instancer IDs",
            "FAIL",
            "PointInstancer ID list is empty.",
            ANIMATED_USDA,
        )

    positions_attr = instancer.GetPositionsAttr()
    scales_attr = instancer.GetScalesAttr()
    proto_indices_attr = instancer.GetProtoIndicesAttr()

    position_samples = positions_attr.GetTimeSamples()
    scale_samples = scales_attr.GetTimeSamples()
    proto_index_samples = proto_indices_attr.GetTimeSamples()

    summary["position_time_samples"] = len(position_samples)
    summary["scale_time_samples"] = len(scale_samples)
    summary["proto_index_time_samples"] = len(proto_index_samples)

    if len(position_samples) > 1:
        add_result(
            rows,
            category,
            "Position time samples",
            "PASS",
            f"Positions are time-sampled across {len(position_samples)} frames.",
            ANIMATED_USDA,
        )
    else:
        add_result(
            rows,
            category,
            "Position time samples",
            "FAIL",
            "Positions are not time-sampled.",
            ANIMATED_USDA,
        )

    if len(scale_samples) == len(position_samples) and len(scale_samples) > 1:
        add_result(
            rows,
            category,
            "Scale time samples",
            "PASS",
            f"Scales are time-sampled across {len(scale_samples)} frames.",
            ANIMATED_USDA,
        )
    else:
        add_result(
            rows,
            category,
            "Scale time samples",
            "WARN",
            f"Scale samples={len(scale_samples)}, position samples={len(position_samples)}.",
            ANIMATED_USDA,
        )

    if len(proto_index_samples) == len(position_samples) and len(proto_index_samples) > 1:
        add_result(
            rows,
            category,
            "Prototype index time samples",
            "PASS",
            f"Prototype indices are time-sampled across {len(proto_index_samples)} frames.",
            ANIMATED_USDA,
        )
    else:
        add_result(
            rows,
            category,
            "Prototype index time samples",
            "WARN",
            f"Prototype index samples={len(proto_index_samples)}, position samples={len(position_samples)}.",
            ANIMATED_USDA,
        )

    expected_metadata_rows = ids_count * len(position_samples)

    if expected_metadata_rows == metadata_count and expected_metadata_rows > 0:
        add_result(
            rows,
            category,
            "Metadata count consistency",
            "PASS",
            f"Metadata rows match IDs x frames: {ids_count} x {len(position_samples)} = {metadata_count}.",
            ANIMATED_METADATA_CSV,
        )
    else:
        add_result(
            rows,
            category,
            "Metadata count consistency",
            "WARN",
            f"Expected {expected_metadata_rows} rows from IDs x frames, found {metadata_count}.",
            ANIMATED_METADATA_CSV,
        )

    return summary


def write_validation_csv(rows: list[dict[str, Any]]) -> None:
    with OUTPUT_VALIDATION_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "category",
                "item",
                "status",
                "details",
                "path",
                "size_mb",
                "sha256_short",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_validation_markdown(
    rows: list[dict[str, Any]],
    static_summary: dict[str, Any],
    animated_summary: dict[str, Any],
) -> None:
    pass_count = sum(1 for row in rows if row["status"] == "PASS")
    warn_count = sum(1 for row in rows if row["status"] == "WARN")
    fail_count = sum(1 for row in rows if row["status"] == "FAIL")

    lines = []

    lines.append("# Phase 5 Week 4 Day 4: USD Export Validation")
    lines.append("")
    lines.append("## Validation Outcome")
    lines.append("")
    lines.append(f"- PASS checks: {pass_count}")
    lines.append(f"- WARN checks: {warn_count}")
    lines.append(f"- FAIL checks: {fail_count}")
    lines.append("")

    if fail_count == 0:
        lines.append("Overall status: **USD export package is valid for Phase 5 documentation.**")
    else:
        lines.append("Overall status: **USD export package has failed checks that must be reviewed.**")

    lines.append("")
    lines.append("## Static USD Export Summary")
    lines.append("")
    lines.append(f"- Static USD file: `{STATIC_USDA}`")
    lines.append(f"- Exported metadata rows: {static_summary.get('metadata_count', 0)}")
    lines.append(f"- Inactive exported platelets: {static_summary.get('inactive_count', 0)}")
    lines.append(f"- Activated exported platelets: {static_summary.get('activated_count', 0)}")
    lines.append(f"- USD platelet mesh prims: {static_summary.get('mesh_prim_count', 0)}")
    lines.append("")

    lines.append("## Animated USD Export Summary")
    lines.append("")
    lines.append(f"- Animated USD file: `{ANIMATED_USDA}`")
    lines.append(f"- Metadata rows: {animated_summary.get('metadata_count', 0)}")
    lines.append(f"- Start time code: {animated_summary.get('start_time')}")
    lines.append(f"- End time code: {animated_summary.get('end_time')}")
    lines.append(f"- Frames per second: {animated_summary.get('fps')}")
    lines.append(f"- PointInstancer IDs: {animated_summary.get('ids_count', 0)}")
    lines.append(f"- Position time samples: {animated_summary.get('position_time_samples', 0)}")
    lines.append(f"- Scale time samples: {animated_summary.get('scale_time_samples', 0)}")
    lines.append(f"- Prototype-index time samples: {animated_summary.get('proto_index_time_samples', 0)}")
    lines.append(f"- Prototype count: {animated_summary.get('prototype_count', 0)}")
    lines.append("")

    lines.append("## Detailed Check Results")
    lines.append("")
    lines.append("| Category | Item | Status | Details |")
    lines.append("|---|---|---:|---|")

    for row in rows:
        lines.append(
            f"| {row['category']} | {row['item']} | {row['status']} | {row['details']} |"
        )

    lines.append("")
    lines.append("## Final Thesis Interpretation")
    lines.append("")
    lines.append(
        "The Phase 5 USD export pipeline produced both a static USD scene and an animation-ready USD scene. "
        "The animated export uses `UsdGeom.PointInstancer` with time-sampled positions, scales, and prototype indices. "
        "This supports an Omniverse/USD-compatible visualization workflow for the platelet simulation."
    )
    lines.append("")
    lines.append("## Important Limitation")
    lines.append("")
    lines.append(
        "This validation confirms that the USD files were generated and can be opened by Pixar USD Python tools. "
        "Manual visual inspection inside NVIDIA Omniverse should be performed separately if Omniverse is available. "
        "If Omniverse is not available, the export should be described as USD-ready rather than Omniverse-rendered."
    )

    OUTPUT_VALIDATION_MD.write_text("\n".join(lines), encoding="utf-8")


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("\nValidation results")
    print("------------------")

    for row in rows:
        print(
            f"{row['status']:5s} | {row['category']:22s} | "
            f"{row['item']:32s} | {row['details']}"
        )

    pass_count = sum(1 for row in rows if row["status"] == "PASS")
    warn_count = sum(1 for row in rows if row["status"] == "WARN")
    fail_count = sum(1 for row in rows if row["status"] == "FAIL")

    print("\nValidation summary")
    print("------------------")
    print(f"PASS: {pass_count}")
    print(f"WARN: {warn_count}")
    print(f"FAIL: {fail_count}")


def main() -> None:
    print("Phase 5 Week 4 Day 4: USD export validation")

    WEEK4_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    static_summary = validate_static_usd(rows)
    animated_summary = validate_animated_usd(rows)

    write_validation_csv(rows)
    write_validation_markdown(rows, static_summary, animated_summary)

    print_summary(rows)

    print("\nSaved validation report:")
    print(OUTPUT_VALIDATION_MD)
    print(OUTPUT_VALIDATION_CSV)

    print("\nWeek 4 Day 4 USD export validation complete.")


if __name__ == "__main__":
    main()