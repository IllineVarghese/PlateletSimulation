from pathlib import Path
import csv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

WEEK3_DIR = PROJECT_ROOT / "results" / "phase5" / "week3"

OUTPUT_REVIEW_MD = WEEK3_DIR / "week3_output_review.md"
OUTPUT_REVIEW_CSV = WEEK3_DIR / "week3_output_review.csv"

DECIMATION_REPORT = WEEK3_DIR / "mesh_decimation_report.csv"
PERFORMANCE_REPORT = WEEK3_DIR / "original_vs_decimated_mesh_performance.csv"


EXPECTED_OUTPUTS = [
    {
        "file": "activated_deformation_test.png",
        "status": "exploratory",
        "use": "Initial visual deformation test; useful for development notes, not final thesis figure.",
    },
    {
        "file": "advanced_activation_deformation_progression.png",
        "status": "selected",
        "use": "Use for thesis/presentation to explain activation-dependent visual deformation.",
    },
    {
        "file": "advanced_activation_deformation_metrics.csv",
        "status": "supporting",
        "use": "Supporting metrics for deformation progression.",
    },
    {
        "file": "mesh_decimation_comparison.png",
        "status": "selected",
        "use": "Use for thesis/presentation to show original vs optimized mesh complexity.",
    },
    {
        "file": "mesh_decimation_report.csv",
        "status": "supporting",
        "use": "Supporting data for mesh reduction values.",
    },
    {
        "file": "original_vs_decimated_mesh_performance.png",
        "status": "selected",
        "use": "Use for thesis/presentation to show rendering speedup from mesh decimation.",
    },
    {
        "file": "original_vs_decimated_mesh_performance.csv",
        "status": "supporting",
        "use": "Supporting data for original vs decimated performance comparison.",
    },
    {
        "file": "_performance_original_100.png",
        "status": "supporting",
        "use": "Visual comparison example using original meshes.",
    },
    {
        "file": "_performance_decimated_100.png",
        "status": "supporting",
        "use": "Visual comparison example using decimated meshes.",
    },
    {
        "file": "phase5_decimated_deformed_thesis_video.mp4",
        "status": "selected",
        "use": "Best Week 3 video output: optimized mesh-based platelet activation visualization.",
    },
]


def file_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0

    return path.stat().st_size / (1024 * 1024)


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def summarize_decimation() -> str:
    rows = read_csv_rows(DECIMATION_REPORT)

    if not rows:
        return "Decimation report not found."

    summaries = []

    for row in rows:
        name = row.get("name", "")
        if "decimated" not in name:
            continue

        reduction = float(row.get("cell_reduction_percent", 0.0))
        points = int(float(row.get("points", 0)))
        cells = int(float(row.get("cells", 0)))

        summaries.append(
            f"{name}: cell reduction={reduction:.1f}%, points={points}, cells={cells}"
        )

    if not summaries:
        return "No decimated mesh rows found in report."

    return "\n".join(summaries)


def summarize_performance() -> str:
    rows = read_csv_rows(PERFORMANCE_REPORT)

    if not rows:
        return "Performance report not found."

    original_200 = None
    decimated_200 = None

    for row in rows:
        mode = row.get("mesh_mode", "")
        count = int(float(row.get("rendered_platelets", 0)))
        time_seconds = float(row.get("render_time_seconds", 0.0))

        if count == 200 and mode == "original":
            original_200 = time_seconds

        if count == 200 and mode == "decimated":
            decimated_200 = time_seconds

    if original_200 is None or decimated_200 is None or decimated_200 <= 0:
        return "Could not compute 200-platelet speedup from report."

    speedup = original_200 / decimated_200

    return (
        f"At 200 rendered platelets, original meshes required {original_200:.3f} s, "
        f"while decimated meshes required {decimated_200:.3f} s. "
        f"Approximate speedup: {speedup:.2f}x."
    )


def write_review_csv(rows: list[dict]) -> None:
    with OUTPUT_REVIEW_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "file",
                "exists",
                "size_mb",
                "status",
                "recommended_use",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def write_review_markdown(rows: list[dict]) -> None:
    decimation_summary = summarize_decimation()
    performance_summary = summarize_performance()

    selected = [row for row in rows if row["status"] == "selected"]
    missing = [row for row in rows if not row["exists"]]

    lines = []

    lines.append("# Phase 5 Week 3 Output Review")
    lines.append("")
    lines.append("## Week 3 Theme")
    lines.append("")
    lines.append(
        "Week 3 focused on improving biological readability and rendering performance "
        "for activation-based platelet mesh visualization."
    )
    lines.append("")
    lines.append("## Main Completed Tasks")
    lines.append("")
    lines.append("- Activation-dependent visual deformation model")
    lines.append("- Pseudo-filopodia overlay for highly activated platelets")
    lines.append("- Mesh decimation for visualization optimization")
    lines.append("- Original vs decimated mesh rendering performance comparison")
    lines.append("- Optimized thesis-style platelet activation video")
    lines.append("")
    lines.append("## Selected Final Week 3 Outputs")
    lines.append("")

    for row in selected:
        lines.append(f"- `{row['file']}` — {row['recommended_use']}")

    lines.append("")
    lines.append("## Decimation Summary")
    lines.append("")
    lines.append("```text")
    lines.append(decimation_summary)
    lines.append("```")
    lines.append("")
    lines.append("## Performance Summary")
    lines.append("")
    lines.append("```text")
    lines.append(performance_summary)
    lines.append("```")
    lines.append("")
    lines.append("## Important Scientific Limitation")
    lines.append("")
    lines.append(
        "The activated deformation and protrusion overlays are visual-only morphology "
        "enhancements. They are not biomechanical soft-body simulations. They are used "
        "to make activation state transitions more readable in dense visualization scenes."
    )
    lines.append("")
    lines.append("## Recommended Usage")
    lines.append("")
    lines.append("- Use original meshes for close-up morphology figures.")
    lines.append("- Use decimated meshes for dense scenes, videos, scaling tests, and later USD/Omniverse export tests.")
    lines.append("- Use activation coloring and limited protrusion overlays for presentation-level videos.")
    lines.append("")
    lines.append("## Missing Outputs")
    lines.append("")

    if missing:
        for row in missing:
            lines.append(f"- Missing: `{row['file']}`")
    else:
        lines.append("No expected Week 3 outputs are missing.")

    OUTPUT_REVIEW_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("Phase 5 Week 3 Day 5: Output review and final checkpoint")

    WEEK3_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for item in EXPECTED_OUTPUTS:
        path = WEEK3_DIR / item["file"]

        rows.append(
            {
                "file": item["file"],
                "exists": path.exists(),
                "size_mb": f"{file_size_mb(path):.3f}",
                "status": item["status"],
                "recommended_use": item["use"],
            }
        )

    write_review_csv(rows)
    write_review_markdown(rows)

    print("\nWeek 3 output check")
    print("-------------------")

    for row in rows:
        symbol = "OK" if row["exists"] else "MISSING"
        print(
            f"{symbol:7s} | {row['file']:55s} | "
            f"{row['status']:10s} | {row['size_mb']} MB"
        )

    print("\nDecimation summary")
    print("------------------")
    print(summarize_decimation())

    print("\nPerformance summary")
    print("-------------------")
    print(summarize_performance())

    print("\nSaved review files:")
    print(OUTPUT_REVIEW_MD)
    print(OUTPUT_REVIEW_CSV)

    print("\nWeek 3 Day 5 review complete.")


if __name__ == "__main__":
    main()