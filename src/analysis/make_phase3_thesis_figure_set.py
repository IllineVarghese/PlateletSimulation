from pathlib import Path
import shutil

FIGURE_DIR = Path("results/month3/thesis_figure_set")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    # GRN structure
    "results/analysis/grn_paper_style/platelet_grn_paper_network.png":
        "Figure_3_1_platelet_grn_network.png",

    # GRN dynamics
    "results/analysis/grn_paper_style/low_shear_low_chemical_selected_nodes.png":
        "Figure_3_2_low_shear_grn_dynamics.png",

    "results/analysis/grn_paper_style/high_shear_low_chemical_selected_nodes.png":
        "Figure_3_3_high_shear_grn_dynamics.png",

    "results/analysis/grn_paper_style/high_shear_high_chemical_selected_nodes.png":
        "Figure_3_4_high_shear_high_chemical_grn_dynamics.png",

    # Shear comparison
    "results/analysis/shear_response_comparison/compare_stickiness.png":
        "Figure_3_5_shear_vs_stickiness.png",

    "results/analysis/shear_response_comparison/compare_secretion.png":
        "Figure_3_6_shear_vs_secretion.png",

    "results/analysis/shear_response_comparison/compare_morphology.png":
        "Figure_3_7_shear_vs_morphology.png",

    # Network pathway views
    "results/analysis/grn_network/grn_full_network.png":
        "Figure_3_8_full_grn_network.png",

    "results/analysis/grn_network/grn_mechanical_pathway.png":
        "Figure_3_9_mechanical_pathway.png",

    "results/analysis/grn_network/grn_chemical_feedback_pathway.png":
        "Figure_3_10_chemical_feedback_pathway.png",

    # Final visualization
    "results/month3/month3_behavior_shear_analysis_3d_snapshot.png":
        "Figure_3_11_3d_behavior_snapshot.png",

    "results/month3/month3_behavior_shear_analysis_3d.mp4":
        "Video_3_1_3d_grn_platelet_behavior.mp4",
}


def main():
    print("\n=== Creating Phase 3 thesis figure set ===\n")

    copied = []
    missing = []

    for src, dst_name in FILES.items():
        src_path = Path(src)
        dst_path = FIGURE_DIR / dst_name

        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            copied.append(dst_path)
            print(f"Copied: {dst_path}")
        else:
            missing.append(src_path)
            print(f"Missing: {src_path}")

    readme = FIGURE_DIR / "README_phase3_figure_set.md"

    with open(readme, "w", encoding="utf-8") as f:
        f.write("# Phase 3 Thesis Figure Set\n\n")
        f.write("This folder contains the final selected figures and video outputs for Phase 3.\n\n")

        f.write("## Figures\n\n")
        f.write("| Figure | File | Purpose |\n")
        f.write("|---|---|---|\n")
        f.write("| Figure 3.1 | Figure_3_1_platelet_grn_network.png | Paper-style platelet GRN topology with activating and inhibitory interactions |\n")
        f.write("| Figure 3.2 | Figure_3_2_low_shear_grn_dynamics.png | GRN node dynamics under low shear and low chemical stimulation |\n")
        f.write("| Figure 3.3 | Figure_3_3_high_shear_grn_dynamics.png | GRN node dynamics under high shear and low chemical stimulation |\n")
        f.write("| Figure 3.4 | Figure_3_4_high_shear_high_chemical_grn_dynamics.png | GRN node dynamics under combined high shear and chemical stimulation |\n")
        f.write("| Figure 3.5 | Figure_3_5_shear_vs_stickiness.png | Shear response of adhesion/stickiness output |\n")
        f.write("| Figure 3.6 | Figure_3_6_shear_vs_secretion.png | Shear response of secretion output |\n")
        f.write("| Figure 3.7 | Figure_3_7_shear_vs_morphology.png | Shear response of morphology output |\n")
        f.write("| Figure 3.8 | Figure_3_8_full_grn_network.png | Full GRN pathway visualization |\n")
        f.write("| Figure 3.9 | Figure_3_9_mechanical_pathway.png | Mechanical/shear sensing pathway |\n")
        f.write("| Figure 3.10 | Figure_3_10_chemical_feedback_pathway.png | Chemical feedback pathway |\n")
        f.write("| Figure 3.11 | Figure_3_11_3d_behavior_snapshot.png | Final 3D GRN-driven platelet behavior snapshot |\n\n")

        f.write("## Video\n\n")
        f.write("| Video | File | Purpose |\n")
        f.write("|---|---|---|\n")
        f.write("| Video 3.1 | Video_3_1_3d_grn_platelet_behavior.mp4 | Final 3D animation showing GRN-driven platelet behavior under shear flow |\n\n")

        f.write("## Notes\n\n")
        f.write("- Node colors in the GRN network represent input, signaling, behavior-program, and output nodes.\n")
        f.write("- Green edges represent activating regulation.\n")
        f.write("- Red edges represent inhibitory regulation.\n")
        f.write("- Time-series plots show normalized GRN node states over simulation time.\n")
        f.write("- The 3D visualization maps secretion to color and morphology to particle size.\n\n")

        if missing:
            f.write("## Missing files during generation\n\n")
            for m in missing:
                f.write(f"- `{m}`\n")

    print("\n=== Done ===")
    print(f"Figure set saved in: {FIGURE_DIR}")
    print(f"README saved as: {readme}")

    if missing:
        print("\nSome files were missing. That is okay if you have not generated them yet.")
        print("Missing files:")
        for m in missing:
            print(f"- {m}")


if __name__ == "__main__":
    main()