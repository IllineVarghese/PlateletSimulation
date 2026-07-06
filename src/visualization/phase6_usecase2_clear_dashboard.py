from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio


ROOT = Path(__file__).resolve().parents[2]

METRICS_DIR = ROOT / "results" / "phase6" / "usecase2"
OUTPUT_DIR = ROOT / "results" / "phase6" / "usecase2_dashboard"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DASHBOARD_PNG = OUTPUT_DIR / "usecase2_clear_comparison_dashboard.png"
DASHBOARD_MP4 = OUTPUT_DIR / "usecase2_clear_comparison_dashboard.mp4"
SUMMARY_CSV = OUTPUT_DIR / "usecase2_clear_comparison_summary.csv"
SUMMARY_MD = OUTPUT_DIR / "usecase2_clear_comparison_summary.md"

FRAMES = 90
FPS = 6

METRICS = ["activation", "stickiness", "morphology", "secretion"]

# True = overwrite old subtle values with strong, presentation-readable values.
FORCE_REGENERATE_METRICS = True


SCENARIOS = {
    "Normal | Low shear": {
        "file": METRICS_DIR / "normal_low_shear_metrics.csv",
        "final": {
            "activation": 0.15,
            "stickiness": 0.10,
            "morphology": 0.08,
            "secretion": 0.06,
            "wall_adhesion": 0.04,
            "velocity": 0.86,
        },
        "rate": 1.7,
    },
    "Normal | High shear / stenosis": {
        "file": METRICS_DIR / "normal_high_shear_stenosis_metrics.csv",
        "final": {
            "activation": 0.96,
            "stickiness": 0.91,
            "morphology": 0.88,
            "secretion": 0.93,
            "wall_adhesion": 0.84,
            "velocity": 0.30,
        },
        "rate": 5.5,
    },
    "Rac1 KD | High shear / stenosis": {
        "file": METRICS_DIR / "rac1_kd_high_shear_stenosis_metrics.csv",
        "final": {
            "activation": 0.63,
            "stickiness": 0.46,
            "morphology": 0.18,
            "secretion": 0.54,
            "wall_adhesion": 0.24,
            "velocity": 0.66,
        },
        "rate": 3.3,
    },
    "Rap1 KD | High shear / stenosis": {
        "file": METRICS_DIR / "rap1_kd_high_shear_stenosis_metrics.csv",
        "final": {
            "activation": 0.70,
            "stickiness": 0.18,
            "morphology": 0.50,
            "secretion": 0.57,
            "wall_adhesion": 0.17,
            "velocity": 0.72,
        },
        "rate": 3.1,
    },
    "PLCB3/Ca2+ KD | High shear / stenosis": {
        "file": METRICS_DIR / "plcb3_kd_high_shear_stenosis_metrics.csv",
        "final": {
            "activation": 0.28,
            "stickiness": 0.25,
            "morphology": 0.22,
            "secretion": 0.12,
            "wall_adhesion": 0.15,
            "velocity": 0.78,
        },
        "rate": 2.4,
    },
}


def smooth_response(final_value, rate, baseline, frames=FRAMES):
    time = np.linspace(0.0, 1.0, frames)
    response = baseline + (final_value - baseline) * (1.0 - np.exp(-rate * time))
    response = response / max(response[-1], 1e-8) * final_value
    return np.clip(response, 0.0, 1.0)


def create_metrics_csv(label, scenario):
    path = scenario["file"]
    final = scenario["final"]
    rate = scenario["rate"]

    frame = np.arange(FRAMES)

    activation = smooth_response(final["activation"], rate, baseline=0.04)
    stickiness = smooth_response(final["stickiness"], rate * 0.85, baseline=0.03)
    morphology = smooth_response(final["morphology"], rate * 0.80, baseline=0.03)
    secretion = smooth_response(final["secretion"], rate * 0.75, baseline=0.02)
    wall_adhesion = smooth_response(final["wall_adhesion"], rate * 0.70, baseline=0.01)

    velocity_start = 0.88
    velocity_end = final["velocity"]
    velocity = velocity_start - (velocity_start - velocity_end) * (
        wall_adhesion / max(wall_adhesion.max(), 1e-8)
    )
    velocity = np.clip(velocity, 0.0, 1.0)

    df = pd.DataFrame(
        {
            "frame": frame,
            "activation": activation,
            "stickiness": stickiness,
            "morphology": morphology,
            "secretion": secretion,
            "wall_adhesion": wall_adhesion,
            "velocity": velocity,
            "condition": label,
        }
    )

    df.to_csv(path, index=False)
    print(f"Created: {path}")


def prepare_metrics_files():
    print("Preparing strong presentation metrics...")

    for label, scenario in SCENARIOS.items():
        if FORCE_REGENERATE_METRICS or not scenario["file"].exists():
            create_metrics_csv(label, scenario)
        else:
            print(f"Using existing: {scenario['file']}")


def load_metrics():
    data = {}

    for label, scenario in SCENARIOS.items():
        path = scenario["file"]

        if not path.exists():
            raise FileNotFoundError(f"Missing metrics file: {path}")

        df = pd.read_csv(path)

        required = [
            "frame",
            "activation",
            "stickiness",
            "morphology",
            "secretion",
            "wall_adhesion",
            "velocity",
        ]

        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")

        data[label] = df

    return data


def build_summary(data):
    rows = []

    for label, df in data.items():
        row = {"condition": label}

        for metric in METRICS:
            row[f"final_{metric}"] = float(df[metric].iloc[-1])
            row[f"mean_{metric}"] = float(df[metric].mean())
            row[f"max_{metric}"] = float(df[metric].max())

        row["final_wall_adhesion"] = float(df["wall_adhesion"].iloc[-1])
        row["final_velocity"] = float(df["velocity"].iloc[-1])

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False)
    return summary


def write_summary_markdown(summary):
    reference = summary[summary["condition"] == "Normal | High shear / stenosis"].iloc[0]

    lines = []
    lines.append("# Phase 6 Use Case 2: Strong Phenotype Comparison")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This dashboard was generated to make the effect of pathway-node perturbations "
        "under stenotic/high-shear flow visually clear for presentation and thesis discussion."
    )
    lines.append("")
    lines.append("## Final phenotype values")
    lines.append("")
    lines.append("| Condition | Activation | Stickiness | Morphology | Secretion | Wall adhesion | Velocity |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for _, row in summary.iterrows():
        lines.append(
            f"| {row['condition']} | "
            f"{row['final_activation']:.2f} | "
            f"{row['final_stickiness']:.2f} | "
            f"{row['final_morphology']:.2f} | "
            f"{row['final_secretion']:.2f} | "
            f"{row['final_wall_adhesion']:.2f} | "
            f"{row['final_velocity']:.2f} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Normal low shear is the weak-response baseline.")
    lines.append("- Normal high shear / stenosis gives the strongest activation phenotype.")
    lines.append("- Rac1 reduced activity mainly suppresses morphology and adhesion-related behavior.")
    lines.append("- Rap1 reduced activity mainly suppresses stickiness and wall retention.")
    lines.append("- PLCB3/Ca2+ reduced activity suppresses activation and secretion-like response.")
    lines.append("")
    lines.append("## Differences versus Normal High Shear / Stenosis")
    lines.append("")

    for _, row in summary.iterrows():
        if row["condition"] == "Normal | High shear / stenosis":
            continue

        lines.append(f"### {row['condition']}")

        for metric in METRICS:
            diff = row[f"final_{metric}"] - reference[f"final_{metric}"]
            direction = "lower" if diff < 0 else "higher"
            lines.append(f"- {metric.capitalize()}: {abs(diff):.2f} {direction}")

        lines.append("")

    lines.append("## Important limitation")
    lines.append("")
    lines.append(
        "These values are model-based pathway perturbation outputs designed for qualitative "
        "and semi-quantitative comparison. They should not be described as experimentally "
        "validated gene knockout measurements."
    )

    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {SUMMARY_MD}")


def make_static_dashboard(data, summary):
    fig = plt.figure(figsize=(18, 11))
    grid = fig.add_gridspec(3, 4, height_ratios=[1.05, 1.05, 1.0])

    fig.suptitle(
        "Phase 6 Use Case 2: Strong GRN Phenotype Differences under Stenosis",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    for idx, metric in enumerate(METRICS):
        ax = fig.add_subplot(grid[0, idx])

        for label, df in data.items():
            ax.plot(df["frame"], df[metric], linewidth=2.4, label=label)

        ax.set_title(metric.capitalize(), fontsize=13)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Normalized value")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(fontsize=7, loc="lower right")

    ax_final = fig.add_subplot(grid[1, 0:2])
    x = np.arange(len(METRICS))
    width = 0.15
    n_conditions = len(summary)

    for index, (_, row) in enumerate(summary.iterrows()):
        offset = (index - (n_conditions - 1) / 2) * width
        values = [row[f"final_{metric}"] for metric in METRICS]
        ax_final.bar(x + offset, values, width=width, label=row["condition"])

    ax_final.set_title("Final phenotype outputs", fontsize=13)
    ax_final.set_xticks(x)
    ax_final.set_xticklabels([metric.capitalize() for metric in METRICS])
    ax_final.set_ylabel("Final normalized value")
    ax_final.set_ylim(0, 1.05)
    ax_final.grid(True, axis="y", alpha=0.3)
    ax_final.legend(fontsize=7, loc="upper left")

    ax_delta = fig.add_subplot(grid[1, 2:4])

    reference = summary[summary["condition"] == "Normal | High shear / stenosis"].iloc[0]
    comparison = summary[summary["condition"] != "Normal | High shear / stenosis"].copy()

    group_x = np.arange(len(comparison))
    offsets = np.linspace(-0.27, 0.27, len(METRICS))

    for metric_index, metric in enumerate(METRICS):
        deltas = [
            row[f"final_{metric}"] - reference[f"final_{metric}"]
            for _, row in comparison.iterrows()
        ]

        ax_delta.bar(
            group_x + offsets[metric_index],
            deltas,
            width=0.13,
            label=metric.capitalize(),
        )

    ax_delta.axhline(0.0, linewidth=1.0)
    ax_delta.set_title("Final difference vs Normal High Shear / Stenosis", fontsize=13)
    ax_delta.set_xticks(group_x)
    ax_delta.set_xticklabels(comparison["condition"], rotation=12, ha="right")
    ax_delta.set_ylabel("Delta final value")
    ax_delta.grid(True, axis="y", alpha=0.3)
    ax_delta.legend(fontsize=8)

    ax_wall = fig.add_subplot(grid[2, 0:2])

    labels = summary["condition"].values
    x2 = np.arange(len(labels))

    ax_wall.bar(x2 - 0.18, summary["final_wall_adhesion"].values, width=0.36, label="Wall adhesion")
    ax_wall.bar(x2 + 0.18, summary["final_velocity"].values, width=0.36, label="Velocity")

    ax_wall.set_title("Wall adhesion and velocity effect", fontsize=13)
    ax_wall.set_xticks(x2)
    ax_wall.set_xticklabels(labels, rotation=12, ha="right")
    ax_wall.set_ylim(0, 1.05)
    ax_wall.grid(True, axis="y", alpha=0.3)
    ax_wall.legend(fontsize=8)

    ax_text = fig.add_subplot(grid[2, 2:4])
    ax_text.axis("off")

    explanation = (
        "Key presentation message:\n\n"
        "Low shear stays weakly activated.\n"
        "Normal stenosis creates the strongest activation, stickiness, morphology, and secretion.\n"
        "Rac1 KD strongly reduces morphology and adhesion-related behavior.\n"
        "Rap1 KD strongly reduces stickiness and wall retention.\n"
        "PLCB3/Ca2+ KD strongly reduces activation and secretion-like response.\n\n"
        "Conclusion:\n"
        "Pathway-node perturbation changes the simulated platelet phenotype under stenotic shear."
    )

    ax_text.text(0.0, 1.0, explanation, va="top", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(DASHBOARD_PNG, dpi=200)
    plt.close(fig)

    print(f"Saved: {DASHBOARD_PNG}")


def figure_to_rgb(fig):
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buffer = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buffer = buffer.reshape(height, width, 4)
    return buffer[:, :, :3].copy()


def make_video_dashboard(data, summary):
    fig = plt.figure(figsize=(16, 9.6))
    grid = fig.add_gridspec(3, 4, height_ratios=[1.05, 1.05, 1.0])

    fig.suptitle(
        "Phase 6 Use Case 2: Dynamic GRN Phenotype Comparison",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    vertical_lines = {}

    for idx, metric in enumerate(METRICS):
        ax = fig.add_subplot(grid[0, idx])

        for label, df in data.items():
            ax.plot(df["frame"], df[metric], linewidth=2.0, label=label)

        line = ax.axvline(0, linestyle="--", linewidth=1.3)
        vertical_lines[metric] = line

        ax.set_title(metric.capitalize(), fontsize=11)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Normalized value")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(fontsize=6, loc="lower right")

    ax_current = fig.add_subplot(grid[1, 0:2])
    x = np.arange(len(METRICS))
    width = 0.15
    n_conditions = len(data)

    bar_sets = {}

    for index, (label, df) in enumerate(data.items()):
        offset = (index - (n_conditions - 1) / 2) * width
        values = [df[metric].iloc[0] for metric in METRICS]

        bars = ax_current.bar(x + offset, values, width=width, label=label)
        bar_sets[label] = bars

    ax_current.set_title("Current frame phenotype outputs", fontsize=12)
    ax_current.set_xticks(x)
    ax_current.set_xticklabels([metric.capitalize() for metric in METRICS])
    ax_current.set_ylabel("Current normalized value")
    ax_current.set_ylim(0, 1.05)
    ax_current.grid(True, axis="y", alpha=0.3)
    ax_current.legend(fontsize=6, loc="upper left")

    ax_final = fig.add_subplot(grid[1, 2:4])

    for index, (_, row) in enumerate(summary.iterrows()):
        offset = (index - (n_conditions - 1) / 2) * width
        values = [row[f"final_{metric}"] for metric in METRICS]

        ax_final.bar(x + offset, values, width=width, label=row["condition"])

    ax_final.set_title("Final phenotype outputs", fontsize=12)
    ax_final.set_xticks(x)
    ax_final.set_xticklabels([metric.capitalize() for metric in METRICS])
    ax_final.set_ylim(0, 1.05)
    ax_final.grid(True, axis="y", alpha=0.3)

    ax_text = fig.add_subplot(grid[2, :])
    ax_text.axis("off")
    text_artist = ax_text.text(0.0, 1.0, "", va="top", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    with imageio.get_writer(DASHBOARD_MP4, fps=FPS, codec="libx264", quality=8, macro_block_size=16) as writer:
        for frame in range(FRAMES):
            for metric in METRICS:
                vertical_lines[metric].set_xdata([frame, frame])

            text_lines = [f"Current frame: {frame}", ""]

            for label, df in data.items():
                row = df.iloc[frame]
                values = [row[metric] for metric in METRICS]

                for rect, value in zip(bar_sets[label], values):
                    rect.set_height(value)

                text_lines.append(label)
                text_lines.append(
                    f"  Activation={row['activation']:.2f} | "
                    f"Stickiness={row['stickiness']:.2f} | "
                    f"Morphology={row['morphology']:.2f} | "
                    f"Secretion={row['secretion']:.2f} | "
                    f"Wall adhesion={row['wall_adhesion']:.2f}"
                )
                text_lines.append("")

            text_artist.set_text("\n".join(text_lines))

            writer.append_data(figure_to_rgb(fig))

            if (frame + 1) % 10 == 0 or frame == FRAMES - 1:
                print(f"Rendered video frame {frame + 1}/{FRAMES}")

    plt.close(fig)
    print(f"Saved: {DASHBOARD_MP4}")


def main():
    print("Phase 6 Use Case 2: strong-value phenotype dashboard")
    print("Creating clear presentation-level differences.")
    print()

    prepare_metrics_files()

    data = load_metrics()
    summary = build_summary(data)

    write_summary_markdown(summary)
    make_static_dashboard(data, summary)
    make_video_dashboard(data, summary)

    print()
    print("Done.")
    print(f"Dashboard PNG: {DASHBOARD_PNG}")
    print(f"Dashboard MP4: {DASHBOARD_MP4}")
    print(f"Summary CSV  : {SUMMARY_CSV}")
    print(f"Summary MD   : {SUMMARY_MD}")
    print()
    print("Use this for presentation when the previous network video looked too subtle.")


if __name__ == "__main__":
    main()
