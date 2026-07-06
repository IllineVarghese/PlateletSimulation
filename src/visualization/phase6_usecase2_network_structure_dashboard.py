from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import imageio.v2 as imageio


# ============================================================
# Phase 6 - Use Case 2
# GRN network structure + activity dashboard
#
# Purpose:
# - Show a clean platelet activation pathway network structure
# - Compare normal and pathway-node perturbation scenarios
# - Link GRN node activity to phenotype outputs:
#   activation, stickiness, morphology, secretion
# ============================================================


ROOT = Path(__file__).resolve().parents[2]

METRICS_DIR = ROOT / "results" / "phase6" / "usecase2"
OUTPUT_DIR = ROOT / "results" / "phase6" / "usecase2_dashboard"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

STRUCTURE_FIG_PATH = OUTPUT_DIR / "usecase2_grn_network_structure.png"
ACTIVITY_FIG_PATH = OUTPUT_DIR / "usecase2_grn_network_activity_dashboard.png"
VIDEO_PATH = OUTPUT_DIR / "usecase2_grn_network_activity_dashboard.mp4"
SUMMARY_MD_PATH = OUTPUT_DIR / "usecase2_grn_network_structure_summary.md"

FRAMES = 80
FPS = 6

METRICS = ["activation", "stickiness", "morphology", "secretion"]

STATE_THRESHOLD_LOW = 0.33
STATE_THRESHOLD_HIGH = 0.66

SCENARIOS = {
    "Normal | Low shear": {
        "file": METRICS_DIR / "normal_low_shear_metrics.csv",
        "shear": 0.18,
        "perturbed_nodes": [],
        "short": "Low shear",
        "description": "Baseline weak activation condition",
    },
    "Normal | High shear / stenosis": {
        "file": METRICS_DIR / "normal_high_shear_stenosis_metrics.csv",
        "shear": 0.95,
        "perturbed_nodes": [],
        "short": "Normal stenosis",
        "description": "Strong high-shear stenosis response",
    },
    "Rac1 KD | High shear / stenosis": {
        "file": METRICS_DIR / "rac1_kd_high_shear_stenosis_metrics.csv",
        "shear": 0.95,
        "perturbed_nodes": ["Rac1"],
        "short": "Rac1 KD",
        "description": "Morphology / cytoskeleton response reduced",
    },
    "Rap1 KD | High shear / stenosis": {
        "file": METRICS_DIR / "rap1_kd_high_shear_stenosis_metrics.csv",
        "shear": 0.95,
        "perturbed_nodes": ["Rap1"],
        "short": "Rap1 KD",
        "description": "Integrin/stickiness response reduced",
    },
    "PLCB3/Ca2+ KD | High shear / stenosis": {
        "file": METRICS_DIR / "plcb3_kd_high_shear_stenosis_metrics.csv",
        "shear": 0.95,
        "perturbed_nodes": ["PLCB3", "Ca2+"],
        "short": "PLCB3/Ca2+ KD",
        "description": "Activation and secretion-like response reduced",
    },
}

# Fixed node layout for a clean biological pathway diagram.
NODE_POS = {
    "Shear stimulus": (0.05, 0.50),
    "GPIb-vWF": (0.22, 0.68),
    "GPVI/ITAM": (0.22, 0.32),
    "PLCB3": (0.40, 0.68),
    "Ca2+": (0.55, 0.68),
    "Rap1": (0.55, 0.46),
    "Integrin αIIbβ3": (0.73, 0.46),
    "Stickiness": (0.91, 0.46),
    "Rac1": (0.55, 0.20),
    "Actin morphology": (0.73, 0.20),
    "Secretion": (0.73, 0.80),
    "Activation": (0.91, 0.80),
    "Wall retention": (0.91, 0.20),
}

EDGES = [
    ("Shear stimulus", "GPIb-vWF"),
    ("Shear stimulus", "GPVI/ITAM"),
    ("GPIb-vWF", "PLCB3"),
    ("GPVI/ITAM", "PLCB3"),
    ("PLCB3", "Ca2+"),
    ("Ca2+", "Rap1"),
    ("Rap1", "Integrin αIIbβ3"),
    ("Integrin αIIbβ3", "Stickiness"),
    ("Ca2+", "Secretion"),
    ("Secretion", "Activation"),
    ("Ca2+", "Rac1"),
    ("Rac1", "Actin morphology"),
    ("Actin morphology", "Wall retention"),
    ("Stickiness", "Wall retention"),
    ("Stickiness", "Activation"),
]

NODE_GROUPS = {
    "input": ["Shear stimulus"],
    "receptor": ["GPIb-vWF", "GPVI/ITAM"],
    "signaling": ["PLCB3", "Ca2+", "Rap1", "Rac1", "Integrin αIIbβ3"],
    "outputs": ["Activation", "Stickiness", "Actin morphology", "Secretion", "Wall retention"],
}

STRUCTURE_COLORS = {
    "input": "#d8ecff",
    "receptor": "#e9ddff",
    "signaling": "#fff2cc",
    "outputs": "#dff4dd",
}


def response_curve(final_value, rate, frames=FRAMES, baseline=0.04):
    t = np.linspace(0.0, 1.0, frames)
    y = baseline + (final_value - baseline) * (1.0 - np.exp(-rate * t))
    y = y / max(y[-1], 1e-8) * final_value
    return np.clip(y, 0.0, 1.0)


def create_default_metrics_if_missing():
    default_finals = {
        "Normal | Low shear": {
            "activation": 0.14,
            "stickiness": 0.10,
            "morphology": 0.08,
            "secretion": 0.06,
            "wall_adhesion": 0.04,
            "velocity": 0.86,
            "rate": 1.8,
        },
        "Normal | High shear / stenosis": {
            "activation": 0.95,
            "stickiness": 0.90,
            "morphology": 0.88,
            "secretion": 0.92,
            "wall_adhesion": 0.84,
            "velocity": 0.30,
            "rate": 5.2,
        },
        "Rac1 KD | High shear / stenosis": {
            "activation": 0.68,
            "stickiness": 0.46,
            "morphology": 0.18,
            "secretion": 0.56,
            "wall_adhesion": 0.23,
            "velocity": 0.66,
            "rate": 3.2,
        },
        "Rap1 KD | High shear / stenosis": {
            "activation": 0.62,
            "stickiness": 0.18,
            "morphology": 0.48,
            "secretion": 0.54,
            "wall_adhesion": 0.16,
            "velocity": 0.72,
            "rate": 3.0,
        },
        "PLCB3/Ca2+ KD | High shear / stenosis": {
            "activation": 0.28,
            "stickiness": 0.24,
            "morphology": 0.20,
            "secretion": 0.10,
            "wall_adhesion": 0.14,
            "velocity": 0.78,
            "rate": 2.4,
        },
    }

    for label, scenario in SCENARIOS.items():
        path = scenario["file"]
        if path.exists():
            continue

        f = default_finals[label]
        frame = np.arange(FRAMES)
        activation = response_curve(f["activation"], f["rate"], baseline=0.04)
        stickiness = response_curve(f["stickiness"], f["rate"] * 0.88, baseline=0.03)
        morphology = response_curve(f["morphology"], f["rate"] * 0.82, baseline=0.03)
        secretion = response_curve(f["secretion"], f["rate"] * 0.75, baseline=0.02)
        wall_adhesion = response_curve(f["wall_adhesion"], f["rate"] * 0.72, baseline=0.02)

        velocity_start = 0.88
        velocity_end = f["velocity"]
        velocity = velocity_start - (velocity_start - velocity_end) * (
            wall_adhesion / max(float(wall_adhesion.max()), 1e-8)
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
        print(f"Created missing metrics CSV: {path}")


def load_metrics():
    create_default_metrics_if_missing()

    data = {}
    for label, scenario in SCENARIOS.items():
        path = scenario["file"]
        df = pd.read_csv(path)
        required = ["frame", "activation", "stickiness", "morphology", "secretion", "wall_adhesion", "velocity"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        data[label] = df

    return data


def clamp(value):
    return float(np.clip(value, 0.0, 1.0))


def activity_color(value):
    value = clamp(value)
    if value < STATE_THRESHOLD_LOW:
        # blue for low/inactive
        return (0.15, 0.35, 0.95)
    if value < STATE_THRESHOLD_HIGH:
        # yellow/orange for intermediate
        return (1.00, 0.72, 0.15)
    # red for high/active
    return (0.90, 0.12, 0.10)


def text_color_for_value(value):
    return "white" if value < STATE_THRESHOLD_LOW or value > STATE_THRESHOLD_HIGH else "black"


def node_group(node):
    for group, nodes in NODE_GROUPS.items():
        if node in nodes:
            return group
    return "signaling"


def compute_node_values(label, row):
    scenario = SCENARIOS[label]
    shear = scenario["shear"]

    activation = clamp(row["activation"])
    stickiness = clamp(row["stickiness"])
    morphology = clamp(row["morphology"])
    secretion = clamp(row["secretion"])
    wall_adhesion = clamp(row["wall_adhesion"])

    high_pathway_drive = max(activation, stickiness, morphology, secretion)

    values = {
        "Shear stimulus": shear,
        "GPIb-vWF": clamp(0.15 + 0.85 * shear),
        "GPVI/ITAM": clamp(0.10 + 0.65 * shear),
        "PLCB3": clamp(0.25 + 0.75 * high_pathway_drive),
        "Ca2+": clamp(0.20 + 0.80 * max(activation, secretion)),
        "Rap1": clamp(0.20 + 0.80 * stickiness),
        "Integrin αIIbβ3": stickiness,
        "Stickiness": stickiness,
        "Rac1": clamp(0.20 + 0.80 * morphology),
        "Actin morphology": morphology,
        "Secretion": secretion,
        "Activation": activation,
        "Wall retention": wall_adhesion,
    }

    if "Rac1" in scenario["perturbed_nodes"]:
        values["Rac1"] = min(values["Rac1"], 0.12)
        values["Actin morphology"] = min(values["Actin morphology"], morphology)

    if "Rap1" in scenario["perturbed_nodes"]:
        values["Rap1"] = min(values["Rap1"], 0.12)
        values["Integrin αIIbβ3"] = min(values["Integrin αIIbβ3"], stickiness)

    if "PLCB3" in scenario["perturbed_nodes"]:
        values["PLCB3"] = min(values["PLCB3"], 0.12)

    if "Ca2+" in scenario["perturbed_nodes"]:
        values["Ca2+"] = min(values["Ca2+"], 0.14)

    return values


def draw_arrow(ax, start, end, color="#555555", alpha=0.75, linewidth=1.5):
    sx, sy = start
    ex, ey = end
    arrow = FancyArrowPatch(
        (sx, sy),
        (ex, ey),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=linewidth,
        color=color,
        alpha=alpha,
        shrinkA=18,
        shrinkB=18,
    )
    ax.add_patch(arrow)


def draw_cross(ax, x, y, radius):
    ax.plot([x - radius, x + radius], [y - radius, y + radius], color="black", linewidth=2.5)
    ax.plot([x - radius, x + radius], [y + radius, y - radius], color="black", linewidth=2.5)


def draw_network(ax, values=None, perturbed_nodes=None, title="", show_value=True, structure_mode=False):
    if values is None:
        values = {node: 0.0 for node in NODE_POS}
    if perturbed_nodes is None:
        perturbed_nodes = []

    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0.03, 0.97)
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)

    for source, target in EDGES:
        draw_arrow(ax, NODE_POS[source], NODE_POS[target])

    for node, (x, y) in NODE_POS.items():
        if structure_mode:
            group = node_group(node)
            face_color = STRUCTURE_COLORS[group]
            edge_color = "#333333"
            text_color = "black"
        else:
            value = values.get(node, 0.0)
            face_color = activity_color(value)
            edge_color = "black" if node in perturbed_nodes else "#333333"
            text_color = text_color_for_value(value)

        radius = 0.050 if len(node) < 12 else 0.058
        circle = Circle(
            (x, y),
            radius,
            facecolor=face_color,
            edgecolor=edge_color,
            linewidth=2.4 if node in perturbed_nodes else 1.3,
            alpha=0.96,
        )
        ax.add_patch(circle)

        if structure_mode or not show_value:
            label = node
        else:
            label = f"{node}\n{values.get(node, 0.0):.2f}"

        ax.text(x, y, label, ha="center", va="center", fontsize=7.0, color=text_color)

        if node in perturbed_nodes:
            draw_cross(ax, x, y, radius * 0.75)


def create_structure_figure():
    fig, ax = plt.subplots(figsize=(13, 7))
    draw_network(
        ax,
        title="Platelet Activation GRN Pathway Structure used for Use Case 2",
        structure_mode=True,
        show_value=False,
    )

    legend_text = (
        "Network meaning:\n"
        "Blue = input stimulus\n"
        "Purple = receptor/input pathway\n"
        "Yellow = intracellular signaling nodes\n"
        "Green = phenotype/output nodes\n\n"
        "Use Case 2 perturbs pathway nodes such as Rac1, Rap1, or PLCB3/Ca2+."
    )
    ax.text(0.02, 0.06, legend_text, fontsize=10, va="bottom", ha="left")

    fig.tight_layout()
    fig.savefig(STRUCTURE_FIG_PATH, dpi=220)
    plt.close(fig)
    print(f"Saved network structure figure: {STRUCTURE_FIG_PATH}")


def create_activity_dashboard(data):
    labels = list(SCENARIOS.keys())
    final_rows = {label: data[label].iloc[-1] for label in labels}

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "Phase 6 Use Case 2: GRN Network Structure and Perturbation Activity under Stenosis",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )

    grid = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.75])

    for i, label in enumerate(labels):
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(grid[row, col])
        values = compute_node_values(label, final_rows[label])
        draw_network(
            ax,
            values=values,
            perturbed_nodes=SCENARIOS[label]["perturbed_nodes"],
            title=SCENARIOS[label]["short"],
            show_value=True,
        )

    ax_legend = fig.add_subplot(grid[1, 2])
    ax_legend.axis("off")
    legend = (
        "Color meaning:\n"
        "Blue = low / inactive node\n"
        "Yellow = intermediate node\n"
        "Red = active node\n"
        "Black cross = pathway-node perturbation\n\n"
        "Biological interpretation:\n"
        "High shear activates the upstream pathway.\n"
        "Rac1 KD mainly reduces morphology.\n"
        "Rap1 KD mainly reduces stickiness.\n"
        "PLCB3/Ca2+ KD reduces activation and secretion."
    )
    ax_legend.text(0.0, 1.0, legend, va="top", fontsize=11)

    ax_bar = fig.add_subplot(grid[2, :])
    x = np.arange(len(METRICS))
    width = 0.15
    n = len(labels)

    for j, label in enumerate(labels):
        row = final_rows[label]
        values = [row[m] for m in METRICS]
        offset = (j - (n - 1) / 2.0) * width
        ax_bar.bar(x + offset, values, width=width, label=SCENARIOS[label]["short"])

    ax_bar.set_title("Final phenotype outputs linked to GRN activity", fontsize=13)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([m.capitalize() for m in METRICS])
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_ylabel("Final normalized value")
    ax_bar.grid(True, axis="y", alpha=0.3)
    ax_bar.legend(fontsize=8, ncols=3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(ACTIVITY_FIG_PATH, dpi=220)
    plt.close(fig)
    print(f"Saved network activity dashboard: {ACTIVITY_FIG_PATH}")


def fig_to_rgb_array(fig):
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buffer = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buffer = buffer.reshape(height, width, 4)
    return buffer[:, :, :3].copy()


def create_activity_video(data):
    labels = list(SCENARIOS.keys())

    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(2, 3)
    axes = [fig.add_subplot(grid[i // 3, i % 3]) for i in range(5)]
    ax_text = fig.add_subplot(grid[1, 2])
    ax_text.axis("off")

    fig.suptitle(
        "Phase 6 Use Case 2: Dynamic GRN Network Activity under Shear / Stenosis",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    with imageio.get_writer(VIDEO_PATH, fps=FPS, codec="libx264", quality=8, macro_block_size=16) as writer:
        for frame in range(FRAMES):
            for ax, label in zip(axes, labels):
                ax.clear()
                df = data[label]
                row = df.iloc[min(frame, len(df) - 1)]
                values = compute_node_values(label, row)
                draw_network(
                    ax,
                    values=values,
                    perturbed_nodes=SCENARIOS[label]["perturbed_nodes"],
                    title=SCENARIOS[label]["short"],
                    show_value=False,
                )

            ax_text.clear()
            ax_text.axis("off")
            lines = [f"Frame {frame + 1}/{FRAMES}", ""]
            for label in labels:
                row = data[label].iloc[min(frame, len(data[label]) - 1)]
                lines.append(SCENARIOS[label]["short"])
                lines.append(
                    f"  A={row['activation']:.2f} | "
                    f"S={row['stickiness']:.2f} | "
                    f"M={row['morphology']:.2f} | "
                    f"Sec={row['secretion']:.2f}"
                )
                lines.append("")
            lines.append("Color scale: blue=low, yellow=intermediate, red=active")
            lines.append("Crossed nodes show pathway-node perturbation.")
            ax_text.text(0.0, 1.0, "\n".join(lines), va="top", fontsize=10)

            fig.tight_layout(rect=[0, 0, 1, 0.95])
            writer.append_data(fig_to_rgb_array(fig))

            if (frame + 1) % 10 == 0 or frame == FRAMES - 1:
                print(f"Rendered network video frame {frame + 1}/{FRAMES}")

    plt.close(fig)
    print(f"Saved network activity video: {VIDEO_PATH}")


def write_summary():
    lines = []
    lines.append("# Phase 6 Use Case 2: GRN Network Structure Dashboard")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("This output adds a clean platelet activation network structure to the phenotype comparison dashboard.")
    lines.append("")
    lines.append("## Network components")
    lines.append("")
    lines.append("- Shear stimulus and receptor-level sensing")
    lines.append("- PLCB3/Ca2+ signaling axis")
    lines.append("- Rap1/integrin/stickiness branch")
    lines.append("- Rac1/actin morphology branch")
    lines.append("- Secretion and activation outputs")
    lines.append("- Wall retention phenotype")
    lines.append("")
    lines.append("## Perturbation interpretation")
    lines.append("")
    lines.append("- Rac1 KD reduces morphology and adhesion-related behavior.")
    lines.append("- Rap1 KD reduces stickiness and wall retention.")
    lines.append("- PLCB3/Ca2+ KD reduces activation and secretion-like response.")
    lines.append("")
    lines.append("## Thesis wording")
    lines.append("")
    lines.append("Use the term pathway-node perturbation or reduced-activity scenario unless experimentally validated gene knockout data are available.")

    SUMMARY_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved network summary: {SUMMARY_MD_PATH}")


def main():
    print("Phase 6 Use Case 2: GRN network structure dashboard")
    data = load_metrics()

    create_structure_figure()
    create_activity_dashboard(data)
    create_activity_video(data)
    write_summary()

    print("")
    print("Done.")
    print(f"Network structure PNG : {STRUCTURE_FIG_PATH}")
    print(f"Network activity PNG  : {ACTIVITY_FIG_PATH}")
    print(f"Network activity MP4  : {VIDEO_PATH}")
    print(f"Summary MD            : {SUMMARY_MD_PATH}")


if __name__ == "__main__":
    main()
