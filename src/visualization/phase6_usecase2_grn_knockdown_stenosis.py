"""
Phase 6 - Use Case 2
GRN-based comparison of normal platelet activation and Rac1 pathway-node perturbation
under low shear and high shear / stenosis conditions.

Run from project root:
    python src/visualization/phase6_usecase2_grn_knockdown_stenosis.py

Outputs:
    results/phase6/usecase2_grn_knockdown_stenosis/
"""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx

try:
    import imageio.v2 as imageio
    IMAGEIO_AVAILABLE = True
except Exception:
    IMAGEIO_AVAILABLE = False


# ============================================================
# PROJECT AND OUTPUT PATHS
# ============================================================


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [Path.cwd(), *here.parents]
    for cand in candidates:
        if (cand / "src").exists() and (cand / "results").exists():
            return cand
    return Path(r"C:\Users\Administrator\Desktop\PlateletSimulation")


PROJECT_ROOT = find_project_root()
OUT_ROOT = PROJECT_ROOT / "results" / "phase6" / "usecase2_grn_knockdown_stenosis"
DIRS = {
    "network": OUT_ROOT / "network_figures",
    "timeseries": OUT_ROOT / "timeseries",
    "summary": OUT_ROOT / "summary_plots",
    "heatmap": OUT_ROOT / "heatmaps",
    "snapshots": OUT_ROOT / "simulation_snapshots",
    "videos": OUT_ROOT / "videos",
    "tables": OUT_ROOT / "tables",
}

for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# Keep this False for the first run so the thesis figures are created quickly.
# Set to True later only if you also want the optional MP4 network animation.
MAKE_VIDEO = False


# ============================================================
# STYLE SETTINGS
# ============================================================

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.20,
    "grid.linewidth": 0.7,
})

COLORS = {
    "normal_low": "#1f77b4",
    "normal_high": "#b2182b",
    "rac1_low": "#4aa3df",
    "rac1_high": "#ef8a62",
    "inactive": "#2b83ba",
    "active": "#1a9850",
    "vessel": "#b2182b",
    "flow": "#67a9cf",
    "perturb": "#d73027",
    "neutral": "#f7f7f7",
    "edge": "#525252",
    "output": "#542788",
}


# ============================================================
# GRN MODEL DEFINITION
# ============================================================

INPUT_NODES = ["Shear input", "Wall/contact signal"]
CORE_NODES = [
    "PLCβ3", "Ca2+", "PI3K", "Rap1", "Rac1", "RhoA",
    "Actin remodeling", "Integrin activation", "Granule release"
]
OUTPUT_NODES = ["Activation", "Stickiness", "Morphology", "Secretion"]
ALL_NODES = INPUT_NODES + CORE_NODES + OUTPUT_NODES
TRACKED_NODES = [
    "Shear input", "PLCβ3", "Ca2+", "PI3K", "Rap1", "Rac1", "RhoA",
    "Actin remodeling", "Integrin activation", "Granule release",
    "Activation", "Stickiness", "Morphology", "Secretion"
]

# Directed weighted edges: source, target, weight.
EDGES: List[Tuple[str, str, float]] = [
    ("Shear input", "PLCβ3", 1.25),
    ("Shear input", "Ca2+", 0.95),
    ("Shear input", "RhoA", 0.85),
    ("Wall/contact signal", "PLCβ3", 1.05),
    ("Wall/contact signal", "PI3K", 0.85),
    ("Wall/contact signal", "Integrin activation", 0.55),
    ("PLCβ3", "Ca2+", 1.55),
    ("Ca2+", "Rap1", 1.15),
    ("Ca2+", "Granule release", 1.05),
    ("Ca2+", "Activation", 0.95),
    ("PI3K", "Rap1", 1.10),
    ("PI3K", "Rac1", 1.15),
    ("Rap1", "Integrin activation", 1.45),
    ("Rap1", "Activation", 0.85),
    ("Rac1", "Actin remodeling", 1.65),
    ("Rac1", "Integrin activation", 0.55),
    ("Rac1", "Morphology", 1.95),
    ("Rac1", "Stickiness", 0.85),
    ("RhoA", "Actin remodeling", 0.95),
    ("RhoA", "Morphology", 0.58),
    ("Actin remodeling", "Morphology", 1.25),
    ("Integrin activation", "Stickiness", 1.55),
    ("Integrin activation", "Activation", 0.70),
    ("Granule release", "Secretion", 1.45),
    ("Granule release", "Activation", 0.55),
    ("Secretion", "PI3K", 0.35),
    ("Activation", "Stickiness", 0.50),
    ("Activation", "Morphology", 0.42),
    ("Activation", "Secretion", 0.35),
]

NODE_BIAS: Dict[str, float] = {
    "PLCβ3": -1.05,
    "Ca2+": -0.95,
    "PI3K": -1.08,
    "Rap1": -1.00,
    "Rac1": -1.00,
    "RhoA": -1.05,
    "Actin remodeling": -1.05,
    "Integrin activation": -1.08,
    "Granule release": -1.15,
    "Activation": -1.08,
    "Stickiness": -1.18,
    "Morphology": -1.10,
    "Secretion": -1.15,
}


@dataclass(frozen=True)
class Condition:
    condition_id: str
    label: str
    grn_state: str
    shear_state: str
    shear_level: str
    perturbation: str
    line_color: str
    linestyle: str


CONDITIONS = [
    Condition(
        condition_id="normal_low_shear",
        label="Normal + low shear",
        grn_state="Normal GRN",
        shear_state="Low shear",
        shear_level="low",
        perturbation="none",
        line_color=COLORS["normal_low"],
        linestyle="-",
    ),
    Condition(
        condition_id="normal_high_shear_stenosis",
        label="Normal + high shear / stenosis",
        grn_state="Normal GRN",
        shear_state="High shear / stenosis",
        shear_level="high",
        perturbation="none",
        line_color=COLORS["normal_high"],
        linestyle="-",
    ),
    Condition(
        condition_id="rac1KD_low_shear",
        label="Rac1 perturbation + low shear",
        grn_state="Rac1 pathway-node perturbation",
        shear_state="Low shear",
        shear_level="low",
        perturbation="rac1_knockdown",
        line_color=COLORS["rac1_low"],
        linestyle="--",
    ),
    Condition(
        condition_id="rac1KD_high_shear_stenosis",
        label="Rac1 perturbation + high shear / stenosis",
        grn_state="Rac1 pathway-node perturbation",
        shear_state="High shear / stenosis",
        shear_level="high",
        perturbation="rac1_knockdown",
        line_color=COLORS["rac1_high"],
        linestyle="--",
    ),
]


# ============================================================
# SIMULATION FUNCTIONS
# ============================================================


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def shear_profile(t_norm: float, level: str) -> float:
    if level == "low":
        return float(np.clip(0.18 + 0.035 * np.sin(2 * np.pi * t_norm), 0.05, 0.30))

    # Stenosis-like exposure: low/moderate initially, then rapid mechanical stimulus increase.
    rise = sigmoid((t_norm - 0.34) * 14.0)
    pulse = 0.08 * np.exp(-((t_norm - 0.62) ** 2) / 0.010)
    return float(np.clip(0.24 + 0.66 * rise + pulse, 0.05, 0.98))


def contact_profile(t_norm: float, level: str) -> float:
    if level == "low":
        return float(np.clip(0.10 + 0.02 * np.sin(2 * np.pi * t_norm + 0.7), 0.05, 0.20))
    return float(np.clip(0.16 + 0.38 * sigmoid((t_norm - 0.46) * 12.0), 0.05, 0.65))


def build_in_edges() -> Dict[str, List[Tuple[str, float]]]:
    in_edges: Dict[str, List[Tuple[str, float]]] = {node: [] for node in ALL_NODES}
    for src, dst, w in EDGES:
        in_edges[dst].append((src, w))
    return in_edges


IN_EDGES = build_in_edges()


def compute_output_overrides(x: Dict[str, float], shear: float) -> None:
    # Output nodes are recomputed as biologically readable aggregate outputs.
    x["Activation"] = float(sigmoid(
        -1.22
        + 0.98 * x["Ca2+"]
        + 0.90 * x["Rap1"]
        + 0.78 * x["Integrin activation"]
        + 0.52 * x["Granule release"]
        + 0.25 * x["Rac1"]
        + 0.55 * shear
    ))
    x["Stickiness"] = float(sigmoid(
        -1.38
        + 1.35 * x["Integrin activation"]
        + 0.62 * x["Rap1"]
        + 1.05 * x["Rac1"]
        + 0.70 * x["Activation"]
    ))
    x["Morphology"] = float(sigmoid(
        -1.32
        + 2.15 * x["Rac1"]
        + 1.20 * x["Actin remodeling"]
        + 0.50 * x["RhoA"]
        + 0.25 * x["Ca2+"]
    ))
    x["Secretion"] = float(sigmoid(
        -1.32
        + 1.30 * x["Granule release"]
        + 0.70 * x["Ca2+"]
        + 0.45 * x["Activation"]
        + 0.30 * x["Rac1"]
    ))


def simulate_condition(condition: Condition, steps: int = 180, dt: float = 0.11) -> pd.DataFrame:
    x = {node: 0.04 for node in ALL_NODES}
    records = []

    for step in range(steps):
        t_norm = step / (steps - 1)
        shear = shear_profile(t_norm, condition.shear_level)
        contact = contact_profile(t_norm, condition.shear_level)

        x["Shear input"] = shear
        x["Wall/contact signal"] = contact

        new_x = x.copy()
        for node in CORE_NODES:
            drive = NODE_BIAS.get(node, -1.0)
            for src, weight in IN_EDGES[node]:
                drive += weight * x[src]

            target = float(sigmoid(drive))
            tau = 1.0
            new_x[node] = float(np.clip(x[node] + dt * (target - x[node]) / tau, 0.0, 1.0))

        if condition.perturbation == "rac1_knockdown":
            # Reduced-activity perturbation. This should be described as a pathway-node perturbation,
            # not as a gene knockout unless explicitly validated as gene-level intervention.
            new_x["Rac1"] = min(new_x["Rac1"], 0.18)

        x.update(new_x)
        compute_output_overrides(x, shear)

        record = {
            "time_step": step,
            "time_normalized": t_norm,
            "condition_id": condition.condition_id,
            "condition_label": condition.label,
            "grn_state": condition.grn_state,
            "shear_state": condition.shear_state,
            "perturbation": condition.perturbation,
        }
        for node in TRACKED_NODES:
            record[node] = x[node]
        records.append(record)

    return pd.DataFrame(records)


# ============================================================
# PLOTTING HELPERS
# ============================================================


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved:", path)


def node_activity_color(value: float, base: str = "#2166ac") -> Tuple[float, float, float, float]:
    # Matplotlib colormap from low grey to high red/orange.
    cmap = plt.get_cmap("YlOrRd")
    return cmap(0.15 + 0.80 * value)


def network_layout() -> Dict[str, Tuple[float, float]]:
    return {
        "Shear input": (0.0, 1.1),
        "Wall/contact signal": (0.0, -0.2),
        "PLCβ3": (1.2, 1.0),
        "Ca2+": (2.3, 1.05),
        "PI3K": (1.5, -0.15),
        "Rap1": (2.7, -0.10),
        "Rac1": (3.0, -1.10),
        "RhoA": (1.8, -1.25),
        "Actin remodeling": (4.15, -1.05),
        "Integrin activation": (4.15, -0.10),
        "Granule release": (3.55, 0.92),
        "Activation": (5.75, 0.80),
        "Stickiness": (5.75, -0.12),
        "Morphology": (5.75, -1.02),
        "Secretion": (5.75, 1.48),
    }


def make_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(ALL_NODES)
    for src, dst, w in EDGES:
        G.add_edge(src, dst, weight=w)
    return G


def draw_network(perturbed: bool = False, final_values: Dict[str, float] | None = None) -> None:
    G = make_graph()
    pos = network_layout()

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_axis_off()

    node_colors = []
    node_edgecolors = []
    node_sizes = []

    for node in G.nodes:
        if node in INPUT_NODES:
            node_colors.append("#e0f3f8")
            node_edgecolors.append("#3288bd")
            node_sizes.append(2100)
        elif node in OUTPUT_NODES:
            node_colors.append("#efe6f7")
            node_edgecolors.append(COLORS["output"])
            node_sizes.append(2300)
        elif node == "Rac1" and perturbed:
            node_colors.append("#fee0d2")
            node_edgecolors.append(COLORS["perturb"])
            node_sizes.append(2500)
        else:
            if final_values is not None:
                node_colors.append(node_activity_color(final_values.get(node, 0.1)))
            else:
                node_colors.append("#f7f7f7")
            node_edgecolors.append("#4d4d4d")
            node_sizes.append(2100)

    edge_widths = [0.75 + 1.00 * G[u][v]["weight"] for u, v in G.edges]

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=15,
        width=edge_widths,
        edge_color="#525252",
        alpha=0.72,
        connectionstyle="arc3,rad=0.04",
        min_source_margin=18,
        min_target_margin=18,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        edgecolors=node_edgecolors,
        linewidths=2.0,
        node_size=node_sizes,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        ax=ax,
        font_size=9,
        font_weight="bold",
    )

    # Region labels
    ax.text(0.0, 1.90, "Mechanical / wall inputs", fontsize=12, fontweight="bold", color="#2166ac")
    ax.text(1.65, 1.90, "Early signaling", fontsize=12, fontweight="bold", color="#525252")
    ax.text(3.20, 1.90, "Cytoskeletal / adhesive signaling", fontsize=12, fontweight="bold", color="#525252")
    ax.text(5.35, 1.90, "Behavior outputs", fontsize=12, fontweight="bold", color=COLORS["output"])

    if perturbed:
        x, y = pos["Rac1"]
        ax.add_patch(patches.Circle((x, y), 0.40, fill=False, edgecolor=COLORS["perturb"], linewidth=3.2))
        ax.plot([x - 0.28, x + 0.28], [y - 0.28, y + 0.28], color=COLORS["perturb"], linewidth=3.2)
        ax.plot([x - 0.28, x + 0.28], [y + 0.28, y - 0.28], color=COLORS["perturb"], linewidth=3.2)
        ax.text(
            x + 0.45,
            y - 0.38,
            "Rac1 reduced\nactivity cap = 0.18",
            fontsize=10,
            color=COLORS["perturb"],
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=COLORS["perturb"], lw=1.4),
        )
        title = "Perturbed platelet activation GRN: Rac1 pathway-node reduced-activity condition"
        filename = DIRS["network"] / "grn_rac1_perturbed_overview.png"
    else:
        title = "Normal platelet activation GRN under shear / stenosis input"
        filename = DIRS["network"] / "grn_normal_overview.png"

    ax.set_title(title, fontsize=16, fontweight="bold", pad=18)
    ax.set_xlim(-0.55, 6.45)
    ax.set_ylim(-1.85, 2.20)

    foot = (
        "Inputs drive continuous GRN activity values in [0, 1]. Outputs aggregate activation, stickiness, morphology, and secretion response."
    )
    ax.text(-0.45, -1.72, foot, fontsize=9, color="#525252")

    savefig(filename)


# ============================================================
# RESULT FIGURES
# ============================================================


def plot_all_outputs(all_data: pd.DataFrame) -> None:
    output_nodes = ["Activation", "Stickiness", "Morphology", "Secretion"]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, out in zip(axes, output_nodes):
        for c in CONDITIONS:
            df = all_data[all_data["condition_id"] == c.condition_id]
            ax.plot(
                df["time_normalized"],
                df[out],
                label=c.label,
                color=c.line_color,
                linestyle=c.linestyle,
                linewidth=2.4,
            )
        ax.set_title(out, fontweight="bold")
        ax.set_xlabel("Normalized simulation time")
        ax.set_ylabel("Activity / response score")
        ax.set_ylim(0, 1.03)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Use Case 2: Normal vs Rac1 pathway-node perturbation under low and stenotic shear",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    savefig(DIRS["timeseries"] / "timeseries_all_outputs_4_conditions.png")


def plot_shear_specific_comparison(all_data: pd.DataFrame, shear_state: str, filename: str, title: str) -> None:
    output_nodes = ["Activation", "Stickiness", "Morphology", "Secretion"]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.5), sharex=True, sharey=True)
    axes = axes.ravel()
    subset_conditions = [c for c in CONDITIONS if c.shear_state == shear_state]

    for ax, out in zip(axes, output_nodes):
        for c in subset_conditions:
            df = all_data[all_data["condition_id"] == c.condition_id]
            ax.plot(
                df["time_normalized"],
                df[out],
                color=c.line_color,
                linestyle=c.linestyle,
                linewidth=2.7,
                label=c.grn_state,
            )
        ax.set_title(out, fontweight="bold")
        ax.set_xlabel("Normalized simulation time")
        ax.set_ylabel("Response score")
        ax.set_ylim(0, 1.03)
        ax.legend(frameon=False)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    savefig(DIRS["timeseries"] / filename)


def plot_summary_bar(summary: pd.DataFrame) -> None:
    outputs = ["Activation", "Stickiness", "Morphology", "Secretion"]
    x = np.arange(len(outputs))
    width = 0.20

    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    for i, c in enumerate(CONDITIONS):
        row = summary[summary["condition_id"] == c.condition_id].iloc[0]
        vals = [row[f"final_{out}"] for out in outputs]
        ax.bar(
            x + (i - 1.5) * width,
            vals,
            width=width,
            label=c.label,
            color=c.line_color,
            alpha=0.92,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(outputs)
    ax.set_ylabel("Final response score")
    ax.set_ylim(0, 1.03)
    ax.set_title("Final output comparison across GRN and shear conditions", fontweight="bold", fontsize=16)
    ax.legend(frameon=False, ncol=2, loc="upper left")

    # Small interpretation annotation.
    ax.text(
        0.02,
        0.96,
        "Expected thesis readout: stenotic shear increases activation; Rac1 perturbation mainly reduces morphology and adhesion-related response.",
        transform=ax.transAxes,
        fontsize=9.5,
        va="top",
        color="#404040",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#bdbdbd", lw=1.0),
    )

    fig.tight_layout()
    savefig(DIRS["summary"] / "final_output_comparison_barplot.png")


def plot_delta_bar(summary: pd.DataFrame) -> None:
    outputs = ["Activation", "Stickiness", "Morphology", "Secretion"]
    rows = []
    for shear_state in ["Low shear", "High shear / stenosis"]:
        normal = summary[(summary["grn_state"] == "Normal GRN") & (summary["shear_state"] == shear_state)].iloc[0]
        kd = summary[(summary["grn_state"] != "Normal GRN") & (summary["shear_state"] == shear_state)].iloc[0]
        for out in outputs:
            rows.append({
                "shear_state": shear_state,
                "output": out,
                "delta_normal_minus_perturbed": normal[f"final_{out}"] - kd[f"final_{out}"],
            })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11.8, 6.6))
    x = np.arange(len(outputs))
    width = 0.34
    low = df[df["shear_state"] == "Low shear"]
    high = df[df["shear_state"] == "High shear / stenosis"]

    ax.bar(x - width / 2, low["delta_normal_minus_perturbed"], width, label="Low shear", color="#67a9cf")
    ax.bar(x + width / 2, high["delta_normal_minus_perturbed"], width, label="High shear / stenosis", color="#ef8a62")

    ax.axhline(0, color="#525252", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(outputs)
    ax.set_ylabel("Normal - Rac1 perturbation")
    ax.set_title("Effect size of Rac1 pathway-node perturbation", fontsize=16, fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig(DIRS["summary"] / "rac1_perturbation_effect_size_barplot.png")


def plot_heatmap(all_data: pd.DataFrame) -> None:
    selected = [
        "Shear input", "PLCβ3", "Ca2+", "PI3K", "Rap1", "Rac1", "RhoA",
        "Actin remodeling", "Integrin activation", "Granule release",
        "Activation", "Stickiness", "Morphology", "Secretion"
    ]

    cols = []
    data = []
    for c in CONDITIONS:
        df = all_data[all_data["condition_id"] == c.condition_id]
        final = df.iloc[-1]
        cols.append(c.label.replace(" + ", "\n+ "))
        data.append([final[node] for node in selected])

    arr = np.array(data).T

    fig, ax = plt.subplots(figsize=(11.5, 8.2))
    im = ax.imshow(arr, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=0, ha="center")
    ax.set_yticks(np.arange(len(selected)))
    ax.set_yticklabels(selected)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=8.5, color="#1a1a1a")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Final node activity / output score")
    ax.set_title("Final GRN node activity heatmap across Use Case 2 conditions", fontsize=16, fontweight="bold")
    fig.tight_layout()
    savefig(DIRS["heatmap"] / "node_activity_heatmap_4_conditions.png")


def draw_platelet(ax, x: float, y: float, active: bool, morphology: float, size: float = 0.10) -> None:
    if active:
        body_color = "#1a9850"
        edge_color = "#00441b"
        # Activated irregular platelet with small projections.
        n = 16
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        radii = size * (1.0 + morphology * 0.45 * np.sin(3 * theta + 0.7))
        px = x + radii * np.cos(theta)
        py = y + 0.70 * radii * np.sin(theta)
        poly = patches.Polygon(np.c_[px, py], closed=True, facecolor=body_color, edgecolor=edge_color, linewidth=1.1, alpha=0.95)
        ax.add_patch(poly)
        if morphology > 0.45:
            for ang in np.linspace(0, 2 * np.pi, 6, endpoint=False):
                ax.plot(
                    [x + size * 0.55 * np.cos(ang), x + size * (0.95 + morphology * 0.35) * np.cos(ang)],
                    [y + size * 0.40 * np.sin(ang), y + size * (0.70 + morphology * 0.22) * np.sin(ang)],
                    color=edge_color,
                    linewidth=0.9,
                    alpha=0.85,
                )
    else:
        ell = patches.Ellipse((x, y), size * 1.7, size * 0.95, angle=10, facecolor="#2b83ba", edgecolor="#084081", linewidth=1.0, alpha=0.92)
        ax.add_patch(ell)


def plot_visual_panel(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.8), sharex=True, sharey=True)
    axes = axes.ravel()

    rng = np.random.default_rng(7)

    for ax, c in zip(axes, CONDITIONS):
        row = summary[summary["condition_id"] == c.condition_id].iloc[0]
        activation = row["final_Activation"]
        stickiness = row["final_Stickiness"]
        morphology = row["final_Morphology"]

        # Vessel background.
        ax.add_patch(patches.FancyBboxPatch(
            (-0.05, 0.15), 1.10, 0.70,
            boxstyle="round,pad=0.02,rounding_size=0.20",
            facecolor="#fddbc7", edgecolor="#b2182b", linewidth=2.2, alpha=0.95,
        ))
        ax.plot([0.0, 1.0], [0.50, 0.50], color="#b2182b", alpha=0.25, linewidth=1.0)

        # Flow arrows. More arrows for high shear/stenosis.
        arrow_count = 4 if c.shear_level == "low" else 7
        for k in range(arrow_count):
            y = 0.28 + k * (0.44 / max(1, arrow_count - 1))
            ax.arrow(0.08, y, 0.78, 0, head_width=0.025, head_length=0.030, length_includes_head=True, color="#0571b0", alpha=0.45, linewidth=1.2)

        total_platelets = 14
        active_count = int(round(activation * total_platelets))
        wall_count = int(round(stickiness * 8))

        # Inactive / circulating platelets.
        for i in range(total_platelets - active_count):
            x = 0.08 + 0.84 * rng.random()
            y = 0.30 + 0.40 * rng.random()
            draw_platelet(ax, x, y, active=False, morphology=morphology, size=0.045)

        # Active / adhesive platelets.
        for i in range(active_count):
            if i < wall_count:
                # Wall-adjacent platelets represent adhesion.
                x = 0.12 + 0.78 * rng.random()
                y = 0.18 if i % 2 == 0 else 0.82
            else:
                x = 0.12 + 0.78 * rng.random()
                y = 0.30 + 0.40 * rng.random()
            draw_platelet(ax, x, y, active=True, morphology=morphology, size=0.052)

        # Stenosis shape cue for high shear.
        if c.shear_level == "high":
            ax.add_patch(patches.Polygon(
                [[0.44, 0.85], [0.56, 0.85], [0.51, 0.63], [0.49, 0.63]],
                closed=True, facecolor="#b2182b", alpha=0.22, edgecolor="none"
            ))
            ax.add_patch(patches.Polygon(
                [[0.44, 0.15], [0.56, 0.15], [0.51, 0.37], [0.49, 0.37]],
                closed=True, facecolor="#b2182b", alpha=0.22, edgecolor="none"
            ))

        if c.perturbation == "rac1_knockdown":
            ax.text(0.02, 0.94, "Rac1 reduced", color=COLORS["perturb"], fontsize=10, fontweight="bold", transform=ax.transAxes)

        ax.set_title(c.label, fontsize=12.5, fontweight="bold")
        ax.text(
            0.02,
            0.03,
            f"Activation {activation:.2f} | Stickiness {stickiness:.2f} | Morphology {morphology:.2f}",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", lw=0.8),
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    fig.suptitle("Use Case 2 visual comparison: GRN state controls platelet behavior under shear", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    savefig(DIRS["snapshots"] / "usecase2_2x2_visual_comparison.png")


def create_simple_video(all_data: pd.DataFrame, summary: pd.DataFrame) -> None:
    if not IMAGEIO_AVAILABLE:
        print("Skipping video: imageio not available")
        return

    frames_dir = DIRS["videos"] / "frames_usecase2_grn_activity"
    frames_dir.mkdir(parents=True, exist_ok=True)

    G = make_graph()
    pos = network_layout()
    output_path = DIRS["videos"] / "usecase2_grn_activity_high_shear_normal_vs_rac1KD.mp4"

    normal = all_data[all_data["condition_id"] == "normal_high_shear_stenosis"].reset_index(drop=True)
    kd = all_data[all_data["condition_id"] == "rac1KD_high_shear_stenosis"].reset_index(drop=True)
    frame_indices = np.linspace(0, len(normal) - 1, 72).astype(int)

    frame_paths = []
    for frame_no, idx in enumerate(frame_indices):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
        for ax, df, title, perturbed in [
            (axes[0], normal, "Normal + high shear / stenosis", False),
            (axes[1], kd, "Rac1 perturbation + high shear / stenosis", True),
        ]:
            ax.set_axis_off()
            row = df.iloc[idx]
            vals = {node: float(row[node]) for node in ALL_NODES if node in row.index}
            node_colors = []
            edgecolors = []
            sizes = []
            for node in G.nodes:
                if node in INPUT_NODES:
                    node_colors.append("#d9f0f7")
                    edgecolors.append("#0571b0")
                    sizes.append(700)
                elif node in OUTPUT_NODES:
                    node_colors.append(node_activity_color(vals.get(node, 0.1)))
                    edgecolors.append(COLORS["output"])
                    sizes.append(780)
                elif node == "Rac1" and perturbed:
                    node_colors.append("#fee0d2")
                    edgecolors.append(COLORS["perturb"])
                    sizes.append(780)
                else:
                    node_colors.append(node_activity_color(vals.get(node, 0.1)))
                    edgecolors.append("#525252")
                    sizes.append(700)

            nx.draw_networkx_edges(
                G, pos, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=9,
                width=1.0, edge_color="#737373", alpha=0.55,
                connectionstyle="arc3,rad=0.04",
            )
            nx.draw_networkx_nodes(
                G, pos, ax=ax, node_color=node_colors, edgecolors=edgecolors,
                linewidths=1.2, node_size=sizes,
            )
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=6.5, font_weight="bold")
            ax.set_xlim(-0.6, 6.5)
            ax.set_ylim(-1.8, 2.05)
            ax.set_title(title, fontsize=12.5, fontweight="bold")
            if perturbed:
                x, y = pos["Rac1"]
                ax.plot([x - 0.20, x + 0.20], [y - 0.20, y + 0.20], color=COLORS["perturb"], linewidth=2.2)
                ax.plot([x - 0.20, x + 0.20], [y + 0.20, y - 0.20], color=COLORS["perturb"], linewidth=2.2)

        t = normal.iloc[idx]["time_normalized"]
        fig.suptitle(f"GRN activity progression under stenotic shear | normalized time = {t:.2f}", fontsize=14, fontweight="bold")
        fig.tight_layout()
        frame_path = frames_dir / f"frame_{frame_no:03d}.png"
        fig.savefig(frame_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        frame_paths.append(frame_path)

    with imageio.get_writer(str(output_path), fps=12, codec="libx264", quality=8, macro_block_size=16) as writer:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))

    print("Saved:", output_path)


# ============================================================
# README / SUMMARY WRITING
# ============================================================


def write_readme(summary: pd.DataFrame) -> None:
    readme = OUT_ROOT / "README_usecase2_grn_knockdown_stenosis.md"
    high_normal = summary[summary["condition_id"] == "normal_high_shear_stenosis"].iloc[0]
    high_kd = summary[summary["condition_id"] == "rac1KD_high_shear_stenosis"].iloc[0]

    text = f"""
# Phase 6 Use Case 2 - GRN knockdown / stenosis comparison

## Objective
This use case compares a normal platelet activation GRN against a Rac1 pathway-node reduced-activity perturbation under low-shear and high-shear / stenosis-like conditions.

## Experimental matrix
1. Normal GRN + low shear
2. Normal GRN + high shear / stenosis
3. Rac1 pathway-node perturbation + low shear
4. Rac1 pathway-node perturbation + high shear / stenosis

## Perturbation wording
The perturbation should be described as **Rac1 pathway-node knockdown**, **Rac1 reduced-activity perturbation**, or **Rac1 pathway-node perturbation**. It should not be described as a gene knockout unless the underlying model explicitly represents a gene-level intervention.

## Model outputs
The four main output variables are:
- Activation
- Stickiness
- Morphology
- Secretion

## High-shear result summary
Under high shear / stenosis:
- Normal final activation: {high_normal['final_Activation']:.3f}
- Perturbed final activation: {high_kd['final_Activation']:.3f}
- Normal final stickiness: {high_normal['final_Stickiness']:.3f}
- Perturbed final stickiness: {high_kd['final_Stickiness']:.3f}
- Normal final morphology: {high_normal['final_Morphology']:.3f}
- Perturbed final morphology: {high_kd['final_Morphology']:.3f}
- Normal final secretion: {high_normal['final_Secretion']:.3f}
- Perturbed final secretion: {high_kd['final_Secretion']:.3f}

## Thesis interpretation
The normal GRN shows stronger activation and adhesion-related response under stenotic high shear. The Rac1 pathway-node perturbation reduces the cytoskeletal and morphology-associated response and lowers adhesion/stickiness despite exposure to the same high mechanical stimulus.

## Generated outputs
- `network_figures/grn_normal_overview.png`
- `network_figures/grn_rac1_perturbed_overview.png`
- `timeseries/timeseries_all_outputs_4_conditions.png`
- `timeseries/timeseries_low_shear_normal_vs_rac1KD.png`
- `timeseries/timeseries_high_shear_stenosis_normal_vs_rac1KD.png`
- `summary_plots/final_output_comparison_barplot.png`
- `summary_plots/rac1_perturbation_effect_size_barplot.png`
- `heatmaps/node_activity_heatmap_4_conditions.png`
- `simulation_snapshots/usecase2_2x2_visual_comparison.png`
- `videos/usecase2_grn_activity_high_shear_normal_vs_rac1KD.mp4`
- `tables/final_output_summary.csv`
- `tables/node_activity_summary.csv`
- `tables/usecase2_all_condition_timeseries.csv`
"""
    readme.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")
    print("Saved:", readme)


# ============================================================
# MAIN PIPELINE
# ============================================================


def main() -> None:
    print("\n====================================================")
    print("PHASE 6 USE CASE 2 - GRN KNOCKDOWN / STENOSIS")
    print("====================================================")
    print("Project root:", PROJECT_ROOT)
    print("Output root:", OUT_ROOT)
    print("====================================================\n")

    # Save condition matrix.
    condition_rows = [c.__dict__ for c in CONDITIONS]
    pd.DataFrame(condition_rows).to_csv(DIRS["tables"] / "condition_matrix.csv", index=False)

    # Run four conditions.
    dfs = []
    for condition in CONDITIONS:
        df = simulate_condition(condition)
        dfs.append(df)
        out_csv = DIRS["tables"] / f"{condition.condition_id}_timeseries.csv"
        df.to_csv(out_csv, index=False)
        print("Saved:", out_csv)

    all_data = pd.concat(dfs, ignore_index=True)
    all_data.to_csv(DIRS["tables"] / "usecase2_all_condition_timeseries.csv", index=False)
    print("Saved:", DIRS["tables"] / "usecase2_all_condition_timeseries.csv")

    # Summary tables.
    rows = []
    node_rows = []
    for condition in CONDITIONS:
        df = all_data[all_data["condition_id"] == condition.condition_id]
        final = df.iloc[-1]
        mean_late = df[df["time_normalized"] >= 0.75].mean(numeric_only=True)
        row = {
            "condition_id": condition.condition_id,
            "condition_label": condition.label,
            "grn_state": condition.grn_state,
            "shear_state": condition.shear_state,
            "perturbation": condition.perturbation,
        }
        for out in OUTPUT_NODES:
            row[f"final_{out}"] = float(final[out])
            row[f"late_mean_{out}"] = float(mean_late[out])
        rows.append(row)

        for node in TRACKED_NODES:
            node_rows.append({
                "condition_id": condition.condition_id,
                "condition_label": condition.label,
                "node": node,
                "final_activity": float(final[node]),
                "late_mean_activity": float(mean_late[node]),
            })

    summary = pd.DataFrame(rows)
    node_summary = pd.DataFrame(node_rows)
    summary.to_csv(DIRS["tables"] / "final_output_summary.csv", index=False)
    node_summary.to_csv(DIRS["tables"] / "node_activity_summary.csv", index=False)
    print("Saved:", DIRS["tables"] / "final_output_summary.csv")
    print("Saved:", DIRS["tables"] / "node_activity_summary.csv")

    # Network diagrams.
    final_normal_high = all_data[all_data["condition_id"] == "normal_high_shear_stenosis"].iloc[-1].to_dict()
    final_kd_high = all_data[all_data["condition_id"] == "rac1KD_high_shear_stenosis"].iloc[-1].to_dict()
    draw_network(perturbed=False, final_values=final_normal_high)
    draw_network(perturbed=True, final_values=final_kd_high)

    # Plots.
    plot_all_outputs(all_data)
    plot_shear_specific_comparison(
        all_data,
        shear_state="Low shear",
        filename="timeseries_low_shear_normal_vs_rac1KD.png",
        title="Low shear: normal GRN vs Rac1 pathway-node perturbation",
    )
    plot_shear_specific_comparison(
        all_data,
        shear_state="High shear / stenosis",
        filename="timeseries_high_shear_stenosis_normal_vs_rac1KD.png",
        title="High shear / stenosis: normal GRN vs Rac1 pathway-node perturbation",
    )
    plot_summary_bar(summary)
    plot_delta_bar(summary)
    plot_heatmap(all_data)
    plot_visual_panel(summary)
    if MAKE_VIDEO:
        create_simple_video(all_data, summary)
    else:
        print("Skipping optional MP4 animation because MAKE_VIDEO = False")
    write_readme(summary)

    print("\n====================================================")
    print("DONE - USE CASE 2 OUTPUTS CREATED")
    print("====================================================")
    print("Open this folder:")
    print(OUT_ROOT)
    print("Start by checking:")
    print(DIRS["network"] / "grn_normal_overview.png")
    print(DIRS["network"] / "grn_rac1_perturbed_overview.png")
    print(DIRS["summary"] / "final_output_comparison_barplot.png")
    print(DIRS["snapshots"] / "usecase2_2x2_visual_comparison.png")
    print("====================================================\n")


if __name__ == "__main__":
    main()
