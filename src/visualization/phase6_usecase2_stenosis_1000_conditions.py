from pathlib import Path
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# PHASE 6 USE CASE 2 EXTENSION
# 1000 platelet stenosis-flow benchmark:
# Normal vs pathway-node perturbation / knockout-like scenarios
#
# This is a prescribed-flow agent simulation, not full CFD.
# It is designed for thesis-level visual comparison and benchmark reporting.
# ============================================================

SEED = 42
N_PLATELETS = 1000
N_STEPS = 260
VIDEO_EVERY = 2
DT = 0.045

L = 12.0
R0 = 1.0
STENOSIS_CENTER = 6.2
STENOSIS_WIDTH = 1.05
STENOSIS_DEPTH = 0.55
U_MAX = 1.20

MAKE_VIDEO = True
VIDEO_FPS = 20
VIDEO_W = 1600
VIDEO_H = 1000

# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

def find_project_root():
    p = Path(__file__).resolve()
    # expected: project/src/visualization/script.py
    if len(p.parents) >= 3 and (p.parents[2] / "src").exists():
        return p.parents[2]
    cwd = Path.cwd()
    if (cwd / "src").exists():
        return cwd
    return p.parents[2]

PROJECT_ROOT = find_project_root()

BASE = (
    PROJECT_ROOT
    / "results"
    / "phase6"
    / "usecase2_grn_knockdown_stenosis"
    / "stenosis_1000_platelet_conditions"
)

DIRS = {
    "tables": BASE / "tables",
    "timeseries": BASE / "timeseries",
    "summary": BASE / "summary_plots",
    "snapshots": BASE / "simulation_snapshots",
    "videos": BASE / "videos",
    "raw": BASE / "raw_data",
}

for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Conditions
# ------------------------------------------------------------

CONDITIONS = [
    {
        "id": "normal",
        "label": "Normal GRN",
        "short": "Normal",
        "color": "#1f77b4",
        "activation_gain": 1.00,
        "stickiness_gain": 1.00,
        "morphology_gain": 1.00,
        "secretion_gain": 1.00,
        "adhesion_strength": 1.00,
        "velocity_penalty": 0.62,
        "description": "Baseline platelet activation pathway under stenotic shear.",
    },
    {
        "id": "rac1KD",
        "label": "Rac1 pathway-node perturbation",
        "short": "Rac1 KD",
        "color": "#ff7f0e",
        "activation_gain": 0.94,
        "stickiness_gain": 0.82,
        "morphology_gain": 0.38,
        "secretion_gain": 0.92,
        "adhesion_strength": 0.56,
        "velocity_penalty": 0.35,
        "description": "Reduced Rac1 activity; expected reduction in morphology and adhesion-related behavior.",
    },
    {
        "id": "rap1KD",
        "label": "Rap1 pathway-node perturbation",
        "short": "Rap1 KD",
        "color": "#2ca02c",
        "activation_gain": 0.92,
        "stickiness_gain": 0.42,
        "morphology_gain": 0.72,
        "secretion_gain": 0.88,
        "adhesion_strength": 0.32,
        "velocity_penalty": 0.24,
        "description": "Reduced Rap1/integrin-related activity; expected weaker sticking and wall retention.",
    },
    {
        "id": "plcb3KD",
        "label": "PLCB3/Ca2+ pathway-node perturbation",
        "short": "PLCB3 KD",
        "color": "#d62728",
        "activation_gain": 0.58,
        "stickiness_gain": 0.62,
        "morphology_gain": 0.66,
        "secretion_gain": 0.42,
        "adhesion_strength": 0.45,
        "velocity_penalty": 0.28,
        "description": "Reduced PLCB3/Ca2+-like signaling; expected lower activation and secretion.",
    },
]

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def vessel_radius(x):
    sten = np.exp(-((x - STENOSIS_CENTER) / STENOSIS_WIDTH) ** 2)
    return R0 * (1.0 - STENOSIS_DEPTH * sten)

def stenosis_intensity(x):
    return np.exp(-((x - STENOSIS_CENTER) / STENOSIS_WIDTH) ** 2)

def sigmoid(x, k=8.0):
    return 1.0 / (1.0 + np.exp(-k * x))

def sample_disc(n, rng, radius=0.75):
    theta = rng.uniform(0, 2 * np.pi, n)
    rr = radius * np.sqrt(rng.uniform(0, 1, n))
    y = rr * np.cos(theta)
    z = rr * np.sin(theta)
    return y, z

def activation_to_rgb(a):
    # blue -> yellow -> red
    a = float(max(0.0, min(1.0, a)))
    if a < 0.5:
        t = a / 0.5
        r = int(30 + t * (245 - 30))
        g = int(120 + t * (210 - 120))
        b = int(220 + t * (30 - 220))
    else:
        t = (a - 0.5) / 0.5
        r = int(245 + t * (210 - 245))
        g = int(210 + t * (35 - 210))
        b = int(30 + t * (45 - 30))
    return (r, g, b)

def get_font(size=20, bold=False):
    paths = (
        [r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\calibrib.ttf", r"C:\Windows\Fonts\segoeuib.ttf"]
        if bold
        else [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\calibri.ttf", r"C:\Windows\Fonts\segoeui.ttf"]
    )
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

FONT_TITLE = get_font(28, True)
FONT_PANEL = get_font(19, True)
FONT_SMALL = get_font(15, False)
FONT_TINY = get_font(12, False)

# ------------------------------------------------------------
# Simulation
# ------------------------------------------------------------

def initialize_agents(seed_offset=0):
    rng = np.random.default_rng(SEED + seed_offset)

    x = rng.uniform(-0.8, L * 0.92, N_PLATELETS)
    y, z = sample_disc(N_PLATELETS, rng, radius=0.72)

    activation = rng.uniform(0.02, 0.12, N_PLATELETS)
    stuck = np.zeros(N_PLATELETS, dtype=bool)

    return x, y, z, activation, stuck, rng

def simulate_condition(params, seed_offset=0):
    start_time = time.perf_counter()

    x, y, z, activation, stuck, rng = initialize_agents(seed_offset=seed_offset)

    frames = []
    timeseries = []

    final_stickiness = np.zeros(N_PLATELETS)
    final_morphology = np.zeros(N_PLATELETS)
    final_secretion = np.zeros(N_PLATELETS)
    final_velocity = np.zeros(N_PLATELETS)

    for step in range(N_STEPS):
        t_norm = step / max(N_STEPS - 1, 1)

        R = vessel_radius(x)
        r = np.sqrt(y * y + z * z)
        rr = np.clip(r / np.maximum(R, 1e-6), 0, 1.4)

        sten = stenosis_intensity(x)
        wall_zone = np.clip((rr - 0.62) / 0.36, 0, 1)

        local_shear = np.clip(0.10 + 0.52 * sten + 0.38 * wall_zone + 0.35 * sten * wall_zone, 0, 1)
        shear_drive = sigmoid(local_shear - 0.35, k=7.0)

        activation += DT * (
            params["activation_gain"] * (0.78 * shear_drive + 0.22 * wall_zone)
            - 0.35 * activation
        )
        activation = np.clip(activation, 0, 1)

        stickiness = np.clip(params["stickiness_gain"] * (0.08 + 0.92 * activation ** 1.35), 0, 1)
        morphology = np.clip(params["morphology_gain"] * activation ** 1.20, 0, 1)
        secretion = np.clip(params["secretion_gain"] * (0.55 * activation + 0.45 * activation * wall_zone), 0, 1)

        # Sticking probability near stenosis and wall
        adhesion_probability = (
            0.11
            * params["adhesion_strength"]
            * stickiness
            * wall_zone
            * sten
            * (0.35 + 0.65 * activation)
        )

        new_stuck = (rng.uniform(0, 1, N_PLATELETS) < adhesion_probability) & (activation > 0.42)
        stuck = stuck | new_stuck

        # Flow velocity: stenosis speeds up center flow; sticking slows wall-adherent platelets
        u = U_MAX * (R0 / np.maximum(R, 0.18)) ** 1.45 * np.clip(1.0 - rr ** 2, 0.04, 1.0)
        u *= (1.0 - params["velocity_penalty"] * stuck.astype(float))
        u = np.clip(u, 0.015, 3.2)

        x += DT * u

        # Radial margination toward wall in stenosis region
        radial_norm_y = y / np.maximum(r, 1e-6)
        radial_norm_z = z / np.maximum(r, 1e-6)
        outward = DT * (0.030 + 0.100 * sten * activation * params["adhesion_strength"])

        y += outward * radial_norm_y + rng.normal(0, 0.006, N_PLATELETS)
        z += outward * radial_norm_z + rng.normal(0, 0.006, N_PLATELETS)

        # Stuck platelets remain close to wall
        r2 = np.sqrt(y * y + z * z)
        rr2 = r2 / np.maximum(R, 1e-6)

        too_far = rr2 > 0.95
        y[too_far] *= (0.95 * R[too_far] / np.maximum(r2[too_far], 1e-6))
        z[too_far] *= (0.95 * R[too_far] / np.maximum(r2[too_far], 1e-6))

        r3 = np.sqrt(y * y + z * z)
        y[stuck] *= (0.90 * R[stuck] / np.maximum(r3[stuck], 1e-6))
        z[stuck] *= (0.90 * R[stuck] / np.maximum(r3[stuck], 1e-6))

        # Periodic inlet respawn for continuous benchmark flow
        outflow = x > L + 0.4
        n_out = int(outflow.sum())
        if n_out > 0:
            x[outflow] = rng.uniform(-0.9, -0.15, n_out)
            yy, zz = sample_disc(n_out, rng, radius=0.62)
            y[outflow] = yy
            z[outflow] = zz
            activation[outflow] = rng.uniform(0.02, 0.10, n_out)
            stuck[outflow] = False

        final_stickiness = stickiness
        final_morphology = morphology
        final_secretion = secretion
        final_velocity = u

        timeseries.append({
            "step": step,
            "time_normalized": t_norm,
            "mean_activation": float(np.mean(activation)),
            "mean_stickiness": float(np.mean(stickiness)),
            "mean_morphology": float(np.mean(morphology)),
            "mean_secretion": float(np.mean(secretion)),
            "stuck_fraction": float(np.mean(stuck.astype(float))),
            "mean_velocity": float(np.mean(u)),
            "mean_shear": float(np.mean(local_shear)),
            "platelet_count": N_PLATELETS,
        })

        if step % VIDEO_EVERY == 0 or step == N_STEPS - 1:
            frames.append({
                "step": step,
                "time_normalized": t_norm,
                "x": x.copy(),
                "y": y.copy(),
                "z": z.copy(),
                "activation": activation.copy(),
                "stuck": stuck.copy(),
                "velocity": u.copy(),
                "stickiness": stickiness.copy(),
                "morphology": morphology.copy(),
                "secretion": secretion.copy(),
            })

    elapsed = time.perf_counter() - start_time

    ts = pd.DataFrame(timeseries)
    summary = {
        "condition_id": params["id"],
        "condition": params["label"],
        "platelet_count": N_PLATELETS,
        "final_activation": float(ts["mean_activation"].iloc[-1]),
        "final_stickiness": float(ts["mean_stickiness"].iloc[-1]),
        "final_morphology": float(ts["mean_morphology"].iloc[-1]),
        "final_secretion": float(ts["mean_secretion"].iloc[-1]),
        "final_stuck_fraction": float(ts["stuck_fraction"].iloc[-1]),
        "mean_velocity_final": float(ts["mean_velocity"].iloc[-1]),
        "mean_shear_final": float(ts["mean_shear"].iloc[-1]),
        "runtime_seconds": elapsed,
        "steps": N_STEPS,
        "benchmark_platelets": N_PLATELETS,
    }

    raw_path = DIRS["raw"] / f"{params['id']}_final_state.npz"
    np.savez_compressed(
        raw_path,
        x=x,
        y=y,
        z=z,
        activation=activation,
        stickiness=final_stickiness,
        morphology=final_morphology,
        secretion=final_secretion,
        velocity=final_velocity,
        stuck=stuck,
    )

    return ts, summary, frames

# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def save_timeseries_plot(all_ts):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.8), dpi=180)
    axes = axes.ravel()

    metrics = [
        ("mean_activation", "Mean activation"),
        ("mean_stickiness", "Mean stickiness"),
        ("mean_morphology", "Mean morphology"),
        ("mean_secretion", "Mean secretion"),
        ("stuck_fraction", "Wall-adherent fraction"),
        ("mean_velocity", "Mean axial velocity"),
    ]

    for ax, (metric, title) in zip(axes, metrics):
        for cond in CONDITIONS:
            ts = all_ts[cond["id"]]
            ax.plot(ts["time_normalized"], ts[metric], label=cond["short"], linewidth=2.5, color=cond["color"])
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Normalized simulation time")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("1000-platelet stenosis-flow simulation: normal vs pathway-node perturbation conditions", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])

    out = DIRS["timeseries"] / "stenosis_1000_timeseries_outputs.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)

def save_summary_barplot(summary_df):
    metrics = [
        ("final_activation", "Activation"),
        ("final_stickiness", "Stickiness"),
        ("final_morphology", "Morphology"),
        ("final_secretion", "Secretion"),
        ("final_stuck_fraction", "Wall-adherent fraction"),
        ("mean_velocity_final", "Mean velocity"),
    ]

    x = np.arange(len(metrics))
    width = 0.18

    fig, ax = plt.subplots(figsize=(15.5, 7.8), dpi=180)

    for i, cond in enumerate(CONDITIONS):
        vals = [float(summary_df.loc[summary_df["condition_id"] == cond["id"], m].iloc[0]) for m, _ in metrics]
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=cond["short"], color=cond["color"], alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics], rotation=0)
    ax.set_ylabel("Final mean score / normalized value")
    ax.set_title("Final benchmark comparison across stenosis-flow conditions", fontsize=16, fontweight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09))

    ax.text(
        0.01,
        0.96,
        "Interpretation: pathway-node perturbations preserve flow transport but reduce specific GRN-controlled outputs.",
        transform=ax.transAxes,
        fontsize=11,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f4f6f8", edgecolor="#cccccc"),
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = DIRS["summary"] / "stenosis_1000_final_condition_summary.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)

def save_final_snapshot(frames_by_condition):
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.0), dpi=180)
    axes = axes.ravel()

    xs = np.linspace(0, L, 400)
    upper = vessel_radius(xs)
    lower = -upper

    for ax, cond in zip(axes, CONDITIONS):
        frame = frames_by_condition[cond["id"]][-1]
        colors = [activation_to_rgb(a) for a in frame["activation"]]

        ax.fill_between(xs, lower, upper, color="#fde5d6", alpha=0.95)
        ax.plot(xs, upper, color="#c9292c", linewidth=2.0)
        ax.plot(xs, lower, color="#c9292c", linewidth=2.0)
        ax.scatter(frame["x"], frame["y"], s=9, c=np.array(colors) / 255.0, alpha=0.75, linewidths=0)

        stuck = frame["stuck"]
        if np.any(stuck):
            ax.scatter(frame["x"][stuck], frame["y"][stuck], s=18, facecolors="none", edgecolors="black", linewidths=0.7)

        ax.axvspan(STENOSIS_CENTER - STENOSIS_WIDTH, STENOSIS_CENTER + STENOSIS_WIDTH, color="red", alpha=0.08)
        ax.set_title(cond["label"], fontweight="bold")
        ax.set_xlim(0, L)
        ax.set_ylim(-1.15, 1.15)
        ax.set_xlabel("Vessel axis")
        ax.set_ylabel("Radial position")
        ax.grid(alpha=0.18)
        ax.set_aspect("auto")

        txt = (
            f"Activation {frame['activation'].mean():.2f} | "
            f"Stuck {frame['stuck'].mean():.2f} | "
            f"Velocity {frame['velocity'].mean():.2f}"
        )
        ax.text(0.02, 0.03, txt, transform=ax.transAxes, fontsize=9, bbox=dict(facecolor="white", alpha=0.8, edgecolor="#cccccc"))

    fig.suptitle("1000 platelet stenosis simulation: flow and wall interaction under normal and perturbed GRN conditions", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = DIRS["snapshots"] / "stenosis_1000_2x2_final_distribution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)

# ------------------------------------------------------------
# Video rendering
# ------------------------------------------------------------

def draw_condition_panel(draw, frame, cond, panel_box):
    x0, y0, x1, y1 = panel_box
    w = x1 - x0
    h = y1 - y0

    draw.rounded_rectangle(panel_box, radius=18, fill=(255, 255, 255), outline=(210, 214, 220), width=2)
    draw.text((x0 + 18, y0 + 14), cond["label"], fill=(24, 24, 24), font=FONT_PANEL)

    plot_x0 = x0 + 28
    plot_y0 = y0 + 60
    plot_x1 = x1 - 24
    plot_y1 = y1 - 46

    def sx(x):
        return plot_x0 + (x / L) * (plot_x1 - plot_x0)

    def sy(y):
        return plot_y0 + (0.5 - y / 2.4) * (plot_y1 - plot_y0)

    xs = np.linspace(0, L, 160)
    ru = vessel_radius(xs)
    upper = [(sx(float(xx)), sy(float(rr))) for xx, rr in zip(xs, ru)]
    lower = [(sx(float(xx)), sy(float(-rr))) for xx, rr in zip(xs[::-1], ru[::-1])]
    poly = upper + lower

    draw.polygon(poly, fill=(252, 225, 211), outline=(206, 35, 40))
    draw.line(upper, fill=(200, 30, 36), width=2)
    draw.line(lower[::-1], fill=(200, 30, 36), width=2)

    # stenosis marker
    st_x0 = sx(STENOSIS_CENTER - STENOSIS_WIDTH)
    st_x1 = sx(STENOSIS_CENTER + STENOSIS_WIDTH)
    draw.rectangle([st_x0, plot_y0, st_x1, plot_y1], fill=(255, 0, 0, 22))

    # flow arrows
    for frac in [0.22, 0.38, 0.54, 0.70]:
        yy = plot_y0 + frac * (plot_y1 - plot_y0)
        draw.line([(plot_x0 + 30, yy), (plot_x1 - 45, yy)], fill=(80, 155, 205), width=2)
        draw.polygon([(plot_x1 - 45, yy - 5), (plot_x1 - 45, yy + 5), (plot_x1 - 32, yy)], fill=(80, 155, 205))

    # platelets
    x = frame["x"]
    y = frame["y"]
    a = frame["activation"]
    stuck = frame["stuck"]

    order = np.argsort(a)
    for idx in order:
        px = int(sx(float(x[idx])))
        py = int(sy(float(y[idx])))
        if px < plot_x0 or px > plot_x1 or py < plot_y0 or py > plot_y1:
            continue
        col = activation_to_rgb(float(a[idx]))
        r = 3 if not stuck[idx] else 5
        draw.ellipse([px - r, py - r, px + r, py + r], fill=col, outline=(35, 35, 35) if stuck[idx] else None)

    # metrics
    metrics = (
        f"N={N_PLATELETS} | activation {np.mean(frame['activation']):.2f} | "
        f"stuck {np.mean(frame['stuck']):.2f} | velocity {np.mean(frame['velocity']):.2f}"
    )
    draw.text((x0 + 18, y1 - 30), metrics, fill=(60, 60, 60), font=FONT_SMALL)

def make_video_frame(frame_index, frames_by_condition):
    img = Image.new("RGB", (VIDEO_W, VIDEO_H), (246, 247, 249))
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, VIDEO_W, 74], fill=(28, 32, 38))
    draw.text((34, 18), "1000 platelet stenosis-flow benchmark: normal vs pathway-node perturbation conditions", fill=(255, 255, 255), font=FONT_TITLE)
    draw.text((34, 50), "Prescribed stenosis flow with GRN-controlled activation, stickiness, morphology, and secretion outputs", fill=(220, 225, 235), font=FONT_SMALL)

    panel_w = (VIDEO_W - 90) // 2
    panel_h = (VIDEO_H - 170) // 2

    panels = [
        (30, 100, 30 + panel_w, 100 + panel_h),
        (60 + panel_w, 100, 60 + 2 * panel_w, 100 + panel_h),
        (30, 125 + panel_h, 30 + panel_w, 125 + 2 * panel_h),
        (60 + panel_w, 125 + panel_h, 60 + 2 * panel_w, 125 + 2 * panel_h),
    ]

    for cond, box in zip(CONDITIONS, panels):
        frames = frames_by_condition[cond["id"]]
        frame = frames[min(frame_index, len(frames) - 1)]
        draw_condition_panel(draw, frame, cond, box)

    # Progress bar
    bar_x0 = 40
    bar_x1 = VIDEO_W - 40
    bar_y = VIDEO_H - 36
    p = frame_index / max(len(next(iter(frames_by_condition.values()))) - 1, 1)

    draw.line([(bar_x0, bar_y), (bar_x1, bar_y)], fill=(180, 185, 195), width=8)
    draw.line([(bar_x0, bar_y), (bar_x0 + p * (bar_x1 - bar_x0), bar_y)], fill=(240, 135, 70), width=8)
    px = int(bar_x0 + p * (bar_x1 - bar_x0))
    draw.ellipse([px - 10, bar_y - 10, px + 10, bar_y + 10], fill=(240, 135, 70), outline=(255, 255, 255), width=2)

    return np.asarray(img)

def save_video(frames_by_condition):
    out = DIRS["videos"] / "stenosis_1000_platelets_conditions_dashboard.mp4"
    nframes = min(len(v) for v in frames_by_condition.values())

    print("Creating video:", out)
    with imageio.get_writer(str(out), fps=VIDEO_FPS, codec="libx264", quality=8, macro_block_size=1) as writer:
        for i in range(nframes):
            writer.append_data(make_video_frame(i, frames_by_condition))
            if i % 25 == 0:
                print("Video frame", i, "/", nframes)

    print("Saved:", out)

# ------------------------------------------------------------
# README
# ------------------------------------------------------------

def save_readme(summary_df):
    readme = BASE / "README_stenosis_1000_platelet_conditions.md"
    text = f"""# Phase 6 Use Case 2 Extension: 1000 Platelet Stenosis-Flow Benchmark

## Purpose

This output set compares a normal platelet activation pathway against multiple pathway-node perturbation conditions in a stenosis vessel scenario using {N_PLATELETS} platelet agents.

## Conditions

1. Normal GRN
2. Rac1 pathway-node perturbation / reduced-activity condition
3. Rap1 pathway-node perturbation / reduced-activity condition
4. PLCB3/Ca2+ pathway-node perturbation / reduced-activity condition

## Important wording

These are pathway-node perturbation or knockout-like reduced-activity scenarios inside the model. They should not be described as experimentally validated gene knockouts unless the underlying GRN explicitly represents gene-level knockout biology.

## Model scope

This is a prescribed-flow, agent-based stenosis simulation. It is not a full CFD solver. The goal is to provide a reproducible visual and quantitative comparison of platelet flow, wall interaction, activation, stickiness, morphology, and secretion under the same 1000-platelet benchmark count.

## Main outputs

- `videos/stenosis_1000_platelets_conditions_dashboard.mp4`
- `simulation_snapshots/stenosis_1000_2x2_final_distribution.png`
- `timeseries/stenosis_1000_timeseries_outputs.png`
- `summary_plots/stenosis_1000_final_condition_summary.png`
- `tables/stenosis_1000_condition_summary.csv`
- `tables/stenosis_1000_benchmark.csv`

## Thesis interpretation

The normal condition is expected to produce stronger wall retention and adhesive behavior under stenotic high shear. Rac1 perturbation primarily reduces morphology and adhesion-related behavior. Rap1 perturbation strongly reduces stickiness and wall retention. PLCB3/Ca2+ perturbation reduces activation and secretion-like response.
"""
    readme.write_text(text, encoding="utf-8")
    print("Saved:", readme)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("\n===================================================")
    print("PHASE 6 USE CASE 2 EXTENSION")
    print("1000 platelet stenosis-flow benchmark")
    print("===================================================")
    print("Output folder:")
    print(BASE)
    print("Platelet count:", N_PLATELETS)
    print("Steps:", N_STEPS)
    print("===================================================\n")

    all_ts = {}
    summaries = []
    frames_by_condition = []

    frames_dict = {}

    for i, cond in enumerate(CONDITIONS):
        print("\nRunning condition:", cond["label"])
        ts, summary, frames = simulate_condition(cond, seed_offset=0)
        all_ts[cond["id"]] = ts
        summaries.append(summary)
        frames_dict[cond["id"]] = frames

        ts_path = DIRS["tables"] / f"{cond['id']}_stenosis_1000_timeseries.csv"
        ts.to_csv(ts_path, index=False)
        print("Saved:", ts_path)

    summary_df = pd.DataFrame(summaries)
    summary_path = DIRS["tables"] / "stenosis_1000_condition_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print("Saved:", summary_path)

    bench_path = DIRS["tables"] / "stenosis_1000_benchmark.csv"
    summary_df[["condition_id", "condition", "platelet_count", "steps", "runtime_seconds"]].to_csv(bench_path, index=False)
    print("Saved:", bench_path)

    save_timeseries_plot(all_ts)
    save_summary_barplot(summary_df)
    save_final_snapshot(frames_dict)

    if MAKE_VIDEO:
        save_video(frames_dict)

    save_readme(summary_df)

    print("\n===================================================")
    print("DONE - STENOSIS 1000 PLATELET CONDITION SIMULATION CREATED")
    print("Start by opening:")
    print(DIRS["videos"] / "stenosis_1000_platelets_conditions_dashboard.mp4")
    print(DIRS["snapshots"] / "stenosis_1000_2x2_final_distribution.png")
    print(DIRS["summary"] / "stenosis_1000_final_condition_summary.png")
    print("===================================================\n")

if __name__ == "__main__":
    main()
