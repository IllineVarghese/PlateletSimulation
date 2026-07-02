from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.backends.backend_agg import FigureCanvasAgg
import imageio.v2 as imageio

SEED = 42
N_PLATELETS = 1000
N_STEPS = 280
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
FIG_W = 18
FIG_H = 10
DPI = 140

CONDITIONS = [
    {
        "id": "normal",
        "label": "Normal GRN",
        "short": "Normal",
        "color": "#1f77b4",
        "activation_gain": 1.08,
        "stickiness_gain": 1.08,
        "morphology_gain": 1.05,
        "secretion_gain": 1.00,
        "adhesion_strength": 1.18,
        "velocity_penalty": 0.68,
    },
    {
        "id": "rac1KD",
        "label": "Rac1 perturbation",
        "short": "Rac1 KD",
        "color": "#ff7f0e",
        "activation_gain": 0.76,
        "stickiness_gain": 0.68,
        "morphology_gain": 0.25,
        "secretion_gain": 0.82,
        "adhesion_strength": 0.38,
        "velocity_penalty": 0.28,
    },
    {
        "id": "rap1KD",
        "label": "Rap1 perturbation",
        "short": "Rap1 KD",
        "color": "#2ca02c",
        "activation_gain": 0.68,
        "stickiness_gain": 0.26,
        "morphology_gain": 0.58,
        "secretion_gain": 0.72,
        "adhesion_strength": 0.20,
        "velocity_penalty": 0.18,
    },
    {
        "id": "plcb3KD",
        "label": "PLCB3/Ca2+ perturbation",
        "short": "PLCB3/Ca2+ KD",
        "color": "#d62728",
        "activation_gain": 0.42,
        "stickiness_gain": 0.42,
        "morphology_gain": 0.48,
        "secretion_gain": 0.25,
        "adhesion_strength": 0.28,
        "velocity_penalty": 0.20,
    },
]

ACT_CMAP = LinearSegmentedColormap.from_list(
    "platelet_activation_high_contrast",
    ["#1d4ed8", "#00c2ff", "#fff176", "#ff8a00", "#b30000"],
)
ACT_NORM = Normalize(vmin=0.18, vmax=0.88)


def find_project_root():
    here = Path(__file__).resolve()
    if len(here.parents) >= 3 and (here.parents[2] / "src").exists():
        return here.parents[2]
    cwd = Path.cwd()
    if (cwd / "src").exists():
        return cwd
    return here.parents[2]


PROJECT_ROOT = find_project_root()
BASE = (
    PROJECT_ROOT
    / "results"
    / "phase6"
    / "usecase2_grn_knockdown_stenosis"
    / "stenosis_1000_platelet_conditions_3d"
)
DIRS = {
    "tables": BASE / "tables",
    "snapshots": BASE / "snapshots",
    "videos": BASE / "videos",
    "timeseries": BASE / "timeseries",
    "raw": BASE / "raw_data",
}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)


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


def initialize_agents(seed_offset=0):
    rng = np.random.default_rng(SEED + seed_offset)
    x = rng.uniform(-0.8, L * 0.92, N_PLATELETS)
    y, z = sample_disc(N_PLATELETS, rng, radius=0.72)
    activation = rng.uniform(0.02, 0.12, N_PLATELETS)
    stuck = np.zeros(N_PLATELETS, dtype=bool)
    return x, y, z, activation, stuck, rng


def simulate_condition(params, seed_offset=0):
    x, y, z, activation, stuck, rng = initialize_agents(seed_offset)
    frames = []
    timeseries = []

    final_stickiness = np.zeros(N_PLATELETS)
    final_morphology = np.zeros(N_PLATELETS)
    final_secretion = np.zeros(N_PLATELETS)
    final_velocity = np.zeros(N_PLATELETS)

    start_time = time.perf_counter()

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

        adhesion_probability = (
            0.11 * params["adhesion_strength"] * stickiness * wall_zone * sten * (0.35 + 0.65 * activation)
        )
        new_stuck = (rng.uniform(0, 1, N_PLATELETS) < adhesion_probability) & (activation > 0.42)
        stuck = stuck | new_stuck

        u = U_MAX * (R0 / np.maximum(R, 0.18)) ** 1.45 * np.clip(1.0 - rr ** 2, 0.04, 1.0)
        u *= (1.0 - params["velocity_penalty"] * stuck.astype(float))
        u = np.clip(u, 0.015, 3.2)
        x += DT * u

        radial_norm_y = y / np.maximum(r, 1e-6)
        radial_norm_z = z / np.maximum(r, 1e-6)
        outward = DT * (0.030 + 0.100 * sten * activation * params["adhesion_strength"])
        y += outward * radial_norm_y + rng.normal(0, 0.006, N_PLATELETS)
        z += outward * radial_norm_z + rng.normal(0, 0.006, N_PLATELETS)

        r2 = np.sqrt(y * y + z * z)
        too_far = r2 > 0.95 * R
        y[too_far] *= 0.95 * R[too_far] / np.maximum(r2[too_far], 1e-6)
        z[too_far] *= 0.95 * R[too_far] / np.maximum(r2[too_far], 1e-6)

        r3 = np.sqrt(y * y + z * z)
        y[stuck] *= 0.90 * R[stuck] / np.maximum(r3[stuck], 1e-6)
        z[stuck] *= 0.90 * R[stuck] / np.maximum(r3[stuck], 1e-6)

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
    summary = {
        "condition_id": params["id"],
        "condition": params["label"],
        "platelet_count": N_PLATELETS,
        "final_activation": float(np.mean(activation)),
        "final_stickiness": float(np.mean(final_stickiness)),
        "final_morphology": float(np.mean(final_morphology)),
        "final_secretion": float(np.mean(final_secretion)),
        "final_stuck_fraction": float(np.mean(stuck.astype(float))),
        "mean_velocity_final": float(np.mean(final_velocity)),
        "runtime_seconds": elapsed,
        "steps": N_STEPS,
    }

    np.savez_compressed(
        DIRS["raw"] / f"{params['id']}_final_state_3d.npz",
        x=x, y=y, z=z,
        activation=activation,
        stickiness=final_stickiness,
        morphology=final_morphology,
        secretion=final_secretion,
        velocity=final_velocity,
        stuck=stuck,
    )

    return pd.DataFrame(timeseries), summary, frames


def draw_vessel(ax):
    xs = np.linspace(0, L, 150)
    theta = np.linspace(0, 2 * np.pi, 48)

    X, TH = np.meshgrid(xs, theta)
    R = vessel_radius(X)
    Y = R * np.cos(TH)
    Z = R * np.sin(TH)

    ax.plot_surface(
        X,
        Y,
        Z,
        color="#d62728",
        alpha=0.14,
        linewidth=0,
        shade=False,
        antialiased=False,
    )

    ax.plot_wireframe(
        X,
        Y,
        Z,
        rstride=8,
        cstride=18,
        color="#b91c1c",
        alpha=0.13,
        linewidth=0.35,
    )

    xs2 = np.linspace(
        STENOSIS_CENTER - STENOSIS_WIDTH,
        STENOSIS_CENTER + STENOSIS_WIDTH,
        55,
    )
    X2, TH2 = np.meshgrid(xs2, theta)
    R2 = vessel_radius(X2)
    Y2 = R2 * np.cos(TH2)
    Z2 = R2 * np.sin(TH2)

    ax.plot_surface(
        X2,
        Y2,
        Z2,
        color="#ff4d4d",
        alpha=0.24,
        linewidth=0,
        shade=False,
        antialiased=False,
    )

    for x_ring in [0.0, STENOSIS_CENTER, L]:
        r_ring = vessel_radius(x_ring)
        yy = r_ring * np.cos(theta)
        zz = r_ring * np.sin(theta)
        xx = np.full_like(theta, x_ring)
        ax.plot(xx, yy, zz, color="#991b1b", linewidth=1.3, alpha=0.75)

    flow_y = np.array([-0.42, -0.22, 0.00, 0.22, 0.42])
    flow_z = np.zeros_like(flow_y)
    flow_x = np.full_like(flow_y, 0.6)
    flow_u = np.full_like(flow_y, 2.0)

    ax.quiver(
        flow_x,
        flow_y,
        flow_z,
        flow_u,
        0 * flow_y,
        0 * flow_y,
        color="#0284c7",
        linewidth=1.4,
        arrow_length_ratio=0.16,
        alpha=0.90,
    )


def style_3d_axes(ax):
    ax.set_xlim(0, L)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)

    ax.set_xticks([0, 3, 6, 9, 12])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlabel("Vessel axis", labelpad=4, fontsize=8)
    ax.set_ylabel("")
    ax.set_zlabel("")

    ax.set_box_aspect((7.5, 1.7, 1.7))
    ax.grid(False)

    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)

    ax.tick_params(axis="x", labelsize=7)

    try:
        ax.dist = 5.8
    except Exception:
        pass


def draw_panel(ax, frame, title, azim=38):
    draw_vessel(ax)

    act = frame["activation"]
    stuck = frame["stuck"]
    free = ~stuck

    free_sizes = 9 + 34 * np.clip(act[free], 0, 1)
    stuck_sizes = 34 + 60 * np.clip(act[stuck], 0, 1)

    ax.scatter(
        frame["x"][free],
        frame["y"][free],
        frame["z"][free],
        c=act[free],
        cmap=ACT_CMAP,
        norm=ACT_NORM,
        s=free_sizes,
        alpha=0.84,
        linewidths=0,
        depthshade=False,
    )

    if np.any(stuck):
        ax.scatter(
            frame["x"][stuck],
            frame["y"][stuck],
            frame["z"][stuck],
            c=act[stuck],
            cmap=ACT_CMAP,
            norm=ACT_NORM,
            s=stuck_sizes,
            alpha=0.98,
            edgecolors="black",
            linewidths=0.85,
            depthshade=False,
        )

    style_3d_axes(ax)
    ax.view_init(elev=15, azim=azim)

    short_title = title.replace("pathway-node ", "")
    ax.set_title(short_title, fontsize=13, fontweight="bold", pad=8)

    info = (
        f"Activation {np.mean(frame['activation']):.2f}   |   "
        f"Stuck {np.mean(frame['stuck']):.2f}   |   "
        f"Velocity {np.mean(frame['velocity']):.2f}"
    )

    ax.text2D(
        0.03,
        0.04,
        info,
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="white",
            alpha=0.90,
            edgecolor="#bbbbbb",
        ),
    )


def save_snapshot(frames_dict):
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)

    for i, cond in enumerate(CONDITIONS, start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        draw_panel(ax, frames_dict[cond["id"]][-1], cond["label"], azim=38)

    sm = plt.cm.ScalarMappable(norm=ACT_NORM, cmap=ACT_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, shrink=0.54, pad=0.01)
    cbar.set_label("Platelet activation state")

    fig.suptitle(
        "3D cylindrical stenosis-flow benchmark: 1000 platelets under normal and pathway-node perturbation conditions",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )

    fig.text(
        0.5,
        0.025,
        "Prescribed-flow 3D cylindrical visualization. Black-edged platelets indicate wall-adherent / stuck platelets.",
        ha="center",
        fontsize=10,
    )

    fig.subplots_adjust(
        left=0.02,
        right=0.92,
        bottom=0.07,
        top=0.89,
        wspace=0.01,
        hspace=0.16,
    )

    out = DIRS["snapshots"] / "stenosis_1000_platelets_3d_cylindrical_snapshot.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


def make_video_frame(frame_idx, frames_dict, total_frames):
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    azim = 34 + 18 * (frame_idx / max(total_frames - 1, 1))

    for i, cond in enumerate(CONDITIONS, start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        frame = frames_dict[cond["id"]][
            min(frame_idx, len(frames_dict[cond["id"]]) - 1)
        ]
        draw_panel(ax, frame, cond["label"], azim=azim)

    sm = plt.cm.ScalarMappable(norm=ACT_NORM, cmap=ACT_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, shrink=0.54, pad=0.01)
    cbar.set_label("Activation state")

    fig.suptitle(
        "3D cylindrical stenosis simulation: normal vs pathway-node perturbation conditions",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )

    fig.text(
        0.5,
        0.945,
        "1000 platelets | prescribed stenosis flow | activation color map enhanced for visual comparison",
        ha="center",
        fontsize=10,
    )

    p = frame_idx / max(total_frames - 1, 1)

    fig.lines.append(
        plt.Line2D(
            [0.10, 0.90],
            [0.035, 0.035],
            transform=fig.transFigure,
            color="#cfcfcf",
            linewidth=7,
        )
    )

    fig.lines.append(
        plt.Line2D(
            [0.10, 0.10 + 0.80 * p],
            [0.035, 0.035],
            transform=fig.transFigure,
            color="#f28b39",
            linewidth=7,
        )
    )

    fig.text(0.10, 0.052, f"Progress {int(100 * p):d}%", fontsize=9)

    fig.subplots_adjust(
        left=0.02,
        right=0.92,
        bottom=0.08,
        top=0.89,
        wspace=0.01,
        hspace=0.16,
    )

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf, (w, h) = canvas.print_to_buffer()
    image = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))[..., :3]
    plt.close(fig)

    return image


def save_video(frames_dict):
    out = DIRS["videos"] / "stenosis_1000_platelets_3d_cylindrical_dashboard.mp4"
    total_frames = min(len(v) for v in frames_dict.values())
    print("Creating video:", out)
    with imageio.get_writer(str(out), fps=VIDEO_FPS, codec="libx264", quality=8, macro_block_size=1) as writer:
        for i in range(total_frames):
            writer.append_data(make_video_frame(i, frames_dict, total_frames))
            if i % 10 == 0:
                print(f"Video frame {i}/{total_frames}")
    print("Saved:", out)


def save_timeseries_plot(all_ts):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), dpi=180)
    axes = axes.ravel()
    metrics = [
        ("mean_activation", "Activation"),
        ("mean_stickiness", "Stickiness"),
        ("mean_morphology", "Morphology"),
        ("mean_secretion", "Secretion"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        for cond in CONDITIONS:
            ts = all_ts[cond["id"]]
            ax.plot(ts["time_normalized"], ts[metric], linewidth=2.0, label=cond["short"], color=cond["color"])
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Normalized simulation time")
        ax.set_ylabel("Mean response")
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("1000-platelet 3D stenosis benchmark: summary outputs", fontsize=15, fontweight="bold")
    out = DIRS["timeseries"] / "stenosis_1000_platelets_3d_timeseries.png"
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


def save_summary_tables(summary_df):
    summary_path = DIRS["tables"] / "stenosis_1000_platelets_3d_condition_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print("Saved:", summary_path)

    readme = BASE / "README_stenosis_1000_platelets_3d.md"
    readme.write_text(
        "# 3D cylindrical stenosis benchmark\n\n"
        "This folder contains a 3D cylindrical visualization of the prescribed-flow 1000-platelet stenosis benchmark. "
        "It compares a normal platelet activation GRN with pathway-node perturbation conditions. "
        "This is a visualization of the simulation output, not a full CFD solver.\n",
        encoding="utf-8",
    )
    print("Saved:", readme)


def main():
    print("\n====================================================")
    print("PHASE 6 USE CASE 2 - 3D CYLINDRICAL STENOSIS VIDEO")
    print("====================================================")
    print("Output folder:")
    print(BASE)
    print("\nConditions:")
    for cond in CONDITIONS:
        print("-", cond["label"])

    all_ts = {}
    summaries = []
    frames_dict = {}

    for i, cond in enumerate(CONDITIONS):
        print("\nRunning:", cond["label"])
        ts, summary, frames = simulate_condition(cond, seed_offset=i)
        all_ts[cond["id"]] = ts
        summaries.append(summary)
        frames_dict[cond["id"]] = frames
        ts_out = DIRS["tables"] / f"{cond['id']}_stenosis_1000_3d_timeseries.csv"
        ts.to_csv(ts_out, index=False)
        print("Saved:", ts_out)

    summary_df = pd.DataFrame(summaries)
    save_summary_tables(summary_df)
    save_timeseries_plot(all_ts)
    save_snapshot(frames_dict)
    if MAKE_VIDEO:
        save_video(frames_dict)

    print("\nDONE")
    print("Open these first:")
    print(DIRS["videos"] / "stenosis_1000_platelets_3d_cylindrical_dashboard.mp4")
    print(DIRS["snapshots"] / "stenosis_1000_platelets_3d_cylindrical_snapshot.png")
    print(DIRS["timeseries"] / "stenosis_1000_platelets_3d_timeseries.png")


if __name__ == "__main__":
    main()
