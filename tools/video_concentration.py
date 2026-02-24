# tools/video_concentration.py
# Creates a video comparing LOW / MEDIUM / HIGH concentration runs.
# Output: results/conc/conc_comparison.mp4
#
# Requirements:
#   pip install matplotlib imageio imageio-ffmpeg
#
# Assumes these files exist from your runs:
#   results/conc/low/positions_saved.npy
#   results/conc/low/activation_saved.npy
#   results/conc/medium/...
#   results/conc/high/...
#
# Notes:
# - Works even if activation saturates; still shows density + motion.
# - Colors by activation (clipped to [0, act_max]) for visibility.
# - Adds a small bar overlay for near-wall fraction (computed per frame).

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio


BASE = Path("results/conc")
RUNS = [
    ("low", BASE / "low"),
    ("medium", BASE / "medium"),
    ("high", BASE / "high"),
]

OUT_MP4 = BASE / "conc_comparison.mp4"

FPS = 6
DPI = 140

# Keep these consistent with your configs
RADIUS = 1.0
NEAR_WALL_DIST_FRAC = 0.10  # activation.near_wall_dist_frac
NEAR_Z = RADIUS * NEAR_WALL_DIST_FRAC

# Visual controls
POINT_SIZE = 2.0
ACT_CLIP_MAX = 0.25  # your activation seems capped ~0.25


def load_run(run_dir: Path):
    pos_path = run_dir / "positions_saved.npy"
    act_path = run_dir / "activation_saved.npy"
    if not pos_path.exists() or not act_path.exists():
        raise FileNotFoundError(f"Missing npy in {run_dir}: {pos_path.name} / {act_path.name}")

    P = np.load(pos_path)  # (T,N,3)
    A = np.load(act_path)  # (T,N)
    if P.ndim != 3 or A.ndim != 2 or P.shape[0] != A.shape[0] or P.shape[1] != A.shape[1]:
        raise ValueError(f"Bad shapes in {run_dir}: P={P.shape}, A={A.shape}")

    return P, A


def _safe_frame_count(all_data):
    T = min(P.shape[0] for (P, _A) in all_data.values())
    return int(T)


def make_video(all_data: dict, out_path: Path, fps: int = 12):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use the minimum number of frames available across runs
    T = _safe_frame_count(all_data)

    # Build figure
    fig = plt.figure(figsize=(10, 9), dpi=DPI)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.35], hspace=0.30, wspace=0.15)

    ax_low = fig.add_subplot(gs[0, 0])
    ax_med = fig.add_subplot(gs[0, 1])
    ax_high = fig.add_subplot(gs[1, 0])
    ax_blank = fig.add_subplot(gs[1, 1])  # used for text/legend box
    ax_bar = fig.add_subplot(gs[2, :])

    ax_blank.axis("off")

    # Prepare axes
    def setup_xy(ax, title):
        ax.set_title(title)
        ax.set_xlim(-RADIUS, RADIUS)
        ax.set_ylim(0.0, RADIUS)  # assuming z in [0, RADIUS] in your cylinder/wall setup
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.grid(True, alpha=0.25)

        # near-wall band
        ax.axhspan(0.0, NEAR_Z, alpha=0.08)

    setup_xy(ax_low, "LOW concentration")
    setup_xy(ax_med, "MEDIUM concentration")
    setup_xy(ax_high, "HIGH concentration")

    fig.suptitle("Concentration Sweep — x–z view (color = activation)", fontsize=14)

    # Bar chart setup
    labels = ["low", "medium", "high"]
    ax_bar.set_title("Near-wall fraction per frame (z <= near-wall threshold)")
    ax_bar.set_ylim(0.0, 1.0)
    ax_bar.set_ylabel("fraction")
    ax_bar.grid(True, axis="y", alpha=0.25)

    # Initialize scatters
    scat_low = ax_low.scatter([], [], s=POINT_SIZE, c=[], vmin=0.0, vmax=ACT_CLIP_MAX, cmap="viridis")
    scat_med = ax_med.scatter([], [], s=POINT_SIZE, c=[], vmin=0.0, vmax=ACT_CLIP_MAX, cmap="viridis")
    scat_high = ax_high.scatter([], [], s=POINT_SIZE, c=[], vmin=0.0, vmax=ACT_CLIP_MAX, cmap="viridis")

    # Add a colorbar once (for the whole figure)
    cbar = fig.colorbar(scat_high, ax=[ax_low, ax_med, ax_high], shrink=0.85, pad=0.02)
    cbar.set_label("activation")

    # Writer
    writer = imageio.get_writer(str(out_path), fps=fps)

    for t in range(T):
        # Update each panel
        def update_scatter(scat, P, A):
            x = P[t, :, 0]
            z = P[t, :, 2]
            a = np.clip(A[t, :], 0.0, ACT_CLIP_MAX)
            scat.set_offsets(np.c_[x, z])
            scat.set_array(a)

            # Near-wall fraction
            frac_near = float(np.mean(z <= NEAR_Z))
            return frac_near, float(np.mean(a))

        P_low, A_low = all_data["low"]
        P_med, A_med = all_data["medium"]
        P_high, A_high = all_data["high"]

        frac_low, meanA_low = update_scatter(scat_low, P_low, A_low)
        frac_med, meanA_med = update_scatter(scat_med, P_med, A_med)
        frac_high, meanA_high = update_scatter(scat_high, P_high, A_high)

        # Text panel
        ax_blank.clear()
        ax_blank.axis("off")
        ax_blank.text(
            0.0,
            0.95,
            f"Frame {t+1}/{T}\n"
            f"Near-wall threshold: z <= {NEAR_Z:.3f} (R={RADIUS}, frac={NEAR_WALL_DIST_FRAC})\n\n"
            f"low    : frac_near={frac_low:.4f} | mean_act={meanA_low:.4f}\n"
            f"medium : frac_near={frac_med:.4f} | mean_act={meanA_med:.4f}\n"
            f"high   : frac_near={frac_high:.4f} | mean_act={meanA_high:.4f}\n",
            va="top",
            fontsize=11,
            family="monospace",
        )

        # Bar chart update
        ax_bar.clear()
        ax_bar.set_title("Near-wall fraction per frame (z <= near-wall threshold)")
        ax_bar.set_ylim(0.0, 1.0)
        ax_bar.set_ylabel("fraction")
        ax_bar.grid(True, axis="y", alpha=0.25)
        ax_bar.bar(labels, [frac_low, frac_med, frac_high])

        # Render to image
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]  # drop alpha
        writer.append_data(img)

    writer.close()
    plt.close(fig)

    print(f"Saved video: {out_path}")


def main():
    all_data = {}
    for name, run_dir in RUNS:
        P, A = load_run(run_dir)
        all_data[name] = (P, A)

    # Ensure keys exist
    for k in ("low", "medium", "high"):
        if k not in all_data:
            raise RuntimeError(f"Missing run '{k}' in loaded data")

    make_video(all_data, OUT_MP4, fps=FPS)


if __name__ == "__main__":
    main()