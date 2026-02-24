# tools/video_shear.py
"""
Week (Shear) — Shear sweep video generator

Reads saved simulation frames from:
  results/shear/low/positions_saved.npy
  results/shear/medium/positions_saved.npy
  results/shear/high/positions_saved.npy

Produces:
  results/shear/shear_flow_xz.mp4   (3-panel particle flow in XZ)
  results/shear/shear_curves.mp4    (dz(t) curves comparing scenarios)

Notes:
- Works with your existing saving format: positions_saved.npy shape = (T, N, 3)
- Uses imageio for MP4 writing.
"""

from __future__ import annotations

import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Config: where the runs live
# -----------------------------
RUNS = [
    ("low_shear", Path("results/shear/low")),
    ("medium_shear", Path("results/shear/medium")),
    ("high_shear", Path("results/shear/high")),
]

OUTPUT_DIR = Path("results/shear")
OUTPUT_FLOW = OUTPUT_DIR / "shear_flow_xz.mp4"
OUTPUT_CURVES = OUTPUT_DIR / "shear_curves.mp4"


# -----------------------------
# Helpers
# -----------------------------
def load_positions(run_dir: Path) -> np.ndarray:
    p = run_dir / "positions_saved.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run the sim first.")
    arr = np.load(p)  # (T, N, 3)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected positions_saved.npy shape (T,N,3), got {arr.shape} at {p}")
    return arr


def select_near_wall_indices(pos0: np.ndarray, near_frac: float = 0.10) -> np.ndarray:
    """
    Define a fixed near-wall cohort based on initial frame.
    Near-wall = top 'near_frac' of z (closest to wall if wall is high z).
    """
    z0 = pos0[:, 2]
    # pick top near_frac by z
    k = max(1, int(math.ceil(len(z0) * near_frac)))
    idx = np.argsort(z0)[-k:]
    return np.sort(idx)


def compute_dz_curve(positions: np.ndarray, idx: np.ndarray) -> dict:
    """
    positions: (T, N, 3)
    idx: cohort indices
    Returns mean_z over time and dz(t) relative to t0.
    """
    z_t = positions[:, idx, 2].mean(axis=1)  # (T,)
    dz_t = z_t - z_t[0]
    return {"mean_z": z_t, "dz": dz_t}


def fig_to_rgb(fig) -> np.ndarray:
    """Convert a matplotlib figure to an RGB image array."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[:, :, :3].copy()
    return rgb


# -----------------------------
# Video 1: 3-panel flow (XZ)
# -----------------------------
def make_flow_video(all_data: list[dict], fps: int = 12):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Use same number of frames across runs (min)
    T = min(d["positions"].shape[0] for d in all_data)
    N = all_data[0]["positions"].shape[1]

    # global axis limits for consistency
    # X range from all runs/frames
    x_all = np.concatenate([d["positions"][:T, :, 0].reshape(-1) for d in all_data])
    z_all = np.concatenate([d["positions"][:T, :, 2].reshape(-1) for d in all_data])
    x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
    z_min, z_max = float(np.min(z_all)), float(np.max(z_all))

    with imageio.get_writer(str(OUTPUT_FLOW), fps=fps) as writer:
        for t in range(T):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=120)
            fig.suptitle("Shear sweep — particle flow (XZ)", fontsize=14)

            for ax, d in zip(axes, all_data):
                pos = d["positions"][t]  # (N,3)
                x = pos[:, 0]
                z = pos[:, 2]
                # highlight a fixed near-wall cohort in a different color
                idx = select_near_wall_indices(d["positions"][0], near_frac=0.10)

                x_all = pos[:, 0]
                z_all = pos[:, 2]

                mask = np.zeros(len(x_all), dtype=bool)
                mask[idx] = True

                # background particles
                ax.scatter(x_all[~mask], z_all[~mask], s=2, alpha=0.25)

                # highlighted cohort
                ax.scatter(x_all[mask], z_all[mask], s=6, alpha=0.9)
                ax.set_title(d["name"])
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(z_min, z_max)
                ax.set_xlabel("x")
                ax.set_ylabel("z")
                ax.grid(True, alpha=0.3)

            fig.tight_layout(rect=[0, 0.02, 1, 0.92])
            frame = fig_to_rgb(fig)
            writer.append_data(frame)
            plt.close(fig)

    print(f"Saved flow video to: {OUTPUT_FLOW}")


# -----------------------------
# Video 2: dz(t) curves
# -----------------------------
def make_curves_video(all_data: list[dict], fps: int = 12):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    T = min(d["positions"].shape[0] for d in all_data)

    # Build curves using a fixed cohort per run (based on each run's t=0)
    curves = []
    for d in all_data:
        pos0 = d["positions"][0]
        idx = select_near_wall_indices(pos0, near_frac=0.10)
        c = compute_dz_curve(d["positions"][:T], idx)
        curves.append({"name": d["name"], "dz": c["dz"], "mean_z": c["mean_z"], "sel": len(idx)})

    # y-limits for stable video
    y_all = np.concatenate([c["dz"] for c in curves])
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    pad = 0.05 * (y_max - y_min + 1e-12)
    y_min -= pad
    y_max += pad

    with imageio.get_writer(str(OUTPUT_CURVES), fps=fps) as writer:
        for t in range(T):
            fig = plt.figure(figsize=(10, 5), dpi=120)
            ax = fig.add_subplot(111)
            ax.set_title("Shear sweep — dz(t) of near-wall cohort (fixed cohort)", fontsize=12)
            ax.set_xlabel("frame")
            ax.set_ylabel("dz (mean z - mean z at t0)")
            ax.grid(True, alpha=0.3)

            for c in curves:
                ax.plot(np.arange(t + 1), c["dz"][: t + 1], label=f"{c['name']} (sel={c['sel']})")

            ax.set_xlim(0, T - 1)
            ax.set_ylim(y_min, y_max)
            ax.legend(loc="lower left", fontsize=8)

            fig.tight_layout()
            frame = fig_to_rgb(fig)
            writer.append_data(frame)
            plt.close(fig)

    print(f"Saved curves video to: {OUTPUT_CURVES}")


def main():
    all_data = []
    for name, run_dir in RUNS:
        positions = load_positions(run_dir)
        all_data.append({"name": name, "dir": run_dir, "positions": positions})

    print("Loaded runs:")
    for d in all_data:
        print(f"  {d['name']:12s} frames={d['positions'].shape[0]} N={d['positions'].shape[1]} dir={d['dir']}")

    make_flow_video(all_data, fps=8)
    make_curves_video(all_data, fps=8)


if __name__ == "__main__":
    main()