import os
from pathlib import Path

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")  # important for headless rendering
import matplotlib.pyplot as plt

import imageio.v2 as imageio


RUNS = [
    ("baseline", "results/week4/baseline"),
    ("weak",     "results/week4/weak"),
    ("medium",   "results/week4/medium"),
    ("strong",   "results/week4/strong"),
]

OUT_DIR = Path("results/week4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_VIDEO_FLOW = OUT_DIR / "week4_flow_xz.mp4"
OUT_VIDEO_CURVES = OUT_DIR / "week4_curves.mp4"


def load_cfg(run_dir: Path) -> dict:
    cfg_path = run_dir / "config_used.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def get_params(cfg: dict):
    radius = float(cfg.get("geometry", {}).get("radius", 1.0))
    length = float(cfg.get("geometry", {}).get("length", 10.0))

    near_frac = float(cfg.get("activation", {}).get("near_wall_dist_frac", 0.10))
    near_wall_dist = near_frac * radius

    act_threshold = float(cfg.get("adhesion", {}).get("act_threshold", 0.02))
    stick_factor = cfg.get("adhesion", {}).get("stick_factor", None)
    adh_enabled = bool(cfg.get("adhesion", {}).get("enabled", True))

    # For baseline where adhesion disabled, it’s nice to show stick as None
    if not adh_enabled:
        stick_factor = None

    return radius, length, near_wall_dist, act_threshold, stick_factor


def load_run_arrays(run_dir: Path):
    pos_path = run_dir / "positions_saved.npy"
    act_path = run_dir / "activation_saved.npy"
    steps_path = run_dir / "positions_saved_steps.npy"

    if not pos_path.exists() or not act_path.exists():
        raise FileNotFoundError(f"Missing .npy files in {run_dir}")

    P = np.load(pos_path)  # (T, N, 3)
    A = np.load(act_path)  # (T, N)

    if steps_path.exists():
        saved_steps = np.load(steps_path)  # (T,)
    else:
        saved_steps = np.arange(P.shape[0], dtype=np.int32)

    return P, A, saved_steps


def canvas_to_rgb(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[:, :, :3].copy()
    return rgb


def make_flow_xz_video(all_data, fps=12):
    """
    2x2 animation: scatter X vs Z, colored by activation.
    This makes the Poiseuille flow (in +Z) visible.
    """
    # Determine common length, radius
    # Use cfg from first run for axis bounds (they’re all same in your setup)
    _, first = all_data[0]
    radius = first["radius"]
    length = first["length"]

    T = min(d["P"].shape[0] for _, d in all_data)

    writer = imageio.get_writer(
        OUT_VIDEO_FLOW,
        fps=fps,
        codec="libx264",
        macro_block_size=1,  # prevents resizing warning
    )

    try:
        for t in range(T):
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes = axes.ravel()

            for ax, (name, d) in zip(axes, all_data):
                P = d["P"][t]
                A = d["A"][t]

                x = P[:, 0]
                z = P[:, 2]

                sc = ax.scatter(x, z, s=4, c=A, vmin=0.0, vmax=1.0)
                ax.set_title(f"{name} | stick={d['stick']}")
                ax.set_xlim(-radius * 1.05, radius * 1.05)
                ax.set_ylim(0.0, length)
                ax.set_xlabel("x")
                ax.set_ylabel("z")
                ax.grid(True, alpha=0.2)

            fig.suptitle("Week 4 — Flow view (X vs Z), colored by activation", fontsize=14)
            fig.colorbar(sc, ax=axes.tolist(), fraction=0.02, pad=0.02, label="activation")

            frame = canvas_to_rgb(fig)
            writer.append_data(frame)
            plt.close(fig)

    finally:
        writer.close()

    print(f"Saved: {OUT_VIDEO_FLOW}")


def make_curves_video(all_data, fps=12):
    """
    Animated plot: mean z(t) for a selected cohort.

    Robust cohort selection:
    - Try to select particles that are near-wall AND activated (A>=threshold)
      at the first frame where this set is non-empty.
    - If still empty (shouldn't happen), fall back to near-wall only.

    This prevents the "need at least one array to concatenate" crash.
    """

    dt_default = 0.001

    curves = []
    T = min(d["P"].shape[0] for _, d in all_data)

    for name, d in all_data:
        cfg = d["cfg"]
        dt = float(cfg.get("simulation", {}).get("dt", dt_default))
        saved_steps = d["saved_steps"][:T]
        t_axis = saved_steps.astype(np.float64) * dt

        # ---- find a frame where near-wall & activated selection is non-empty ----
        sel = None
        sel_count = 0
        chosen_frame = 0

        for t_sel in range(min(T, 10)):  # only search first ~10 frames
            Pt = d["P"][t_sel]
            At = d["A"][t_sel]

            rt = np.sqrt(Pt[:, 0] ** 2 + Pt[:, 1] ** 2)
            dist_to_wall = d["radius"] - rt

            sel_try = (dist_to_wall <= d["near_wall_dist"]) & (At >= d["act_threshold"])
            cnt = int(sel_try.sum())
            if cnt > 0:
                sel = sel_try
                sel_count = cnt
                chosen_frame = t_sel
                break

        # ---- fallback: near-wall only ----
        if sel is None:
            Pt = d["P"][0]
            rt = np.sqrt(Pt[:, 0] ** 2 + Pt[:, 1] ** 2)
            dist_to_wall = d["radius"] - rt
            sel = (dist_to_wall <= d["near_wall_dist"])
            sel_count = int(sel.sum())
            chosen_frame = 0

        # ---- compute mean z(t) for that fixed cohort ----
        mean_z = []
        for t in range(T):
            zt = d["P"][t, :, 2]
            if sel_count > 0:
                mean_z.append(float(np.mean(zt[sel])))
            else:
                mean_z.append(float("nan"))

        mean_z = np.array(mean_z, dtype=np.float64)

        curves.append({
            "name": name,
            "stick": d["stick"],
            "t": t_axis,
            "mean_z": mean_z,
            "sel_count": sel_count,
            "chosen_frame": chosen_frame,
        })

    # ---- axis limits (robust: handle all-nan safely) ----
    t_max = max(c["t"][-1] for c in curves)

    finite_vals = []
    for c in curves:
        finite_vals.append(c["mean_z"][np.isfinite(c["mean_z"])])
    finite_vals = [v for v in finite_vals if v.size > 0]

    if len(finite_vals) == 0:
        y_min, y_max = 0.0, 1.0
    else:
        all_y = np.concatenate(finite_vals)
        y_min, y_max = float(all_y.min()), float(all_y.max())
        pad = 0.05 * (y_max - y_min + 1e-9)
        y_min -= pad
        y_max += pad

    writer = imageio.get_writer(
        OUT_VIDEO_CURVES,
        fps=fps,
        codec="libx264",
        macro_block_size=1,
    )

    try:
        for k in range(T):
            fig, ax = plt.subplots(figsize=(10, 6))

            for c in curves:
                ax.plot(
                    c["t"][:k+1],
                    c["mean_z"][:k+1],
                    label=f"{c['name']} stick={c['stick']} sel={c['sel_count']} (frame={c['chosen_frame']})"
                )

            ax.set_title("Week 4 — mean z(t) of near-wall activated cohort (fixed cohort)")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("mean z")
            ax.set_xlim(0.0, t_max)
            ax.set_ylim(y_min, y_max)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")

            frame = canvas_to_rgb(fig)
            writer.append_data(frame)
            plt.close(fig)

    finally:
        writer.close()

    print(f"Saved: {OUT_VIDEO_CURVES}")


def main():
    all_data = []
    for name, run_dir_str in RUNS:
        run_dir = Path(run_dir_str)

        cfg = load_cfg(run_dir)
        radius, length, near_wall_dist, act_threshold, stick = get_params(cfg)
        P, A, saved_steps = load_run_arrays(run_dir)

        all_data.append((name, {
            "cfg": cfg,
            "radius": radius,
            "length": length,
            "near_wall_dist": near_wall_dist,
            "act_threshold": act_threshold,
            "stick": stick,
            "P": P,
            "A": A,
            "saved_steps": saved_steps,
        }))

    print("Loaded runs:")
    for name, d in all_data:
        print(f"  {name:8s} frames={d['P'].shape[0]} N={d['P'].shape[1]} stick={d['stick']}")

    make_flow_xz_video(all_data, fps=12)
    make_curves_video(all_data, fps=12)


if __name__ == "__main__":
    main()