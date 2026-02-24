import csv
from pathlib import Path

import numpy as np
import yaml

BASE = Path("results/conc")
RUNS = [
    ("low", BASE / "low"),
    ("medium", BASE / "medium"),
    ("high", BASE / "high"),
]

# Defaults (will be overridden per-run from config_used.yaml when available)
NEAR_WALL_DIST_FRAC_DEFAULT = 0.10   # activation.near_wall_dist_frac
RADIUS_DEFAULT = 1.0                 # geometry.radius
ACT_THRESH_DEFAULT = 0.005           # adhesion.act_threshold

OUT_CSV = BASE / "summary_concentration.csv"


def load_run(run_dir: Path):
    pos = np.load(run_dir / "positions_saved.npy")     # (T,N,3)
    act = np.load(run_dir / "activation_saved.npy")    # (T,N)
    return pos, act


def read_params_from_config(run_dir: Path):
    """
    Pull the actual parameters used for this run from results/.../config_used.yaml
    so the measurement matches the run.
    """
    cfg_path = run_dir / "config_used.yaml"
    near_frac = NEAR_WALL_DIST_FRAC_DEFAULT
    radius = RADIUS_DEFAULT
    act_thresh = ACT_THRESH_DEFAULT

    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        near_frac = float(cfg.get("activation", {}).get("near_wall_dist_frac", near_frac))
        radius = float(cfg.get("geometry", {}).get("radius", radius))
        act_thresh = float(cfg.get("adhesion", {}).get("act_threshold", act_thresh))

    return near_frac, radius, act_thresh


def main():
    rows = []

    for name, run_dir in RUNS:
        if not run_dir.exists():
            print(f"[SKIP] Missing folder: {run_dir}")
            continue

        pos_path = run_dir / "positions_saved.npy"
        act_path = run_dir / "activation_saved.npy"
        if not pos_path.exists() or not act_path.exists():
            print(f"[SKIP] Missing npy in {run_dir}")
            continue

        P, A = load_run(run_dir)

        if P.ndim != 3 or A.ndim != 2:
            print(f"[SKIP] Unexpected shapes in {run_dir}: P={P.shape}, A={A.shape}")
            continue

        T, N, D = P.shape
        if D < 3 or A.shape != (T, N):
            print(f"[SKIP] Shape mismatch in {run_dir}: P={P.shape}, A={A.shape}")
            continue

        near_frac, radius, act_thresh = read_params_from_config(run_dir)

        # Near-wall definition for CYLINDER: near the radial wall (r close to R)
        # near if r >= R*(1 - near_frac)
        near_r_threshold = radius * (1.0 - near_frac)

        # -----------------------------
        # 1) Last-frame stats
        # -----------------------------
        r_last = np.sqrt(P[-1, :, 0] ** 2 + P[-1, :, 1] ** 2)
        a_last = A[-1, :]
        near_last = r_last >= near_r_threshold

        frac_near_last = float(np.mean(near_last))
        mean_act_near_last = float(np.mean(a_last[near_last])) if np.any(near_last) else 0.0

        # -----------------------------
        # 2) Mean-over-time stats (stronger signal than just last frame)
        # -----------------------------
        r_all = np.sqrt(P[:, :, 0] ** 2 + P[:, :, 1] ** 2)     # (T,N)
        near_all = r_all >= near_r_threshold                   # (T,N)

        frac_near_mean = float(np.mean(near_all))
        mean_act_near_mean = float(np.mean(A[near_all])) if np.any(near_all) else 0.0

        # -----------------------------
        # 3) dz_near_sel using a fixed cohort
        # Select cohort AFTER activation has had a moment to appear
        # -----------------------------
        cohort_frame = min(5, T - 1) if T >= 2 else 0

        r_cohort = np.sqrt(P[cohort_frame, :, 0] ** 2 + P[cohort_frame, :, 1] ** 2)
        a_cohort = A[cohort_frame, :]

        near_cohort = r_cohort >= near_r_threshold
        eligible = a_cohort >= act_thresh
        sel = near_cohort & eligible
        sel_count = int(np.sum(sel))

        if sel_count > 0:
            # IMPORTANT: compare last frame against the same cohort_frame used for selection
            dz = np.abs(P[-1, sel, 2] - P[cohort_frame, sel, 2])
            dz_near_sel = float(np.mean(dz))
        else:
            dz_near_sel = float("nan")

        rows.append([
            name,
            N,
            near_frac,
            radius,
            act_thresh,
            frac_near_last,
            frac_near_mean,
            mean_act_near_last,
            mean_act_near_mean,
            dz_near_sel,
            sel_count,
        ])

        print(
            f"{name:6s} N={N:4d} | "
            f"frac_near_last={frac_near_last:.4f} | frac_near_mean={frac_near_mean:.4f} | "
            f"mean_act_near_last={mean_act_near_last:.4f} | mean_act_near_mean={mean_act_near_mean:.4f} | "
            f"dz_near_sel={dz_near_sel:.6f} | sel={sel_count}"
        )

    BASE.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "run",
            "num_agents",
            "near_wall_dist_frac",
            "radius",
            "act_threshold",
            "frac_near_last",
            "frac_near_mean",
            "mean_act_near_last",
            "mean_act_near_mean",
            "dz_near_sel",
            "sel_count",
        ])
        w.writerows(rows)

    print(f"\nWrote: {OUT_CSV}")


if __name__ == "__main__":
    main()