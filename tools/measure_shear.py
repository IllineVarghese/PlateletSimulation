import os
import numpy as np
import csv
import yaml


# -----------------------------
# Shear study folders
# -----------------------------
RUNS = [
    ("low_shear", "results/shear/low"),
    ("medium_shear", "results/shear/medium"),
    ("high_shear", "results/shear/high"),
]

OUT_CSV = "results/shear/summary.csv"


def load_config(folder):
    """
    Tries to load config.yaml saved inside the run folder.
    If not found, falls back to default values.
    """
    cfg_path = os.path.join(folder, "config.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)
    return None


def measure_run(name, folder):
    pos_path = os.path.join(folder, "positions_saved.npy")
    act_path = os.path.join(folder, "activation_saved.npy")

    if not os.path.exists(pos_path):
        raise FileNotFoundError(f"{pos_path} not found")

    P = np.load(pos_path)  # shape: (T, N, 3)
    A = np.load(act_path) if os.path.exists(act_path) else None

    T, N, _ = P.shape

    # Use first and last saved frames
    P0 = P[0]
    P1 = P[-1]

    # radial distance
    r = np.sqrt(P0[:, 0] ** 2 + P0[:, 1] ** 2)

    radius = 1.0  # matches your geometry
    near_wall_dist_frac = 0.10
    near_wall_dist = radius * near_wall_dist_frac

    dist_to_wall = radius - r

    # near-wall mask
    near_mask = dist_to_wall <= near_wall_dist

    # activation threshold
    act_threshold = 0.0001

    if A is not None:
        A0 = A[0]
        act_mask = A0 >= act_threshold
    else:
        act_mask = np.ones(N, dtype=bool)

    # selected cohort = near-wall + activated
    sel_mask = near_mask & act_mask
    sel_count = int(sel_mask.sum())

    dz = P1[:, 2] - P0[:, 2]

    dz_near_sel = float(np.mean(dz[sel_mask])) if sel_count > 0 else float("nan")
    dz_far = float(np.mean(dz[~near_mask]))

    A_near = float(np.mean(A0[near_mask])) if A is not None else float("nan")

    return {
        "run": name,
        "dz_near_sel": dz_near_sel,
        "dz_far": dz_far,
        "A_near": A_near,
        "sel_count": sel_count,
    }


def main():
    rows = []

    print("\n=== Shear Study Summary ===\n")

    for name, folder in RUNS:
        result = measure_run(name, folder)
        rows.append([
            result["run"],
            result["dz_near_sel"],
            result["dz_far"],
            result["A_near"],
            result["sel_count"],
        ])

        print(
            f"{name:12s} | "
            f"dz_near_sel={result['dz_near_sel']:.6f} | "
            f"dz_far={result['dz_far']:.6f} | "
            f"A_near={result['A_near']:.4f} | "
            f"sel={result['sel_count']}"
        )

    os.makedirs("results/shear", exist_ok=True)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "dz_near_sel", "dz_far", "A_near", "sel_count"])
        writer.writerows(rows)

    print(f"\nSaved summary to: {OUT_CSV}\n")


if __name__ == "__main__":
    main()