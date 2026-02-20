import csv
from pathlib import Path
import numpy as np

L = 10.0
R = 1.0
near_band = 0.10 * R
act_threshold = 0.02

RUNS = [
    ("baseline", "results/week4/baseline", None),
    ("weak",     "results/week4/weak",     0.5),
    ("medium",   "results/week4/medium",   0.2),
    ("strong",   "results/week4/strong",   0.05),
]

def measure_one(folder: Path):
    P = np.load(folder / "positions_saved.npy")      # (T, N, 3)
    A = np.load(folder / "activation_saved.npy")     # (T, N)

    p0 = P[-2]
    p1 = P[-1]
    a  = A[-1]

    dz = (p1[:, 2] - p0[:, 2]) % L
    r  = np.sqrt(p0[:, 0] ** 2 + p0[:, 1] ** 2)

    near = (R - r) <= near_band
    far  = ~near
    sel  = near & (a >= act_threshold)

    dz_near_sel = float(dz[sel].mean()) if sel.sum() > 0 else float("nan")
    dz_far      = float(dz[far].mean()) if far.sum() > 0 else float("nan")
    A_near      = float(a[near].mean()) if near.sum() > 0 else float("nan")
    A_far       = float(a[far].mean()) if far.sum() > 0 else float("nan")
    sel_count   = int(sel.sum())

    return dz_near_sel, dz_far, A_near, A_far, sel_count


def main():
    out_csv = Path("results/week4/summary.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, folder, stick in RUNS:
        folder = Path(folder)
        if not folder.exists():
            print(f"[skip] missing folder: {folder}")
            continue

        dz_near_sel, dz_far, A_near, A_far, sel_count = measure_one(folder)

        rows.append([name, stick, dz_near_sel, dz_far, A_near, A_far, sel_count])

        print(
            f"{name:8s} stick={stick} | "
            f"dz_near_sel={dz_near_sel:.6f} dz_far={dz_far:.6f} | "
            f"A_near={A_near:.4f} A_far={A_far:.4f} | sel={sel_count}"
        )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "stick_factor", "dz_near_sel", "dz_far", "A_near", "A_far", "sel_count"])
        w.writerows(rows)

    print(f"\nWrote: {out_csv}")


if __name__ == "__main__":
    main()