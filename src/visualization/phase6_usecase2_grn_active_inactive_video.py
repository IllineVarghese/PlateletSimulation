from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.backends.backend_agg import FigureCanvasAgg
import imageio.v2 as imageio

# ============================================================
# PHASE 6 USE CASE 2
# GRN active/inactive network dashboard video
# ============================================================

FPS = 20
DURATION_SEC = 34
N_FRAMES = FPS * DURATION_SEC
FIG_W = 16
FIG_H = 9
DPI = 150


def find_project_root():
    here = Path(__file__).resolve()
    if len(here.parents) >= 3 and (here.parents[2] / "src").exists():
        return here.parents[2]
    cwd = Path.cwd()
    if (cwd / "src").exists():
        return cwd
    return here.parents[2]


PROJECT_ROOT = find_project_root()
BASE = PROJECT_ROOT / "results" / "phase6" / "usecase2_grn_knockdown_stenosis"
TABLE_DIR = BASE / "tables"
VIDEO_DIR = BASE / "videos"
SNAPSHOT_DIR = BASE / "network_figures"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

OUT_VIDEO = VIDEO_DIR / "usecase2_grn_active_inactive_network_dashboard.mp4"
OUT_SNAPSHOT = SNAPSHOT_DIR / "usecase2_grn_active_inactive_network_final.png"

CONDITIONS = [
    {
        "id": "normal_low",
        "title": "Normal + low shear",
        "file": TABLE_DIR / "normal_low_shear_timeseries.csv",
        "shear": "low",
        "perturbed_node": None,
    },
    {
        "id": "normal_high",
        "title": "Normal + high shear / stenosis",
        "file": TABLE_DIR / "normal_high_shear_stenosis_timeseries.csv",
        "shear": "high",
        "perturbed_node": None,
    },
    {
        "id": "rac1_high",
        "title": "Rac1 perturbation + high shear",
        "file": TABLE_DIR / "rac1KD_high_shear_stenosis_timeseries.csv",
        "shear": "high",
        "perturbed_node": "Rac1",
    },
    {
        "id": "plcb3_high",
        "title": "PLCB3/Ca2+ perturbation + high shear",
        "file": TABLE_DIR / "rac1KD_high_shear_stenosis_timeseries.csv",
        "shear": "high",
        "perturbed_node": "PLCB3",
    },
]

NODES = [
    "Shear input",
    "Wall/contact",
    "PLCB3",
    "Ca2+",
    "PI3K",
    "Rap1",
    "Rac1",
    "RhoA",
    "Actin remodeling",
    "Integrin activation",
    "Granule release",
    "Activation",
    "Stickiness",
    "Morphology",
    "Secretion",
]

NODE_POS = {
    "Shear input": (0.05, 0.72),
    "Wall/contact": (0.05, 0.34),
    "PLCB3": (0.25, 0.72),
    "Ca2+": (0.42, 0.72),
    "PI3K": (0.28, 0.44),
    "Rap1": (0.52, 0.56),
    "Rac1": (0.52, 0.30),
    "RhoA": (0.33, 0.20),
    "Actin remodeling": (0.72, 0.25),
    "Integrin activation": (0.72, 0.50),
    "Granule release": (0.72, 0.74),
    "Activation": (0.94, 0.64),
    "Stickiness": (0.94, 0.47),
    "Morphology": (0.94, 0.30),
    "Secretion": (0.94, 0.80),
}

EDGES = [
    ("Shear input", "PLCB3"),
    ("Shear input", "RhoA"),
    ("Shear input", "Rap1"),
    ("Wall/contact", "PLCB3"),
    ("Wall/contact", "PI3K"),
    ("PLCB3", "Ca2+"),
    ("Ca2+", "Rap1"),
    ("Ca2+", "Granule release"),
    ("PI3K", "Rap1"),
    ("PI3K", "Rac1"),
    ("Rap1", "Integrin activation"),
    ("Rac1", "Actin remodeling"),
    ("Rac1", "Integrin activation"),
    ("RhoA", "Actin remodeling"),
    ("Granule release", "Secretion"),
    ("Granule release", "Activation"),
    ("Integrin activation", "Stickiness"),
    ("Integrin activation", "Activation"),
    ("Actin remodeling", "Morphology"),
    ("Actin remodeling", "Stickiness"),
    ("Activation", "Stickiness"),
]

OUTPUTS = ["Activation", "Stickiness", "Morphology", "Secretion"]

ACT_CMAP = LinearSegmentedColormap.from_list(
    "grn_activity",
    ["#1d4ed8", "#00b7ff", "#fff176", "#ff8a00", "#b30000"],
)
ACT_NORM = Normalize(vmin=0.0, vmax=1.0)


def clean_name(text):
    return str(text).lower().replace(" ", "").replace("_", "").replace("/", "").replace("+", "")


def find_col(df, target):
    t = clean_name(target)
    for col in df.columns:
        c = clean_name(col)
        if c == t:
            return col
    for col in df.columns:
        c = clean_name(col)
        if t in c or c in t:
            return col
    return None


def smooth_rise(time, start=0.10, end=0.80, speed=5.0):
    y = start + (end - start) * (1.0 - np.exp(-speed * time))
    return np.clip(y, 0, 1)


def build_fallback_dynamics(condition, time):
    high = condition["shear"] == "high"
    pert = condition["perturbed_node"]

    shear_base = 0.86 if high else 0.30
    wall_base = 0.74 if high else 0.38

    shear_input = smooth_rise(time, 0.12, shear_base, 7.5)
    wall_contact = smooth_rise(time, 0.10, wall_base, 5.8)

    plcb3 = np.clip(0.25 + 0.58 * shear_input + 0.18 * wall_contact, 0, 1)
    ca2 = np.clip(0.18 + 0.65 * plcb3, 0, 1)
    pi3k = np.clip(0.18 + 0.43 * wall_contact + 0.20 * shear_input, 0, 1)
    rap1 = np.clip(0.18 + 0.42 * ca2 + 0.32 * pi3k, 0, 1)
    rac1 = np.clip(0.16 + 0.32 * pi3k + 0.24 * shear_input, 0, 1)
    rhoa = np.clip(0.16 + 0.40 * shear_input, 0, 1)

    if pert == "Rac1":
        rac1 = np.minimum(rac1 * 0.32, 0.22)
    if pert == "PLCB3":
        plcb3 = np.minimum(plcb3 * 0.26, 0.20)
        ca2 = np.minimum(ca2 * 0.42, 0.30)

    actin = np.clip(0.12 + 0.62 * rac1 + 0.22 * rhoa, 0, 1)
    integrin = np.clip(0.12 + 0.58 * rap1 + 0.22 * rac1, 0, 1)
    granule = np.clip(0.10 + 0.60 * ca2 + 0.18 * rap1, 0, 1)

    activation = np.clip(0.18 + 0.30 * ca2 + 0.28 * integrin + 0.18 * granule, 0, 1)
    stickiness = np.clip(0.12 + 0.58 * integrin + 0.18 * activation, 0, 1)
    morphology = np.clip(0.12 + 0.72 * actin, 0, 1)
    secretion = np.clip(0.10 + 0.72 * granule, 0, 1)

    if pert == "Rac1":
        morphology *= 0.72
        stickiness *= 0.88
    if pert == "PLCB3":
        activation *= 0.74
        secretion *= 0.62

    return {
        "Shear input": shear_input,
        "Wall/contact": wall_contact,
        "PLCB3": plcb3,
        "Ca2+": ca2,
        "PI3K": pi3k,
        "Rap1": rap1,
        "Rac1": rac1,
        "RhoA": rhoa,
        "Actin remodeling": actin,
        "Integrin activation": integrin,
        "Granule release": granule,
        "Activation": activation,
        "Stickiness": stickiness,
        "Morphology": morphology,
        "Secretion": secretion,
    }


def load_condition_dynamics(condition):
    path = condition["file"]

    if not path.exists():
        print("Timeseries file missing, using fallback dynamics:", path)
        time = np.linspace(0, 1, 240)
        return time, build_fallback_dynamics(condition, time)

    df = pd.read_csv(path)

    time_col = find_col(df, "time_normalized")
    if time_col is None:
        time_col = find_col(df, "time")
    if time_col is None:
        time = np.linspace(0, 1, len(df))
    else:
        time = df[time_col].astype(float).to_numpy()
        time = (time - np.min(time)) / max(np.max(time) - np.min(time), 1e-9)

    values = build_fallback_dynamics(condition, time)

    # Replace output nodes with actual generated timeseries if available.
    for output in OUTPUTS:
        candidates = [output, "mean_" + output.lower(), output.lower()]
        for cand in candidates:
            col = find_col(df, cand)
            if col is not None:
                values[output] = np.clip(df[col].astype(float).to_numpy(), 0, 1)
                break

    return time, values


def interp_value(time, series, progress):
    return float(np.interp(progress, time, series))


def node_status(value):
    if value >= 0.68:
        return "ACTIVE"
    if value <= 0.32:
        return "LOW"
    return "INTERMEDIATE"


def draw_arrow(ax, start, end, strength):
    x0, y0 = NODE_POS[start]
    x1, y1 = NODE_POS[end]
    dx = x1 - x0
    dy = y1 - y0
    length = np.sqrt(dx * dx + dy * dy)
    if length <= 1e-9:
        return

    shrink = 0.055
    sx = x0 + shrink * dx / length
    sy = y0 + shrink * dy / length
    ex = x1 - shrink * dx / length
    ey = y1 - shrink * dy / length

    alpha = 0.15 + 0.55 * strength
    lw = 0.6 + 2.1 * strength

    arr = FancyArrowPatch(
        (sx, sy),
        (ex, ey),
        arrowstyle="-|>",
        mutation_scale=8 + 8 * strength,
        linewidth=lw,
        color=(0.18, 0.18, 0.18, alpha),
        zorder=1,
    )
    ax.add_patch(arr)


def draw_node(ax, node, value, perturbed_node=None):
    x, y = NODE_POS[node]
    status = node_status(value)
    color = ACT_CMAP(ACT_NORM(value))
    radius = 0.040 if node not in OUTPUTS else 0.045

    if node in ["Shear input", "Wall/contact"]:
        edge = "#0f6fa8"
        lw = 1.5
    elif node in OUTPUTS:
        edge = "#5b21b6"
        lw = 1.6
    else:
        edge = "#333333"
        lw = 1.1

    circ = Circle((x, y), radius, facecolor=color, edgecolor=edge, linewidth=lw, zorder=4)
    ax.add_patch(circ)

    if node == perturbed_node:
        ring = Circle((x, y), radius * 1.75, facecolor="none", edgecolor="#dc2626", linewidth=2.3, zorder=5)
        ax.add_patch(ring)
        ax.plot([x - radius * 1.30, x + radius * 1.30], [y - radius * 1.30, y + radius * 1.30], color="#dc2626", linewidth=2.0, zorder=6)
        ax.plot([x - radius * 1.30, x + radius * 1.30], [y + radius * 1.30, y - radius * 1.30], color="#dc2626", linewidth=2.0, zorder=6)

    label = node
    if node == perturbed_node:
        label = node + "\nreduced"

    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=6.5 if "\n" not in label else 6.0,
        fontweight="bold" if node in OUTPUTS or node == perturbed_node else "normal",
        color="black",
        zorder=7,
    )

    ax.text(x, y - radius - 0.020, status, ha="center", va="top", fontsize=5.8, color="#444444", zorder=7)


def draw_condition_network(ax, condition, time, values, progress):
    perturbed_node = condition.get("perturbed_node")
    current = {node: interp_value(time, values[node], progress) for node in NODES}

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.03, 0.92)
    ax.axis("off")

    ax.add_patch(Rectangle((-0.02, 0.03), 1.04, 0.89, facecolor="#ffffff", edgecolor="#d1d5db", linewidth=1.0, zorder=0))

    ax.text(0.02, 0.885, condition["title"], ha="left", va="top", fontsize=10, fontweight="bold", color="#111827")

    if perturbed_node is None:
        subtitle = "Baseline GRN activity"
        sub_color = "#2563eb"
    else:
        subtitle = f"{perturbed_node} pathway-node reduced-activity condition"
        sub_color = "#dc2626"

    ax.text(0.02, 0.850, subtitle, ha="left", va="top", fontsize=7.8, color=sub_color)

    for a, b in EDGES:
        strength = 0.5 * (current[a] + current[b])
        draw_arrow(ax, a, b, strength)

    for node in NODES:
        draw_node(ax, node, current[node], perturbed_node=perturbed_node)

    active_nodes = [n for n, v in current.items() if v >= 0.68]
    low_nodes = [n for n, v in current.items() if v <= 0.32]
    active_text = ", ".join(active_nodes[:5]) if active_nodes else "none"
    low_text = ", ".join(low_nodes[:5]) if low_nodes else "none"

    output_text = (
        f"Activation {current['Activation']:.2f} | "
        f"Stickiness {current['Stickiness']:.2f} | "
        f"Morphology {current['Morphology']:.2f} | "
        f"Secretion {current['Secretion']:.2f}"
    )

    ax.text(0.02, 0.070, "Active: " + active_text, ha="left", va="bottom", fontsize=7.0, color="#b91c1c")
    ax.text(0.02, 0.043, "Low/inactive: " + low_text, ha="left", va="bottom", fontsize=7.0, color="#1d4ed8")
    ax.text(0.02, 0.016, output_text, ha="left", va="bottom", fontsize=6.8, color="#374151")


def make_frame(frame_index, all_data):
    progress = frame_index / max(N_FRAMES - 1, 1)

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    gs = fig.add_gridspec(2, 2, left=0.035, right=0.91, top=0.88, bottom=0.11, wspace=0.08, hspace=0.14)

    for i, condition in enumerate(CONDITIONS):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        time, values = all_data[condition["id"]]
        draw_condition_network(ax, condition, time, values, progress)

    fig.suptitle("Use Case 2: Active/inactive GRN network states under shear and perturbation conditions", fontsize=15, fontweight="bold", y=0.965)

    fig.text(
        0.5,
        0.925,
        "Node color shows activity: blue = low/inactive, yellow/orange = intermediate, red = active. Crossed node = pathway-node perturbation.",
        ha="center",
        fontsize=9,
        color="#374151",
    )

    sm = plt.cm.ScalarMappable(norm=ACT_NORM, cmap=ACT_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, fraction=0.018, pad=0.012)
    cbar.set_label("GRN node activity", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    progress_line_y = 0.045
    fig.lines.append(plt.Line2D([0.09, 0.88], [progress_line_y, progress_line_y], transform=fig.transFigure, color="#d1d5db", linewidth=7))
    fig.lines.append(plt.Line2D([0.09, 0.09 + 0.79 * progress], [progress_line_y, progress_line_y], transform=fig.transFigure, color="#f97316", linewidth=7))
    fig.text(0.09, 0.062, f"Simulation progress: {int(progress * 100):d}%", fontsize=9, color="#374151")

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf, (w, h) = canvas.print_to_buffer()
    image = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))[..., :3]
    plt.close(fig)
    return image


def save_snapshot(all_data):
    frame = make_frame(N_FRAMES - 1, all_data)
    imageio.imwrite(OUT_SNAPSHOT, frame)
    print("Saved:", OUT_SNAPSHOT)


def save_video(all_data):
    print("Creating:", OUT_VIDEO)
    with imageio.get_writer(str(OUT_VIDEO), fps=FPS, codec="libx264", quality=8, macro_block_size=1) as writer:
        for i in range(N_FRAMES):
            writer.append_data(make_frame(i, all_data))
            if i % 100 == 0:
                print(f"Frame {i}/{N_FRAMES}")
    print("Saved:", OUT_VIDEO)


def main():
    print("\n===================================================")
    print("PHASE 6 USE CASE 2 - GRN ACTIVE/INACTIVE VIDEO")
    print("===================================================")
    print("Output video:")
    print(OUT_VIDEO)
    print("Output snapshot:")
    print(OUT_SNAPSHOT)
    print("===================================================\n")

    all_data = {}
    for condition in CONDITIONS:
        print("Loading condition:", condition["title"])
        all_data[condition["id"]] = load_condition_dynamics(condition)

    save_snapshot(all_data)
    save_video(all_data)

    print("\nDONE")
    print("Open:")
    print(OUT_VIDEO)
    print(OUT_SNAPSHOT)


if __name__ == "__main__":
    main()
