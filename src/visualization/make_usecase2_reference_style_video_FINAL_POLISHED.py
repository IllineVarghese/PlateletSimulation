from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

FPS = 20
DURATION_SEC = 36
W, H = 1280, 720

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results" / "phase6" / "usecase2_grn_knockdown_stenosis"

NORMAL_GRN = BASE / "network_figures" / "grn_normal_overview.png"
PERT_GRN = BASE / "network_figures" / "grn_rac1_perturbed_overview.png"
TABLE_DIR = BASE / "tables"
OUT_DIR = BASE / "videos"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_VIDEO = OUT_DIR / "usecase2_reference_style_grn_dashboard_FINAL_POLISHED.mp4"

CONDITION_FILES = {
    "Normal + low shear": TABLE_DIR / "normal_low_shear_timeseries.csv",
    "Normal + high shear / stenosis": TABLE_DIR / "normal_high_shear_stenosis_timeseries.csv",
    "Rac1 perturbation + low shear": TABLE_DIR / "rac1KD_low_shear_timeseries.csv",
    "Rac1 perturbation + high shear / stenosis": TABLE_DIR / "rac1KD_high_shear_stenosis_timeseries.csv",
}

OUTPUTS = ["Activation", "Stickiness", "Morphology", "Secretion"]

BG = (246, 247, 249)
WHITE = (255, 255, 255)
DARK = (28, 32, 38)
PANEL = (255, 255, 255)
GRID = (226, 229, 234)
TEXT = (25, 25, 25)
MUTED = (92, 99, 110)
RED = (210, 30, 45)
ORANGE = (240, 135, 70)
BLUE = (45, 125, 210)
LIGHT_BLUE = (95, 175, 235)
PURPLE = (112, 70, 170)

COND_COLORS = {
    "Normal + low shear": BLUE,
    "Normal + high shear / stenosis": RED,
    "Rac1 perturbation + low shear": LIGHT_BLUE,
    "Rac1 perturbation + high shear / stenosis": ORANGE,
}

TRACKED_LIST = [
    "Shear input", "PLCB3", "Ca2+", "PI3K", "Rap1", "Rac1", "RhoA",
    "Actin remodeling", "Integrin activation", "Granule release",
    "Activation", "Stickiness", "Morphology", "Secretion",
]


def font(size, bold=False):
    candidates = (
        [r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\calibrib.ttf", r"C:\Windows\Fonts\segoeuib.ttf"]
        if bold
        else [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\calibri.ttf", r"C:\Windows\Fonts\segoeui.ttf"]
    )
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


FONT_TITLE = font(25, True)
FONT_SMALL = font(13, False)
FONT_TINY = font(11, False)
FONT_LABEL = font(12, True)
FONT_NODE = font(13, False)
FONT_NODE_BOLD = font(13, True)


def find_column(df, wanted):
    wanted_clean = wanted.lower().replace(" ", "").replace("_", "")
    for col in df.columns:
        c = str(col).lower().replace(" ", "").replace("_", "")
        if c == wanted_clean:
            return col
    for col in df.columns:
        c = str(col).lower().replace(" ", "").replace("_", "")
        if wanted_clean in c or c in wanted_clean:
            return col
    return None


def load_condition(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing timeseries file: {path}")

    df = pd.read_csv(path)
    time_col = find_column(df, "time")
    if time_col is None:
        time_col = df.columns[0]

    t = df[time_col].astype(float).to_numpy()
    t = (t - np.min(t)) / max(np.max(t) - np.min(t), 1e-9)

    data = {"time": t}
    for out in OUTPUTS:
        col = find_column(df, out)
        if col is None:
            data[out] = np.zeros_like(t)
        else:
            data[out] = np.clip(df[col].astype(float).to_numpy(), 0, 1)
    return data


def load_all_data():
    return {label: load_condition(path) for label, path in CONDITION_FILES.items()}


def fit_image(img, box_w, box_h):
    img = img.convert("RGB")
    scale = min(box_w / img.width, box_h / img.height)
    nw = max(1, int(img.width * scale))
    nh = max(1, int(img.height * scale))
    resized = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (box_w, box_h), WHITE)
    canvas.paste(resized, ((box_w - nw) // 2, (box_h - nh) // 2))
    return canvas


def draw_rounded_panel(draw, box, fill=PANEL, outline=(210, 214, 220), radius=14):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2)


def interp_series(t, y, progress):
    mask = t <= progress
    if not np.any(mask):
        return t[:1], y[:1]
    last = np.where(mask)[0][-1]
    return t[: last + 1], y[: last + 1]


def map_point(x, y, box):
    x0, y0, x1, y1 = box
    px = x0 + x * (x1 - x0)
    py = y1 - y * (y1 - y0)
    return int(px), int(py)


def draw_curve(draw, t, y, box, color, width=3, dashed=False):
    if len(t) < 2:
        return
    pts = [map_point(float(ti), float(yi), box) for ti, yi in zip(t, y)]
    if not dashed:
        draw.line(pts, fill=color, width=width)
    else:
        for i in range(len(pts) - 1):
            if i % 2 == 0:
                draw.line([pts[i], pts[i + 1]], fill=color, width=width)


def draw_plot(draw, box, title, output_name, data, progress):
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=WHITE, outline=(205, 210, 218), width=1)

    for k in range(1, 5):
        gx = x0 + k * (x1 - x0) / 5
        gy = y0 + k * (y1 - y0) / 5
        draw.line([(gx, y0), (gx, y1)], fill=GRID, width=1)
        draw.line([(x0, gy), (x1, gy)], fill=GRID, width=1)

    draw.text((x0 + 8, y0 + 5), title, fill=TEXT, font=FONT_LABEL)
    plot_box = (x0 + 34, y0 + 28, x1 - 12, y1 - 24)
    draw.line([(plot_box[0], plot_box[3]), (plot_box[2], plot_box[3])], fill=(120, 120, 120), width=1)
    draw.line([(plot_box[0], plot_box[1]), (plot_box[0], plot_box[3])], fill=(120, 120, 120), width=1)

    draw.text((x0 + 5, plot_box[1] - 4), "1.0", fill=MUTED, font=FONT_TINY)
    draw.text((x0 + 5, plot_box[3] - 8), "0.0", fill=MUTED, font=FONT_TINY)

    for label, cond_data in data.items():
        t = cond_data["time"]
        y = cond_data[output_name]
        tt, yy = interp_series(t, y, progress)
        is_perturbed = "Rac1" in label
        draw_curve(draw, tt, yy, plot_box, COND_COLORS[label], width=3, dashed=is_perturbed)

    cx = int(plot_box[0] + progress * (plot_box[2] - plot_box[0]))
    draw.line([(cx, plot_box[1]), (cx, plot_box[3])], fill=(70, 70, 70), width=1)


def draw_legend(draw, x, y):
    items = [
        ("Normal low shear", BLUE, False),
        ("Normal high shear / stenosis", RED, False),
        ("Rac1 perturbation low shear", LIGHT_BLUE, True),
        ("Rac1 perturbation high shear / stenosis", ORANGE, True),
    ]
    for text, color, dashed in items:
        if dashed:
            draw.line([(x, y + 8), (x + 24, y + 8)], fill=color, width=3)
            draw.line([(x + 33, y + 8), (x + 50, y + 8)], fill=color, width=3)
        else:
            draw.line([(x, y + 8), (x + 50, y + 8)], fill=color, width=3)
        draw.text((x + 58, y), text, fill=TEXT, font=FONT_SMALL)
        y += 22


def draw_tracked_list(draw, box, progress, perturbation_phase=False):
    x0, y0, x1, y1 = box
    draw_rounded_panel(draw, box, fill=(252, 253, 255), outline=(205, 210, 218), radius=10)
    draw.text((x0 + 12, y0 + 12), "Tracked nodes", fill=TEXT, font=FONT_LABEL)

    y = y0 + 42
    for item in TRACKED_LIST:
        is_rac = item.lower() == "rac1"
        is_output = item in OUTPUTS
        if is_rac:
            draw.rounded_rectangle([x0 + 8, y - 2, x1 - 8, y + 18], radius=5, fill=(255, 230, 232))
            draw.text((x0 + 14, y), "Rac1 reduced", fill=RED, font=FONT_NODE_BOLD)
        elif is_output:
            draw.text((x0 + 14, y), item, fill=PURPLE, font=FONT_NODE_BOLD)
        else:
            draw.text((x0 + 14, y), item, fill=MUTED, font=FONT_NODE)
        y += 22

    draw.rounded_rectangle([x0 + 10, y1 - 82, x1 - 10, y1 - 18], radius=8, fill=(245, 248, 252), outline=(215, 220, 228))
    draw.text((x0 + 18, y1 - 72), "Simulation time", fill=MUTED, font=FONT_SMALL)
    draw.text((x0 + 18, y1 - 48), f"{progress:0.2f}", fill=TEXT, font=FONT_TITLE)


def draw_progress_bar(draw, progress):
    y = H - 38
    x0, x1 = 40, W - 40
    draw.line([(x0, y), (x1, y)], fill=(185, 190, 200), width=7)
    draw.line([(x0, y), (x0 + progress * (x1 - x0), y)], fill=ORANGE, width=7)
    px = int(x0 + progress * (x1 - x0))
    draw.ellipse([px - 9, y - 9, px + 9, y + 9], fill=ORANGE, outline=WHITE, width=2)
    draw.text((40, H - 28), "0.0", fill=MUTED, font=FONT_TINY)
    draw.text((W - 66, H - 28), "1.0", fill=MUTED, font=FONT_TINY)


def make_frame(frame_idx, total_frames, data, normal_img, pert_img):
    progress = frame_idx / max(total_frames - 1, 1)

    if progress < 0.33:
        current_network = normal_img
        phase_title = "Normal platelet activation GRN"
        phase_subtitle = "Baseline response under low shear and stenotic high shear"
    elif progress < 0.66:
        current_network = pert_img
        phase_title = "Rac1 pathway-node reduced-activity condition"
        phase_subtitle = "Perturbation highlights cytoskeletal and adhesion-related effects"
    else:
        current_network = pert_img
        phase_title = "Normal vs Rac1 perturbation under shear / stenosis"
        phase_subtitle = "Dynamic comparison of activation, stickiness, morphology, and secretion"

    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, W, 66], fill=DARK)
    draw.text((28, 13), "Use Case 2: GRN-based platelet activation comparison", fill=WHITE, font=FONT_TITLE)
    draw.text((28, 43), phase_subtitle, fill=(220, 225, 235), font=FONT_SMALL)

    left_box = (24, 84, 785, 635)
    curve_box = (802, 84, 1114, 635)
    list_box = (1128, 84, 1258, 635)

    draw_rounded_panel(draw, left_box, fill=WHITE, outline=(204, 208, 216), radius=16)
    draw.text((left_box[0] + 16, left_box[1] + 12), phase_title, fill=TEXT, font=FONT_LABEL)

    network_area_w = left_box[2] - left_box[0] - 24
    network_area_h = left_box[3] - left_box[1] - 54
    fitted_net = fit_image(current_network, network_area_w, network_area_h)
    img.paste(fitted_net, (left_box[0] + 12, left_box[1] + 42))

    if progress >= 0.33:
        draw.rounded_rectangle(
            [left_box[2] - 245, left_box[3] - 72, left_box[2] - 22, left_box[3] - 22],
            radius=8, fill=(255, 238, 238), outline=RED, width=2
        )
        draw.text((left_box[2] - 232, left_box[3] - 62), "Rac1 reduced-activity", fill=RED, font=FONT_SMALL)
        draw.text((left_box[2] - 232, left_box[3] - 42), "pathway-node perturbation", fill=RED, font=FONT_TINY)

    draw_rounded_panel(draw, curve_box, fill=WHITE, outline=(204, 208, 216), radius=16)
    draw.text((curve_box[0] + 12, curve_box[1] + 10), "Live GRN output dynamics", fill=TEXT, font=FONT_LABEL)

    plot_w, plot_h = 140, 188
    gap_x, gap_y = 14, 20
    p1 = (curve_box[0] + 12, curve_box[1] + 42, curve_box[0] + 12 + plot_w, curve_box[1] + 42 + plot_h)
    p2 = (p1[2] + gap_x, p1[1], p1[2] + gap_x + plot_w, p1[3])
    p3 = (p1[0], p1[3] + gap_y, p1[2], p1[3] + gap_y + plot_h)
    p4 = (p2[0], p3[1], p2[2], p3[3])

    draw_plot(draw, p1, "Activation", "Activation", data, progress)
    draw_plot(draw, p2, "Stickiness", "Stickiness", data, progress)
    draw_plot(draw, p3, "Morphology", "Morphology", data, progress)
    draw_plot(draw, p4, "Secretion", "Secretion", data, progress)

    draw_legend(draw, curve_box[0] + 16, curve_box[3] - 98)
    draw_tracked_list(draw, list_box, progress, perturbation_phase=(progress >= 0.33))

    bottom_y = 650
    draw.text((30, bottom_y), "Interpretation:", fill=RED, font=FONT_LABEL)

    if progress < 0.33:
        msg = "Normal GRN converts shear and wall-contact inputs into activation and adhesion-related outputs."
    elif progress < 0.66:
        msg = "Rac1 pathway-node perturbation keeps global activation possible but weakens cytoskeletal morphology response."
    else:
        msg = "High shear / stenosis increases activation; Rac1 perturbation mainly reduces morphology and adhesion-related behavior."

    draw.text((130, bottom_y), msg, fill=TEXT, font=FONT_SMALL)
    draw_progress_bar(draw, progress)
    return img


def main():
    print("Creating FINAL polished reference-style Use Case 2 dashboard video...")

    for p in [NORMAL_GRN, PERT_GRN]:
        if not p.exists():
            raise FileNotFoundError(f"Missing network figure: {p}")

    data = load_all_data()
    normal_img = Image.open(NORMAL_GRN).convert("RGB")
    pert_img = Image.open(PERT_GRN).convert("RGB")
    total_frames = FPS * DURATION_SEC

    print("Output:")
    print(OUT_VIDEO)

    with imageio.get_writer(str(OUT_VIDEO), fps=FPS, codec="libx264", quality=8, macro_block_size=1) as writer:
        for i in range(total_frames):
            frame = make_frame(i, total_frames, data, normal_img, pert_img)
            writer.append_data(np.asarray(frame))
            if i % 100 == 0:
                print("Frame", i, "/", total_frames)

    print("\nDONE")
    print("Saved video here:")
    print(OUT_VIDEO)


if __name__ == "__main__":
    main()
