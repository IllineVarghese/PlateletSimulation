from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

# ============================================================
# USE CASE 2 THESIS-STYLE VIDEO
# Normal vs Rac1 pathway-node perturbation under
# low shear and high shear / stenosis
# ============================================================

FPS = 20
VIDEO_W = 1280
VIDEO_H = 720
HEADER_H = 72
FOOTER_H = 125
TRANSITION_SEC = 0.6

BG = (248, 249, 251)
HEADER_BG = (18, 42, 76)
FOOTER_BG = (235, 239, 244)
TEXT = (20, 20, 20)
SUBTEXT = (70, 70, 70)
WHITE = (255, 255, 255)
ACCENT_RED = (190, 35, 45)
ACCENT_BLUE = (45, 120, 210)
ACCENT_GREEN = (30, 150, 80)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results" / "phase6" / "usecase2_grn_knockdown_stenosis"

FIG_NORMAL_GRN = BASE / "network_figures" / "grn_normal_overview.png"
FIG_PERTURBED_GRN = BASE / "network_figures" / "grn_rac1_perturbed_overview.png"
FIG_HEATMAP = BASE / "heatmaps" / "node_activity_heatmap_4_conditions.png"
FIG_TIMESERIES_ALL = BASE / "timeseries" / "timeseries_all_outputs_4_conditions.png"
FIG_TIMESERIES_LOW = BASE / "timeseries" / "timeseries_low_shear_normal_vs_rac1KD.png"
FIG_TIMESERIES_HIGH = BASE / "timeseries" / "timeseries_high_shear_stenosis_normal_vs_rac1KD.png"
FIG_VISUAL_2X2 = BASE / "simulation_snapshots" / "usecase2_2x2_visual_comparison.png"
FIG_EFFECT = BASE / "summary_plots" / "rac1_perturbation_effect_size_barplot.png"
FIG_BAR = BASE / "summary_plots" / "final_output_comparison_barplot.png"

OUT_DIR = BASE / "videos"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_VIDEO = OUT_DIR / "usecase2_grn_knockdown_stenosis_thesis_video.mp4"

# ------------------------------------------------------------
# Fonts
# ------------------------------------------------------------
def get_font(size=24, bold=False):
    candidates = []
    if bold:
        candidates = [
            r"C:\Windows\Fonts\arialbd.ttf",
            r"C:\Windows\Fonts\calibrib.ttf",
            r"C:\Windows\Fonts\segoeuib.ttf",
        ]
    else:
        candidates = [
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\calibri.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
        ]

    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)

    return ImageFont.load_default()

FONT_TITLE = get_font(34, bold=True)
FONT_SUBTITLE = get_font(21, bold=False)
FONT_BULLET = get_font(22, bold=False)
FONT_BULLET_BOLD = get_font(22, bold=True)
FONT_SMALL = get_font(18, bold=False)
FONT_BIG = get_font(42, bold=True)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def check_required_files():
    required = [
        FIG_NORMAL_GRN,
        FIG_PERTURBED_GRN,
        FIG_HEATMAP,
        FIG_TIMESERIES_ALL,
        FIG_TIMESERIES_LOW,
        FIG_TIMESERIES_HIGH,
        FIG_VISUAL_2X2,
        FIG_EFFECT,
        FIG_BAR,
    ]

    missing = [p for p in required if not p.exists()]
    if missing:
        print("Missing files:")
        for m in missing:
            print(" -", m)
        raise FileNotFoundError("One or more required figure files are missing.")

def fit_image_to_box(img, box_w, box_h):
    img = img.convert("RGB")
    scale = min(box_w / img.width, box_h / img.height)
    new_w = max(1, int(img.width * scale))
    new_h = max(1, int(img.height * scale))
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (box_w, box_h), (255, 255, 255))
    x = (box_w - new_w) // 2
    y = (box_h - new_h) // 2
    canvas.paste(resized, (x, y))
    return canvas

def wrap_text(draw, text, font, max_width):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = word if current == "" else current + " " + word
        bbox = draw.textbbox((0, 0), test, font=font)
        width = bbox[2] - bbox[0]
        if width <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines

def draw_bullets(draw, bullets, x, y, max_width, line_gap=8):
    for bullet in bullets:
        bullet_text = f"• {bullet}"
        lines = wrap_text(draw, bullet_text, FONT_BULLET, max_width)
        for line in lines:
            draw.text((x, y), line, fill=TEXT, font=FONT_BULLET)
            bbox = draw.textbbox((x, y), line, font=FONT_BULLET)
            y += (bbox[3] - bbox[1]) + line_gap
        y += 4
    return y

def make_title_only_slide(title, subtitle_lines, seconds=3.0):
    img = Image.new("RGB", (VIDEO_W, VIDEO_H), BG)
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, VIDEO_W, HEADER_H], fill=HEADER_BG)
    draw.text((36, 18), title, fill=WHITE, font=FONT_TITLE)

    center_y = 180
    for i, line in enumerate(subtitle_lines):
        font = FONT_BIG if i == 0 else FONT_BULLET
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        draw.text(((VIDEO_W - w) // 2, center_y), line, fill=TEXT, font=font)
        center_y += 55 if i == 0 else 40

    return {"image": img, "seconds": seconds}

def make_figure_slide(title, subtitle, figure_path, bullets, seconds=4.0):
    img = Image.new("RGB", (VIDEO_W, VIDEO_H), BG)
    draw = ImageDraw.Draw(img)

    # Header
    draw.rectangle([0, 0, VIDEO_W, HEADER_H], fill=HEADER_BG)
    draw.text((28, 16), title, fill=WHITE, font=FONT_TITLE)
    draw.text((28, 46), subtitle, fill=(220, 230, 245), font=FONT_SMALL)

    # Figure region
    fig_x = 30
    fig_y = HEADER_H + 14
    fig_w = VIDEO_W - 60
    fig_h = VIDEO_H - HEADER_H - FOOTER_H - 26

    draw.rounded_rectangle(
        [fig_x - 4, fig_y - 4, fig_x + fig_w + 4, fig_y + fig_h + 4],
        radius=14,
        fill=(255, 255, 255),
        outline=(210, 214, 220),
        width=2
    )

    figure = Image.open(figure_path)
    fitted = fit_image_to_box(figure, fig_w, fig_h)
    img.paste(fitted, (fig_x, fig_y))

    # Footer
    foot_y = VIDEO_H - FOOTER_H
    draw.rectangle([0, foot_y, VIDEO_W, VIDEO_H], fill=FOOTER_BG)
    draw.line([0, foot_y, VIDEO_W, foot_y], fill=(210, 214, 220), width=2)

    draw.text((28, foot_y + 12), "Key interpretation", fill=ACCENT_RED, font=FONT_BULLET_BOLD)
    draw_bullets(draw, bullets, 40, foot_y + 42, VIDEO_W - 80)

    return {"image": img, "seconds": seconds}

def blend_frames(img_a, img_b, n_frames):
    frames = []
    for i in range(n_frames):
        alpha = (i + 1) / n_frames
        blended = Image.blend(img_a, img_b, alpha)
        frames.append(np.array(blended))
    return frames

def append_hold(writer, pil_img, seconds):
    n = int(seconds * FPS)
    arr = np.array(pil_img)
    for _ in range(n):
        writer.append_data(arr)

# ------------------------------------------------------------
# Build slides
# ------------------------------------------------------------
def build_slides():
    slides = []

    slides.append(
        make_title_only_slide(
            "Use Case 2 — GRN-based biological comparison",
            [
                "Normal platelet activation vs Rac1 pathway-node perturbation",
                "Low shear and high shear / stenosis conditions",
                "Outputs: activation, stickiness, morphology, and secretion",
            ],
            seconds=3.2
        )
    )

    slides.append(
        make_figure_slide(
            "Normal platelet activation GRN",
            "Baseline signaling architecture under shear / wall-contact input",
            FIG_NORMAL_GRN,
            [
                "The normal GRN links shear and wall-contact inputs to intracellular signaling and behavior outputs.",
                "Outputs aggregate into activation, stickiness, morphology, and secretion.",
            ],
            seconds=4.0
        )
    )

    slides.append(
        make_figure_slide(
            "Rac1 pathway-node perturbation GRN",
            "Reduced-activity condition used as the perturbation scenario",
            FIG_PERTURBED_GRN,
            [
                "Rac1 is perturbed as a meaningful cytoskeletal / adhesion-related pathway node.",
                "This perturbation is expected to most strongly affect morphology and adhesive response.",
            ],
            seconds=4.2
        )
    )

    slides.append(
        make_figure_slide(
            "Node-activity heatmap across four conditions",
            "Normal vs perturbation under low shear and high shear / stenosis",
            FIG_HEATMAP,
            [
                "High shear / stenosis elevates several signaling outputs in both GRN states.",
                "Rac1 perturbation most clearly lowers morphology-related downstream response.",
            ],
            seconds=4.2
        )
    )

    slides.append(
        make_figure_slide(
            "Dynamic response overview",
            "All four outputs tracked over normalized simulation time",
            FIG_TIMESERIES_ALL,
            [
                "Normal high-shear / stenosis reaches the strongest activation and stickiness levels.",
                "The perturbation preserves general response structure but suppresses selected outputs.",
            ],
            seconds=4.6
        )
    )

    slides.append(
        make_figure_slide(
            "Low-shear comparison",
            "Normal GRN vs Rac1 pathway-node perturbation",
            FIG_TIMESERIES_LOW,
            [
                "Under low shear, perturbation effects are present but relatively modest.",
                "Morphology shows the clearest drop even in the low-shear condition.",
            ],
            seconds=4.0
        )
    )

    slides.append(
        make_figure_slide(
            "High shear / stenosis comparison",
            "Normal GRN vs Rac1 pathway-node perturbation",
            FIG_TIMESERIES_HIGH,
            [
                "Under stenotic shear, the system becomes more activated overall.",
                "The perturbation effect becomes more visible, especially for stickiness and morphology.",
            ],
            seconds=4.2
        )
    )

    slides.append(
        make_figure_slide(
            "2×2 biological visual comparison",
            "Visual translation of GRN state into platelet behavior under flow",
            FIG_VISUAL_2X2,
            [
                "The 2×2 panel shows how GRN state and shear regime jointly shape observed behavior.",
                "High shear / stenosis visibly increases activation-related response, while perturbation weakens morphology.",
            ],
            seconds=4.8
        )
    )

    slides.append(
        make_figure_slide(
            "Final output comparison",
            "Summary of activation, stickiness, morphology, and secretion",
            FIG_BAR,
            [
                "Normal + high shear / stenosis gives the strongest overall response profile.",
                "Rac1 perturbation lowers outputs, with morphology showing the strongest reduction.",
            ],
            seconds=4.0
        )
    )

    slides.append(
        make_figure_slide(
            "Perturbation effect size",
            "Normal minus Rac1 perturbation under low and stenotic shear",
            FIG_EFFECT,
            [
                "Morphology has the largest perturbation effect size.",
                "The effect is slightly stronger under high shear / stenosis than under low shear.",
            ],
            seconds=4.0
        )
    )

    slides.append(
        make_title_only_slide(
            "Conclusion",
            [
                "High shear / stenosis amplifies platelet activation behavior.",
                "Rac1 pathway-node perturbation most strongly weakens morphology and adhesion-related response.",
                "Use Case 2 provides a clear thesis-level GRN comparison under biomechanical stress.",
            ],
            seconds=4.0
        )
    )

    return slides

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    check_required_files()
    slides = build_slides()
    transition_frames = max(1, int(TRANSITION_SEC * FPS))

    print("Creating video:")
    print(OUT_VIDEO)

    with imageio.get_writer(
        OUT_VIDEO,
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=None
    ) as writer:

        for i, slide in enumerate(slides):
            append_hold(writer, slide["image"], slide["seconds"])

            if i < len(slides) - 1:
                next_img = slides[i + 1]["image"]
                for frame in blend_frames(slide["image"], next_img, transition_frames):
                    writer.append_data(frame)

    print("\nDONE")
    print("Saved video to:")
    print(OUT_VIDEO)

if __name__ == "__main__":
    main()