from pathlib import Path
from PIL import Image, ImageEnhance
import imageio.v2 as imageio
import numpy as np

PROJECT_ROOT = Path(r"C:\Users\Administrator\Desktop\PlateletSimulation")

VIDEO_DIR = (
    PROJECT_ROOT
    / "results"
    / "phase6"
    / "usecase1_actual_mesh"
    / "blender_consistent_flow_video"
)

INPUT_VIDEO = VIDEO_DIR / "usecase1_consistent_flow_10_platelets_EDITMODELOOK_FINAL.mp4"
OUTPUT_VIDEO = VIDEO_DIR / "usecase1_consistent_flow_10_platelets_EDITMODELOOK_FINAL_COLORFUL_RELIABLE.mp4"

if not INPUT_VIDEO.exists():
    raise FileNotFoundError(f"Input video not found: {INPUT_VIDEO}")

print("Input video:", INPUT_VIDEO)
print("Output video:", OUTPUT_VIDEO)

reader = imageio.get_reader(str(INPUT_VIDEO))
meta = reader.get_meta_data()

fps = meta.get("fps", 12)
writer = imageio.get_writer(
    str(OUTPUT_VIDEO),
    fps=fps,
    codec="libx264",
    quality=8,
    macro_block_size=1
)

frame_count = 0

for frame in reader:
    img = Image.fromarray(frame)

    # Safe visible enhancement only: no glow, no sequencer, no white blank output.
    img = ImageEnhance.Color(img).enhance(1.45)
    img = ImageEnhance.Contrast(img).enhance(1.12)
    img = ImageEnhance.Brightness(img).enhance(1.04)
    img = ImageEnhance.Sharpness(img).enhance(1.08)

    out = np.asarray(img)
    writer.append_data(out)

    frame_count += 1
    if frame_count % 50 == 0:
        print("Processed frames:", frame_count)

reader.close()
writer.close()

print("DONE")
print("Frames processed:", frame_count)
print("Saved colorful video here:")
print(OUTPUT_VIDEO)
