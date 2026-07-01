from pathlib import Path
import imageio.v2 as imageio

PROJECT_ROOT = Path(r"C:\Users\Administrator\Desktop\PlateletSimulation")

FRAME_DIR = (
    PROJECT_ROOT
    / "results"
    / "phase6"
    / "usecase1_actual_mesh"
    / "blender_cinematic_inside_vessel"
    / "slow_png_frames"
)

OUT_MP4 = (
    PROJECT_ROOT
    / "results"
    / "phase6"
    / "usecase1_actual_mesh"
    / "blender_cinematic_inside_vessel"
    / "usecase1_cinematic_10_platelets_activation_FINAL_FIXED.mp4"
)

FPS = 10

frames = sorted(FRAME_DIR.glob("frame_*.png"))

if not frames:
    raise SystemExit(f"No PNG frames found in: {FRAME_DIR}")

print(f"[OK] Found {len(frames)} PNG frames")
print(f"[OK] Writing MP4 to: {OUT_MP4}")

with imageio.get_writer(
    OUT_MP4,
    fps=FPS,
    codec="libx264",
    quality=8,
    macro_block_size=16,
) as writer:
    for i, frame_path in enumerate(frames):
        frame = imageio.imread(frame_path)
        writer.append_data(frame)

        if i % 25 == 0:
            print(f"Added frame {i}/{len(frames)}")

print("\nDONE.")
print(f"Final MP4 saved at:\n{OUT_MP4}")
print(f"Duration: about {len(frames) / FPS:.1f} seconds")