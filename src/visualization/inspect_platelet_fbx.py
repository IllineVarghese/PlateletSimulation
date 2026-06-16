from pathlib import Path
import pyvista as pv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FBX_MESH = PROJECT_ROOT / "data" / "meshes" / "platelet" / "plateletDemo.fbx"
OUTPUT_DIR = PROJECT_ROOT / "results" / "phase5" / "week1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("Phase 5 Week 1: FBX platelet mesh inspection")

    if not FBX_MESH.exists():
        raise FileNotFoundError(f"FBX mesh not found: {FBX_MESH}")

    print(f"Found FBX mesh: {FBX_MESH}")

    try:
        mesh = pv.read(FBX_MESH)

    except Exception as error:
        print("\nFBX loading failed.")
        print("-------------------")
        print("This is expected because PyVista/VTK does not reliably support FBX files.")
        print("For Phase 5 Week 1 and Week 2, we will use inactive.obj and activated.obj.")
        print("If plateletDemo.fbx is needed later, convert it to OBJ or PLY using Blender.")
        print("\nOriginal error:")
        print(error)

        note_path = OUTPUT_DIR / "fbx_loading_note.txt"
        note_path.write_text(
            "Phase 5 Week 1 FBX inspection note\n"
            "==================================\n\n"
            f"FBX file found: {FBX_MESH}\n\n"
            "Result:\n"
            "PyVista/VTK could not directly load the FBX file.\n\n"
            "Decision:\n"
            "Use inactive.obj and activated.obj for the current mesh-switching pipeline.\n"
            "Convert plateletDemo.fbx later using Blender if a better activated mesh is required.\n\n"
            f"Original error:\n{error}\n",
            encoding="utf-8",
        )

        print(f"\nSaved note: {note_path}")
        return

    print("\nFBX mesh information")
    print("--------------------")
    print(f"Type:   {type(mesh)}")
    print(f"Points: {mesh.n_points}")
    print(f"Cells:  {mesh.n_cells}")
    print(f"Bounds: {mesh.bounds}")
    print(f"Center: {mesh.center}")

    plotter = pv.Plotter(off_screen=True, window_size=(1600, 900))
    plotter.set_background("white")
    plotter.add_mesh(mesh, smooth_shading=True, show_edges=True, line_width=0.3)
    plotter.add_text(
        "plateletDemo.fbx inspection",
        position=(600, 840),
        font_size=16,
        color="black",
    )
    plotter.add_axes()
    plotter.camera_position = "iso"

    output_path = OUTPUT_DIR / "platelet_demo_fbx_inspection.png"
    plotter.screenshot(str(output_path))
    plotter.close()

    print("\nSaved output:")
    print(output_path)


if __name__ == "__main__":
    main()