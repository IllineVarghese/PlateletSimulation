"""
Phase 6 - Use Case 1
Blender-like activated platelet render from VS Code.

This script must be run with Blender, not normal Python.

It creates a Blender/Edit-Mode-like visual:
- original activated platelet mesh
- grey surface
- orange wireframe overlay
- orthographic scientific view
- no decimation

Output:
results/phase6/usecase1_actual_mesh/blender_like_render/
"""

import bpy
from pathlib import Path
import math


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(r"C:\Users\Administrator\Desktop\PlateletSimulation")

# Prefer recovered Use Case 1 activated mesh.
ACTIVATED_OBJ = PROJECT_ROOT / "data" / "meshes" / "platelet" / "activated_usecase1.obj"

# Fallback to old activated mesh if needed.
if not ACTIVATED_OBJ.exists():
    ACTIVATED_OBJ = PROJECT_ROOT / "data" / "meshes" / "platelet" / "activated.obj"

OUTPUT_DIR = PROJECT_ROOT / "results" / "phase6" / "usecase1_actual_mesh" / "blender_like_render"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_IMAGE = OUTPUT_DIR / "activated_platelet_blender_editmode_like.png"
OUTPUT_BLEND = OUTPUT_DIR / "activated_platelet_blender_editmode_like.blend"
OUTPUT_METADATA = OUTPUT_DIR / "activated_platelet_blender_editmode_like_metadata.txt"


# ------------------------------------------------------------
# Clean scene
# ------------------------------------------------------------

def clean_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


clean_scene()


# ------------------------------------------------------------
# Materials
# ------------------------------------------------------------

def make_material(name, color, roughness=0.5):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True

    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = 0.0

    return mat


surface_mat = make_material(
    "Light grey activated platelet surface",
    (0.78, 0.80, 0.78, 1.0),
    roughness=0.55,
)

wire_mat = make_material(
    "Orange edit-mode-like wireframe",
    (1.0, 0.45, 0.0, 1.0),
    roughness=0.35,
)


# ------------------------------------------------------------
# Import OBJ
# ------------------------------------------------------------

def import_obj(obj_path):
    if not obj_path.exists():
        raise FileNotFoundError(f"Activated OBJ not found: {obj_path}")

    before = set(bpy.data.objects)

    try:
        bpy.ops.wm.obj_import(filepath=str(obj_path))
    except Exception:
        bpy.ops.import_scene.obj(filepath=str(obj_path))

    after = set(bpy.data.objects)
    imported = list(after - before)

    if not imported:
        raise RuntimeError(f"No object imported from {obj_path}")

    bpy.ops.object.select_all(action="DESELECT")

    for obj in imported:
        obj.select_set(True)

    bpy.context.view_layer.objects.active = imported[0]

    if len(imported) > 1:
        bpy.ops.object.join()

    obj = bpy.context.view_layer.objects.active
    obj.name = "Original activated platelet mesh - Use Case 1"
    obj.data.name = "activated_usecase1_original_mesh"

    return obj


activated_surface = import_obj(ACTIVATED_OBJ)

# Assign grey surface material
activated_surface.data.materials.clear()
activated_surface.data.materials.append(surface_mat)

# Smooth surface for better morphology visibility
bpy.context.view_layer.objects.active = activated_surface
activated_surface.select_set(True)
bpy.ops.object.shade_smooth()

# Put origin to geometry and normalize scale for render only
bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")

max_dim = max(activated_surface.dimensions)
if max_dim > 0:
    scale_factor = 4.2 / max_dim
    activated_surface.scale = (
        activated_surface.scale.x * scale_factor,
        activated_surface.scale.y * scale_factor,
        activated_surface.scale.z * scale_factor,
    )

activated_surface.location = (0.0, 0.0, 0.0)

# Apply transform so duplicate/wireframe behaves predictably
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

vertices_count = len(activated_surface.data.vertices)
faces_count = len(activated_surface.data.polygons)
edges_count = len(activated_surface.data.edges)
triangles_count = sum(len(poly.vertices) - 2 for poly in activated_surface.data.polygons)

print("--------------------------------------------------")
print("Activated platelet mesh imported")
print(f"File: {ACTIVATED_OBJ}")
print(f"Vertices: {vertices_count}")
print(f"Edges: {edges_count}")
print(f"Faces: {faces_count}")
print(f"Estimated triangles: {triangles_count}")
print("No decimation applied.")
print("--------------------------------------------------")


# ------------------------------------------------------------
# Duplicate mesh for orange wireframe overlay
# ------------------------------------------------------------

activated_wire = activated_surface.copy()
activated_wire.data = activated_surface.data.copy()
activated_wire.name = "Orange wireframe overlay - Blender edit mode style"
activated_wire.data.name = "activated_wireframe_overlay_mesh"
bpy.context.collection.objects.link(activated_wire)

activated_wire.data.materials.clear()
activated_wire.data.materials.append(wire_mat)

# Slightly enlarge to avoid z-fighting with grey surface
activated_wire.scale = (1.003, 1.003, 1.003)

# Add wireframe modifier to create thick visible orange topology
wire_mod = activated_wire.modifiers.new("Edit mode style orange wireframe", "WIREFRAME")
wire_mod.thickness = 0.006
wire_mod.use_even_offset = True
wire_mod.use_replace = True

# Keep surface below wireframe
activated_surface.hide_render = False
activated_wire.hide_render = False


# ------------------------------------------------------------
# Add a small title label
# ------------------------------------------------------------

label_mat = make_material(
    "Black label material",
    (0.02, 0.02, 0.02, 1.0),
    roughness=0.5,
)

bpy.ops.object.text_add(
    location=(0.0, -2.65, 0.0),
    rotation=(math.radians(90), 0, 0),
)

label = bpy.context.object
label.name = "Use Case 1 activated mesh label"
label.data.body = (
    "Original activated platelet mesh before decimation\n"
    f"Vertices: {vertices_count} | Faces: {faces_count}"
)
label.data.align_x = "CENTER"
label.data.align_y = "CENTER"
label.data.size = 0.16
label.data.materials.append(label_mat)


# ------------------------------------------------------------
# Lighting
# ------------------------------------------------------------

bpy.ops.object.light_add(type="AREA", location=(0, -3.5, 4.5))
key_light = bpy.context.object
key_light.name = "Large soft key light"
key_light.data.energy = 600
key_light.data.size = 5.0

bpy.ops.object.light_add(type="POINT", location=(-3, 3, 3))
fill_light = bpy.context.object
fill_light.name = "Small fill light"
fill_light.data.energy = 100


# ------------------------------------------------------------
# Camera: orthographic Blender-like mesh inspection view
# ------------------------------------------------------------

# Use top/bottom orthographic-style camera.
# This gives a clean scientific mesh view similar to the Blender viewport screenshot.
bpy.ops.object.camera_add(location=(0, 0, 8.0), rotation=(0, 0, 0))
camera = bpy.context.object
bpy.context.scene.camera = camera

camera.name = "Orthographic mesh inspection camera"
camera.data.type = "ORTHO"
camera.data.ortho_scale = 5.4


# ------------------------------------------------------------
# Render settings
# ------------------------------------------------------------

scene = bpy.context.scene

# Use Eevee for faster rendering and good viewport-like output
scene.render.engine = "BLENDER_EEVEE"

try:
    scene.eevee.taa_render_samples = 64
    scene.eevee.use_gtao = True
    scene.eevee.gtao_distance = 3
    scene.eevee.gtao_factor = 1.2
except Exception:
    pass

scene.render.resolution_x = 2200
scene.render.resolution_y = 1600
scene.render.film_transparent = False

# White background
if scene.world is None:
    scene.world = bpy.data.worlds.new("World")

scene.world.color = (1.0, 1.0, 1.0)

# Color management
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "Medium High Contrast"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0

# Save PNG
scene.render.filepath = str(OUTPUT_IMAGE)
bpy.ops.render.render(write_still=True)

# Save blend file for reproducibility
bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_BLEND))

# Save metadata
metadata = f"""Phase 6 Use Case 1 - Blender-like activated platelet render

Input mesh:
{ACTIVATED_OBJ}

Output image:
{OUTPUT_IMAGE}

Output blend:
{OUTPUT_BLEND}

Mesh statistics:
Vertices: {vertices_count}
Edges: {edges_count}
Faces: {faces_count}
Estimated triangles: {triangles_count}

Rendering method:
Blender background render from VS Code.
Grey original surface + orange wireframe overlay.
No decimation applied before import.

Scientific interpretation:
This image shows the original activated platelet morphology before decimation.
It can be used as the high-activation / morphology-based proxy for increased
adhesion or stickiness in Use Case 1. The mesh itself does not store a numerical
stickiness value; quantitative activation/stickiness is derived from simulation output.
"""

OUTPUT_METADATA.write_text(metadata, encoding="utf-8")

print("\nDONE.")
print(f"Saved image: {OUTPUT_IMAGE}")
print(f"Saved blend: {OUTPUT_BLEND}")
print(f"Saved metadata: {OUTPUT_METADATA}")