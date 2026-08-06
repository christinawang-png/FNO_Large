#!/usr/bin/env python
import bpy
import os
import sys
import csv
import math
import numpy as np
from pathlib import Path

# ==============================
# CONFIGURATION
# ==============================

PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DIR     = PROJECT_ROOT / "plane_dataset_4"

VOLUME_METADATA_CSV = str(BASE_DIR / "metadata_volumes.csv")

RENDER_DIR = BASE_DIR / "renders_preview"
os.makedirs(RENDER_DIR, exist_ok=True)

RES_X = 64
RES_Y = 64
SAMPLES = 64

# fixed camera sphere pose
RADIUS = 1.5
PHI    = math.radians(55)    # elevation
THETA  = math.radians(45)    # azimuth

# ==============================
# UTILITIES
# ==============================

def clean_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for mesh in list(bpy.data.meshes):
        if not mesh.users:
            bpy.data.meshes.remove(mesh)
    for mat in list(bpy.data.materials):
        if not mat.users:
            bpy.data.materials.remove(mat)
    for light in list(bpy.data.lights):
        if not light.users:
            bpy.data.lights.remove(light)

def setup_world_and_lighting(scene):
    # simple dark world
    world = bpy.data.worlds.new("World") if scene.world is None else scene.world
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    for n in list(nodes):
        nodes.remove(n)
    out = nodes.new("ShaderNodeOutputWorld")
    bg  = nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.02, 0.02, 0.02, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    links.new(bg.outputs["Background"], out.inputs["Surface"])

    # remove any existing lights
    for o in list(bpy.data.objects):
        if o.type == 'LIGHT':
            bpy.data.objects.remove(o, do_unlink=True)

    # simple key light
    key_data = bpy.data.lights.new(name="KeyLight", type='AREA')
    key_data.energy = 1000.0
    key_data.size   = 2.0
    key_obj = bpy.data.objects.new("KeyLight", key_data)
    scene.collection.objects.link(key_obj)
    key_obj.location = (2.0, -2.0, 2.0)
    key_obj.rotation_euler = (math.radians(60), 0.0, math.radians(45))

    # fill light
    fill_data = bpy.data.lights.new(name="FillLight", type='AREA')
    fill_data.energy = 400.0
    fill_data.size   = 3.0
    fill_obj = bpy.data.objects.new("FillLight", fill_data)
    scene.collection.objects.link(fill_obj)
    fill_obj.location = (-2.0, -1.0, 1.5)
    fill_obj.rotation_euler = (math.radians(50), 0.0, math.radians(-30))

def create_camera(scene, target_obj):
    cam_data = bpy.data.cameras.new(name="Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    con = cam.constraints.new(type='TRACK_TO')
    con.target = target_obj
    con.track_axis = 'TRACK_NEGATIVE_Z'
    con.up_axis = 'UP_Y'
    return cam

def set_camera_from_spherical(cam, radius, phi, theta):
    x = radius * math.sin(phi) * math.cos(theta)
    y = radius * math.sin(phi) * math.sin(theta)
    z = radius * math.cos(phi)
    cam.location = (x, y, z)

def make_simple_material():
    mat = bpy.data.materials.new(name="PreviewMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.7, 0.7, 0.7, 1.0)  # light gray
    bsdf.inputs["Metallic"].default_value   = 0.0
    bsdf.inputs["Roughness"].default_value  = 0.4
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat

# ==============================
# MAIN
# ==============================

def main():
    start_id = None
    end_id   = None

    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if "--start_id" in argv:
        start_id = int(argv[argv.index("--start_id") + 1])
    if "--end_id" in argv:
        end_id = int(argv[argv.index("--end_id") + 1])

    scene = bpy.context.scene
    clean_scene()

    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = SAMPLES
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    setup_world_and_lighting(scene)

    # load volume metadata
    with open(VOLUME_METADATA_CSV, "r", newline="") as f:
        vol_reader = csv.DictReader(f)
        vol_rows = [row for row in vol_reader]

    # optional filter by sample_id
    if start_id is not None or end_id is not None:
        filtered = []
        for row in vol_rows:
            sid = int(row["sample_id"])
            if (start_id is None or sid >= start_id) and (end_id is None or sid <= end_id):
                filtered.append(row)
        vol_rows = filtered
        print(f"Previewing sample_id in [{start_id}, {end_id}], count={len(vol_rows)}")

    # simple material
    preview_mat = make_simple_material()

    for vol_row in vol_rows:
        sample_id = int(vol_row["sample_id"])
        mesh_path = vol_row["mesh_path"]

        # remove existing non-light, non-camera objects
        for o in list(bpy.data.objects):
            if o.type not in {'LIGHT', 'CAMERA'}:
                bpy.data.objects.remove(o, do_unlink=True)

        # load verts/faces from .npz
        full_mesh_path = os.path.abspath(mesh_path)
        data = np.load(full_mesh_path)
        verts = data["verts"].astype(np.float32)
        faces = data["faces"]

        # recenter mesh
        vmin   = verts.min(axis=0)
        vmax   = verts.max(axis=0)
        center = 0.5 * (vmin + vmax)
        verts_centered = verts - center

        mesh = bpy.data.meshes.new(f"mesh_{sample_id:04d}")
        mesh.from_pydata(verts_centered.tolist(), [], faces.tolist())
        mesh.update()

        shape_obj = bpy.data.objects.new(mesh.name, mesh)
        scene.collection.objects.link(shape_obj)
        shape_obj.name = f"shape_{sample_id:04d}"

        # assign simple material
        shape_obj.data.materials.clear()
        shape_obj.data.materials.append(preview_mat)

        # smooth shading
        bpy.context.view_layer.objects.active = shape_obj
        shape_obj.select_set(True)
        bpy.ops.object.shade_smooth()
        shape_obj.select_set(False)

        # remove existing cameras, create new one
        for o in list(bpy.data.objects):
            if o.type == 'CAMERA':
                bpy.data.objects.remove(o, do_unlink=True)
        cam = create_camera(scene, shape_obj)
        set_camera_from_spherical(cam, RADIUS, PHI, THETA)

        # render a single PNG per shape
        img_name = f"preview_s{sample_id:04d}.png"
        img_path = str(RENDER_DIR / img_name)
        scene.render.filepath = img_path
        bpy.ops.render.render(write_still=True)
        print("Rendered preview:", img_path)

    print("Done. Previews written to", RENDER_DIR)


if __name__ == "__main__":
    main()