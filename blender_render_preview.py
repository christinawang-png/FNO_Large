#!/usr/bin/env python
import bpy
import os
import sys
import csv
import math
from pathlib import Path

# ==============================
# CONFIGURATION
# ==============================

PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DIR     = PROJECT_ROOT / "plane_dataset_4"

VOLUME_METADATA_CSV = str(BASE_DIR / "metadata_volumes.csv")

RENDER_DIR = PROJECT_ROOT / "renders_preview_vdb"
os.makedirs(RENDER_DIR, exist_ok=True)

RES_X = 64
RES_Y = 64
SAMPLES = 1024

# fixed camera sphere pose
RADIUS = 1.5
PHI    = math.radians(55)    # elevation
THETA  = math.radians(45)    # azimuth

# name of the density grid in your VDBs (from export_to_vdb.py)
VDB_GRID_NAME = "density"

# ==============================
# UTILITIES
# ==============================

def clean_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for mesh in list(bpy.data.meshes):
        if not mesh.users:
            bpy.data.meshes.remove(mesh)
    for vol in list(bpy.data.volumes):
        if not vol.users:
            bpy.data.volumes.remove(vol)
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
    key_data.energy = 1500.0
    key_data.size   = 2.0
    key_obj = bpy.data.objects.new("KeyLight", key_data)
    scene.collection.objects.link(key_obj)
    key_obj.location = (2.0, -2.0, 2.0)
    key_obj.rotation_euler = (math.radians(60), 0.0, math.radians(45))

    # fill light
    fill_data = bpy.data.lights.new(name="FillLight", type='AREA')
    fill_data.energy = 600.0
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

def make_volume_material(density_scale=5.0, color=(0.6, 0.6, 0.6, 1.0)):
    """
    Simple Principled Volume material that uses the VDB's density grid
    and scales it by density_scale.
    """
    mat = bpy.data.materials.new(name="PreviewVolumeMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (400, 0)

    vol = nodes.new("ShaderNodeVolumePrincipled")
    vol.location = (0, 0)
    vol.inputs["Color"].default_value  = color
    vol.inputs["Density"].default_value = 1.0

    # Volume Info node: reads density/color from the Volume object
    vol_info = nodes.new("ShaderNodeVolumeInfo")
    vol_info.location = (-300, 0)

    # Multiply density by a global scale
    mul = nodes.new("ShaderNodeMath")
    mul.operation = 'MULTIPLY'
    mul.inputs[1].default_value = density_scale
    mul.location = (-100, -50)

    links.new(vol_info.outputs["Density"], mul.inputs[0])
    links.new(mul.outputs["Value"], vol.inputs["Density"])
    links.new(vol.outputs["Volume"], out.inputs["Volume"])

    return mat

def load_vdb_volume(sample_id, volume_rel_path):
    """
    Load volume_XXXX.vdb corresponding to volume_XXXX.npy as a Volume object.

    volume_rel_path: relative path from metadata (e.g. "volume_0001.npy").
    """
    vol_npy = Path(volume_rel_path)
    if vol_npy.suffix != ".npy":
        raise ValueError(f"Expected .npy in volume_path, got: {volume_rel_path}")
    vol_vdb = vol_npy.with_suffix(".vdb")  # same name, .vdb extension

    full_vdb = (BASE_DIR / vol_vdb).resolve()
    if not full_vdb.is_file():
        raise FileNotFoundError(f"VDB file not found: {full_vdb}")

    # Import VDB as a Volume object; the operator will create an object
    # and make it the active object.
    bpy.ops.object.volume_import(filepath=str(full_vdb))
    vol_obj = bpy.context.object    # newly created volume object

    # Name it nicely
    vol_obj.name = f"vol_{sample_id:04d}"

    # Our export used voxelSize=1/nx with origin at (0,0,0),
    # so the volume spans roughly [0,1]^3. Move it so cube is centered.
    vol_obj.location = (-0.5, -0.5, -0.5)

    return vol_obj

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

    # somewhat finer volume sampling for preview
    scene.cycles.volume_step_rate = 1.0
    scene.cycles.volume_max_steps = 1024

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

    # volume material
    preview_vol_mat = make_volume_material(density_scale=5.0)

    for vol_row in vol_rows:
        sample_id  = int(vol_row["sample_id"])
        volume_rel = vol_row["volume_path"]   # e.g. "volume_0001.npy"

        # remove existing non-light, non-camera objects
        for o in list(bpy.data.objects):
            if o.type not in {'LIGHT', 'CAMERA'}:
                bpy.data.objects.remove(o, do_unlink=True)

        # load VDB volume
        try:
            vol_obj = load_vdb_volume(sample_id, volume_rel)
        except FileNotFoundError as e:
            print("[WARN]", e)
            continue

        # assign volume material
        vol_obj.data.materials.clear()
        vol_obj.data.materials.append(preview_vol_mat)

        # remove existing cameras, create new one
        for o in list(bpy.data.objects):
            if o.type == 'CAMERA':
                bpy.data.objects.remove(o, do_unlink=True)
        cam = create_camera(scene, vol_obj)
        set_camera_from_spherical(cam, RADIUS, PHI, THETA)

        # render a single PNG per shape
        img_name = f"preview_vdb_s{sample_id:04d}.png"
        img_path = str(RENDER_DIR / img_name)
        scene.render.filepath = img_path
        bpy.ops.render.render(write_still=True)
        print("Rendered VDB preview:", img_path)

    print("Done. VDB previews written to", RENDER_DIR)


if __name__ == "__main__":
    main()