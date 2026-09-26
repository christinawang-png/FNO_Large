#!/usr/bin/env python
import bpy
import csv
import math
import sys
from pathlib import Path

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

BASE_DIR = PROJECT_ROOT / "implicit_bspline_dataset" / "volumes"
METADATA_CSV = BASE_DIR / "metadata_volumes.csv"

RENDER_DIR = PROJECT_ROOT / "implicit_bspline_dataset" / "surface_previews"
RENDER_DIR.mkdir(parents=True, exist_ok=True)

# Render settings.
RES_X = 256
RES_Y = 256
SAMPLES = 256

# Fixed camera pose.
CAMERA_RADIUS = 2.4
CAMERA_PHI = math.radians(55.0)
CAMERA_THETA = math.radians(45.0)

# Render one preview per geometry rather than five duplicate previews,
# one for each sigma.
ONE_RENDER_PER_GEOMETRY = True

# Neutral preview material.
SURFACE_COLOR = (0.55, 0.72, 0.95, 1.0)
METALLIC = 0.0
ROUGHNESS = 0.35


# ============================================================
# UTILITIES
# ============================================================

def resolve_csv_path(path_from_csv, base_dir: Path) -> Path:
    """
    Resolve an absolute or relative path from metadata.

    Absolute paths are used directly.
    """
    path = Path(str(path_from_csv))

    if path.is_absolute():
        return path

    candidate = base_dir / path
    if candidate.is_file():
        return candidate

    candidate = base_dir.parent / path
    if candidate.is_file():
        return candidate

    return base_dir / path


def clean_scene():
    """Delete all scene objects and unused data blocks."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    for mesh in list(bpy.data.meshes):
        if not mesh.users:
            bpy.data.meshes.remove(mesh)

    for camera in list(bpy.data.cameras):
        if not camera.users:
            bpy.data.cameras.remove(camera)

    for material in list(bpy.data.materials):
        if not material.users:
            bpy.data.materials.remove(material)

    for light in list(bpy.data.lights):
        if not light.users:
            bpy.data.lights.remove(light)


def clear_previous_preview_objects():
    """Remove previous mesh, camera, empty, and lights."""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    for mesh in list(bpy.data.meshes):
        if not mesh.users:
            bpy.data.meshes.remove(mesh)

    for camera in list(bpy.data.cameras):
        if not camera.users:
            bpy.data.cameras.remove(camera)

    for light in list(bpy.data.lights):
        if not light.users:
            bpy.data.lights.remove(light)


# ============================================================
# WORLD / LIGHTING / CAMERA
# ============================================================

def setup_world(scene):
    """Set up a soft gray world background."""
    world = bpy.data.worlds.new("World") if scene.world is None else scene.world
    scene.world = world
    world.use_nodes = True

    nodes = world.node_tree.nodes
    links = world.node_tree.links

    for node in list(nodes):
        nodes.remove(node)

    output_node = nodes.new("ShaderNodeOutputWorld")
    background_node = nodes.new("ShaderNodeBackground")

    background_node.inputs["Color"].default_value = (0.04, 0.04, 0.04, 1.0)
    background_node.inputs["Strength"].default_value = 0.25

    links.new(
        background_node.outputs["Background"],
        output_node.inputs["Surface"],
    )


def point_light_at(light_obj, target=(0.0, 0.0, 0.0)):
    """Rotate a light so its local negative Z axis points at target."""
    direction = (
        target[0] - light_obj.location.x,
        target[1] - light_obj.location.y,
        target[2] - light_obj.location.z,
    )
    light_obj.rotation_euler = (
        bpy.mathutils.Vector(direction)
        .to_track_quat("-Z", "Y")
        .to_euler()
    )


def add_area_light(
    scene,
    name,
    location,
    energy,
    size,
):
    """Create an area light pointing at world origin."""
    light_data = bpy.data.lights.new(name=name, type="AREA")
    light_data.energy = float(energy)
    light_data.shape = "DISK"
    light_data.size = float(size)

    light_obj = bpy.data.objects.new(name, light_data)
    scene.collection.objects.link(light_obj)
    light_obj.location = location

    # Avoid bpy.mathutils ambiguity by importing locally.
    from mathutils import Vector
    direction = Vector((0.0, 0.0, 0.0)) - light_obj.location
    light_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    return light_obj


def setup_studio_lighting(scene):
    """Create three neutral area lights for geometry inspection."""
    add_area_light(
        scene,
        name="KeyLight",
        location=(2.5, -2.5, 3.0),
        energy=900.0,
        size=3.0,
    )

    add_area_light(
        scene,
        name="FillLight",
        location=(-3.0, -1.5, 1.5),
        energy=450.0,
        size=4.0,
    )

    add_area_light(
        scene,
        name="RimLight",
        location=(0.0, 3.0, 2.5),
        energy=700.0,
        size=2.5,
    )


def create_camera(scene, target_obj):
    """Create a camera tracking target_obj."""
    camera_data = bpy.data.cameras.new(name="Camera")
    camera_obj = bpy.data.objects.new("Camera", camera_data)

    scene.collection.objects.link(camera_obj)
    scene.camera = camera_obj

    constraint = camera_obj.constraints.new(type="TRACK_TO")
    constraint.target = target_obj
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"

    return camera_obj


def set_camera_from_spherical(camera, radius, phi, theta):
    """Position camera around the origin."""
    x = radius * math.sin(phi) * math.cos(theta)
    y = radius * math.sin(phi) * math.sin(theta)
    z = radius * math.cos(phi)

    camera.location = (x, y, z)


# ============================================================
# MESH / MATERIAL
# ============================================================

def make_preview_material():
    """Create a neutral Principled surface material."""
    material = bpy.data.materials.new(name="SurfacePreviewMaterial")
    material.use_nodes = True

    nodes = material.node_tree.nodes
    links = material.node_tree.links

    for node in list(nodes):
        nodes.remove(node)

    output_node = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")

    bsdf.inputs["Base Color"].default_value = SURFACE_COLOR
    bsdf.inputs["Metallic"].default_value = METALLIC
    bsdf.inputs["Roughness"].default_value = ROUGHNESS

    links.new(bsdf.outputs["BSDF"], output_node.inputs["Surface"])

    return material


def load_mesh_object(sample_id, mesh_path_from_csv, material):
    """
    Load a marching-cubes mesh from NPZ and center it using its mesh
    bounding-box center.

    Returns:
        mesh_obj
        original mesh center, as ndarray shape (3,)
    """
    mesh_path = resolve_csv_path(mesh_path_from_csv, BASE_DIR)

    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    data = np.load(mesh_path)

    verts = data["verts"].astype(np.float32)
    faces = data["faces"].astype(np.int32)

    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"Unexpected verts shape {verts.shape} in {mesh_path}")

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Unexpected faces shape {faces.shape} in {mesh_path}")

    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    center = 0.5 * (vmin + vmax)

    centered_verts = verts - center

    blender_mesh = bpy.data.meshes.new(f"mesh_{sample_id:06d}")
    blender_mesh.from_pydata(
        centered_verts.tolist(),
        [],
        faces.tolist(),
    )
    blender_mesh.update()

    mesh_obj = bpy.data.objects.new(
        f"surface_{sample_id:06d}",
        blender_mesh,
    )
    bpy.context.scene.collection.objects.link(mesh_obj)

    mesh_obj.data.materials.append(material)

    bpy.context.view_layer.objects.active = mesh_obj
    mesh_obj.select_set(True)
    bpy.ops.object.shade_smooth()
    mesh_obj.select_set(False)

    return mesh_obj, center


# ============================================================
# METADATA SELECTION
# ============================================================

def parse_args():
    """
    Supported command-line options:

      --start_id 1
      --end_id 20
      --max_geometries 10

    Example:
      blender -b -P preview_surfaces.py -- --start_id 1 --end_id 20
    """
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []

    start_id = None
    end_id = None
    max_geometries = None

    if "--start_id" in argv:
        start_id = int(argv[argv.index("--start_id") + 1])

    if "--end_id" in argv:
        end_id = int(argv[argv.index("--end_id") + 1])

    if "--max_geometries" in argv:
        max_geometries = int(argv[argv.index("--max_geometries") + 1])

    return start_id, end_id, max_geometries


def select_rows(rows, start_id, end_id, max_geometries):
    """Filter sample IDs and optionally remove repeated geometry rows."""
    rows = sorted(rows, key=lambda row: int(row["sample_id"]))

    if start_id is not None or end_id is not None:
        rows = [
            row
            for row in rows
            if (
                (start_id is None or int(row["sample_id"]) >= start_id)
                and (end_id is None or int(row["sample_id"]) <= end_id)
            )
        ]

    if ONE_RENDER_PER_GEOMETRY:
        selected_rows = []
        seen_geometry_ids = set()

        for row in rows:
            geometry_id = int(row["geometry_id"])

            if geometry_id in seen_geometry_ids:
                continue

            seen_geometry_ids.add(geometry_id)
            selected_rows.append(row)

        rows = selected_rows

    if max_geometries is not None:
        rows = rows[:max_geometries]

    return rows


# ============================================================
# MAIN
# ============================================================

def main():
    start_id, end_id, max_geometries = parse_args()

    if not METADATA_CSV.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_CSV}")

    scene = bpy.context.scene
    clean_scene()

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = SAMPLES

    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.resolution_percentage = 100

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"

    scene.render.film_transparent = False

    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    setup_world(scene)

    with open(METADATA_CSV, "r", newline="") as f:
        all_rows = list(csv.DictReader(f))

    required_fields = {"sample_id", "geometry_id", "mesh_path", "sigma"}
    missing = required_fields - set(all_rows[0].keys())

    if missing:
        raise KeyError(
            f"Metadata CSV is missing required fields: {sorted(missing)}"
        )

    rows = select_rows(
        rows=all_rows,
        start_id=start_id,
        end_id=end_id,
        max_geometries=max_geometries,
    )

    print("=" * 70)
    print("Implicit B-spline surface preview")
    print(f"Metadata: {METADATA_CSV}")
    print(f"Selected preview rows: {len(rows)}")
    print(f"One render per geometry: {ONE_RENDER_PER_GEOMETRY}")
    print(f"Output directory: {RENDER_DIR}")
    print("=" * 70)

    if not rows:
        print("No rows selected.")
        return

    # Reuse the same material throughout all previews.
    preview_material = make_preview_material()

    for preview_index, row in enumerate(rows, start=1):
        sample_id = int(row["sample_id"])
        geometry_id = int(row["geometry_id"])
        sigma = float(row["sigma"])
        mesh_path = row["mesh_path"]

        clear_previous_preview_objects()

        # clear_previous_preview_objects() can remove the material only if
        # its user count is zero. Recreate if Blender removed it.
        if "SurfacePreviewMaterial" not in bpy.data.materials:
            preview_material = make_preview_material()

        try:
            mesh_obj, mesh_center = load_mesh_object(
                sample_id=sample_id,
                mesh_path_from_csv=mesh_path,
                material=preview_material,
            )
        except (FileNotFoundError, ValueError, KeyError) as exc:
            print(
                f"[WARN] sample_id={sample_id}, "
                f"geometry_id={geometry_id}: {exc}"
            )
            continue

        # Camera tracks this empty at the centered object's origin.
        origin_empty = bpy.data.objects.new(
            f"origin_{sample_id:06d}",
            None,
        )
        origin_empty.location = (0.0, 0.0, 0.0)
        scene.collection.objects.link(origin_empty)

        setup_world(scene)
        setup_studio_lighting(scene)

        camera = create_camera(scene, origin_empty)
        set_camera_from_spherical(
            camera,
            radius=CAMERA_RADIUS,
            phi=CAMERA_PHI,
            theta=CAMERA_THETA,
        )

        image_name = (
            f"surface_geom{geometry_id:06d}_"
            f"sample{sample_id:06d}_"
            f"sigma{sigma:.3f}.png"
        )
        image_path = RENDER_DIR / image_name

        scene.render.filepath = str(image_path)
        bpy.ops.render.render(write_still=True)

        print(
            f"[{preview_index}/{len(rows)}] rendered "
            f"geometry_id={geometry_id}, sample_id={sample_id}, "
            f"sigma={sigma:.3f}: {image_path}"
        )

    print("Done. Surface previews written to", RENDER_DIR)


if __name__ == "__main__":
    main()