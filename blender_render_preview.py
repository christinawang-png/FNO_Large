#!/usr/bin/env python
import bpy
import os
import sys
import csv
import math
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

BASE_DIR = PROJECT_ROOT / "implicit_bspline_dataset" / "volumes"
VDB_DIR = BASE_DIR / "vdb"

VOLUME_METADATA_CSV = BASE_DIR / "metadata_volumes.csv"

RENDER_DIR = PROJECT_ROOT / "renders_preview_vdb"
RENDER_DIR.mkdir(parents=True, exist_ok=True)

RES_X = 32
RES_Y = 32
SAMPLES = 100

# Fixed camera pose.
RADIUS = 2.3
PHI = math.radians(55.0)
THETA = math.radians(45.0)

# Approximate opacity for a camera ray crossing the Gaussian sheet
# approximately perpendicular to its surface.
#
# 0.05 = mostly transparent
# 0.50 = half opaque
# 0.95 = strongly opaque
# 0.98 = nearly opaque
TARGET_OPACITY = 0.98

# Must match the VDB export script.
BAND_SIGMAS = 3.0


# ============================================================
# SCENE UTILITIES
# ============================================================

def clean_scene():
    """Remove all objects and unused Blender datablocks."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    for mesh in list(bpy.data.meshes):
        if not mesh.users:
            bpy.data.meshes.remove(mesh)

    for volume in list(bpy.data.volumes):
        if not volume.users:
            bpy.data.volumes.remove(volume)

    for material in list(bpy.data.materials):
        if not material.users:
            bpy.data.materials.remove(material)

    for camera in list(bpy.data.cameras):
        if not camera.users:
            bpy.data.cameras.remove(camera)

    for light in list(bpy.data.lights):
        if not light.users:
            bpy.data.lights.remove(light)


def remove_previous_volume_and_camera():
    """Remove the prior preview volume/camera while preserving the world."""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    for volume in list(bpy.data.volumes):
        if not volume.users:
            bpy.data.volumes.remove(volume)

    for camera in list(bpy.data.cameras):
        if not camera.users:
            bpy.data.cameras.remove(camera)


def setup_dark_world(scene):
    """Use a dark neutral world for preview rendering."""
    world = bpy.data.worlds.new("World") if scene.world is None else scene.world
    scene.world = world
    world.use_nodes = True

    nodes = world.node_tree.nodes
    links = world.node_tree.links

    for node in list(nodes):
        nodes.remove(node)

    output_node = nodes.new("ShaderNodeOutputWorld")
    background_node = nodes.new("ShaderNodeBackground")

    background_node.inputs["Color"].default_value = (
        0.02,
        0.02,
        0.02,
        1.0,
    )
    background_node.inputs["Strength"].default_value = 1.0

    links.new(
        background_node.outputs["Background"],
        output_node.inputs["Surface"],
    )


def create_camera(scene, target_obj):
    """Create a camera that tracks the volume object's origin."""
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
    """Set camera position using the project spherical-coordinate convention."""
    x = radius * math.sin(phi) * math.cos(theta)
    y = radius * math.sin(phi) * math.sin(theta)
    z = radius * math.cos(phi)

    camera.location = (x, y, z)


# ============================================================
# VOLUME MATERIAL
# ============================================================

def density_scale_for_sigma(
    sigma,
    target_opacity=TARGET_OPACITY,
    band_sigmas=BAND_SIGMAS,
):
    """
    Convert desired opacity into a density multiplier for the Gaussian VDB.

    Generated density is:

        V(d) = exp(-0.5 * (d / sigma)^2)

    The VDB stores approximately the interval [-3 sigma, +3 sigma].
    This scaling makes thin and thick shells have approximately equal
    opacity when seen front-facing.
    """
    sigma = float(sigma)
    target_opacity = float(target_opacity)

    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    if target_opacity <= 0.0:
        return 0.0

    # Avoid log(0) and infinitely large extinction.
    target_opacity = min(target_opacity, 1.0 - 1e-6)

    optical_depth = -math.log(1.0 - target_opacity)

    retained_integral = (
        math.sqrt(2.0 * math.pi)
        * sigma
        * math.erf(float(band_sigmas) / math.sqrt(2.0))
    )

    return optical_depth / retained_integral


def make_volume_material(
    density_scale,
    color=(0.65, 0.65, 0.65, 1.0),
):
    """
    Create a Principled Volume material.

    Volume Info reads the density grid stored in the imported VDB.
    A math node multiplies it by density_scale.
    """
    material = bpy.data.materials.new(name="PreviewVolumeMaterial")
    material.use_nodes = True

    nodes = material.node_tree.nodes
    links = material.node_tree.links

    for node in list(nodes):
        nodes.remove(node)

    output_node = nodes.new("ShaderNodeOutputMaterial")
    output_node.location = (450, 0)

    principled_volume = nodes.new("ShaderNodeVolumePrincipled")
    principled_volume.location = (120, 0)
    principled_volume.inputs["Color"].default_value = color
    principled_volume.inputs["Density"].default_value = 1.0

    volume_info = nodes.new("ShaderNodeVolumeInfo")
    volume_info.location = (-320, 0)

    density_multiplier = nodes.new("ShaderNodeMath")
    density_multiplier.operation = "MULTIPLY"
    density_multiplier.location = (-100, -50)
    density_multiplier.inputs[1].default_value = float(density_scale)

    links.new(
        volume_info.outputs["Density"],
        density_multiplier.inputs[0],
    )
    links.new(
        density_multiplier.outputs["Value"],
        principled_volume.inputs["Density"],
    )
    links.new(
        principled_volume.outputs["Volume"],
        output_node.inputs["Volume"],
    )

    return material


# ============================================================
# VDB LOADING
# ============================================================

def load_vdb_volume(sample_id):
    """
    Load the VDB corresponding to this CSV sample ID.

    Expected VDB location:
        BASE_DIR / 'vdb' / 'volume_000001.vdb'
    """
    vdb_path = VDB_DIR / f"volume_{sample_id:06d}.vdb"

    if not vdb_path.is_file():
        raise FileNotFoundError(f"VDB file not found: {vdb_path}")

    bpy.ops.object.volume_import(filepath=str(vdb_path))

    volume_obj = bpy.context.object
    volume_obj.name = f"volume_{sample_id:06d}"

    return volume_obj


# ============================================================
# MAIN
# ============================================================

def main():
    start_id = None
    end_id = None

    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []

    if "--start_id" in argv:
        start_id = int(argv[argv.index("--start_id") + 1])

    if "--end_id" in argv:
        end_id = int(argv[argv.index("--end_id") + 1])

    if not VOLUME_METADATA_CSV.is_file():
        raise FileNotFoundError(
            f"Metadata CSV not found: {VOLUME_METADATA_CSV}"
        )

    scene = bpy.context.scene
    clean_scene()

    # Render setup.
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = SAMPLES

    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.resolution_percentage = 100

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    scene.cycles.volume_step_rate = 1.0
    scene.cycles.volume_max_steps = 1024

    # Preview with a visible background rather than alpha.
    scene.render.film_transparent = False

    setup_dark_world(scene)

    # Read volume metadata.
    with open(VOLUME_METADATA_CSV, "r", newline="") as f:
        volume_rows = list(csv.DictReader(f))

    if start_id is not None or end_id is not None:
        volume_rows = [
            row
            for row in volume_rows
            if (
                (start_id is None or int(row["sample_id"]) >= start_id)
                and (end_id is None or int(row["sample_id"]) <= end_id)
            )
        ]

    volume_rows.sort(key=lambda row: int(row["sample_id"]))

    print("=" * 70)
    print("VDB preview rendering")
    print(f"Metadata rows selected: {len(volume_rows):,}")
    print(f"Target opacity: {TARGET_OPACITY:.3f}")
    print(f"VDB directory: {VDB_DIR}")
    print("=" * 70)

    if not volume_rows:
        print("No samples selected.")
        return

    for index, row in enumerate(volume_rows, start=1):
        sample_id = int(row["sample_id"])
        sigma = float(row["sigma"])

        remove_previous_volume_and_camera()

        try:
            volume_obj = load_vdb_volume(sample_id)
        except FileNotFoundError as exc:
            print(f"[WARN] {exc}")
            continue

        density_scale = density_scale_for_sigma(
            sigma=sigma,
            target_opacity=TARGET_OPACITY,
        )

        material = make_volume_material(
            density_scale=density_scale,
            color=(0.65, 0.65, 0.65, 1.0),
        )

        volume_obj.data.materials.clear()
        volume_obj.data.materials.append(material)

        camera = create_camera(scene, volume_obj)
        set_camera_from_spherical(
            camera,
            radius=RADIUS,
            phi=PHI,
            theta=THETA,
        )

        image_name = (
            f"preview_vdb_s{sample_id:06d}_"
            f"sigma{sigma:.3f}_"
            f"opacity{TARGET_OPACITY:.2f}.png"
        )

        image_path = RENDER_DIR / image_name

        scene.render.filepath = str(image_path)
        bpy.ops.render.render(write_still=True)

        # Remove material after rendering, avoiding memory accumulation.
        volume_obj.data.materials.clear()

        if material.users == 0:
            bpy.data.materials.remove(material)

        print(
            f"[{index}/{len(volume_rows)}] rendered "
            f"sample_id={sample_id}, "
            f"sigma={sigma:.4f}, "
            f"density_scale={density_scale:.4f}: "
            f"{image_path}"
        )

    print("Done. Preview images written to", RENDER_DIR)


if __name__ == "__main__":
    main()