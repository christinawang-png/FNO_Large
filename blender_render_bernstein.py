#!/usr/bin/env python
import bpy
import os
import csv
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import imageio.v2 as imageio


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

BASE_DIR = PROJECT_ROOT / "implicit_bspline_dataset" / "volumes"
VDB_DIR = BASE_DIR / "vdb"

VOLUME_METADATA_CSV = BASE_DIR / "metadata_volumes.csv"

RENDER_DIR = PROJECT_ROOT / "implicit_bspline_dataset" / "renders_balanced"
ALPHA_DIR = PROJECT_ROOT / "implicit_bspline_dataset" / "hard_alpha_balanced"

RENDER_DIR.mkdir(parents=True, exist_ok=True)
ALPHA_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Balanced geometry selection
# ------------------------------------------------------------

# Number of distinct 2x2x2 control grids / geometries selected.
#
# If each geometry has 5 sigma values:
#     10,000 geometries x 5 sigmas = 50,000 volume samples.
TARGET_NUM_GEOMETRIES = 10000

# Makes the geometry subset reproducible.
GEOMETRY_SELECTION_SEED = 2025

# ------------------------------------------------------------
# Views and render modes
# ------------------------------------------------------------

NUM_VIEWS_PER_VOLUME = 16

RENDER_SURFACE = True
RENDER_VOLUME = True

# 16 views x 2 modes = 32 images/volume.
CAMERA_RADIUS = 2.2

# ------------------------------------------------------------
# Render settings
# ------------------------------------------------------------

RES_X = 32
RES_Y = 32

# Start lower for a test run. Increase only if noise is a problem.
SAMPLES = 512

# Opacity applies to both modes:
#
# Surface:
#   shader mixture between Transparent and Principled BSDF.
#
# Volume:
#   converted to a sigma-normalized density multiplier.
MIN_OPACITY = 0.01
MAX_OPACITY = 0.99

# Must agree with export_to_vdb_shards.py.
BAND_SIGMAS = 3.0

# ------------------------------------------------------------
# Environments
# ------------------------------------------------------------

NUM_GLOBAL_ENVS = 128

ENV_DIR = BASE_DIR / "envmaps"
ENV_DIR.mkdir(parents=True, exist_ok=True)

ENV_H = 32
ENV_W = 32
SH_ORDER = 2

# ------------------------------------------------------------
# Shard output
# ------------------------------------------------------------

SHARD_SIZE = 5000


# ============================================================
# PATH / SCENE UTILITIES
# ============================================================

def resolve_csv_path(path_from_csv, base_dir: Path) -> Path:
    """
    Resolve absolute paths and relative metadata paths.

    Your current project can use absolute paths directly.
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
    """Remove all objects and unused Blender datablocks."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    for mesh in list(bpy.data.meshes):
        if not mesh.users:
            bpy.data.meshes.remove(mesh)

    for volume in list(bpy.data.volumes):
        if not volume.users:
            bpy.data.volumes.remove(volume)

    for camera in list(bpy.data.cameras):
        if not camera.users:
            bpy.data.cameras.remove(camera)

    for material in list(bpy.data.materials):
        if not material.users:
            bpy.data.materials.remove(material)


def clear_shape_objects():
    """
    Remove objects belonging to the previous geometry/sample.

    Reusable materials and world data remain.
    """
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    for mesh in list(bpy.data.meshes):
        if not mesh.users:
            bpy.data.meshes.remove(mesh)

    for volume in list(bpy.data.volumes):
        if not volume.users:
            bpy.data.volumes.remove(volume)

    for camera in list(bpy.data.cameras):
        if not camera.users:
            bpy.data.cameras.remove(camera)


# ============================================================
# WORLD / CAMERA
# ============================================================

def set_black_world(scene):
    """Black non-illuminating world used for silhouette-mask rendering."""
    world = bpy.data.worlds.new("World") if scene.world is None else scene.world
    scene.world = world
    world.use_nodes = True

    nodes = world.node_tree.nodes
    links = world.node_tree.links

    for node in list(nodes):
        nodes.remove(node)

    output_node = nodes.new("ShaderNodeOutputWorld")
    background_node = nodes.new("ShaderNodeBackground")

    background_node.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    background_node.inputs["Strength"].default_value = 0.0

    links.new(
        background_node.outputs["Background"],
        output_node.inputs["Surface"],
    )


def create_camera(scene, target_obj):
    """Create a camera tracking target_obj."""
    camera_data = bpy.data.cameras.new(name="Camera")
    camera_obj = bpy.data.objects.new("Camera", camera_data)

    scene.collection.objects.link(camera_obj)
    scene.camera = camera_obj

    track = camera_obj.constraints.new(type="TRACK_TO")
    track.target = target_obj
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    return camera_obj


def set_camera_from_direction(camera, direction, radius):
    """Place camera at radius * normalized(direction)."""
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)

    camera.location = (
        float(radius * direction[0]),
        float(radius * direction[1]),
        float(radius * direction[2]),
    )


def direction_to_phi_theta(direction):
    """
    Convert direction to this convention:

        x = r sin(phi) cos(theta)
        y = r sin(phi) sin(theta)
        z = r cos(phi)
    """
    x, y, z = np.asarray(direction, dtype=np.float64)

    phi = math.acos(float(np.clip(z, -1.0, 1.0)))
    theta = math.atan2(float(y), float(x))

    if theta < 0.0:
        theta += 2.0 * math.pi

    return float(phi), float(theta)


# ============================================================
# MATERIALS
# ============================================================

def make_surface_material():
    """Create one reusable transparent/Principled surface material."""
    material = bpy.data.materials.new(name="ReusableSurfaceMaterial")
    material.use_nodes = True

    nodes = material.node_tree.nodes
    links = material.node_tree.links

    for node in list(nodes):
        nodes.remove(node)

    output_node = nodes.new("ShaderNodeOutputMaterial")
    output_node.location = (450, 0)

    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.name = "SurfacePrincipled"
    principled.location = (0, 100)

    transparent = nodes.new("ShaderNodeBsdfTransparent")
    transparent.location = (0, -50)

    opacity_mix = nodes.new("ShaderNodeMixShader")
    opacity_mix.name = "SurfaceOpacityMix"
    opacity_mix.location = (220, 40)

    links.new(transparent.outputs["BSDF"], opacity_mix.inputs[1])
    links.new(principled.outputs["BSDF"], opacity_mix.inputs[2])
    links.new(opacity_mix.outputs["Shader"], output_node.inputs["Surface"])

    return material


def update_surface_material(
    material,
    rgb,
    metallic,
    roughness,
    specular,
    opacity,
):
    """Update the reusable surface material for one frame."""
    nodes = material.node_tree.nodes

    principled = nodes["SurfacePrincipled"]
    opacity_mix = nodes["SurfaceOpacityMix"]

    r, g, b = [float(x) for x in rgb]

    principled.inputs["Base Color"].default_value = (r, g, b, 1.0)
    principled.inputs["Metallic"].default_value = float(metallic)
    principled.inputs["Roughness"].default_value = float(roughness)

    # Blender version compatibility.
    if "Specular" in principled.inputs:
        principled.inputs["Specular"].default_value = float(specular)
    elif "Specular IOR Level" in principled.inputs:
        principled.inputs["Specular IOR Level"].default_value = float(specular)

    opacity_mix.inputs["Fac"].default_value = float(opacity)

    material_type = "plastic" if metallic <= 1e-3 else "metal"

    return (r, g, b, float(opacity)), material_type


def make_volume_material():
    """Create one reusable VDB density material."""
    material = bpy.data.materials.new(name="ReusableVolumeMaterial")
    material.use_nodes = True

    nodes = material.node_tree.nodes
    links = material.node_tree.links

    for node in list(nodes):
        nodes.remove(node)

    output_node = nodes.new("ShaderNodeOutputMaterial")
    output_node.location = (450, 0)

    principled_volume = nodes.new("ShaderNodeVolumePrincipled")
    principled_volume.name = "VolumePrincipled"
    principled_volume.location = (120, 0)

    volume_info = nodes.new("ShaderNodeVolumeInfo")
    volume_info.location = (-300, 0)

    multiplier = nodes.new("ShaderNodeMath")
    multiplier.name = "VolumeDensityMultiplier"
    multiplier.operation = "MULTIPLY"
    multiplier.location = (-80, -50)
    multiplier.inputs[1].default_value = 1.0

    links.new(
        volume_info.outputs["Density"],
        multiplier.inputs[0],
    )
    links.new(
        multiplier.outputs["Value"],
        principled_volume.inputs["Density"],
    )
    links.new(
        principled_volume.outputs["Volume"],
        output_node.inputs["Volume"],
    )

    return material


def update_volume_material(material, rgb, density_scale):
    """Update reusable volume material values."""
    nodes = material.node_tree.nodes

    principled_volume = nodes["VolumePrincipled"]
    multiplier = nodes["VolumeDensityMultiplier"]

    r, g, b = [float(x) for x in rgb]

    principled_volume.inputs["Color"].default_value = (r, g, b, 1.0)
    multiplier.inputs[1].default_value = float(density_scale)

    return (r, g, b, 1.0)


def make_mask_material():
    """Opaque white emission material for surface silhouette alpha masks."""
    material = bpy.data.materials.new(name="ReusableMaskMaterial")
    material.use_nodes = True

    nodes = material.node_tree.nodes
    links = material.node_tree.links

    for node in list(nodes):
        nodes.remove(node)

    output_node = nodes.new("ShaderNodeOutputMaterial")
    output_node.location = (220, 0)

    emission = nodes.new("ShaderNodeEmission")
    emission.location = (0, 0)
    emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission.inputs["Strength"].default_value = 1.0

    links.new(emission.outputs["Emission"], output_node.inputs["Surface"])

    return material


# ============================================================
# VOLUME OPACITY
# ============================================================

def density_scale_for_sigma(sigma, target_opacity):
    """
    Convert desired front-facing opacity to a density multiplier.

    Stored VDB density is approximately:

        V(d) = exp(-0.5 * (d / sigma)^2)

    The VDB retains approximately ±3 sigma. This scaling compensates for
    sigma, so thin and thick Gaussian shells have approximately equal
    normal-incidence opacity for a common target_opacity.

    Grazing-angle views naturally look more opaque.
    """
    sigma = float(sigma)
    target_opacity = float(target_opacity)

    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    if target_opacity <= 0.0:
        return 0.0

    target_opacity = min(target_opacity, 1.0 - 1e-6)

    optical_depth = -math.log(1.0 - target_opacity)

    retained_integral = (
        math.sqrt(2.0 * math.pi)
        * sigma
        * math.erf(BAND_SIGMAS / math.sqrt(2.0))
    )

    return optical_depth / retained_integral


# ============================================================
# MESH / VDB LOADING
# ============================================================

def load_mesh_object(sample_id, mesh_path_from_csv):
    """
    Load the F=0 marching-cubes mesh and center it by bounding-box center.

    The corresponding VDB receives the same shift to remain aligned.
    """
    mesh_path = resolve_csv_path(mesh_path_from_csv, BASE_DIR)

    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    mesh_data_np = np.load(mesh_path)

    verts = mesh_data_np["verts"].astype(np.float32)
    faces = mesh_data_np["faces"].astype(np.int32)

    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"Unexpected verts shape: {verts.shape}")

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Unexpected faces shape: {faces.shape}")

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
        f"shape_mesh_{sample_id:06d}",
        blender_mesh,
    )
    bpy.context.scene.collection.objects.link(mesh_obj)

    bpy.context.view_layer.objects.active = mesh_obj
    mesh_obj.select_set(True)
    bpy.ops.object.shade_smooth()
    mesh_obj.select_set(False)

    return mesh_obj, center


def load_vdb_volume(sample_id):
    """
    Load VDB written by export_to_vdb_shards.py:

        volumes/vdb/volume_000001.vdb
    """
    vdb_path = VDB_DIR / f"volume_{sample_id:06d}.vdb"

    if not vdb_path.is_file():
        raise FileNotFoundError(f"VDB not found: {vdb_path}")

    bpy.ops.object.volume_import(filepath=str(vdb_path))

    volume_obj = bpy.context.object
    volume_obj.name = f"volume_{sample_id:06d}"

    return volume_obj


# ============================================================
# CAMERA DISTRIBUTION
# ============================================================

def fibonacci_sphere_directions(num_views):
    """
    Approximate uniform directions on the sphere.

    Better coverage than independent random views when using only 16 cameras.
    """
    directions = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))

    for i in range(num_views):
        z = 1.0 - 2.0 * ((i + 0.5) / num_views)
        radius_xy = math.sqrt(max(0.0, 1.0 - z * z))
        angle = golden_angle * i

        x = radius_xy * math.cos(angle)
        y = radius_xy * math.sin(angle)

        directions.append(np.array([x, y, z], dtype=np.float64))

    return directions


def random_rotation_matrix(rng):
    """
    Uniform random 3D rotation.

    Each volume receives a deterministic rotation of the base camera pattern.
    """
    u1, u2, u3 = rng.uniform(0.0, 1.0, size=3)

    qx = math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2)
    qy = math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2)
    qz = math.sqrt(u1) * math.sin(2.0 * math.pi * u3)
    qw = math.sqrt(u1) * math.cos(2.0 * math.pi * u3)

    return np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


# ============================================================
# SPHERICAL HARMONICS / ENVIRONMENT MAPS
# ============================================================

def sh_lm_list(order):
    pairs = []

    for l in range(order + 1):
        for m in range(-l, l + 1):
            pairs.append((l, m))

    return pairs


def sh_basis_dir_l2(x, y, z):
    c0 = 0.28209479177387814
    c1 = 0.4886025119029199
    c2 = 1.0925484305920792
    c3 = 0.31539156525252005
    c4 = 0.5462742152960396

    Y = np.empty(9, dtype=np.float32)

    Y[0] = c0
    Y[1] = -c1 * y
    Y[2] = c1 * z
    Y[3] = -c1 * x
    Y[4] = c2 * x * y
    Y[5] = -c2 * y * z
    Y[6] = c3 * (3.0 * z * z - 1.0)
    Y[7] = -c2 * x * z
    Y[8] = c4 * (x * x - y * y)

    return Y


def env_from_sh(height, width, sh_coeffs):
    """Create an RGB latitude-longitude environment map from SH coefficients."""
    env = np.zeros((height, width, 3), dtype=np.float32)

    dtheta = math.pi / height
    dphi = 2.0 * math.pi / width

    for row in range(height):
        theta = (row + 0.5) * dtheta
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)

        for col in range(width):
            phi = (col + 0.5) * dphi

            x = sin_theta * math.cos(phi)
            y = sin_theta * math.sin(phi)
            z = cos_theta

            basis = sh_basis_dir_l2(x, y, z)
            env[row, col] = (sh_coeffs.T @ basis).astype(np.float32)

    env -= env.min()

    if env.max() > 0.0:
        env /= env.max()

    return env


def sh_for_global_env(env_id, order=2):
    """Generate deterministic SH coefficients for environment env_id."""
    pairs = sh_lm_list(order)
    coeffs = np.zeros((len(pairs), 3), dtype=np.float32)

    u = env_id / max(1.0, float(NUM_GLOBAL_ENVS - 1))
    t = 2.0 * math.pi * u

    r = 0.5 + 0.4 * math.sin(t)
    g = 0.5 + 0.4 * math.sin(t + 2.0 * math.pi / 3.0)
    b = 0.5 + 0.4 * math.sin(t + 4.0 * math.pi / 3.0)

    rgb = np.array([r, g, b], dtype=np.float32)
    gray = np.full(3, rgb.mean(), dtype=np.float32)

    if u < 1.0 / 3.0:
        saturation = 0.1
    elif u < 2.0 / 3.0:
        saturation = 0.5
    else:
        saturation = 1.0

    rgb_scale = (1.0 - saturation) * gray + saturation * rgb

    coeffs[0] = rgb_scale * 0.4

    for idx, (l, m) in enumerate(pairs):
        if l == 1:
            if m == -1:
                coeffs[idx] = rgb_scale * (
                    0.2 * math.sin(2.0 * math.pi * u)
                )
            elif m == 0:
                coeffs[idx] = rgb_scale * (
                    0.2 * math.cos(2.0 * math.pi * u)
                )
            elif m == 1:
                coeffs[idx] = rgb_scale * (
                    0.2 * math.sin(2.0 * math.pi * u + 1.0)
                )

        elif l == 2 and m == 0:
            coeffs[idx] += rgb_scale * (
                0.05 * math.cos(4.0 * math.pi * u)
            )

    return coeffs


def save_envmap(env, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    env_u8 = np.clip(env, 0.0, 1.0)
    env_u8 = (env_u8 * 255.0).astype(np.uint8)

    imageio.imwrite(filepath, env_u8)


def set_env_texture(scene, image_path, strength=1.0):
    """Set Blender world to an environment texture."""
    world = scene.world

    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world

    world.use_nodes = True

    nodes = world.node_tree.nodes
    links = world.node_tree.links

    for node in list(nodes):
        nodes.remove(node)

    output_node = nodes.new("ShaderNodeOutputWorld")
    output_node.location = (300, 0)

    background = nodes.new("ShaderNodeBackground")
    background.location = (0, 0)
    background.inputs["Strength"].default_value = float(strength)

    environment = nodes.new("ShaderNodeTexEnvironment")
    environment.location = (-300, 0)

    image = bpy.data.images.load(
        str(image_path),
        check_existing=True,
    )
    environment.image = image

    links.new(environment.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output_node.inputs["Surface"])


# ============================================================
# DATASET SELECTION
# ============================================================

def select_geometry_rows(all_rows, target_num_geometries, seed):
    """
    Select geometry IDs, retaining every sigma volume for each selected geometry.

    This expects the sharded volume generator metadata to include:

        geometry_id
        sample_id
        sigma
    """
    groups = {}

    for row in all_rows:
        geometry_id = int(row["geometry_id"])
        groups.setdefault(geometry_id, []).append(row)

    geometry_ids = sorted(groups.keys())

    if target_num_geometries is None:
        selected_geometry_ids = geometry_ids
    else:
        target_num_geometries = min(
            int(target_num_geometries),
            len(geometry_ids),
        )

        rng = np.random.RandomState(seed)

        selected_indices = rng.choice(
            len(geometry_ids),
            size=target_num_geometries,
            replace=False,
        )

        selected_geometry_ids = [
            geometry_ids[i]
            for i in sorted(selected_indices)
        ]

    selected_rows = []

    for geometry_id in selected_geometry_ids:
        geometry_rows = groups[geometry_id]

        # Stable sigma/sample ordering.
        geometry_rows = sorted(
            geometry_rows,
            key=lambda row: (
                float(row["sigma"]),
                int(row["sample_id"]),
            ),
        )

        selected_rows.extend(geometry_rows)

    selected_rows.sort(key=lambda row: int(row["sample_id"]))

    return selected_rows, len(selected_geometry_ids)


# ============================================================
# RENDER / SHARD HELPERS
# ============================================================

def render_rgba(scene, filepath):
    """Render a PNG, read it as float32 RGBA, then delete temp file."""
    scene.render.filepath = str(filepath)
    bpy.ops.render.render(write_still=True)

    image = imageio.imread(filepath).astype(np.float32) / 255.0

    if filepath.is_file():
        filepath.unlink()

    return image


def render_surface_mask(
    scene,
    mesh_obj,
    volume_obj,
    mask_material,
    temp_dir,
    job_id,
    sample_id,
    view_idx,
):
    """Render opaque mesh silhouette and return alpha [H, W]."""
    mesh_obj.data.materials.clear()
    mesh_obj.data.materials.append(mask_material)

    mesh_obj.hide_render = False

    if volume_obj is not None:
        volume_obj.hide_render = True

    set_black_world(scene)

    mask_path = temp_dir / (
        f"mask_{job_id}_{os.getpid()}_"
        f"s{sample_id:06d}_v{view_idx:03d}.png"
    )

    image = render_rgba(scene, mask_path)

    mesh_obj.data.materials.clear()

    if image.ndim != 3 or image.shape[2] < 4:
        print(
            f"[WARN] Mask failure: sample_id={sample_id}, "
            f"view={view_idx}, shape={image.shape}"
        )
        return None

    return image[:, :, 3]


def save_shard(
    rgb_array,
    alpha_array,
    rgb_rows,
    alpha_rows,
    count,
    job_id,
    local_shard_id,
):
    """Save one RGB shard, one alpha shard, and their metadata CSVs."""
    if count <= 0:
        return

    shard_tag = f"{job_id}_shard_{local_shard_id:04d}"

    rgb_path = RENDER_DIR / (
        f"images_{RES_X}x{RES_Y}_{shard_tag}.npy"
    )
    np.save(rgb_path, rgb_array[:count])

    rgb_csv = RENDER_DIR / f"metadata_{shard_tag}.csv"
    pd.DataFrame(rgb_rows).to_csv(rgb_csv, index=False)

    alpha_path = ALPHA_DIR / (
        f"alpha_{RES_X}x{RES_Y}_{shard_tag}.npy"
    )
    np.save(alpha_path, alpha_array[:count])

    alpha_csv = ALPHA_DIR / f"metadata_alpha_{shard_tag}.csv"
    pd.DataFrame(alpha_rows).to_csv(alpha_csv, index=False)

    print(f"Saved RGB shard: {rgb_path}")
    print(f"Saved RGB metadata: {rgb_csv}")
    print(f"Saved alpha shard: {alpha_path}")
    print(f"Saved alpha metadata: {alpha_csv}")


# ============================================================
# MAIN
# ============================================================

def main():
    # Slurm array settings.
    task_id = 0
    num_tasks = 1
    job_id = "job0"

    # Optional test override.
    max_geometries = None

    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []

    if "--task_id" in argv:
        task_id = int(argv[argv.index("--task_id") + 1])

    if "--num_tasks" in argv:
        num_tasks = int(argv[argv.index("--num_tasks") + 1])

    if "--job_id" in argv:
        job_id = argv[argv.index("--job_id") + 1]

    if "--max_geometries" in argv:
        max_geometries = int(argv[argv.index("--max_geometries") + 1])

    if num_tasks <= 0:
        raise ValueError(f"num_tasks must be positive, got {num_tasks}")

    if not (0 <= task_id < num_tasks):
        raise ValueError(
            f"task_id must be in [0, {num_tasks - 1}], got {task_id}"
        )

    if not VOLUME_METADATA_CSV.is_file():
        raise FileNotFoundError(
            f"Metadata CSV not found: {VOLUME_METADATA_CSV}"
        )

    temp_dir = RENDER_DIR / f"_tmp_{job_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Blender setup
    # --------------------------------------------------------

    scene = bpy.context.scene
    clean_scene()

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = SAMPLES

    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.resolution_percentage = 100

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"

    # Needed for alpha channels.
    scene.render.film_transparent = True

    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    scene.cycles.transparent_max_bounces = 8
    scene.cycles.volume_step_rate = 0.5
    scene.cycles.volume_max_steps = 1024

    set_black_world(scene)

    # Reusable materials prevent per-frame Blender datablock growth.
    surface_material = make_surface_material()
    volume_material = make_volume_material()
    mask_material = make_mask_material()

    # --------------------------------------------------------
    # Environment maps
    # --------------------------------------------------------

    global_env_sh = {}
    global_env_path = {}

    for env_id in range(NUM_GLOBAL_ENVS):
        sh_coeffs = sh_for_global_env(env_id, order=SH_ORDER)
        env = env_from_sh(ENV_H, ENV_W, sh_coeffs)

        env_path = ENV_DIR / f"env_e{env_id:03d}.png"

        if not env_path.is_file():
            save_envmap(env, env_path)

        global_env_sh[env_id] = sh_coeffs
        global_env_path[env_id] = env_path

    # --------------------------------------------------------
    # Read and select geometry groups
    # --------------------------------------------------------

    with open(VOLUME_METADATA_CSV, "r", newline="") as f:
        all_rows = list(csv.DictReader(f))

    required_fields = {"sample_id", "geometry_id", "mesh_path", "sigma"}
    missing_fields = required_fields - set(all_rows[0].keys())

    if missing_fields:
        raise KeyError(
            "Metadata is missing required columns: "
            f"{sorted(missing_fields)}"
        )

    requested_geometries = (
        max_geometries
        if max_geometries is not None
        else TARGET_NUM_GEOMETRIES
    )

    selected_rows, selected_geometry_count = select_geometry_rows(
        all_rows=all_rows,
        target_num_geometries=requested_geometries,
        seed=GEOMETRY_SELECTION_SEED,
    )

    # Evenly distribute rows over Slurm tasks.
    #
    # Does not assume sample IDs are contiguous.
    task_rows = selected_rows[task_id::num_tasks]

    num_modes = int(RENDER_SURFACE) + int(RENDER_VOLUME)

    print("=" * 72)
    print("Balanced implicit B-spline renderer")
    print(f"Selected geometries: {selected_geometry_count:,}")
    print(f"Selected volume samples: {len(selected_rows):,}")
    print(f"Task ID: {task_id}/{num_tasks - 1}")
    print(f"Volumes assigned to task: {len(task_rows):,}")
    print(f"Views per volume: {NUM_VIEWS_PER_VOLUME}")
    print(f"Render modes per view: {num_modes}")
    print(
        f"Expected images for task: "
        f"{len(task_rows) * NUM_VIEWS_PER_VOLUME * num_modes:,}"
    )
    print("=" * 72)

    if not task_rows:
        print("No samples assigned to this task.")
        return

    # --------------------------------------------------------
    # Output buffers
    # --------------------------------------------------------

    rgb_shard = np.empty(
        (SHARD_SIZE, 3, RES_Y, RES_X),
        dtype=np.float32,
    )

    alpha_shard = np.empty(
        (SHARD_SIZE, RES_Y, RES_X),
        dtype=np.float32,
    )

    rgb_metadata_rows = []
    alpha_metadata_rows = []

    current_shard_count = 0
    local_shard_id = 0

    sh_pairs = sh_lm_list(SH_ORDER)
    base_directions = fibonacci_sphere_directions(NUM_VIEWS_PER_VOLUME)

    # --------------------------------------------------------
    # Frame writing helper
    # --------------------------------------------------------

    def add_frame(
        rgb,
        alpha,
        metadata_row,
        alpha_metadata_row,
    ):
        nonlocal current_shard_count
        nonlocal local_shard_id
        nonlocal rgb_metadata_rows
        nonlocal alpha_metadata_rows

        if rgb.shape != (RES_Y, RES_X, 3):
            raise ValueError(f"Unexpected RGB shape: {rgb.shape}")

        if alpha.shape != (RES_Y, RES_X):
            raise ValueError(f"Unexpected alpha shape: {alpha.shape}")

        rgb_shard[current_shard_count] = np.transpose(rgb, (2, 0, 1))
        alpha_shard[current_shard_count] = alpha

        metadata_row["shard_id"] = f"{job_id}_{local_shard_id}"
        metadata_row["idx_in_shard"] = current_shard_count

        alpha_metadata_row["alpha_shard_id"] = (
            f"{job_id}_{local_shard_id}"
        )
        alpha_metadata_row["idx_in_alpha_shard"] = current_shard_count
        alpha_metadata_row["img_shard_id"] = f"{job_id}_{local_shard_id}"
        alpha_metadata_row["idx_in_img_shard"] = current_shard_count

        rgb_metadata_rows.append(metadata_row)
        alpha_metadata_rows.append(alpha_metadata_row)

        current_shard_count += 1

        if current_shard_count >= SHARD_SIZE:
            save_shard(
                rgb_array=rgb_shard,
                alpha_array=alpha_shard,
                rgb_rows=rgb_metadata_rows,
                alpha_rows=alpha_metadata_rows,
                count=current_shard_count,
                job_id=job_id,
                local_shard_id=local_shard_id,
            )

            local_shard_id += 1
            current_shard_count = 0
            rgb_metadata_rows = []
            alpha_metadata_rows = []

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    for row_index, row in enumerate(task_rows, start=1):
        sample_id = int(row["sample_id"])
        geometry_id = int(row["geometry_id"])

        mesh_path = row["mesh_path"]
        sigma = float(row["sigma"])

        clear_shape_objects()

        # ---- Load mesh ----
        try:
            mesh_obj, mesh_center = load_mesh_object(
                sample_id=sample_id,
                mesh_path_from_csv=mesh_path,
            )
        except Exception as exc:
            print(
                f"[WARN] sample_id={sample_id}: mesh loading failed: {exc}"
            )
            continue

        # ---- Load VDB ----
        try:
            volume_obj = load_vdb_volume(sample_id)

            # Mesh was centered by subtracting mesh_center.
            # Apply identical translation to VDB.
            volume_obj.location = (
                -float(mesh_center[0]),
                -float(mesh_center[1]),
                -float(mesh_center[2]),
            )

        except Exception as exc:
            print(
                f"[WARN] sample_id={sample_id}: VDB loading failed: {exc}"
            )
            volume_obj = None

        # ---- Camera target ----
        origin = bpy.data.objects.new(
            f"origin_{sample_id:06d}",
            None,
        )
        origin.location = (0.0, 0.0, 0.0)
        scene.collection.objects.link(origin)

        camera = create_camera(scene, origin)

        # Reproducibly rotate the 16-view pattern per volume sample.
        view_rng = np.random.RandomState(sample_id + 10_000_019)
        rotation = random_rotation_matrix(view_rng)

        rotated_directions = [
            rotation @ direction
            for direction in base_directions
        ]

        for view_idx, direction in enumerate(rotated_directions):
            set_camera_from_direction(
                camera,
                direction=direction,
                radius=CAMERA_RADIUS,
            )

            phi, theta = direction_to_phi_theta(direction)

            # Deterministic but distinct appearance sampling per sample/view.
            appearance_rng = np.random.RandomState(
                sample_id * 100_003 + view_idx * 101
            )

            # ====================================================
            # SURFACE MODE
            # ====================================================

            if RENDER_SURFACE:
                base_rgb = appearance_rng.uniform(0.02, 1.0, size=3)

                metallic = float(appearance_rng.choice([0.0, 1.0]))
                roughness = float(appearance_rng.uniform(0.1, 0.9))
                specular = 0.5
                opacity = float(
                    appearance_rng.uniform(MIN_OPACITY, MAX_OPACITY)
                )

                env_id = int(
                    appearance_rng.randint(0, NUM_GLOBAL_ENVS)
                )
                sh_coeffs = global_env_sh[env_id]
                env_path = global_env_path[env_id]

                mask_alpha = render_surface_mask(
                    scene=scene,
                    mesh_obj=mesh_obj,
                    volume_obj=volume_obj,
                    mask_material=mask_material,
                    temp_dir=temp_dir,
                    job_id=job_id,
                    sample_id=sample_id,
                    view_idx=view_idx,
                )

                set_env_texture(scene, env_path, strength=1.0)

                base_color, material_type = update_surface_material(
                    material=surface_material,
                    rgb=base_rgb,
                    metallic=metallic,
                    roughness=roughness,
                    specular=specular,
                    opacity=opacity,
                )

                mesh_obj.data.materials.clear()
                mesh_obj.data.materials.append(surface_material)
                mesh_obj.hide_render = False

                if volume_obj is not None:
                    volume_obj.hide_render = True

                image_path = temp_dir / (
                    f"surface_{job_id}_{os.getpid()}_"
                    f"s{sample_id:06d}_v{view_idx:03d}.png"
                )

                image = render_rgba(scene, image_path)

                mesh_obj.data.materials.clear()

                if (
                    image.ndim == 3
                    and image.shape == (RES_Y, RES_X, 4)
                ):
                    rgb = image[:, :, :3]

                    if mask_alpha is not None:
                        alpha = (mask_alpha * opacity).astype(np.float32)
                    else:
                        alpha = image[:, :, 3]

                    metadata_row = {
                        "sample_id": sample_id,
                        "geometry_id": geometry_id,
                        "mesh_path": mesh_path,
                        "sigma": sigma,
                        "render_mode": "surface",
                        "view_idx": view_idx,
                        "env_id": env_id,
                        "env_path": str(env_path),
                        "metallic": metallic,
                        "roughness": roughness,
                        "specular": specular,
                        "material_type": material_type,
                        "base_color_r": float(base_color[0]),
                        "base_color_g": float(base_color[1]),
                        "base_color_b": float(base_color[2]),
                        "opacity": opacity,
                        "density_scale": np.nan,
                        "phi": phi,
                        "theta": theta,
                        "radius": CAMERA_RADIUS,
                    }

                    alpha_metadata_row = {
                        "sample_id": sample_id,
                        "geometry_id": geometry_id,
                        "mesh_path": mesh_path,
                        "sigma": sigma,
                        "render_mode": "surface",
                        "view_idx": view_idx,
                        "opacity": opacity,
                        "density_scale": np.nan,
                        "phi": phi,
                        "theta": theta,
                        "radius": CAMERA_RADIUS,
                    }

                    for sh_idx, (l, m) in enumerate(sh_pairs):
                        r_coeff, g_coeff, b_coeff = sh_coeffs[sh_idx]

                        metadata_row[f"sh_l{l}_m{m}_r"] = float(r_coeff)
                        metadata_row[f"sh_l{l}_m{m}_g"] = float(g_coeff)
                        metadata_row[f"sh_l{l}_m{m}_b"] = float(b_coeff)

                    add_frame(
                        rgb=rgb,
                        alpha=alpha,
                        metadata_row=metadata_row,
                        alpha_metadata_row=alpha_metadata_row,
                    )

                else:
                    print(
                        f"[WARN] Bad surface image: "
                        f"sample_id={sample_id}, view={view_idx}, "
                        f"shape={image.shape}"
                    )

            # ====================================================
            # VOLUME MODE
            # ====================================================

            if RENDER_VOLUME and volume_obj is not None:
                base_rgb = appearance_rng.uniform(0.02, 1.0, size=3)

                opacity = float(
                    appearance_rng.uniform(MIN_OPACITY, MAX_OPACITY)
                )

                density_scale = density_scale_for_sigma(
                    sigma=sigma,
                    target_opacity=opacity,
                )

                env_id = int(
                    appearance_rng.randint(0, NUM_GLOBAL_ENVS)
                )
                sh_coeffs = global_env_sh[env_id]
                env_path = global_env_path[env_id]

                set_env_texture(scene, env_path, strength=1.0)

                base_color = update_volume_material(
                    material=volume_material,
                    rgb=base_rgb,
                    density_scale=density_scale,
                )

                volume_obj.data.materials.clear()
                volume_obj.data.materials.append(volume_material)

                volume_obj.hide_render = False
                mesh_obj.hide_render = True

                image_path = temp_dir / (
                    f"volume_{job_id}_{os.getpid()}_"
                    f"s{sample_id:06d}_v{view_idx:03d}.png"
                )

                image = render_rgba(scene, image_path)

                volume_obj.data.materials.clear()

                if (
                    image.ndim == 3
                    and image.shape == (RES_Y, RES_X, 4)
                ):
                    rgb = image[:, :, :3]
                    alpha = image[:, :, 3]

                    metadata_row = {
                        "sample_id": sample_id,
                        "geometry_id": geometry_id,
                        "mesh_path": mesh_path,
                        "sigma": sigma,
                        "render_mode": "volume",
                        "view_idx": view_idx,
                        "env_id": env_id,
                        "env_path": str(env_path),
                        "metallic": np.nan,
                        "roughness": np.nan,
                        "specular": np.nan,
                        "material_type": "volume",
                        "base_color_r": float(base_color[0]),
                        "base_color_g": float(base_color[1]),
                        "base_color_b": float(base_color[2]),
                        "opacity": opacity,
                        "density_scale": float(density_scale),
                        "phi": phi,
                        "theta": theta,
                        "radius": CAMERA_RADIUS,
                    }

                    alpha_metadata_row = {
                        "sample_id": sample_id,
                        "geometry_id": geometry_id,
                        "mesh_path": mesh_path,
                        "sigma": sigma,
                        "render_mode": "volume",
                        "view_idx": view_idx,
                        "opacity": opacity,
                        "density_scale": float(density_scale),
                        "phi": phi,
                        "theta": theta,
                        "radius": CAMERA_RADIUS,
                    }

                    for sh_idx, (l, m) in enumerate(sh_pairs):
                        r_coeff, g_coeff, b_coeff = sh_coeffs[sh_idx]

                        metadata_row[f"sh_l{l}_m{m}_r"] = float(r_coeff)
                        metadata_row[f"sh_l{l}_m{m}_g"] = float(g_coeff)
                        metadata_row[f"sh_l{l}_m{m}_b"] = float(b_coeff)

                    add_frame(
                        rgb=rgb,
                        alpha=alpha,
                        metadata_row=metadata_row,
                        alpha_metadata_row=alpha_metadata_row,
                    )

                else:
                    print(
                        f"[WARN] Bad volume image: "
                        f"sample_id={sample_id}, view={view_idx}, "
                        f"shape={image.shape}"
                    )

        print(
            f"[{row_index}/{len(task_rows)}] completed "
            f"sample_id={sample_id}, geometry_id={geometry_id}, "
            f"sigma={sigma:.4f}"
        )

    # Save partial final shard.
    if current_shard_count > 0:
        save_shard(
            rgb_array=rgb_shard,
            alpha_array=alpha_shard,
            rgb_rows=rgb_metadata_rows,
            alpha_rows=alpha_metadata_rows,
            count=current_shard_count,
            job_id=job_id,
            local_shard_id=local_shard_id,
        )

    print("Done.")


if __name__ == "__main__":
    main()