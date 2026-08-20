#!/usr/bin/env python
import bpy
import os
import csv
import math
import numpy as np
import colorsys
import sys
from pathlib import Path
import imageio.v2 as imageio
import pandas as pd

# ==============================
# CONFIGURATION
# ==============================
PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DIR = PROJECT_ROOT / "plane_dataset_4"

VOLUME_METADATA_CSV = os.path.join(BASE_DIR, "metadata_volumes.csv")

RENDER_DIR = os.path.join(BASE_DIR, "renders")
os.makedirs(RENDER_DIR, exist_ok=True)

ALPHA_DIR = os.path.join(BASE_DIR, "hard_alpha")
os.makedirs(ALPHA_DIR, exist_ok=True)

# ENV
NUM_GLOBAL_ENVS = 128

# render settings
RES_X = 64
RES_Y = 64
SAMPLES = 1024

# ENV MAP + SH
ENV_DIR = os.path.join(BASE_DIR, "envmaps")
os.makedirs(ENV_DIR, exist_ok=True)

ENV_H = 16
ENV_W = 32
SH_ORDER = 2

IMAGES_PER_SHAPE = 300
SHARD_SIZE = 5000
ALPHA_SHARD_SIZE = SHARD_SIZE

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
    world = bpy.data.worlds.new("World") if scene.world is None else scene.world
    scene.world = world
    world.use_nodes = True
    w_nodes = world.node_tree.nodes
    w_links = world.node_tree.links
    for n in list(w_nodes):
        w_nodes.remove(n)
    world_output = w_nodes.new("ShaderNodeOutputWorld")
    bg = w_nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1)
    bg.inputs["Strength"].default_value = 0.0
    w_links.new(bg.outputs["Background"], world_output.inputs["Surface"])
    for o in list(bpy.data.objects):
        if o.type == 'LIGHT':
            bpy.data.objects.remove(o, do_unlink=True)

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
    cam.location.x = x
    cam.location.y = y
    cam.location.z = z

def make_material_from_params(hue, saturation, metallic, roughness, specular, opacity):
    # surface: Transparent + Principled, mixed by opacity
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, 0.4)
    base_color = (r, g, b, 1.0)

    mat = bpy.data.materials.new(name="SurfMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (400, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 100)
    bsdf.inputs["Base Color"].default_value = base_color
    bsdf.inputs["Metallic"].default_value   = metallic
    bsdf.inputs["Roughness"].default_value  = roughness
    if "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = specular

    transp = nodes.new("ShaderNodeBsdfTransparent")
    transp.location = (0, -50)

    mix = nodes.new("ShaderNodeMixShader")
    mix.location = (200, 50)
    mix.inputs["Fac"].default_value = opacity  # 0=transparent, 1=opaque

    # (1-opacity)*Transparent + opacity*BSDF
    links.new(transp.outputs["BSDF"], mix.inputs[1])
    links.new(bsdf.outputs["BSDF"],   mix.inputs[2])
    links.new(mix.outputs["Shader"],  out.inputs["Surface"])

    material_type = "plastic" if metallic <= 1e-3 else "metal"
    return mat, (r, g, b, opacity), material_type

def make_volume_material_from_params(hue, saturation, density_scale):
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, 0.5)
    color = (r, g, b, 1.0)

    mat = bpy.data.materials.new(name="VolMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (400, 0)

    vol = nodes.new("ShaderNodeVolumePrincipled")
    vol.location = (0, 0)
    vol.inputs["Color"].default_value   = color
    vol.inputs["Density"].default_value = 1.0

    vol_info = nodes.new("ShaderNodeVolumeInfo")
    vol_info.location = (-300, 0)

    mul = nodes.new("ShaderNodeMath")
    mul.operation = 'MULTIPLY'
    mul.inputs[1].default_value = density_scale
    mul.location = (-100, -50)

    links.new(vol_info.outputs["Density"], mul.inputs[0])
    links.new(mul.outputs["Value"], vol.inputs["Density"])
    links.new(vol.outputs["Volume"], out.inputs["Volume"])

    return mat

def make_mask_material():
    """
    Opaque white emission; we only care about alpha coverage.
    """
    mat = bpy.data.materials.new(name="MaskMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (200, 0)

    emit = nodes.new("ShaderNodeEmission")
    emit.location = (0, 0)
    emit.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emit.inputs["Strength"].default_value = 1.0

    links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat

def load_vdb_volume(sample_id, volume_rel_path):
    vol_npy = Path(volume_rel_path)
    if vol_npy.suffix != ".npy":
        raise ValueError(f"Expected .npy in volume_path, got: {volume_rel_path}")
    vol_vdb = vol_npy.with_suffix(".vdb")
    full_vdb = (BASE_DIR / vol_vdb).resolve()
    if not full_vdb.is_file():
        raise FileNotFoundError(f"VDB file not found: {full_vdb}")
    bpy.ops.object.volume_import(filepath=str(full_vdb))
    vol_obj = bpy.context.object
    vol_obj.name = f"vol_{sample_id:04d}"
    return vol_obj

# --- SH helpers ---
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
    Y[6] = c3 * (3.0 * z*z - 1.0)
    Y[7] = -c2 * x * z
    Y[8] = c4 * (x*x - y*y)
    return Y

def env_from_sh(H, W, sh_coeffs):
    H = int(H); W = int(W)
    env = np.zeros((H, W, 3), dtype=np.float32)
    dtheta = math.pi / H
    dphi   = 2.0 * math.pi / W
    for y in range(H):
        theta = (y + 0.5) * dtheta
        sin_theta = math.sin(theta)
        ct = math.cos(theta)
        for x in range(W):
            phi = (x + 0.5) * dphi
            cp = math.cos(phi)
            sp = math.sin(phi)
            vx = sin_theta * cp
            vy = sin_theta * sp
            vz = ct
            Y = sh_basis_dir_l2(vx, vy, vz)
            rgb = (sh_coeffs.T @ Y).astype(np.float32)
            env[y, x, :] = rgb
    env -= env.min()
    if env.max() > 0:
        env /= env.max()
    return env

def sh_for_global_env(env_id, order=2):
    pairs = sh_lm_list(order)
    num_coeffs = len(pairs)
    coeffs = np.zeros((num_coeffs, 3), dtype=np.float32)
    u = env_id / max(1.0, float(NUM_GLOBAL_ENVS - 1))
    t = 2.0 * math.pi * u
    r = 0.5 + 0.4 * math.sin(t)
    g = 0.5 + 0.4 * math.sin(t + 2.0 * math.pi / 3.0)
    b = 0.5 + 0.4 * math.sin(t + 4.0 * math.pi / 3.0)
    rgb = np.array([r, g, b], dtype=np.float32)
    gray = np.full(3, rgb.mean(), dtype=np.float32)
    if u < 1.0/3.0:
        alpha = 0.1
    elif u < 2.0/3.0:
        alpha = 0.5
    else:
        alpha = 1.0
    rgb_scale = (1.0 - alpha) * gray + alpha * rgb
    coeffs[0, :] = rgb_scale * 0.4
    for idx, (l, m) in enumerate(pairs):
        if l == 1:
            if m == -1:
                coeffs[idx, :] = rgb_scale * (0.2 * math.sin(2.0 * math.pi * u))
            elif m == 0:
                coeffs[idx, :] = rgb_scale * (0.2 * math.cos(2.0 * math.pi * u))
            elif m == 1:
                coeffs[idx, :] = rgb_scale * (0.2 * math.sin(2.0 * math.pi * u + 1.0))
    for idx, (l, m) in enumerate(pairs):
        if l == 2 and m == 0:
            coeffs[idx, :] += rgb_scale * (0.05 * math.cos(4.0 * math.pi * u))
    return coeffs

def set_env_texture(scene, image_path, strength=1.0):
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nodes = nt.nodes
    links = nt.links
    for n in list(nodes):
        nodes.remove(n)
    world_output = nodes.new("ShaderNodeOutputWorld")
    world_output.location = (300, 0)
    bg = nodes.new("ShaderNodeBackground")
    bg.location = (0, 0)
    env_tex = nodes.new("ShaderNodeTexEnvironment")
    env_tex.location = (-300, 0)
    img = bpy.data.images.load(image_path)
    env_tex.image = img
    bg.inputs["Strength"].default_value = strength
    links.new(env_tex.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], world_output.inputs["Surface"])

def save_envmap(env, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    env8 = np.clip(env, 0.0, 1.0)
    env8 = (env8 * 255.0).astype(np.uint8)
    imageio.imwrite(filepath, env8)

# ==============================
# MAIN
# ==============================

def main():
    start_id = None
    end_id = None
    job_id = "job0"
    temp_dir = os.path.join(RENDER_DIR, f"_tmp_{job_id}")
    os.makedirs(temp_dir, exist_ok=True)

    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if "--start_id" in argv:
        start_id = int(argv[argv.index("--start_id") + 1])
    if "--end_id" in argv:
        end_id = int(argv[argv.index("--end_id") + 1])
    if "--job_id" in argv:
        job_id = argv[argv.index("--job_id") + 1]

    shard_base = 0
    if "--shard_base" in argv:
        shard_base = int(argv[argv.index("--shard_base") + 1])

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
    scene.cycles.transparent_max_bounces = 8

    scene.cycles.volume_step_rate = 0.5
    scene.cycles.volume_max_steps = 1024

    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.film_transparent = True

    setup_world_and_lighting(scene)

    # Mask material (for silhouette renders in surface mode)
    mask_mat = make_mask_material()

    # Precompute env SH + envmaps
    global_env_sh = {}
    global_env_path = {}
    for env_id in range(NUM_GLOBAL_ENVS):
        sh_coeffs = sh_for_global_env(env_id, order=SH_ORDER)
        env = env_from_sh(ENV_H, ENV_W, sh_coeffs)
        env_name = f"env_e{env_id:03d}.png"
        env_path = os.path.join(ENV_DIR, env_name)
        if not os.path.exists(env_path):
            save_envmap(env, env_path)
        global_env_sh[env_id] = sh_coeffs
        global_env_path[env_id] = env_path

    # read volume metadata
    with open(VOLUME_METADATA_CSV, "r", newline="") as f:
        vol_reader = csv.DictReader(f)
        vol_rows = [row for row in vol_reader]

    if start_id is not None or end_id is not None:
        filtered = []
        for row in vol_rows:
            sid = int(row["sample_id"])
            if (start_id is None or sid >= start_id) and (end_id is None or sid <= end_id):
                filtered.append(row)
        vol_rows = filtered
        print(f"Processing sample_id in [{start_id}, {end_id}], count={len(vol_rows)}")

    H, W = RES_Y, RES_X

    current_shard_id = 0
    current_shard_count = 0
    shard_array = np.empty((SHARD_SIZE, 3, H, W), dtype=np.float32)
    rows = []

    alpha_shard_array = np.empty((ALPHA_SHARD_SIZE, H, W), dtype=np.float32)
    alpha_rows = []

    sh_pairs = sh_lm_list(SH_ORDER)

    for vol_row in vol_rows:
        sample_id   = int(vol_row["sample_id"])
        mesh_path   = vol_row["mesh_path"]
        coeff_path  = vol_row["coeff_path"]
        volume_path = vol_row["volume_path"]

        # remove existing non-light, non-camera objects
        for o in list(bpy.data.objects):
            if o.type not in {'LIGHT', 'CAMERA'}:
                bpy.data.objects.remove(o, do_unlink=True)

        # ---- load mesh (surface mode) ----
        full_mesh_path = os.path.abspath(mesh_path)
        data = np.load(full_mesh_path)
        verts = data["verts"].astype(np.float32)
        faces = data["faces"]

        vmin = verts.min(axis=0)
        vmax = verts.max(axis=0)
        center = 0.5 * (vmin + vmax)
        verts_centered = verts - center

        mesh = bpy.data.meshes.new(f"mesh_{sample_id:04d}")
        mesh.from_pydata(verts_centered.tolist(), [], faces.tolist())
        mesh.update()
        mesh_obj = bpy.data.objects.new(mesh.name, mesh)
        scene.collection.objects.link(mesh_obj)
        mesh_obj.name = f"shape_mesh_{sample_id:04d}"

        bpy.context.view_layer.objects.active = mesh_obj
        mesh_obj.select_set(True)
        bpy.ops.object.shade_smooth()
        mesh_obj.select_set(False)

        # ---- load VDB volume (volume mode) ----
        try:
            vol_obj = load_vdb_volume(sample_id, volume_path)
            vol_obj.location = (-center[0], -center[1], -center[2])
        except FileNotFoundError as e:
            print("[WARN]", e)
            vol_obj = None

        # ---- create an empty at origin and a camera that tracks it ----
        origin_empty = bpy.data.objects.new(f"origin_{sample_id:04d}", None)
        origin_empty.location = (0.0, 0.0, 0.0)
        scene.collection.objects.link(origin_empty)

        for o in list(bpy.data.objects):
            if o.type == 'CAMERA':
                bpy.data.objects.remove(o, do_unlink=True)
        cam = create_camera(scene, origin_empty)

        rng = np.random.RandomState(sample_id)

        for img_idx in range(IMAGES_PER_SHAPE):
            # 3) camera pose (set first so both mask & main use same pose)
            radius = 2.2
            phi    = float(rng.uniform(0.0, math.pi))
            theta  = float(rng.uniform(0.0, 2.0 * math.pi))
            set_camera_from_spherical(cam, radius, phi, theta)

            # 2) material + mode
            hue        = float(rng.uniform(0.0, 1.0))
            saturation = float(rng.uniform(0.3, 0.9))
            metallic   = float(rng.choice([0.0, 1.0]))
            roughness  = float(rng.uniform(0.1, 0.9))
            specular   = 0.5
            opacity    = float(rng.uniform(0.1, 1.0))

            if vol_obj is not None:
                render_mode = rng.choice(["surface", "volume"])
            else:
                render_mode = "surface"

            # ---- mask render for surface mode ----
            mask_alpha = None
            if render_mode == "surface":
                # mesh only, mask material
                mesh_obj.data.materials.clear()
                mesh_obj.data.materials.append(mask_mat)
                mesh_obj.hide_render = False
                if vol_obj is not None:
                    vol_obj.hide_render = True

                # simple black world for mask
                setup_world_and_lighting(scene)

                pid = os.getpid()
                tmp_mask_name = f"_tmpmask_{job_id}_{pid}_s{sample_id:04d}_i{img_idx:05d}.png"
                tmp_mask_path = os.path.join(temp_dir, tmp_mask_name)

                scene.render.filepath = tmp_mask_path
                scene.render.use_file_extension = True
                bpy.ops.render.render(write_still=True)

                mask_img = imageio.imread(tmp_mask_path).astype(np.float32) / 255.0
                os.remove(tmp_mask_path)

                if mask_img.ndim == 3 and mask_img.shape[2] >= 4:
                    mask_alpha = mask_img[:, :, 3]  # [H,W]
                else:
                    print(f"[WARN] Mask render missing alpha, shape={mask_img.shape}")
                    mask_alpha = None

            # 1) env (for main render)
            env_id = int(rng.randint(0, NUM_GLOBAL_ENVS))
            sh_coeffs = global_env_sh[env_id]
            env_path  = global_env_path[env_id]
            set_env_texture(scene, env_path, strength=1.0)

            # assign real materials for main render
            if render_mode == "surface":
                mat, base_color, material_type = make_material_from_params(
                    hue, saturation, metallic, roughness, specular, opacity
                )
                mesh_obj.data.materials.clear()
                mesh_obj.data.materials.append(mat)
                mesh_obj.hide_render = False
                if vol_obj is not None:
                    vol_obj.hide_render = True
            else:  # volume
                density_scale = 0.05 + 0.95 * opacity
                mat = make_volume_material_from_params(hue, saturation, density_scale)
                if vol_obj is not None:
                    vol_obj.data.materials.clear()
                    vol_obj.data.materials.append(mat)
                    vol_obj.hide_render = False
                material_type = "volume"
                base_color = (0.0, 0.0, 0.0, 1.0)
                mesh_obj.hide_render = True

            # 4) main render to temp PNG
            pid = os.getpid()
            tmp_name = f"_tmp_{job_id}_{pid}_s{sample_id:04d}_i{img_idx:05d}.png"
            tmp_path = os.path.join(temp_dir, tmp_name)

            scene.render.filepath = tmp_path
            scene.render.use_file_extension = True
            bpy.ops.render.render(write_still=True)

            img = imageio.imread(tmp_path).astype(np.float32) / 255.0
            os.remove(tmp_path)

            if img.ndim != 3 or img.shape[2] < 4:
                print(f"[WARN] Expected RGBA, got shape {img.shape}; skipping frame")
                continue

            rgb   = img[:, :, :3]
            alpha = img[:, :, 3]

            # replace alpha with mask * opacity in surface mode
            if render_mode == "surface" and mask_alpha is not None:
                alpha = (mask_alpha * opacity).astype(np.float32)

            h, w, _ = rgb.shape
            if h != RES_Y or w != RES_X:
                print(f"[WARN] Got render size ({w},{h}), expected ({RES_X},{RES_Y}); skipping frame")
                continue

            img_np = np.transpose(rgb, (2, 0, 1))
            shard_array[current_shard_count] = img_np
            alpha_shard_array[current_shard_count] = alpha

            row = {
                "sample_id": sample_id,
                "env_id": env_id,
                "coeff_path": coeff_path,
                "mesh_path": mesh_path,
                "hue": float(hue),
                "saturation": float(saturation),
                "metallic": float(metallic),
                "roughness": float(roughness),
                "specular": float(specular),
                "material_type": material_type,
                "base_color_r": float(base_color[0]),
                "base_color_g": float(base_color[1]),
                "base_color_b": float(base_color[2]),
                "opacity": float(opacity),
                "phi": float(phi),
                "theta": float(theta),
                "radius": float(radius),
                "env_path": env_path,
                "shard_id": f"{job_id}_{shard_base + current_shard_id}",
                "idx_in_shard": current_shard_count,
                "render_mode": render_mode,
            }
            for idx_sh, (l, m) in enumerate(sh_pairs):
                r_c, g_c, b_c = sh_coeffs[idx_sh]
                row[f"sh_l{l}_m{m}_r"] = float(r_c)
                row[f"sh_l{l}_m{m}_g"] = float(g_c)
                row[f"sh_l{l}_m{m}_b"] = float(b_c)

            rows.append(row)

            alpha_row = {
                "sample_id": sample_id,
                "env_id": env_id,
                "coeff_path": coeff_path,
                "volume_path": volume_path,
                "mesh_path": mesh_path,
                "phi": float(phi),
                "theta": float(theta),
                "radius": float(radius),
                "render_mode": render_mode,
                "alpha_shard_id": f"{job_id}_{shard_base + current_shard_id}",
                "idx_in_alpha_shard": current_shard_count,
                "img_shard_id": f"{job_id}_{shard_base + current_shard_id}",
                "idx_in_img_shard": current_shard_count,
            }
            alpha_rows.append(alpha_row)

            current_shard_count += 1

            if current_shard_count == SHARD_SIZE:
                shard_name = f"images_64x64_{job_id}_shard_{shard_base + current_shard_id:03d}.npy"
                shard_path = Path(RENDER_DIR) / shard_name
                np.save(shard_path, shard_array[:current_shard_count])
                print("Saved RGB shard:", shard_path)

                rgb_csv_path = Path(RENDER_DIR) / f"metadata_{job_id}_shard_{shard_base + current_shard_id:03d}.csv"
                pd.DataFrame(rows).to_csv(rgb_csv_path, index=False)
                print("Saved RGB metadata:", rgb_csv_path)

                alpha_shard_name = f"alpha_64x64_{job_id}_shard_{shard_base + current_shard_id:03d}.npy"
                alpha_shard_path = Path(ALPHA_DIR) / alpha_shard_name
                np.save(alpha_shard_path, alpha_shard_array[:current_shard_count])
                print("Saved alpha shard:", alpha_shard_path)

                alpha_csv_path = Path(ALPHA_DIR) / f"metadata_alpha_{job_id}_shard_{shard_base + current_shard_id:03d}.csv"
                pd.DataFrame(alpha_rows).to_csv(alpha_csv_path, index=False)
                print("Saved alpha metadata:", alpha_csv_path)

                current_shard_id += 1
                current_shard_count = 0
                rows = []
                alpha_rows = []

        print("Finished shape", sample_id)

    if current_shard_count > 0:
        shard_name = f"images_64x64_{job_id}_shard_{shard_base + current_shard_id:03d}.npy"
        shard_path = Path(RENDER_DIR) / shard_name
        np.save(shard_path, shard_array[:current_shard_count])
        print("Saved final RGB shard:", shard_path)

        rgb_csv_path = Path(RENDER_DIR) / f"metadata_{job_id}_shard_{shard_base + current_shard_id:03d}.csv"
        pd.DataFrame(rows).to_csv(rgb_csv_path, index=False)
        print("Saved RGB metadata:", rgb_csv_path)

        alpha_shard_name = f"alpha_64x64_{job_id}_shard_{shard_base + current_shard_id:03d}.npy"
        alpha_shard_path = Path(ALPHA_DIR) / alpha_shard_name
        np.save(alpha_shard_path, alpha_shard_array[:current_shard_count])
        print("Saved final alpha shard:", alpha_shard_path)

        alpha_csv_path = Path(ALPHA_DIR) / f"metadata_alpha_{job_id}_shard_{shard_base + current_shard_id:03d}.csv"
        pd.DataFrame(alpha_rows).to_csv(alpha_csv_path, index=False)
        print("Saved alpha metadata:", alpha_csv_path)

    print("Done.")

if __name__ == "__main__":
    main()