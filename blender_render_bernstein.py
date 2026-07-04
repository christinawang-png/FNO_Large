import bpy
import os
import csv
import math
import numpy as np
import colorsys
import sys
from pathlib import Path
import imageio  # pip install imageio in Blender's Python if you don't have it



# ==============================
# CONFIGURATION
# ==============================
# Directory where this Python file lives
PROJECT_ROOT = Path(__file__).resolve().parent

# Base directory created by generate_bernstein_meshes.py
BASE_DIR = PROJECT_ROOT / "plane_dataset_1"

VOLUME_METADATA_CSV = os.path.join(BASE_DIR, "metadata_volumes.csv")

# Where to write rendered images and image metadata
RENDER_DIR = os.path.join(BASE_DIR, "renders")
os.makedirs(RENDER_DIR, exist_ok=True)

# camera sphere params
RADIUS = 1.5
NUM_PHI = 6
NUM_THETA = 12

# material param ranges (continuous color via hue)
HUE_VALUES       = [0.0]                  # fixed color
METALLIC_VALUES  = [0.0, 1.0]             # plastic vs metal
ROUGHNESS_VALUES = [0.1, 0.5, 0.9]   # main variation
SPECULAR_VALUES  = [0.5]                  # fixed

# render settings
RES_X = 128
RES_Y = 128
SAMPLES = 64

# ==============================
# ENVIRONMENT MAP + SH CONFIG
# ==============================

ENV_DIR = os.path.join(BASE_DIR, "envmaps")
os.makedirs(ENV_DIR, exist_ok=True)

ENV_H = 16   # envmap height (equirectangular)
ENV_W = 32   # envmap width  (equirectangular)
SH_ORDER = 2  # spherical harmonic order (2 -> 9 coeffs)


# ==============
# ENV SAMPLING
# ==============

NUM_GLOBAL_ENVS = 128        # total distinct environments
NUM_ENVS_PER_SHAPE = NUM_GLOBAL_ENVS       # envs used per shape

# ==============================
# UTILITIES
# ==============================

def clean_scene():
    # delete all objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # clean meshes & materials & lights
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
    # Basic dark world; we will override with an Environment Texture per image
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

    # remove lights so only envmap contributes
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

def make_material_from_params(hue, metallic, roughness, specular):
    r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.4)
    base_color = (r, g, b, 1.0)

    mat_name = f"Mat_h{hue:.2f}_m{metallic:.2f}_r{roughness:.2f}_s{specular:.2f}"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (300, 0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)

    bsdf.inputs["Base Color"].default_value = base_color
    bsdf.inputs["Metallic"].default_value   = metallic
    bsdf.inputs["Roughness"].default_value  = roughness

    # Set specular-like control depending on available socket
    if "Specular" in bsdf.inputs:
        # older Principled BSDF
        bsdf.inputs["Specular"].default_value = specular
    elif "specular_ior_level" in bsdf.inputs:
        # Principled v2 socket name in Blender 5.1
        bsdf.inputs["specular_ior_level"].default_value = specular
    # else: leave default, but still record 'specular' in metadata

    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    material_type = "plastic" if metallic <= 1e-3 else "metal"

    return mat, base_color, material_type

def set_env_texture(scene, image_path, strength=1.0):
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nodes = nt.nodes
    links = nt.links

    # clear existing nodes
    for n in list(nodes):
        nodes.remove(n)

    # create nodes: Environment Texture -> Background -> World Output
    world_output = nodes.new("ShaderNodeOutputWorld")
    world_output.location = (300, 0)

    bg = nodes.new("ShaderNodeBackground")
    bg.location = (0, 0)

    env_tex = nodes.new("ShaderNodeTexEnvironment")
    env_tex.location = (-300, 0)

    # load image
    img = bpy.data.images.load(image_path)
    env_tex.image = img

    bg.inputs["Strength"].default_value = strength

    links.new(env_tex.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], world_output.inputs["Surface"])

def generate_random_envmap(H, W):
    """
    Generate a simple random environment map for testing.
    Shape: (H, W, 3), float32 in [0, 1].
    You can replace this with something more structured later.
    """
    # Low-frequency random by upsampling a tiny random grid
    small_h, small_w = 4, 8
    base = np.random.rand(small_h, small_w, 3).astype(np.float32)

    # Nearest-neighbor upscale to (H, W)
    ys = (np.linspace(0, small_h - 1, H)).astype(np.int32)
    xs = (np.linspace(0, small_w - 1, W)).astype(np.int32)
    env = base[ys[:, None], xs[None, :], :]  # (H, W, 3)
    return env


def save_envmap(env, filepath):
    """
    env: (H, W, 3) float32, [0,1]
    Saves as 8‑bit PNG.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    env8 = np.clip(env, 0.0, 1.0)
    env8 = (env8 * 255.0).astype(np.uint8)  # (H, W, 3), uint8
    imageio.imwrite(filepath, env8)


def sh_lm_list(order):
    """
    List (l, m) pairs in a fixed order up to given order.
    For order=2: (0,0),
                 (1,-1),(1,0),(1,1),
                 (2,-2),(2,-1),(2,0),(2,1),(2,2)
    """
    pairs = []
    for l in range(order + 1):
        for m in range(-l, l + 1):
            pairs.append((l, m))
    return pairs


def sh_basis_dir_l2(x, y, z):
    """
    Real SH basis up to l=2 evaluated at direction (x, y, z), ||dir||=1.
    Returns 9 coefficients in the (l,m) order from sh_lm_list(2).
    """
    # constants for real SH (see e.g. "Stupid Spherical Harmonics (SH) Tricks" notes)
    # Y00
    c0 = 0.28209479177387814

    # l=1
    c1 = 0.4886025119029199
    # l=2
    c2 = 1.0925484305920792
    c3 = 0.31539156525252005
    c4 = 0.5462742152960396

    Y = np.empty(9, dtype=np.float32)
    # l=0, m=0
    Y[0] = c0
    # l=1
    Y[1] = -c1 * y          # (1,-1)
    Y[2] = c1 * z           # (1, 0)
    Y[3] = -c1 * x          # (1, 1)
    # l=2
    Y[4] = c2 * x * y       # (2,-2)
    Y[5] = -c2 * y * z      # (2,-1)
    Y[6] = c3 * (3.0 * z*z - 1.0)  # (2, 0)
    Y[7] = -c2 * x * z      # (2, 1)
    Y[8] = c4 * (x*x - y*y) # (2, 2)
    return Y


def env_from_sh(H, W, sh_coeffs):
    """
    Reconstruct envmap from SH.
    sh_coeffs: (num_coeffs, 3) for RGB, in the order of sh_lm_list(SH_ORDER).
    Returns env: (H, W, 3), float32 in [0, +inf) (you can later normalize/clamp).
    """
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

            Y = sh_basis_dir_l2(vx, vy, vz)  # (9,)
            # sum_lm c_lm * Y_lm
            rgb = (sh_coeffs.T @ Y).astype(np.float32)  # (3,)
            env[y, x, :] = rgb

    # Simple normalization to [0,1] for testing
    env -= env.min()
    if env.max() > 0:
        env /= env.max()
    return env


def sh_for_global_env(env_id, order=2):
    pairs = sh_lm_list(order)
    num_coeffs = len(pairs)
    coeffs = np.zeros((num_coeffs, 3), dtype=np.float32)

    u = env_id / max(1.0, float(NUM_GLOBAL_ENVS - 1))  # 0..1
    t = 2.0 * math.pi * u

    # base color wheel (strong tint)
    r = 0.5 + 0.4 * math.sin(t)
    g = 0.5 + 0.4 * math.sin(t + 2.0 * math.pi / 3.0)
    b = 0.5 + 0.4 * math.sin(t + 4.0 * math.pi / 3.0)
    rgb = np.array([r, g, b], dtype=np.float32)

    # compute gray level and interpolate towards it for part of u
    gray = np.full(3, rgb.mean(), dtype=np.float32)

    # e.g. first 1/3 of env_ids almost white/gray, last 2/3 more colorful
    if u < 1.0/3.0:
        alpha = 0.1   # very low saturation
    elif u < 2.0/3.0:
        alpha = 0.5   # medium saturation
    else:
        alpha = 1.0   # full tint

    rgb_scale = (1.0 - alpha) * gray + alpha * rgb

    # ambient
    coeffs[0, :] = rgb_scale * 0.4
    
    # First-order band: loop directionally with u
    for idx, (l, m) in enumerate(pairs):
        if l == 1:
            if m == -1:    # Y-ish
                coeffs[idx, :] = rgb_scale * (0.2 * math.sin(2.0 * math.pi * u))
            elif m == 0:   # Z-ish
                coeffs[idx, :] = rgb_scale * (0.2 * math.cos(2.0 * math.pi * u))
            elif m == 1:   # X-ish
                coeffs[idx, :] = rgb_scale * (0.2 * math.sin(2.0 * math.pi * u + 1.0))

    # Small l=2 term for extra variation
    for idx, (l, m) in enumerate(pairs):
        if l == 2 and m == 0:
            coeffs[idx, :] += rgb_scale * (0.05 * math.cos(4.0 * math.pi * u))

    return coeffs


def env_ids_for_shape(sample_id):
    """
    Deterministically pick NUM_ENVS_PER_SHAPE env_ids for this sample_id
    from 0..NUM_GLOBAL_ENVS-1, spread along the snake.
    """
    base = (sample_id * 7) % NUM_GLOBAL_ENVS
    step = max(1, NUM_GLOBAL_ENVS // NUM_ENVS_PER_SHAPE)
    ids = [ (base + k * step) % NUM_GLOBAL_ENVS for k in range(NUM_ENVS_PER_SHAPE) ]
    return ids


def project_env_to_sh(env, order=3):
    """
    Project an equirectangular envmap to SH coefficients.
    env: (H, W, 3), float32 in [0,1], representing radiance over directions.
         v in [0,1] -> theta in [0, pi]
         u in [0,1] -> phi in [0, 2*pi]
    Returns coeffs: (num_coeffs, 3) for RGB.
    """
    H, W, _ = env.shape
    pairs = sh_lm_list(order)
    num_coeffs = len(pairs)

    coeffs = np.zeros((num_coeffs, 3), dtype=np.float64)

    dtheta = math.pi / H
    dphi = 2.0 * math.pi / W

    for y in range(H):
        # center of pixel in theta
        theta = (y + 0.5) * dtheta
        sin_theta = math.sin(theta)
        for x in range(W):
            phi = (x + 0.5) * dphi

            # direction on unit sphere
            st = sin_theta
            ct = math.cos(theta)
            cp = math.cos(phi)
            sp = math.sin(phi)
            vx = st * cp
            vy = st * sp
            vz = ct

            L = env[y, x, :]  # RGB
            if sin_theta < 1e-6:
                continue

            Y = sh_basis_dir_l2(vx, vy, vz)  # 9 basis values

            weight = sin_theta * dtheta * dphi  # area element on sphere
            coeffs += (Y[:, None] * L[None, :] * weight)

    # coeffs now approximate integral over the sphere of L * Y_lm dω
    # No additional normalization needed for our convention.
    return coeffs.astype(np.float32)  # (num_coeffs, 3)

def sh_for_shape_env(sample_id, env_id, order=2):
    """
    Env 'snake': env_id moves along a smooth curve through RGB & SH space.
    Neighboring env_id are similar; over many env_id you cover a range.
    """
    pairs = sh_lm_list(order)
    num_coeffs = len(pairs)
    coeffs = np.zeros((num_coeffs, 3), dtype=np.float32)

    # Base shape index (so different shapes get offset snakes)
    base_t = float(sample_id) * 0.1

    # Env parameter along the snake
    k = float(env_id)
    t = base_t + k * 0.2          # step size along curve
    u = (k - 1) / max(1, NUM_ENVS_PER_SHAPE - 1)  # in [0,1]

    # 1) Make RGB walk around a color wheel (slowly varying)
    #    This gives you different color casts per env, but adjacent ones are close.
    r = 0.5 + 0.4 * math.sin(t)
    g = 0.5 + 0.4 * math.sin(t + 2.0 * math.pi / 3.0)
    b = 0.5 + 0.4 * math.sin(t + 4.0 * math.pi / 3.0)
    rgb_scale = np.array([r, g, b], dtype=np.float32)

    # 2) Base ambient term (l=0,m=0): slightly colored
    coeffs[0, :] = rgb_scale * 0.4

    # 3) First-order band: direction varies smoothly with env
    #    Interpret (u) as going around a small ellipsoid in SH(1) space.
    for idx, (l, m) in enumerate(pairs):
        if l == 1:
            if m == -1:    # roughly Y
                coeffs[idx, :] = rgb_scale * (0.2 * math.sin(2.0 * math.pi * u))
            elif m == 0:   # roughly Z
                coeffs[idx, :] = rgb_scale * (0.2 * math.cos(2.0 * math.pi * u))
            elif m == 1:   # roughly X
                coeffs[idx, :] = rgb_scale * (0.2 * math.sin(2.0 * math.pi * u + 1.0))

    # 4) Small l=2 term to add variation without huge effect
    for idx, (l, m) in enumerate(pairs):
        if l == 2 and m == 0:
            coeffs[idx, :] += rgb_scale * (0.05 * math.cos(4.0 * math.pi * u))

    return coeffs

# ==============================
# MAIN
# ==============================

def main():
    start_id = None
    end_id = None
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if "--start_id" in argv:
        start_id = int(argv[argv.index("--start_id") + 1])
    if "--end_id" in argv:
        end_id = int(argv[argv.index("--end_id") + 1])

    scene = bpy.context.scene
    clean_scene()
    IMAGE_METADATA_CSV = os.path.join(RENDER_DIR, f"metadata_images_{start_id}.csv")

    # basic render settings (same as before)
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'   # force CPU
    scene.cycles.samples = SAMPLES
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    setup_world_and_lighting(scene)

    # precompute camera angles (same as before)
    # single fixed camera angle
    phi_fixed   = 0.5 * math.pi     # 90°, level with the object; tweak if needed
    theta_fixed = math.radians(45)  # 45° around

    camera_angles = [(phi_fixed, theta_fixed)]



    # Precompute global env SH + envmaps
    global_env_sh = {}   # env_id -> sh_coeffs
    global_env_path = {} # env_id -> file path

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

    # filter by sample_id range if given
    if start_id is not None or end_id is not None:
        filtered = []
        for row in vol_rows:
            sid = int(row["sample_id"])
            if (start_id is None or sid >= start_id) and (end_id is None or sid <= end_id):
                filtered.append(row)
        vol_rows = filtered
        print(f"Processing sample_id in [{start_id}, {end_id}], count={len(vol_rows)}")

    fieldnames = [
        "image_path",
        "sample_id",
        "env_id",          # add this
        "coeff_path",
        "mesh_path",
        "hue",
        "metallic",
        "roughness",
        "specular",
        "material_type",
        "base_color_r",
        "base_color_g",
        "base_color_b",
        "phi",
        "theta",
        "radius",
        "env_path",
    ]
    # then append SH fields as you already do
    sh_pairs = sh_lm_list(SH_ORDER)
    for (l, m) in sh_pairs:
        for c in ["r", "g", "b"]:
            fieldnames.append(f"sh_l{l}_m{m}_{c}")

    f_img = open(IMAGE_METADATA_CSV, "w", newline="")
    writer = csv.DictWriter(f_img, fieldnames=fieldnames)
    writer.writeheader()

    # main loops over meshes, materials, camera views
    for vol_row in vol_rows:
        sample_id = int(vol_row["sample_id"])
        mesh_path = vol_row["mesh_path"]
        coeff_path = vol_row["coeff_path"]

        # remove existing non-light, non-camera objects
        for o in list(bpy.data.objects):
            if o.type not in {'LIGHT', 'CAMERA'}:
                bpy.data.objects.remove(o, do_unlink=True)

        # load verts/faces from .npz
        full_mesh_path = os.path.abspath(mesh_path)
        data = np.load(full_mesh_path)
        verts = data["verts"].astype(np.float32)   # (N, 3)
        faces = data["faces"]                      # (M, 3)
        
        # recenter: subtract bbox center so object is around (0,0,0)
        vmin = verts.min(axis=0)
        vmax = verts.max(axis=0)
        center = 0.5 * (vmin + vmax)
        verts_centered = verts - center  # now roughly symmetric around origin
        
        mesh = bpy.data.meshes.new(f"mesh_{sample_id:04d}")
        mesh.from_pydata(verts_centered.tolist(), [], faces.tolist())
        mesh.update()
        
        shape_obj = bpy.data.objects.new(mesh.name, mesh)
        scene.collection.objects.link(shape_obj)
        shape_obj.name = f"shape_{sample_id:04d}"
        
        # smooth shading to reduce blocky look
        bpy.context.view_layer.objects.active = shape_obj
        shape_obj.select_set(True)
        bpy.ops.object.shade_smooth()
        shape_obj.select_set(False)

        # remove existing cameras, then create camera targeting this shape
        for o in list(bpy.data.objects):
            if o.type == 'CAMERA':
                bpy.data.objects.remove(o, do_unlink=True)
        cam = create_camera(scene, shape_obj)
# -------- environments for this shape --------
        env_ids = env_ids_for_shape(sample_id)  # e.g. range(NUM_GLOBAL_ENVS)

        for env_id in env_ids:
            sh_coeffs = global_env_sh[env_id]
            env_path = global_env_path[env_id]

            # set world env for this image batch
            set_env_texture(scene, env_path, strength=1.0)

            mat_id = 0
            for hue in HUE_VALUES:
                for metallic in METALLIC_VALUES:
                    for roughness in ROUGHNESS_VALUES:
                        for specular in SPECULAR_VALUES:
                            mat_id += 1
                            mat, base_color, material_type = make_material_from_params(
                                float(hue),
                                float(metallic),
                                float(roughness),
                                float(specular),
                            )
                            shape_obj.data.materials.clear()
                            shape_obj.data.materials.append(mat)

                            cam_view_idx = 0
                            for phi, theta in camera_angles:
                                cam_view_idx += 1
                                set_camera_from_spherical(cam, RADIUS, phi, theta)

                                img_name = (
                                    f"s{sample_id:04d}_e{env_id:03d}_m{mat_id:03d}_v{cam_view_idx:03d}.png"
                                )
                                img_path = os.path.join(RENDER_DIR, img_name)

                                scene.render.filepath = img_path
                                bpy.ops.render.render(write_still=True)
                                print("Rendered:", img_path)

                                row = {
                                    "image_path": img_path,
                                    "sample_id": sample_id,
                                    "env_id": env_id,
                                    "coeff_path": coeff_path,
                                    "mesh_path": mesh_path,
                                    "hue": float(hue),
                                    "metallic": float(metallic),
                                    "roughness": float(roughness),
                                    "specular": float(specular),
                                    "material_type": material_type,
                                    "base_color_r": float(base_color[0]),
                                    "base_color_g": float(base_color[1]),
                                    "base_color_b": float(base_color[2]),
                                    "phi": float(phi),
                                    "theta": float(theta),
                                    "radius": float(RADIUS),
                                    "env_path": env_path,
                                }

                                for idx, (l, m) in enumerate(sh_pairs):
                                    r_c, g_c, b_c = sh_coeffs[idx]
                                    row[f"sh_l{l}_m{m}_r"] = float(r_c)
                                    row[f"sh_l{l}_m{m}_g"] = float(g_c)
                                    row[f"sh_l{l}_m{m}_b"] = float(b_c)

                                writer.writerow(row)
    f_img.close()
    print("Done. Image metadata written to", IMAGE_METADATA_CSV)

if __name__ == "__main__":
    main()