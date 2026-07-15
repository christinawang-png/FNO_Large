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
BASE_DIR = PROJECT_ROOT / "plane_dataset_3"

VOLUME_METADATA_CSV = os.path.join(BASE_DIR, "metadata_volumes.csv")

# Where to write rendered images and image metadata
RENDER_DIR = os.path.join(BASE_DIR, "renders")
os.makedirs(RENDER_DIR, exist_ok=True)

# ENV
NUM_GLOBAL_ENVS    = 128          # instead of 128
NUM_ENVS_PER_SHAPE = 8           # 8 different envs per shape

# MATERIALS
HUE_VALUES       = [0.0, 1/3, 2/3]  # red, green, blue-ish
SAT_VALUES       = [0.4, 0.8]       # muted / saturated
METALLIC_VALUES  = [0.0, 1.0]       # plastic / metal
ROUGHNESS_VALUES = [0.2, 0.6]       # semi-gloss / rough
SPECULAR_VALUES  = [0.5]            # fixed
OPACITY_VALUES   = [1.0, 0.3]       # opaque / semi

# CAMERA
RADIUS_VALUES = [0.8, 1.1]                       # close / mid

# render settings
RES_X = 64
RES_Y = 64
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
NUM_ENVS_PER_SHAPE = NUM_GLOBAL_ENVS       # envs used per shape
IMAGES_PER_SHAPE = 500

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

def make_material_from_params(hue, saturation, metallic, roughness, specular, opacity):
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, 0.4)
    base_color = (r, g, b, opacity)

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
    bsdf.inputs["Alpha"].default_value      = opacity

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
    Pick NUM_ENVS_PER_SHAPE env_ids for this sample_id
    from 0..NUM_GLOBAL_ENVS-1, spread along the snake.
    """
    base = (sample_id * 7) % NUM_GLOBAL_ENVS
    step = max(1, NUM_GLOBAL_ENVS // NUM_ENVS_PER_SHAPE)
    ids = [ (base + k * step) % NUM_GLOBAL_ENVS for k in range(NUM_ENVS_PER_SHAPE) ]
    return ids

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
    phi_values   = [math.radians(45), math.radians(65)]
    theta_values = [math.radians(t) for t in [0, 120, 240]]  # 3 azimuths

    camera_poses = []
    for radius in RADIUS_VALUES:
        for phi in phi_values:
            for theta in theta_values:
                camera_poses.append((radius, phi, theta))


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
        "saturation",
        "metallic",
        "roughness",
        "specular",
        "material_type",
        "base_color_r",
        "base_color_g",
        "base_color_b",
        "opacity",
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
# remove existing cameras, then create camera targeting this shape

    # ---------- RANDOM SAMPLING PER SHAPE ----------
    # Optional: seed per shape for reproducibility
        rng = np.random.RandomState(sample_id)

        for img_idx in range(IMAGES_PER_SHAPE):
            # 1) Sample environment
            env_id = int(rng.randint(0, NUM_GLOBAL_ENVS))
            sh_coeffs = global_env_sh[env_id]
            env_path  = global_env_path[env_id]
            set_env_texture(scene, env_path, strength=1.0)

            # 2) Sample material parameters
            hue        = float(rng.uniform(0.0, 1.0))            # full hue wheel
            saturation = float(rng.uniform(0.3, 0.9))            # avoid extremes
            metallic   = float(rng.choice([0.0, 1.0]))           # binary
            roughness  = float(rng.uniform(0.1, 0.9))            # continuous
            specular   = 0.5                                     # keep fixed
            opacity    = float(rng.uniform(1.0, 0.1))           # opaque / transparent

            mat, base_color, material_type = make_material_from_params(
                hue, saturation, metallic, roughness, specular, opacity
            )
            shape_obj.data.materials.clear()
            shape_obj.data.materials.append(mat)

            # 3) Sample camera pose
            radius = float(rng.uniform(0.8, 1.2))
            phi    = float(rng.uniform(0.0, math.pi))
            theta  = float(rng.uniform(0.0, 2.0 * math.pi))

            set_camera_from_spherical(cam, radius, phi, theta)

            # 4) Render
            img_name = f"s{sample_id:04d}_i{img_idx:05d}.png"
            img_path = os.path.join(RENDER_DIR, img_name)

            scene.render.filepath = img_path
            bpy.ops.render.render(write_still=True)
            print("Rendered:", img_path)

            # 5) Write metadata
            row = {
                "image_path": img_path,
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
            }

            for idx, (l, m) in enumerate(sh_pairs):
                r_c, g_c, b_c = sh_coeffs[idx]
                row[f"sh_l{l}_m{m}_r"] = float(r_c)
                row[f"sh_l{l}_m{m}_g"] = float(g_c)
                row[f"sh_l{l}_m{m}_b"] = float(b_c)

            writer.writerow(row)
# -----------------------------------------------
    f_img.close()
    print("Done. Image metadata written to", IMAGE_METADATA_CSV)

if __name__ == "__main__":
    main()