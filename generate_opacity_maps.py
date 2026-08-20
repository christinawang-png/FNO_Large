#!/usr/bin/env python
import os
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd

import torch

# ==============================
# CONFIG
# ==============================

PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DIR     = PROJECT_ROOT / "plane_dataset_4"

VOLUME_METADATA_CSV = BASE_DIR / "metadata_volumes.csv"
IMAGE_METADATA_CSV = BASE_DIR / "renders" / "metadata_images_all_sharded.csv"

OPACITY_DIR = BASE_DIR / "opacity_shards"
os.makedirs(OPACITY_DIR, exist_ok=True)

GRID_NX = 64
GRID_NY = 64
GRID_NZ = 64

IMG_H = 64
IMG_W = 64

# for now: 1 front-view per shape; you can bump this if you want multiple sigmas/materials
NUM_VIEWS_PER_SHAPE = 300

SHARD_SIZE = 5000
OUTPUT_CSV = OPACITY_DIR / "metadata_opacity_sharded.csv"

GLOBAL_SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

SIGMA_E_BASE = 5.0            # tune as you like
NUM_SAMPLES_ALONG_RAY = 256   # or 128
V_THR = 0.9                   # thin band threshold

# vertical FOV in radians;
# Blender default camera: ~26.99° vertical (50mm on 24mm sensor height)
BLENDER_FOV_Y_DEG = 26.99
BLENDER_FOV_Y_RAD = math.radians(BLENDER_FOV_Y_DEG)

# ==============================
# MAIN
# ==============================

def integrate_volume_pinhole_torch(
        V,
        R,
        phi,
        theta,
        center,        # <-- NEW: 3D center of the object in volume coords
        H,
        W,
        fov_y,
        num_samples=128,
        sigma_e=1.0,
        v_thr=0.9,
    ):
    """
    V: [nx,ny,nz] volume in [0,1]^3 (volume coordinates).
    R, phi, theta: same as Blender set_camera_from_spherical.
    center: torch.tensor([cx,cy,cz]) in volume coordinates; this is
            the bbox center used to recenter verts in Blender.
    H, W: output image resolution (64x64).
    fov_y: vertical FOV in radians (Blender: camera.data.angle_y).
    """

    device = V.device
    nx, ny, nz = V.shape

    # 1) Camera direction as in Blender (from origin to camera)
    cam_dir = torch.tensor([
        math.sin(phi) * math.cos(theta),
        math.sin(phi) * math.sin(theta),
        math.cos(phi),
    ], dtype=torch.float32, device=device)
    cam_dir = cam_dir / cam_dir.norm()

    # In Blender: object center is at (0,0,0), camera at R * cam_dir
    # In volume coords: we *translate* everything so that that object
    # center is at `center`. So camera center in volume coords is:
    cam_center = center + R * cam_dir   # [3]

    # forward direction: from camera to object center
    fwd = (center - cam_center)
    fwd = fwd / fwd.norm()

    # 2) Camera basis {right, up, forward}
    # choose arbitrary world up not parallel to fwd
    if abs(fwd[2].item()) < 0.9:
        a = torch.tensor([0.0, 0.0, 1.0], device=device)
    else:
        a = torch.tensor([1.0, 0.0, 0.0], device=device)

    right = torch.cross(fwd, a); right = right / right.norm()
    up    = torch.cross(right, fwd); up = up / up.norm()

    # 3) Pinhole camera: FOV_y
    half_h = math.tan(fov_y / 2.0)
    aspect = W / H
    half_w = half_h * aspect

    # Pixel grid: match Blender's rasterization
    ys = torch.linspace(0, H - 1, H, device=device)
    xs = torch.linspace(0, W - 1, W, device=device)
    j_grid, i_grid = torch.meshgrid(xs, ys, indexing="xy")  # j=col, i=row

    # Normalize pixel coordinates to [-1,1]
    u = (2.0 * (j_grid + 0.5) / W - 1.0)   # [H,W], x
    v = (1.0 - 2.0 * (i_grid + 0.5) / H)   # [H,W], y (flip so +v is up)

    # Camera-space/world-space ray directions
    dir_world = (
        u[..., None] * half_w * right[None, None, :] +
        v[..., None] * half_h * up[None, None, :] +
        1.0          * fwd[None, None, :]
    )  # [H,W,3]

    dir_world = dir_world / torch.norm(dir_world, dim=-1, keepdim=True)  # [H,W,3]
    origins = cam_center[None, None, :].expand(H, W, 3)                  # [H,W,3]

    # 4) Ray-box intersection with [0,1]^3 in *volume* coordinates
    t_min = torch.full((H, W), -float("inf"), device=device)
    t_max = torch.full((H, W),  float("inf"), device=device)

    for axis in range(3):
        o  = origins[..., axis]
        dd = dir_world[..., axis]
        mask_nonzero = (dd.abs() > 1e-6)
        inv_d = torch.zeros_like(dd)
        inv_d[mask_nonzero] = 1.0 / dd[mask_nonzero]

        t1 = (0.0 - o) * inv_d
        t2 = (1.0 - o) * inv_d
        t_near = torch.minimum(t1, t2)
        t_far  = torch.maximum(t1, t2)

        t_min = torch.maximum(t_min, t_near)
        t_max = torch.minimum(t_max, t_far)

    hit_mask = (t_max > t_min) & (t_max > 0)

    # 5) Sample along rays inside the box
    num_samples = int(num_samples)
    ts = torch.linspace(0.0, 1.0, num_samples, device=device)

    t_min_exp = t_min[..., None]
    t_max_exp = t_max[..., None]
    s_all     = t_min_exp + ts * (t_max_exp - t_min_exp)  # [H,W,S]

    o_exp = origins[..., None, :]   # [H,W,1,3]
    d_exp = dir_world[..., None, :] # [H,W,1,3]
    pos   = o_exp + s_all[..., None] * d_exp  # [H,W,S,3] in volume coords

    # clip to volume cube
    pos = pos.clamp(0.0, 1.0)

    ix = (pos[..., 0] * (nx-1)).round().long().clamp(0, nx-1)
    iy = (pos[..., 1] * (ny-1)).round().long().clamp(0, ny-1)
    iz = (pos[..., 2] * (nz-1)).round().long().clamp(0, nz-1)

    # thin band of high density
    V_band = (V > v_thr).float()     # [nx,ny,nz]
    rho = V_band[ix, iy, iz]         # [H,W,S]

    hit_mask_exp = hit_mask[..., None]
    rho = rho * hit_mask_exp

    seg_len = (t_max - t_min).clamp(min=0.0)  # [H,W]
    ds = seg_len / max(num_samples, 1)

    tau = sigma_e * (rho.sum(dim=2) * ds)  # [H,W]
    T   = torch.exp(-tau)
    alpha = 1.0 - T
    return alpha  # [H,W]
    
def random_unit_vectors(num, rng):
    """
    Sample 'num' random unit vectors on the sphere.
    rng: np.random.RandomState
    Returns: (num, 3) numpy array of unit vectors.
    """
    zs = rng.uniform(-1.0, 1.0, size=num)
    thetas = rng.uniform(0.0, 2.0 * math.pi, size=num)
    rs = np.sqrt(1.0 - zs**2)
    xs = rs * np.cos(thetas)
    ys = rs * np.sin(thetas)
    dirs = np.stack([xs, ys, zs], axis=1)  # (num, 3)
    return dirs

def main():
    rng = np.random.RandomState(GLOBAL_SEED)

    # read volume metadata
    df_vol = pd.read_csv(VOLUME_METADATA_CSV)
    vol_rows = df_vol.to_dict("records")
    
    df_img = pd.read_csv(IMAGE_METADATA_CSV)      # plane_dataset_4/renders/metadata_images_all_sharded.csv
    df_vol = pd.read_csv(VOLUME_METADATA_CSV).set_index("sample_id")
    
    # map sample_id -> volume_path
    vol_paths = {int(sid): row["volume_path"] for sid, row in df_vol.iterrows()}

    shard_id = 0
    idx_in_shard = 0
    shard_array = np.empty((SHARD_SIZE, IMG_H, IMG_W), dtype=np.float32)

    fieldnames = [
        "img_row_idx",           # link back to render CSV
        "sample_id",
        "coeff_path",
        "volume_path",
        "view_idx",
        "radius",
        "phi",
        "theta",
        "sigma_e",
        "roughness",
        "metallic",
        "shard_id",
        "idx_in_shard",
    ]

    with open(OUTPUT_CSV, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
    
        for img_row_idx, row in df_img.iterrows():
            sample_id = int(row["sample_id"])
            radius    = float(row["radius"])
            phi       = float(row["phi"])
            theta     = float(row["theta"])
    
            coeff_path  = row.get("coeff_path", "")
            volume_path = vol_paths.get(sample_id, None)
            if volume_path is None:
                print(f"[WARN] no volume_path for sample_id={sample_id}")
                continue
    
            full_volume_path = (
                BASE_DIR / volume_path
                if not os.path.isabs(volume_path)
                else Path(volume_path)
            )
            full_volume_path = full_volume_path.resolve()
            if not full_volume_path.is_file():
                print(f"[WARN] volume not found for sample_id={sample_id}: {full_volume_path}")
                continue
    
            # NEW: load mesh and compute bbox center in volume coords
            mesh_path = row.get("mesh_path", None)
            if mesh_path is None:
                print(f"[WARN] no mesh_path for sample_id={sample_id}")
                continue
    
            full_mesh_path = (
                BASE_DIR / mesh_path
                if not os.path.isabs(mesh_path)
                else Path(mesh_path)
            )
            full_mesh_path = full_mesh_path.resolve()
            if not full_mesh_path.is_file():
                print(f"[WARN] mesh not found for sample_id={sample_id}: {full_mesh_path}")
                continue
    
            mesh_data = np.load(full_mesh_path)
            verts = mesh_data["verts"].astype(np.float32)  # these are in [0,1]^3 coords
            vmin = verts.min(axis=0)
            vmax = verts.max(axis=0)
            center_np = 0.5 * (vmin + vmax)               # same as Blender's 'center'
            center_t  = torch.from_numpy(center_np).to(device)
    
            # load volume
            V_np = np.load(full_volume_path).astype(np.float32)
            if V_np.shape != (GRID_NX, GRID_NY, GRID_NZ):
                print(f"[WARN] volume shape mismatch for sample_id={sample_id}: {V_np.shape}")
                continue
            V = torch.from_numpy(V_np).to(device)
    
            sigma_e = float(row.get("opacity", 1.0)) * SIGMA_E_BASE
            roughness = float(row.get("roughness", 0.5))
            metallic  = float(row.get("metallic", 0.0))
    
            alpha = integrate_volume_pinhole_torch(
                V,
                R=radius,
                phi=phi,
                theta=theta,
                center=center_t,
                H=IMG_H,
                W=IMG_W,
                fov_y=BLENDER_FOV_Y_RAD,
                num_samples=NUM_SAMPLES_ALONG_RAY,
                sigma_e=sigma_e,
                v_thr=V_THR,
            )  # [64,64]
    
            alpha_np = alpha.clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
            shard_array[idx_in_shard] = alpha_np
    
            out_row = {
                "img_row_idx": img_row_idx,           # link back to render CSV
                "sample_id": sample_id,
                "coeff_path": coeff_path,
                "volume_path": volume_path,
                "view_idx": int(row.get("view_idx", 0)),
                "radius": radius,
                "phi": phi,
                "theta": theta,
                "sigma_e": sigma_e,
                "roughness": roughness,
                "metallic": metallic,
                "shard_id": shard_id,
                "idx_in_shard": idx_in_shard,
            }
            writer.writerow(out_row)
    
            idx_in_shard += 1
            if idx_in_shard == SHARD_SIZE:
                shard_name = f"opacity_64x64_from_images_shard_{shard_id:05d}.npy"
                shard_path = OPACITY_DIR / shard_name
                np.save(shard_path, shard_array[:idx_in_shard])
                print("Saved shard:", shard_path)
                shard_id += 1
                idx_in_shard = 0
    
        # final partial shard
        if idx_in_shard > 0:
            shard_name = f"opacity_64x64_from_images_shard_{shard_id:05d}.npy"
            shard_path = OPACITY_DIR / shard_name
            np.save(shard_path, shard_array[:idx_in_shard])
            print("Saved final shard:", shard_path)

    print("Done. Metadata:", OUTPUT_CSV)
    print("Opacity shards in:", OPACITY_DIR)


if __name__ == "__main__":
    main()