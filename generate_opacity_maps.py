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

# ==============================
# MAIN
# ==============================

def integrate_volume_along_camera_torch(V, R, phi, theta,
                                        num_samples=128,
                                        sigma_e=1.0,
                                        R_ref=1.0):
    """
    V: torch.Tensor [nx,ny,nz] on device, density in [0, +inf)
    R: radius of camera (float)
    phi, theta: spherical angles (radians), same convention as in your render script
    num_samples: integration samples along each ray
    sigma_e: extinction coefficient
    R_ref: reference radius for zoom (R<R_ref => zoom in, R>R_ref => zoom out)

    Returns:
        alpha: [nx,ny] tensor on same device, in [0,1]
    """

    device = V.device
    nx, ny, nz = V.shape

    # Camera direction in world: same as set_camera_from_spherical
    # cam_dir points from origin to camera; we want view_dir from camera to origin
    cam_dir = torch.tensor([
        math.sin(phi) * math.cos(theta),
        math.sin(phi) * math.sin(theta),
        math.cos(phi),
    ], dtype=torch.float32, device=device)
    cam_dir = cam_dir / cam_dir.norm()

    # viewing direction: from camera to center
    d = -cam_dir  # [3], unit

    # Orthonormal basis {u, v, d}
    if abs(d[2]) < 0.9:
        a = torch.tensor([0.0, 0.0, 1.0], device=device)
    else:
        a = torch.tensor([1.0, 0.0, 0.0], device=device)

    u = torch.cross(d, a); u = u / u.norm()
    v = torch.cross(d, u); v = v / v.norm()

    # Image plane: center at volume center, extents scale with 1/R
    center = torch.tensor([0.5, 0.5, 0.5], device=device)
    H, W = nx, ny

    ys = torch.linspace(-1.0, 1.0, H, device=device)
    xs = torch.linspace(-1.0, 1.0, W, device=device)
    Ts, Ss = torch.meshgrid(ys, xs, indexing="ij")  # [H,W]

    # half size of cube is 0.5; scale by R_ref/R for zoom
    half_extent = 0.5 * (R_ref / max(R, 1e-6))

    # base plane (before pushing in front of cube)
    plane_pts = center[None,None,:] \
        + half_extent * (Ss[...,None]*u[None,None,:] + Ts[...,None]*v[None,None,:])  # [H,W,3]

    # move plane in front of cube along -d so rays enter from "front"
    L = math.sqrt(3.0)  # a bit more than cube diagonal/2
    origins = plane_pts - L * d[None,None,:]   # [H,W,3]
    d_vec   = d[None,None,:].expand(H, W, 3)   # [H,W,3]

    # Ray-box intersection with [0,1]^3
    t_min = torch.full((H, W), -float("inf"), device=device)
    t_max = torch.full((H, W),  float("inf"), device=device)

    for axis in range(3):
        o = origins[..., axis]
        dd = d_vec[..., axis]
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

    # Sample along rays
    num_samples = int(num_samples)
    ts = torch.linspace(0.0, 1.0, num_samples, device=device)

    t_min_exp = t_min[..., None]
    t_max_exp = t_max[..., None]
    s_all     = t_min_exp + ts * (t_max_exp - t_min_exp)  # [H,W,S]

    o_exp = origins[..., None, :]   # [H,W,1,3]
    d_exp = d_vec[..., None, :]     # [H,W,1,3]
    pos   = o_exp + s_all[...,None] * d_exp  # [H,W,S,3]

    # clamp to volume cube
    pos = pos.clamp(0.0, 1.0)

    ix = (pos[...,0] * (nx-1)).round().long().clamp(0, nx-1)
    iy = (pos[...,1] * (ny-1)).round().long().clamp(0, ny-1)
    iz = (pos[...,2] * (nz-1)).round().long().clamp(0, nz-1)

    rho = V[ix, iy, iz]   # [H,W,S]

    hit_mask_exp = hit_mask[..., None]
    rho = rho * hit_mask_exp

    seg_len = (t_max - t_min).clamp(min=0.0)
    ds = seg_len / max(num_samples, 1)

    tau = sigma_e * (rho.sum(dim=2) * ds)  # [H,W]
    T   = torch.exp(-tau)
    alpha = 1.0 - T
    return alpha


def integrate_volume_along_direction_torch(V, d, num_samples=128, sigma_e=1.0):
    """
    V: torch.Tensor [nx, ny, nz] on device, density in [0, +inf)
    d: torch.Tensor [3], unit direction vector (dx,dy,dz)
    num_samples: number of integration samples along each ray
    Returns:
        alpha: [nx, ny] float32 tensor on same device, in [0,1]
    We assume:
      - Volume lives in cube [0,1]^3.
      - Orthographic camera: all rays are parallel to d.
      - Image plane is chosen to cover projection of the cube.
    """

    device = V.device
    nx, ny, nz = V.shape

    # Orthonormal basis {u,v,d}
    # Choose an arbitrary vector not parallel to d for constructing u
    d = d / torch.norm(d)
    if abs(d[2]) < 0.9:
        a = torch.tensor([0.0, 0.0, 1.0], device=device)
    else:
        a = torch.tensor([1.0, 0.0, 0.0], device=device)

    u = torch.cross(d, a)
    u = u / torch.norm(u)
    v = torch.cross(d, u)
    v = v / torch.norm(v)

    # Image plane grid in (s,t) ∈ [-1,1]^2 (wider window)
    H, W = nx, ny  # or your higher preview resolution
    ys = torch.linspace(-0.7, 0.7, H, device=device)
    xs = torch.linspace(-0.7, 0.7, W, device=device)
    Ts, Ss = torch.meshgrid(ys, xs, indexing="ij")  # [H,W]
    
    # Larger scale factor: full cube diagonal ≈ sqrt(3)
    L = 1.3  # full diagonal + margin
    
    center = torch.tensor([0.5, 0.5, 0.5], device=device)
    origins = center[None, None, :] + L * (Ss[..., None] * u[None,None,:] + Ts[...,None] * v[None,None,:])
    # All ray directions = d
    d_vec = d[None, None, :].expand(H, W, 3)

    # Ray-box intersection with [0,1]^3
    t_min = torch.full((H, W), -float("inf"), device=device)
    t_max = torch.full((H, W),  float("inf"), device=device)

    for axis in range(3):
        o = origins[..., axis]
        dd = d_vec[..., axis]
        # Avoid division by zero: if dd == 0, ray is parallel to that slab
        # and either always inside or outside depending on origin.
        mask_nonzero = (dd.abs() > 1e-6)
        inv_d = torch.zeros_like(dd)
        inv_d[mask_nonzero] = 1.0 / dd[mask_nonzero]

        t1 = (0.0 - o) * inv_d
        t2 = (1.0 - o) * inv_d
        t_near = torch.minimum(t1, t2)
        t_far  = torch.maximum(t1, t2)

        t_min = torch.maximum(t_min, t_near)
        t_max = torch.minimum(t_max, t_far)

    # mask of rays that hit the box
    hit_mask = (t_max > t_min) & (t_max > 0)

    # initialize tau
    tau = torch.zeros((H, W), device=device, dtype=torch.float32)

    # sample along rays only for hits
    num_samples = int(num_samples)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    ts = torch.linspace(0.0, 1.0, num_samples, device=device)
    # For each pixel, parameter along [t_min, t_max]:
    # s = t_min + ts * (t_max - t_min)
    t_min_exp = t_min[..., None]
    t_max_exp = t_max[..., None]
    s_all = t_min_exp + ts * (t_max_exp - t_min_exp)  # [H,W,S]

    # Positions: x = o + s*d
    o_exp = origins[..., None, :]   # [H,W,1,3]
    d_exp = d_vec[..., None, :]     # [H,W,1,3]
    pos = o_exp + s_all[..., None] * d_exp  # [H,W,S,3]

    # Clip positions to [0,1] to stay inside grid bounds
    pos = torch.clamp(pos, 0.0, 1.0)

    # Map positions to voxel indices (nearest neighbor)
    ix = (pos[..., 0] * (nx - 1)).round().long().clamp(0, nx - 1)  # [H,W,S]
    iy = (pos[..., 1] * (ny - 1)).round().long().clamp(0, ny - 1)
    iz = (pos[..., 2] * (nz - 1)).round().long().clamp(0, nz - 1)

    # Gather densities
    rho = V[ix, iy, iz]  # [H,W,S]

    # Only integrate where rays actually hit the box
    # Zero out contributions for non-hit pixels
    hit_mask_exp = hit_mask[..., None]  # [H,W,1]
    rho = rho * hit_mask_exp

    # approximate integral: tau ≈ sigma_e * Δs * sum rho
    # Δs = average length of segment divided by num_samples
    # We can use average segment length L_seg ~ |t_max - t_min|
    seg_len = (t_max - t_min).clamp(min=0.0)  # [H,W]
    ds = seg_len / num_samples
    tau = sigma_e * (rho.sum(dim=2) * ds)  # [H,W]

    # transmittance and opacity
    T = torch.exp(-tau)
    alpha = 1.0 - T  # [H,W] in [0,1]

    return alpha
    
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

    shard_id = 0
    idx_in_shard = 0
    shard_array = np.empty((SHARD_SIZE, IMG_H, IMG_W), dtype=np.float32)

    fieldnames = [
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

        for vol_row in vol_rows:
            sample_id   = int(vol_row["sample_id"])
            volume_path = vol_row["volume_path"]
            coeff_path  = vol_row.get("coeff_path", "")

            full_volume_path = (
                BASE_DIR / volume_path
                if not os.path.isabs(volume_path)
                else Path(volume_path)
            )
            full_volume_path = full_volume_path.resolve()
            if not full_volume_path.is_file():
                print(f"[WARN] volume not found for sample_id={sample_id}: {full_volume_path}")
                continue

            # load volume to GPU
            V_np = np.load(full_volume_path).astype(np.float32)  # (nx,ny,nz)
            if V_np.shape != (GRID_NX, GRID_NY, GRID_NZ):
                print(f"[WARN] volume shape mismatch for sample_id={sample_id}: {V_np.shape}")
                continue

            V = torch.from_numpy(V_np).to(device)  # [64,64,64]

            for view_idx in range(NUM_VIEWS_PER_SHAPE):
                # sample camera params in same ranges as your renderer
                R      = float(rng.uniform(0.8, 1.2))
                phi    = float(rng.uniform(0.0, math.pi))  # e.g. 30°–150° in radians
                theta  = float(rng.uniform(0.0, 2.0*math.pi))
            
                sigma_e   = float(rng.uniform(0.5, 2.0))
                roughness = float(rng.uniform(0.1, 0.9))
                metallic  = float(rng.choice([0.0, 1.0]))
            
                alpha = integrate_volume_along_camera_torch(
                    V, R, phi, theta,
                    num_samples=128,
                    sigma_e=sigma_e,
                    R_ref=1.0,
                )  # [64,64]
            
                alpha_np = alpha.clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
            
                shard_array[idx_in_shard] = alpha_np
            
                row = {
                    "sample_id": sample_id,
                    "coeff_path": str(coeff_path),
                    "volume_path": str(volume_path),
                    "view_idx": view_idx,
                    "radius": R,
                    "phi": phi,
                    "theta": theta,
                    "sigma_e": sigma_e,
                    "roughness": roughness,
                    "metallic": metallic,
                    "shard_id": shard_id,
                    "idx_in_shard": idx_in_shard,
                }
                writer.writerow(row)
            
                idx_in_shard += 1
                if idx_in_shard == SHARD_SIZE:
                    shard_name = f"opacity_64x64_shard_{shard_id:05d}.npy"
                    shard_path = OPACITY_DIR / shard_name
                    np.save(shard_path, shard_array[:idx_in_shard])
                    print("Saved shard:", shard_path)
                    shard_id += 1
                    idx_in_shard = 0

        # save final partial shard
        if idx_in_shard > 0:
            shard_name = f"opacity_128x128_shard_{shard_id:05d}.npy"
            shard_path = OPACITY_DIR / shard_name
            np.save(shard_path, shard_array[:idx_in_shard])
            print("Saved final shard:", shard_path)

    print("Done. Metadata:", OUTPUT_CSV)
    print("Opacity shards in:", OPACITY_DIR)


if __name__ == "__main__":
    main()