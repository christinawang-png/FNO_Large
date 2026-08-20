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

# Ground-truth image metadata (the one you used to train the renderer)
IMAGE_METADATA_CSV = BASE_DIR / "renders" / "metadata_images_all_sharded.csv"

# Volume metadata
VOLUME_METADATA_CSV = BASE_DIR / "metadata_volumes.csv"

# Where to write opacity shards and metadata
OPACITY_DIR = BASE_DIR / "opacity_from_images_shards"
os.makedirs(OPACITY_DIR, exist_ok=True)

OPACITY_META_CSV = OPACITY_DIR / "metadata_opacity_from_images_sharded.csv"

GRID_NX = 64
GRID_NY = 64
GRID_NZ = 64

IMG_H = 64
IMG_W = 64

SHARD_SIZE = 5000
GLOBAL_SEED = 42

# integration params
NUM_SAMPLES_ALONG_RAY = 128
V_THR = 0.5        # threshold on V to define "surface band"
SIGMA_E_BASE = 2.0 # base extinction; you can scale by row["opacity"] if desired

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ==============================
# INTEGRATION
# ==============================

def integrate_volume_along_camera_thin_band(V, R, phi, theta,
                                            num_samples=128,
                                            sigma_e=1.0,
                                            R_ref=1.0,
                                            v_thr=0.5):
    """
    V: torch.Tensor [nx,ny,nz] on device, scalar field (e.g. Gaussian)
    R, phi, theta: camera pose parameters (same convention as your render script)
    num_samples: samples along each ray
    sigma_e: extinction coefficient
    R_ref: reference radius for zoom
    v_thr: threshold on V to define thin surface band (V>v_thr => inside band)

    Returns:
        alpha: [nx,ny] tensor on same device, in [0,1]
    """

    device = V.device
    nx, ny, nz = V.shape

    # camera direction (from origin to camera)
    cam_dir = torch.tensor([
        math.sin(phi) * math.cos(theta),
        math.sin(phi) * math.sin(theta),
        math.cos(phi),
    ], dtype=torch.float32, device=device)
    cam_dir = cam_dir / cam_dir.norm()

    # viewing direction: from camera to center
    d = -cam_dir  # [3], unit

    # Orthonormal basis {u, v, d}
    if abs(d[2].item()) < 0.9:
        a = torch.tensor([0.0, 0.0, 1.0], device=device)
    else:
        a = torch.tensor([1.0, 0.0, 0.0], device=device)

    u = torch.cross(d, a); u = u / u.norm()
    v = torch.cross(d, u); v = v / v.norm()

    # Image plane centered at cube center, extent scales with 1/R
    center = torch.tensor([0.5, 0.5, 0.5], device=device)
    H, W = nx, ny

    ys = torch.linspace(-1.0, 1.0, H, device=device)
    xs = torch.linspace(-1.0, 1.0, W, device=device)
    Ts, Ss = torch.meshgrid(ys, xs, indexing="ij")  # [H,W]

    half_extent = 0.5 * (R_ref / max(R, 1e-6))  # zoom factor

    plane_pts = center[None,None,:] \
        + half_extent * (Ss[...,None]*u[None,None,:] + Ts[...,None]*v[None,None,:])  # [H,W,3]

    # Move plane behind cube along -d so rays pass through cube toward camera
    L = math.sqrt(3.0)  # slightly larger than cube diagonal
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

    o_exp = origins[..., None, :]  # [H,W,1,3]
    d_exp = d_vec[..., None, :]    # [H,W,1,3]
    pos   = o_exp + s_all[...,None] * d_exp  # [H,W,S,3]

    pos = pos.clamp(0.0, 1.0)

    ix = (pos[...,0] * (nx-1)).round().long().clamp(0, nx-1)
    iy = (pos[...,1] * (ny-1)).round().long().clamp(0, ny-1)
    iz = (pos[...,2] * (nz-1)).round().long().clamp(0, nz-1)

    V_band = (V > v_thr).float()  # thin band around surface
    rho = V_band[ix, iy, iz]      # [H,W,S]

    hit_mask_exp = hit_mask[..., None]
    rho = rho * hit_mask_exp

    seg_len = (t_max - t_min).clamp(min=0.0)
    ds = seg_len / max(num_samples, 1)

    tau = sigma_e * (rho.sum(dim=2) * ds)  # [H,W]
    T   = torch.exp(-tau)
    alpha = 1.0 - T
    return alpha
    
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

IMG_META_CSV = BASE_DIR / "renders" / "metadata_images_all_sharded.csv"
VOL_META_CSV = BASE_DIR / "metadata_volumes.csv"

OPACITY_DIR  = BASE_DIR / "opacity_from_images_shards"
os.makedirs(OPACITY_DIR, exist_ok=True)

OPACITY_META_CSV = BASE_DIR / "metadata_opacity_from_images_sharded.csv"

GRID_NX = 64
GRID_NY = 64
GRID_NZ = 64

IMG_H = 64
IMG_W = 64

SHARD_SIZE = 5000
GLOBAL_SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def main():
    rng = np.random.RandomState(GLOBAL_SEED)

    # load metadata
    df_img = pd.read_csv(IMG_META_CSV)
    df_vol = pd.read_csv(VOL_META_CSV).set_index("sample_id")

    # map sample_id -> volume_path
    vol_paths = {}
    for sid, row in df_vol.iterrows():
        vol_paths[int(sid)] = row["volume_path"]

    shard_id = 0
    idx_in_shard = 0
    shard_array = np.empty((SHARD_SIZE, IMG_H, IMG_W), dtype=np.float32)

    fieldnames = [
        "img_row_idx",        # index into metadata_images_all_sharded
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

    with open(OPACITY_META_CSV, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        # optional: shuffle or just go in order
        for img_idx, row in df_img.iterrows():
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

            # load volume
            V_np = np.load(full_volume_path).astype(np.float32)
            if V_np.shape != (GRID_NX, GRID_NY, GRID_NZ):
                print(f"[WARN] volume shape mismatch for sample_id={sample_id}: {V_np.shape}")
                continue
            V = torch.from_numpy(V_np).to(device)

            # extinction scaled by rendered opacity if you want, or fixed
            # e.g. use row["opacity"] from image metadata
            sigma_e = float(row.get("opacity", 1.0)) * 2.0

            # random material-like extras (for labeling only)
            roughness = float(row.get("roughness", 0.5))
            metallic  = float(row.get("metallic", 0.0))

            alpha = integrate_volume_along_camera_thin_band(
                V,
                R=radius,
                phi=phi,
                theta=theta,
                num_samples=128,
                sigma_e=sigma_e,
                R_ref=1.0,
                v_thr=0.5,
            )  # [64,64]

            alpha_np = alpha.clamp(0.0, 1.0).cpu().numpy().astype(np.float32)

            shard_array[idx_in_shard] = alpha_np

            out_row = {
                "img_row_idx": img_idx,
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

    print("Done. Metadata:", OPACITY_META_CSV)
    print("Opacity shards in:", OPACITY_DIR)


if __name__ == "__main__":
    main()