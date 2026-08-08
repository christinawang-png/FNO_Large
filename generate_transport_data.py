#!/usr/bin/env python
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F

# import your Dataset/model definitions from train.py
from train import FNOPlusResNet  # adjust if your module name differs


# ========= SH utilities =========

def precompute_sh_weights(H, W, order=2, device="cpu"):
    """
    Precompute Y_lm(direction) * area_weight for each pixel in an HxW grid.
    Returns Yw: [HW, num_coeffs] tensor on device.
    """
    ys = torch.linspace(0, H - 1, H, device=device)
    xs = torch.linspace(0, W - 1, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # [H,W]

    dtheta = math.pi / H
    dphi   = 2.0 * math.pi / W

    theta = (yy + 0.5) * dtheta
    phi   = (xx + 0.5) * dphi

    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    cos_phi   = torch.cos(phi)
    sin_phi   = torch.sin(phi)

    vx = sin_theta * cos_phi
    vy = sin_theta * sin_phi
    vz = cos_theta

    # real SH up to l=2, 9 coeffs
    # constants
    c0 = 0.28209479177387814
    c1 = 0.4886025119029199
    c2 = 1.0925484305920792
    c3 = 0.31539156525252005
    c4 = 0.5462742152960396

    # Y_lm per pixel, shape [H,W,9]
    Y = torch.empty((H, W, 9), device=device, dtype=torch.float32)
    Y[..., 0] = c0
    Y[..., 1] = -c1 * vy
    Y[..., 2] =  c1 * vz
    Y[..., 3] = -c1 * vx
    Y[..., 4] =  c2 * vx * vy
    Y[..., 5] = -c2 * vy * vz
    Y[..., 6] =  c3 * (3.0 * vz * vz - 1.0)
    Y[..., 7] = -c2 * vx * vz
    Y[..., 8] =  c4 * (vx * vx - vy * vy)

    area = sin_theta * dtheta * dphi  # [H,W]
    Yw = (Y * area[..., None]).reshape(H * W, 9)  # [HW,9]

    return Yw      # each row: Y_lm * dω for that pixel

def sh_lm_list(order):
    pairs = []
    for l in range(order + 1):
        for m in range(-l, l + 1):
            pairs.append((l, m))
    return pairs

def sh_basis_dir_l2(x, y, z):
    # real SH up to l=2, same as before
    c0 = 0.28209479177387814
    c1 = 0.4886025119029199
    c2 = 1.0925484305920792
    c3 = 0.31539156525252005
    c4 = 0.5462742152960396
    Y = np.empty(9, dtype=np.float32)
    Y[0] = c0
    Y[1] = -c1 * y
    Y[2] =  c1 * z
    Y[3] = -c1 * x
    Y[4] =  c2 * x * y
    Y[5] = -c2 * y * z
    Y[6] =  c3 * (3.0 * z*z - 1.0)
    Y[7] = -c2 * x * z
    Y[8] =  c4 * (x*x - y*y)
    return Y

def project_image_to_sh(img_np, order=2):
    """
    img_np: [3,H,W] in [0,1], predicted image
    Treat it as a view-dependent 'env' around camera direction and
    project to SH (rough approximation, but consistent).
    Returns coeffs: (num_coeffs, 3)
    """
    C, H, W = img_np.shape
    pairs = sh_lm_list(order)
    num_coeffs = len(pairs)
    coeffs = np.zeros((num_coeffs, 3), dtype=np.float64)

    # simple mapping: interpret pixel (y,x) as a direction on the unit sphere
    # with a naive equirectangular mapping
    dtheta = math.pi / H
    dphi   = 2.0 * math.pi / W

    # convert to HxWx3
    img = np.transpose(img_np, (1, 2, 0))  # [H,W,3]

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

            L = img[y, x, :]  # [3]
            Y = sh_basis_dir_l2(vx, vy, vz)  # [9]

            weight = sin_theta * dtheta * dphi  # area element
            coeffs += (Y[:, None] * L[None, :] * weight)

    return coeffs.astype(np.float32)  # (num_coeffs,3)


def project_batch_to_sh(preds_t, Yw):
    """
    preds_t: [B,3,H,W] on same device as Yw
    Yw: [HW,9] precomputed weights
    Returns coeffs: [B, 9, 3]
    """
    B, C, H, W = preds_t.shape
    assert C == 3
    L = preds_t.reshape(B, C, H * W).permute(0, 2, 1)  # [B,HW,3]
    # coeffs[b,q,c] = sum_h L[b,h,c] * Yw[h,q]
    coeffs = torch.einsum("bhc,hq->bqc", L, Yw)        # [B,9,3]
    return coeffs


# ========= Param vector builder (matches your new generator) =========

def build_param_vector(row, shape_meta, ctrl_cols, use_sh=True):
    """
    row: pandas Series from metadata_images_all_combined.csv
    shape_meta: dict sample_id -> {ctrl_*, sigma}
    ctrl_cols: list of ctrl_* column names in sorted order
    """
    sid = int(row["sample_id"])
    shp = shape_meta[sid]

    # shape
    ctrl_vals = [float(shp[c]) for c in ctrl_cols]
    sigma     = float(shp["sigma"])

    # material
    hue        = float(row["hue"])
    saturation = float(row["saturation"])
    metallic   = float(row["metallic"])
    roughness  = float(row["roughness"])
    opacity    = float(row["opacity"])
    specular   = float(row["specular"])

    # camera
    phi        = float(row["phi"])
    theta      = float(row["theta"])
    radius     = float(row["radius"])
    sin_phi, cos_phi = math.sin(phi), math.cos(phi)
    sin_th,  cos_th  = math.sin(theta), math.cos(theta)

    scalars = (
        ctrl_vals
        + [sigma]
        + [hue, saturation, metallic, roughness, opacity, specular]
        + [sin_phi, cos_phi, sin_th, cos_th, radius]
    )

    if use_sh:
        for col in row.index:
            if col.startswith("sh_l") and col.endswith(("_r", "_g", "_b")):
                scalars.append(float(row[col]))

    return np.array(scalars, dtype=np.float32)


# ========= Main generation =========

def main():
    base_dir   = Path("./plane_dataset_4")
    image_csv  = base_dir / "renders" / "metadata_images_all_sharded.csv"
    volume_csv = base_dir / "metadata_volumes.csv"
    ckpt_path  = Path("fno_params_to_image_cameras_larger120_finetuned_finetuned.pt")  # adjust to your best ckpt

    # 1) Load metadata
    df_img = pd.read_csv(image_csv, low_memory=False)
    df_vol = pd.read_csv(volume_csv).set_index("sample_id")

    # shape metadata: ctrl_* + sigma
    ctrl_cols = [c for c in df_vol.columns if c.startswith("ctrl_")]
    shape_cols = ctrl_cols + ["sigma"]
    shape_meta = df_vol[shape_cols].to_dict("index")

    print("Rows in image CSV:", len(df_img))
    print("Control cols:", ctrl_cols)

    # 2) Load renderer checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    latent_dim = ckpt["latent_dim"]
    param_mean = ckpt["param_mean"]  # np.array
    param_std  = ckpt["param_std"]

    model = FNOPlusResNet(latent_dim=latent_dim, img_size=(64,64)).to(device)
    state = ckpt["model_state"]
    state.pop("_metadata", None)
    model.load_state_dict(state)
    model.eval()
    print("Loaded renderer from", ckpt_path)

    # 3) Prepare storage (option: subset if you don’t want everything)
    SHARD_SIZE_SAMPLES = 10000  # or 100k, tune as you like
    H = W = 64
    Yw = precompute_sh_weights(H, W, order=2, device=device)
    N = len(df_img)
    in_dim  = latent_dim
    sh_pairs = sh_lm_list(order=2)
    out_dim = len(sh_pairs) * 3  # 27

    out_dir = base_dir / "transport_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_id = 0
    shard_count = 0
    X_shard = np.empty((SHARD_SIZE_SAMPLES, in_dim),  dtype=np.float32)
    Y_shard = np.empty((SHARD_SIZE_SAMPLES, out_dim), dtype=np.float32)

    sample_idx = 0  # global sample counter
    batch_size = 1024

    with torch.no_grad():
        for i_start in range(0, N, batch_size):
            i_end = min(N, i_start + batch_size)
            rows = df_img.iloc[i_start:i_end]

            # build and normalize params_norm as before
            param_list = []
            for _, row in rows.iterrows():
                p = build_param_vector(row, shape_meta, ctrl_cols, use_sh=True)
                param_list.append(p)
            params_np = np.stack(param_list, axis=0)
            params_norm = (params_np - param_mean) / param_std

            params_t = torch.from_numpy(params_norm).to(device)
            preds_t  = model(params_t).clamp(0,1)          # [B,3,64,64]

            # batched SH projection on GPU
            sh_coeffs_t = project_batch_to_sh(preds_t, Yw)  # [B,9,3]
            sh_coeffs_np = sh_coeffs_t.cpu().numpy().reshape(-1, out_dim)  # [B,27]

            B = sh_coeffs_np.shape[0]
            for j in range(B):
                X_shard[shard_count] = params_norm[j]
                Y_shard[shard_count] = sh_coeffs_np[j]
                shard_count += 1
                sample_idx += 1

                # flush full shard
                if shard_count == SHARD_SIZE_SAMPLES:
                    X_path = out_dir / f"X_params_shard_{shard_id:03d}.npy"
                    Y_path = out_dir / f"Y_sh_out_shard_{shard_id:03d}.npy"
                    np.save(X_path, X_shard)
                    np.save(Y_path, Y_shard)
                    print(f"Saved shard {shard_id} with {shard_count} samples to", X_path, Y_path)

                    shard_id += 1
                    shard_count = 0

            print(f"[{i_end}/{N}] processed, total samples so far: {sample_idx}")

    # save final partial shard, if any
    if shard_count > 0:
        X_path = out_dir / f"X_params_shard_{shard_id:03d}.npy"
        Y_path = out_dir / f"Y_sh_out_shard_{shard_id:03d}.npy"
        np.save(X_path, X_shard[:shard_count])
        np.save(Y_path, Y_shard[:shard_count])
        print(f"Saved final shard {shard_id} with {shard_count} samples to", X_path, Y_path)


if __name__ == "__main__":
    main()