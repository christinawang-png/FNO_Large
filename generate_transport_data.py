#!/usr/bin/env python
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

# import your Dataset/model definitions from train.py
from train import FNOPlusResNet  # adjust if your module name differs


# ========= SH utilities =========

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
    N = len(df_img)
    # if you want a subset:
    # N = min(N, 200000)
    in_dim  = latent_dim
    sh_pairs = sh_lm_list(order=2)
    out_dim = len(sh_pairs) * 3  # 9 coeffs * 3 channels = 27

    X = np.empty((N, in_dim),  dtype=np.float32)
    Y = np.empty((N, out_dim), dtype=np.float32)

    batch_size = 1024
    H = W = 64

    with torch.no_grad():
        for i_start in range(0, N, batch_size):
            i_end = min(N, i_start + batch_size)
            rows = df_img.iloc[i_start:i_end]

            # build param vectors (unnormalized) on CPU
            param_list = []
            for _, row in rows.iterrows():
                p = build_param_vector(row, shape_meta, ctrl_cols, use_sh=True)
                param_list.append(p)
            params_np = np.stack(param_list, axis=0)  # [B, in_dim]

            # normalize using ckpt stats
            params_norm = (params_np - param_mean) / param_std

            params_t = torch.from_numpy(params_norm).to(device)  # [B, D]
            preds_t  = model(params_t)                            # [B,3,64,64]
            preds_t  = preds_t.clamp(0,1)

            preds_np = preds_t.cpu().numpy()  # [B,3,H,W]

            # project each pred to SH
            B = preds_np.shape[0]
            for j in range(B):
                sh_coeffs = project_image_to_sh(preds_np[j], order=2)  # (9,3)
                X[i_start + j] = params_norm[j]
                Y[i_start + j] = sh_coeffs.reshape(-1)  # 27

            print(f"[{i_end}/{N}] processed")

    # 4) Save transport dataset
    out_dir = base_dir / "transport_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X_params.npy", X)
    np.save(out_dir / "Y_sh_out.npy", Y)
    print("Saved X_params.npy and Y_sh_out.npy to", out_dir)


if __name__ == "__main__":
    main()