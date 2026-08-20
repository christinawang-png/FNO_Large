# test_wow_params.py
import math
import numpy as np
import pandas as pd
import torch

from pathlib import Path

# import Dataset and wow helpers
from train import PlaneDatasetParamsToImageSharded  # adjust module name if needed
from wow_video import sh_lm_list, build_param_vec  # import from your wow script

base_dir   = Path("./plane_dataset_4")
image_csv  = base_dir / "renders" / "metadata_images_all_sharded.csv"
volume_csv = base_dir / "metadata_volumes.csv"
shards_dir = base_dir / "renders"

# 1) Build dataset (same as training)
dataset = PlaneDatasetParamsToImageSharded(
    image_csv_path=str(image_csv),
    volume_csv_path=str(volume_csv),
    img_size=(64, 64),
    use_sh=True,
    normalize_params=True,
    shards_dir=str(shards_dir),
)

print("latent_dim from dataset:", dataset.latent_dim)

# 2) Pick an index to test (e.g. one with hue ~0.9)
df_img = dataset.df_img  # underlying DataFrame used by the dataset

mask = (df_img["hue"] > 0.85) & (df_img["hue"] < 0.95)
if not mask.any():
    print("No hue in [0.85,0.95] found; using idx=0")
    idx = 0
else:
    idx = df_img[mask].index[0]

print("Testing idx:", idx)
row = df_img.iloc[idx]
sample_id = int(row["sample_id"])
print("sample_id:", sample_id, "hue:", row["hue"])

# 3) Get param_vec from dataset (ground truth)
param_vec_ds, img_gt = dataset[idx]   # param_vec_ds: [latent_dim]

# 4) Rebuild param vector via wow's build_param_vec for the SAME row

# --- shape params: control grid + sigma ---
shp = dataset.shape_meta[sample_id]         # dict with ctrl_* and sigma
ctrl_cols = dataset.ctrl_cols              # you stored this in __init__
ctrl_vals = [float(shp[c]) for c in ctrl_cols]
sigma     = float(shp["sigma"])

# --- material params ---
hue        = float(row["hue"])
saturation = float(row["saturation"])
metallic   = float(row["metallic"])
roughness  = float(row["roughness"])
opacity    = float(row["opacity"])
specular   = float(row["specular"])

# --- camera params ---
phi        = float(row["phi"])
theta      = float(row["theta"])
radius     = float(row["radius"])

# --- SH coeffs from row ---
pairs = sh_lm_list(order=2)
sh_coeffs = []
for (l, m) in pairs:
    r_c = float(row[f"sh_l{l}_m{m}_r"])
    g_c = float(row[f"sh_l{l}_m{m}_g"])
    b_c = float(row[f"sh_l{l}_m{m}_b"])
    sh_coeffs.append([r_c, g_c, b_c])
sh_coeffs = np.array(sh_coeffs, dtype=np.float32)  # (num_coeffs,3)

# 5) Build wow-style param vector (unnormalized inside build_param_vec)
param_vec_wow = build_param_vec(
    ctrl_vals, sigma,
    hue, saturation, metallic, roughness, opacity, specular,
    phi, theta, radius,
    sh_coeffs,
    dataset,        # so build_param_vec can use dataset.param_mean/std
)  # [latent_dim] tensor

# 6) Compare the two vectors
print("param_vec_ds shape:", param_vec_ds.shape)
print("param_vec_wow shape:", param_vec_wow.shape)

diff = (param_vec_ds - param_vec_wow).abs()
print("max abs diff:", diff.max().item())
print("mean abs diff:", diff.mean().item())

# 7) Optionally, print a few entries to see where they differ
for i in range(diff.numel()):
    print(f"i={i}: ds={param_vec_ds[i].item():.5f}, wow={param_vec_wow[i].item():.5f}, diff={diff[i].item():.5e}")