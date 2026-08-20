import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from train import PlaneDatasetParamsToImageSharded, FNOPlusResNet

# ============= PATHS =============
base_dir = Path("/orcd/home/002/yuanxiuw/FNO_Large/plane_dataset_4")
shards_dir = base_dir / "renders"

image_csv  = base_dir / "renders" / "metadata_images_all_sharded.csv"
volume_csv = base_dir / "metadata_volumes.csv"

opacity_meta_csv   = base_dir / "metadata_opacity_from_images_sharded.csv"
opacity_shards_dir = base_dir / "opacity_from_images_shards"

ckpt_path = "fno_params_to_image_cameras_larger120_finetuned_finetuned.pt"

out_dir = base_dir / "opacity_model_preview"
out_dir.mkdir(parents=True, exist_ok=True)

# ============= DEVICE =============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============= LOAD DATASET FOR STATS + PARAM BUILDING =============
full_dataset = PlaneDatasetParamsToImageSharded(
    image_csv_path=str(image_csv),
    volume_csv_path=str(volume_csv),
    img_size=(64, 64),
    use_sh=True,
    normalize_params=True,
    shards_dir=str(shards_dir),
)
latent_dim = full_dataset.latent_dim
param_mean = full_dataset.param_mean
param_std  = full_dataset.param_std
shape_meta = full_dataset.shape_meta

# IMPORTANT: raw image metadata (unfiltered, original indices)
df_img_raw = pd.read_csv(image_csv)

# ============= LOAD RENDERER MODEL =============
model = FNOPlusResNet(latent_dim=latent_dim, img_size=(64, 64)).to(device)
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
state = ckpt["model_state"]
state.pop("_metadata", None)
model.load_state_dict(state)
model.eval()
print("Loaded renderer checkpoint:", ckpt_path)

# ============= LOAD OPACITY SHARDS =============
df_op = pd.read_csv(opacity_meta_csv)

shard_files = sorted(opacity_shards_dir.glob("opacity_64x64_from_images_shard_*.npy"))
opacity_shards = {}
for f in shard_files:
    sid = int(f.stem.split("_")[-1])  # ..._shard_00012.npy -> 12
    opacity_shards[sid] = np.load(f, mmap_mode="r")

print("Found opacity shards:", len(opacity_shards))


# ============= HELPERS =============

def build_param_vec_from_image_row(img_row):
    """
    Build normalized param vector exactly as PlaneDatasetParamsToImageSharded does.
    Uses full_dataset._build_param_vector_np but with raw row from df_img_raw.
    """
    scalars_np = full_dataset._build_param_vector_np(img_row)
    scalars_np = (scalars_np - param_mean) / param_std
    return torch.from_numpy(scalars_np).float()  # [latent_dim]


def load_rgb_gt_from_img_row(img_row):
    """
    Load ground-truth RGB [64,64,3] from shard file, using shard_id / idx_in_shard
    exactly like PlaneDatasetParamsToImageSharded.__init__ expects.
    """
    shard_id_str = str(img_row["shard_id"])
    idx_in_shard = int(img_row["idx_in_shard"])

    # Rebuild shard filename, mirroring PlaneDatasetParamsToImageSharded
    if "_" in shard_id_str:
        # NEW FORMAT: "job0_0" -> images_64x64_job0_shard_000.npy
        job, local = shard_id_str.split("_")
        local_id = int(local)
        shard_name = f"images_64x64_{job}_shard_{local_id:03d}.npy"
    else:
        # OLD FORMAT compatibility, if any
        old_id = int(shard_id_str)
        shard_name = f"images_64x64_shard_{old_id:02d}.npy"

    shard_path = shards_dir / shard_name
    if not shard_path.is_file():
        raise FileNotFoundError(f"Shard not found: {shard_path}")

    shard_arr = np.load(shard_path, mmap_mode="r")  # [N,3,H,W]
    img_np = shard_arr[idx_in_shard]  # [3,64,64]
    rgb = np.transpose(img_np, (1, 2, 0))  # [64,64,3]
    return rgb


# ============= VISUALIZE FEW SAMPLES =============

n_show = 20
indices = random.sample(range(len(df_op)), min(n_show, len(df_op)))

for idx in indices:
    row_op = df_op.iloc[idx]

    # opacity metadata
    shard_id_op     = int(row_op["shard_id"])
    idx_in_shard_op = int(row_op["idx_in_shard"])
    img_row_idx     = int(row_op["img_row_idx"])
    sample_id       = int(row_op["sample_id"])
    view_idx        = int(row_op.get("view_idx", 0))

    # Load opacity map
    if shard_id_op not in opacity_shards:
        print(f"[WARN] opacity shard_id {shard_id_op} not loaded; skipping")
        continue
    alpha = opacity_shards[shard_id_op][idx_in_shard_op]  # [64,64], float32

    # Corresponding image row from the ORIGINAL metadata CSV
    row_img = df_img_raw.iloc[img_row_idx]

    # Ground truth RGB from shards on disk
    try:
        rgb_gt = load_rgb_gt_from_img_row(row_img)  # [64,64,3]
    except FileNotFoundError as e:
        print("[WARN]", e)
        continue

    # Build param vector & render RGB from model
    with torch.no_grad():
        p_vec = build_param_vec_from_image_row(row_img).unsqueeze(0).to(device)  # [1,D]
        pred  = model(p_vec).clamp(0.0, 1.0)  # [1,3,64,64]

    pred_np = pred[0].cpu().numpy()  # [3,64,64]
    rgb_pred = np.transpose(pred_np, (1, 2, 0))  # [64,64,3]

    # Composite model RGB with opacity over white background
    a   = alpha[..., None]           # [64,64,1]
    bg  = np.ones_like(rgb_pred)     # white
    comp = a * rgb_pred + (1.0 - a) * bg
    comp = np.clip(comp, 0.0, 1.0)

    # Plot
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 3))

    # 1) GT RGB
    plt.subplot(1, 4, 1)
    plt.imshow(np.clip(rgb_gt, 0.0, 1.0))
    plt.axis("off")
    plt.title(f"GT s{sample_id:04d} v{view_idx:02d}")

    # 2) Model RGB
    plt.subplot(1, 4, 2)
    plt.imshow(rgb_pred, vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.title("Model RGB")

    # 3) Opacity α
    plt.subplot(1, 4, 3)
    im = plt.imshow(alpha, cmap="inferno", vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.title("Opacity α")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    # 4) Overlay
    plt.subplot(1, 4, 4)
    plt.imshow(comp, vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.title("Model RGB ⊗ α")

    out_path = out_dir / f"preview_s{sample_id:04d}_row{img_row_idx:06d}.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved preview:", out_path)