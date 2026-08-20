import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---- paths ----
base_dir   = Path("plane_dataset_4")
render_dir = base_dir / "renders"
alpha_dir  = base_dir / "hard_alpha"

job_id     = "job0"
shard_idx  = 800   # debug_shard_000

# RGB metadata + shard
rgb_csv   = render_dir / f"metadata_{job_id}_shard_{shard_idx:03d}.csv"
rgb_npy   = render_dir / f"images_64x64_{job_id}_shard_{shard_idx:03d}.npy"

# alpha metadata + shard
alpha_csv = alpha_dir / f"metadata_alpha_{job_id}_shard_{shard_idx:03d}.csv"
alpha_npy = alpha_dir / f"alpha_64x64_{job_id}_shard_{shard_idx:03d}.npy"

out_dir = Path("debug_previews")
out_dir.mkdir(parents=True, exist_ok=True)

df_rgb   = pd.read_csv(rgb_csv)
df_alpha = pd.read_csv(alpha_csv)

rgb_shard   = np.load(rgb_npy)    # [N,3,H,W]
alpha_shard = np.load(alpha_npy)  # [N,H,W]

print("RGB shard shape:", rgb_shard.shape)
print("Alpha shard shape:", alpha_shard.shape)

# ---- visualize & save a few samples ----
n_show = 100
indices = np.linspace(0, len(df_rgb)-1, num=min(n_show, len(df_rgb)), dtype=int)

for idx in indices:
    row_rgb   = df_rgb.iloc[idx]
    row_alpha = df_alpha.iloc[idx]

    # sanity: check alignment
    assert row_alpha["idx_in_img_shard"] == row_rgb["idx_in_shard"]
    i = int(row_rgb["idx_in_shard"])

    rgb   = rgb_shard[i]          # [3,H,W]
    alpha = alpha_shard[i]        # [H,W]

    rgb_hw = np.transpose(rgb, (1, 2, 0))     # [H,W,3]
    comp   = rgb_hw * alpha[..., None]        # simple premultiply

    plt.figure(figsize=(9, 3))
    plt.suptitle(
        f"s{int(row_rgb['sample_id']):04d} | mode={row_rgb['render_mode']} | idx={i}"
    )

    plt.subplot(1, 3, 1)
    plt.imshow(np.clip(rgb_hw, 0.0, 1.0))
    plt.axis("off")
    plt.title("RGB")

    plt.subplot(1, 3, 2)
    im = plt.imshow(alpha, cmap="inferno", vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.title("Alpha")

    plt.subplot(1, 3, 3)
    plt.imshow(np.clip(comp, 0.0, 1.0))
    plt.axis("off")
    plt.title("RGB × Alpha")

    fname = out_dir / f"preview_s{int(row_rgb['sample_id']):04d}_idx{i:04d}.png"
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved", fname)