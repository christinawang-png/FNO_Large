#!/usr/bin/env python
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from train_premult_split import (
    PlaneDatasetParamsToPremultRGBA,
    FNOPlusResNetSplit,
)

def main():
    # ---- config ----
    base_dir   = Path("./plane_dataset_4")
    renders_dir = base_dir / "renders"
    alpha_dir   = base_dir / "hard_alpha"
    img_meta_csv = renders_dir / "metadata_images_all_sharded.csv"
    vol_meta_csv = base_dir / "metadata_volumes.csv"
    alpha_meta_csv = alpha_dir / "metadata_alpha_all.csv"

    ckpt_path = "fno_premult_split_epoch010.pt"  # change to your .pt
    out_dir   = base_dir / "vis_premult_split"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_show = 20  # how many examples to visualize

    # ---- dataset (no split, we just need some samples) ----
    dataset = PlaneDatasetParamsToPremultRGBA(
        base_dir=base_dir,
        img_meta_csv=img_meta_csv,
        vol_meta_csv=vol_meta_csv,
        renders_dir=renders_dir,
        alpha_dir=alpha_dir,
        alpha_meta_csv=alpha_meta_csv,
        img_size=(64, 64),
        use_sh=True,
        normalize_params=True,
    )

    latent_dim = dataset.latent_dim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- model ----
    model = FNOPlusResNetSplit(latent_dim=latent_dim, img_size=(64, 64)).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    state.pop("_metadata", None)
    model.load_state_dict(state)
    model.eval()
    print("Loaded checkpoint:", ckpt_path)

    # ---- sample indices ----
    n = len(dataset)
    indices = random.sample(range(n), min(n_show, n))

    with torch.no_grad():
        for idx in indices:
            param_vec, target_rgba, is_volume = dataset[idx]
            param_vec  = param_vec.unsqueeze(0).to(device)     # [1,D]
            target_rgba_t = target_rgba.unsqueeze(0).to(device)  # [1,4,H,W]
            is_volume_t = is_volume.unsqueeze(0).to(device)    # [1]

            pred_rgba = model(param_vec, is_volume_t)          # [1,4,H,W]
            pred_rgba = pred_rgba.squeeze(0).cpu().numpy()     # [4,H,W]
            tgt_rgba  = target_rgba.cpu().numpy()              # [4,H,W]

            # split into C and alpha
            C_gt   = tgt_rgba[:3]        # [3,H,W]
            A_gt   = tgt_rgba[3]         # [H,W]
            C_pred = pred_rgba[:3]
            A_pred = pred_rgba[3]

            # convert premultiplied to RGB over black and overlays
            rgb_gt   = np.transpose(C_gt,   (1, 2, 0))  # [H,W,3]
            rgb_pred = np.transpose(C_pred, (1, 2, 0))

            overlay_gt   = rgb_gt   # premultiplied over black
            overlay_pred = rgb_pred

            # make sure values in [0,1]
            rgb_gt        = np.clip(rgb_gt, 0.0, 1.0)
            rgb_pred      = np.clip(rgb_pred, 0.0, 1.0)
            overlay_gt    = np.clip(overlay_gt, 0.0, 1.0)
            overlay_pred  = np.clip(overlay_pred, 0.0, 1.0)
            A_gt_img      = np.clip(A_gt, 0.0, 1.0)
            A_pred_img    = np.clip(A_pred, 0.0, 1.0)

            # retrieve some metadata for naming
            row = dataset.df.iloc[idx]
            sample_id = int(row["sample_id"])
            mode = row.get("render_mode", "surface")

            # ---- plot 2x3 grid ----
            fig, axes = plt.subplots(2, 3, figsize=(9, 6))
            fig.suptitle(f"s{sample_id:04d} | mode={mode} | idx={idx}", fontsize=10)

            # Row 0: GT
            axes[0, 0].imshow(rgb_gt)
            axes[0, 0].set_title("GT RGB×α", fontsize=9)
            axes[0, 0].axis("off")

            axes[0, 1].imshow(A_gt_img, cmap="gray", vmin=0.0, vmax=1.0)
            axes[0, 1].set_title("GT α", fontsize=9)
            axes[0, 1].axis("off")

            axes[0, 2].imshow(overlay_gt)
            axes[0, 2].set_title("GT overlay", fontsize=9)
            axes[0, 2].axis("off")

            # Row 1: Pred
            axes[1, 0].imshow(rgb_pred)
            axes[1, 0].set_title("Pred RGB×α", fontsize=9)
            axes[1, 0].axis("off")

            axes[1, 1].imshow(A_pred_img, cmap="gray", vmin=0.0, vmax=1.0)
            axes[1, 1].set_title("Pred α", fontsize=9)
            axes[1, 1].axis("off")

            axes[1, 2].imshow(overlay_pred)
            axes[1, 2].set_title("Pred overlay", fontsize=9)
            axes[1, 2].axis("off")

            fig.tight_layout()
            out_path = out_dir / f"vis_s{sample_id:04d}_idx{idx:06d}.png"
            fig.savefig(out_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            print("Saved", out_path)

if __name__ == "__main__":
    main()