#!/usr/bin/env python
import sys
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from train_premult_single_mode import (
    FNOPlusResNetSingle,
    PlaneDatasetParamsToPremultRGBA,
    ModeFilteredPremultDataset,
)

def main():
    # ---- parse args ----
    mode = "surface"
    ckpt_path = "fno_premult_surface_epoch128_color.pt"
    if "--mode" in sys.argv:
        i = sys.argv.index("--mode") + 1
        if i < len(sys.argv):
            mode = sys.argv[i]
    if "--ckpt" in sys.argv:
        i = sys.argv.index("--ckpt") + 1
        if i < len(sys.argv):
            ckpt_path = sys.argv[i]

    assert mode in ("surface", "volume"), f"mode must be 'surface' or 'volume', got {mode}"
    ckpt_path = Path(ckpt_path)

    # ---- paths ----
    base_dir   = Path("./plane_dataset_4")
    renders_dir = base_dir / "renders"
    alpha_dir   = base_dir / "hard_alpha"
    img_meta_csv   = renders_dir / "metadata_images_all_sharded.csv"
    vol_meta_csv   = base_dir / "metadata_volumes.csv"
    alpha_meta_csv = alpha_dir / "metadata_alpha_all.csv"

    out_dir = base_dir / "vis_premult_single_mode"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Visualizing mode={mode} from checkpoint={ckpt_path}")

    # ---- dataset ----
    base_dataset = PlaneDatasetParamsToPremultRGBA(
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
    dataset = ModeFilteredPremultDataset(base_dataset, mode=mode)

    latent_dim = base_dataset.latent_dim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- model ----
    model = FNOPlusResNetSingle(latent_dim=latent_dim, img_size=(64, 64)).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    state.pop("_metadata", None)
    model.load_state_dict(state)
    model.eval()
    print("Loaded checkpoint:", ckpt_path)

    # ---- sample indices ----
    n_show = 20
    n = len(dataset)
    indices = random.sample(range(n), min(n_show, n))

    with torch.no_grad():
        for local_idx in indices:
            param_vec, target_rgba, _ = dataset[local_idx]  # is_volume is fixed in wrapper

            # map local index back to base_dataset index for metadata
            base_idx = dataset.indices[local_idx]
            row = base_dataset.df.iloc[base_idx]
            sample_id = int(row["sample_id"])
            mode_str  = row.get("render_mode", mode)

            param_vec_t  = param_vec.unsqueeze(0).to(device)     # [1,D]
            target_rgba_t = target_rgba.unsqueeze(0).to(device)  # [1,4,H,W]

            pred_rgba = model(param_vec_t)                       # [1,4,H,W]
            pred_rgba = pred_rgba.squeeze(0).cpu().numpy()       # [4,H,W]
            tgt_rgba  = target_rgba.cpu().numpy()                # [4,H,W]

            # split into C and alpha
            C_gt   = tgt_rgba[:3]        # [3,H,W]
            A_gt   = tgt_rgba[3]         # [H,W]
            C_pred = pred_rgba[:3]
            A_pred = pred_rgba[3]

            # convert premultiplied to RGB over black
            rgb_gt   = np.transpose(C_gt,   (1, 2, 0))  # [H,W,3]
            rgb_pred = np.transpose(C_pred, (1, 2, 0))

            overlay_gt   = rgb_gt
            overlay_pred = rgb_pred

            # clip to [0,1]
            rgb_gt        = np.clip(rgb_gt, 0.0, 1.0)
            rgb_pred      = np.clip(rgb_pred, 0.0, 1.0)
            overlay_gt    = np.clip(overlay_gt, 0.0, 1.0)
            overlay_pred  = np.clip(overlay_pred, 0.0, 1.0)
            A_gt_img      = np.clip(A_gt, 0.0, 1.0)
            A_pred_img    = np.clip(A_pred, 0.0, 1.0)

            # ---- plot 2x3 grid ----
            fig, axes = plt.subplots(2, 3, figsize=(9, 6))
            fig.suptitle(f"s{sample_id:04d} | mode={mode_str} | base_idx={base_idx}", fontsize=10)

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
            out_path = out_dir / f"vis_{mode}_s{sample_id:04d}_base{base_idx:06d}.png"
            fig.savefig(out_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            print("Saved", out_path)

if __name__ == "__main__":
    main()