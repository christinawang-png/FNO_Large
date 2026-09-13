#!/usr/bin/env python
import os
import sys
from pathlib import Path
import math

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from neuralop.models import FNO

from train_premult_split import (
    make_fourier_pos_features,
    ResBlock,
    ImageRefiner,
    PlaneDatasetParamsToPremultRGBA,
    ModeFilteredPremultDataset,
)

class FNOPlusResNetSingle(nn.Module):
    """
    Single-branch FNO+ResNet that predicts premultiplied RGBA [4,H,W].
    Used for either surface-only or volume-only training.
    """
    def __init__(self, latent_dim, img_size=(64, 64)):
        super().__init__()
        H, W = img_size
        self.H, self.W = H, W
        self.latent_dim = latent_dim

        self.register_buffer(
            "pos_features",
            make_fourier_pos_features(H, W, num_freqs=4)
        )
        C_pos = self.pos_features.shape[0]

        in_channels = latent_dim + C_pos
        self.input_proj = nn.Conv2d(in_channels, 64, 1)

        self.fno = FNO(
            n_modes=(32, 32),
            hidden_channels=96,
            in_channels=64,
            out_channels=4,
        )

        self.head = ImageRefiner(in_ch=4, hidden=32)

    def forward(self, params):
        B, D = params.shape
        device = params.device

        z_grid = params.view(B, D, 1, 1).expand(B, D, self.H, self.W)
        pos = self.pos_features.to(device).unsqueeze(0).expand(B, -1, -1, -1)
        field = torch.cat([z_grid, pos], dim=1)
        x = self.input_proj(field)

        coarse = self.fno(x)
        out    = self.head(coarse)
        return torch.clamp(out, 0.0, 1.0)

def loss_fn(preds, targets, alpha_weight=2.0):
    C_pred, A_pred = preds[:, :3], preds[:, 3:4]
    C_gt,   A_gt   = targets[:, :3], targets[:, 3:4]

    loss_C = F.mse_loss(C_pred, C_gt)
    loss_A = F.mse_loss(A_pred, A_gt)
    return loss_C + alpha_weight * loss_A

def main():
    # ---- parse mode from CLI ----
    mode = "surface"
    if "--mode" in sys.argv:
        mode_idx = sys.argv.index("--mode") + 1
        if mode_idx < len(sys.argv):
            mode = sys.argv[mode_idx]
    assert mode in ("surface", "volume"), f"mode must be 'surface' or 'volume', got {mode}"

    print(f"Training single-mode model for mode={mode}")
    
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")

    base_dir   = Path("./plane_dataset_4")
    renders_dir = base_dir / "renders"
    alpha_dir   = base_dir / "hard_alpha"

    img_meta_csv = renders_dir / "metadata_images_all_sharded.csv"
    vol_meta_csv = base_dir / "metadata_volumes.csv"
    alpha_meta_csv = alpha_dir / "metadata_alpha_all.csv"

    # full base dataset
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

    # filtered dataset: only surface or only volume
    full_dataset = ModeFilteredPremultDataset(base_dataset, mode=mode)

    N = len(full_dataset)
    val_frac = 0.1
    N_val = int(N * val_frac)
    N_train = N - N_val

    train_dataset, val_dataset = random_split(
        full_dataset,
        [N_train, N_val],
        generator=torch.Generator().manual_seed(42),
    )

    print("N_train:", len(train_dataset), "N_val:", len(val_dataset))

    latent_dim = base_dataset.latent_dim  # same for both
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FNOPlusResNetSingle(latent_dim=latent_dim, img_size=(64, 64)).to(device)
    print(f"Using device: {device}")

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=512, shuffle=False,
                              num_workers=8, pin_memory=True, persistent_workers=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # ---- optional resume from checkpoint ----
    resume_path = None  # e.g. "fno_premult_surface_epoch020.pt"
    start_epoch = 120

    if "--resume" in sys.argv:
        i = sys.argv.index("--resume") + 1
        if i < len(sys.argv):
            resume_path = sys.argv[i]

    if resume_path is not None and os.path.isfile(resume_path):
        print(f"[{mode}] Resuming from", resume_path)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)

        state = ckpt.get("model_state", ckpt)
        state.pop("_metadata", None)
        model.load_state_dict(state)

        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"]
            print("  checkpoint epoch:", start_epoch)

        # optional: sanity check mode
        if "mode" in ckpt and ckpt["mode"] != mode:
            print(f"  [WARN] checkpoint mode={ckpt['mode']} but current mode={mode}")

        print("Resume complete.")
    else:
        print(f"[{mode}] No resume checkpoint found or resume_path=None; starting from scratch.")

    num_epochs = 150
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_train = 0.0

        for param_vec, target_rgba, is_volume in train_loader:
            param_vec   = param_vec.to(device)
            target_rgba = target_rgba.to(device)

            preds = model(param_vec)
            loss  = loss_fn(preds, target_rgba)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train += loss.item() * param_vec.size(0)

        avg_train = total_train / len(train_dataset)

        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for param_vec, target_rgba, is_volume in val_loader:
                param_vec   = param_vec.to(device)
                target_rgba = target_rgba.to(device)

                preds = model(param_vec)
                loss  = loss_fn(preds, target_rgba)
                total_val += loss.item() * param_vec.size(0)

        avg_val = total_val / len(val_dataset)

        if (epoch + 1) % 2 == 0:
            print(f"[{mode}] Epoch {epoch+1}/{num_epochs} "
                  f"train_loss={avg_train:.6f} val_loss={avg_val:.6f}")

            ckpt_path = f"fno_premult_{mode}_epoch{epoch+1:03d}_color.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "param_mean": base_dataset.param_mean,
                "param_std": base_dataset.param_std,
                "latent_dim": latent_dim,
                "mode": mode,
            }, ckpt_path)
            print("Saved checkpoint:", ckpt_path)

    out_path = f"fno_premult_{mode}_color_final.pt"
    torch.save({
        "model_state": model.state_dict(),
        "param_mean": base_dataset.param_mean,
        "param_std": base_dataset.param_std,
        "latent_dim": latent_dim,
        "mode": mode,
    }, out_path)
    print("Saved final model to", out_path)

if __name__ == "__main__":
    main()