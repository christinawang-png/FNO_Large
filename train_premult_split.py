#!/usr/bin/env python
import os
from pathlib import Path
import math
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset

from neuralop.models import FNO  # make sure neuralop is installed

# ==============================
# Positional features
# ==============================

def make_fourier_pos_features(H, W, num_freqs=4, device="cpu"):
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # (H,W)

    feats = [xx, yy]
    for k in range(1, num_freqs + 1):
        feats.append(torch.sin(k * math.pi * xx))
        feats.append(torch.cos(k * math.pi * xx))
        feats.append(torch.sin(k * math.pi * yy))
        feats.append(torch.cos(k * math.pi * yy))
    grid = torch.stack(feats, dim=0)  # (C_pos, H, W)
    return grid

# ==============================
# Models
# ==============================

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return x + y

class ImageRefiner(nn.Module):
    def __init__(self, in_ch=4, hidden=32, num_blocks=3):
        super().__init__()
        self.entry = nn.Conv2d(in_ch, hidden, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(num_blocks)])
        self.exit  = nn.Conv2d(hidden, in_ch, 3, padding=1)

    def forward(self, x):
        h = self.entry(x)
        h = self.blocks(h)
        y = self.exit(h)
        out = x + y
        return torch.clamp(out, 0.0, 1.0)

class FNOPlusResNetSplit(nn.Module):
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

        self.fno_surface = FNO(
            n_modes=(32, 32),
            hidden_channels=96,
            in_channels=64,
            out_channels=4,
        )
        self.fno_volume = FNO(
            n_modes=(32, 32),
            hidden_channels=96,
            in_channels=64,
            out_channels=4,
        )

        self.surface_head = ImageRefiner(in_ch=4, hidden=32)
        self.volume_head  = ImageRefiner(in_ch=4, hidden=32)

    def forward(self, params, is_volume):
        B, D = params.shape
        device = params.device

        z_grid = params.view(B, D, 1, 1).expand(B, D, self.H, self.W)
        pos = self.pos_features.to(device).unsqueeze(0).expand(B, -1, -1, -1)
        field = torch.cat([z_grid, pos], dim=1)
        x = self.input_proj(field)

        is_vol = is_volume.view(B, 1, 1, 1).to(device)

        # run both branches, then pick per-sample
        coarse_surf = self.fno_surface(x)
        coarse_vol  = self.fno_volume(x)

        out_surf = self.surface_head(coarse_surf)
        out_vol  = self.volume_head(coarse_vol)

        out = torch.where(is_vol > 0.5, out_vol, out_surf)
        return torch.clamp(out, 0.0, 1.0)

# ==============================
# Dataset (RGB + alpha → premult RGBA)
# ==============================

class PlaneDatasetParamsToPremultRGBA(Dataset):
    """
    - Reads RGB metadata (metadata_images_all_sharded.csv) under renders/.
    - Reads merged alpha metadata (metadata_alpha_all.csv) under hard_alpha/.
    - Joins them via (shard_id, idx_in_shard).
    - Param vector similar to PlaneDatasetParamsToImageSharded, plus is_volume flag.
    - Target: [4,H,W] = [C_r,C_g,C_b,alpha] where C = alpha * rgb.
    """
    def __init__(self,
                 base_dir,
                 img_meta_csv=None,
                 vol_meta_csv=None,
                 renders_dir=None,
                 alpha_dir=None,
                 alpha_meta_csv=None,
                 img_size=(64, 64),
                 use_sh=True,
                 normalize_params=True):
        base_dir = Path(base_dir)

        if renders_dir is None:
            renders_dir = base_dir / "renders"
        else:
            renders_dir = Path(renders_dir)

        if alpha_dir is None:
            alpha_dir = base_dir / "hard_alpha"
        else:
            alpha_dir = Path(alpha_dir)

        if img_meta_csv is None:
            img_meta_csv = renders_dir / "metadata_images_all_sharded.csv"
        else:
            img_meta_csv = Path(img_meta_csv)

        if vol_meta_csv is None:
            vol_meta_csv = base_dir / "metadata_volumes.csv"
        else:
            vol_meta_csv = Path(vol_meta_csv)

        if alpha_meta_csv is None:
            alpha_meta_csv = alpha_dir / "metadata_alpha_all.csv"
        else:
            alpha_meta_csv = Path(alpha_meta_csv)

        self.img_size = img_size
        self.use_sh = use_sh
        self.normalize_params = normalize_params

        # ---- load image metadata (RGB) ----
        df_img = pd.read_csv(img_meta_csv, low_memory=False)
        df_img["shard_id"]     = df_img["shard_id"].astype(str)
        df_img["idx_in_shard"] = df_img["idx_in_shard"].astype(int)

        # ---- load alpha metadata from merged CSV ----
        df_alpha = pd.read_csv(alpha_meta_csv)
        df_alpha["img_shard_id"]     = df_alpha["img_shard_id"].astype(str)
        df_alpha["idx_in_img_shard"] = df_alpha["idx_in_img_shard"].astype(int)

        # Merge: alpha side gets "_alpha", image side keeps original names
        df_join = df_alpha.merge(
            df_img,
            left_on=["img_shard_id", "idx_in_img_shard"],
            right_on=["shard_id", "idx_in_shard"],
            suffixes=("_alpha", ""),  # IMPORTANT
        )

        # Normalize sample_id name if needed (usually sample_id already present)
        if "sample_id_img" in df_join.columns and "sample_id" not in df_join.columns:
            df_join.rename(columns={"sample_id_img": "sample_id"}, inplace=True)

        # Drop duplicate alpha-side sample_id if present
        if "sample_id_alpha" in df_join.columns:
            df_join.drop(columns=["sample_id_alpha"], inplace=True)

        self.df = df_join.reset_index(drop=True)

        # Sanity check: render_mode must exist
        if "render_mode" not in self.df.columns:
            raise RuntimeError("render_mode column missing after merge; check suffixes/metadata.")

        # ---- set up RGB shards ----
        self.rgb_shards = {}
        unique_img_sids = sorted(self.df["shard_id"].unique().tolist())
        for sid in unique_img_sids:
            if "_" in sid:
                job, local = sid.split("_")
                local_id = int(local)
                shard_name = f"images_64x64_{job}_shard_{local_id:03d}.npy"
            else:
                old_id = int(sid)
                shard_name = f"images_64x64_shard_{old_id:02d}.npy"
            shard_path = renders_dir / shard_name
            if shard_path.is_file():
                self.rgb_shards[sid] = np.load(shard_path, mmap_mode="r")
            else:
                print(f"[WARN] RGB shard missing: {shard_path}")

        # Filter rows to those with RGB shard present
        mask_rgb = self.df["shard_id"].isin(self.rgb_shards.keys())
        self.df = self.df[mask_rgb].reset_index(drop=True)

        # ---- set up alpha shards ----
        self.alpha_shards = {}
        unique_alpha_sids = sorted(self.df["alpha_shard_id"].unique().tolist())
        for sid in unique_alpha_sids:
            sid_str = str(sid)
            if "_" in sid_str:
                job, local = sid_str.split("_")
                local_id = int(local)
                shard_name = f"alpha_64x64_{job}_shard_{local_id:03d}.npy"
            else:
                local_id = int(sid_str)
                shard_name = f"alpha_64x64_shard_{local_id:03d}.npy"
            shard_path = alpha_dir / shard_name
            if shard_path.is_file():
                self.alpha_shards[sid_str] = np.load(shard_path, mmap_mode="r")
            else:
                print(f"[WARN] Alpha shard missing: {shard_path}")

        mask_alpha = self.df["alpha_shard_id"].astype(str).isin(self.alpha_shards.keys())
        self.df = self.df[mask_alpha].reset_index(drop=True)

        print("Using rows:", len(self.df))

        # ---- volume metadata ----
        df_vol = pd.read_csv(vol_meta_csv).set_index("sample_id")
        ctrl_cols = [c for c in df_vol.columns if c.startswith("ctrl_")]
        shape_cols = ctrl_cols + ["sigma"]
        self.shape_meta = df_vol[shape_cols].to_dict("index")
        self.ctrl_cols = ctrl_cols

        # ---- build param stats ----
        param_list = []
        for _, row in self.df.iterrows():
            param_list.append(self._build_param_vector_np(row))
        vals = np.stack(param_list, axis=0)
        self.latent_dim = vals.shape[1]

        if normalize_params:
            self.param_mean = vals.mean(axis=0)
            self.param_std  = vals.std(axis=0) + 1e-6
        else:
            self.param_mean = np.zeros(self.latent_dim, dtype=np.float32)
            self.param_std  = np.ones(self.latent_dim, dtype=np.float32)

        self.param_matrix = torch.empty((len(self.df), self.latent_dim), dtype=torch.float32)
        for i, row in self.df.iterrows():
            scalars_np = self._build_param_vector_np(row)
            scalars_np = (scalars_np - self.param_mean) / self.param_std
            self.param_matrix[i] = torch.from_numpy(scalars_np)

    def __len__(self):
        return len(self.df)

    def _build_param_vector_np(self, row):
        sid = int(row["sample_id"])
        shp = self.shape_meta[sid]

        ctrl_vals = [float(shp[c]) for c in self.ctrl_cols]
        sigma     = float(shp["sigma"])

        hue        = float(row["hue"])
        saturation = float(row["saturation"])
        metallic   = float(row["metallic"])
        roughness  = float(row["roughness"])
        opacity    = float(row["opacity"])
        specular   = float(row.get("specular", 0.5))

        phi        = float(row["phi"])
        theta      = float(row["theta"])
        radius     = float(row["radius"])
        sin_phi, cos_phi = math.sin(phi), math.cos(phi)
        sin_th,  cos_th  = math.sin(theta), math.cos(theta)

        mode = row["render_mode"] if "render_mode" in row.index else "surface"
        is_volume = 1.0 if mode == "volume" else 0.0

        # Optional: zero out surface-only params in volume mode
        if is_volume == 1.0:
            metallic  = 0.0
            roughness = 0.0
            specular  = 0.0

        scalars = (
            ctrl_vals
            + [sigma]
            + [hue, saturation, metallic, roughness, opacity, specular]
            + [sin_phi, cos_phi, sin_th, cos_th, radius]
        )

        if self.use_sh:
            for c in row.index:
                if c.startswith("sh_l") and c.endswith(("_r", "_g", "_b")):
                    scalars.append(float(row[c]))

        scalars.append(is_volume)

        return np.array(scalars, dtype=np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # RGB
        shard_id_str = str(row["shard_id"])
        idx_in_shard = int(row["idx_in_shard"])
        rgb_np = self.rgb_shards[shard_id_str][idx_in_shard]  # [3,H,W]

        # Alpha
        alpha_sid_str = str(row["alpha_shard_id"])
        idx_a = int(row["idx_in_alpha_shard"])
        alpha_np = self.alpha_shards[alpha_sid_str][idx_a]    # [H,W]

        H, W = self.img_size
        assert rgb_np.shape[1] == H and rgb_np.shape[2] == W

        alpha_ch = alpha_np[None, ...]                       # [1,H,W]
        C_np     = rgb_np * alpha_ch                         # [3,H,W]
        target   = np.concatenate([C_np, alpha_ch], axis=0)  # [4,H,W]

        target_t = torch.from_numpy(target.astype(np.float32))
        param_vec = self.param_matrix[idx]                   # [latent_dim]

        mode = row["render_mode"] if "render_mode" in row.index else "surface"
        is_volume = 1.0 if mode == "volume" else 0.0
        is_volume_t = torch.tensor(is_volume, dtype=torch.float32)

        return param_vec, target_t, is_volume_t
        

class ModeFilteredPremultDataset(Dataset):
    """
    Thin wrapper around PlaneDatasetParamsToPremultRGBA that filters
    by render_mode == "surface" or "volume".
    """
    def __init__(self, base_dataset, mode="surface"):
        assert mode in ("surface", "volume")
        self.base = base_dataset

        modes = self.base.df["render_mode"].values
        if mode == "surface":
            self.indices = np.where(modes == "surface")[0]
        else:
            self.indices = np.where(modes == "volume")[0]

        print(f"[ModeFilteredPremultDataset] mode={mode}, "
              f"keeping {len(self.indices)} / {len(self.base)} samples")

        self.mode = mode

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        base_idx = self.indices[i]
        param_vec, target_rgba, is_volume = self.base[base_idx]

        # For surface dataset, force is_volume=0; for volume, force is_volume=1
        if self.mode == "surface":
            is_volume = torch.tensor(0.0, dtype=torch.float32)
        else:
            is_volume = torch.tensor(1.0, dtype=torch.float32)

        return param_vec, target_rgba, is_volume

# ==============================
# Loss
# ==============================

def loss_fn(preds, targets, alpha_weight=2.0):
    # preds, targets: [B,4,H,W] = [Cr,Cg,Cb,alpha]
    C_pred, A_pred = preds[:, :3], preds[:, 3:4]
    C_gt,   A_gt   = targets[:, :3], targets[:, 3:4]

    loss_C = F.mse_loss(C_pred, C_gt)
    loss_A = F.mse_loss(A_pred, A_gt)
    return loss_C + alpha_weight * loss_A

# ==============================
# Training
# ==============================

def main():
    base_dir = Path("./plane_dataset_4")
    renders_dir = base_dir / "renders"
    alpha_dir   = base_dir / "hard_alpha"
    img_meta_csv = renders_dir / "metadata_images_all_sharded.csv"
    vol_meta_csv = base_dir / "metadata_volumes.csv"

    full_dataset = PlaneDatasetParamsToPremultRGBA(
        base_dir=base_dir,
        img_meta_csv=img_meta_csv,
        vol_meta_csv=vol_meta_csv,
        renders_dir=renders_dir,
        alpha_dir=alpha_dir,
        alpha_meta_csv=alpha_dir / "metadata_alpha_all.csv",
        img_size=(64, 64),
        use_sh=True,
        normalize_params=True,
    )

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

    latent_dim = full_dataset.latent_dim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FNOPlusResNetSplit(latent_dim=latent_dim, img_size=(64, 64)).to(device)
    print(f"Using device: {device}")

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=1024, shuffle=False,
                              num_workers=8, pin_memory=True, persistent_workers=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ---- optional resume from checkpoint ----
    resume_path = None  # or None, or pass via CLI
    start_epoch = 0

    if resume_path is not None and os.path.isfile(resume_path):
        print("Resuming from", resume_path)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)

        # load model weights
        state = ckpt.get("model_state", ckpt)
        # some PyTorch versions add '_metadata' key; it's safe to drop
        state.pop("_metadata", None)
        model.load_state_dict(state)

        # if you saved these, you can restore them or at least check
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"]
            print("  checkpoint epoch:", start_epoch)

        if "param_mean" in ckpt and "param_std" in ckpt:
            # optional: sanity-check against current dataset stats
            print("  (param_mean/param_std in checkpoint, current dataset stats will still be used)")

        print("Resume complete.")
    else:
        print("No resume checkpoint found or resume_path=None; starting from scratch.")

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_train = 0.0

        for param_vec, target_rgba, is_volume in train_loader:
            param_vec   = param_vec.to(device)
            target_rgba = target_rgba.to(device)
            is_volume   = is_volume.to(device)

            preds = model(param_vec, is_volume)
            loss = loss_fn(preds, target_rgba)

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
                is_volume   = is_volume.to(device)

                preds = model(param_vec, is_volume)
                loss = loss_fn(preds, target_rgba)
                total_val += loss.item() * param_vec.size(0)

        avg_val = total_val / len(val_dataset)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} "
                  f"train_loss={avg_train:.6f} val_loss={avg_val:.6f}")

        if (epoch + 1) % 5 == 0:
            ckpt_path = f"fno_premult_split_epoch{epoch+1:03d}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "param_mean": full_dataset.param_mean,
                "param_std": full_dataset.param_std,
                "latent_dim": latent_dim,
            }, ckpt_path)
            print("Saved checkpoint:", ckpt_path)

    # final save
    out_path = "fno_premult_split_final.pt"
    torch.save({
        "model_state": model.state_dict(),
        "param_mean": full_dataset.param_mean,
        "param_std": full_dataset.param_std,
        "latent_dim": latent_dim,
    }, out_path)
    print("Saved final model to", out_path)

if __name__ == "__main__":
    main()