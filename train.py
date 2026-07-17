from pathlib import Path

import math

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from neuralop.models import FNO  # must be installed

from torchvision.utils import save_image
import random

from torch.utils.data import Dataset
import torch.nn.functional as F
import os


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
# Model
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
    def __init__(self, in_ch=3, hidden=32, num_blocks=3):
        super().__init__()
        self.entry = nn.Conv2d(in_ch, hidden, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(num_blocks)])
        self.exit  = nn.Conv2d(hidden, in_ch, 3, padding=1)

    def forward(self, x):
        h = self.entry(x)
        h = self.blocks(h)
        y = self.exit(h)
        out = x + y
        return torch.sigmoid(out)


class FNOPlusResNet(nn.Module):
    def __init__(self, latent_dim, img_size=(64, 64)):
        super().__init__()
        H, W = img_size
        self.H, self.W = H, W
        self.latent_dim = latent_dim

        # 2D positional encodings
        self.register_buffer(
            "pos_features",
            make_fourier_pos_features(H, W, num_freqs=4)
        )
        C_pos = self.pos_features.shape[0]

        in_channels = latent_dim + C_pos
        self.input_proj = nn.Conv2d(in_channels, 64, 1)

        # FNO backbone
        self.fno = FNO(
            n_modes=(40, 40),
            hidden_channels=128,
            in_channels=64,
            out_channels=3
        )

        # Image-space refiner
        self.refiner = ImageRefiner(in_ch=3, hidden=64)

    def forward(self, params):
        B, D = params.shape
        device = params.device

        z_grid = params.view(B, D, 1, 1).expand(B, D, self.H, self.W)
        pos = self.pos_features.to(device).unsqueeze(0).expand(B, -1, -1, -1)

        field = torch.cat([z_grid, pos], dim=1)
        x = self.input_proj(field)

        coarse = self.fno(x)        # [B,3,H,W]
        out = self.refiner(coarse)
        return out


# ==============================
# Dataset
# ==============================


class PlaneDatasetParamsToImageSharded(Dataset):
    def __init__(self,
                 image_csv_path,
                 volume_csv_path,
                 img_size=(64, 64),
                 use_sh=True,
                 normalize_params=True,
                 shards_dir=None):
        self.df_img = pd.read_csv(image_csv_path, low_memory=False)
        self.img_size = img_size
        self.use_sh = use_sh
        self.normalize_params = normalize_params

        # ----- volume metadata as before -----
        df_vol = pd.read_csv(volume_csv_path).set_index("sample_id")
        self.shape_meta = df_vol[["p1", "p2", "sigma"]].to_dict("index")

        # ----- build param stats (same as your non-sharded dataset) -----
        param_list = []
        for _, row in self.df_img.iterrows():
            param_list.append(self._build_param_vector_np(row))
        vals = np.stack(param_list, axis=0)
        self.latent_dim = vals.shape[1]
        if normalize_params:
            self.param_mean = vals.mean(axis=0)
            self.param_std  = vals.std(axis=0) + 1e-6
        else:
            self.param_mean = np.zeros(self.latent_dim, dtype=np.float32)
            self.param_std  = np.ones(self.latent_dim, dtype=np.float32)

        # ----- load shards (old + new), keyed by string shard_id -----
        if shards_dir is None:
            shards_dir = Path(image_csv_path).parent.parent  # BASE_DIR
        else:
            shards_dir = Path(shards_dir)

        # ensure shard_id is string
        self.df_img["shard_id"] = self.df_img["shard_id"].astype(str)

        self.shards = {}  # key: shard_id string
        unique_ids = sorted(self.df_img["shard_id"].unique().tolist())

        for sid in unique_ids:
            if "_" in sid:
                # NEW FORMAT: "job0_0" -> images_64x64_job0_shard_000.npy
                job, local = sid.split("_")
                local_id = int(local)
                shard_name = f"images_64x64_{job}_shard_{local_id:03d}.npy"
            else:
                # OLD FORMAT: "0", "1", ... -> images_64x64_shard_00.npy (or similar)
                old_id = int(sid)
                shard_name = f"images_64x64_shard_{old_id:02d}.npy"  # adjust digits if needed

            shard_path = shards_dir / shard_name
            if not shard_path.is_file():
                print(f"[WARN] shard file not found for shard_id={sid} -> {shard_path}")
                continue

            self.shards[sid] = np.load(shard_path, mmap_mode="r")

        # Optionally drop rows whose shard_id we couldn't load
        mask = self.df_img["shard_id"].isin(self.shards.keys())
        self.df_img = self.df_img[mask].reset_index(drop=True)
        print("Using rows:", len(self.df_img), "unique shard_ids:", len(self.shards))

    def __len__(self):
        return len(self.df_img)

    def _build_param_vector_np(self, row):
        # same as before: use sample_id, hue, saturation, metallic, roughness, opacity, phi/theta/radius, SH, etc.
        sid = int(row["sample_id"])
        shp = self.shape_meta[sid]
        p1 = float(shp["p1"]); p2 = float(shp["p2"]); sigma = float(shp["sigma"])
        hue        = float(row["hue"])
        saturation = float(row["saturation"])
        metallic   = float(row["metallic"])
        roughness  = float(row["roughness"])
        opacity    = float(row["opacity"])
        specular   = float(row["specular"])
        phi        = float(row["phi"])
        theta      = float(row["theta"])
        radius     = float(row["radius"])
        sin_phi, cos_phi = math.sin(phi), math.cos(phi)
        sin_th, cos_th   = math.sin(theta), math.cos(theta)

        scalars = [
            p1, p2, sigma,
            hue, saturation, metallic, roughness, opacity, specular,
            sin_phi, cos_phi, sin_th, cos_th,
            radius,
        ]
        if self.use_sh:
            for col in self.df_img.columns:
                if col.startswith("sh_l") and col.endswith(("_r", "_g", "_b")):
                    scalars.append(float(row[col]))
        return np.array(scalars, dtype=np.float32)

    def __getitem__(self, idx):
        row = self.df_img.iloc[idx]

        sid_str = str(row["shard_id"])      # "job0_0" or "0"
        local_i = int(row["idx_in_shard"])
        img_np  = self.shards[sid_str][local_i]   # [3,H,W]

        img = torch.from_numpy(img_np)

        scalars_np = self._build_param_vector_np(row)
        scalars_np = (scalars_np - self.param_mean) / self.param_std
        param_vec  = torch.from_numpy(scalars_np)

        return param_vec, img
    

def loss_fn(preds, targets):
    return 0.5 * F.l1_loss(preds, targets) + 0.5 * F.mse_loss(preds, targets)


# ==============================
# Training
# ==============================

def main():
    base_dir = Path("./plane_dataset_3")
    image_csv = base_dir / "renders" / "metadata_images_all_combined.csv"   # or shard
    volume_csv = base_dir / "metadata_volumes.csv"

    full_dataset = PlaneDatasetParamsToImageSharded(
        image_csv_path=str(image_csv),
        volume_csv_path=str(volume_csv),
        img_size=(64,64),
        use_sh=True,
        normalize_params=True,
        shards_dir=str(base_dir),  # wherever you saved images_64x64_shard_*.npy
    )

    N = len(full_dataset)
    val_frac = 0.1   # 10% for validation
    N_val = int(N * val_frac)
    N_train = N - N_val

    train_dataset, val_dataset = random_split(
        full_dataset,
        [N_train, N_val],
        generator=torch.Generator().manual_seed(42),  # reproducible split
    )

    print("N_train:", len(train_dataset), "N_val:", len(val_dataset))

    latent_dim = full_dataset.latent_dim  # same for both

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FNOPlusResNet(latent_dim=latent_dim, img_size=(64, 64)).to(device)
    print(f"Using device: {device}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=2)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    resume_path = "fno_params_to_image_more_envs_30.pt"  # or None
    start_epoch = 0
    num_epochs  = 150  # total epochs you want to reach

    resume = False

    if resume_path is not None and os.path.isfile(resume_path) and resume:
        print("Resuming from", resume_path)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        state = ckpt["model_state"]
        state.pop("_metadata", None)  # if needed
        model.load_state_dict(state)
        print("Resumed at epoch", start_epoch)
        

    for epoch in range(start_epoch, num_epochs):
        # ---- train ----
        model.train()
        total_train = 0.0
        for param_vec, images in train_loader:
            param_vec = param_vec.to(device)
            images = images.to(device)

            preds = model(param_vec)
            loss = criterion(preds, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train += loss.item() * param_vec.size(0)

        avg_train = total_train / len(train_dataset)

        # ---- val ----
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for param_vec, images in val_loader:
                param_vec = param_vec.to(device)
                images = images.to(device)

                preds = model(param_vec)
                loss = criterion(preds, images)
                total_val += loss.item() * param_vec.size(0)

        avg_val = total_val / len(val_dataset)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, "
                f"train_loss={avg_train:.6f}, val_loss={avg_val:.6f}")
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} Saved.")
            # save checkpoint
            ckpt_path = f"fno_params_to_image_cameras_{epoch+1:03d}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "param_mean": full_dataset.param_mean,
                "param_std": full_dataset.param_std,
                "latent_dim": latent_dim,
            }, ckpt_path)
    # ==============================
    # Quick qualitative check: GT vs Pred
    # ==============================
    model.eval()
    out_vis_dir = Path("qualitative_preds")
    (out_vis_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_vis_dir / "val").mkdir(parents=True, exist_ok=True)

    n_show = 4

    # ---- random train examples ----
    train_indices = random.sample(range(len(train_dataset)), min(n_show, len(train_dataset)))
    with torch.no_grad():
        for i, idx in enumerate(train_indices):
            param_vec, img_gt = train_dataset[idx]
            param_vec = param_vec.unsqueeze(0).to(device)
            img_gt    = img_gt.unsqueeze(0).to(device)

            img_pred = model(param_vec).clamp(0, 1)

            sbs = torch.cat([img_gt, img_pred], dim=3)
            fname = out_vis_dir / "train" / f"train_{idx:06d}.png"
            save_image(sbs, fname)
            print("Saved train example:", fname)

    # ---- random val examples ----
    val_indices = random.sample(range(len(val_dataset)), min(n_show, len(val_dataset)))
    with torch.no_grad():
        for i, idx in enumerate(val_indices):
            param_vec, img_gt = val_dataset[idx]
            param_vec = param_vec.unsqueeze(0).to(device)
            img_gt    = img_gt.unsqueeze(0).to(device)

            img_pred = model(param_vec).clamp(0, 1)

            sbs = torch.cat([img_gt, img_pred], dim=3)
            fname = out_vis_dir / "val" / f"val_{idx:06d}.png"
            save_image(sbs, fname)
            print("Saved val example:", fname)

    # save model + normalization stats
    out_path = "fno_params_to_image_cameras.pt"
    torch.save({
        "model_state": model.state_dict(),
        "param_mean": full_dataset.param_mean,
        "param_std": full_dataset.param_std,
        "latent_dim": latent_dim,
    }, out_path)
    print("Saved model to", out_path)


if __name__ == "__main__":
    main()