import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
import piq

# adjust this import to wherever you defined these
from train import PlaneDatasetParamsToImageSharded, FNOPlusResNet  

def color_sensitive_loss(preds, targets, alpha_luma=0.3, beta_chroma=0.7):
    """
    preds, targets: [B,3,H,W] in [0,1]
    alpha_luma, beta_chroma: weights for luma vs chroma; beta > alpha
    """
    # Luminance
    w = preds.new_tensor([0.299, 0.587, 0.114]).view(1,3,1,1)
    Y_pred = (preds * w).sum(dim=1, keepdim=True)   # [B,1,H,W]
    Y_gt   = (targets * w).sum(dim=1, keepdim=True)

    # Chroma (color component)
    C_pred = preds - Y_pred      # [B,3,H,W]
    C_gt   = targets - Y_gt

    loss_luma   = F.mse_loss(Y_pred, Y_gt)
    loss_chroma = F.mse_loss(C_pred, C_gt)

    return alpha_luma * loss_luma + beta_chroma * loss_chroma

def loss_fn_color(preds, targets):
    base = 0.5 * F.mse_loss(preds, targets) + 0.5 * F.l1_loss(preds, targets)
    color = color_sensitive_loss(preds, targets, alpha_luma=0.3, beta_chroma=0.7)
    return 0.3 * base + 0.7 * color

def loss_fn(preds, targets):
    # mixed L1 + MSE
    return 0.5 * F.l1_loss(preds, targets) + 0.5 * F.mse_loss(preds, targets)

def main():
    # -------- paths & settings --------
    base_dir   = Path("./plane_dataset_4")
    image_csv  = base_dir / "renders" / "metadata_images_all_sharded.csv"
    volume_csv = base_dir / "metadata_volumes.csv"
    shards_dir = base_dir / "renders"

    checkpoint_path = Path("fno_params_to_image_cameras_larger120_finetuned.pt")  # your MSE-trained checkpoint
    finetune_epochs = 30
    batch_size = 1024
    val_frac = 0.1
    lr = 1e-4   # lower LR for finetune

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------- dataset & split --------
    full_dataset = PlaneDatasetParamsToImageSharded(
        image_csv_path=str(image_csv),
        volume_csv_path=str(volume_csv),
        img_size=(64,64),
        use_sh=True,
        normalize_params=True,
        shards_dir=str(shards_dir),  # wherever you saved images_64x64_shard_*.npy
    )

    latent_dim = full_dataset.latent_dim
    N = len(full_dataset)
    N_val = int(N * val_frac)
    N_train = N - N_val

    train_dataset, val_dataset = random_split(
        full_dataset,
        [N_train, N_val],
        generator=torch.Generator().manual_seed(42),
    )
    print("N_train:", len(train_dataset), "N_val:", len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=4)

    # -------- model & checkpoint load --------
    model = FNOPlusResNet(latent_dim=latent_dim, img_size=(64, 64)).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state = ckpt["model_state"]
    # remove problematic key if present
    state.pop("_metadata", None)

    model.load_state_dict(state)
    print("Loaded checkpoint from", checkpoint_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -------- finetuning loop --------
    for epoch in range(finetune_epochs):
        # train
        model.train()
        total_train = 0.0
        for param_vec, images in train_loader:
            param_vec = param_vec.to(device)
            images    = images.to(device)

            preds = model(param_vec)
            loss  = loss_fn(preds, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train += loss.item() * param_vec.size(0)
        avg_train = total_train / len(train_dataset)

        # val
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for param_vec, images in val_loader:
                param_vec = param_vec.to(device)
                images    = images.to(device)

                preds = model(param_vec)
                loss  = loss_fn(preds, images)
                total_val += loss.item() * param_vec.size(0)
        avg_val = total_val / len(val_dataset)

        if (epoch + 1) % 5 == 0:
            print(f"[Finetune] Epoch {epoch+1}/{finetune_epochs}, "
                  f"train_loss={avg_train:.6f}, val_loss={avg_val:.6f}")

    # -------- qualitative eval on train & val --------
    model.eval()
    out_vis_dir = Path("qualitative_preds_finetune")
    train_vis_dir = out_vis_dir / "train"
    val_vis_dir   = out_vis_dir / "val"
    train_vis_dir.mkdir(parents=True, exist_ok=True)
    val_vis_dir.mkdir(parents=True, exist_ok=True)

    n_show = 8  # more examples than before

    # random train samples
    train_indices = random.sample(range(len(train_dataset)), min(n_show, len(train_dataset)))
    with torch.no_grad():
        for idx in train_indices:
            param_vec, img_gt = train_dataset[idx]
            param_vec = param_vec.unsqueeze(0).to(device)  # [1,D]
            img_gt    = img_gt.unsqueeze(0).to(device)     # [1,3,H,W]

            img_pred = model(param_vec).clamp(0, 1)

            sbs = torch.cat([img_gt, img_pred], dim=3)
            fname = train_vis_dir / f"train_{idx:06d}.png"
            save_image(sbs, fname)
            print("Saved finetune train example:", fname)

    # random val samples
    val_indices = random.sample(range(len(val_dataset)), min(n_show, len(val_dataset)))
    with torch.no_grad():
        for idx in val_indices:
            param_vec, img_gt = val_dataset[idx]
            param_vec = param_vec.unsqueeze(0).to(device)
            img_gt    = img_gt.unsqueeze(0).to(device)

            img_pred = model(param_vec).clamp(0, 1)

            sbs = torch.cat([img_gt, img_pred], dim=3)
            fname = val_vis_dir / f"val_{idx:06d}.png"
            save_image(sbs, fname)
            print("Saved finetune val example:", fname)

    # -------- save finetuned model --------
    out_ckpt = "fno_params_to_image_cameras_larger120_finetuned_finetuned.pt"
    torch.save({
        "model_state": model.state_dict(),
        "param_mean":  full_dataset.param_mean,
        "param_std":   full_dataset.param_std,
        "latent_dim":  latent_dim,
    }, out_ckpt)
    print("Saved finetuned model to", out_ckpt)


if __name__ == "__main__":
    main()