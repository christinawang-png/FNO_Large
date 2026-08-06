import os
import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# import your Dataset and model from train.py or paste their definitions here
from train import PlaneDatasetParamsToImageSharded, FNOPlusResNet  # adjust if module name differs


def loss_fn_mse(preds, targets):
    return F.mse_loss(preds, targets)


# If you want mixed loss instead:
# def loss_fn_mixed(preds, targets):
#     return 0.5 * F.l1_loss(preds, targets) + 0.5 * F.mse_loss(preds, targets)


def main():
    base_dir   = Path("./plane_dataset_4")
    image_csv  = base_dir / "renders" / "metadata_images_all_sharded.csv"
    volume_csv = base_dir / "metadata_volumes.csv"
    shards_dir = base_dir / "renders"

    # ----- rebuild dataset and train/val split -----
    full_dataset = PlaneDatasetParamsToImageSharded(
        image_csv_path=str(image_csv),
        volume_csv_path=str(volume_csv),
        img_size=(64, 64),
        use_sh=True,
        normalize_params=True,
        shards_dir=str(shards_dir),
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

    # small-ish batch is enough for eval; can increase if you want
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    # choose loss
    criterion = loss_fn_mse  # or loss_fn_mixed

    # ----- find all checkpoints matching pattern -----
    ckpt_paths = glob.glob("fno_params_to_image_cameras_larger*.pt")
    # extract epoch numbers
    def get_epoch(path):
        # expects names like fno_params_to_image_cameras_larger120.pt
        base = os.path.basename(path)
        # take last 3 digits before .pt
        ep_str = base.split("larger")[-1].split(".pt")[0]
        try:
            return int(ep_str)
        except ValueError:
            return -1

    ckpt_paths = [(p, get_epoch(p)) for p in ckpt_paths]
    ckpt_paths = [p for p in ckpt_paths if p[1] >= 0]
    ckpt_paths.sort(key=lambda x: x[1])

    print("Found checkpoints:", [e for _, e in ckpt_paths])

    # ----- evaluate each checkpoint -----
    for ckpt_path, epoch in ckpt_paths:
        # rebuild model fresh each time
        model = FNOPlusResNet(latent_dim=latent_dim, img_size=(64, 64)).to(device)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt["model_state"]
        state.pop("_metadata", None)
        model.load_state_dict(state)
        model.eval()

        # eval train
        total_train = 0.0
        with torch.no_grad():
            for param_vec, images in train_loader:
                param_vec = param_vec.to(device)
                images    = images.to(device)
                preds     = model(param_vec)
                loss      = criterion(preds, images)
                total_train += loss.item() * param_vec.size(0)
        avg_train = total_train / len(train_dataset)

        # eval val
        total_val = 0.0
        with torch.no_grad():
            for param_vec, images in val_loader:
                param_vec = param_vec.to(device)
                images    = images.to(device)
                preds     = model(param_vec)
                loss      = criterion(preds, images)
                total_val += loss.item() * param_vec.size(0)
        avg_val = total_val / len(val_dataset)

        print(f"Epoch {epoch:3d}: train_loss={avg_train:.6f}, val_loss={avg_val:.6f}")


if __name__ == "__main__":
    main()