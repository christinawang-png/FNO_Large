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
import torch.optim as optim


class TransportDataset(Dataset):
    def __init__(self, X_path, Y_path, normalize_X=True):
        # X: [N, D_in], Y: [N, D_out]
        self.X = np.load(X_path, mmap_mode='r')
        self.Y = np.load(Y_path, mmap_mode='r')
        assert self.X.shape[0] == self.Y.shape[0]
        self.N, self.D_in = self.X.shape
        self.D_out = self.Y.shape[1]

        self.normalize_X = normalize_X
        if normalize_X:
            Xnp = self.X.astype(np.float32)
            self.X_mean = Xnp.mean(axis=0)
            self.X_std  = Xnp.std(axis=0) + 1e-6
        else:
            self.X_mean = np.zeros(self.D_in, dtype=np.float32)
            self.X_std  = np.ones(self.D_in, dtype=np.float32)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)
        if self.normalize_X:
            x = (x - self.X_mean) / self.X_std
        y = self.Y[idx].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


class TransportMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, num_layers=4):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.GELU())
            d = hidden
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
        

def make_sh_weights(order=2, w_l0=3.0, w_l1=2.0, w_l2=1.0, device="cpu"):
    """
    For order=2, returns weights of shape [27] matching flattened SH (9 coeffs * 3 channels).
    """
    # per-coefficient weights for one channel: [Y00, Y1-1, Y10, Y11, Y2-2,...,Y22]
    w_per_coeff = torch.tensor(
        [w_l0,              # l=0, m=0 (idx 0)
         w_l1, w_l1, w_l1,  # l=1, m=-1,0,1 (idx 1-3)
         w_l2, w_l2, w_l2, w_l2, w_l2],  # l=2, m=-2..2 (idx 4-8)
        dtype=torch.float32,
        device=device,
    )  # [9]

    # repeat for 3 channels (R,G,B): [9*3]=27
    w_all = w_per_coeff.repeat(3)  # [27]
    return w_all


def sh_weighted_mse(preds, targets, w):
    """
    preds, targets: [B, 27]
    w: [27] weights
    """
    diff2 = (preds - targets) ** 2  # [B,27]
    weighted = diff2 * w.unsqueeze(0)  # broadcast [1,27]
    return weighted.mean()


def main():
    base_dir = Path("./plane_dataset_4") / "transport_data"
    X_path = base_dir / "X_params_all.npy"
    Y_path = base_dir / "Y_sh_out_all.npy"

    full_dataset = TransportDataset(str(X_path), str(Y_path), normalize_X=True)
    N = len(full_dataset)
    print("Transport dataset size:", N)

    # train/val split
    val_frac = 0.1
    N_val = int(N * val_frac)
    N_train = N - N_val

    train_dataset, val_dataset = random_split(
        full_dataset,
        [N_train, N_val],
        generator=torch.Generator().manual_seed(42),
    )
    print("N_train:", len(train_dataset), "N_val:", len(val_dataset))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w_sh = make_sh_weights(order=2, w_l0=3.0, w_l1=2.0, w_l2=1.0, device=device)

    in_dim  = full_dataset.D_in
    out_dim = full_dataset.D_out
    model = TransportMLP(in_dim, out_dim, hidden=384, num_layers=4).to(device)

    batch_size = 1024
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 100

    for epoch in range(1, num_epochs+1):
        # ---- train ----
        model.train()
        total_train = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = sh_weighted_mse(pred, y, w_sh)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train += loss.item() * x.size(0)
        avg_train = total_train / len(train_dataset)

        # ---- val ----
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                loss = sh_weighted_mse(pred, y, w_sh)
                total_val += loss.item() * x.size(0)
        avg_val = total_val / len(val_dataset)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{num_epochs}, train_loss={avg_train:.6e}, val_loss={avg_val:.6e}")

    # save model + X normalization stats
    out_ckpt = base_dir / "transport_mlp.pt"
    torch.save({
        "model_state": model.state_dict(),
        "X_mean": full_dataset.X_mean,
        "X_std": full_dataset.X_std,
        "in_dim": in_dim,
        "out_dim": out_dim,
    }, out_ckpt)
    print("Saved transport model to", out_ckpt)


if __name__ == "__main__":
    from pathlib import Path
    main()