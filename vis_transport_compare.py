import math
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from train import FNOPlusResNet       # renderer model
from train_faces import TransportMLP  # or paste the TransportMLP class here


# ----- SH utilities -----

def sh_lm_list(order):
    pairs = []
    for l in range(order + 1):
        for m in range(-l, l + 1):
            pairs.append((l, m))
    return pairs

def sh_basis_dir_l2(x, y, z):
    c0 = 0.28209479177387814
    c1 = 0.4886025119029199
    c2 = 1.0925484305920792
    c3 = 0.31539156525252005
    c4 = 0.5462742152960396
    Y = np.empty(9, dtype=np.float32)
    Y[0] = c0
    Y[1] = -c1 * y
    Y[2] =  c1 * z
    Y[3] = -c1 * x
    Y[4] =  c2 * x * y
    Y[5] = -c2 * y * z
    Y[6] =  c3 * (3.0*z*z - 1.0)
    Y[7] = -c2 * x * z
    Y[8] =  c4 * (x*x - y*y)
    return Y

def env_from_sh(H, W, sh_coeffs):
    """
    sh_coeffs: (9,3) for RGB, order=2
    Returns env: (H,W,3) in [0,1]
    """
    H = int(H); W = int(W)
    env = np.zeros((H, W, 3), dtype=np.float32)
    dtheta = math.pi / H
    dphi   = 2.0 * math.pi / W

    for y in range(H):
        theta = (y + 0.5) * dtheta
        sin_theta = math.sin(theta)
        ct = math.cos(theta)
        for x in range(W):
            phi = (x + 0.5) * dphi
            cp = math.cos(phi)
            sp = math.sin(phi)

            vx = sin_theta * cp
            vy = sin_theta * sp
            vz = ct

            Y = sh_basis_dir_l2(vx, vy, vz)   # (9,)
            rgb = (sh_coeffs.T @ Y).astype(np.float32)  # (3,)
            env[y, x, :] = rgb
            #env = np.maximum(env, 0.0)
            #if env.max() > 0:
                #env = env / (env.max())
    return env


def main():
    base_dir = Path("./plane_dataset_4")
    trans_dir = base_dir / "transport_data"
    out_dir = Path("transport_vis")
    out_dir.mkdir(parents=True, exist_ok=True)

    X_path = trans_dir / "X_params_all.npy"
    Y_path = trans_dir / "Y_sh_out_all.npy"
    trans_ckpt_path = trans_dir / "transport_mlp.pt"
    rend_ckpt_path  = Path("fno_params_to_image_cameras_larger120_finetuned_finetuned.pt")

    # load X,Y
    X = np.load(X_path, mmap_mode="r")   # [N,D]
    Y = np.load(Y_path, mmap_mode="r")   # [N,27]
    N, D_in = X.shape
    print("Loaded transport data:", N, "samples,", D_in, "input dims")

    # load transport MLP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_ckpt = torch.load(trans_ckpt_path, map_location=device, weights_only=False)
    in_dim  = t_ckpt["in_dim"]
    out_dim = t_ckpt["out_dim"]
    X_mean  = t_ckpt["X_mean"]
    X_std   = t_ckpt["X_std"]
    t_model = TransportMLP(in_dim, out_dim, hidden=384, num_layers=4).to(device)
    t_model.load_state_dict(t_ckpt["model_state"])
    t_model.eval()
    print("Loaded transport MLP from", trans_ckpt_path)

    # load renderer
    r_ckpt = torch.load(rend_ckpt_path, map_location=device, weights_only=False)
    latent_dim = r_ckpt["latent_dim"]
    param_mean = r_ckpt["param_mean"]
    param_std  = r_ckpt["param_std"]
    r_model = FNOPlusResNet(latent_dim=latent_dim, img_size=(64,64)).to(device)
    state = r_ckpt["model_state"]
    state.pop("_metadata", None)
    r_model.load_state_dict(state)
    r_model.eval()
    print("Loaded renderer from", rend_ckpt_path)

    # pick a few random indices
    n_show = 10
    idxs = random.sample(range(N), n_show)

    H_img = W_img = 64
    H_env = 64
    W_env = 64

    for idx in idxs:
        x_norm = X[idx].astype(np.float32)              # this X is already normalized for renderer
        y_gt   = Y[idx].astype(np.float32)             # [27]

        # 1) renderer GT image
        # un-normalize back to renderer param space, then normalize again is redundant here;
        # X was produced as normalized renderer params, so we can feed x_norm directly if
        # X_mean/std in transport were identity. If you used extra normalization in TransportDataset,
        # invert it:
        x_rend = torch.from_numpy(x_norm).unsqueeze(0).to(device)
        with torch.no_grad():
            img_pred = r_model(x_rend).clamp(0, 1)[0]
        img_gt_np = np.transpose(img_pred.cpu().numpy(), (1,2,0))  # [64,64,3]

        # 2) GT SH env from Y
        sh_gt = y_gt.reshape(9,3)                      # [9,3]
        env_gt = env_from_sh(H_env, W_env, sh_gt)      # [H_env,W_env,3]

        # 3) MLP-predicted SH env
        x_raw = X[idx].astype(np.float32)      # row from disk
        x_mlp = (x_raw - X_mean) / X_std       # SAME normalization as training
        
        x_t = torch.from_numpy(x_mlp).unsqueeze(0).to(device)
        with torch.no_grad():
            y_pred = t_model(x_t)[0].cpu().numpy()
        sh_pred = y_pred.reshape(9,3)
        env_pred = env_from_sh(H_env, W_env, sh_pred)

        # 4) visualize
        fig, axes = plt.subplots(1, 3, figsize=(9,3))
        axes[0].imshow(img_gt_np)
        axes[0].set_title(f"Renderer image (idx={idx})")
        axes[0].axis("off")

        axes[1].imshow(env_gt)
        axes[1].set_title("Env from GT SH")
        axes[1].axis("off")

        axes[2].imshow(env_pred)
        axes[2].set_title("Env from MLP SH")
        axes[2].axis("off")

        plt.tight_layout()
        save_path = out_dir / f"vis_idx_{idx:07d}.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print("Saved", save_path)

if __name__ == "__main__":
    main()