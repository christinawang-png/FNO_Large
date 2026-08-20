import math
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from train import FNOPlusResNet         # renderer model
from train_faces import TransportMLP  # or paste your TransportMLP here


# ---- SH utilities (order 2) ----

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
    sh_coeffs: (9,3)
    returns env: (H,W,3) in [0,1], no aggressive normalization
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
            Y = sh_basis_dir_l2(vx, vy, vz)
            rgb = (sh_coeffs.T @ Y).astype(np.float32)
            env[y, x, :] = rgb

    env = np.maximum(env, 0.0)
    m = env.max()
    if m > 1.0:
        env /= m
    return env


# ---- simple CPU SH projection for a single image ----

def project_image_to_sh_cpu(img_np):
    """
    img_np: [3,H,W], in [0,1]
    returns (9,3)
    """
    C, H, W = img_np.shape
    dtheta = math.pi / H
    dphi   = 2.0 * math.pi / W

    coeffs = np.zeros((9, 3), dtype=np.float64)
    img = np.transpose(img_np, (1,2,0))  # [H,W,3]

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
            L = img[y, x, :]  # [3]
            Y = sh_basis_dir_l2(vx, vy, vz)
            weight = sin_theta * dtheta * dphi
            coeffs += (Y[:, None] * L[None, :] * weight)
    return coeffs.astype(np.float32)


def main():
    base_dir   = Path("./plane_dataset_4")
    trans_dir  = base_dir / "transport_data"

    X_path = trans_dir / "X_params_all.npy"
    Y_path = trans_dir / "Y_sh_out_all.npy"  # not strictly needed here
    trans_ckpt_path = trans_dir / "transport_mlp.pt"
    rend_ckpt_path  = Path("fno_params_to_image_cameras_larger120_finetuned_finetuned.pt")

    # load X (renderer-normalized params used for transport data)
    X_all = np.load(X_path, mmap_mode="r")
    N, D = X_all.shape
    print("Loaded X:", X_all.shape)

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

    # we assume SH dims are at the tail of the latent vector:
    # ctrl(4) + sigma(1) + material(6) + camera(5) = 16, so SH dims start at 16
    sh_start = 16
    sh_dim   = 27  # 9 coeffs * 3 channels
    assert latent_dim == sh_start + sh_dim, "latent_dim structure assumption broken"

    out_dir = Path("transport_vis_ood")
    out_dir.mkdir(parents=True, exist_ok=True)

    H_img = W_img = 64
    H_env = 64
    W_env = 64

    n_show = 6
    idxs = random.sample(range(N), n_show)

    for idx in idxs:
        # base normalized renderer params (as used when generating transport data)
        x_base = X_all[idx].astype(np.float32)  # [D]

        # create an OOD variant by perturbing SH part
        x_ood = x_base.copy()
        # simple OOD: scale SH and add a bias
        x_ood[sh_start:sh_start+sh_dim] *= 1.5
        x_ood[sh_start:sh_start+sh_dim] += 1.0  # big shift in renderer-normalized space

        # 1) Renderer "GT" image for this OOD param
        # x_ood is in renderer-normalized space already (what renderer expects)
        x_r = torch.from_numpy(x_ood).unsqueeze(0).to(device)  # [1,D]
        with torch.no_grad():
            img_t = r_model(x_r).clamp(0,1)[0]   # [3,64,64]
        img_np = img_t.cpu().numpy()
        img_vis = np.transpose(img_np, (1,2,0))  # [64,64,3]

        # 2) GT SH from renderer image
        sh_gt = project_image_to_sh_cpu(img_np)  # [9,3]

        # 3) MLP-predicted SH for the same OOD param
        # transport MLP expects an extra normalization over X_all: (x_base - X_mean)/X_std
        x_t_in = (x_ood - X_mean) / X_std
        x_t_in = torch.from_numpy(x_t_in).unsqueeze(0).to(device)  # [1,in_dim]
        with torch.no_grad():
            y_pred = t_model(x_t_in)[0].cpu().numpy()  # [27]
        sh_pred = y_pred.reshape(9,3)

        # 4) Reconstruct envs
        env_gt   = env_from_sh(H_env, W_env, sh_gt)
        env_pred = env_from_sh(H_env, W_env, sh_pred)
        
        
        # 2.5) weighted MSE in SH space (2x weight on l=0,1)
        # sh_gt, sh_pred: [9,3] where indices:
        #   0 -> l=0
        #   1,2,3 -> l=1
        #   4..8 -> l=2
        diff = sh_pred - sh_gt              # [9,3]
        w = np.ones((9, 1), dtype=np.float32)
        w[0] = 2.0       # l=0
        w[1:4] = 2.0     # l=1
        # l=2 stays at 1.0

        diff2 = (diff ** 2) * w            # broadcast over RGB
        weighted_mse = diff2.mean()        # simple average with weights applied
        print(f"idx={idx}: weighted SH MSE = {weighted_mse:.6e}")

        # 5) Save side-by-side
        fig, axes = plt.subplots(1, 3, figsize=(9,3))
        axes[0].imshow(img_vis)
        axes[0].set_title(f"Renderer OOD img\nidx={idx}")
        axes[0].axis("off")

        axes[1].imshow(env_gt)
        axes[1].set_title("Env from GT SH")
        axes[1].axis("off")

        axes[2].imshow(env_pred)
        axes[2].set_title("Env from MLP SH")
        axes[2].axis("off")

        plt.tight_layout()
        save_path = out_dir / f"ood_vis_{idx:07d}.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print("Saved", save_path)


if __name__ == "__main__":
    main()