import math
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image
import imageio.v2 as imageio

# import your classes from training script/module
from train import PlaneDatasetParamsToImageSharded, FNOPlusResNet # adjust import path


# ---------- Helper: build SH env from global "snake" (same as training) ----------
def sh_lm_list(order):
    """
    List (l, m) pairs in a fixed order up to given order.
    For order=2: (0,0),
                 (1,-1),(1,0),(1,1),
                 (2,-2),(2,-1),(2,0),(2,1),(2,2)
    """
    pairs = []
    for l in range(order + 1):
        for m in range(-l, l + 1):
            pairs.append((l, m))
    return pairs


def sh_for_global_env(env_id_float, num_global_envs, order=2):
    """
    Continuous version of sh_for_global_env:
    env_id_float can be non-integer; we just treat it as u in [0,1] directly.
    """
    pairs = sh_lm_list(order)
    num_coeffs = len(pairs)
    coeffs = np.zeros((num_coeffs, 3), dtype=np.float32)

    # map env_id_float to u in [0,1] but allow slight extrapolation
    u = env_id_float / max(1.0, float(num_global_envs - 1))
    # allow a bit OOD: u in [-0.2, 1.2] -> clamp for safety
    u = max(-0.2, min(1.2, u))
    t = 2.0 * math.pi * u

    # base color wheel
    r = 0.5 + 0.4 * math.sin(t)
    g = 0.5 + 0.4 * math.sin(t + 2.0 * math.pi / 3.0)
    b = 0.5 + 0.4 * math.sin(t + 4.0 * math.pi / 3.0)
    rgb = np.array([r, g, b], dtype=np.float32)
    gray = np.full(3, rgb.mean(), dtype=np.float32)

    if u < 1.0/3.0:
        alpha = 0.1
    elif u < 2.0/3.0:
        alpha = 0.5
    else:
        alpha = 1.0
    rgb_scale = (1.0 - alpha) * gray + alpha * rgb

    # ambient
    coeffs[0, :] = rgb_scale * 0.4

    pairs = sh_lm_list(order)
    for idx, (l, m) in enumerate(pairs):
        if l == 1:
            if m == -1:
                coeffs[idx, :] = rgb_scale * (0.2 * math.sin(2.0 * math.pi * u))
            elif m == 0:
                coeffs[idx, :] = rgb_scale * (0.2 * math.cos(2.0 * math.pi * u))
            elif m == 1:
                coeffs[idx, :] = rgb_scale * (0.2 * math.sin(2.0 * math.pi * u + 1.0))
    for idx, (l, m) in enumerate(pairs):
        if l == 2 and m == 0:
            coeffs[idx, :] += rgb_scale * (0.05 * math.cos(4.0 * math.pi * u))

    rgb_scale = 0.8 * gray + 0.2 * rgb   # mostly gray with a hint of tint
    coeffs[0, :] = rgb_scale * 0.8       # brighter ambient than before
    # optional: scale all SH up a bit so env is brighter than object
    #coeffs *= 1.2

    return coeffs.astype(np.float32)  # (num_coeffs,3)


# ---------- Build param vector in same order as PlaneDatasetParamsToImage ----------
def build_param_vec(ctrl_vals, sigma,
                    hue, saturation, metallic, roughness, opacity, specular,
                    phi, theta, radius,
                    sh_coeffs,
                    dataset):
    """
    ctrl_vals: list/array of control heights in the same order as ctrl_cols
    sigma: scalar

    Matches _build_param_vector_np in training:
      [ctrl_vals..., sigma, hue, saturation, metallic, roughness, opacity, specular,
       sin(phi), cos(phi), sin(theta), cos(theta), radius, SH...]
    """
    sin_phi, cos_phi = math.sin(phi), math.cos(phi)
    sin_th,  cos_th  = math.sin(theta), math.cos(theta)

    scalars = list(ctrl_vals) + [
        float(sigma),
        float(hue), float(saturation), float(metallic), float(roughness),
        float(opacity), float(specular),
        sin_phi, cos_phi, sin_th, cos_th,
        float(radius),
    ]

    if sh_coeffs is not None:
        # flatten SH in l,m,rgb order
        pairs = sh_lm_list(order=2)
        for idx, (l, m) in enumerate(pairs):
            r_c, g_c, b_c = sh_coeffs[idx]
            scalars.extend([r_c, g_c, b_c])

    scalars_np = np.array(scalars, dtype=np.float32)
    # normalize with training stats
    scalars_np = (scalars_np - dataset.param_mean) / dataset.param_std
    return torch.from_numpy(scalars_np)

# t in [0,1]: go cyan -> magenta -> golden -> back
def nice_hue_path(t):
    # piecewise: 0–1/3 cyan, 1/3–2/3 magenta, 2/3–1 gold
    if t < 1/3:
        h0, h1 = 0.5, 0.83      # cyan(~0.5) to magenta(~0.83)
        u = t * 3.0
    elif t < 2/3:
        h0, h1 = 0.83, 0.13     # magenta to orange/gold
        u = (t - 1/3) * 3.0
    else:
        h0, h1 = 0.13, 0.5      # gold back to cyan
        u = (t - 2/3) * 3.0
    return (1 - u) * h0 + u * h1


def main():
    # ---------- Load dataset just to get normalization stats ----------
    base_dir   = Path("./plane_dataset_4")  # adjust if needed
    image_csv  = base_dir / "renders" / "metadata_images_all_sharded.csv"
    volume_csv = base_dir / "metadata_volumes.csv"

    dataset = PlaneDatasetParamsToImageSharded(
        image_csv_path=str(image_csv),
        volume_csv_path=str(volume_csv),
        img_size=(64,64),
        use_sh=True,
        normalize_params=True,
        shards_dir=str(base_dir),  # wherever you saved images_64x64_shard_*.npy
    )
    latent_dim = dataset.latent_dim

    # ---------- Load trained model ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FNOPlusResNet(latent_dim=latent_dim, img_size=(64, 64)).to(device)

    ckpt = torch.load("fno_params_to_image_cameras_larger130_finetuned_finetuned_color.pt", map_location=device, weights_only=False)
    state = ckpt["model_state"]
    state.pop("_metadata", None)
    model.load_state_dict(state)
    model.eval()

    print("Loaded model; generating WOW video...")

    # ---------- Trajectory settings ----------
    NUM_GLOBAL_ENVS = 128  # or whatever you used in generation
    num_frames = 120
    fps = 12

    out_dir = Path("wow_video6_frames")
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path("wow_video6.mp4")

    # simple neutral env: approximate white ambient + gentle top light
    sh_coeffs_neutral = np.zeros((9, 3), dtype=np.float32)
    sh_coeffs_neutral[0, :] = 0.8  # strong Y_00, white-ish ambient
    # small directional term to get some shading
    sh_coeffs_neutral[2, :] = 0.2  # Y_10 (z-ish)

    with torch.no_grad(), imageio.get_writer(video_path, fps=fps) as writer:
        for i in range(num_frames):
            t = i / (num_frames - 1)  # 0..1

            # ---- Shape (p1, p2, sigma) ----
            # Loop in (p1,p2), slight sigma wiggle. Slightly OOD at edges.
            p1 = 0.8 * math.cos(2.0 * math.pi * t)
            p2 = 0.6 * math.sin(2.0 * math.pi * t)
            sigma = 0.03 + 0.03 * (0.5 * (1.0 + math.sin(2.0 * math.pi * t)))
            # sigma in [0.03, 0.06]

            # ---- Env SH (snake + slight extrapolation) ----
            # SH_ORDER=2 → 9 coeffs per channel
            env_id_float = (NUM_GLOBAL_ENVS - 1) * t
            env_id = int(env_id_float)
            sh_coeffs = sh_for_global_env(env_id, NUM_GLOBAL_ENVS, order=2)
            # make env brighter
            sh_coeffs *= 1.5   # global gain

            # make it whiter (less tinted)
            mean_rgb = sh_coeffs.mean(axis=1, keepdims=True)  # (9,1)
            sh_coeffs = 0.8 * mean_rgb + 0.2 * sh_coeffs      # pull toward gray/white

            # ---- Material ----
            # Strongly colored, darker object
            # hue oscillates between warm and cool
            hue        = 0.9
            saturation = 0.7     # vivid, but not neon
            metallic   = 0.0     # keep to dielectric for clean color
            roughness  = 0.1    # semi-gloss
            opacity    = 1.0
            specular   = 0.5

            # ---- Camera ----
            theta = 2.0 * math.pi * t           # full horizontal orbit
            phi   = math.radians(55)            # fixed elevation
            radius = 1.2

            # ---- Build param vec & predict ----
            param_vec = build_param_vec(
                p1, p2, sigma,
                hue, saturation, metallic, roughness, opacity, specular,
                phi, theta, radius,
                sh_coeffs,
                dataset,
            ).unsqueeze(0).to(device)  # [1, latent_dim]

            pred = model(param_vec)[0]  # [3,H,W]
            pred = pred.clamp(0, 1)

            # Save frame as png (optional) and append to video
            frame_name = out_dir / f"frame_{i:04d}.png"
            save_image(pred.cpu(), frame_name)

            frame_np = (
                pred.mul(255).byte().cpu().permute(1, 2, 0).numpy()
            )
            writer.append_data(frame_np)
            print(f"Frame {i+1}/{num_frames} written")

    print("Wrote wow_video.mp4")


if __name__ == "__main__":
    main()