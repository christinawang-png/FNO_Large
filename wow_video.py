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
    coeffs *= 1.2

    return coeffs.astype(np.float32)  # (num_coeffs,3)


# ---------- Build param vector in same order as PlaneDatasetParamsToImage ----------
def build_param_vec(p1, p2, sigma,
                    hue, saturation, metallic, roughness, opacity, specular,
                    phi, theta, radius,
                    sh_coeffs,  # (num_coeffs,3) or None
                    dataset):
    """
    Matches _build_param_vector_np logic in PlaneDatasetParamsToImage.
    """
    sin_phi, cos_phi = math.sin(phi), math.cos(phi)
    sin_th, cos_th   = math.sin(theta), math.cos(theta)

    scalars = [
        float(p1), float(p2), float(sigma),
        float(hue), float(saturation),
        float(metallic), float(roughness),
        float(opacity), float(specular),
        sin_phi, cos_phi, sin_th, cos_th,
        float(radius),
    ]

    if sh_coeffs is not None:
        # Flatten SH in same order as CSV columns
        # dataset expects columns in order sh_l.._r, sh_l.._g, sh_l.._b
        pairs = sh_lm_list(order=2)
        for idx, (l, m) in enumerate(pairs):
            r_c, g_c, b_c = sh_coeffs[idx]
            scalars.extend([r_c, g_c, b_c])

    scalars_np = np.array(scalars, dtype=np.float32)
    # normalize with training stats
    scalars_np = (scalars_np - dataset.param_mean) / dataset.param_std
    return torch.from_numpy(scalars_np)  # [latent_dim]


def main():
    # ---------- Load dataset just to get normalization stats ----------
    base_dir   = Path("./plane_dataset_3")  # adjust if needed
    image_csv  = base_dir / "renders" / "metadata_images_all_combined.csv"
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

    ckpt = torch.load("fno_params_to_image_cameras_130_finetuned_finetuned.pt", map_location=device, weights_only=False)
    state = ckpt["model_state"]
    state.pop("_metadata", None)
    model.load_state_dict(state)
    model.eval()

    print("Loaded model; generating WOW video...")

    # ---------- Trajectory settings ----------
    NUM_GLOBAL_ENVS = 128  # or whatever you used in generation
    num_frames = 120
    fps = 12

    out_dir = Path("wow_video4_frames")
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path("wow_video4.mp4")

    with torch.no_grad(), imageio.get_writer(video_path, fps=fps) as writer:
        for i in range(num_frames):
            t = i / (num_frames - 1)  # 0..1

            # ---- Shape (p1, p2, sigma) ----
            # Loop in (p1,p2), slight sigma wiggle. Slightly OOD at edges.
            p1 = math.cos(2.0 * math.pi * t)
            p2 = math.sin(2.0 * math.pi * t)
            sigma = 0.02 + 0.14 * (0.5 * (1.0 + math.sin(2.0 * math.pi * t)))

            # ---- Env SH (snake + slight extrapolation) ----
            env_id_float = (NUM_GLOBAL_ENVS - 1) * (t * 1.4 - 0.2)  # goes a bit beyond [0, N-1]
            sh_coeffs = sh_for_global_env(env_id_float, NUM_GLOBAL_ENVS, order=2)

            # ---- Material ----
            # Strongly colored, darker object
            hue        = (0.2 + 0.6 * t) % 1.0           # cycles through colors
            saturation = 0.7                             # fairly saturated

            if t < 0.25:
                metallic = 0.0
            elif t < 0.5:
                metallic = 1.0
            elif t < 0.75:
                metallic = 0.0
            else:
                metallic = 1.0

            # Roughness: mid-range, oscillates between slightly glossy and quite rough
            roughness  = 0.1 + 0.8 * (0.5 * (1.0 + math.sin(4.0 * math.pi * t)))
            # ~0.2 to ~0.7

            # Opacity: keep mostly opaque for clean edges, tiny variation only
            opacity    = 0.8 + 0.2 * math.sin(2.0 * math.pi * t + 0.5)
            # ~0.8–1.0

            specular   = 0.5

            # ---- Camera ----
            theta  = 2.0 * math.pi * t          # full orbit
            phi_center = math.pi / 2.0   # 90°, side view
            phi_amp    = math.radians(25)  # swing ±25°

            phi = phi_center + phi_amp * math.sin(2.0 * math.pi * t)
            phi = math.radians(65)
            radius = 1.2 + 0.3 * math.sin(2.0 * math.pi * t)
            # radius in [1.3, 1.5] instead of [0.8, 1.2]

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