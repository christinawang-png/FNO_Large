#!/usr/bin/env python

import sys
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# PATHS
# ============================================================

REPO_DIR = Path(__file__).resolve().parent
FNO_ROOT = REPO_DIR.parent

sys.path.insert(0, str(FNO_ROOT))
sys.path.insert(0, str(REPO_DIR))

from train_premult_single_mode import (
    PlaneDatasetParamsToPremultRGBA,
    FNOPlusResNetSingle,
)


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BASE_DIR = FNO_ROOT / "plane_dataset_4"
RENDERS_DIR = BASE_DIR / "renders"
ALPHA_DIR = BASE_DIR / "hard_alpha"

IMG_META_CSV = RENDERS_DIR / "metadata_images_all_sharded.csv"
VOL_META_CSV = BASE_DIR / "metadata_volumes.csv"
ALPHA_META_CSV = ALPHA_DIR / "metadata_alpha_all.csv"

SURFACE_CHECKPOINT = (
    FNO_ROOT / "fno_premult_surface_final.pt"
)

VOLUME_CHECKPOINT = (
    FNO_ROOT / "fno_premult_volume_epoch025.pt"
)

OUTPUT_DIR = (
    REPO_DIR / "synthetic_two_slice_optimization_outputs"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 64
CANVAS_H = 512
CANVAS_W = 512

OPTIMIZATION_STEPS = 2000
LEARNING_RATE = 1e-2

# Ground-truth synthetic placements.
TARGET_SURFACE = {
    "x": 350.0,
    "y": 180.0,
    "size": 230.0,
}

TARGET_VOLUME = {
    "x": 220.0,
    "y": 320.0,
    "size": 170.0,
}

# Intentionally incorrect initial placements.
INITIAL_SURFACE = {
    "x": 320.0,
    "y": 210.0,
    "size": 190.0,
}

INITIAL_VOLUME = {
    "x": 250.0,
    "y": 300.0,
    "size": 140.0,
}

SURFACE_TEMPLATE_INDEX = 0
VOLUME_TEMPLATE_INDEX = 1000


# ============================================================
# DIFFERENTIABLE PLACEMENT
# ============================================================

def place_patch_uniform(
    patch,
    canvas_height,
    canvas_width,
    center_x,
    center_y,
    patch_size,
):
    """
    Uniformly resize and place [B,C,H,W] into a larger canvas.

    The same patch_size is used horizontally and vertically.
    """
    if patch.ndim != 4:
        raise ValueError(
            f"Expected [B,C,H,W], got {patch.shape}"
        )

    batch_size = patch.shape[0]
    dtype = patch.dtype
    device = patch.device

    center_x = torch.as_tensor(
        center_x,
        dtype=dtype,
        device=device,
    ).reshape(-1)

    center_y = torch.as_tensor(
        center_y,
        dtype=dtype,
        device=device,
    ).reshape(-1)

    patch_size = torch.as_tensor(
        patch_size,
        dtype=dtype,
        device=device,
    ).reshape(-1)

    if center_x.numel() == 1:
        center_x = center_x.expand(batch_size)

    if center_y.numel() == 1:
        center_y = center_y.expand(batch_size)

    if patch_size.numel() == 1:
        patch_size = patch_size.expand(batch_size)

    patch_size = patch_size.clamp_min(2.0)

    canvas_wm1 = float(max(canvas_width - 1, 1))
    canvas_hm1 = float(max(canvas_height - 1, 1))

    # Exact pixel-coordinate mapping for align_corners=True.
    scale_x = canvas_wm1 / (patch_size - 1.0)
    scale_y = canvas_hm1 / (patch_size - 1.0)

    center_x_norm = (
        2.0 * center_x / canvas_wm1 - 1.0
    )

    center_y_norm = (
        2.0 * center_y / canvas_hm1 - 1.0
    )

    theta = torch.zeros(
        batch_size,
        2,
        3,
        dtype=dtype,
        device=device,
    )

    theta[:, 0, 0] = scale_x
    theta[:, 1, 1] = scale_y

    theta[:, 0, 2] = -scale_x * center_x_norm
    theta[:, 1, 2] = -scale_y * center_y_norm

    grid = F.affine_grid(
        theta,
        size=(
            batch_size,
            patch.shape[1],
            canvas_height,
            canvas_width,
        ),
        align_corners=True,
    )

    return F.grid_sample(
        patch,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


# ============================================================
# COMPOSITING
# ============================================================

def alpha_over(back, front):
    """
    Premultiplied RGBA front-over-back compositing.

    Inputs:
        [B,4,H,W]
    """
    back_color = back[:, :3]
    back_alpha = back[:, 3:4]

    front_color = front[:, :3]
    front_alpha = front[:, 3:4]

    output_color = (
        front_color
        + (1.0 - front_alpha) * back_color
    )

    output_alpha = (
        front_alpha
        + (1.0 - front_alpha) * back_alpha
    )

    return torch.cat(
        [output_color, output_alpha],
        dim=1,
    )


# ============================================================
# FNO HELPERS
# ============================================================

def load_model(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    state = dict(checkpoint["model_state"])
    state.pop("_metadata", None)

    latent_dim = int(checkpoint["latent_dim"])

    model = FNOPlusResNetSingle(
        latent_dim=latent_dim,
        img_size=(64, 64),
    ).to(DEVICE)

    model.load_state_dict(state)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    param_mean = np.asarray(
        checkpoint["param_mean"],
        dtype=np.float32,
    )

    param_std = np.asarray(
        checkpoint["param_std"],
        dtype=np.float32,
    )

    return model, param_mean, param_std, latent_dim


def build_fno_input(
    dataset,
    template_row,
    mode,
    param_mean,
    param_std,
):
    """
    Build one normalized vector from a real training metadata row.

    The resulting FNO patch is fixed during this optimization test.
    """
    row = template_row.copy()
    row["render_mode"] = mode

    if mode == "volume":
        row["metallic"] = 0.0
        row["roughness"] = 0.0
        row["specular"] = 0.0

    raw = dataset._build_param_vector_np(row)

    if raw.shape[0] != param_mean.shape[0]:
        raise RuntimeError(
            f"Parameter mismatch for {mode}: "
            f"raw={raw.shape[0]}, "
            f"checkpoint={param_mean.shape[0]}"
        )

    normalized = (
        raw - param_mean
    ) / param_std

    return torch.from_numpy(
        normalized.astype(np.float32)
    ).unsqueeze(0).to(DEVICE)


# ============================================================
# VISUALIZATION
# ============================================================

def chw_to_hwc(array):
    return np.transpose(array, (1, 2, 0))


def save_rgb(rgb_chw, path):
    image = chw_to_hwc(
        rgb_chw.detach().cpu().numpy()
        if torch.is_tensor(rgb_chw)
        else rgb_chw
    )

    image = np.clip(image, 0.0, 1.0)

    plt.imsave(
        path,
        image,
    )


def save_alpha(alpha, path):
    if torch.is_tensor(alpha):
        alpha = alpha.detach().cpu().numpy()

    alpha = np.clip(alpha, 0.0, 1.0)

    plt.imsave(
        path,
        alpha,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )


def save_preview(
    target,
    initial,
    optimized,
    path,
):
    target_rgb = target[0, :3].detach().cpu()
    initial_rgb = initial[0, :3].detach().cpu()
    optimized_rgb = optimized[0, :3].detach().cpu()

    target_alpha = target[0, 3].detach().cpu()
    initial_alpha = initial[0, 3].detach().cpu()
    optimized_alpha = optimized[0, 3].detach().cpu()

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12, 8),
    )

    axes[0, 0].imshow(
        chw_to_hwc(target_rgb.numpy())
    )
    axes[0, 0].set_title("Target composite")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(
        chw_to_hwc(initial_rgb.numpy())
    )
    axes[0, 1].set_title("Initial composite")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(
        chw_to_hwc(optimized_rgb.numpy())
    )
    axes[0, 2].set_title("Optimized composite")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(
        target_alpha.numpy(),
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1, 0].set_title("Target alpha")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(
        initial_alpha.numpy(),
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1, 1].set_title("Initial alpha")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(
        optimized_alpha.numpy(),
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1, 2].set_title("Optimized alpha")
    axes[1, 2].axis("off")

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    print("Using device:", DEVICE)

    if DEVICE.type != "cuda":
        print("[WARN] Running on CPU.")

    # --------------------------------------------------------
    # Load metadata and models.
    # --------------------------------------------------------
    dataset = PlaneDatasetParamsToPremultRGBA(
        base_dir=BASE_DIR,
        img_meta_csv=IMG_META_CSV,
        vol_meta_csv=VOL_META_CSV,
        renders_dir=RENDERS_DIR,
        alpha_dir=ALPHA_DIR,
        alpha_meta_csv=ALPHA_META_CSV,
        img_size=(64, 64),
        use_sh=True,
        normalize_params=True,
    )

    print("Dataset rows:", len(dataset.df))
    print(dataset.df["render_mode"].value_counts())

    surface_model, surface_mean, surface_std, surface_dim = \
        load_model(SURFACE_CHECKPOINT)

    volume_model, volume_mean, volume_std, volume_dim = \
        load_model(VOLUME_CHECKPOINT)

    if surface_dim != volume_dim:
        raise RuntimeError(
            "Surface and volume latent dimensions differ."
        )

    if surface_dim != dataset.latent_dim:
        raise RuntimeError(
            "Checkpoint and dataset latent dimensions differ."
        )

    # --------------------------------------------------------
    # Pick fixed templates.
    # --------------------------------------------------------
    surface_rows = dataset.df[
        dataset.df["render_mode"].astype(str) == "surface"
    ].reset_index(drop=True)

    volume_rows = dataset.df[
        dataset.df["render_mode"].astype(str) == "volume"
    ].reset_index(drop=True)

    surface_template = surface_rows.iloc[
        SURFACE_TEMPLATE_INDEX
    ]

    volume_template = volume_rows.iloc[
        VOLUME_TEMPLATE_INDEX
    ]

    surface_input = build_fno_input(
        dataset,
        surface_template,
        "surface",
        surface_mean,
        surface_std,
    )

    volume_input = build_fno_input(
        dataset,
        volume_template,
        "volume",
        volume_mean,
        volume_std,
    )

    # --------------------------------------------------------
    # Generate fixed FNO patches.
    # --------------------------------------------------------
    with torch.no_grad():
        surface_patch = surface_model(
            surface_input
        ).clamp(0.0, 1.0)

        volume_patch = volume_model(
            volume_input
        ).clamp(0.0, 1.0)

    surface_patch = surface_patch.detach()
    volume_patch = volume_patch.detach()

    print(
        "Surface patch shape:",
        tuple(surface_patch.shape),
    )

    print(
        "Volume patch shape:",
        tuple(volume_patch.shape),
    )

    # --------------------------------------------------------
    # Ground-truth synthetic placement.
    #
    # This is the target the optimizer must recover.
    # The target is generated from the same two fixed patches.
    # --------------------------------------------------------
    with torch.no_grad():
        target_surface_layer = place_patch_uniform(
            patch=surface_patch,
            canvas_height=CANVAS_H,
            canvas_width=CANVAS_W,
            center_x=TARGET_SURFACE["x"],
            center_y=TARGET_SURFACE["y"],
            patch_size=TARGET_SURFACE["size"],
        )

        target_volume_layer = place_patch_uniform(
            patch=volume_patch,
            canvas_height=CANVAS_H,
            canvas_width=CANVAS_W,
            center_x=TARGET_VOLUME["x"],
            center_y=TARGET_VOLUME["y"],
            patch_size=TARGET_VOLUME["size"],
        )

        # Surface is the back layer; volume is the front layer.
        target_canvas = alpha_over(
            back=target_surface_layer,
            front=target_volume_layer,
        ).detach()

    # --------------------------------------------------------
    # Learnable initial placements.
    # --------------------------------------------------------
    surface_x = nn.Parameter(
        torch.tensor(
            INITIAL_SURFACE["x"],
            dtype=torch.float32,
            device=DEVICE,
        )
    )

    surface_y = nn.Parameter(
        torch.tensor(
            INITIAL_SURFACE["y"],
            dtype=torch.float32,
            device=DEVICE,
        )
    )

    surface_log_size = nn.Parameter(
        torch.tensor(
            math.log(INITIAL_SURFACE["size"]),
            dtype=torch.float32,
            device=DEVICE,
        )
    )

    volume_x = nn.Parameter(
        torch.tensor(
            INITIAL_VOLUME["x"],
            dtype=torch.float32,
            device=DEVICE,
        )
    )

    volume_y = nn.Parameter(
        torch.tensor(
            INITIAL_VOLUME["y"],
            dtype=torch.float32,
            device=DEVICE,
        )
    )

    volume_log_size = nn.Parameter(
        torch.tensor(
            math.log(INITIAL_VOLUME["size"]),
            dtype=torch.float32,
            device=DEVICE,
        )
    )

    optimizer = torch.optim.Adam(
        [
            surface_x,
            surface_y,
            surface_log_size,
            volume_x,
            volume_y,
            volume_log_size,
        ],
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------------
    # Initial composite.
    # --------------------------------------------------------
    with torch.no_grad():
        initial_surface_layer = place_patch_uniform(
            patch=surface_patch,
            canvas_height=CANVAS_H,
            canvas_width=CANVAS_W,
            center_x=INITIAL_SURFACE["x"],
            center_y=INITIAL_SURFACE["y"],
            patch_size=INITIAL_SURFACE["size"],
        )

        initial_volume_layer = place_patch_uniform(
            patch=volume_patch,
            canvas_height=CANVAS_H,
            canvas_width=CANVAS_W,
            center_x=INITIAL_VOLUME["x"],
            center_y=INITIAL_VOLUME["y"],
            patch_size=INITIAL_VOLUME["size"],
        )

        initial_canvas = alpha_over(
            back=initial_surface_layer,
            front=initial_volume_layer,
        ).detach()

    # --------------------------------------------------------
    # Optimize placements.
    # --------------------------------------------------------
    print("Starting optimization...")

    for step in range(OPTIMIZATION_STEPS):
        surface_size = torch.exp(
            surface_log_size
        ).clamp(2.0, 600.0)

        volume_size = torch.exp(
            volume_log_size
        ).clamp(2.0, 600.0)

        surface_layer = place_patch_uniform(
            patch=surface_patch,
            canvas_height=CANVAS_H,
            canvas_width=CANVAS_W,
            center_x=surface_x,
            center_y=surface_y,
            patch_size=surface_size,
        )

        volume_layer = place_patch_uniform(
            patch=volume_patch,
            canvas_height=CANVAS_H,
            canvas_width=CANVAS_W,
            center_x=volume_x,
            center_y=volume_y,
            patch_size=volume_size,
        )

        predicted_canvas = alpha_over(
            back=surface_layer,
            front=volume_layer,
        )

        loss = F.mse_loss(
            predicted_canvas,
            target_canvas,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or \
           step == OPTIMIZATION_STEPS - 1:
            print(
                f"step={step:04d} "
                f"loss={loss.item():.8f} "
                f"surface_center=("
                f"{surface_x.item():.2f},"
                f"{surface_y.item():.2f}) "
                f"surface_size={surface_size.item():.2f} "
                f"volume_center=("
                f"{volume_x.item():.2f},"
                f"{volume_y.item():.2f}) "
                f"volume_size={volume_size.item():.2f}"
            )

    # --------------------------------------------------------
    # Final output.
    # --------------------------------------------------------
    with torch.no_grad():
        final_surface_size = torch.exp(
            surface_log_size
        ).clamp(2.0, 600.0)

        final_volume_size = torch.exp(
            volume_log_size
        ).clamp(2.0, 600.0)

        final_surface_layer = place_patch_uniform(
            patch=surface_patch,
            canvas_height=CANVAS_H,
            canvas_width=CANVAS_W,
            center_x=surface_x,
            center_y=surface_y,
            patch_size=final_surface_size,
        )

        final_volume_layer = place_patch_uniform(
            patch=volume_patch,
            canvas_height=CANVAS_H,
            canvas_width=CANVAS_W,
            center_x=volume_x,
            center_y=volume_y,
            patch_size=final_volume_size,
        )

        final_canvas = alpha_over(
            back=final_surface_layer,
            front=final_volume_layer,
        ).clamp(0.0, 1.0)

    preview_path = OUTPUT_DIR / "synthetic_two_slice_optimization.png"

    save_preview(
        target=target_canvas,
        initial=initial_canvas,
        optimized=final_canvas,
        path=preview_path,
    )

    print("\nExpected surface placement:")
    print(
        TARGET_SURFACE
    )

    print("Recovered surface placement:")
    print(
        {
            "x": surface_x.item(),
            "y": surface_y.item(),
            "size": final_surface_size.item(),
        }
    )

    print("\nExpected volume placement:")
    print(
        TARGET_VOLUME
    )

    print("Recovered volume placement:")
    print(
        {
            "x": volume_x.item(),
            "y": volume_y.item(),
            "size": final_volume_size.item(),
        }
    )

    print("\nSaved preview:", preview_path)


if __name__ == "__main__":
    main()