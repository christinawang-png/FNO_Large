#!/usr/bin/env python

import sys
import math
from pathlib import Path
from argparse import ArgumentParser

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

BASE_DIR = FNO_ROOT / "plane_dataset_4"

RENDERS_DIR = BASE_DIR / "renders"
ALPHA_DIR = BASE_DIR / "hard_alpha"

IMG_META_CSV = RENDERS_DIR / "metadata_images_all_sharded.csv"
VOL_META_CSV = BASE_DIR / "metadata_volumes.csv"
ALPHA_META_CSV = ALPHA_DIR / "metadata_alpha_all.csv"

DEFAULT_SURFACE_CHECKPOINT = (
    FNO_ROOT / "fno_premult_surface_final.pt"
)

DEFAULT_VOLUME_CHECKPOINT = (
    FNO_ROOT / "fno_premult_volume_epoch025.pt"
)

OUTPUT_DIR = (
    REPO_DIR / "differentiable_fno_placement_outputs"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMG_SIZE = (64, 64)

CANVAS_H = 512
CANVAS_W = 512

OPTIMIZATION_STEPS = 1000
LEARNING_RATE = 1e-2

SIZE_LOSS_WEIGHT = 0.02

# Known target placement for this test.
TARGET_CENTER_X = 350.0
TARGET_CENTER_Y = 180.0
TARGET_WIDTH = 230.0
TARGET_HEIGHT = 160.0

# Initial placement.
INITIAL_CENTER_X = 310.0
INITIAL_CENTER_Y = 220.0
INITIAL_WIDTH = 180.0
INITIAL_HEIGHT = 190.0


# ============================================================
# DIFFERENTIABLE PLACEMENT
# ============================================================

def place_patch_differentiable(
    patch,
    canvas_height,
    canvas_width,
    center_x,
    center_y,
    patch_width,
    patch_height,
):
    """
    Differentiably resize and place a patch into a canvas.

    patch:
        [B,C,H,W]

    center_x, center_y:
        Center in output-canvas pixel coordinates.

    patch_width, patch_height:
        Desired patch size in output pixels.

    Returns:
        [B,C,canvas_height,canvas_width]
    """
    if patch.ndim != 4:
        raise ValueError(
            f"Expected patch [B,C,H,W], got {patch.shape}"
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

    patch_width = torch.as_tensor(
        patch_width,
        dtype=dtype,
        device=device,
    ).reshape(-1)

    patch_height = torch.as_tensor(
        patch_height,
        dtype=dtype,
        device=device,
    ).reshape(-1)

    if center_x.numel() == 1:
        center_x = center_x.expand(batch_size)

    if center_y.numel() == 1:
        center_y = center_y.expand(batch_size)

    if patch_width.numel() == 1:
        patch_width = patch_width.expand(batch_size)

    if patch_height.numel() == 1:
        patch_height = patch_height.expand(batch_size)

    patch_width = patch_width.clamp_min(2.0)
    patch_height = patch_height.clamp_min(2.0)

    canvas_w_minus_one = float(
        max(canvas_width - 1, 1)
    )

    canvas_h_minus_one = float(
        max(canvas_height - 1, 1)
    )

    center_x_norm = (
        2.0 * center_x / canvas_w_minus_one - 1.0
    )

    center_y_norm = (
        2.0 * center_y / canvas_h_minus_one - 1.0
    )

    # Mapping from output canvas coordinates to input patch
    # normalized coordinates.
    scale_x = canvas_w_minus_one / patch_width
    scale_y = canvas_h_minus_one / patch_height

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

    canvas = F.grid_sample(
        patch,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    return canvas


# ============================================================
# FNO LOADING
# ============================================================

def load_fno_checkpoint(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    print("Loading checkpoint:", checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state = checkpoint.get(
        "model_state",
        checkpoint,
    )

    state = dict(state)
    state.pop("_metadata", None)

    latent_dim = int(checkpoint["latent_dim"])

    param_mean = np.asarray(
        checkpoint["param_mean"],
        dtype=np.float32,
    )

    param_std = np.asarray(
        checkpoint["param_std"],
        dtype=np.float32,
    )

    model = FNOPlusResNetSingle(
        latent_dim=latent_dim,
        img_size=IMG_SIZE,
    ).to(device)

    model.load_state_dict(state)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    print("  latent_dim:", latent_dim)
    print(
        "  mode:",
        checkpoint.get("mode", "unknown"),
    )

    return model, param_mean, param_std, latent_dim


def build_fno_input_from_template(
    dataset,
    template_row,
    mode,
    param_mean,
    param_std,
    device,
):
    """
    Build one normalized FNO parameter vector from a metadata
    template row.

    The template provides shape/material/environment values.
    For this test, the camera parameters remain those of the
    selected metadata row.

    The vector ordering is delegated to the same dataset method
    used during training.
    """
    row = template_row.copy()
    row["render_mode"] = mode

    if mode == "volume":
        row["metallic"] = 0.0
        row["roughness"] = 0.0
        row["specular"] = 0.0

    raw_np = dataset._build_param_vector_np(row)

    if raw_np.shape[0] != param_mean.shape[0]:
        raise RuntimeError(
            "Parameter dimension mismatch: "
            f"constructed={raw_np.shape[0]}, "
            f"checkpoint={param_mean.shape[0]}"
        )

    normalized_np = (
        raw_np - param_mean
    ) / param_std

    return torch.from_numpy(
        normalized_np.astype(np.float32)
    ).unsqueeze(0).to(device)


# ============================================================
# VISUALIZATION
# ============================================================

def rgba_to_rgb_image(rgba):
    """
    Convert [1,4,H,W] premultiplied RGBA to [H,W,3].
    """
    rgba = rgba.detach().cpu().clamp(0.0, 1.0)

    rgb = rgba[0, :3]
    rgb = rgb.permute(1, 2, 0)

    return rgb.numpy()


def rgba_to_alpha_image(rgba):
    """
    Convert [1,4,H,W] to [H,W].
    """
    rgba = rgba.detach().cpu().clamp(0.0, 1.0)
    return rgba[0, 3].numpy()


def save_preview(
    patch,
    target_canvas,
    final_canvas,
    output_path,
):
    patch_rgb = rgba_to_rgb_image(patch)
    target_rgb = rgba_to_rgb_image(target_canvas)
    final_rgb = rgba_to_rgb_image(final_canvas)
    final_alpha = rgba_to_alpha_image(final_canvas)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4),
    )

    axes[0].imshow(patch_rgb)
    axes[0].set_title("FNO patch")
    axes[0].axis("off")

    axes[1].imshow(target_rgb)
    axes[1].set_title("Target placement")
    axes[1].axis("off")

    axes[2].imshow(final_rgb)
    axes[2].set_title("Optimized placement")
    axes[2].axis("off")

    axes[3].imshow(
        final_alpha,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    axes[3].set_title("Optimized alpha")
    axes[3].axis("off")

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = ArgumentParser(
        description=(
            "Test differentiable placement using a real "
            "pretrained FNO RGBA output."
        )
    )

    parser.add_argument(
        "--mode",
        choices=["surface", "volume"],
        default="surface",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--template_index",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    print("Using device:", DEVICE)
    print("Mode:", args.mode)

    # --------------------------------------------------------
    # Select checkpoint.
    # --------------------------------------------------------
    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint)
    elif args.mode == "surface":
        checkpoint_path = DEFAULT_SURFACE_CHECKPOINT
    else:
        checkpoint_path = DEFAULT_VOLUME_CHECKPOINT

    # --------------------------------------------------------
    # Load the training dataset metadata.
    #
    # This also gives us the exact parameter construction logic.
    # --------------------------------------------------------
    dataset = PlaneDatasetParamsToPremultRGBA(
        base_dir=BASE_DIR,
        img_meta_csv=IMG_META_CSV,
        vol_meta_csv=VOL_META_CSV,
        renders_dir=RENDERS_DIR,
        alpha_dir=ALPHA_DIR,
        alpha_meta_csv=ALPHA_META_CSV,
        img_size=IMG_SIZE,
        use_sh=True,
        normalize_params=True,
    )

    if "render_mode" not in dataset.df.columns:
        raise RuntimeError(
            "render_mode is missing from dataset metadata."
        )

    print("Dataset rows:", len(dataset.df))
    print(dataset.df["render_mode"].value_counts())

    # --------------------------------------------------------
    # Load FNO.
    # --------------------------------------------------------
    model, param_mean, param_std, latent_dim = \
        load_fno_checkpoint(
            checkpoint_path,
            DEVICE,
        )

    if latent_dim != dataset.latent_dim:
        raise RuntimeError(
            f"Checkpoint latent_dim={latent_dim}, "
            f"dataset latent_dim={dataset.latent_dim}"
        )

    # --------------------------------------------------------
    # Select a template row for this mode.
    # --------------------------------------------------------
    mode_rows = dataset.df[
        dataset.df["render_mode"].astype(str) == args.mode
    ].reset_index(drop=True)

    if len(mode_rows) == 0:
        raise RuntimeError(
            f"No rows found for mode={args.mode}."
        )

    if not 0 <= args.template_index < len(mode_rows):
        raise IndexError(
            f"template_index={args.template_index} is out of range. "
            f"Available rows: {len(mode_rows)}."
        )

    template_row = mode_rows.iloc[
        args.template_index
    ]

    print(
        "Template index:",
        args.template_index,
        "sample_id:",
        int(template_row["sample_id"]),
    )

    # --------------------------------------------------------
    # Build one real FNO input and evaluate it.
    #
    # The model is frozen. Only placement parameters below
    # will be optimized.
    # --------------------------------------------------------
    param_vec = build_fno_input_from_template(
        dataset=dataset,
        template_row=template_row,
        mode=args.mode,
        param_mean=param_mean,
        param_std=param_std,
        device=DEVICE,
    )

    with torch.no_grad():
        patch = model(param_vec).clamp(0.0, 1.0)

    if patch.shape != (1, 4, 64, 64):
        raise RuntimeError(
            f"Expected FNO output [1,4,64,64], "
            f"got {tuple(patch.shape)}"
        )

    patch = patch.detach()

    print("FNO patch shape:", tuple(patch.shape))
    print(
        "FNO patch range:",
        float(patch.min()),
        float(patch.max()),
    )

    # --------------------------------------------------------
    # Build a fixed target canvas using the same FNO patch.
    #
    # This isolates placement from scene/model mismatch.
    # --------------------------------------------------------
    target_center_x = torch.tensor(
        TARGET_CENTER_X,
        dtype=torch.float32,
        device=DEVICE,
    )

    target_center_y = torch.tensor(
        TARGET_CENTER_Y,
        dtype=torch.float32,
        device=DEVICE,
    )

    target_width = torch.tensor(
        TARGET_WIDTH,
        dtype=torch.float32,
        device=DEVICE,
    )

    target_height = torch.tensor(
        TARGET_HEIGHT,
        dtype=torch.float32,
        device=DEVICE,
    )

    with torch.no_grad():
        target_canvas = place_patch_differentiable(
            patch=patch,
            canvas_height=CANVAS_H,
            canvas_width=CANVAS_W,
            center_x=target_center_x,
            center_y=target_center_y,
            patch_width=target_width,
            patch_height=target_height,
        )

    target_canvas = target_canvas.detach()

    # --------------------------------------------------------
    # Learnable placement variables.
    # --------------------------------------------------------
    optimized_center_x = nn.Parameter(
        torch.tensor(
            INITIAL_CENTER_X,
            dtype=torch.float32,
            device=DEVICE,
        )
    )

    optimized_center_y = nn.Parameter(
        torch.tensor(
            INITIAL_CENTER_Y,
            dtype=torch.float32,
            device=DEVICE,
        )
    )

    optimized_log_width = nn.Parameter(
        torch.tensor(
            math.log(INITIAL_WIDTH),
            dtype=torch.float32,
            device=DEVICE,
        )
    )

    optimized_log_height = nn.Parameter(
        torch.tensor(
            math.log(INITIAL_HEIGHT),
            dtype=torch.float32,
            device=DEVICE,
        )
    )

    optimizer = torch.optim.Adam(
        [
            optimized_center_x,
            optimized_center_y,
            optimized_log_width,
            optimized_log_height,
        ],
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------------
    # Optimize placement.
    # --------------------------------------------------------
    print("\nStarting optimization...")

    for step in range(OPTIMIZATION_STEPS):
        optimized_width = torch.exp(
            optimized_log_width
        )

        optimized_height = torch.exp(
            optimized_log_height
        )

        predicted_canvas = place_patch_differentiable(
            patch=patch,
            canvas_height=CANVAS_H,
            canvas_width=CANVAS_W,
            center_x=optimized_center_x,
            center_y=optimized_center_y,
            patch_width=optimized_width,
            patch_height=optimized_height,
        )

        image_loss = F.mse_loss(
            predicted_canvas,
            target_canvas,
        )

        size_loss = (
            (
                (optimized_width - target_width)
                / target_width
            ) ** 2
            + (
                (optimized_height - target_height)
                / target_height
            ) ** 2
        )

        loss = (
            image_loss
            + SIZE_LOSS_WEIGHT * size_loss
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()
        optimizer.step()

        if step == 0:
            print(
                "center_x gradient:",
                optimized_center_x.grad,
            )
            print(
                "center_y gradient:",
                optimized_center_y.grad,
            )
            print(
                "log_width gradient:",
                optimized_log_width.grad,
            )
            print(
                "log_height gradient:",
                optimized_log_height.grad,
            )

        if step % 50 == 0 or \
           step == OPTIMIZATION_STEPS - 1:

            print(
                f"step={step:04d} "
                f"loss={loss.item():.8f} "
                f"image_loss={image_loss.item():.8f} "
                f"size_loss={size_loss.item():.8f} "
                f"center=("
                f"{optimized_center_x.item():.2f}, "
                f"{optimized_center_y.item():.2f}) "
                f"size=("
                f"{optimized_width.item():.2f}, "
                f"{optimized_height.item():.2f})"
            )

    # --------------------------------------------------------
    # Final placement.
    # --------------------------------------------------------
    with torch.no_grad():
        final_width = torch.exp(
            optimized_log_width
        )

        final_height = torch.exp(
            optimized_log_height
        )

        final_canvas = place_patch_differentiable(
            patch=patch,
            canvas_height=CANVAS_H,
            canvas_width=CANVAS_W,
            center_x=optimized_center_x,
            center_y=optimized_center_y,
            patch_width=final_width,
            patch_height=final_height,
        )

    # --------------------------------------------------------
    # Save result.
    # --------------------------------------------------------
    preview_path = OUTPUT_DIR / (
        f"fno_{args.mode}_placement_test.png"
    )

    save_preview(
        patch=patch,
        target_canvas=target_canvas,
        final_canvas=final_canvas,
        output_path=preview_path,
    )

    print("\nExpected placement:")
    print(
        f"  center=({TARGET_CENTER_X:.2f}, "
        f"{TARGET_CENTER_Y:.2f})"
    )
    print(
        f"  size=({TARGET_WIDTH:.2f}, "
        f"{TARGET_HEIGHT:.2f})"
    )

    print("\nRecovered placement:")
    print(
        f"  center=("
        f"{optimized_center_x.item():.2f}, "
        f"{optimized_center_y.item():.2f})"
    )
    print(
        f"  size=("
        f"{final_width.item():.2f}, "
        f"{final_height.item():.2f})"
    )

    print("\nSaved preview:")
    print(preview_path)


if __name__ == "__main__":
    main()