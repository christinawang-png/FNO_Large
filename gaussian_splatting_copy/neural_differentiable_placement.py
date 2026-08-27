#!/usr/bin/env python

import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "differentiable_placement_outputs"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PATCH_H = 64
PATCH_W = 64

CANVAS_H = 512
CANVAS_W = 512

OPTIMIZATION_STEPS = 500
LEARNING_RATE = 5e-2

# Known target placement.
TARGET_CENTER_X = 350.0
TARGET_CENTER_Y = 180.0
TARGET_WIDTH = 230.0
TARGET_HEIGHT = 160.0

# Initial incorrect placement.
INITIAL_CENTER_X = 310.0
INITIAL_CENTER_Y = 220.0
INITIAL_WIDTH = 180.0
INITIAL_HEIGHT = 190.0

# Weight that keeps the learned size from collapsing.
SIZE_LOSS_WEIGHT = 0.01


# ============================================================
# DIFFERENTIABLE PATCH PLACEMENT
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

    Parameters
    ----------
    patch:
        [B,C,H_patch,W_patch]

    center_x, center_y:
        Center in output-canvas pixel coordinates.

    patch_width, patch_height:
        Desired output size in pixels.

    Returns
    -------
    canvas:
        [B,C,canvas_height,canvas_width]
    """
    if patch.ndim != 4:
        raise ValueError(
            f"Expected patch with shape [B,C,H,W], got {patch.shape}"
        )

    batch_size = patch.shape[0]
    dtype = patch.dtype
    device = patch.device

    # Convert scalar tensors to shape [B].
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

    canvas_w_minus_one = float(max(canvas_width - 1, 1))
    canvas_h_minus_one = float(max(canvas_height - 1, 1))

    # Convert desired output-center pixels into normalized coordinates.
    center_x_norm = (
        2.0 * center_x / canvas_w_minus_one - 1.0
    )

    center_y_norm = (
        2.0 * center_y / canvas_h_minus_one - 1.0
    )

    # Map output canvas coordinates into input patch coordinates.
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
# ALPHA COMPOSITING
# ============================================================

def alpha_over(back, front):
    """
    Composite premultiplied RGBA front over back.

    Inputs:
        back/front: [B,4,H,W]
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
# SYNTHETIC PATCH
# ============================================================

def make_test_patch(device):
    """
    Create a fixed premultiplied RGBA patch.

    This is only for testing differentiable placement.
    Later, replace this with an FNO output:
        patch = fno_model(param_vec)
    """
    ys = torch.linspace(
        -1.0,
        1.0,
        PATCH_H,
        device=device,
    )

    xs = torch.linspace(
        -1.0,
        1.0,
        PATCH_W,
        device=device,
    )

    yy, xx = torch.meshgrid(
        ys,
        xs,
        indexing="ij",
    )

    distance = torch.sqrt(xx ** 2 + yy ** 2)

    alpha = torch.sigmoid(
        30.0 * (0.8 - distance)
    )

    red = 0.25 + 0.75 * (xx + 1.0) * 0.5
    green = 0.25 + 0.75 * (yy + 1.0) * 0.5
    blue = 0.15 * torch.ones_like(red)

    rgb = torch.stack(
        [red, green, blue],
        dim=0,
    )

    premultiplied_rgb = rgb * alpha.unsqueeze(0)

    rgba = torch.cat(
        [
            premultiplied_rgb,
            alpha.unsqueeze(0),
        ],
        dim=0,
    )

    return rgba.unsqueeze(0).float()


# ============================================================
# VISUALIZATION
# ============================================================

def rgba_to_rgb_image(rgba):
    """
    Convert [1,4,H,W] premultiplied RGBA to [H,W,3].
    Composites over black.
    """
    rgba = rgba.detach().cpu().clamp(0.0, 1.0)

    rgb = rgba[0, :3]
    rgb = rgb.permute(1, 2, 0)

    return rgb.numpy()


def rgba_to_alpha_image(rgba):
    """
    Convert [1,4,H,W] RGBA to [H,W].
    """
    rgba = rgba.detach().cpu().clamp(0.0, 1.0)
    return rgba[0, 3].numpy()


def save_preview(
    patch,
    target_canvas,
    final_canvas,
    output_path,
):
    """
    Save:
        original patch
        target placement
        optimized placement
        optimized alpha
    """
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
    axes[0].set_title("Original patch")
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
    print("Using device:", DEVICE)

    # --------------------------------------------------------
    # 1. Create a fixed patch.
    # --------------------------------------------------------
    patch = make_test_patch(DEVICE).detach()

    print("Patch shape:", tuple(patch.shape))

    # --------------------------------------------------------
    # 2. Create fixed target placement.
    #
    # No autograd graph is needed for the target.
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

    # --------------------------------------------------------
    # 3. Learnable placement parameters.
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
    # 4. Optimize placement.
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

        # Exactly one backward pass for this graph.
        loss.backward()

        # Optional gradient diagnostics.
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

        optimizer.step()

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
    # 5. Render the final optimized placement.
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
    # 6. Save preview.
    # --------------------------------------------------------
    preview_path = OUTPUT_DIR / "placement_test.png"

    save_preview(
        patch=patch,
        target_canvas=target_canvas,
        final_canvas=final_canvas,
        output_path=preview_path,
    )

    print("\nExpected placement:")
    print(
        f"  center=("
        f"{TARGET_CENTER_X:.2f}, "
        f"{TARGET_CENTER_Y:.2f})"
    )
    print(
        f"  size=("
        f"{TARGET_WIDTH:.2f}, "
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