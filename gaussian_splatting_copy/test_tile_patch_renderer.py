#!/usr/bin/env python

from pathlib import Path

import imageio.v2 as imageio
import torch
import torch.nn.functional as F

from tile_patch_renderer import TilePatchRenderer


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUT_DIR = (
    Path(__file__).resolve().parent
    / "tile_renderer_test_outputs"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

N_PATCHES = 12
PATCH_H = 64
PATCH_W = 64

IMAGE_H = 512
IMAGE_W = 512


def make_random_patches(
    n,
    device,
):
    """
    Produce premultiplied random RGBA patches.
    """
    rgb = torch.rand(
        n,
        3,
        PATCH_H,
        PATCH_W,
        device=device,
        requires_grad=True,
    )

    alpha = torch.rand(
        n,
        1,
        PATCH_H,
        PATCH_W,
        device=device,
        requires_grad=True,
    )

    alpha = alpha * 0.8

    color = rgb * alpha

    return torch.cat(
        [color, alpha],
        dim=1,
    )


def save_rgba_as_visible_rgb(
    rgba,
    path,
):
    rgba = rgba.detach().cpu().clamp(0.0, 1.0)

    rgb = rgba[0, :3]
    alpha = rgba[0, 3:4]

    # Visualize over white.
    visible = rgb + (
        1.0 - alpha
    ) * torch.ones_like(rgb)

    image = visible.permute(
        1,
        2,
        0,
    ).numpy()

    imageio.imwrite(
        path,
        (image * 255.0 + 0.5).astype("uint8"),
    )


def main():
    torch.manual_seed(42)

    print("Using device:", DEVICE)

    # Patch output is non-leaf because it is built with torch.cat.
    # retain_grad() lets us inspect its gradient for this test.
    patches = make_random_patches(
        N_PATCHES,
        DEVICE,
    )
    patches.retain_grad()

    # These are leaf tensors, so .grad will be populated.
    center_x = torch.empty(
        N_PATCHES,
        device=DEVICE,
    ).uniform_(0.0, float(IMAGE_W)).requires_grad_()

    center_y = torch.empty(
        N_PATCHES,
        device=DEVICE,
    ).uniform_(0.0, float(IMAGE_H)).requires_grad_()

    patch_size = torch.empty(
        N_PATCHES,
        device=DEVICE,
    ).uniform_(32.0, 160.0).requires_grad_()

    # Depth is discrete for sorting, so it does not need gradients.
    depths = torch.empty(
        N_PATCHES,
        device=DEVICE,
    ).uniform_(0.0, 10.0)

    renderer = TilePatchRenderer(
        tile_size=128,
        roi_margin=16,
        max_patches_per_tile=None,
    ).to(DEVICE)
    
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    rendered, tile_bins = renderer(
        patches=patches,
        center_x=center_x,
        center_y=center_y,
        patch_size=patch_size,
        depths=depths,
        image_height=IMAGE_H,
        image_width=IMAGE_W,
        return_tile_bins=True,
    )
    
    loss = rendered.mean()
    loss.backward()

    print(
        "center_x grad exists:",
        center_x.grad is not None,
    )

    print(
        "center_x grad finite:",
        torch.isfinite(center_x.grad).all().item()
        if center_x.grad is not None
        else False,
    )

    print(
        "center_y grad exists:",
        center_y.grad is not None,
    )

    print(
        "center_y grad finite:",
        torch.isfinite(center_y.grad).all().item()
        if center_y.grad is not None
        else False,
    )

    print(
        "patch_size grad exists:",
        patch_size.grad is not None,
    )

    print(
        "patch_size grad finite:",
        torch.isfinite(patch_size.grad).all().item()
        if patch_size.grad is not None
        else False,
    )

    print(
        "patches grad exists:",
        patches.grad is not None,
    )

    print(
        "patches grad finite:",
        torch.isfinite(patches.grad).all().item()
        if patches.grad is not None
        else False,
    )

    print(
        "center_x grad finite:",
        torch.isfinite(center_x.grad).all().item(),
    )

    print(
        "center_y grad finite:",
        torch.isfinite(center_y.grad).all().item(),
    )

    # patch_size is not a leaf after arithmetic above. Retain
    # grad on the actual tensor only if you want to inspect it.
    print(
        "patches grad finite:",
        torch.isfinite(patches.grad).all().item()
        if patches.grad is not None
        else "patches is non-leaf / no retained grad",
    )

    if DEVICE.type == "cuda":
        peak_gib = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

        print(
            f"Peak allocated memory: {peak_gib:.3f} GiB"
        )

    save_rgba_as_visible_rgb(
        rendered,
        OUT_DIR / "tile_renderer_output.png",
    )

    print(
        "Saved:",
        OUT_DIR / "tile_renderer_output.png",
    )


if __name__ == "__main__":
    main()