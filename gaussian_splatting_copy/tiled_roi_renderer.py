#!/usr/bin/env python

import math
import torch
import torch.nn.functional as F


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_TILE_SIZE = 128
DEFAULT_ROI_MARGIN = 16
DEFAULT_MAX_ROI_SIDE = 512


# ============================================================
# PREMULTIPLIED ALPHA COMPOSITING
# ============================================================

def alpha_over(back, front):
    """
    Premultiplied RGBA front-over-back compositing.

    back/front:
        [1, 4, H, W]

    Returns:
        [1, 4, H, W]
    """
    back_color = back[:, :3]
    back_alpha = back[:, 3:4]

    front_color = front[:, :3]
    front_alpha = front[:, 3:4]

    out_color = (
        front_color
        + (1.0 - front_alpha) * back_color
    )

    out_alpha = (
        front_alpha
        + (1.0 - front_alpha) * back_alpha
    )

    return torch.cat(
        [out_color, out_alpha],
        dim=1,
    )


# ============================================================
# LOCAL ROI PATCH WARP
# ============================================================

def place_patch_uniform_roi(
    patch,
    center_x,
    center_y,
    patch_size,
    canvas_height,
    canvas_width,
    margin_pixels=DEFAULT_ROI_MARGIN,
    max_roi_side=DEFAULT_MAX_ROI_SIDE,
):
    """
    Render one canonical patch only into its local screen-space ROI.

    Inputs:
        patch:
            [1, 4, H_patch, W_patch]

        center_x, center_y:
            Scalar tensors in full-image pixel coordinates.

        patch_size:
            Scalar tensor; desired projected square side in pixels.

    Returns:
        roi_rgba:
            [1, 4, roi_h, roi_w], or None if fully outside.

        roi_x0, roi_y0:
            Integer upper-left pixel location in the full canvas.

    Important:
        ROI bounds are selected discretely using detached values.
        The local warp remains differentiable with respect to
        center_x, center_y, and patch_size inside the chosen ROI.
    """
    if patch.ndim != 4 or patch.shape[0] != 1:
        raise ValueError(
            "ROI placement expects patch shape [1,C,H,W], "
            f"got {tuple(patch.shape)}"
        )

    device = patch.device
    dtype = patch.dtype

    # Discrete ROI bounds only.
    cx_value = float(center_x.detach().cpu().item())
    cy_value = float(center_y.detach().cpu().item())
    size_value = float(patch_size.detach().cpu().item())

    size_value = max(2.0, size_value)

    half_extent = (
        0.5 * size_value
        + float(margin_pixels)
    )

    roi_x0 = max(
        0,
        int(math.floor(cx_value - half_extent)),
    )

    roi_x1 = min(
        canvas_width,
        int(math.ceil(cx_value + half_extent)),
    )

    roi_y0 = max(
        0,
        int(math.floor(cy_value - half_extent)),
    )

    roi_y1 = min(
        canvas_height,
        int(math.ceil(cy_value + half_extent)),
    )

    roi_w = roi_x1 - roi_x0
    roi_h = roi_y1 - roi_y0

    if roi_w <= 0 or roi_h <= 0:
        return None, roi_x0, roi_y0

    # Cap unusually large nearby patches.
    if roi_w > max_roi_side or roi_h > max_roi_side:
        roi_w = min(roi_w, max_roi_side)
        roi_h = min(roi_h, max_roi_side)

        roi_x0 = max(
            0,
            min(
                canvas_width - roi_w,
                int(round(cx_value - 0.5 * roi_w)),
            ),
        )

        roi_y0 = max(
            0,
            min(
                canvas_height - roi_h,
                int(round(cy_value - 0.5 * roi_h)),
            ),
        )

    canvas_wm1 = float(max(canvas_width - 1, 1))
    canvas_hm1 = float(max(canvas_height - 1, 1))

    # Differentiable projected center in normalized full-image space.
    center_x_norm = (
        2.0 * center_x / canvas_wm1 - 1.0
    )

    center_y_norm = (
        2.0 * center_y / canvas_hm1 - 1.0
    )

    safe_patch_size = patch_size.clamp_min(2.0)

    # Full canvas normalized coordinate -> canonical patch coordinate.
    scale_x = canvas_wm1 / safe_patch_size
    scale_y = canvas_hm1 / safe_patch_size

    ys = torch.arange(
        roi_y0,
        roi_y0 + roi_h,
        device=device,
        dtype=dtype,
    )

    xs = torch.arange(
        roi_x0,
        roi_x0 + roi_w,
        device=device,
        dtype=dtype,
    )

    yy, xx = torch.meshgrid(
        ys,
        xs,
        indexing="ij",
    )

    full_x_norm = (
        2.0 * xx / canvas_wm1 - 1.0
    )

    full_y_norm = (
        2.0 * yy / canvas_hm1 - 1.0
    )

    patch_x = scale_x * (
        full_x_norm - center_x_norm
    )

    patch_y = scale_y * (
        full_y_norm - center_y_norm
    )

    grid = torch.stack(
        [patch_x, patch_y],
        dim=-1,
    ).unsqueeze(0)

    roi_rgba = F.grid_sample(
        patch,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    return roi_rgba, roi_x0, roi_y0


# ============================================================
# TILE HELPERS
# ============================================================

def get_tile_bounds(
    tile_y,
    tile_x,
    canvas_height,
    canvas_width,
    tile_size,
):
    """
    Return full-image bounds for one tile:
        x0, x1, y0, y1
    """
    x0 = tile_x * tile_size
    y0 = tile_y * tile_size

    x1 = min(x0 + tile_size, canvas_width)
    y1 = min(y0 + tile_size, canvas_height)

    return x0, x1, y0, y1


def create_empty_tile(
    tile_y,
    tile_x,
    canvas_height,
    canvas_width,
    tile_size,
    device,
    dtype,
):
    """
    Allocate one transparent premultiplied RGBA tile.
    """
    x0, x1, y0, y1 = get_tile_bounds(
        tile_y=tile_y,
        tile_x=tile_x,
        canvas_height=canvas_height,
        canvas_width=canvas_width,
        tile_size=tile_size,
    )

    return torch.zeros(
        1,
        4,
        y1 - y0,
        x1 - x0,
        device=device,
        dtype=dtype,
    )


def update_tile_region(
    old_tile,
    updated_region,
    local_x0,
    local_x1,
    local_y0,
    local_y1,
):
    """
    Return a new small tile with one rectangular region replaced.

    This intentionally rebuilds only a tile, not the full image.
    """
    top = old_tile[:, :, :local_y0, :]

    middle_left = old_tile[
        :,
        :,
        local_y0:local_y1,
        :local_x0,
    ]

    middle_right = old_tile[
        :,
        :,
        local_y0:local_y1,
        local_x1:,
    ]

    middle = torch.cat(
        [
            middle_left,
            updated_region,
            middle_right,
        ],
        dim=3,
    )

    bottom = old_tile[:, :, local_y1:, :]

    return torch.cat(
        [
            top,
            middle,
            bottom,
        ],
        dim=2,
    )


# ============================================================
# ROI -> TILE COMPOSITING
# ============================================================

def composite_roi_into_tiles(
    tiles,
    roi_rgba,
    roi_x0,
    roi_y0,
    canvas_height,
    canvas_width,
    tile_size=DEFAULT_TILE_SIZE,
):
    """
    Composite one ROI into only tiles that it overlaps.

    Parameters
    ----------
    tiles:
        Dictionary:
            (tile_y, tile_x) -> [1,4,tile_h,tile_w]

    roi_rgba:
        [1,4,roi_h,roi_w] or None.

    roi_x0, roi_y0:
        Upper-left ROI location in full-image pixel coordinates.

    Returns
    -------
    tiles:
        Updated tile dictionary.

    This avoids full-image cloning per slice.
    """
    if roi_rgba is None:
        return tiles

    _, _, roi_h, roi_w = roi_rgba.shape

    roi_x1 = min(
        roi_x0 + roi_w,
        canvas_width,
    )

    roi_y1 = min(
        roi_y0 + roi_h,
        canvas_height,
    )

    if roi_x0 >= roi_x1 or roi_y0 >= roi_y1:
        return tiles

    first_tile_x = roi_x0 // tile_size
    last_tile_x = (roi_x1 - 1) // tile_size

    first_tile_y = roi_y0 // tile_size
    last_tile_y = (roi_y1 - 1) // tile_size

    for tile_y in range(
        first_tile_y,
        last_tile_y + 1,
    ):
        for tile_x in range(
            first_tile_x,
            last_tile_x + 1,
        ):
            tile_x0, tile_x1, tile_y0, tile_y1 = (
                get_tile_bounds(
                    tile_y=tile_y,
                    tile_x=tile_x,
                    canvas_height=canvas_height,
                    canvas_width=canvas_width,
                    tile_size=tile_size,
                )
            )

            # Full-image intersection rectangle.
            x0 = max(roi_x0, tile_x0)
            x1 = min(roi_x1, tile_x1)
            y0 = max(roi_y0, tile_y0)
            y1 = min(roi_y1, tile_y1)

            if x0 >= x1 or y0 >= y1:
                continue

            # ROI-local crop coordinates.
            roi_lx0 = x0 - roi_x0
            roi_lx1 = x1 - roi_x0
            roi_ly0 = y0 - roi_y0
            roi_ly1 = y1 - roi_y0

            # Tile-local destination coordinates.
            tile_lx0 = x0 - tile_x0
            tile_lx1 = x1 - tile_x0
            tile_ly0 = y0 - tile_y0
            tile_ly1 = y1 - tile_y0

            key = (tile_y, tile_x)

            if key not in tiles:
                tiles[key] = create_empty_tile(
                    tile_y=tile_y,
                    tile_x=tile_x,
                    canvas_height=canvas_height,
                    canvas_width=canvas_width,
                    tile_size=tile_size,
                    device=roi_rgba.device,
                    dtype=roi_rgba.dtype,
                )

            old_tile = tiles[key]

            roi_crop = roi_rgba[
                :,
                :,
                roi_ly0:roi_ly1,
                roi_lx0:roi_lx1,
            ]

            old_region = old_tile[
                :,
                :,
                tile_ly0:tile_ly1,
                tile_lx0:tile_lx1,
            ]

            new_region = alpha_over(
                back=old_region,
                front=roi_crop,
            )

            # Only reconstruct this tile.
            tiles[key] = update_tile_region(
                old_tile=old_tile,
                updated_region=new_region,
                local_x0=tile_lx0,
                local_x1=tile_lx1,
                local_y0=tile_ly0,
                local_y1=tile_ly1,
            )

    return tiles


# ============================================================
# TILE ASSEMBLY
# ============================================================

def assemble_tiles(
    tiles,
    canvas_height,
    canvas_width,
    tile_size=DEFAULT_TILE_SIZE,
    device=None,
    dtype=torch.float32,
):
    """
    Assemble tiles into one full [1,4,H,W] image.

    A full-resolution canvas is allocated only once here.
    """
    if device is None:
        raise ValueError("device must be specified")

    num_tile_rows = (
        canvas_height + tile_size - 1
    ) // tile_size

    num_tile_cols = (
        canvas_width + tile_size - 1
    ) // tile_size

    image_rows = []

    for tile_y in range(num_tile_rows):
        row = []

        for tile_x in range(num_tile_cols):
            key = (tile_y, tile_x)

            if key in tiles:
                tile = tiles[key]
            else:
                tile = create_empty_tile(
                    tile_y=tile_y,
                    tile_x=tile_x,
                    canvas_height=canvas_height,
                    canvas_width=canvas_width,
                    tile_size=tile_size,
                    device=device,
                    dtype=dtype,
                )

            row.append(tile)

        image_rows.append(
            torch.cat(row, dim=3)
        )

    return torch.cat(
        image_rows,
        dim=2,
    )


# ============================================================
# COMPLETE TILED ROI COMPOSITOR
# ============================================================

def render_rois_to_tiled_canvas(
    sorted_patches,
    sorted_center_x,
    sorted_center_y,
    sorted_patch_sizes,
    canvas_height,
    canvas_width,
    tile_size=DEFAULT_TILE_SIZE,
    margin_pixels=DEFAULT_ROI_MARGIN,
    max_roi_side=DEFAULT_MAX_ROI_SIDE,
):
    """
    Render patches into a tiled final canvas.

    Inputs must already be sorted far -> near.

    sorted_patches:
        [N,4,64,64]

    sorted_center_x/y:
        [N]

    sorted_patch_sizes:
        [N]

    Returns:
        composite:
            [1,4,H,W]
    """
    if sorted_patches.ndim != 4:
        raise ValueError(
            "Expected sorted_patches [N,4,H,W], got "
            f"{tuple(sorted_patches.shape)}"
        )

    num_slices = sorted_patches.shape[0]

    if num_slices == 0:
        return torch.zeros(
            1,
            4,
            canvas_height,
            canvas_width,
            device=sorted_patches.device,
            dtype=sorted_patches.dtype,
        )

    tiles = {}

    # Patches must be processed far -> near so later chunks
    # correctly alpha-composite over earlier chunks.
    for i in range(num_slices):
        patch = sorted_patches[i:i + 1]

        roi_rgba, roi_x0, roi_y0 = (
            place_patch_uniform_roi(
                patch=patch,
                center_x=sorted_center_x[i],
                center_y=sorted_center_y[i],
                patch_size=sorted_patch_sizes[i],
                canvas_height=canvas_height,
                canvas_width=canvas_width,
                margin_pixels=margin_pixels,
                max_roi_side=max_roi_side,
            )
        )

        tiles = composite_roi_into_tiles(
            tiles=tiles,
            roi_rgba=roi_rgba,
            roi_x0=roi_x0,
            roi_y0=roi_y0,
            canvas_height=canvas_height,
            canvas_width=canvas_width,
            tile_size=tile_size,
        )

    return assemble_tiles(
        tiles=tiles,
        canvas_height=canvas_height,
        canvas_width=canvas_width,
        tile_size=tile_size,
        device=sorted_patches.device,
        dtype=sorted_patches.dtype,
    )