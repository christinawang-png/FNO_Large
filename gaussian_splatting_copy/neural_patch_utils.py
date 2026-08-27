#!/usr/bin/env python

import sys
from pathlib import Path

# ------------------------------------------------------------
# Make FNO_Large importable
# ------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
GS_ROOT = THIS_FILE.parent
FNO_ROOT = GS_ROOT.parent

TRAIN_SINGLE_FILE = FNO_ROOT / "train_premult_single_mode.py"
TRAIN_SPLIT_FILE = FNO_ROOT / "train_premult_split.py"

if not TRAIN_SINGLE_FILE.is_file():
    raise FileNotFoundError(
        f"Could not find:\n{TRAIN_SINGLE_FILE}\n"
        f"neural_patch_utils.py is located at:\n{THIS_FILE}\n"
        f"Expected train file in:\n{FNO_ROOT}"
    )

# Insert before imports from train_premult_single_mode.py.
if str(FNO_ROOT) not in sys.path:
    sys.path.insert(0, str(FNO_ROOT))

print(f"[neural_patch_utils] FNO_ROOT={FNO_ROOT}")

# ------------------------------------------------------------
# Normal imports
# ------------------------------------------------------------

import math
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_premult_single_mode import FNOPlusResNetSingle

IMG_SIZE = (64, 64)

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
    Differentiably resize and place an RGBA patch into a canvas.

    patch:
        Tensor with shape [B, C, H, W].
        For the FNO output:
            [B, 4, 64, 64]

    center_x, center_y:
        Patch center in canvas pixel coordinates.

    patch_width, patch_height:
        Desired patch size in canvas pixels.

    Returns:
        Tensor with shape [B, C, canvas_height, canvas_width].
    """
    if patch.ndim != 4:
        raise ValueError(
            f"Expected patch [B,C,H,W], got {tuple(patch.shape)}"
        )

    batch_size = patch.shape[0]
    dtype = patch.dtype
    device = patch.device

    # Convert inputs to tensors while preserving gradients when they
    # are already tensors requiring gradients.
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

    # Convert patch center from pixel coordinates to normalized
    # canvas coordinates.
    center_x_norm = (
        2.0 * center_x / canvas_w_minus_one - 1.0
    )

    center_y_norm = (
        2.0 * center_y / canvas_h_minus_one - 1.0
    )

    # Scale from canvas-normalized coordinates to patch-normalized
    # coordinates.
    scale_x = canvas_w_minus_one / patch_width
    scale_y = canvas_h_minus_one / patch_height

    # Construct theta without breaking the autograd graph.
    theta = torch.stack(
        [
            torch.stack(
                [
                    scale_x,
                    torch.zeros_like(scale_x),
                    -scale_x * center_x_norm,
                ],
                dim=1,
            ),
            torch.stack(
                [
                    torch.zeros_like(scale_y),
                    scale_y,
                    -scale_y * center_y_norm,
                ],
                dim=1,
            ),
        ],
        dim=1,
    )

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


def camera_to_slice_pose(viewpoint_camera, slice_center):
    """
    Compute the camera pose relative to a neural slice.

    Parameters
    ----------
    viewpoint_camera:
        3DGS Camera object with camera_center.

    slice_center:
        Tensor [3] containing the slice center in world coordinates.

    Returns
    -------
    phi:
        Polar angle in radians.

    theta:
        Azimuth angle in radians.

    actual_radius:
        Actual camera-to-slice distance.

    relative:
        Camera position relative to the slice center.
    """
    camera_center = viewpoint_camera.camera_center

    slice_center = slice_center.to(
        device=camera_center.device,
        dtype=camera_center.dtype,
    )

    relative = camera_center - slice_center

    actual_radius = torch.linalg.norm(
        relative
    ).clamp_min(1e-8)

    x = relative[0]
    y = relative[1]
    z = relative[2]

    phi = torch.acos(
        torch.clamp(
            z / actual_radius,
            -1.0,
            1.0,
        )
    )

    theta = torch.atan2(y, x)
    theta = torch.remainder(
        theta,
        2.0 * math.pi,
    )

    return (
        phi,
        theta,
        actual_radius,
        relative,
    )
    
def compute_scene_median_center(gaussian_model):
    """
    Compute a scene-specific robust center from the initialized
    point cloud.
    """
    xyz = gaussian_model.get_xyz.detach()

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise RuntimeError(
            f"Expected point cloud with shape [N,3], got {xyz.shape}"
        )

    return xyz.median(dim=0).values

    return phi, theta, actual_radius, relative

def reconstruct_camera_position(phi, theta, radius):
    """
    Reconstruct a relative camera position from spherical values.
    """
    return torch.stack([
        radius * torch.sin(phi) * torch.cos(theta),
        radius * torch.sin(phi) * torch.sin(theta),
        radius * torch.cos(phi),
    ])
    
def load_fno_checkpoint(checkpoint_path, device):
    """
    Load one single-mode FNO checkpoint and its normalization stats.
    """
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

    state = checkpoint.get("model_state", checkpoint)

    # Do not modify the checkpoint dictionary's original state in place
    state = dict(state)
    state.pop("_metadata", None)

    if "latent_dim" not in checkpoint:
        raise RuntimeError(
            f"Checkpoint does not contain latent_dim: {checkpoint_path}"
        )

    if "param_mean" not in checkpoint:
        raise RuntimeError(
            f"Checkpoint does not contain param_mean: {checkpoint_path}"
        )

    if "param_std" not in checkpoint:
        raise RuntimeError(
            f"Checkpoint does not contain param_std: {checkpoint_path}"
        )

    latent_dim = int(checkpoint["latent_dim"])

    param_mean = np.asarray(
        checkpoint["param_mean"],
        dtype=np.float32,
    )

    param_std = np.asarray(
        checkpoint["param_std"],
        dtype=np.float32,
    )

    if len(param_mean) != latent_dim:
        raise RuntimeError(
            f"param_mean has length {len(param_mean)}, "
            f"but latent_dim={latent_dim}"
        )

    if len(param_std) != latent_dim:
        raise RuntimeError(
            f"param_std has length {len(param_std)}, "
            f"but latent_dim={latent_dim}"
        )

    model = FNOPlusResNetSingle(
        latent_dim=latent_dim,
        img_size=IMG_SIZE,
    ).to(device)

    model.load_state_dict(state)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    checkpoint_mode = checkpoint.get("mode", "unknown")

    print("  latent_dim:", latent_dim)
    print("  checkpoint mode:", checkpoint_mode)

    return model, param_mean, param_std, latent_dim


# ============================================================
# FNO PARAMETER BUILDER
# ============================================================

def build_fno_param_vector(
    dataset,
    template_row,
    mode,
    phi,
    theta,
    fno_radius,
    param_mean,
    param_std,
    device,
):
    """
    Build the exact normalized parameter vector used during FNO
    training.

    The template row supplies:
        shape/sample ID
        shape control points through sample_id
        material values
        environment SH coefficients

    Camera values are replaced with the current 3DGS camera pose.

    The FNO receives canonical radius=2.2, not the actual scene radius.
    """
    if mode not in ("surface", "volume"):
        raise ValueError(f"Invalid mode: {mode}")

    row = template_row.copy()

    phi_value = float(phi.detach().cpu().item())
    theta_value = float(theta.detach().cpu().item())
    radius_value = float(fno_radius)

    row["phi"] = phi_value
    row["theta"] = theta_value
    row["radius"] = radius_value
    row["render_mode"] = mode

    # These variables were zeroed during volume training.
    if mode == "volume":
        row["metallic"] = 0.0
        row["roughness"] = 0.0
        row["specular"] = 0.0

    # This is the same feature builder used by the training dataset.
    raw_np = dataset._build_param_vector_np(row)

    if raw_np.shape[0] != param_mean.shape[0]:
        raise RuntimeError(
            f"{mode} parameter dimension mismatch: "
            f"constructed={raw_np.shape[0]}, "
            f"checkpoint={param_mean.shape[0]}"
        )

    normalized_np = (
        raw_np - param_mean
    ) / param_std

    return torch.from_numpy(
        normalized_np.astype(np.float32)
    ).unsqueeze(0).to(device)


def project_world_point(viewpoint_camera, point_world):
    """
    Project a world-space point into the camera image.

    Returns:
        px, py, depth
    """
    device = point_world.device

    point_h = torch.cat([
        point_world,
        torch.ones(1, device=device),
    ])

    clip = viewpoint_camera.full_proj_transform @ point_h

    if torch.abs(clip[3]) < 1e-8:
        raise RuntimeError("Projected point has invalid homogeneous coordinate.")

    ndc = clip[:3] / clip[3]

    width = float(viewpoint_camera.image_width)
    height = float(viewpoint_camera.image_height)

    px = (ndc[0] + 1.0) * 0.5 * width
    py = (1.0 - ndc[1]) * 0.5 * height

    return px, py, ndc[2]

    
def projected_patch_bbox(
    viewpoint_camera,
    slice_center,
    world_width,
    world_height,
):
    """
    Project an unrotated rectangular slice and return its pixel bbox.

    slice_center: [3]
    world_width/world_height: physical scene dimensions
    """

    device = slice_center.device

    hx = 0.5 * world_width
    hy = 0.5 * world_height

    corners = torch.stack([
        slice_center + torch.tensor([-hx, -hy, 0.0], device=device),
        slice_center + torch.tensor([ hx, -hy, 0.0], device=device),
        slice_center + torch.tensor([ hx,  hy, 0.0], device=device),
        slice_center + torch.tensor([-hx,  hy, 0.0], device=device),
    ])

    pixels = []

    for corner in corners:
        px, py, depth = project_world_point(
            viewpoint_camera,
            corner,
        )
        pixels.append(torch.stack([px, py, depth]))

    pixels = torch.stack(pixels, dim=0)

    min_x = pixels[:, 0].min()
    max_x = pixels[:, 0].max()
    min_y = pixels[:, 1].min()
    max_y = pixels[:, 1].max()

    center_x, center_y, _ = project_world_point(
        viewpoint_camera,
        slice_center,
    )

    return {
        "center_x": center_x,
        "center_y": center_y,
        "width_px": (max_x - min_x).clamp_min(1.0),
        "height_px": (max_y - min_y).clamp_min(1.0),
        "corners_px": pixels,
    }


def resize_neural_patch(rgba, width_px, height_px):
    """
    rgba: [1,4,64,64]
    """
    width_px = max(1, int(round(float(width_px))))
    height_px = max(1, int(round(float(height_px))))

    return F.interpolate(
        rgba,
        size=(height_px, width_px),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    
def paste_patch(canvas, patch, center_x, center_y):
    """
    canvas: [1,4,H,W]
    patch:  [1,4,h,w]
    """

    _, _, H, W = canvas.shape
    _, _, h, w = patch.shape

    cx = int(round(float(center_x)))
    cy = int(round(float(center_y)))

    x0 = cx - w // 2
    y0 = cy - h // 2
    x1 = x0 + w
    y1 = y0 + h

    canvas_x0 = max(0, x0)
    canvas_y0 = max(0, y0)
    canvas_x1 = min(W, x1)
    canvas_y1 = min(H, y1)

    if canvas_x0 >= canvas_x1 or canvas_y0 >= canvas_y1:
        return canvas

    patch_x0 = canvas_x0 - x0
    patch_y0 = canvas_y0 - y0
    patch_x1 = patch_x0 + canvas_x1 - canvas_x0
    patch_y1 = patch_y0 + canvas_y1 - canvas_y0

    canvas[
        :,
        :,
        canvas_y0:canvas_y1,
        canvas_x0:canvas_x1,
    ] = patch[
        :,
        :,
        patch_y0:patch_y1,
        patch_x0:patch_x1,
    ]

    return canvas
    

def alpha_over(back, front):
    """
    Both tensors: [1,4,H,W].
    Assumes premultiplied RGB.
    """
    back_color = back[:, :3]
    back_alpha = back[:, 3:4]

    front_color = front[:, :3]
    front_alpha = front[:, 3:4]

    out_color = front_color + (
        1.0 - front_alpha
    ) * back_color

    out_alpha = front_alpha + (
        1.0 - front_alpha
    ) * back_alpha

    return torch.cat([out_color, out_alpha], dim=1)
    