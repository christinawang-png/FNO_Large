#!/usr/bin/env python

import sys
import math
import copy
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import imageio.v2 as imageio

import torch
import torch.nn as nn
import torch.nn.functional as F

import random


# ============================================================
# PATH SETUP
# ============================================================

REPO_DIR = Path(__file__).resolve().parent
FNO_ROOT = REPO_DIR.parent

sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(FNO_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from scene import Scene, GaussianModel
from arguments import ModelParams

from train_premult_single_mode import FNOPlusResNetSingle

from neural_parameters import (
    LearnableNeuralSlice,
    build_tensor_fno_vector,
)

from neural_lifecycle import (
    make_gradient_stats,
    accumulate_gradient_stats,
    finalize_gradient_stats,
    collect_slice_statistics,
    prune_slices,
    split_top_slices,
    voxel_chunk_seeds,
)


# ============================================================
# CONFIGURATION
# ============================================================

SURFACE_CHECKPOINT = (
    FNO_ROOT / "fno_premult_surface_final.pt"
)

VOLUME_CHECKPOINT = (
    FNO_ROOT / "fno_premult_volume_epoch025.pt"
)

DEFAULT_OUTPUT_DIR = (
    REPO_DIR / "neural_scene_random_init_outputs"
)

IMG_SIZE = (64, 64)
FNO_RADIUS = 2.2

NUM_GLOBAL_ENVS = 128
SH_ORDER = 2

CTRL_LEVELS = np.array(
    [0.1, 0.3, 0.5, 0.7, 0.9],
    dtype=np.float32,
)

SIGMA_VALUES = np.array(
    [0.02, 0.08, 0.20, 0.50, 0.70],
    dtype=np.float32,
)

BACKGROUND_RGB = [1.0, 1.0, 1.0]

# Parameter regularization.
POSITION_REG_WEIGHT = 1e-5
SIZE_REG_WEIGHT = 1e-3
PARAMETER_REG_WEIGHT = 1e-5

# Lighting regularization.
GLOBAL_SH_BOUND = 0.005
GLOBAL_SH_REG_WEIGHT = 1e-2
NEIGHBOR_SH_REG_WEIGHT = 1e-1

# Gradient clipping.
MAX_GRAD_NORM = 1.0

# Optimization schedule.
SHAPE_WARMUP_ITERS = 500
LIGHTING_START_ITERS = 1000

# Early stage: shape/material adapt fast.
EARLY_PLACEMENT_LR = 1e-2
EARLY_SHAPE_LR = 5e-2
EARLY_MATERIAL_LR = 7e-2

# Middle stage: stabilize geometry.
MID_PLACEMENT_LR = 5e-4
MID_SHAPE_LR = 1e-3
MID_MATERIAL_LR = 2e-3

# Final stage: mostly refine appearance/light.
LATE_PLACEMENT_LR = 1e-4
LATE_SHAPE_LR = 5e-4
LATE_MATERIAL_LR = 5e-4

# Lighting is deliberately very slow.
LOCAL_SH_LR = 1e-7
GLOBAL_SH_LR = 1e-8

# Local patch rendering. Avoid full HxW layer per slice.
USE_LOCAL_ROI_RENDERING = True

# Extra room around the estimated projected square footprint.
ROI_MARGIN_PIXELS = 16

# Safety cap for a single local region.
MAX_ROI_SIDE_PIXELS = 1024


# ============================================================
# CHECKPOINT LOADING
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

    state = dict(checkpoint["model_state"])
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

    # Frozen model weights, but gradients through input vectors
    # remain valid.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    print(
        "  latent_dim:",
        latent_dim,
        "mode:",
        checkpoint.get("mode", "unknown"),
    )

    return model, param_mean, param_std, latent_dim


# ============================================================
# PROCEDURAL SH BANK
# Matches Blender generation code.
# ============================================================

def sh_lm_list(order):
    pairs = []

    for l in range(order + 1):
        for m in range(-l, l + 1):
            pairs.append((l, m))

    return pairs


def sh_for_global_env(env_id, order=2):
    pairs = sh_lm_list(order)

    coeffs = np.zeros(
        (len(pairs), 3),
        dtype=np.float32,
    )

    u = env_id / max(
        1.0,
        float(NUM_GLOBAL_ENVS - 1),
    )

    t = 2.0 * math.pi * u

    r = 0.5 + 0.4 * math.sin(t)
    g = 0.5 + 0.4 * math.sin(
        t + 2.0 * math.pi / 3.0
    )
    b = 0.5 + 0.4 * math.sin(
        t + 4.0 * math.pi / 3.0
    )

    rgb = np.array(
        [r, g, b],
        dtype=np.float32,
    )

    gray = np.full(
        3,
        rgb.mean(),
        dtype=np.float32,
    )

    if u < 1.0 / 3.0:
        alpha = 0.1
    elif u < 2.0 / 3.0:
        alpha = 0.5
    else:
        alpha = 1.0

    rgb_scale = (
        (1.0 - alpha) * gray
        + alpha * rgb
    )

    coeffs[0, :] = rgb_scale * 0.4

    for idx, (l, m) in enumerate(pairs):
        if l != 1:
            continue

        if m == -1:
            coeffs[idx, :] = rgb_scale * (
                0.2 * math.sin(2.0 * math.pi * u)
            )
        elif m == 0:
            coeffs[idx, :] = rgb_scale * (
                0.2 * math.cos(2.0 * math.pi * u)
            )
        elif m == 1:
            coeffs[idx, :] = rgb_scale * (
                0.2 * math.sin(
                    2.0 * math.pi * u + 1.0
                )
            )

    for idx, (l, m) in enumerate(pairs):
        if l == 2 and m == 0:
            coeffs[idx, :] += rgb_scale * (
                0.05 * math.cos(4.0 * math.pi * u)
            )

    return coeffs


def build_sh_bank(device):
    rows = []

    for env_id in range(NUM_GLOBAL_ENVS):
        coeffs = sh_for_global_env(
            env_id,
            order=SH_ORDER,
        )

        # [9,3] -> [27], matching training input order.
        rows.append(coeffs.reshape(-1))

    sh_bank = torch.tensor(
        np.stack(rows, axis=0),
        dtype=torch.float32,
        device=device,
    )

    print("Built SH bank:", tuple(sh_bank.shape))
    return sh_bank


# ============================================================
# RANDOM SLICE PRIOR
# ============================================================

def random_slice_init(rng, shared_sh):
    return {
        "ctrl_values": rng.choice(
            CTRL_LEVELS,
            size=4,
            replace=True,
        ).astype(np.float32),

        "sigma": float(
            rng.choice(SIGMA_VALUES)
        ),

        "hue": float(
            rng.uniform(0.0, 1.0)
        ),

        "saturation": float(
            rng.uniform(0.3, 0.9)
        ),

        "opacity": float(
            rng.uniform(0.1, 1.0)
        ),

        "roughness": float(
            rng.uniform(0.1, 0.9)
        ),

        "metallic": float(
            rng.choice([0.0, 1.0])
        ),

        "specular": 0.5,

        # Initial slice SH is the scene's shared environment.
        "sh_values": shared_sh.detach().clone(),
    }


def make_slice_from_seed(
    seed,
    mode,
    init_params,
    device,
    optimize_sh,
):
    if mode == "surface":
        roughness = init_params["roughness"]
        metallic = init_params["metallic"]
        specular = init_params["specular"]
    else:
        roughness = 0.5
        metallic = 0.0
        specular = 0.0

    return LearnableNeuralSlice(
        mode=mode,
        center=seed["center"].to(
            device=device,
            dtype=torch.float32,
        ),
        world_size=float(seed["world_size"]),
        ctrl_values=init_params["ctrl_values"],
        sigma=init_params["sigma"],
        hue=init_params["hue"],
        saturation=init_params["saturation"],
        opacity=init_params["opacity"],
        roughness=roughness,
        sh_values=init_params["sh_values"],
        metallic=metallic,
        specular=specular,
        optimize_environment=optimize_sh,
    ).to(device)


# ============================================================
# SHARED LIGHTING / NEURAL SCENE
# ============================================================

class SharedLighting(nn.Module):
    def __init__(
        self,
        initial_global_sh,
        optimize_sh,
        bound=GLOBAL_SH_BOUND,
    ):
        super().__init__()

        initial_global_sh = torch.as_tensor(
            initial_global_sh,
            dtype=torch.float32,
        )

        self.register_buffer(
            "initial_global_sh",
            initial_global_sh.clone(),
        )

        self.bound = float(bound)

        self.raw_global_sh_delta = nn.Parameter(
            torch.zeros_like(initial_global_sh),
            requires_grad=optimize_sh,
        )

    @property
    def global_sh(self):
        return self.initial_global_sh + (
            self.bound
            * torch.tanh(self.raw_global_sh_delta)
        )


class NeuralScene(nn.Module):
    def __init__(
        self,
        slices,
        initial_global_sh,
        optimize_sh,
    ):
        super().__init__()

        self.slices = nn.ModuleList(slices)

        self.lighting = SharedLighting(
            initial_global_sh=initial_global_sh,
            optimize_sh=optimize_sh,
        )

    @property
    def global_sh(self):
        return self.lighting.global_sh


# ============================================================
# CAMERA / PROJECTION HELPERS
# ============================================================

def camera_to_slice_pose(camera, slice_center):
    camera_center = camera.camera_center

    slice_center = slice_center.to(
        device=camera_center.device,
        dtype=camera_center.dtype,
    )

    relative = camera_center - slice_center

    actual_radius = torch.linalg.norm(
        relative
    ).clamp_min(1e-8)

    phi = torch.acos(
        torch.clamp(
            relative[2] / actual_radius,
            -1.0,
            1.0,
        )
    )

    theta = torch.atan2(
        relative[1],
        relative[0],
    )

    theta = torch.remainder(
        theta,
        2.0 * math.pi,
    )

    return phi, theta, actual_radius, relative


def project_world_point(
    camera,
    point_world,
    flip_projection_y=False,
):
    matrix = camera.full_proj_transform

    point_world = point_world.to(
        device=matrix.device,
        dtype=matrix.dtype,
    )

    point_h = torch.cat([
        point_world,
        torch.ones(
            1,
            dtype=matrix.dtype,
            device=matrix.device,
        ),
    ])

    # 3DGS matrix convention is row-vector multiplication.
    clip = point_h @ matrix

    ndc = clip[:3] / clip[3].clamp_min(1e-8)

    pixel_x = (
        (ndc[0] + 1.0)
        * 0.5
        * float(camera.image_width)
    )

    if flip_projection_y:
        pixel_y = (
            (ndc[1] + 1.0)
            * 0.5
            * float(camera.image_height)
        )
    else:
        pixel_y = (
            (1.0 - ndc[1])
            * 0.5
            * float(camera.image_height)
        )

    return pixel_x, pixel_y, ndc[2]


# ============================================================
# BATCHED PATCH PLACEMENT
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
    Inputs:
        patch:      [B,4,64,64]
        center_x:   [B] or scalar
        center_y:   [B] or scalar
        patch_size: [B] or scalar

    Returns:
        [B,4,H,W]
    """
    if patch.ndim != 4:
        raise ValueError(
            f"Expected [B,C,H,W], got {tuple(patch.shape)}"
        )

    B, C, _, _ = patch.shape

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
        center_x = center_x.expand(B)

    if center_y.numel() == 1:
        center_y = center_y.expand(B)

    if patch_size.numel() == 1:
        patch_size = patch_size.expand(B)

    if (
        center_x.numel() != B
        or center_y.numel() != B
        or patch_size.numel() != B
    ):
        raise RuntimeError(
            "Patch-placement batch dimensions do not match."
        )

    patch_size = patch_size.clamp_min(2.0)

    canvas_wm1 = float(max(canvas_width - 1, 1))
    canvas_hm1 = float(max(canvas_height - 1, 1))

    center_x_norm = (
        2.0 * center_x / canvas_wm1 - 1.0
    )

    center_y_norm = (
        2.0 * center_y / canvas_hm1 - 1.0
    )

    scale_x = canvas_wm1 / patch_size
    scale_y = canvas_hm1 / patch_size

    zero = torch.zeros_like(scale_x)

    theta_row_0 = torch.stack(
        [
            scale_x,
            zero,
            -scale_x * center_x_norm,
        ],
        dim=1,
    )

    theta_row_1 = torch.stack(
        [
            zero,
            scale_y,
            -scale_y * center_y_norm,
        ],
        dim=1,
    )

    theta = torch.stack(
        [
            theta_row_0,
            theta_row_1,
        ],
        dim=1,
    )

    grid = F.affine_grid(
        theta,
        size=(B, C, canvas_height, canvas_width),
        align_corners=True,
    )

    return F.grid_sample(
        patch,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    

def place_patch_uniform_roi(
    patch,
    center_x,
    center_y,
    patch_size,
    canvas_height,
    canvas_width,
    margin_pixels=ROI_MARGIN_PIXELS,
    max_roi_side=MAX_ROI_SIDE_PIXELS,
):
    """
    Render ONE premultiplied RGBA patch into only its local ROI.

    Inputs:
        patch:      [1,4,64,64]
        center_x/y: scalar tensors in full-image pixel coordinates
        patch_size: scalar tensor, desired square side in pixels

    Returns:
        roi_rgba: [1,4,roi_h,roi_w]
        x0, y0: Python integer location of ROI in the full canvas

    Notes:
        - patch position remains differentiable inside the ROI.
        - ROI bounds use detached integer values, so crossing an ROI
          boundary is not differentiable.
        - Use margin_pixels to make boundary changes infrequent.
    """
    if patch.shape[0] != 1:
        raise ValueError(
            f"ROI placement currently expects batch size 1, got {patch.shape}"
        )

    device = patch.device
    dtype = patch.dtype

    # We use detached values only to select the local storage region.
    # The placement grid below continues to use original tensor values.
    center_x_value = float(center_x.detach().cpu().item())
    center_y_value = float(center_y.detach().cpu().item())
    patch_size_value = float(patch_size.detach().cpu().item())

    # Clamp pathological sizes before determining ROI.
    patch_size_value = max(2.0, patch_size_value)

    half = 0.5 * patch_size_value + float(margin_pixels)

    x0 = max(0, int(math.floor(center_x_value - half)))
    x1 = min(canvas_width, int(math.ceil(center_x_value + half)))

    y0 = max(0, int(math.floor(center_y_value - half)))
    y1 = min(canvas_height, int(math.ceil(center_y_value + half)))

    roi_w = x1 - x0
    roi_h = y1 - y0

    if roi_w <= 0 or roi_h <= 0:
        return None, x0, y0

    # Limit pathological nearby slices.
    if roi_w > max_roi_side or roi_h > max_roi_side:
        roi_w = min(roi_w, max_roi_side)
        roi_h = min(roi_h, max_roi_side)

        # Recenter capped ROI around the projected center.
        x0 = max(
            0,
            min(
                canvas_width - roi_w,
                int(round(center_x_value - roi_w / 2)),
            ),
        )

        y0 = max(
            0,
            min(
                canvas_height - roi_h,
                int(round(center_y_value - roi_h / 2)),
            ),
        )

    # Coordinates of ROI center in full-canvas normalized coordinates.
    canvas_wm1 = float(max(canvas_width - 1, 1))
    canvas_hm1 = float(max(canvas_height - 1, 1))

    center_x_norm = (
        2.0 * center_x / canvas_wm1 - 1.0
    )

    center_y_norm = (
        2.0 * center_y / canvas_hm1 - 1.0
    )

    # Full image -> patch transform scale.
    scale_x = canvas_wm1 / patch_size.clamp_min(2.0)
    scale_y = canvas_hm1 / patch_size.clamp_min(2.0)

    # We must construct the local grid manually because affine_grid
    # normally assumes its output covers the whole canvas.
    ys = torch.arange(
        y0,
        y0 + roi_h,
        dtype=dtype,
        device=device,
    )

    xs = torch.arange(
        x0,
        x0 + roi_w,
        dtype=dtype,
        device=device,
    )

    yy, xx = torch.meshgrid(
        ys,
        xs,
        indexing="ij",
    )

    # Convert full image ROI pixels to normalized full-canvas coords.
    full_x_norm = 2.0 * xx / canvas_wm1 - 1.0
    full_y_norm = 2.0 * yy / canvas_hm1 - 1.0

    # Map full-canvas normalized coordinates -> patch coordinates.
    patch_x = scale_x * (
        full_x_norm - center_x_norm
    )

    patch_y = scale_y * (
        full_y_norm - center_y_norm
    )

    grid = torch.stack(
        [patch_x, patch_y],
        dim=-1,
    ).unsqueeze(0)  # [1, roi_h, roi_w, 2]

    roi_rgba = F.grid_sample(
        patch,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    return roi_rgba, x0, y0
    

def alpha_over_roi(
    canvas,
    roi_rgba,
    x0,
    y0,
):
    """
    Alpha-composite one premultiplied ROI patch over a full canvas.

    canvas:
        [1,4,H,W]

    roi_rgba:
        [1,4,roi_h,roi_w]

    x0/y0:
        integer top-left ROI placement on canvas.
    """
    if roi_rgba is None:
        return canvas

    _, _, canvas_h, canvas_w = canvas.shape
    _, _, roi_h, roi_w = roi_rgba.shape

    x1 = min(x0 + roi_w, canvas_w)
    y1 = min(y0 + roi_h, canvas_h)

    if x0 >= x1 or y0 >= y1:
        return canvas

    valid_w = x1 - x0
    valid_h = y1 - y0

    roi = roi_rgba[
        :,
        :,
        :valid_h,
        :valid_w,
    ]

    back = canvas[
        :,
        :,
        y0:y1,
        x0:x1,
    ]

    back_color = back[:, :3]
    back_alpha = back[:, 3:4]

    front_color = roi[:, :3]
    front_alpha = roi[:, 3:4]

    output_color = (
        front_color
        + (1.0 - front_alpha) * back_color
    )

    output_alpha = (
        front_alpha
        + (1.0 - front_alpha) * back_alpha
    )

    output_roi = torch.cat(
        [output_color, output_alpha],
        dim=1,
    )

    # This CopySlices operation preserves gradients from the output
    # canvas back to output_roi / roi_rgba.
    canvas = canvas.clone()
    canvas[
        :,
        :,
        y0:y1,
        x0:x1,
    ] = output_roi

    return canvas


def run_fno_in_chunks(
    model,
    params,
    batch_size,
):
    """
    params: [N,D]
    returns: [N,4,64,64]
    """
    if params.shape[0] == 0:
        raise RuntimeError("Cannot run FNO on empty batch.")

    outputs = []

    for start in range(
        0,
        params.shape[0],
        batch_size,
    ):
        end = min(
            start + batch_size,
            params.shape[0],
        )

        outputs.append(
            model(params[start:end])
        )

    return torch.cat(
        outputs,
        dim=0,
    )


# ============================================================
# BATCHED NEURAL SCENE RENDERER
# ============================================================

class NeuralSceneRenderer(nn.Module):
    def __init__(
        self,
        surface_model,
        volume_model,
        surface_mean,
        surface_std,
        volume_mean,
        volume_std,
        fno_radius=FNO_RADIUS,
        fno_batch_size=16,
        placement_batch_size=8,
        flip_projection_y=False,
        flip_fno_vertical=False,
    ):
        super().__init__()

        self.surface_model = surface_model
        self.volume_model = volume_model

        self.surface_mean = surface_mean
        self.surface_std = surface_std
        self.volume_mean = volume_mean
        self.volume_std = volume_std

        self.fno_radius = float(fno_radius)
        self.fno_batch_size = int(fno_batch_size)
        self.placement_batch_size = int(
            placement_batch_size
        )

        self.flip_projection_y = bool(
            flip_projection_y
        )

        self.flip_fno_vertical = bool(
            flip_fno_vertical
        )

        for model in [
            self.surface_model,
            self.volume_model,
        ]:
            model.eval()

            for p in model.parameters():
                p.requires_grad_(False)

    @staticmethod
    def alpha_over(back, front):
        back_color = back[:, :3]
        back_alpha = back[:, 3:4]

        front_color = front[:, :3]
        front_alpha = front[:, 3:4]

        out_color = (
            front_color
            + (1.0 - front_alpha)
            * back_color
        )

        out_alpha = (
            front_alpha
            + (1.0 - front_alpha)
            * back_alpha
        )

        return torch.cat(
            [out_color, out_alpha],
            dim=1,
        )

    @staticmethod
    def composite_layers_far_to_near(layers):
        """
        layers:
            [B,4,H,W]

        layers[0] is farthest and layers[-1] is nearest.
        """
        if layers.ndim != 4 or layers.shape[1] != 4:
            raise ValueError(
                f"Expected [B,4,H,W], got {tuple(layers.shape)}"
            )

        color = layers[:, :3]
        alpha = layers[:, 3:4]

        transmittance = 1.0 - alpha

        reversed_trans = transmittance.flip(0)

        cumulative = torch.cumprod(
            reversed_trans,
            dim=0,
        )

        shifted = torch.cat(
            [
                torch.ones_like(cumulative[:1]),
                cumulative[:-1],
            ],
            dim=0,
        )

        weight = shifted.flip(0)

        output_color = torch.sum(
            color * weight,
            dim=0,
            keepdim=True,
        )

        output_alpha = 1.0 - torch.prod(
            transmittance,
            dim=0,
            keepdim=True,
        )

        return torch.cat(
            [output_color, output_alpha],
            dim=1,
        )

    def render_slice_list_batched(
        self,
        camera,
        slices,
        shared_sh,
        placement_batch_size=None,
        collect_diagnostics=False,
    ):
        
        active_slices = [
            s for s in slices
            if is_slice_potentially_visible(
                camera=camera,
                neural_slice=s,
                fno_radius=self.fno_radius,
                margin_px=64.0,
                min_patch_size_px=2.0,
            )
        ]
        
        if len(active_slices) == 0:
            H = int(camera.image_height)
            W = int(camera.image_width)
        
            return (
                torch.zeros(
                    1, 4, H, W,
                    device=camera.camera_center.device,
                    dtype=torch.float32,
                ),
                [],
            )
        
        slices = active_slices
        """
        Batched rendering path.

        Per camera:
            - build all slice records
            - run surface FNO in chunks
            - run volume FNO in chunks
            - place patches in chunks
            - GPU sort far-to-near
            - composite
        """
        if len(slices) == 0:
            raise RuntimeError("No slices to render.")

        if placement_batch_size is None:
            placement_batch_size = (
                self.placement_batch_size
            )

        device = camera.camera_center.device

        canvas_h = int(camera.image_height)
        canvas_w = int(camera.image_width)

        records = []

        # ----------------------------------------------------
        # Phase A: build per-slice tensor records.
        # ----------------------------------------------------
        for slice_index, neural_slice in enumerate(slices):
            phi, theta, actual_radius, relative = (
                camera_to_slice_pose(
                    camera,
                    neural_slice.center,
                )
            )

            if neural_slice.mode == "surface":
                param_mean = self.surface_mean
                param_std = self.surface_std
            elif neural_slice.mode == "volume":
                param_mean = self.volume_mean
                param_std = self.volume_std
            else:
                raise ValueError(
                    f"Unknown mode: {neural_slice.mode}"
                )

            param_vec = build_tensor_fno_vector(
                neural_slice=neural_slice,
                param_mean=param_mean,
                param_std=param_std,
                phi=phi,
                theta=theta,
                fno_radius=self.fno_radius,
                device=device,
                shared_sh=shared_sh,
            )

            center_x, center_y, center_depth = (
                project_world_point(
                    camera,
                    neural_slice.center,
                    flip_projection_y=self.flip_projection_y,
                )
            )

            patch_size = (
                float(canvas_h)
                * neural_slice.world_size
                * self.fno_radius
                / actual_radius
            )

            records.append(
                {
                    "slice_index": slice_index,
                    "mode": neural_slice.mode,
                    "param_vec": param_vec,
                    "center_x": center_x,
                    "center_y": center_y,
                    "patch_size": patch_size,
                    "actual_radius": actual_radius,
                    "center_depth": center_depth,
                    "phi": phi,
                    "theta": theta,
                    "relative": relative,
                }
            )

        surface_indices = [
            i
            for i, r in enumerate(records)
            if r["mode"] == "surface"
        ]

        volume_indices = [
            i
            for i, r in enumerate(records)
            if r["mode"] == "volume"
        ]

        num_slices = len(records)
        patches_by_index = [None] * num_slices

        # ----------------------------------------------------
        # Phase B: batched FNO calls by mode.
        # ----------------------------------------------------
        if surface_indices:
            surface_params = torch.cat(
                [
                    records[i]["param_vec"]
                    for i in surface_indices
                ],
                dim=0,
            )

            surface_patches = run_fno_in_chunks(
                self.surface_model,
                surface_params,
                batch_size=self.fno_batch_size,
            )

            if self.flip_fno_vertical:
                surface_patches = torch.flip(
                    surface_patches,
                    dims=[2],
                )

            for local_i, record_i in enumerate(
                surface_indices
            ):
                patches_by_index[record_i] = (
                    surface_patches[
                        local_i:local_i + 1
                    ]
                )

        if volume_indices:
            volume_params = torch.cat(
                [
                    records[i]["param_vec"]
                    for i in volume_indices
                ],
                dim=0,
            )

            volume_patches = run_fno_in_chunks(
                self.volume_model,
                volume_params,
                batch_size=self.fno_batch_size,
            )

            if self.flip_fno_vertical:
                volume_patches = torch.flip(
                    volume_patches,
                    dims=[2],
                )

            for local_i, record_i in enumerate(
                volume_indices
            ):
                patches_by_index[record_i] = (
                    volume_patches[
                        local_i:local_i + 1
                    ]
                )

        if any(
            patch is None
            for patch in patches_by_index
        ):
            raise RuntimeError(
                "At least one slice did not receive an FNO patch."
            )

        all_patches = torch.cat(
            patches_by_index,
            dim=0,
        )

        center_x = torch.stack(
            [
                r["center_x"]
                for r in records
            ],
            dim=0,
        )

        center_y = torch.stack(
            [
                r["center_y"]
                for r in records
            ],
            dim=0,
        )

        patch_sizes = torch.stack(
            [
                r["patch_size"]
                for r in records
            ],
            dim=0,
        )

        depths = torch.stack(
            [
                r["actual_radius"]
                for r in records
            ],
            dim=0,
        )

        # Far -> near on GPU.
        sort_indices = torch.argsort(
            depths.detach(),
            descending=True,
        )

        sorted_patches = all_patches[sort_indices]
        sorted_center_x = center_x[sort_indices]
        sorted_center_y = center_y[sort_indices]
        sorted_patch_sizes = patch_sizes[sort_indices]

        # --------------------------------------------------------
        # Phase C: local ROI placement and compositing.
        #
        # sorted_patches are already ordered far -> near.
        # --------------------------------------------------------
        composite = torch.zeros(
            1,
            4,
            canvas_h,
            canvas_w,
            dtype=all_patches.dtype,
            device=device,
        )
        
        for sorted_index in range(num_slices):
            patch = sorted_patches[
                sorted_index:sorted_index + 1
            ]
        
            roi_rgba, x0, y0 = place_patch_uniform_roi(
                patch=patch,
                center_x=sorted_center_x[sorted_index],
                center_y=sorted_center_y[sorted_index],
                patch_size=sorted_patch_sizes[sorted_index],
                canvas_height=canvas_h,
                canvas_width=canvas_w,
                margin_pixels=ROI_MARGIN_PIXELS,
                max_roi_side=MAX_ROI_SIDE_PIXELS,
            )
        
            composite = alpha_over_roi(
                canvas=composite,
                roi_rgba=roi_rgba,
                x0=x0,
                y0=y0,
            )

        diagnostics = []

        if collect_diagnostics:
            order_cpu = sort_indices.detach().cpu().tolist()

            for record_index in order_cpu:
                r = records[record_index]

                diagnostics.append(
                    {
                        "slice_index": r["slice_index"],
                        "mode": r["mode"],
                        "phi": r["phi"],
                        "theta": r["theta"],
                        "actual_radius": r["actual_radius"],
                        "relative": r["relative"],
                        "center_x": r["center_x"],
                        "center_y": r["center_y"],
                        "center_depth": r["center_depth"],
                        "patch_size": r["patch_size"],
                    }
                )

        return composite.clamp(0.0, 1.0), diagnostics

    def forward(
        self,
        camera,
        neural_scene,
    ):
        return self.render_slice_list_batched(
            camera=camera,
            slices=neural_scene.slices,
            shared_sh=neural_scene.global_sh,
            placement_batch_size=self.placement_batch_size,
            collect_diagnostics=False,
        )


# ============================================================
# LOSS / REGULARIZATION
# ============================================================

def visible_rgb_from_rgba(rgba):
    background = torch.tensor(
        BACKGROUND_RGB,
        dtype=rgba.dtype,
        device=rgba.device,
    ).view(1, 3, 1, 1)

    return rgba[:, :3] + (
        1.0 - rgba[:, 3:4]
    ) * background


def image_loss(predicted_rgba, target_rgb):
    predicted_rgb = visible_rgb_from_rgba(
        predicted_rgba
    )

    return (
        0.5 * F.l1_loss(predicted_rgb, target_rgb)
        + 0.5 * F.mse_loss(predicted_rgb, target_rgb)
    )


def parameter_regularization(neural_scene):
    total = torch.zeros(
        (),
        dtype=torch.float32,
        device=neural_scene.slices[0].center.device,
    )

    for s in neural_scene.slices:
        values = s.fno_values(
            shared_sh=neural_scene.global_sh
        )

        center_loss = F.mse_loss(
            s.center,
            s.initial_center,
        )

        size_loss = (
            (
                s.world_size
                - s.initial_world_size
            )
            / s.initial_world_size.clamp_min(1e-4)
        ) ** 2

        appearance_loss = torch.mean(
            (values["ctrl"] - s.initial_ctrl) ** 2
        )

        appearance_loss = appearance_loss + (
            values["sigma"] - s.initial_sigma
        ) ** 2

        appearance_loss = appearance_loss + (
            values["hue"] - s.initial_hue
        ) ** 2

        appearance_loss = appearance_loss + (
            values["saturation"]
            - s.initial_saturation
        ) ** 2

        appearance_loss = appearance_loss + (
            values["opacity"]
            - s.initial_opacity
        ) ** 2

        if s.mode == "surface":
            appearance_loss = appearance_loss + (
                values["roughness"]
                - s.initial_roughness
            ) ** 2

        local_sh_loss = torch.mean(
            s.local_sh_delta ** 2
        )

        total = total + (
            POSITION_REG_WEIGHT * center_loss
            + SIZE_REG_WEIGHT * size_loss
            + PARAMETER_REG_WEIGHT * appearance_loss
            + GLOBAL_SH_REG_WEIGHT * local_sh_loss
        )

    return total


def build_knn_neighbors(neural_scene, k=3):
    if len(neural_scene.slices) <= 1:
        return []

    centers = torch.stack(
        [
            s.center.detach()
            for s in neural_scene.slices
        ],
        dim=0,
    )

    distances = torch.cdist(centers, centers)
    distances.fill_diagonal_(float("inf"))

    pairs = set()

    for i in range(len(neural_scene.slices)):
        indices = torch.topk(
            distances[i],
            k=min(
                k,
                len(neural_scene.slices) - 1,
            ),
            largest=False,
        ).indices.tolist()

        for j in indices:
            pairs.add(tuple(sorted((i, j))))

    return sorted(pairs)


def neighbor_sh_regularization(
    neural_scene,
    neighbor_pairs,
):
    if not neighbor_pairs:
        return torch.zeros(
            (),
            dtype=torch.float32,
            device=neural_scene.slices[0].center.device,
        )

    effective_sh = [
        s.fno_values(
            shared_sh=neural_scene.global_sh
        )["sh"]
        for s in neural_scene.slices
    ]

    total = torch.zeros(
        (),
        dtype=torch.float32,
        device=effective_sh[0].device,
    )

    for i, j in neighbor_pairs:
        total = total + torch.mean(
            (effective_sh[i] - effective_sh[j]) ** 2
        )

    return NEIGHBOR_SH_REG_WEIGHT * total


def global_sh_regularization(neural_scene):
    return GLOBAL_SH_REG_WEIGHT * torch.mean(
        (
            neural_scene.global_sh
            - neural_scene.lighting.initial_global_sh
        ) ** 2
    )


# ============================================================
# OPTIMIZATION SCHEDULE
# ============================================================

def get_optimization_stage(
    iteration,
    optimize_sh,
):
    if iteration < SHAPE_WARMUP_ITERS:
        return "early"

    if (
        iteration < LIGHTING_START_ITERS
        or not optimize_sh
    ):
        return "mid"

    return "lighting"


def make_optimizer(
    neural_scene,
    stage,
    optimize_sh,
):
    placement_params = []
    shape_params = []
    material_params = []
    local_sh_params = []
    global_sh_params = []

    for s in neural_scene.slices:
        placement_params.extend([
            s.center,
            s.raw_world_size,
        ])

        shape_params.extend([
            s.raw_ctrl,
            s.raw_sigma,
        ])

        material_params.extend([
            s.raw_hue,
            s.raw_saturation,
            s.raw_opacity,
            s.raw_roughness,
        ])

        if optimize_sh:
            local_sh_params.append(
                s.raw_local_sh_delta
            )

    if optimize_sh:
        global_sh_params.append(
            neural_scene.lighting.raw_global_sh_delta
        )

    if stage == "early":
        placement_lr = EARLY_PLACEMENT_LR
        shape_lr = EARLY_SHAPE_LR
        material_lr = EARLY_MATERIAL_LR
        local_sh_lr = 0.0
        global_sh_lr = 0.0

    elif stage == "mid":
        placement_lr = MID_PLACEMENT_LR
        shape_lr = MID_SHAPE_LR
        material_lr = MID_MATERIAL_LR
        local_sh_lr = 0.0
        global_sh_lr = 0.0

    elif stage == "lighting":
        placement_lr = LATE_PLACEMENT_LR
        shape_lr = LATE_SHAPE_LR
        material_lr = LATE_MATERIAL_LR
        local_sh_lr = LOCAL_SH_LR
        global_sh_lr = GLOBAL_SH_LR

    else:
        raise ValueError(
            f"Unknown optimizer stage: {stage}"
        )

    groups = [
        {
            "params": placement_params,
            "lr": placement_lr,
        },
        {
            "params": shape_params,
            "lr": shape_lr,
        },
        {
            "params": material_params,
            "lr": material_lr,
        },
    ]

    if optimize_sh and local_sh_lr > 0:
        groups.append(
            {
                "params": local_sh_params,
                "lr": local_sh_lr,
            }
        )

    if optimize_sh and global_sh_lr > 0:
        groups.append(
            {
                "params": global_sh_params,
                "lr": global_sh_lr,
            }
        )

    return torch.optim.Adam(groups)


# ============================================================
# MODE CANDIDATE SELECTION
# ============================================================

@torch.no_grad()
def evaluate_scene_loss(
    renderer,
    neural_scene,
    cameras,
):
    total = 0.0

    for camera in cameras:
        predicted_rgba, _ = renderer(
            camera,
            neural_scene,
        )

        target = camera.original_image

        if target.ndim == 3:
            target = target.unsqueeze(0)

        target = target.to(
            device=predicted_rgba.device,
            dtype=predicted_rgba.dtype,
        )

        total += image_loss(
            predicted_rgba,
            target,
        ).item()

    return total / max(len(cameras), 1)


@torch.no_grad()
def choose_initial_slice_mode(
    renderer,
    selected_slices,
    base_candidate,
    initial_global_sh,
    optimize_sh,
    cameras,
    device,
):
    surface_candidate = copy.deepcopy(
        base_candidate
    )
    surface_candidate.mode = "surface"

    volume_candidate = copy.deepcopy(
        base_candidate
    )
    volume_candidate.mode = "volume"
    volume_candidate.metallic_value.fill_(0.0)
    volume_candidate.specular_value.fill_(0.0)

    surface_scene = NeuralScene(
        slices=selected_slices + [surface_candidate],
        initial_global_sh=initial_global_sh,
        optimize_sh=optimize_sh,
    ).to(device)

    volume_scene = NeuralScene(
        slices=selected_slices + [volume_candidate],
        initial_global_sh=initial_global_sh,
        optimize_sh=optimize_sh,
    ).to(device)

    surface_score = evaluate_scene_loss(
        renderer,
        surface_scene,
        cameras,
    )

    volume_score = evaluate_scene_loss(
        renderer,
        volume_scene,
        cameras,
    )

    if surface_score <= volume_score:
        chosen = surface_candidate
        chosen_mode = "surface"
    else:
        chosen = volume_candidate
        chosen_mode = "volume"

    print(
        f"  initial candidate: "
        f"surface={surface_score:.6f}, "
        f"volume={volume_score:.6f}, "
        f"chosen={chosen_mode}"
    )

    return chosen


@torch.no_grad()
def set_slice_mode(neural_slice, mode):
    candidate = copy.deepcopy(neural_slice)
    candidate.mode = mode

    if mode == "volume":
        candidate.metallic_value.fill_(0.0)
        candidate.specular_value.fill_(0.0)

    return candidate


@torch.no_grad()
def score_child_pair_modes(
    renderer,
    neural_scene,
    parent_index,
    child_a,
    child_b,
    mode_a,
    mode_b,
    cameras,
):
    candidate_a = set_slice_mode(
        child_a,
        mode_a,
    )

    candidate_b = set_slice_mode(
        child_b,
        mode_b,
    )

    temporary_slices = (
        list(neural_scene.slices[:parent_index])
        + [candidate_a, candidate_b]
        + list(neural_scene.slices[parent_index + 1:])
    )

    total = 0.0

    for camera in cameras:
        predicted_rgba, _ = (
            renderer.render_slice_list_batched(
                camera=camera,
                slices=temporary_slices,
                shared_sh=neural_scene.global_sh,
                placement_batch_size=(
                    renderer.placement_batch_size
                ),
                collect_diagnostics=False,
            )
        )

        target = camera.original_image

        if target.ndim == 3:
            target = target.unsqueeze(0)

        target = target.to(
            device=predicted_rgba.device,
            dtype=predicted_rgba.dtype,
        )

        total += image_loss(
            predicted_rgba,
            target,
        ).item()

    return (
        total / max(len(cameras), 1),
        candidate_a,
        candidate_b,
    )


@torch.no_grad()
def choose_split_child_modes(
    renderer,
    neural_scene,
    parent_index,
    child_a,
    child_b,
    cameras,
):
    combinations = [
        ("surface", "surface"),
        ("surface", "volume"),
        ("volume", "surface"),
        ("volume", "volume"),
    ]

    best_score = float("inf")
    best_children = None
    best_modes = None

    for mode_a, mode_b in combinations:
        score, candidate_a, candidate_b = (
            score_child_pair_modes(
                renderer=renderer,
                neural_scene=neural_scene,
                parent_index=parent_index,
                child_a=child_a,
                child_b=child_b,
                mode_a=mode_a,
                mode_b=mode_b,
                cameras=cameras,
            )
        )

        print(
            f"  split parent={parent_index} "
            f"candidate=({mode_a}, {mode_b}) "
            f"loss={score:.6f}"
        )

        if score < best_score:
            best_score = score
            best_modes = (mode_a, mode_b)
            best_children = (
                candidate_a,
                candidate_b,
            )

    print(
        f"  selected child modes "
        f"for parent={parent_index}: "
        f"{best_modes}, loss={best_score:.6f}"
    )

    return best_children


# ============================================================
# CONTRIBUTION PRUNING
# ============================================================

@torch.no_grad()
def mean_scene_loss(
    renderer,
    neural_scene,
    cameras,
):
    total = 0.0

    for camera in cameras:
        prediction, _ = (
            renderer.render_slice_list_batched(
                camera=camera,
                slices=list(neural_scene.slices),
                shared_sh=neural_scene.global_sh,
                placement_batch_size=(
                    renderer.placement_batch_size
                ),
                collect_diagnostics=False,
            )
        )

        target = camera.original_image

        if target.ndim == 3:
            target = target.unsqueeze(0)

        target = target.to(
            device=prediction.device,
            dtype=prediction.dtype,
        )

        total += image_loss(
            prediction,
            target,
        ).item()

    return total / max(len(cameras), 1)


@torch.no_grad()
def slice_removal_contribution(
    renderer,
    neural_scene,
    remove_index,
    cameras,
    full_loss,
):
    remaining = [
        s
        for i, s in enumerate(neural_scene.slices)
        if i != remove_index
    ]

    if len(remaining) == 0:
        return float("inf"), float("inf")

    total = 0.0

    for camera in cameras:
        prediction, _ = (
            renderer.render_slice_list_batched(
                camera=camera,
                slices=remaining,
                shared_sh=neural_scene.global_sh,
                placement_batch_size=(
                    renderer.placement_batch_size
                ),
                collect_diagnostics=False,
            )
        )

        target = camera.original_image

        if target.ndim == 3:
            target = target.unsqueeze(0)

        target = target.to(
            device=prediction.device,
            dtype=prediction.dtype,
        )

        total += image_loss(
            prediction,
            target,
        ).item()

    loss_without = total / max(len(cameras), 1)
    contribution = loss_without - full_loss

    return contribution, loss_without


@torch.no_grad()
def choose_contribution_prune_indices(
    renderer,
    neural_scene,
    cameras,
    slice_stats,
    max_candidates=4,
    max_prunes=2,
    contribution_threshold=1e-4,
):
    if len(neural_scene.slices) <= 1:
        return []

    weak_order = sorted(
        range(len(slice_stats)),
        key=lambda i: (
            slice_stats[i]["mean_alpha_mass"],
            slice_stats[i]["max_alpha"],
            slice_stats[i]["visible_views"],
        ),
    )

    candidate_indices = weak_order[
        :min(max_candidates, len(weak_order))
    ]

    full_loss = mean_scene_loss(
        renderer=renderer,
        neural_scene=neural_scene,
        cameras=cameras,
    )

    print(
        f"[CONTRIBUTION PRUNE] "
        f"full_loss={full_loss:.8f}"
    )

    prune_indices = []

    for slice_index in candidate_indices:
        contribution, loss_without = (
            slice_removal_contribution(
                renderer=renderer,
                neural_scene=neural_scene,
                remove_index=slice_index,
                cameras=cameras,
                full_loss=full_loss,
            )
        )

        print(
            f"  slice={slice_index:02d} "
            f"loss_without={loss_without:.8f} "
            f"contribution={contribution:.8e}"
        )

        if contribution <= contribution_threshold:
            prune_indices.append(slice_index)

    max_allowed = min(
        max_prunes,
        len(neural_scene.slices) - 1,
    )

    return prune_indices[:max_allowed]


# ============================================================
# CHECKPOINTING / RESUME
# ============================================================

def serialize_slice(slice_obj):
    """
    Store constructor metadata. State dict later restores
    exact learnable parameters and buffers.
    """
    return {
        "mode": slice_obj.mode,
        "optimize_environment": bool(
            slice_obj.optimize_environment
        ),
        "local_sh_bound": float(
            slice_obj.local_sh_bound
        ),
    }


def save_neural_scene_checkpoint(
    path,
    neural_scene,
    optimizer,
    iteration,
    metadata=None,
):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "iteration": int(iteration),
            "scene_state": neural_scene.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "slice_specs": [
                serialize_slice(s)
                for s in neural_scene.slices
            ],
            "initial_global_sh": (
                neural_scene.lighting.initial_global_sh
                .detach()
                .cpu()
            ),
            "metadata": metadata or {},
        },
        path,
    )


def rebuild_neural_scene_from_checkpoint(
    checkpoint,
    device,
    optimize_sh,
):
    """
    Recreate the correct number/type of slices, then load
    saved state including raw parameters and buffers.
    """
    slices = []

    scene_state = checkpoint["scene_state"]

    for i, spec in enumerate(
        checkpoint["slice_specs"]
    ):
        prefix = f"slices.{i}."

        initial_center = scene_state[
            prefix + "initial_center"
        ]

        initial_world_size = scene_state[
            prefix + "initial_world_size"
        ]

        initial_ctrl = scene_state[
            prefix + "initial_ctrl"
        ]

        initial_sigma = scene_state[
            prefix + "initial_sigma"
        ]

        initial_hue = scene_state[
            prefix + "initial_hue"
        ]

        initial_saturation = scene_state[
            prefix + "initial_saturation"
        ]

        initial_opacity = scene_state[
            prefix + "initial_opacity"
        ]

        initial_roughness = scene_state[
            prefix + "initial_roughness"
        ]

        initial_sh = scene_state[
            prefix + "initial_sh"
        ]

        metallic_value = scene_state[
            prefix + "metallic_value"
        ]

        specular_value = scene_state[
            prefix + "specular_value"
        ]

        slice_obj = LearnableNeuralSlice(
            mode=spec["mode"],
            center=initial_center,
            world_size=float(initial_world_size),
            ctrl_values=initial_ctrl,
            sigma=float(initial_sigma),
            hue=float(initial_hue),
            saturation=float(initial_saturation),
            opacity=float(initial_opacity),
            roughness=float(initial_roughness),
            sh_values=initial_sh,
            metallic=float(metallic_value),
            specular=float(specular_value),
            optimize_environment=(
                bool(spec["optimize_environment"])
                and optimize_sh
            ),
            local_sh_bound=float(
                spec["local_sh_bound"]
            ),
        ).to(device)

        slices.append(slice_obj)

    neural_scene = NeuralScene(
        slices=slices,
        initial_global_sh=checkpoint[
            "initial_global_sh"
        ].to(device),
        optimize_sh=optimize_sh,
    ).to(device)

    neural_scene.load_state_dict(
        checkpoint["scene_state"]
    )

    return neural_scene


# ============================================================
# PREVIEW SAVING
# ============================================================

@torch.no_grad()
def save_checkpoint_previews(
    renderer,
    neural_scene,
    cameras,
    output_dir,
    iteration,
    max_views=3,
):
    preview_dir = (
        Path(output_dir)
        / "previews"
        / f"iter_{iteration:06d}"
    )

    preview_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for view_idx, camera in enumerate(
        cameras[:max_views]
    ):
        predicted_rgba, _ = renderer(
            camera,
            neural_scene,
        )

        predicted_rgba = predicted_rgba.clamp(
            0.0,
            1.0,
        )

        visible_rgb = visible_rgb_from_rgba(
            predicted_rgba
        )[0]

        target_rgb = camera.original_image.detach()

        pred_np = visible_rgb.detach().cpu().numpy()
        target_np = target_rgb.detach().cpu().numpy()
        alpha_np = (
            predicted_rgba[0, 3]
            .detach()
            .cpu()
            .numpy()
        )

        pred_hwc = np.transpose(
            pred_np,
            (1, 2, 0),
        )

        target_hwc = np.transpose(
            target_np,
            (1, 2, 0),
        )

        alpha_rgb = np.repeat(
            alpha_np[..., None],
            3,
            axis=2,
        )

        imageio.imwrite(
            preview_dir / (
                f"view_{view_idx:02d}_prediction.png"
            ),
            (
                np.clip(pred_hwc, 0.0, 1.0)
                * 255.0
                + 0.5
            ).astype(np.uint8),
        )

        imageio.imwrite(
            preview_dir / (
                f"view_{view_idx:02d}_alpha.png"
            ),
            (
                np.clip(alpha_rgb, 0.0, 1.0)
                * 255.0
                + 0.5
            ).astype(np.uint8),
        )

        imageio.imwrite(
            preview_dir / (
                f"view_{view_idx:02d}_target.png"
            ),
            (
                np.clip(target_hwc, 0.0, 1.0)
                * 255.0
                + 0.5
            ).astype(np.uint8),
        )

        comparison = np.concatenate(
            [
                np.clip(target_hwc, 0.0, 1.0),
                np.clip(pred_hwc, 0.0, 1.0),
            ],
            axis=1,
        )

        imageio.imwrite(
            preview_dir / (
                f"view_{view_idx:02d}_comparison.png"
            ),
            (
                comparison * 255.0 + 0.5
            ).astype(np.uint8),
        )

    print("Saved previews:", preview_dir)


# ============================================================
# SMALL IMAGE SAVERS
# ============================================================

def save_rgb_chw(rgb_chw, path):
    if torch.is_tensor(rgb_chw):
        rgb_chw = rgb_chw.detach().cpu().numpy()

    rgb_hwc = np.transpose(
        rgb_chw,
        (1, 2, 0),
    )

    imageio.imwrite(
        path,
        (
            np.clip(rgb_hwc, 0.0, 1.0)
            * 255.0
            + 0.5
        ).astype(np.uint8),
    )


def save_rgba_chw(rgba_chw, path):
    if torch.is_tensor(rgba_chw):
        rgba_chw = rgba_chw.detach().cpu().numpy()

    rgba_hwc = np.transpose(
        rgba_chw,
        (1, 2, 0),
    )

    imageio.imwrite(
        path,
        (
            np.clip(rgba_hwc, 0.0, 1.0)
            * 255.0
            + 0.5
        ).astype(np.uint8),
    )


def save_alpha_hw(alpha_hw, path):
    if torch.is_tensor(alpha_hw):
        alpha_hw = alpha_hw.detach().cpu().numpy()

    alpha_rgb = np.repeat(
        alpha_hw[..., None],
        3,
        axis=2,
    )

    imageio.imwrite(
        path,
        (
            np.clip(alpha_rgb, 0.0, 1.0)
            * 255.0
            + 0.5
        ).astype(np.uint8),
    )
    

def is_slice_potentially_visible(
    camera,
    neural_slice,
    fno_radius,
    margin_px=64.0,
    min_patch_size_px=2.0,
):
    """
    Fast differentiable-ish visibility screening.
    Returns a Python bool for render culling.

    This is intentionally discrete; do not use it for gradient-based
    position optimization near image boundaries.
    """
    center_x, center_y, depth = project_world_point(
        camera,
        neural_slice.center,
        flip_projection_y=False,
    )

    relative = camera.camera_center - neural_slice.center
    radius = torch.linalg.norm(relative).clamp_min(1e-8)

    patch_size = (
        float(camera.image_height)
        * neural_slice.world_size
        * float(fno_radius)
        / radius
    )

    # Convert only these small scalar decisions to Python.
    x = float(center_x.detach().cpu())
    y = float(center_y.detach().cpu())
    z = float(depth.detach().cpu())
    size = float(patch_size.detach().cpu())

    H = float(camera.image_height)
    W = float(camera.image_width)

    # Outside screen with margin, behind camera, or too tiny.
    return (
        z > 0.0
        and size >= min_patch_size_px
        and x >= -margin_px
        and x <= W + margin_px
        and y >= -margin_px
        and y <= H + margin_px
    )


# ============================================================
# MAIN
# ============================================================

def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass

    parser = ArgumentParser(
        description=(
            "Dataset-free neural scene optimization with "
            "batched FNO rendering."
        )
    )

    # Adds standard 3DGS arguments:
    # --source_path, --model_path, --resolution, etc.
    lp = ModelParams(parser)

    parser.add_argument(
        "--camera_start",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--num_cameras",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=2000,
        help="Final total iteration count.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--min_points_per_voxel",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--max_initial_slices",
        type=int,
        default=0,
        help=(
            "Maximum voxel seeds. Use 0 for every valid voxel."
        ),
    )

    parser.add_argument(
        "--disable_initial_mode_selection",
        action="store_true",
        help=(
            "Skip expensive initial surface/volume selection. "
            "Initial modes are randomized."
        ),
    )

    parser.add_argument(
        "--mode_selection_cameras",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--disable_child_mode_selection",
        action="store_true",
    )

    parser.add_argument(
        "--child_selection_cameras",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--optimize_sh",
        action="store_true",
    )

    parser.add_argument(
        "--global_sh_warmup_iterations",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--split_interval",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--max_splits_per_update",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max_slices",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--prune_interval",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--contribution_prune_candidates",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max_prunes_per_update",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--contribution_threshold",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--prune_selection_cameras",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--fno_batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--placement_batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--preview_views",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--flip_projection_y",
        action="store_true",
    )

    parser.add_argument(
        "--flip_fno_vertical",
        action="store_true",
    )

    parser.add_argument(
        "--surface_checkpoint",
        type=str,
        default=str(SURFACE_CHECKPOINT),
    )

    parser.add_argument(
        "--volume_checkpoint",
        type=str,
        default=str(VOLUME_CHECKPOINT),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = torch.device("cuda")
    rng = np.random.default_rng(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Using device:", device)
    print("Random seed:", args.seed)
    print("Optimize SH:", args.optimize_sh)
    print("FNO batch size:", args.fno_batch_size)
    print(
        "Placement batch size:",
        args.placement_batch_size,
    )

    # --------------------------------------------------------
    # Frozen FNO models / normalization.
    # --------------------------------------------------------
    surface_model, surface_mean, surface_std, surface_dim = (
        load_fno_checkpoint(
            args.surface_checkpoint,
            device,
        )
    )

    volume_model, volume_mean, volume_std, volume_dim = (
        load_fno_checkpoint(
            args.volume_checkpoint,
            device,
        )
    )

    if surface_dim != 44 or volume_dim != 44:
        raise RuntimeError(
            f"Expected latent_dim=44, got "
            f"surface={surface_dim}, volume={volume_dim}"
        )

    renderer = NeuralSceneRenderer(
        surface_model=surface_model,
        volume_model=volume_model,
        surface_mean=surface_mean,
        surface_std=surface_std,
        volume_mean=volume_mean,
        volume_std=volume_std,
        fno_radius=FNO_RADIUS,
        fno_batch_size=args.fno_batch_size,
        placement_batch_size=args.placement_batch_size,
        flip_projection_y=args.flip_projection_y,
        flip_fno_vertical=args.flip_fno_vertical,
    ).to(device)

    # --------------------------------------------------------
    # Load 3DGS scene / camera list / point cloud.
    # --------------------------------------------------------
    scene_args = lp.extract(args)

    if not scene_args.model_path:
        scene_args.model_path = str(
            REPO_DIR / "output" / "random_init_scene"
        )

    Path(scene_args.model_path).mkdir(
        parents=True,
        exist_ok=True,
    )

    gaussian_model = GaussianModel(
        args.sh_degree,
    )

    scene = Scene(
        scene_args,
        gaussian_model,
        shuffle=False,
        resolution_scales=[
            float(args.resolution)
        ],
    )

    cameras = scene.getTrainCameras(
        scale=float(args.resolution)
    )

    camera_end = min(
        args.camera_start + args.num_cameras,
        len(cameras),
    )

    cameras_used = cameras[
        args.camera_start:camera_end
    ]

    if len(cameras_used) == 0:
        raise RuntimeError("No cameras selected.")

    print(
        "Using cameras:",
        args.camera_start,
        "through",
        camera_end - 1,
    )

    xyz = gaussian_model.get_xyz.detach()

    scene_center = xyz.median(
        dim=0
    ).values.to(device)

    print(
        "Scene center:",
        scene_center.detach().cpu().numpy(),
    )

    sh_bank = build_sh_bank(device)

    # --------------------------------------------------------
    # Initialize or resume neural scene.
    # --------------------------------------------------------
    start_iteration = 0

    if args.resume_checkpoint is not None:
        print("Resuming from:", args.resume_checkpoint)

        checkpoint = torch.load(
            args.resume_checkpoint,
            map_location=device,
            weights_only=False,
        )

        neural_scene = rebuild_neural_scene_from_checkpoint(
            checkpoint=checkpoint,
            device=device,
            optimize_sh=args.optimize_sh,
        )

        start_iteration = (
            int(checkpoint["iteration"]) + 1
        )

        initial_env_id = checkpoint[
            "metadata"
        ].get("initial_env_id", -1)

    else:
        initial_env_id = int(
            rng.integers(0, NUM_GLOBAL_ENVS)
        )

        initial_global_sh = sh_bank[
            initial_env_id
        ].detach().clone()

        max_chunks = (
            None
            if args.max_initial_slices <= 0
            else args.max_initial_slices
        )

        seeds = voxel_chunk_seeds(
            xyz=xyz,
            voxel_size=args.voxel_size,
            min_points=args.min_points_per_voxel,
            max_chunks=max_chunks,
        )

        if len(seeds) == 0:
            raise RuntimeError(
                "No valid voxel seeds. Increase voxel size "
                "or lower min-points-per-voxel."
            )

        print(
            "Voxel initialization produced",
            len(seeds),
            "seeds.",
        )

        mode_selection_cameras = cameras_used[
            :min(
                args.mode_selection_cameras,
                len(cameras_used),
            )
        ]

        slices = []

        for i, seed in enumerate(seeds):
            init_params = random_slice_init(
                rng,
                initial_global_sh,
            )

            base_candidate = make_slice_from_seed(
                seed=seed,
                mode="surface",
                init_params=init_params,
                device=device,
                optimize_sh=args.optimize_sh,
            )

            if args.disable_initial_mode_selection:
                chosen_mode = random.choice(
                    ["surface", "volume"]
                )

                chosen = copy.deepcopy(
                    base_candidate
                )

                chosen.mode = chosen_mode

                if chosen_mode == "volume":
                    chosen.metallic_value.fill_(0.0)
                    chosen.specular_value.fill_(0.0)

            else:
                chosen = choose_initial_slice_mode(
                    renderer=renderer,
                    selected_slices=slices,
                    base_candidate=base_candidate,
                    initial_global_sh=initial_global_sh,
                    optimize_sh=args.optimize_sh,
                    cameras=mode_selection_cameras,
                    device=device,
                )

                chosen_mode = chosen.mode

            slices.append(chosen)

            if i < 20 or i % 50 == 0:
                print(
                    f"seed={i:03d} "
                    f"points={seed['num_points']} "
                    f"size={seed['world_size']:.3f} "
                    f"mode={chosen_mode}"
                )

        neural_scene = NeuralScene(
            slices=slices,
            initial_global_sh=initial_global_sh,
            optimize_sh=args.optimize_sh,
        ).to(device)

    current_stage = get_optimization_stage(
        iteration=start_iteration,
        optimize_sh=args.optimize_sh,
    )

    # Global SH stays frozen until lighting stage.
    if args.optimize_sh:
        enable_global_sh = (
            start_iteration
            >= args.global_sh_warmup_iterations
            and current_stage == "lighting"
        )

        neural_scene.lighting.raw_global_sh_delta.requires_grad_(
            enable_global_sh
        )

    optimizer = make_optimizer(
        neural_scene=neural_scene,
        stage=current_stage,
        optimize_sh=args.optimize_sh,
    )

    # Restore optimizer state only after topology is rebuilt.
    if args.resume_checkpoint is not None:
        try:
            optimizer.load_state_dict(
                checkpoint["optimizer_state"]
            )
            print("Restored optimizer state.")
        except Exception as exc:
            print(
                "[WARN] Could not restore optimizer state:",
                exc,
            )

    print(
        f"[SCHEDULE] Starting stage: {current_stage}"
    )

    child_selection_cameras = cameras_used[
        :min(
            args.child_selection_cameras,
            len(cameras_used),
        )
    ]

    # Defined outside resume/init branch so splitting works
    # after checkpoint resume too.
    def split_mode_selector(
        parent_index,
        child_a,
        child_b,
    ):
        if args.disable_child_mode_selection:
            return child_a, child_b

        chosen_a, chosen_b = choose_split_child_modes(
            renderer=renderer,
            neural_scene=neural_scene,
            parent_index=parent_index,
            child_a=child_a,
            child_b=child_b,
            cameras=child_selection_cameras,
        )

        return chosen_a, chosen_b

    gradient_stats = make_gradient_stats(
        len(neural_scene.slices)
    )

    neighbor_pairs = build_knn_neighbors(
        neural_scene,
        k=3,
    )

    print(
        "Initial slice count:",
        len(neural_scene.slices),
    )

    # --------------------------------------------------------
    # Batched-renderer sanity check.
    # --------------------------------------------------------
    with torch.no_grad():
        sanity_rgba, _ = renderer(
            cameras_used[0],
            neural_scene,
        )

    if not torch.isfinite(sanity_rgba).all():
        raise RuntimeError(
            "Non-finite output from batched renderer."
        )

    print(
        "Batched renderer sanity output:",
        tuple(sanity_rgba.shape),
    )

    # --------------------------------------------------------
    # Optimization loop.
    # --------------------------------------------------------
    for iteration in range(
        start_iteration,
        args.iterations,
    ):
        new_stage = get_optimization_stage(
            iteration=iteration,
            optimize_sh=args.optimize_sh,
        )

        if new_stage != current_stage:
            current_stage = new_stage

            if (
                args.optimize_sh
                and current_stage == "lighting"
            ):
                neural_scene.lighting.raw_global_sh_delta.requires_grad_(
                    iteration
                    >= args.global_sh_warmup_iterations
                )

            optimizer = make_optimizer(
                neural_scene=neural_scene,
                stage=current_stage,
                optimize_sh=args.optimize_sh,
            )

            print(
                f"[SCHEDULE] Iteration {iteration}: "
                f"stage={current_stage}"
            )

        if (
            args.optimize_sh
            and iteration == args.global_sh_warmup_iterations
            and current_stage == "lighting"
        ):
            neural_scene.lighting.raw_global_sh_delta.requires_grad_(
                True
            )

            optimizer = make_optimizer(
                neural_scene=neural_scene,
                stage=current_stage,
                optimize_sh=args.optimize_sh,
            )

            print(
                "[LIGHTING] Enabled global SH optimization."
            )

        optimizer.zero_grad(set_to_none=True)

        # Kept as GPU tensor to avoid CPU sync per camera.
        total_image_loss = torch.zeros(
            (),
            device=device,
            dtype=torch.float32,
        )

        valid_camera_count = 0

        for camera in cameras_used:
            predicted_rgba, _ = renderer(
                camera,
                neural_scene,
            )

            target_rgb = camera.original_image.detach()

            if target_rgb.ndim == 3:
                target_rgb = target_rgb.unsqueeze(0)

            target_rgb = target_rgb.to(
                device=device,
                dtype=predicted_rgba.dtype,
            )

            camera_loss = image_loss(
                predicted_rgba,
                target_rgb,
            )

            if (
                not torch.isfinite(camera_loss)
                or not torch.isfinite(predicted_rgba).all()
            ):
                print(
                    f"[WARN] Non-finite render/loss at "
                    f"iteration={iteration}, "
                    f"camera={camera.image_name}; skipped."
                )
                continue

            scaled_loss = camera_loss / len(
                cameras_used
            )

            scaled_loss.backward()

            total_image_loss = (
                total_image_loss
                + scaled_loss.detach()
            )

            valid_camera_count += 1

            del predicted_rgba
            del target_rgb
            del camera_loss
            del scaled_loss

        if valid_camera_count == 0:
            print(
                f"[WARN] No valid cameras at iteration "
                f"{iteration}; skipping update."
            )
            optimizer.zero_grad(set_to_none=True)
            continue

        loss_parameter_reg = parameter_regularization(
            neural_scene
        )

        if args.optimize_sh:
            loss_neighbor_sh = neighbor_sh_regularization(
                neural_scene,
                neighbor_pairs,
            )

            loss_global_sh = global_sh_regularization(
                neural_scene
            )
        else:
            loss_neighbor_sh = torch.zeros(
                (),
                device=device,
            )

            loss_global_sh = torch.zeros(
                (),
                device=device,
            )

        loss_regularization = (
            loss_parameter_reg
            + loss_neighbor_sh
            + loss_global_sh
        )

        if not torch.isfinite(loss_regularization):
            print(
                f"[WARN] Non-finite regularization at "
                f"iteration={iteration}; skipping update."
            )
            optimizer.zero_grad(set_to_none=True)
            continue

        loss_regularization.backward()

        accumulate_gradient_stats(
            neural_scene,
            gradient_stats,
        )

        bad_gradients = []

        for name, parameter in (
            neural_scene.named_parameters()
        ):
            if (
                parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
            ):
                bad_gradients.append(name)

        if bad_gradients:
            print(
                f"[WARN] Non-finite gradients at "
                f"iteration={iteration}; skipping update."
            )
            print("Bad parameters:", bad_gradients)
            optimizer.zero_grad(set_to_none=True)
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(
            neural_scene.parameters(),
            max_norm=MAX_GRAD_NORM,
            error_if_nonfinite=True,
        )

        if iteration % 100 == 0:
            print(
                f"[DEBUG] iter={iteration} "
                f"grad_norm={grad_norm.item():.6e}"
            )

        optimizer.step()

        # ----------------------------------------------------
        # Structural updates.
        # ----------------------------------------------------
        after_warmup = (
            iteration >= args.warmup_iterations
        )

        do_prune = (
            after_warmup
            and args.prune_interval > 0
            and iteration > 0
            and iteration % args.prune_interval == 0
        )

        do_split = (
            after_warmup
            and args.split_interval > 0
            and iteration > 0
            and iteration % args.split_interval == 0
        )

        if do_prune or do_split:
            gradient_summary = finalize_gradient_stats(
                gradient_stats
            )

            slice_stats = collect_slice_statistics(
                renderer=renderer,
                neural_scene=neural_scene,
                cameras=cameras_used,
            )

            print("\n[STRUCTURE UPDATE]")
            print(
                "Slice count before:",
                len(neural_scene.slices),
            )

            if do_prune:
                prune_cameras = cameras_used[
                    :min(
                        args.prune_selection_cameras,
                        len(cameras_used),
                    )
                ]

                prune_indices = (
                    choose_contribution_prune_indices(
                        renderer=renderer,
                        neural_scene=neural_scene,
                        cameras=prune_cameras,
                        slice_stats=slice_stats,
                        max_candidates=(
                            args.contribution_prune_candidates
                        ),
                        max_prunes=args.max_prunes_per_update,
                        contribution_threshold=(
                            args.contribution_threshold
                        ),
                    )
                )

                if prune_indices:
                    print(
                        "Pruning slices:",
                        prune_indices,
                    )

                    prune_slices(
                        neural_scene,
                        prune_indices,
                    )

                    # Do not split during same update.
                    do_split = False

            if do_split:
                split_indices = split_top_slices(
                    neural_scene=neural_scene,
                    slice_stats=slice_stats,
                    gradient_stats=gradient_summary,
                    max_splits=args.max_splits_per_update,
                    max_total_slices=args.max_slices,
                    min_alpha_mass=10.0,
                    min_patch_size=40.0,
                    mode_selector=split_mode_selector,
                )

                if split_indices:
                    print(
                        "Split slices:",
                        split_indices,
                    )

            print(
                "Slice count after:",
                len(neural_scene.slices),
            )

            # Keep current schedule stage after topology change.
            current_stage = get_optimization_stage(
                iteration=iteration,
                optimize_sh=args.optimize_sh,
            )

            optimizer = make_optimizer(
                neural_scene=neural_scene,
                stage=current_stage,
                optimize_sh=args.optimize_sh,
            )

            gradient_stats = make_gradient_stats(
                len(neural_scene.slices)
            )

            neighbor_pairs = build_knn_neighbors(
                neural_scene,
                k=3,
            )

        # ----------------------------------------------------
        # Checkpoint / preview.
        # ----------------------------------------------------
        if (
            args.checkpoint_interval > 0
            and iteration > 0
            and iteration % args.checkpoint_interval == 0
        ):
            checkpoint_path = output_dir / (
                f"checkpoint_iter_{iteration:06d}.pt"
            )

            save_neural_scene_checkpoint(
                path=checkpoint_path,
                neural_scene=neural_scene,
                optimizer=optimizer,
                iteration=iteration,
                metadata={
                    "num_slices": len(neural_scene.slices),
                    "initial_env_id": initial_env_id,
                    "optimize_sh": args.optimize_sh,
                    "fno_radius": FNO_RADIUS,
                },
            )

            save_checkpoint_previews(
                renderer=renderer,
                neural_scene=neural_scene,
                cameras=cameras_used,
                output_dir=output_dir,
                iteration=iteration,
                max_views=args.preview_views,
            )

            print(
                "Saved checkpoint:",
                checkpoint_path,
            )

        # ----------------------------------------------------
        # Logging.
        # ----------------------------------------------------
        if (
            iteration % 25 == 0
            or iteration == args.iterations - 1
        ):
            effective_global_delta = (
                neural_scene.global_sh
                - neural_scene.lighting.initial_global_sh
            )

            total_value = (
                total_image_loss
                + loss_regularization.detach()
            )

            print(
                f"iter={iteration:05d} "
                f"loss={total_value.item():.8f} "
                f"image={total_image_loss.item():.8f} "
                f"reg={loss_regularization.item():.8f} "
                f"slices={len(neural_scene.slices)}"
            )

            print(
                "  global SH raw delta norm:",
                neural_scene.lighting.raw_global_sh_delta
                .detach()
                .norm()
                .item(),
            )

            print(
                "  global SH effective delta norm:",
                effective_global_delta.detach()
                .norm()
                .item(),
            )

            print(
                "  global SH max abs delta:",
                effective_global_delta.detach()
                .abs()
                .max()
                .item(),
            )

            for i, s in enumerate(
                neural_scene.slices
            ):
                values = s.fno_values(
                    shared_sh=neural_scene.global_sh
                )

                print(
                    f"  slice={i:03d} "
                    f"mode={s.mode} "
                    f"size={s.world_size.item():.4f} "
                    f"sigma={values['sigma'].item():.4f} "
                    f"opacity={values['opacity'].item():.4f} "
                    f"hue={values['hue'].item():.4f}"
                )

    # --------------------------------------------------------
    # Final output.
    # --------------------------------------------------------
    first_camera = cameras_used[0]

    with torch.no_grad():
        final_rgba, _ = renderer(
            first_camera,
            neural_scene,
        )

    final_rgba = final_rgba.clamp(0.0, 1.0)

    final_np = final_rgba[0].cpu().numpy()
    target_rgb = first_camera.original_image.detach()

    save_rgba_chw(
        final_np,
        output_dir / "final_rgba.png",
    )

    save_rgb_chw(
        final_np[:3],
        output_dir / "final_premult_rgb.png",
    )

    save_rgb_chw(
        visible_rgb_from_rgba(final_rgba)[0],
        output_dir / "final_white_background_rgb.png",
    )

    save_alpha_hw(
        final_np[3],
        output_dir / "final_alpha.png",
    )

    save_rgb_chw(
        target_rgb,
        output_dir / "target_rgb.png",
    )

    target_np = target_rgb.cpu().numpy()
    visible_np = (
        visible_rgb_from_rgba(final_rgba)[0]
        .detach()
        .cpu()
        .numpy()
    )

    comparison = np.concatenate(
        [
            np.transpose(target_np, (1, 2, 0)),
            np.transpose(visible_np, (1, 2, 0)),
        ],
        axis=1,
    )

    imageio.imwrite(
        output_dir / "target_vs_final.png",
        (
            np.clip(comparison, 0.0, 1.0)
            * 255.0
            + 0.5
        ).astype(np.uint8),
    )

    # Save final checkpoint too.
    final_checkpoint = output_dir / (
        f"checkpoint_iter_{args.iterations:06d}_final.pt"
    )

    save_neural_scene_checkpoint(
        path=final_checkpoint,
        neural_scene=neural_scene,
        optimizer=optimizer,
        iteration=args.iterations,
        metadata={
            "num_slices": len(neural_scene.slices),
            "initial_env_id": initial_env_id,
            "optimize_sh": args.optimize_sh,
            "fno_radius": FNO_RADIUS,
        },
    )

    print("Saved final checkpoint:", final_checkpoint)
    print("Done.")
    print("Outputs:", output_dir)


if __name__ == "__main__":
    main()