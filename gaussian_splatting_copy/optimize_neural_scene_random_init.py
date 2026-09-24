#!/usr/bin/env python
"""
Optimize a scene composed of local neural implicit-B-spline slices.

Each slice has:
    - 3D center and world size;
    - 8 trilinear implicit B-spline corner controls;
    - sigma;
    - RGB/material/opacity parameters;
    - local SH lighting residual.

The pretrained FNO models are frozen. Optimization updates only slice
parameters and optional SH lighting, then renders each predicted local
RGBA patch into a full image.

Compatible with:
    implicit_model.py
    neural_parameters.py
    neural_lifecycle.py
    tile_roi_renderer.py
    tile_patch_renderer.py
"""

from __future__ import annotations

import copy
import math
import random
import sys
from argparse import ArgumentParser
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ============================================================
# PATH SETUP
# ============================================================

REPO_DIR = Path(__file__).resolve().parent
FNO_ROOT = REPO_DIR.parent

sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(FNO_ROOT))

# ============================================================
# PROJECT IMPORTS
# ============================================================

from scene import Scene, GaussianModel
from arguments import ModelParams

from implicit_model import ImplicitFNOImageModel

from neural_parameters import (
    FNO_FEATURE_DIM,
    LearnableNeuralSlice,
    build_tensor_fno_vector,
)

from neural_lifecycle import (
    accumulate_gradient_stats,
    choose_batched_prune_candidates,
    choose_batched_split_candidates,
    collect_batched_lifecycle_stats,
    finalize_gradient_stats,
    make_gradient_stats,
    prune_slices,
    split_top_slices,
    voxel_chunk_seeds,
)

from tiled_roi_renderer import render_rois_to_tiled_canvas
from tile_patch_renderer import TilePatchRenderer


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_SURFACE_CHECKPOINT = (
    FNO_ROOT
    / "checkpoints_implicit_modes16_l1"
    / "surface"
    / "best.pt"
)

DEFAULT_VOLUME_CHECKPOINT = (
    FNO_ROOT
    / "checkpoints_implicit_modes16_l1"
    / "volume"
    / "best.pt"
)

DEFAULT_OUTPUT_DIR = REPO_DIR / "neural_scene_outputs"

# Must match the 4-level implicit B-spline dataset.
CTRL_LEVELS = np.array(
    [-0.5, -1.0 / 6.0, 1.0 / 6.0, 0.5],
    dtype=np.float32,
)

SIGMA_VALUES = np.array(
    [0.02, 0.08, 0.20, 0.50, 0.70],
    dtype=np.float32,
)

NUM_GLOBAL_ENVS = 128
SH_ORDER = 2

# Used only for projected local patch footprint:
#
# patch_size_px ≈ image_height * world_size * FNO_RADIUS / camera_distance
#
# It is NOT an FNO conditioning feature.
FNO_RADIUS = 2.2

BACKGROUND_RGB = (0.0, 0.0, 0.0)

# Regularization.
POSITION_REG_WEIGHT = 1e-5
SIZE_REG_WEIGHT = 1e-3
PARAMETER_REG_WEIGHT = 1e-5
LOCAL_SH_REG_WEIGHT = 1e-2
GLOBAL_SH_REG_WEIGHT = 1e-2
NEIGHBOR_SH_REG_WEIGHT = 1e-1

GLOBAL_SH_BOUND = 0.005
MAX_GRAD_NORM = 1.0

# Optimization schedule.
SHAPE_WARMUP_ITERS = 100
LIGHTING_START_ITERS = 300

EARLY_PLACEMENT_LR = 1e-3
EARLY_SHAPE_LR = 5e-2
EARLY_MATERIAL_LR = 7e-2

MID_PLACEMENT_LR = 5e-4
MID_SHAPE_LR = 1e-3
MID_MATERIAL_LR = 2e-3

LATE_PLACEMENT_LR = 1e-4
LATE_SHAPE_LR = 5e-4
LATE_MATERIAL_LR = 2e-3

LOCAL_SH_LR = 1e-4
GLOBAL_SH_LR = 1e-3


# ============================================================
# FNO CHECKPOINT LOADING
# ============================================================

def load_fno_checkpoint(checkpoint_path, device, expected_mode):
    """
    Load one frozen current-format implicit FNO checkpoint.

    Supports checkpoints from train_implicit.py and removes legacy
    '_metadata' if present in model_state.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"FNO checkpoint not found:\n  {checkpoint_path}"
        )

    print("Loading FNO checkpoint:", checkpoint_path)

    checkpoint_data = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint_data.get("config", {})

    checkpoint_mode = config.get(
        "mode",
        checkpoint_data.get("mode"),
    )

    if checkpoint_mode is not None and checkpoint_mode != expected_mode:
        raise RuntimeError(
            f"Checkpoint mode mismatch:\n"
            f"  expected: {expected_mode}\n"
            f"  checkpoint: {checkpoint_mode}"
        )

    param_mean = np.asarray(
        checkpoint_data["param_mean"],
        dtype=np.float32,
    )

    param_std = np.asarray(
        checkpoint_data["param_std"],
        dtype=np.float32,
    )

    latent_dim = int(len(param_mean))

    if latent_dim != FNO_FEATURE_DIM:
        raise RuntimeError(
            f"Expected {FNO_FEATURE_DIM} checkpoint conditioning features, "
            f"got {latent_dim}."
        )

    image_height = int(config.get("height", 32))
    image_width = int(config.get("width", 32))
    fno_modes = int(config.get("fno_modes", 16))

    model = ImplicitFNOImageModel(
        latent_dim=latent_dim,
        image_height=image_height,
        image_width=image_width,
        fno_modes=fno_modes,
    ).to(device)

    state = dict(checkpoint_data["model_state"])
    state.pop("_metadata", None)

    model.load_state_dict(state, strict=True)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    print(
        f"  mode={expected_mode}, "
        f"latent_dim={latent_dim}, "
        f"patch={image_width}x{image_height}, "
        f"fno_modes={fno_modes}"
    )

    return {
        "model": model,
        "param_mean": param_mean,
        "param_std": param_std,
        "latent_dim": latent_dim,
        "height": image_height,
        "width": image_width,
        "checkpoint": checkpoint_data,
    }


# ============================================================
# SH LIGHTING
# ============================================================

def sh_lm_list(order):
    return [
        (l, m)
        for l in range(order + 1)
        for m in range(-l, l + 1)
    ]


def sh_for_global_env(env_id, order=2):
    """
    Must match the procedural SH environment setup used during rendering.
    """
    pairs = sh_lm_list(order)
    coeffs = np.zeros((len(pairs), 3), dtype=np.float32)

    u = env_id / max(1.0, float(NUM_GLOBAL_ENVS - 1))
    t = 2.0 * math.pi * u

    r = 0.5 + 0.4 * math.sin(t)
    g = 0.5 + 0.4 * math.sin(t + 2.0 * math.pi / 3.0)
    b = 0.5 + 0.4 * math.sin(t + 4.0 * math.pi / 3.0)

    rgb = np.array([r, g, b], dtype=np.float32)
    gray = np.full(3, rgb.mean(), dtype=np.float32)

    if u < 1.0 / 3.0:
        saturation = 0.1
    elif u < 2.0 / 3.0:
        saturation = 0.5
    else:
        saturation = 1.0

    rgb_scale = (1.0 - saturation) * gray + saturation * rgb

    coeffs[0, :] = rgb_scale * 0.4

    for index, (l, m) in enumerate(pairs):
        if l != 1:
            continue

        if m == -1:
            coeffs[index, :] = rgb_scale * (
                0.2 * math.sin(2.0 * math.pi * u)
            )
        elif m == 0:
            coeffs[index, :] = rgb_scale * (
                0.2 * math.cos(2.0 * math.pi * u)
            )
        elif m == 1:
            coeffs[index, :] = rgb_scale * (
                0.2 * math.sin(2.0 * math.pi * u + 1.0)
            )

    for index, (l, m) in enumerate(pairs):
        if l == 2 and m == 0:
            coeffs[index, :] += rgb_scale * (
                0.05 * math.cos(4.0 * math.pi * u)
            )

    return coeffs


def build_sh_bank(device):
    sh_bank = np.stack(
        [
            sh_for_global_env(env_id, order=SH_ORDER).reshape(-1)
            for env_id in range(NUM_GLOBAL_ENVS)
        ],
        axis=0,
    )

    sh_bank = torch.tensor(
        sh_bank,
        dtype=torch.float32,
        device=device,
    )

    print("Built SH bank:", tuple(sh_bank.shape))
    return sh_bank


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
        ).reshape(9, 3)

        self.register_buffer(
            "initial_global_sh",
            initial_global_sh.clone(),
        )

        self.bound = float(bound)

        self.raw_global_sh_delta = nn.Parameter(
            torch.zeros_like(initial_global_sh),
            requires_grad=bool(optimize_sh),
        )

    @property
    def global_sh(self):
        return self.initial_global_sh + (
            self.bound * torch.tanh(self.raw_global_sh_delta)
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
# CAMERA / PROJECTION
# ============================================================

def camera_to_slice_pose(camera, slice_center):
    """
    Return the camera direction encoded exactly as in training.

    phi/theta describe the camera position relative to this slice center.
    """
    camera_center = camera.camera_center

    slice_center = slice_center.to(
        device=camera_center.device,
        dtype=camera_center.dtype,
    )

    relative = camera_center - slice_center

    radius = torch.linalg.norm(relative).clamp_min(1e-8)

    phi = torch.acos(
        torch.clamp(
            relative[2] / radius,
            -1.0,
            1.0,
        )
    )

    theta = torch.remainder(
        torch.atan2(relative[1], relative[0]),
        2.0 * math.pi,
    )

    return phi, theta, radius, relative


def project_world_point(
    camera,
    point_world,
    flip_projection_y=False,
):
    """
    Project one 3D point using the row-vector convention used by 3DGS.

    Returns:
        pixel_x, pixel_y, ndc_z, valid_w
    """
    matrix = camera.full_proj_transform

    point_world = point_world.to(
        device=matrix.device,
        dtype=matrix.dtype,
    )

    point_h = torch.cat(
        [
            point_world,
            torch.ones(
                1,
                dtype=matrix.dtype,
                device=matrix.device,
            ),
        ]
    )

    clip = point_h @ matrix
    clip_w = clip[3]

    valid_w = clip_w > 1e-8

    safe_w = torch.where(
        valid_w,
        clip_w,
        torch.ones_like(clip_w),
    )

    ndc = clip[:3] / safe_w

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

    return pixel_x, pixel_y, ndc[2], valid_w


@torch.no_grad()
def select_visible_slice_indices_batched(
    camera,
    slices,
    fno_radius,
    min_patch_size_px=2.0,
    margin_px=64.0,
    flip_projection_y=False,
    max_active_slices=0,
):
    """
    Batched detached visibility culling.

    Culling itself is intentionally non-differentiable. Rendering of active
    slices remains differentiable.
    """
    if not slices:
        return torch.empty(
            0,
            dtype=torch.long,
            device=camera.camera_center.device,
        )

    device = camera.camera_center.device
    dtype = camera.camera_center.dtype
    n = len(slices)

    centers = torch.stack(
        [
            neural_slice.center.detach().to(
                device=device,
                dtype=dtype,
            )
            for neural_slice in slices
        ],
        dim=0,
    )

    sizes = torch.stack(
        [
            neural_slice.world_size.detach().to(
                device=device,
                dtype=dtype,
            )
            for neural_slice in slices
        ],
        dim=0,
    )

    relative = camera.camera_center[None, :] - centers

    radius = torch.linalg.norm(
        relative,
        dim=1,
    ).clamp_min(1e-8)

    points_h = torch.cat(
        [
            centers,
            torch.ones(n, 1, dtype=dtype, device=device),
        ],
        dim=1,
    )

    clip = points_h @ camera.full_proj_transform
    clip_w = clip[:, 3]

    valid_w = clip_w > 1e-8

    ndc = torch.zeros_like(clip[:, :3])
    ndc[valid_w] = (
        clip[valid_w, :3]
        / clip_w[valid_w, None]
    )

    image_width = float(camera.image_width)
    image_height = float(camera.image_height)

    pixel_x = (ndc[:, 0] + 1.0) * 0.5 * image_width

    if flip_projection_y:
        pixel_y = (ndc[:, 1] + 1.0) * 0.5 * image_height
    else:
        pixel_y = (1.0 - ndc[:, 1]) * 0.5 * image_height

    patch_size = (
        image_height
        * sizes
        * float(fno_radius)
        / radius
    )

    active_mask = (
        valid_w
        & (ndc[:, 2] > 0.0)
        & torch.isfinite(pixel_x)
        & torch.isfinite(pixel_y)
        & torch.isfinite(patch_size)
        & (patch_size >= float(min_patch_size_px))
        & (pixel_x >= -float(margin_px))
        & (pixel_x <= image_width + float(margin_px))
        & (pixel_y >= -float(margin_px))
        & (pixel_y <= image_height + float(margin_px))
    )

    active_indices = torch.nonzero(
        active_mask,
        as_tuple=True,
    )[0]

    if (
        max_active_slices > 0
        and active_indices.numel() > max_active_slices
    ):
        projected_area = patch_size[active_indices].square()

        _, keep_local = torch.topk(
            projected_area,
            k=max_active_slices,
            largest=True,
            sorted=False,
        )

        active_indices = active_indices[keep_local]

    return active_indices


# ============================================================
# PATCH NETWORK EVALUATION
# ============================================================

def run_fno_in_chunks(
    model,
    params,
    batch_size,
    use_activation_checkpointing=True,
):
    """
    Evaluate a frozen FNO on [N, D] parameter vectors in chunks.

    Gradients remain enabled with respect to params, which flow through local
    slice parameters. FNO weights remain frozen.
    """
    if params.shape[0] == 0:
        raise RuntimeError("Cannot evaluate FNO on zero parameters.")

    outputs = []

    for start in range(0, params.shape[0], batch_size):
        end = min(start + batch_size, params.shape[0])
        parameter_chunk = params[start:end]

        if use_activation_checkpointing and torch.is_grad_enabled():
            output_chunk = checkpoint(
                model,
                parameter_chunk,
                use_reentrant=False,
            )
        else:
            output_chunk = model(parameter_chunk)

        outputs.append(output_chunk)

    return torch.cat(outputs, dim=0)


# ============================================================
# NEURAL SCENE RENDERER
# ============================================================

class NeuralSceneRenderer(nn.Module):
    """
    Render all active local neural slices for one 3DGS camera.
    """

    def __init__(
        self,
        surface_bundle,
        volume_bundle,
        fno_radius=FNO_RADIUS,
        fno_batch_size=16,
        flip_projection_y=False,
        flip_fno_vertical=False,
        use_tile_renderer=False,
        tile_size=128,
        tile_roi_margin=16,
        max_patches_per_tile=None,
        use_visibility_culling=False,
        use_fno_activation_checkpointing=True,
        max_active_slices_per_camera=0,
        min_projected_patch_size=2.0,
        active_slice_margin=64.0,
    ):
        super().__init__()

        self.surface_model = surface_bundle["model"]
        self.volume_model = volume_bundle["model"]

        self.surface_mean = surface_bundle["param_mean"]
        self.surface_std = surface_bundle["param_std"]

        self.volume_mean = volume_bundle["param_mean"]
        self.volume_std = volume_bundle["param_std"]

        self.fno_radius = float(fno_radius)
        self.fno_batch_size = int(fno_batch_size)

        self.flip_projection_y = bool(flip_projection_y)
        self.flip_fno_vertical = bool(flip_fno_vertical)

        self.use_tile_renderer = bool(use_tile_renderer)
        self.use_visibility_culling = bool(use_visibility_culling)

        self.use_fno_activation_checkpointing = bool(
            use_fno_activation_checkpointing
        )

        self.max_active_slices_per_camera = int(
            max_active_slices_per_camera
        )

        self.min_projected_patch_size = float(
            min_projected_patch_size
        )

        self.active_slice_margin = float(active_slice_margin)

        self.tile_renderer = TilePatchRenderer(
            tile_size=int(tile_size),
            roi_margin=int(tile_roi_margin),
            max_patches_per_tile=max_patches_per_tile,
        )

        for model in [self.surface_model, self.volume_model]:
            model.eval()

            for parameter in model.parameters():
                parameter.requires_grad_(False)

    def render_slice_list_batched(
        self,
        camera,
        slices,
        shared_sh,
        collect_diagnostics=False,
    ):
        if self.use_visibility_culling:
            active_indices = select_visible_slice_indices_batched(
                camera=camera,
                slices=slices,
                fno_radius=self.fno_radius,
                min_patch_size_px=self.min_projected_patch_size,
                margin_px=self.active_slice_margin,
                flip_projection_y=self.flip_projection_y,
                max_active_slices=self.max_active_slices_per_camera,
            )

            active_indices_cpu = active_indices.detach().cpu().tolist()

            active_slices = [
                slices[index]
                for index in active_indices_cpu
            ]
        else:
            active_indices_cpu = list(range(len(slices)))
            active_slices = list(slices)

        canvas_height = int(camera.image_height)
        canvas_width = int(camera.image_width)
        device = camera.camera_center.device

        if not active_slices:
            empty = torch.zeros(
                1,
                4,
                canvas_height,
                canvas_width,
                dtype=torch.float32,
                device=device,
            )

            return empty, []

        records = []

        for local_index, neural_slice in enumerate(active_slices):
            slice_index = active_indices_cpu[local_index]

            phi, theta, radius, relative = camera_to_slice_pose(
                camera,
                neural_slice.center,
            )

            if neural_slice.mode == "surface":
                param_mean = self.surface_mean
                param_std = self.surface_std
            elif neural_slice.mode == "volume":
                param_mean = self.volume_mean
                param_std = self.volume_std
            else:
                raise RuntimeError(
                    f"Unexpected slice mode: {neural_slice.mode}"
                )

            param_vector = build_tensor_fno_vector(
                neural_slice=neural_slice,
                param_mean=param_mean,
                param_std=param_std,
                phi=phi,
                theta=theta,
                device=device,
                shared_sh=shared_sh,
            )

            center_x, center_y, center_depth, valid_w = project_world_point(
                camera,
                neural_slice.center,
                flip_projection_y=self.flip_projection_y,
            )

            projected_patch_size = (
                float(canvas_height)
                * neural_slice.world_size
                * self.fno_radius
                / radius
            )

            records.append(
                {
                    "slice_index": slice_index,
                    "mode": neural_slice.mode,
                    "param_vector": param_vector,
                    "center_x": center_x,
                    "center_y": center_y,
                    "patch_size": projected_patch_size,
                    "depth": radius,
                    "center_depth": center_depth,
                    "valid_w": valid_w,
                    "phi": phi,
                    "theta": theta,
                    "radius": radius,
                    "relative": relative,
                }
            )

        surface_record_indices = [
            index
            for index, record in enumerate(records)
            if record["mode"] == "surface"
        ]

        volume_record_indices = [
            index
            for index, record in enumerate(records)
            if record["mode"] == "volume"
        ]

        patches_by_record = [None] * len(records)

        if surface_record_indices:
            surface_params = torch.cat(
                [
                    records[index]["param_vector"]
                    for index in surface_record_indices
                ],
                dim=0,
            )

            surface_patches = run_fno_in_chunks(
                model=self.surface_model,
                params=surface_params,
                batch_size=self.fno_batch_size,
                use_activation_checkpointing=(
                    self.use_fno_activation_checkpointing
                ),
            )

            if self.flip_fno_vertical:
                surface_patches = torch.flip(
                    surface_patches,
                    dims=[2],
                )

            for local_index, record_index in enumerate(
                surface_record_indices
            ):
                patches_by_record[record_index] = (
                    surface_patches[local_index:local_index + 1]
                )

        if volume_record_indices:
            volume_params = torch.cat(
                [
                    records[index]["param_vector"]
                    for index in volume_record_indices
                ],
                dim=0,
            )

            volume_patches = run_fno_in_chunks(
                model=self.volume_model,
                params=volume_params,
                batch_size=self.fno_batch_size,
                use_activation_checkpointing=(
                    self.use_fno_activation_checkpointing
                ),
            )

            if self.flip_fno_vertical:
                volume_patches = torch.flip(
                    volume_patches,
                    dims=[2],
                )

            for local_index, record_index in enumerate(
                volume_record_indices
            ):
                patches_by_record[record_index] = (
                    volume_patches[local_index:local_index + 1]
                )

        if any(patch is None for patch in patches_by_record):
            raise RuntimeError(
                "At least one active slice failed to receive an FNO patch."
            )

        all_patches = torch.cat(patches_by_record, dim=0)

        center_x = torch.stack(
            [record["center_x"] for record in records],
            dim=0,
        )

        center_y = torch.stack(
            [record["center_y"] for record in records],
            dim=0,
        )

        patch_sizes = torch.stack(
            [record["patch_size"] for record in records],
            dim=0,
        )

        depths = torch.stack(
            [record["depth"] for record in records],
            dim=0,
        )

        # Larger camera distance means farther away.
        # Sorting is detached/discrete, as in Gaussian splatting.
        sort_indices = torch.argsort(
            depths.detach(),
            descending=True,
        )

        sorted_patches = all_patches[sort_indices]
        sorted_center_x = center_x[sort_indices]
        sorted_center_y = center_y[sort_indices]
        sorted_patch_sizes = patch_sizes[sort_indices]

        if self.use_tile_renderer:
            composite = self.tile_renderer(
                patches=all_patches,
                center_x=center_x,
                center_y=center_y,
                patch_size=patch_sizes,
                depths=depths,
                image_height=canvas_height,
                image_width=canvas_width,
            )
        else:
            composite = render_rois_to_tiled_canvas(
                sorted_patches=sorted_patches,
                sorted_center_x=sorted_center_x,
                sorted_center_y=sorted_center_y,
                sorted_patch_sizes=sorted_patch_sizes,
                canvas_height=canvas_height,
                canvas_width=canvas_width,
                tile_size=128,
                margin_pixels=16,
                max_roi_side=max(canvas_height, canvas_width),
            )

        diagnostics = []

        if collect_diagnostics:
            sorted_record_indices = sort_indices.detach().cpu().tolist()

            for record_index in sorted_record_indices:
                diagnostics.append(records[record_index])

        return composite.clamp(0.0, 1.0), diagnostics

    def forward(self, camera, neural_scene):
        return self.render_slice_list_batched(
            camera=camera,
            slices=neural_scene.slices,
            shared_sh=neural_scene.global_sh,
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
    """
    Current inverse-render objective.

    Target images from 3DGS Scene are ordinary RGB, so predicted
    premultiplied RGBA is composited onto the configured white background.
    """
    predicted_rgb = visible_rgb_from_rgba(predicted_rgba)

    return (
        0.5 * F.l1_loss(predicted_rgb, target_rgb)
        + 0.5 * F.mse_loss(predicted_rgb, target_rgb)
    )


def parameter_regularization(neural_scene):
    if not neural_scene.slices:
        return torch.zeros((), device=neural_scene.global_sh.device)

    total = torch.zeros(
        (),
        dtype=torch.float32,
        device=neural_scene.slices[0].center.device,
    )

    for neural_slice in neural_scene.slices:
        values = neural_slice.fno_values(
            shared_sh=neural_scene.global_sh
        )

        position_loss = F.mse_loss(
            neural_slice.center,
            neural_slice.initial_center,
        )

        relative_size_change = (
            neural_slice.world_size
            - neural_slice.initial_world_size
        ) / neural_slice.initial_world_size.clamp_min(1e-4)

        size_loss = relative_size_change.square()

        appearance_loss = torch.mean(
            (values["ctrl"] - neural_slice.initial_ctrl).square()
        )

        appearance_loss = appearance_loss + (
            values["sigma"] - neural_slice.initial_sigma
        ).square()

        appearance_loss = appearance_loss + (
            values["base_color_r"]
            - neural_slice.initial_base_color_r
        ).square()

        appearance_loss = appearance_loss + (
            values["base_color_g"]
            - neural_slice.initial_base_color_g
        ).square()

        appearance_loss = appearance_loss + (
            values["base_color_b"]
            - neural_slice.initial_base_color_b
        ).square()

        appearance_loss = appearance_loss + (
            values["opacity"]
            - neural_slice.initial_opacity
        ).square()

        if neural_slice.mode == "surface":
            appearance_loss = appearance_loss + (
                values["roughness"]
                - neural_slice.initial_roughness
            ).square()

        local_sh_loss = torch.mean(
            neural_slice.local_sh_delta.square()
        )

        total = total + (
            POSITION_REG_WEIGHT * position_loss
            + SIZE_REG_WEIGHT * size_loss
            + PARAMETER_REG_WEIGHT * appearance_loss
            + LOCAL_SH_REG_WEIGHT * local_sh_loss
        )

    return total


def global_sh_regularization(neural_scene):
    return GLOBAL_SH_REG_WEIGHT * torch.mean(
        (
            neural_scene.global_sh
            - neural_scene.lighting.initial_global_sh
        ).square()
    )


def build_knn_neighbors(neural_scene, k=3):
    if len(neural_scene.slices) <= 1:
        return []

    centers = torch.stack(
        [
            neural_slice.center.detach()
            for neural_slice in neural_scene.slices
        ],
        dim=0,
    )

    distances = torch.cdist(centers, centers)
    distances.fill_diagonal_(float("inf"))

    pairs = set()

    for index in range(len(neural_scene.slices)):
        neighbors = torch.topk(
            distances[index],
            k=min(k, len(neural_scene.slices) - 1),
            largest=False,
        ).indices.tolist()

        for neighbor in neighbors:
            pairs.add(tuple(sorted((index, neighbor))))

    return sorted(pairs)


def neighbor_sh_regularization(neural_scene, neighbor_pairs):
    if not neighbor_pairs:
        return torch.zeros(
            (),
            dtype=torch.float32,
            device=neural_scene.global_sh.device,
        )

    effective_sh = [
        neural_slice.fno_values(
            shared_sh=neural_scene.global_sh
        )["sh"]
        for neural_slice in neural_scene.slices
    ]

    total = torch.zeros(
        (),
        dtype=torch.float32,
        device=effective_sh[0].device,
    )

    for index_a, index_b in neighbor_pairs:
        total = total + torch.mean(
            (effective_sh[index_a] - effective_sh[index_b]).square()
        )

    return NEIGHBOR_SH_REG_WEIGHT * total


# ============================================================
# OPTIMIZER / SCHEDULE
# ============================================================

def get_optimization_stage(iteration, optimize_sh):
    if iteration < SHAPE_WARMUP_ITERS:
        return "early"

    if iteration < LIGHTING_START_ITERS or not optimize_sh:
        return "mid"

    return "lighting"


def set_lighting_requires_grad(neural_scene, enabled):
    neural_scene.lighting.raw_global_sh_delta.requires_grad_(enabled)

    for neural_slice in neural_scene.slices:
        neural_slice.optimize_environment = bool(enabled)
        neural_slice.raw_local_sh_delta.requires_grad_(enabled)


def make_optimizer(neural_scene, stage, optimize_sh):
    placement_params = []
    shape_params = []
    material_params = []
    local_sh_params = []

    for neural_slice in neural_scene.slices:
        placement_params.extend(
            [
                neural_slice.center,
                neural_slice.raw_world_size,
            ]
        )

        shape_params.extend(
            [
                neural_slice.raw_ctrl,
                neural_slice.raw_sigma,
            ]
        )

        material_params.extend(
            [
                neural_slice.raw_base_color_r,
                neural_slice.raw_base_color_g,
                neural_slice.raw_base_color_b,
                neural_slice.raw_opacity,
                neural_slice.raw_roughness,
            ]
        )

        if (
            optimize_sh
            and neural_slice.raw_local_sh_delta.requires_grad
        ):
            local_sh_params.append(
                neural_slice.raw_local_sh_delta
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
        raise ValueError(f"Unknown stage: {stage}")

    groups = [
        {"params": placement_params, "lr": placement_lr},
        {"params": shape_params, "lr": shape_lr},
        {"params": material_params, "lr": material_lr},
    ]

    if optimize_sh and local_sh_params and local_sh_lr > 0.0:
        groups.append(
            {"params": local_sh_params, "lr": local_sh_lr}
        )

    if (
        optimize_sh
        and neural_scene.lighting.raw_global_sh_delta.requires_grad
        and global_sh_lr > 0.0
    ):
        groups.append(
            {
                "params": [
                    neural_scene.lighting.raw_global_sh_delta
                ],
                "lr": global_sh_lr,
            }
        )

    return torch.optim.Adam(groups)


# ============================================================
# POINT-CLOUD COLOR EXTRACTION
# ============================================================

@torch.no_grad()
def try_get_point_colors(gaussian_model):
    """
    Best-effort RGB extraction from common 3DGS GaussianModel APIs.

    Returns:
        colors [N,3] in [0,1], or None if unavailable.

    Standard 3DGS often stores DC SH coefficients in get_features[:,0,:].
    In that case SH2RGB converts them to RGB.
    """
    if hasattr(gaussian_model, "get_colors"):
        colors = gaussian_model.get_colors.detach()
        return colors.clamp(0.0, 1.0)

    if hasattr(gaussian_model, "get_rgb"):
        colors = gaussian_model.get_rgb.detach()
        return colors.clamp(0.0, 1.0)

    if hasattr(gaussian_model, "get_features"):
        features = gaussian_model.get_features.detach()

        if features.ndim == 3 and features.shape[1] >= 1:
            dc = features[:, 0, :]

            try:
                from utils.sh_utils import SH2RGB
                return SH2RGB(dc).clamp(0.0, 1.0)
            except Exception as exc:
                print(
                    "[WARN] Could not import/use SH2RGB for point colors:",
                    exc,
                )

    print(
        "[WARN] Could not extract point RGB from GaussianModel. "
        "Voxel seeds will use random base colors."
    )

    return None


# ============================================================
# RANDOM INITIALIZATION
# ============================================================

def random_slice_init(rng, shared_sh, seed_color=None):
    if seed_color is not None:
        seed_color = torch.as_tensor(
            seed_color,
            dtype=torch.float32,
        ).reshape(3)

        base_color_r = float(seed_color[0])
        base_color_g = float(seed_color[1])
        base_color_b = float(seed_color[2])
    else:
        base_color_r = float(rng.uniform(0.02, 1.0))
        base_color_g = float(rng.uniform(0.02, 1.0))
        base_color_b = float(rng.uniform(0.02, 1.0))

    return {
        "ctrl_values": rng.choice(
            CTRL_LEVELS,
            size=8,
            replace=True,
        ).astype(np.float32),
        "sigma": float(rng.choice(SIGMA_VALUES)),
        "base_color_r": base_color_r,
        "base_color_g": base_color_g,
        "base_color_b": base_color_b,
        "opacity": 0.90,
        "roughness": float(rng.uniform(0.10, 0.90)),
        "metallic": float(rng.choice([0.0, 1.0])),
        "specular": 0.5,
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
        base_color_r=init_params["base_color_r"],
        base_color_g=init_params["base_color_g"],
        base_color_b=init_params["base_color_b"],
        opacity=init_params["opacity"],
        roughness=roughness,
        sh_values=init_params["sh_values"],
        metallic=metallic,
        specular=specular,
        optimize_environment=optimize_sh,
    ).to(device)


# ============================================================
# CHECKPOINT RECONSTRUCTION
# ============================================================

def serialize_slice(slice_obj):
    return {
        "mode": slice_obj.mode,
        "optimize_environment": bool(
            slice_obj.optimize_environment
        ),
        "local_sh_bound": float(slice_obj.local_sh_bound),
    }


def save_neural_scene_checkpoint(
    path,
    neural_scene,
    optimizer,
    iteration,
    metadata=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "iteration": int(iteration),
            "scene_state": neural_scene.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "slice_specs": [
                serialize_slice(neural_slice)
                for neural_slice in neural_scene.slices
            ],
            "initial_global_sh": (
                neural_scene.lighting.initial_global_sh.detach().cpu()
            ),
            "metadata": metadata or {},
        },
        path,
    )


def rebuild_neural_scene_from_checkpoint(
    checkpoint_data,
    device,
    optimize_sh,
):
    """
    Recreate topology, then load exact raw parameters/buffers.
    """
    scene_state = checkpoint_data["scene_state"]
    slices = []

    for index, spec in enumerate(checkpoint_data["slice_specs"]):
        prefix = f"slices.{index}."

        slice_obj = LearnableNeuralSlice(
            mode=spec["mode"],
            center=scene_state[prefix + "initial_center"],
            world_size=float(
                scene_state[prefix + "initial_world_size"]
            ),
            ctrl_values=scene_state[prefix + "initial_ctrl"],
            sigma=float(scene_state[prefix + "initial_sigma"]),
            base_color_r=float(
                scene_state[prefix + "initial_base_color_r"]
            ),
            base_color_g=float(
                scene_state[prefix + "initial_base_color_g"]
            ),
            base_color_b=float(
                scene_state[prefix + "initial_base_color_b"]
            ),
            opacity=float(
                scene_state[prefix + "initial_opacity"]
            ),
            roughness=float(
                scene_state[prefix + "initial_roughness"]
            ),
            sh_values=scene_state[prefix + "initial_sh"],
            metallic=float(scene_state[prefix + "metallic_value"]),
            specular=float(scene_state[prefix + "specular_value"]),
            optimize_environment=(
                bool(spec["optimize_environment"])
                and bool(optimize_sh)
            ),
            local_sh_bound=float(spec["local_sh_bound"]),
        ).to(device)

        slices.append(slice_obj)

    neural_scene = NeuralScene(
        slices=slices,
        initial_global_sh=checkpoint_data["initial_global_sh"].to(device),
        optimize_sh=optimize_sh,
    ).to(device)

    neural_scene.load_state_dict(checkpoint_data["scene_state"])

    return neural_scene


# ============================================================
# PREVIEW HELPERS
# ============================================================

def save_rgb_chw(rgb_chw, path):
    if torch.is_tensor(rgb_chw):
        rgb_chw = rgb_chw.detach().cpu().numpy()

    rgb_hwc = np.transpose(rgb_chw, (1, 2, 0))

    imageio.imwrite(
        path,
        (
            np.clip(rgb_hwc, 0.0, 1.0) * 255.0 + 0.5
        ).astype(np.uint8),
    )


def save_alpha_hw(alpha_hw, path):
    if torch.is_tensor(alpha_hw):
        alpha_hw = alpha_hw.detach().cpu().numpy()

    alpha_rgb = np.repeat(alpha_hw[..., None], 3, axis=2)

    imageio.imwrite(
        path,
        (
            np.clip(alpha_rgb, 0.0, 1.0) * 255.0 + 0.5
        ).astype(np.uint8),
    )


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

    preview_dir.mkdir(parents=True, exist_ok=True)

    for view_index, camera in enumerate(cameras[:max_views]):
        prediction_rgba, _ = renderer(camera, neural_scene)

        prediction_rgba = prediction_rgba.clamp(0.0, 1.0)

        predicted_rgb = visible_rgb_from_rgba(prediction_rgba)[0]
        target_rgb = camera.original_image.detach()

        save_rgb_chw(
            predicted_rgb,
            preview_dir / f"view_{view_index:02d}_prediction.png",
        )

        save_alpha_hw(
            prediction_rgba[0, 3],
            preview_dir / f"view_{view_index:02d}_alpha.png",
        )

        save_rgb_chw(
            target_rgb,
            preview_dir / f"view_{view_index:02d}_target.png",
        )

        target_hwc = np.transpose(
            target_rgb.detach().cpu().numpy(),
            (1, 2, 0),
        )

        prediction_hwc = np.transpose(
            predicted_rgb.detach().cpu().numpy(),
            (1, 2, 0),
        )

        comparison = np.concatenate(
            [
                np.clip(target_hwc, 0.0, 1.0),
                np.clip(prediction_hwc, 0.0, 1.0),
            ],
            axis=1,
        )

        imageio.imwrite(
            preview_dir / f"view_{view_index:02d}_comparison.png",
            (comparison * 255.0 + 0.5).astype(np.uint8),
        )

    print("Saved previews:", preview_dir)
    

# ============================================================
# CONTRIBUTION-BASED PRUNING
# ============================================================

@torch.no_grad()
def mean_scene_loss(
    renderer,
    neural_scene,
    cameras,
):
    """
    Mean reconstruction loss across a small camera subset.
    """
    if not cameras:
        return float("inf")

    total_loss = 0.0
    valid_count = 0

    for camera in cameras:
        prediction_rgba, _ = renderer(
            camera,
            neural_scene,
        )

        target_rgb = camera.original_image.detach()

        if target_rgb.ndim == 3:
            target_rgb = target_rgb.unsqueeze(0)

        target_rgb = target_rgb.to(
            device=prediction_rgba.device,
            dtype=prediction_rgba.dtype,
        )

        loss = image_loss(
            prediction_rgba,
            target_rgb,
        )

        if torch.isfinite(loss):
            total_loss += float(loss.item())
            valid_count += 1

    if valid_count == 0:
        return float("inf")

    return total_loss / valid_count


@torch.no_grad()
def slice_removal_contribution(
    renderer,
    neural_scene,
    remove_index,
    cameras,
    full_scene_loss,
):
    """
    Measure how much one slice contributes.

    Returns:
        contribution = loss_without_slice - full_scene_loss

    Interpretation:
        contribution < 0:
            Removing the slice improves the loss; prune it.

        contribution ≈ 0:
            Slice is redundant/negligible.

        contribution > 0:
            Slice helps reconstruction; keep it.
    """
    remaining_slices = [
        neural_slice
        for index, neural_slice in enumerate(neural_scene.slices)
        if index != remove_index
    ]

    # Never permit contribution testing to remove the only slice.
    if not remaining_slices:
        return float("inf"), float("inf")

    total_loss = 0.0
    valid_count = 0

    for camera in cameras:
        prediction_rgba, _ = renderer.render_slice_list_batched(
            camera=camera,
            slices=remaining_slices,
            shared_sh=neural_scene.global_sh,
            collect_diagnostics=False,
        )

        target_rgb = camera.original_image.detach()

        if target_rgb.ndim == 3:
            target_rgb = target_rgb.unsqueeze(0)

        target_rgb = target_rgb.to(
            device=prediction_rgba.device,
            dtype=prediction_rgba.dtype,
        )

        loss = image_loss(
            prediction_rgba,
            target_rgb,
        )

        if torch.isfinite(loss):
            total_loss += float(loss.item())
            valid_count += 1

    if valid_count == 0:
        return float("inf"), float("inf")

    loss_without_slice = total_loss / valid_count
    contribution = loss_without_slice - full_scene_loss

    return contribution, loss_without_slice


@torch.no_grad()
def choose_contribution_prune_indices(
    renderer,
    neural_scene,
    cameras,
    candidate_indices,
    max_prunes=2,
    contribution_threshold=0.0,
):
    """
    Return low-value slices selected by explicit removal tests.

    A threshold of 0.0 means prune only if removal improves the loss.

    A small positive threshold such as 1e-4 permits pruning slices whose
    removal causes only a negligible reconstruction-loss increase.
    """
    if len(neural_scene.slices) <= 1:
        return []

    if not candidate_indices:
        return []

    full_scene_loss = mean_scene_loss(
        renderer=renderer,
        neural_scene=neural_scene,
        cameras=cameras,
    )

    print(
        "[CONTRIBUTION PRUNE] "
        f"full_scene_loss={full_scene_loss:.8f}"
    )

    prune_indices = []

    for slice_index in candidate_indices:
        slice_index = int(slice_index)

        if slice_index < 0 or slice_index >= len(neural_scene.slices):
            continue

        contribution, loss_without = slice_removal_contribution(
            renderer=renderer,
            neural_scene=neural_scene,
            remove_index=slice_index,
            cameras=cameras,
            full_scene_loss=full_scene_loss,
        )

        print(
            f"  slice={slice_index:03d} "
            f"loss_without={loss_without:.8f} "
            f"contribution={contribution:+.8e}"
        )

        if contribution <= contribution_threshold:
            prune_indices.append(slice_index)

        if len(prune_indices) >= max_prunes:
            break

    # Defensive guard: never prune every slice.
    return prune_indices[:max(0, len(neural_scene.slices) - 1)]


# ============================================================
# MAIN
# ============================================================

def main():
    parser = ArgumentParser(
        description=(
            "Optimize a neural scene using frozen implicit B-spline "
            "surface and volume FNO renderers."
        )
    )

    lp = ModelParams(parser)

    parser.add_argument("--camera_start", type=int, default=0)
    parser.add_argument("--num_cameras", type=int, default=8)

    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--voxel_size", type=float, default=0.5)
    parser.add_argument("--min_points_per_voxel", type=int, default=50)
    parser.add_argument("--max_initial_slices", type=int, default=64)

    parser.add_argument(
        "--initial_mode",
        choices=["surface", "volume", "random"],
        default="random",
        help="Initial mode for each voxel seed.",
    )

    parser.add_argument("--optimize_sh", action="store_true")

    parser.add_argument("--fno_batch_size", type=int, default=16)

    parser.add_argument("--use_tile_renderer", action="store_true")
    parser.add_argument("--tile_size", type=int, default=128)
    parser.add_argument("--tile_roi_margin", type=int, default=16)
    parser.add_argument("--max_patches_per_tile", type=int, default=0)

    parser.add_argument("--use_visibility_culling", action="store_true")
    parser.add_argument("--min_projected_patch_size", type=float, default=2.0)
    parser.add_argument("--active_slice_margin", type=float, default=64.0)
    parser.add_argument("--max_active_slices_per_camera", type=int, default=0)

    parser.add_argument(
        "--disable_fno_activation_checkpointing",
        action="store_true",
    )

    parser.add_argument("--cameras_per_step", type=int, default=8)

    parser.add_argument("--split_interval", type=int, default=0)
    parser.add_argument("--max_splits_per_update", type=int, default=1)
    parser.add_argument("--max_slices", type=int, default=128)
    parser.add_argument("--lifecycle_cameras", type=int, default=1)
    parser.add_argument("--lifecycle_split_candidates", type=int, default=16)

    parser.add_argument("--prune_interval", type=int, default=0)
    parser.add_argument("--lifecycle_prune_candidates", type=int, default=16)
    parser.add_argument("--max_prunes_per_update", type=int, default=2)

    parser.add_argument(
        "--surface_checkpoint",
        type=Path,
        default=DEFAULT_SURFACE_CHECKPOINT,
    )

    parser.add_argument(
        "--volume_checkpoint",
        type=Path,
        default=DEFAULT_VOLUME_CHECKPOINT,
    )

    parser.add_argument(
        "--resume_checkpoint",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=250,
    )

    parser.add_argument("--preview_views", type=int, default=2)

    parser.add_argument(
        "--flip_projection_y",
        action="store_true",
    )

    parser.add_argument(
        "--flip_fno_vertical",
        action="store_true",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    
    parser.add_argument(
        "--prune_selection_cameras",
        type=int,
        default=3,
        help=(
            "Number of cameras used for expensive contribution-based "
            "slice-removal tests."
        ),
    )
    
    parser.add_argument(
        "--contribution_threshold",
        type=float,
        default=0.0,
        help=(
            "Prune when loss_without_slice - full_loss is <= this value. "
            "Use 0.0 to prune only slices whose removal improves loss. "
            "Use a small positive value such as 1e-4 to also prune nearly "
            "redundant slices."
        ),
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for neural scene optimization.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass

    device = torch.device("cuda")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rng = np.random.default_rng(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Device:", device)
    print("Seed:", args.seed)
    print("Output directory:", output_dir)

    # --------------------------------------------------------
    # Load frozen FNO bundles.
    # --------------------------------------------------------

    surface_bundle = load_fno_checkpoint(
        args.surface_checkpoint,
        device=device,
        expected_mode="surface",
    )

    volume_bundle = load_fno_checkpoint(
        args.volume_checkpoint,
        device=device,
        expected_mode="volume",
    )

    if surface_bundle["latent_dim"] != volume_bundle["latent_dim"]:
        raise RuntimeError(
            "Surface/volume FNO conditioning dimensions differ."
        )

    # --------------------------------------------------------
    # Build neural renderer.
    # --------------------------------------------------------

    max_patches_per_tile = (
        None
        if args.max_patches_per_tile <= 0
        else int(args.max_patches_per_tile)
    )

    renderer = NeuralSceneRenderer(
        surface_bundle=surface_bundle,
        volume_bundle=volume_bundle,
        fno_radius=FNO_RADIUS,
        fno_batch_size=args.fno_batch_size,
        flip_projection_y=args.flip_projection_y,
        flip_fno_vertical=args.flip_fno_vertical,
        use_tile_renderer=args.use_tile_renderer,
        tile_size=args.tile_size,
        tile_roi_margin=args.tile_roi_margin,
        max_patches_per_tile=max_patches_per_tile,
        use_visibility_culling=args.use_visibility_culling,
        use_fno_activation_checkpointing=(
            not args.disable_fno_activation_checkpointing
        ),
        min_projected_patch_size=args.min_projected_patch_size,
        active_slice_margin=args.active_slice_margin,
        max_active_slices_per_camera=args.max_active_slices_per_camera,
    ).to(device)

    # --------------------------------------------------------
    # Load 3DGS scene/cameras/point cloud.
    # --------------------------------------------------------

    scene_args = lp.extract(args)

    if not scene_args.model_path:
        scene_args.model_path = str(
            REPO_DIR / "output" / "neural_scene"
        )

    Path(scene_args.model_path).mkdir(
        parents=True,
        exist_ok=True,
    )

    gaussian_model = GaussianModel(args.sh_degree)

    scene = Scene(
        scene_args,
        gaussian_model,
        shuffle=False,
        resolution_scales=[1.0],
    )

    all_cameras = scene.getTrainCameras(scale=1.0)

    camera_end = min(
        args.camera_start + args.num_cameras,
        len(all_cameras),
    )

    cameras = all_cameras[args.camera_start:camera_end]

    if not cameras:
        raise RuntimeError("No cameras selected.")

    print(
        f"Using cameras [{args.camera_start}, {camera_end - 1}], "
        f"count={len(cameras)}"
    )

    xyz = gaussian_model.get_xyz.detach()
    point_colors = try_get_point_colors(gaussian_model)

    # --------------------------------------------------------
    # Build or resume neural scene.
    # --------------------------------------------------------

    checkpoint_data = None
    initial_env_id = -1
    start_iteration = 0

    if args.resume_checkpoint is not None:
        if not args.resume_checkpoint.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {args.resume_checkpoint}"
            )

        checkpoint_data = torch.load(
            args.resume_checkpoint,
            map_location=device,
            weights_only=False,
        )

        neural_scene = rebuild_neural_scene_from_checkpoint(
            checkpoint_data=checkpoint_data,
            device=device,
            optimize_sh=args.optimize_sh,
        )

        start_iteration = int(checkpoint_data["iteration"]) + 1

        initial_env_id = int(
            checkpoint_data.get("metadata", {}).get(
                "initial_env_id",
                -1,
            )
        )

        print(
            f"Resumed scene at iteration {start_iteration}, "
            f"slices={len(neural_scene.slices)}"
        )

    else:
        sh_bank = build_sh_bank(device)

        initial_env_id = int(
            rng.integers(0, NUM_GLOBAL_ENVS)
        )

        initial_global_sh = sh_bank[initial_env_id].detach().clone()

        max_chunks = (
            None
            if args.max_initial_slices <= 0
            else int(args.max_initial_slices)
        )

        seeds = voxel_chunk_seeds(
            xyz=xyz,
            colors=point_colors,
            voxel_size=args.voxel_size,
            min_points=args.min_points_per_voxel,
            max_chunks=max_chunks,
        )

        if not seeds:
            raise RuntimeError(
                "No voxel seeds were produced. Increase --voxel_size "
                "or lower --min_points_per_voxel."
            )

        print(f"Voxel seeds: {len(seeds)}")

        slices = []

        for seed_index, seed in enumerate(seeds):
            init_params = random_slice_init(
                rng=rng,
                shared_sh=initial_global_sh,
                seed_color=seed.get("color"),
            )

            if args.initial_mode == "random":
                mode = random.choice(["surface", "volume"])
            else:
                mode = args.initial_mode

            neural_slice = make_slice_from_seed(
                seed=seed,
                mode=mode,
                init_params=init_params,
                device=device,
                optimize_sh=False,
            )

            slices.append(neural_slice)

            if seed_index < 10 or seed_index % 25 == 0:
                print(
                    f"seed={seed_index:03d} "
                    f"points={seed['num_points']} "
                    f"size={seed['world_size']:.4f} "
                    f"mode={mode} "
                    f"color={seed.get('color')}"
                )

        neural_scene = NeuralScene(
            slices=slices,
            initial_global_sh=initial_global_sh,
            optimize_sh=False,
        ).to(device)

    # --------------------------------------------------------
    # Optimization setup.
    # --------------------------------------------------------

    current_stage = get_optimization_stage(
        start_iteration,
        args.optimize_sh,
    )

    lighting_enabled = (
        args.optimize_sh
        and current_stage == "lighting"
    )

    set_lighting_requires_grad(
        neural_scene,
        enabled=lighting_enabled,
    )

    optimizer = make_optimizer(
        neural_scene=neural_scene,
        stage=current_stage,
        optimize_sh=args.optimize_sh,
    )

    # Optimizer state is only restored when topology and stage match.
    if (
        checkpoint_data is not None
        and checkpoint_data.get("optimizer_state") is not None
    ):
        try:
            optimizer.load_state_dict(
                checkpoint_data["optimizer_state"]
            )
            print("Restored optimizer state.")
        except Exception as exc:
            print(
                "[WARN] Could not restore optimizer state; "
                "using a fresh optimizer.",
                exc,
            )

    gradient_stats = make_gradient_stats(
        len(neural_scene.slices)
    )

    neighbor_pairs = build_knn_neighbors(neural_scene, k=3)

    print(
        f"Starting optimization at iteration={start_iteration}, "
        f"stage={current_stage}, "
        f"slices={len(neural_scene.slices)}"
    )

    # --------------------------------------------------------
    # Sanity render.
    # --------------------------------------------------------

    with torch.no_grad():
        sanity_rgba, _ = renderer(cameras[0], neural_scene)

    if not torch.isfinite(sanity_rgba).all():
        raise RuntimeError("Initial neural render has NaN/Inf values.")

    print("Sanity render shape:", tuple(sanity_rgba.shape))

    # --------------------------------------------------------
    # Optimization loop.
    # --------------------------------------------------------

    for iteration in range(start_iteration, args.iterations):
        new_stage = get_optimization_stage(
            iteration,
            args.optimize_sh,
        )

        if new_stage != current_stage:
            current_stage = new_stage

            should_enable_lighting = (
                args.optimize_sh
                and current_stage == "lighting"
            )

            set_lighting_requires_grad(
                neural_scene,
                enabled=should_enable_lighting,
            )

            optimizer = make_optimizer(
                neural_scene=neural_scene,
                stage=current_stage,
                optimize_sh=args.optimize_sh,
            )

            print(
                f"[SCHEDULE] iteration={iteration}, "
                f"new stage={current_stage}, "
                f"lighting={should_enable_lighting}"
            )

        optimizer.zero_grad(set_to_none=True)

        cameras_per_step = min(
            int(args.cameras_per_step),
            len(cameras),
        )

        selected_camera_indices = rng.choice(
            len(cameras),
            size=cameras_per_step,
            replace=False,
        )

        step_cameras = [
            cameras[index]
            for index in selected_camera_indices
        ]

        total_image_loss = torch.zeros(
            (),
            device=device,
            dtype=torch.float32,
        )

        valid_camera_count = 0

        for camera in step_cameras:
            predicted_rgba, _ = renderer(camera, neural_scene)

            # Fully culled/no-slice output has no graph.
            if not predicted_rgba.requires_grad:
                continue

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

            if not torch.isfinite(camera_loss):
                print(
                    f"[WARN] Invalid camera loss at iter={iteration}, "
                    f"camera={camera.image_name}; skipped."
                )
                continue

            scaled_loss = camera_loss / float(cameras_per_step)
            scaled_loss.backward()

            total_image_loss = total_image_loss + scaled_loss.detach()
            valid_camera_count += 1

        if valid_camera_count == 0:
            print(
                f"[WARN] No valid differentiable camera renders at "
                f"iteration={iteration}; skipping."
            )
            optimizer.zero_grad(set_to_none=True)
            continue

        loss_regularization = parameter_regularization(neural_scene)

        if args.optimize_sh and current_stage == "lighting":
            loss_regularization = (
                loss_regularization
                + global_sh_regularization(neural_scene)
                + neighbor_sh_regularization(
                    neural_scene,
                    neighbor_pairs,
                )
            )

        loss_regularization.backward()

        accumulate_gradient_stats(
            neural_scene,
            gradient_stats,
        )

        invalid_gradients = [
            name
            for name, parameter in neural_scene.named_parameters()
            if (
                parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
            )
        ]

        if invalid_gradients:
            print(
                f"[WARN] Non-finite gradients at iter={iteration}: "
                f"{invalid_gradients}"
            )
            optimizer.zero_grad(set_to_none=True)
            continue

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            neural_scene.parameters(),
            max_norm=MAX_GRAD_NORM,
            error_if_nonfinite=True,
        )

        optimizer.step()

        # ----------------------------------------------------
        # Optional topology updates.
        # ----------------------------------------------------

        after_warmup = iteration >= SHAPE_WARMUP_ITERS

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
            lifecycle_camera_count = min(
                args.lifecycle_cameras,
                len(cameras),
            )

            lifecycle_cameras = cameras[:lifecycle_camera_count]

            gradient_summary = finalize_gradient_stats(
                gradient_stats
            )

            lifecycle_stats = collect_batched_lifecycle_stats(
                cameras=lifecycle_cameras,
                neural_scene=neural_scene,
                fno_radius=FNO_RADIUS,
                min_patch_size_px=args.min_projected_patch_size,
                margin_px=args.active_slice_margin,
                flip_projection_y=args.flip_projection_y,
            )

            topology_changed = False

            if do_prune and len(neural_scene.slices) > 1:
                # First use cheap visibility/opacity/projected-area statistics
                # to rank weak candidates.
                prune_candidates = choose_batched_prune_candidates(
                    lifecycle_stats=lifecycle_stats,
                    max_candidates=args.lifecycle_prune_candidates,
                )
            
                prune_camera_count = min(
                    int(args.prune_selection_cameras),
                    len(cameras),
                )
            
                prune_cameras = cameras[:prune_camera_count]
            
                # Then run expensive actual contribution tests only on those candidates.
                prune_indices = choose_contribution_prune_indices(
                    renderer=renderer,
                    neural_scene=neural_scene,
                    cameras=prune_cameras,
                    candidate_indices=prune_candidates,
                    max_prunes=args.max_prunes_per_update,
                    contribution_threshold=args.contribution_threshold,
                )
            
                if prune_indices:
                    print(
                        "[PRUNE] Removing contribution-tested slices:",
                        prune_indices,
                    )
            
                    prune_slices(
                        neural_scene=neural_scene,
                        prune_indices=prune_indices,
                    )
            
                    topology_changed = True
            
                    # Avoid splitting in the same topology-update cycle because
                    # candidate statistics were calculated before pruning.
                    do_split = False

            if do_split:
                split_candidates = choose_batched_split_candidates(
                    lifecycle_stats=lifecycle_stats,
                    gradient_summary=gradient_summary,
                    max_candidates=args.lifecycle_split_candidates,
                )

                max_total_slices = (
                    None
                    if args.max_slices <= 0
                    else int(args.max_slices)
                )

                split_indices = split_top_slices(
                    neural_scene=neural_scene,
                    candidate_indices=split_candidates,
                    gradient_stats=gradient_summary,
                    max_splits=args.max_splits_per_update,
                    max_total_slices=max_total_slices,
                    mode_selector=None,
                )

                if split_indices:
                    print("Split slices:", split_indices)
                    topology_changed = True

            if topology_changed:
                # Critical: new child Parameters / changed ModuleList require
                # a fresh optimizer.
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
                    "surface_checkpoint": str(args.surface_checkpoint),
                    "volume_checkpoint": str(args.volume_checkpoint),
                },
            )

            save_checkpoint_previews(
                renderer=renderer,
                neural_scene=neural_scene,
                cameras=cameras,
                output_dir=output_dir,
                iteration=iteration,
                max_views=args.preview_views,
            )

            print("Saved checkpoint:", checkpoint_path)

        if iteration % 25 == 0 or iteration == args.iterations - 1:
            total_loss = total_image_loss + loss_regularization.detach()

            print(
                f"iter={iteration:05d} "
                f"loss={total_loss.item():.8f} "
                f"image={total_image_loss.item():.8f} "
                f"regularization={loss_regularization.item():.8f} "
                f"grad_norm={gradient_norm.item():.6e} "
                f"slices={len(neural_scene.slices)}"
            )

    # --------------------------------------------------------
    # Final render / checkpoint.
    # --------------------------------------------------------

    final_camera = cameras[0]

    with torch.no_grad():
        final_rgba, _ = renderer(final_camera, neural_scene)

    final_rgba = final_rgba.clamp(0.0, 1.0)

    save_rgb_chw(
        visible_rgb_from_rgba(final_rgba)[0],
        output_dir / "final_visible_rgb.png",
    )

    save_rgb_chw(
        final_rgba[0, :3],
        output_dir / "final_premultiplied_rgb.png",
    )

    save_alpha_hw(
        final_rgba[0, 3],
        output_dir / "final_alpha.png",
    )

    save_rgb_chw(
        final_camera.original_image.detach(),
        output_dir / "target_rgb.png",
    )

    final_iteration = max(start_iteration, args.iterations - 1)

    final_checkpoint = output_dir / (
        f"checkpoint_iter_{final_iteration:06d}_final.pt"
    )

    save_neural_scene_checkpoint(
        path=final_checkpoint,
        neural_scene=neural_scene,
        optimizer=optimizer,
        iteration=final_iteration,
        metadata={
            "num_slices": len(neural_scene.slices),
            "initial_env_id": initial_env_id,
            "optimize_sh": args.optimize_sh,
            "fno_radius": FNO_RADIUS,
            "surface_checkpoint": str(args.surface_checkpoint),
            "volume_checkpoint": str(args.volume_checkpoint),
        },
    )

    print("Saved final checkpoint:", final_checkpoint)
    print("Done.")
    print("Outputs:", output_dir)


if __name__ == "__main__":
    main()