#!/usr/bin/env python
"""
Initial fixed-voxel neural-scene optimizer.

This is the first optimizer for the fixed voxel hierarchy design:

    - Voxel centers and world sizes are fixed after initialization.
    - Each voxel softly chooses surface vs. volume rendering.
    - Each voxel has a soft alive/dead gate.
    - Frozen surface/volume FNO checkpoints produce local RGBA patches.
    - Patches are projected and alpha-composited into each training camera.

This version intentionally does not yet schedule:
    - octree splitting;
    - hard removal of dead blocks;
    - neighbor activation.

Those operations should be added only after the fixed-grid renderer and
optimization behavior are verified.
"""

from __future__ import annotations

import math
import random
import sys
from argparse import ArgumentParser
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from pytorch_msssim import ssim
import torch.nn.functional as F

# ============================================================
# PATH SETUP
# ============================================================

REPO_DIR = Path(__file__).resolve().parent
FNO_ROOT = REPO_DIR.parent

sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(FNO_ROOT))

# ============================================================
# 3DGS / PROJECT IMPORTS
# ============================================================

from arguments import ModelParams
from scene import GaussianModel, Scene

from implicit_model import ImplicitFNOImageModel

from neural_lifecycle import voxel_chunk_seeds

from fixed_voxel_parameters import FNO_FEATURE_DIM

from fixed_voxel_scene import (
    FixedVoxelScene,
    activate_neighbor_from_source,
    available_face_neighbors_nonoverlapping,
    make_blocks_from_voxel_seeds,
    root_grid_index_from_center,
    split_scene_block,
)

from fixed_voxel_renderer import FixedVoxelRenderer

from fixed_voxel_lifecycle import (
    accumulate_voxel_gradient_stats,
    choose_dead_block_keys,
    choose_neighbor_activation_candidates,
    choose_stably_dead_blocks,
    choose_voxel_split_candidates,
    collect_voxel_lifecycle_stats,
    finalize_voxel_gradient_stats,
    make_voxel_gradient_stats,
    print_voxel_lifecycle_summary,
    update_dead_streaks,
)


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_OUTPUT_DIR = REPO_DIR / "fixed_voxel_outputs"

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

FNO_RADIUS = 2.2
BACKGROUND_RGB = (0.0, 0.0, 0.0)

# Fixed-voxel optimization learning rates.
SHAPE_LR = 2e-2
MATERIAL_LR = 3e-3
GATE_LR = 2e-3

# Keep SH optional and conservative initially.
LOCAL_SH_LR = 1e-4
GLOBAL_SH_LR = 1e-3

MAX_GRAD_NORM = 1.0

# Regularization.
PARAMETER_REG_WEIGHT = 1.0
ALIVE_REG_WEIGHT = 1e-5
MODE_ENTROPY_WEIGHT = 0.0

GLOBAL_SH_BOUND = 0.005
GLOBAL_SH_REG_WEIGHT = 1e-2


# ============================================================
# FNO CHECKPOINT LOADING
# ============================================================

def load_fno_checkpoint(checkpoint_path, device, expected_mode):
    """
    Load a current ImplicitFNOImageModel checkpoint.

    The checkpoint must come from train_implicit.py.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"FNO checkpoint not found:\n  {checkpoint_path}"
        )

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
            f"Expected a '{expected_mode}' checkpoint, "
            f"but checkpoint reports '{checkpoint_mode}'."
        )

    param_mean = np.asarray(
        checkpoint_data["param_mean"],
        dtype=np.float32,
    )

    param_std = np.asarray(
        checkpoint_data["param_std"],
        dtype=np.float32,
    )

    latent_dim = len(param_mean)

    if latent_dim != FNO_FEATURE_DIM:
        raise RuntimeError(
            f"Expected {FNO_FEATURE_DIM} FNO features, got {latent_dim}. "
            "Check that this is a new implicit-B-spline checkpoint."
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
        f"Loaded {expected_mode} FNO: "
        f"features={latent_dim}, "
        f"patch={image_width}x{image_height}, "
        f"modes={fno_modes}"
    )

    return {
        "model": model,
        "param_mean": param_mean,
        "param_std": param_std,
        "checkpoint_path": str(checkpoint_path),
        "height": image_height,
        "width": image_width,
    }


# ============================================================
# SH ENVIRONMENT BANK
# ============================================================

def sh_lm_list(order):
    return [
        (l, m)
        for l in range(order + 1)
        for m in range(-l, l + 1)
    ]


def sh_for_global_env(env_id, order=2):
    """
    Matches the renderer SH environment generation.
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
            sh_for_global_env(env_id, SH_ORDER)
            for env_id in range(NUM_GLOBAL_ENVS)
        ],
        axis=0,
    )

    return torch.tensor(
        sh_bank,
        dtype=torch.float32,
        device=device,
    )


# ============================================================
# POINT-CLOUD COLOR EXTRACTION
# ============================================================

@torch.no_grad()
def try_get_point_colors(gaussian_model):
    """
    Best-effort extraction of RGB priors from a common 3DGS GaussianModel.

    Returns [N, 3] RGB in [0, 1], or None.
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
            dc_coefficients = features[:, 0, :]

            try:
                from utils.sh_utils import SH2RGB

                colors = SH2RGB(dc_coefficients)
                return colors.clamp(0.0, 1.0)

            except Exception as exc:
                print(
                    "[WARN] Found Gaussian SH/DC features but SH2RGB "
                    f"conversion failed: {exc}"
                )

    print(
        "[WARN] Could not extract point-cloud colors. "
        "Blocks will use random color initialization."
    )

    return None


# ============================================================
# INITIAL BLOCK PARAMETER PRIOR
# ============================================================

def make_block_initialization(
    rng,
    initial_global_sh,
    seed,
):
    """
    Callback used by make_blocks_from_voxel_seeds().

    Point-cloud voxel median RGB is used when present. Other parameters are
    initialized from the same ranges as your training distribution.
    """
    seed_color = seed.get("color")

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
    }


# ============================================================
# OPTIMIZER / LOSSES
# ============================================================

def make_optimizer(voxel_scene, optimize_sh):
    """
    Construct a fresh optimizer.

    Rebuild this optimizer after future split/death/activation topology
    changes, because ModuleDict parameters will have changed.
    """
    shape_parameters = []
    material_parameters = []
    gate_parameters = []
    local_sh_parameters = []

    for block in voxel_scene.blocks.values():
        shape_parameters.extend(
            [
                block.raw_ctrl,
                block.raw_sigma,
            ]
        )

        material_parameters.extend(
            [
                block.raw_base_color_r,
                block.raw_base_color_g,
                block.raw_base_color_b,
                block.raw_opacity,
                block.raw_roughness,
            ]
        )

        gate_parameters.extend(
            [
                block.raw_alive_logit,
                block.raw_mode_logit,
            ]
        )

        if (
            optimize_sh
            and block.raw_local_sh_delta.requires_grad
        ):
            local_sh_parameters.append(block.raw_local_sh_delta)

    groups = [
        {
            "params": shape_parameters,
            "lr": SHAPE_LR,
        },
        {
            "params": material_parameters,
            "lr": MATERIAL_LR,
        },
        {
            "params": gate_parameters,
            "lr": GATE_LR,
        },
    ]

    if optimize_sh and local_sh_parameters:
        groups.append(
            {
                "params": local_sh_parameters,
                "lr": LOCAL_SH_LR,
            }
        )

    if (
        optimize_sh
        and voxel_scene.lighting.raw_global_sh_delta.requires_grad
    ):
        groups.append(
            {
                "params": [
                    voxel_scene.lighting.raw_global_sh_delta
                ],
                "lr": GLOBAL_SH_LR,
            }
        )

    return torch.optim.Adam(groups)


def visible_rgb_from_rgba(rgba):
    background = torch.tensor(
        BACKGROUND_RGB,
        dtype=rgba.dtype,
        device=rgba.device,
    ).view(1, 3, 1, 1)

    return rgba[:, :3] + (
        1.0 - rgba[:, 3:4]
    ) * background


def image_loss(
    predicted_rgba,
    target_rgb,
    l1_weight=0.4,
    mse_weight=0.4,
    ssim_weight=0.2,
):
    """
    Mixed RGB reconstruction loss.

    The neural renderer produces premultiplied RGBA. It is composited onto
    BACKGROUND_RGB before comparison with the standard RGB target image.

    Returns:
        total_loss: differentiable scalar tensor
        metrics: detached tensors for logging
    """
    if abs(
        float(l1_weight)
        + float(mse_weight)
        + float(ssim_weight)
        - 1.0
    ) > 1e-6:
        raise ValueError(
            "l1_weight + mse_weight + ssim_weight must equal 1.0"
        )

    predicted_rgb = visible_rgb_from_rgba(
        predicted_rgba
    ).clamp(0.0, 1.0)

    target_rgb = target_rgb.clamp(0.0, 1.0)

    l1_value = F.l1_loss(
        predicted_rgb,
        target_rgb,
    )

    mse_value = F.mse_loss(
        predicted_rgb,
        target_rgb,
    )

    # SSIM returns similarity, where 1.0 is identical.
    ssim_value = ssim(
        predicted_rgb,
        target_rgb,
        data_range=1.0,
        size_average=True,
    )

    ssim_loss = 1.0 - ssim_value

    total_loss = (
        float(l1_weight) * l1_value
        + float(mse_weight) * mse_value
        + float(ssim_weight) * ssim_loss
    )

    metrics = {
        "l1": l1_value.detach(),
        "mse": mse_value.detach(),
        "ssim": ssim_value.detach(),
        "ssim_loss": ssim_loss.detach(),
    }

    return total_loss, metrics


def regularization_loss(
    voxel_scene,
    alive_weight=ALIVE_REG_WEIGHT,
    mode_entropy_weight=MODE_ENTROPY_WEIGHT,
):
    """
    Sum block regularization plus optional global lighting regularization.
    """
    if len(voxel_scene) == 0:
        return torch.zeros(
            (),
            device=voxel_scene.global_sh.device,
        )

    total = torch.zeros(
        (),
        dtype=torch.float32,
        device=voxel_scene.global_sh.device,
    )

    for block in voxel_scene.blocks.values():
        total = total + PARAMETER_REG_WEIGHT * block.regularization_loss(
            shared_sh=voxel_scene.global_sh,
            alive_weight=alive_weight,
            mode_entropy_weight=mode_entropy_weight,
        )

    global_delta = (
        voxel_scene.global_sh
        - voxel_scene.lighting.initial_global_sh
    )

    total = total + GLOBAL_SH_REG_WEIGHT * torch.mean(
        global_delta.square()
    )

    return total


# ============================================================
# PREVIEW / CHECKPOINT HELPERS
# ============================================================

def save_rgb_chw(rgb_chw, path):
    if torch.is_tensor(rgb_chw):
        rgb_chw = rgb_chw.detach().cpu().numpy()

    rgb_hwc = np.transpose(rgb_chw, (1, 2, 0))

    imageio.imwrite(
        path,
        (
            np.clip(rgb_hwc, 0.0, 1.0)
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


@torch.no_grad()
def save_previews(
    renderer,
    voxel_scene,
    cameras,
    output_dir,
    iteration,
    max_views,
):
    preview_dir = (
        Path(output_dir)
        / "previews"
        / f"iter_{iteration:06d}"
    )

    preview_dir.mkdir(parents=True, exist_ok=True)

    for view_index, camera in enumerate(cameras[:max_views]):
        predicted_rgba, _ = renderer(camera, voxel_scene)

        predicted_rgba = predicted_rgba.clamp(0.0, 1.0)

        visible_rgb = visible_rgb_from_rgba(predicted_rgba)[0]
        target_rgb = camera.original_image.detach()

        save_rgb_chw(
            visible_rgb,
            preview_dir / f"view_{view_index:02d}_prediction.png",
        )

        save_alpha_hw(
            predicted_rgba[0, 3],
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
            visible_rgb.detach().cpu().numpy(),
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
            (
                comparison * 255.0
                + 0.5
            ).astype(np.uint8),
        )

    print("Saved previews:", preview_dir)


def block_summary(voxel_scene):
    """
    Small detached summary for logging/checkpoints.
    """
    if len(voxel_scene) == 0:
        return {
            "num_blocks": 0,
            "mean_alive": 0.0,
            "mean_volume_weight": 0.0,
        }

    alive_weights = torch.stack(
        [
            block.alive_weight.detach()
            for block in voxel_scene.blocks.values()
        ]
    )

    volume_weights = torch.stack(
        [
            block.volume_weight.detach()
            for block in voxel_scene.blocks.values()
        ]
    )

    return {
        "num_blocks": len(voxel_scene),
        "mean_alive": float(alive_weights.mean().cpu()),
        "mean_volume_weight": float(volume_weights.mean().cpu()),
    }


def save_checkpoint(
    path,
    voxel_scene,
    optimizer,
    iteration,
    metadata,
):
    """
    Save fixed voxel scene topology and learnable state.

    The frozen FNO checkpoints are not embedded; their paths are stored in
    metadata and must be supplied again when resuming/rendering.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    block_specs = []

    for key, block in voxel_scene.iter_blocks_with_keys():
        block_specs.append(
            {
                "key": key,
                "level": block.level,
                "grid_index": block.grid_index,
                "parent_key": block.parent_key,
                "world_size": float(block.world_size.detach().cpu()),
            }
        )

    torch.save(
        {
            "iteration": int(iteration),
            "scene_state": voxel_scene.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "block_specs": block_specs,
            "root_voxel_size": voxel_scene.root_voxel_size,
            "metadata": metadata,
        },
        path,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = ArgumentParser(
        description=(
            "Optimize a fixed voxel scene using frozen surface and "
            "volume implicit-B-spline FNO renderers."
        )
    )

    lp = ModelParams(parser)

    parser.add_argument("--camera_start", type=int, default=0)
    parser.add_argument("--num_cameras", type=int, default=8)
    parser.add_argument("--cameras_per_step", type=int, default=4)

    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--voxel_size", type=float, default=0.5)
    parser.add_argument("--min_points_per_voxel", type=int, default=50)

    parser.add_argument(
        "--max_initial_blocks",
        type=int,
        default=64,
        help="Use 0 for every valid initialization voxel.",
    )

    parser.add_argument(
        "--initial_volume_probability",
        type=float,
        default=0.5,
        help=(
            "Initial soft volume probability for every block. "
            "0 = surface, 1 = volume, 0.5 = neutral."
        ),
    )

    parser.add_argument(
        "--initial_alive_probability",
        type=float,
        default=0.99,
    )

    parser.add_argument("--optimize_sh", action="store_true")

    parser.add_argument("--fno_batch_size", type=int, default=16)

    parser.add_argument("--use_tile_renderer", action="store_true")
    parser.add_argument("--tile_size", type=int, default=128)
    parser.add_argument("--tile_roi_margin", type=int, default=16)
    parser.add_argument("--max_patches_per_tile", type=int, default=0)

    parser.add_argument(
        "--min_projected_patch_size",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--active_slice_margin",
        type=float,
        default=64.0,
    )

    parser.add_argument(
        "--alive_cull_threshold",
        type=float,
        default=0.0,
        help=(
            "Skip rendering blocks below this detached alive weight. "
            "Use 0 initially so every block remains differentiable."
        ),
    )

    parser.add_argument(
        "--disable_fno_activation_checkpointing",
        action="store_true",
    )

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
        "--checkpoint_interval",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--preview_views",
        type=int,
        default=2,
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
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    
    parser.add_argument(
        "--l1_weight",
        type=float,
        default=0.4,
        help="Weight of RGB L1 reconstruction loss.",
    )
    
    parser.add_argument(
        "--mse_weight",
        type=float,
        default=0.4,
        help="Weight of RGB MSE reconstruction loss.",
    )
    
    parser.add_argument(
        "--ssim_weight",
        type=float,
        default=0.2,
        help="Weight of 1 - SSIM structural loss.",
    )
    
    parser.add_argument(
        "--lifecycle_interval",
        type=int,
        default=0,
        help=(
            "Run fixed-voxel lifecycle reporting/topology every N iterations. "
            "Use 0 to disable."
        ),
    )
    
    parser.add_argument(
        "--max_blocks",
        type=int,
        default=128,
        help=(
            "Maximum number of voxel blocks after splitting. "
            "Use 0 for no cap, though this is not recommended."
        ),
    )
    
    parser.add_argument(
        "--max_voxel_level",
        type=int,
        default=2,
        help=(
            "Maximum octree refinement level. Root blocks are level 0. "
            "A root split creates level-1 children."
        ),
    )
    
    parser.add_argument(
        "--max_splits_per_update",
        type=int,
        default=1,
        help=(
            "Maximum parent blocks split in one lifecycle update. "
            "Each split creates 8 children and adds 7 net blocks."
        ),
    )
    
    parser.add_argument(
        "--split_candidate_count",
        type=int,
        default=8,
        help="Number of high-gradient blocks considered for splitting.",
    )
    
    parser.add_argument(
        "--split_min_alive_weight",
        type=float,
        default=0.25,
        help="Do not split blocks below this alive weight.",
    )
    
    parser.add_argument(
        "--split_min_world_size",
        type=float,
        default=0.0,
        help=(
            "Do not split blocks smaller than this world size. "
            "Use 0 to rely only on --max_voxel_level."
        ),
    )
    
    parser.add_argument(
        "--split_min_shape_gradient",
        type=float,
        default=0.0,
        help="Minimum average shape-gradient norm required to split.",
    )
    
    parser.add_argument(
        "--enable_hard_pruning",
        action="store_true",
        help=(
            "Hard-remove blocks whose soft alive weight is below "
            "--dead_alive_threshold."
        ),
    )
    
    parser.add_argument(
        "--disable_splitting",
        action="store_true",
        help=(
            "Report split candidates but do not create octree children. "
            "Useful for checking whether candidate selection is sensible."
        ),
    )
    
    parser.add_argument(
        "--dead_alive_threshold",
        type=float,
        default=0.02,
        help=(
            "A block is considered nearly dead when alive_weight is at "
            "or below this value."
        ),
    )
    
    parser.add_argument(
        "--dead_intervals_before_prune",
        type=int,
        default=2,
        help=(
            "Number of consecutive lifecycle intervals a block must remain "
            "nearly dead before hard removal."
        ),
    )
    
    parser.add_argument(
        "--max_prunes_per_update",
        type=int,
        default=2,
        help="Maximum blocks hard-pruned during one lifecycle update.",
    )
    
    
    parser.add_argument(
        "--neighbor_activation_interval",
        type=int,
        default=0,
        help=(
            "Try activating empty face-neighbor blocks every N iterations. "
            "Use 0 to disable."
        ),
    )
    
    parser.add_argument(
        "--max_neighbor_activations_per_update",
        type=int,
        default=1,
        help="Maximum new neighboring blocks activated per update.",
    )
    
    parser.add_argument(
        "--neighbor_candidate_count",
        type=int,
        default=8,
        help="Number of high-gradient source blocks considered for expansion.",
    )
    
    parser.add_argument(
        "--neighbor_min_alive_weight",
        type=float,
        default=0.50,
        help="Only expand from blocks with at least this alive weight.",
    )
    
    parser.add_argument(
        "--neighbor_min_gradient",
        type=float,
        default=0.0,
        help="Minimum combined gradient score for neighbor expansion.",
    )
    
    parser.add_argument(
        "--disable_neighbor_activation",
        action="store_true",
        help="Disable neighbor activation even when interval is nonzero.",
    )
    
    parser.add_argument(
        "--prune_interval",
        type=int,
        default=0,
        help=(
            "Check for hard-prunable dead blocks every N iterations. "
            "Use 0 to disable."
        ),
    )
    
    parser.add_argument(
        "--split_interval",
        type=int,
        default=0,
        help=(
            "Try octree splitting every N iterations. Use 0 to disable."
        ),
    )

    args = parser.parse_args()
    
    loss_weight_sum = (
        args.l1_weight
        + args.mse_weight
        + args.ssim_weight
    )
    
    if abs(loss_weight_sum - 1.0) > 1e-6:
        raise ValueError(
            "Loss weights must sum to 1.0, got "
            f"{loss_weight_sum:.6f}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this fixed voxel optimizer."
        )

    if not (0.0 < args.initial_alive_probability < 1.0):
        raise ValueError(
            "--initial_alive_probability must be strictly in (0, 1)."
        )

    if not (0.0 < args.initial_volume_probability < 1.0):
        raise ValueError(
            "--initial_volume_probability must be strictly in (0, 1)."
        )

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
    print("Output:", output_dir)

    # --------------------------------------------------------
    # Frozen FNO checkpoints.
    # --------------------------------------------------------

    surface_bundle = load_fno_checkpoint(
        checkpoint_path=args.surface_checkpoint,
        device=device,
        expected_mode="surface",
    )

    volume_bundle = load_fno_checkpoint(
        checkpoint_path=args.volume_checkpoint,
        device=device,
        expected_mode="volume",
    )

    # --------------------------------------------------------
    # Fixed voxel renderer.
    # --------------------------------------------------------

    max_patches_per_tile = (
        None
        if args.max_patches_per_tile <= 0
        else int(args.max_patches_per_tile)
    )

    renderer = FixedVoxelRenderer(
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
        min_projected_patch_size=args.min_projected_patch_size,
        active_slice_margin=args.active_slice_margin,
        alive_cull_threshold=args.alive_cull_threshold,
        use_fno_activation_checkpointing=(
            not args.disable_fno_activation_checkpointing
        ),
    ).to(device)

    # --------------------------------------------------------
    # Load 3DGS data / training cameras / point cloud.
    # --------------------------------------------------------

    scene_args = lp.extract(args)

    if not scene_args.model_path:
        scene_args.model_path = str(
            REPO_DIR / "output" / "fixed_voxel_scene"
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
        raise RuntimeError("No training cameras were selected.")

    print(
        f"Using cameras {args.camera_start} through {camera_end - 1}; "
        f"count={len(cameras)}"
    )

    point_xyz = gaussian_model.get_xyz.detach()
    point_colors = try_get_point_colors(gaussian_model)

    # --------------------------------------------------------
    # Fixed voxel initialization.
    # --------------------------------------------------------

    sh_bank = build_sh_bank(device)

    initial_env_id = int(
        rng.integers(0, NUM_GLOBAL_ENVS)
    )

    initial_global_sh = sh_bank[initial_env_id].detach().clone()

    max_chunks = (
        None
        if args.max_initial_blocks <= 0
        else int(args.max_initial_blocks)
    )

    seeds = voxel_chunk_seeds(
        xyz=point_xyz,
        colors=point_colors,
        voxel_size=args.voxel_size,
        min_points=args.min_points_per_voxel,
        max_chunks=max_chunks,
    )

    if not seeds:
        raise RuntimeError(
            "No valid voxel seeds. Increase --voxel_size or lower "
            "--min_points_per_voxel."
        )

    # Important fixed-grid convention:
    #
    # voxel_chunk_seeds() returns median point locations. For a true fixed
    # hierarchy, root block centers must lie exactly at grid-cell centers.
    #
    # We convert each seed's median back to its floor-grid index, then use:
    #
    #     center = (index + 0.5) * voxel_size
    #
    # This ensures child splits and same-level neighbors align correctly.
    aligned_seeds = []

    for seed in seeds:
        grid_index = root_grid_index_from_center(
            center=seed["center"],
            root_voxel_size=args.voxel_size,
        )

        aligned_center = (
            torch.tensor(
                grid_index,
                dtype=torch.float32,
                device=seed["center"].device,
            )
            + 0.5
        ) * float(args.voxel_size)

        aligned_seed = dict(seed)
        aligned_seed["center"] = aligned_center
        aligned_seeds.append(aligned_seed)

    def init_parameter_fn(seed):
        return make_block_initialization(
            rng=rng,
            initial_global_sh=initial_global_sh,
            seed=seed,
        )

    blocks = make_blocks_from_voxel_seeds(
        seeds=aligned_seeds,
        root_voxel_size=args.voxel_size,
        init_parameter_fn=init_parameter_fn,
        initial_global_sh=initial_global_sh,
        device=device,
        optimize_environment=args.optimize_sh,
        initial_volume_probability=args.initial_volume_probability,
        initial_alive_probability=args.initial_alive_probability,
    )

    voxel_scene = FixedVoxelScene(
        blocks=blocks,
        initial_global_sh=initial_global_sh,
        optimize_sh=args.optimize_sh,
        global_sh_bound=GLOBAL_SH_BOUND,
        root_voxel_size=args.voxel_size,
    ).to(device)

    print(
        f"Initialized fixed voxel scene with {len(voxel_scene)} blocks."
    )

    # --------------------------------------------------------
    # Optimizer.
    # --------------------------------------------------------

    optimizer = make_optimizer(
        voxel_scene=voxel_scene,
        optimize_sh=args.optimize_sh,
    )
    
    gradient_stats = make_voxel_gradient_stats(voxel_scene)
    
    dead_streaks = {}

    # --------------------------------------------------------
    # Initial sanity render.
    # --------------------------------------------------------

    with torch.no_grad():
        sanity_rgba, _ = renderer(
            cameras[0],
            voxel_scene,
        )

    if not torch.isfinite(sanity_rgba).all():
        raise RuntimeError(
            "Initial fixed voxel render contains NaN or Inf."
        )

    print("Sanity RGBA shape:", tuple(sanity_rgba.shape))

    # --------------------------------------------------------
    # Optimization.
    # --------------------------------------------------------

    for iteration in range(args.iterations):
        optimizer.zero_grad(set_to_none=True)

        cameras_per_step = min(
            int(args.cameras_per_step),
            len(cameras),
        )

        selected_indices = rng.choice(
            len(cameras),
            size=cameras_per_step,
            replace=False,
        )

        step_cameras = [
            cameras[index]
            for index in selected_indices
        ]

        total_image_loss = torch.zeros(
            (),
            dtype=torch.float32,
            device=device,
        )

        valid_camera_count = 0
        
        total_l1_loss = torch.zeros(
            (),
            dtype=torch.float32,
            device=device,
        )
        
        total_mse_loss = torch.zeros(
            (),
            dtype=torch.float32,
            device=device,
        )
        
        total_ssim = torch.zeros(
            (),
            dtype=torch.float32,
            device=device,
        )

        for camera in step_cameras:
            predicted_rgba, _ = renderer(
                camera,
                voxel_scene,
            )

            if not predicted_rgba.requires_grad:
                continue

            target_rgb = camera.original_image.detach()

            if target_rgb.ndim == 3:
                target_rgb = target_rgb.unsqueeze(0)

            target_rgb = target_rgb.to(
                device=device,
                dtype=predicted_rgba.dtype,
            )

            camera_loss, camera_metrics = image_loss(
                predicted_rgba=predicted_rgba,
                target_rgb=target_rgb,
                l1_weight=args.l1_weight,
                mse_weight=args.mse_weight,
                ssim_weight=args.ssim_weight,
            )

            if not torch.isfinite(camera_loss):
                print(
                    f"[WARN] Non-finite camera loss at iteration "
                    f"{iteration}, camera={camera.image_name}."
                )
                continue

            scaled_loss = camera_loss / float(cameras_per_step)
            scaled_loss.backward()

            total_image_loss = total_image_loss + scaled_loss.detach()
            
            total_l1_loss = total_l1_loss + (
                camera_metrics["l1"] / float(cameras_per_step)
            )
            
            total_mse_loss = total_mse_loss + (
                camera_metrics["mse"] / float(cameras_per_step)
            )
            
            total_ssim = total_ssim + (
                camera_metrics["ssim"] / float(cameras_per_step)
            )
            valid_camera_count += 1

        if valid_camera_count == 0:
            print(
                f"[WARN] No valid camera losses at iteration={iteration}; "
                "skipping optimizer update."
            )
            optimizer.zero_grad(set_to_none=True)
            continue

        loss_reg = regularization_loss(
            voxel_scene=voxel_scene,
            alive_weight=ALIVE_REG_WEIGHT,
            mode_entropy_weight=MODE_ENTROPY_WEIGHT,
        )

        loss_reg.backward()
        
        accumulate_voxel_gradient_stats(
            voxel_scene,
            gradient_stats,
        )

        bad_gradient_names = [
            name
            for name, parameter in voxel_scene.named_parameters()
            if (
                parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
            )
        ]

        if bad_gradient_names:
            print(
                f"[WARN] Non-finite gradients at iteration={iteration}: "
                f"{bad_gradient_names}"
            )
            optimizer.zero_grad(set_to_none=True)
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(
            voxel_scene.parameters(),
            max_norm=MAX_GRAD_NORM,
            error_if_nonfinite=True,
        )

        optimizer.step()

        # ----------------------------------------------------
        # Logging.
        # ----------------------------------------------------

        if iteration % 25 == 0 or iteration == args.iterations - 1:
            summary = block_summary(voxel_scene)

            total_loss = total_image_loss + loss_reg.detach()

            print(
                f"iter={iteration:05d} "
                f"loss={total_loss.item():.8f} "
                f"image={total_image_loss.item():.8f} "
                f"reg={loss_reg.item():.8f} "
                f"l1={total_l1_loss.item():.6f} "
                f"mse={total_mse_loss.item():.6f} "
                f"ssim={total_ssim.item():.6f} "
                f"grad={grad_norm.item():.4e} "
                f"blocks={summary['num_blocks']} "
                f"mean_alive={summary['mean_alive']:.4f} "
                f"mean_volume={summary['mean_volume_weight']:.4f}"
            )
            
        # --------------------------------------------------------
        # Fixed voxel lifecycle / topology update.
        # --------------------------------------------------------
        
        # --------------------------------------------------------
        # Fixed-voxel scheduled lifecycle / topology update.
        #
        # Operations have independent schedules:
        #   1. neighbor activation
        #   2. pruning
        #   3. splitting
        #
        # If multiple are due on the same iteration, priority is:
        #   neighbor activation -> prune -> split
        #
        # Once one operation changes topology, stop there, rebuild the
        # optimizer, and use fresh stats at the next scheduled update.
        # --------------------------------------------------------
        
        do_neighbor_activation = (
            not args.disable_neighbor_activation
            and args.neighbor_activation_interval > 0
            and iteration > 0
            and iteration % args.neighbor_activation_interval == 0
        )
        
        do_prune = (
            args.enable_hard_pruning
            and args.prune_interval > 0
            and iteration > 0
            and iteration % args.prune_interval == 0
        )
        
        do_split = (
            not args.disable_splitting
            and args.split_interval > 0
            and iteration > 0
            and iteration % args.split_interval == 0
        )
        
        do_topology_update = (
            do_neighbor_activation
            or do_prune
            or do_split
        )
        
        if do_topology_update:
            lifecycle_stats = collect_voxel_lifecycle_stats(
                voxel_scene
            )
        
            gradient_summary = finalize_voxel_gradient_stats(
                gradient_stats
            )
        
            print_voxel_lifecycle_summary(
                lifecycle_stats=lifecycle_stats,
                gradient_summary=gradient_summary,
                max_rows=16,
            )
        
            topology_changed = False
        
            max_blocks = (
                None
                if args.max_blocks <= 0
                else int(args.max_blocks)
            )
        
            # ====================================================
            # 1. Neighbor activation: highest priority
            # ====================================================
        
            if do_neighbor_activation:
                print(
                    f"[SCHEDULE] iteration={iteration}: "
                    "checking neighbor activation"
                )
        
                neighbor_sources = choose_neighbor_activation_candidates(
                    lifecycle_stats=lifecycle_stats,
                    gradient_summary=gradient_summary,
                    max_candidates=args.neighbor_candidate_count,
                    min_alive_weight=args.neighbor_min_alive_weight,
                    min_gradient=args.neighbor_min_gradient,
                )
        
                activated_count = 0
        
                for source_key in neighbor_sources:
                    if (
                        activated_count
                        >= args.max_neighbor_activations_per_update
                    ):
                        break
        
                    # Candidate statistics are detached. Defensive check in case
                    # scene topology was changed unexpectedly elsewhere.
                    if not voxel_scene.has_key(source_key):
                        continue
        
                    if (
                        max_blocks is not None
                        and len(voxel_scene) >= max_blocks
                    ):
                        print(
                            "[NEIGHBOR ACTIVATE] Reached max_blocks="
                            f"{max_blocks}; cannot add another block."
                        )
                        break
        
                    available_neighbors = available_face_neighbors_nonoverlapping(
                        scene=voxel_scene,
                        source_key=source_key,
                    )
        
                    if not available_neighbors:
                        continue
        
                    # Pick one available face-neighbor without directional bias.
                    target_key = available_neighbors[
                        int(rng.integers(0, len(available_neighbors)))
                    ]
        
                    try:
                        created_key, _ = activate_neighbor_from_source(
                            scene=voxel_scene,
                            source_key=source_key,
                            neighbor_grid_index=target_key[1:],
                        )
        
                        print(
                            "[NEIGHBOR ACTIVATE] "
                            f"source={source_key} -> target={created_key}"
                        )
        
                        activated_count += 1
                        topology_changed = True
        
                    except Exception as exc:
                        print(
                            "[WARN] Neighbor activation failed for "
                            f"source={source_key}: {exc}"
                        )
        
            # ====================================================
            # 2. Hard pruning: only if activation changed nothing
            # ====================================================
        
            if do_prune and not topology_changed:
                print(
                    f"[SCHEDULE] iteration={iteration}: "
                    "checking hard pruning"
                )
        
                dead_streaks = update_dead_streaks(
                    lifecycle_stats=lifecycle_stats,
                    dead_streaks=dead_streaks,
                    alive_threshold=args.dead_alive_threshold,
                )
        
                dead_keys = choose_stably_dead_blocks(
                    lifecycle_stats=lifecycle_stats,
                    dead_streaks=dead_streaks,
                    min_dead_intervals=args.dead_intervals_before_prune,
                    max_prunes=args.max_prunes_per_update,
                )
        
                # Always retain at least one scene block.
                max_removable = max(0, len(voxel_scene) - 1)
                dead_keys = dead_keys[:max_removable]
        
                if dead_keys:
                    print(
                        "[FIXED PRUNE] Removing stably dead blocks:",
                        dead_keys,
                    )
        
                    for dead_key in dead_keys:
                        if voxel_scene.has_key(dead_key):
                            voxel_scene.remove_block(dead_key)
                            dead_streaks.pop(dead_key, None)
        
                    topology_changed = True
        
            # ====================================================
            # 3. Octree splitting: only if nothing else changed
            # ====================================================
        
            if do_split and not topology_changed:
                print(
                    f"[SCHEDULE] iteration={iteration}: "
                    "checking octree splitting"
                )
        
                split_keys = choose_voxel_split_candidates(
                    lifecycle_stats=lifecycle_stats,
                    gradient_summary=gradient_summary,
                    max_candidates=args.split_candidate_count,
                    min_alive_weight=args.split_min_alive_weight,
                    min_world_size=args.split_min_world_size,
                    min_shape_gradient=args.split_min_shape_gradient,
                    max_level=args.max_voxel_level,
                )
        
                successful_splits = 0
        
                for parent_key in split_keys:
                    if successful_splits >= args.max_splits_per_update:
                        break
        
                    if not voxel_scene.has_key(parent_key):
                        continue
        
                    # One parent disappears; eight children appear:
                    # net growth = +7 blocks.
                    if (
                        max_blocks is not None
                        and len(voxel_scene) + 7 > max_blocks
                    ):
                        print(
                            "[FIXED SPLIT] Cannot split because max_blocks "
                            f"would be exceeded: current={len(voxel_scene)}, "
                            f"cap={max_blocks}."
                        )
                        break
        
                    try:
                        child_keys = split_scene_block(
                            scene=voxel_scene,
                            parent_key=parent_key,
                            remove_parent=True,
                        )
        
                        print(
                            "[FIXED SPLIT] "
                            f"parent={parent_key} -> "
                            f"{len(child_keys)} children"
                        )
        
                        successful_splits += 1
                        topology_changed = True
        
                    except Exception as exc:
                        print(
                            f"[WARN] Could not split block {parent_key}: {exc}"
                        )
        
            # ====================================================
            # Rebuild optimizer / reset lifecycle window.
            # ====================================================
        
            if topology_changed:
                optimizer = make_optimizer(
                    voxel_scene=voxel_scene,
                    optimize_sh=args.optimize_sh,
                )
        
                print(
                    "[FIXED TOPOLOGY] Optimizer rebuilt; "
                    f"blocks={len(voxel_scene)}"
                )
        
            # Whether topology changed or not, begin collecting a fresh
            # gradient window for the next scheduled lifecycle decision.
            gradient_stats = make_voxel_gradient_stats(
                voxel_scene
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

            metadata = {
                "initial_env_id": initial_env_id,
                "optimize_sh": bool(args.optimize_sh),
                "fno_radius": FNO_RADIUS,
                "surface_checkpoint": str(args.surface_checkpoint),
                "volume_checkpoint": str(args.volume_checkpoint),
                "summary": block_summary(voxel_scene),
            }

            save_checkpoint(
                path=checkpoint_path,
                voxel_scene=voxel_scene,
                optimizer=optimizer,
                iteration=iteration,
                metadata=metadata,
            )

            save_previews(
                renderer=renderer,
                voxel_scene=voxel_scene,
                cameras=cameras,
                output_dir=output_dir,
                iteration=iteration,
                max_views=args.preview_views,
            )

            print("Saved checkpoint:", checkpoint_path)

    # --------------------------------------------------------
    # Final render/checkpoint.
    # --------------------------------------------------------

    final_camera = cameras[0]

    with torch.no_grad():
        final_rgba, _ = renderer(
            final_camera,
            voxel_scene,
        )

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

    final_checkpoint = output_dir / (
        f"checkpoint_iter_{args.iterations - 1:06d}_final.pt"
    )

    save_checkpoint(
        path=final_checkpoint,
        voxel_scene=voxel_scene,
        optimizer=optimizer,
        iteration=args.iterations - 1,
        metadata={
            "initial_env_id": initial_env_id,
            "optimize_sh": bool(args.optimize_sh),
            "fno_radius": FNO_RADIUS,
            "surface_checkpoint": str(args.surface_checkpoint),
            "volume_checkpoint": str(args.volume_checkpoint),
            "summary": block_summary(voxel_scene),
        },
    )

    print("Saved final checkpoint:", final_checkpoint)
    print("Done.")


if __name__ == "__main__":
    main()