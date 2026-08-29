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
    choose_prune_indices,
    prune_slices,
    split_top_slices,
    save_neural_scene_checkpoint,
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

DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_ITERATIONS = 1000

POSITION_REG_WEIGHT = 1e-5
SIZE_REG_WEIGHT = 1e-3
PARAMETER_REG_WEIGHT = 1e-5

GLOBAL_SH_BOUND = 0.01
GLOBAL_SH_REG_WEIGHT = 1e-2
NEIGHBOR_SH_REG_WEIGHT = 1e-1

MAX_GRAD_NORM = 1.0


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

    # Frozen model weights; gradients through input remain valid.
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
# PROCEDURAL SH ENVIRONMENT BANK
# Matches the Blender render-generation script.
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
        if l == 1:
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

        # [9,3] -> [27], matching training metadata ordering.
        rows.append(coeffs.reshape(-1))

    sh_bank = torch.tensor(
        np.stack(rows, axis=0),
        dtype=torch.float32,
        device=device,
    )

    print("Built SH bank:", tuple(sh_bank.shape))
    return sh_bank


# ============================================================
# RANDOM INITIALIZATION PRIOR
# ============================================================

def random_slice_init(rng, shared_sh):
    """
    Sample shape/material values that stay within the ranges
    used by the FNO training data.
    """
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

        # All initial slices use the same scene-level environment.
        "sh_values": shared_sh.detach().clone(),
    }


def make_slice_from_seed(
    seed,
    mode,
    init_params,
    device,
    optimize_sh,
):
    """
    Create one learnable slice at a voxel seed.
    """
    if mode == "surface":
        roughness = init_params["roughness"]
        metallic = init_params["metallic"]
        specular = init_params["specular"]
    else:
        # Must match the convention used by volume FNO training.
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
# LIGHTING / SCENE STRUCTURE
# ============================================================

class SharedLighting(nn.Module):
    """
    Global bounded SH environment:

        SH_global = SH_initial + bound * tanh(raw_delta)
    """
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
# CAMERA / PROJECTION
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


def project_world_point(camera, point_world):
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

    # 3DGS uses row-vector convention.
    clip = point_h @ matrix

    ndc = clip[:3] / clip[3].clamp_min(1e-8)

    pixel_x = (
        (ndc[0] + 1.0)
        * 0.5
        * float(camera.image_width)
    )

    pixel_y = (
        (1.0 - ndc[1])
        * 0.5
        * float(camera.image_height)
    )

    return pixel_x, pixel_y, ndc[2]


# ============================================================
# DIFFERENTIABLE UNIFORM PATCH PLACEMENT
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
    Differentiably place patch using equal X/Y scale.
    """
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

    scale_x = canvas_wm1 / patch_size
    scale_y = canvas_hm1 / patch_size

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
# RENDERER
# ============================================================

class NeuralSceneRenderer(nn.Module):
    """
    Renders arbitrary number of neural slices.
    """
    def __init__(
        self,
        surface_model,
        volume_model,
        surface_mean,
        surface_std,
        volume_mean,
        volume_std,
        fno_radius=FNO_RADIUS,
    ):
        super().__init__()

        self.surface_model = surface_model
        self.volume_model = volume_model

        self.surface_mean = surface_mean
        self.surface_std = surface_std
        self.volume_mean = volume_mean
        self.volume_std = volume_std

        self.fno_radius = float(fno_radius)

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

    def render_slice(
        self,
        camera,
        neural_slice,
        shared_sh,
    ):
        device = camera.camera_center.device

        phi, theta, actual_radius, relative = \
            camera_to_slice_pose(
                camera,
                neural_slice.center,
            )

        if neural_slice.mode == "surface":
            model = self.surface_model
            param_mean = self.surface_mean
            param_std = self.surface_std
        else:
            model = self.volume_model
            param_mean = self.volume_mean
            param_std = self.volume_std

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

        patch = model(param_vec).clamp(0.0, 1.0)

        center_x, center_y, center_depth = \
            project_world_point(
                camera,
                neural_slice.center,
            )

        patch_size = (
            float(camera.image_height)
            * neural_slice.world_size
            * self.fno_radius
            / actual_radius
        )

        layer = place_patch_uniform(
            patch=patch,
            canvas_height=int(camera.image_height),
            canvas_width=int(camera.image_width),
            center_x=center_x,
            center_y=center_y,
            patch_size=patch_size,
        )

        info = {
            "phi": phi,
            "theta": theta,
            "actual_radius": actual_radius,
            "center_depth": center_depth,
            "center_x": center_x,
            "center_y": center_y,
            "patch_size": patch_size,
            "relative": relative,
        }

        return layer.clamp(0.0, 1.0), info
        
    def render_slice_list(
        self,
        camera,
        slices,
        shared_sh,
    ):
        """
        Render an explicit list of slices without mutating the live
        NeuralScene ModuleList.
    
        `slices` is an ordinary Python list or ModuleList.
        """
        if len(slices) == 0:
            raise RuntimeError("No slices to render.")
    
        rendered = []
    
        for slice_index, neural_slice in enumerate(slices):
            layer, info = self.render_slice(
                camera=camera,
                neural_slice=neural_slice,
                shared_sh=shared_sh,
            )
    
            rendered.append(
                {
                    "slice_index": slice_index,
                    "layer": layer,
                    "info": info,
                    "depth": float(
                        info["actual_radius"]
                        .detach()
                        .cpu()
                        .item()
                    ),
                }
            )
    
        # Far -> near. Near slices are composited last.
        rendered.sort(
            key=lambda x: x["depth"],
            reverse=True,
        )
    
        composite = torch.zeros_like(
            rendered[0]["layer"]
        )
    
        diagnostics = []
    
        for item in rendered:
            composite = self.alpha_over(
                back=composite,
                front=item["layer"],
            )
    
            info = dict(item["info"])
            info["slice_index"] = item["slice_index"]
            info["sort_depth"] = item["depth"]
            diagnostics.append(info)
    
        return composite.clamp(0.0, 1.0), diagnostics

    def forward(self, camera, neural_scene):
        return self.render_slice_list(
            camera=camera,
            slices=neural_scene.slices,
            shared_sh=neural_scene.global_sh,
        )


# ============================================================
# LOSSES / REGULARIZATION
# ============================================================

def visible_rgb_from_rgba(rgba):
    bg = torch.tensor(
        BACKGROUND_RGB,
        dtype=rgba.dtype,
        device=rgba.device,
    ).view(1, 3, 1, 1)

    return rgba[:, :3] + (
        1.0 - rgba[:, 3:4]
    ) * bg


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
                s.world_size - s.initial_world_size
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
            values["opacity"] - s.initial_opacity
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

    centers = torch.stack([
        s.center.detach()
        for s in neural_scene.slices
    ], dim=0)

    distances = torch.cdist(
        centers,
        centers,
    )

    distances.fill_diagonal_(float("inf"))

    pairs = set()

    for i in range(len(neural_scene.slices)):
        neighbors = torch.topk(
            distances[i],
            k=min(k, len(neural_scene.slices) - 1),
            largest=False,
        ).indices.tolist()

        for j in neighbors:
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
# OPTIMIZER
# ============================================================

def make_optimizer(
    neural_scene,
    base_lr,
    optimize_sh,
):
    placement_params = []
    shape_params = []
    material_params = []
    lighting_params = []

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
            lighting_params.append(
                s.raw_local_sh_delta
            )

    if optimize_sh:
        lighting_params.append(
            neural_scene.lighting.raw_global_sh_delta
        )

    groups = [
        {
            "params": placement_params,
            "lr": base_lr,
        },
        {
            "params": shape_params,
            "lr": base_lr * 2.0,
        },
        {
            "params": material_params,
            "lr": base_lr * 3.0,
        },
    ]

    if lighting_params:
        groups.append(
            {
                "params": lighting_params,
                "lr": base_lr * 0.001,
            }
        )

    return torch.optim.Adam(groups)


# ============================================================
# INITIAL SURFACE/VOLUME CANDIDATE SELECTION
# ============================================================

@torch.no_grad()
def evaluate_scene_loss(
    renderer,
    neural_scene,
    cameras,
):
    total_loss = 0.0

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

        total_loss += image_loss(
            predicted_rgba,
            target,
        ).item()

    return total_loss / max(len(cameras), 1)


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
    """
    Greedy candidate selection:
      selected existing slices + current candidate
      → evaluate surface candidate vs volume candidate.
    """
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
        f"  candidate: "
        f"surface={surface_score:.6f}, "
        f"volume={volume_score:.6f}, "
        f"chosen={chosen_mode}"
    )

    return chosen
    
@torch.no_grad()
def set_slice_mode(neural_slice, mode):
    """
    Return a copied slice configured as surface or volume.
    """
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
    """
    Score a proposed child-mode pair without mutating the real scene.
    """
    candidate_a = set_slice_mode(
        child_a,
        mode_a,
    )

    candidate_b = set_slice_mode(
        child_b,
        mode_b,
    )

    # Keep every old slice except the parent, then insert children.
    temporary_slices = (
        list(neural_scene.slices[:parent_index])
        + [candidate_a, candidate_b]
        + list(neural_scene.slices[parent_index + 1:])
    )

    total_loss = 0.0

    for camera in cameras:
        predicted_rgba, _ = renderer.render_slice_list(
            camera=camera,
            slices=temporary_slices,
            shared_sh=neural_scene.global_sh,
        )

        target_rgb = camera.original_image

        if target_rgb.ndim == 3:
            target_rgb = target_rgb.unsqueeze(0)

        target_rgb = target_rgb.to(
            device=predicted_rgba.device,
            dtype=predicted_rgba.dtype,
        )

        total_loss += image_loss(
            predicted_rgba,
            target_rgb
        ).item()

    return (
        total_loss / max(len(cameras), 1),
        candidate_a,
        candidate_b,
    )
    return score, candidate_a, candidate_b


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
        score, candidate_a, candidate_b = score_child_pair_modes(
            renderer=renderer,
            neural_scene=neural_scene,
            parent_index=parent_index,
            child_a=child_a,
            child_b=child_b,
            mode_a=mode_a,
            mode_b=mode_b,
            cameras=cameras,
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
        f"  selected child modes for parent={parent_index}: "
        f"{best_modes}, loss={best_score:.6f}"
    )

    return best_children


# ============================================================
# SAVING
# ============================================================

def save_rgb_chw(rgb_chw, path):
    if torch.is_tensor(rgb_chw):
        rgb_chw = rgb_chw.detach().cpu().numpy()

    rgb_hwc = np.transpose(rgb_chw, (1, 2, 0))
    rgb_hwc = np.clip(rgb_hwc, 0.0, 1.0)

    imageio.imwrite(
        path,
        (rgb_hwc * 255.0 + 0.5).astype(np.uint8),
    )


def save_rgba_chw(rgba_chw, path):
    if torch.is_tensor(rgba_chw):
        rgba_chw = rgba_chw.detach().cpu().numpy()

    rgba_hwc = np.transpose(rgba_chw, (1, 2, 0))
    rgba_hwc = np.clip(rgba_hwc, 0.0, 1.0)

    imageio.imwrite(
        path,
        (rgba_hwc * 255.0 + 0.5).astype(np.uint8),
    )


def save_alpha_hw(alpha_hw, path):
    if torch.is_tensor(alpha_hw):
        alpha_hw = alpha_hw.detach().cpu().numpy()

    alpha_hw = np.clip(alpha_hw, 0.0, 1.0)

    alpha_rgb = np.repeat(
        alpha_hw[..., None],
        3,
        axis=2,
    )

    imageio.imwrite(
        path,
        (alpha_rgb * 255.0 + 0.5).astype(np.uint8),
    )


def save_comparison(
    predicted_rgba,
    target_rgb,
    path,
):
    visible_rgb = visible_rgb_from_rgba(
        predicted_rgba
    )[0]

    if target_rgb.ndim == 4:
        target_rgb = target_rgb[0]

    pred_np = visible_rgb.detach().cpu().numpy()
    target_np = target_rgb.detach().cpu().numpy()

    pred_hwc = np.transpose(pred_np, (1, 2, 0))
    target_hwc = np.transpose(target_np, (1, 2, 0))

    comparison = np.concatenate(
        [
            np.clip(target_hwc, 0.0, 1.0),
            np.clip(pred_hwc, 0.0, 1.0),
        ],
        axis=1,
    )

    imageio.imwrite(
        path,
        (comparison * 255.0 + 0.5).astype(np.uint8),
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = ArgumentParser(
        description=(
            "Dataset-free neural scene optimization using "
            "random training-range slice priors."
        )
    )

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
        default=DEFAULT_ITERATIONS,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--initial_num_slices",
        type=int,
        default=8,
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
        "--mode_selection_cameras",
        type=int,
        default=3,
        help=(
            "Small camera subset used for initial "
            "surface/volume candidate selection."
        ),
    )

    parser.add_argument(
        "--disable_initial_mode_selection",
        action="store_true",
    )

    parser.add_argument(
        "--optimize_sh",
        action="store_true",
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
        default=32,
    )

    parser.add_argument(
        "--prune_interval",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=500,
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

    # --------------------------------------------------------
    # Load FNO models and their normalization only.
    # --------------------------------------------------------
    surface_model, surface_mean, surface_std, surface_dim = \
        load_fno_checkpoint(
            args.surface_checkpoint,
            device,
        )

    volume_model, volume_mean, volume_std, volume_dim = \
        load_fno_checkpoint(
            args.volume_checkpoint,
            device,
        )

    if surface_dim != 44 or volume_dim != 44:
        raise RuntimeError(
            f"Expected latent_dim=44; got "
            f"surface={surface_dim}, volume={volume_dim}"
        )

    # --------------------------------------------------------
    # Build valid environment SH bank and select one global env.
    # --------------------------------------------------------
    sh_bank = build_sh_bank(device)

    initial_env_id = int(
        rng.integers(0, NUM_GLOBAL_ENVS)
    )

    initial_global_sh = sh_bank[
        initial_env_id
    ].detach().clone()

    print(
        "Initial environment ID:",
        initial_env_id,
    )

    # --------------------------------------------------------
    # Load 3DGS scene.
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

    if not cameras_used:
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

    # --------------------------------------------------------
    # Renderer must exist before candidate selection.
    # --------------------------------------------------------
    renderer = NeuralSceneRenderer(
        surface_model=surface_model,
        volume_model=volume_model,
        surface_mean=surface_mean,
        surface_std=surface_std,
        volume_mean=volume_mean,
        volume_std=volume_std,
        fno_radius=FNO_RADIUS,
    ).to(device)

    # --------------------------------------------------------
    # Voxel seeds.
    # --------------------------------------------------------
    seeds = voxel_chunk_seeds(
        xyz=xyz,
        voxel_size=args.voxel_size,
        min_points=args.min_points_per_voxel,
        max_chunks=args.initial_num_slices,
    )

    if not seeds:
        raise RuntimeError(
            "No valid voxel seeds. Try larger voxel size "
            "or lower min_points_per_voxel."
        )

    print(
        f"Voxel initialization created {len(seeds)} seeds."
    )

    # --------------------------------------------------------
    # Initial surface/volume mode selection.
    # --------------------------------------------------------
    selection_cameras = cameras_used[
        :min(
            args.mode_selection_cameras,
            len(cameras_used),
        )
    ]
    
    # Keep candidate selection cheap.
    # Two or three cameras is enough for an initial decision.
    child_selection_cameras = cameras_used[
        :min(3, len(cameras_used))
    ]
    
    def split_mode_selector(parent_index, child_a, child_b):
        chosen_a, chosen_b = choose_split_child_modes(
            renderer=renderer,
            neural_scene=neural_scene,
            parent_index=parent_index,
            child_a=child_a,
            child_b=child_b,
            cameras=child_selection_cameras,
        )
    
        return chosen_a, chosen_b

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
            chosen = base_candidate
            chosen_mode = "surface"
        else:
            chosen = choose_initial_slice_mode(
                renderer=renderer,
                selected_slices=slices,
                base_candidate=base_candidate,
                initial_global_sh=initial_global_sh,
                optimize_sh=args.optimize_sh,
                cameras=selection_cameras,
                device=device,
            )

            chosen_mode = chosen.mode

        slices.append(chosen)

        print(
            f"seed={i:02d} "
            f"points={seed['num_points']} "
            f"center={seed['center'].detach().cpu().numpy()} "
            f"size={seed['world_size']:.3f} "
            f"mode={chosen_mode}"
        )

    neural_scene = NeuralScene(
        slices=slices,
        initial_global_sh=initial_global_sh,
        optimize_sh=args.optimize_sh,
    ).to(device)

    optimizer = make_optimizer(
        neural_scene,
        base_lr=args.lr,
        optimize_sh=args.optimize_sh,
    )

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
    # Optimization loop.
    # --------------------------------------------------------
    for iteration in range(args.iterations):
        optimizer.zero_grad(
            set_to_none=True
        )

        total_image_loss = 0.0

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
            ) / len(cameras_used)

            camera_loss.backward()

            total_image_loss += (
                camera_loss.detach().item()
                * len(cameras_used)
            )

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

        loss_regularization.backward()

        accumulate_gradient_stats(
            neural_scene,
            gradient_stats,
        )

        torch.nn.utils.clip_grad_norm_(
            neural_scene.parameters(),
            max_norm=MAX_GRAD_NORM,
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
                prune_indices = choose_prune_indices(
                    slice_stats=slice_stats,
                    min_max_alpha=0.01,
                    min_alpha_mass=5.0,
                    min_visible_views=1,
                    min_remaining_slices=1,
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

                    # Skip split after prune for this first version.
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

            optimizer = make_optimizer(
                neural_scene,
                base_lr=args.lr,
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
        # Checkpoint.
        # ----------------------------------------------------
        if (
            args.checkpoint_interval > 0
            and iteration > 0
            and iteration % args.checkpoint_interval == 0
        ):
            path = output_dir / (
                f"checkpoint_iter_{iteration:06d}.pt"
            )

            save_neural_scene_checkpoint(
                path=path,
                neural_scene=neural_scene,
                optimizer=optimizer,
                iteration=iteration,
                metadata={
                    "num_slices": len(neural_scene.slices),
                    "initial_env_id": initial_env_id,
                    "optimize_sh": args.optimize_sh,
                },
            )

            print("Saved checkpoint:", path)

        # ----------------------------------------------------
        # Logging.
        # ----------------------------------------------------
        if (
            iteration % 25 == 0
            or iteration == args.iterations - 1
        ):
            image_value = (
                total_image_loss / len(cameras_used)
            )

            total_value = (
                image_value
                + loss_regularization.detach().item()
            )

            print(
                f"iter={iteration:05d} "
                f"loss={total_value:.8f} "
                f"image={image_value:.8f} "
                f"reg={loss_regularization.item():.8f}"
            )

            print(
                "  global SH raw delta norm:",
                neural_scene.lighting.raw_global_sh_delta
                .detach()
                .norm()
                .item(),
            )

            for i, neural_slice in enumerate(
                neural_scene.slices
            ):
                values = neural_slice.fno_values(
                    shared_sh=neural_scene.global_sh
                )

                print(
                    f"  slice={i:02d} "
                    f"mode={neural_slice.mode} "
                    f"size={neural_slice.world_size.item():.4f} "
                    f"sigma={values['sigma'].item():.4f} "
                    f"opacity={values['opacity'].item():.4f} "
                    f"hue={values['hue'].item():.4f}"
                )

    # --------------------------------------------------------
    # Final image outputs.
    # --------------------------------------------------------
    first_camera = cameras_used[0]

    with torch.no_grad():
        final_rgba, _ = renderer(
            first_camera,
            neural_scene,
        )

    final_rgba = final_rgba.clamp(0.0, 1.0)

    final_np = final_rgba[0].cpu().numpy()
    target = first_camera.original_image.detach()

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
        target,
        output_dir / "target_rgb.png",
    )

    save_comparison(
        predicted_rgba=final_rgba,
        target_rgb=target,
        path=output_dir / "target_vs_final.png",
    )

    # --------------------------------------------------------
    # Save compact result metadata.
    # --------------------------------------------------------
    slice_data = []

    for i, s in enumerate(neural_scene.slices):
        values = s.fno_values(
            shared_sh=neural_scene.global_sh
        )

        slice_data.append(
            {
                "index": i,
                "mode": s.mode,
                "center": s.center.detach()
                .cpu()
                .numpy()
                .tolist(),
                "world_size": s.world_size.item(),
                "ctrl": values["ctrl"].detach()
                .cpu()
                .numpy()
                .tolist(),
                "sigma": values["sigma"].item(),
                "hue": values["hue"].item(),
                "saturation": values["saturation"].item(),
                "opacity": values["opacity"].item(),
                "roughness": values["roughness"].item(),
                "sh": values["sh"].detach()
                .cpu()
                .numpy()
                .tolist(),
            }
        )

    torch.save(
        {
            "slices": slice_data,
            "global_sh": neural_scene.global_sh.detach()
            .cpu()
            .numpy(),
            "initial_environment_id": initial_env_id,
            "fno_radius": FNO_RADIUS,
            "optimize_sh": args.optimize_sh,
        },
        output_dir / "optimized_neural_slices.pt",
    )

    print("Done.")
    print("Outputs:", output_dir)


if __name__ == "__main__":
    main()