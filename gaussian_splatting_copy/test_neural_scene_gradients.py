#!/usr/bin/env python

import sys
import math
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

from train_premult_single_mode import (
    PlaneDatasetParamsToPremultRGBA,
    FNOPlusResNetSingle,
)

from neural_scene import (
    NeuralSlice,
    NeuralScene,
    NeuralSceneRenderer,
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

SURFACE_CHECKPOINT = (
    FNO_ROOT / "fno_premult_surface_final.pt"
)

VOLUME_CHECKPOINT = (
    FNO_ROOT / "fno_premult_volume_epoch025.pt"
)

DEFAULT_OUTPUT_DIR = (
    REPO_DIR / "neural_scene_gradient_test_outputs"
)

IMG_SIZE = (64, 64)
FNO_RADIUS = 2.2

OPTIMIZATION_STEPS = 300
LEARNING_RATE = 1e-2


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

    state = dict(
        checkpoint.get(
            "model_state",
            checkpoint,
        )
    )

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

    print(
        "  latent_dim:",
        latent_dim,
        "mode:",
        checkpoint.get("mode", "unknown"),
    )

    return model, param_mean, param_std, latent_dim


# ============================================================
# FNO PARAMETER BUILDER
# ============================================================

def make_parameter_builder(dataset, device):
    """
    Build a callback compatible with NeuralSceneRenderer.

    Shape, material, and environment values come from the
    selected metadata template row.

    Camera phi/theta/radius are replaced per view.
    """

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
        row = template_row.copy()

        # These values are intentionally converted to floats
        # because the current training dataset builder is NumPy/
        # metadata based.
        row["phi"] = float(
            phi.detach().cpu().item()
        )

        row["theta"] = float(
            theta.detach().cpu().item()
        )

        row["radius"] = float(fno_radius)
        row["render_mode"] = mode

        if mode == "volume":
            row["metallic"] = 0.0
            row["roughness"] = 0.0
            row["specular"] = 0.0

        raw_np = dataset._build_param_vector_np(row)

        raw_np = np.asarray(
            raw_np,
            dtype=np.float32,
        )

        param_mean = np.asarray(
            param_mean,
            dtype=np.float32,
        )

        param_std = np.asarray(
            param_std,
            dtype=np.float32,
        )

        if raw_np.shape[0] != param_mean.shape[0]:
            raise RuntimeError(
                f"{mode} parameter dimension mismatch: "
                f"raw={raw_np.shape[0]}, "
                f"mean={param_mean.shape[0]}"
            )

        normalized_np = (
            raw_np - param_mean
        ) / param_std

        return torch.from_numpy(
            normalized_np.astype(np.float32)
        ).unsqueeze(0).to(device)

    return build_fno_param_vector


# ============================================================
# IMAGE SAVING
# ============================================================

def save_rgb_chw(rgb_chw, path):
    if torch.is_tensor(rgb_chw):
        rgb_chw = rgb_chw.detach().cpu().numpy()

    rgb_hwc = np.transpose(
        rgb_chw,
        (1, 2, 0),
    )

    rgb_hwc = np.clip(
        rgb_hwc,
        0.0,
        1.0,
    )

    imageio.imwrite(
        path,
        (rgb_hwc * 255.0 + 0.5).astype(np.uint8),
    )


def save_alpha_hw(alpha_hw, path):
    if torch.is_tensor(alpha_hw):
        alpha_hw = alpha_hw.detach().cpu().numpy()

    alpha_hw = np.clip(
        alpha_hw,
        0.0,
        1.0,
    )

    alpha_rgb = np.repeat(
        alpha_hw[..., None],
        3,
        axis=2,
    )

    imageio.imwrite(
        path,
        (alpha_rgb * 255.0 + 0.5).astype(np.uint8),
    )


def save_rgba_chw(rgba_chw, path):
    if torch.is_tensor(rgba_chw):
        rgba_chw = rgba_chw.detach().cpu().numpy()

    rgba_hwc = np.transpose(
        rgba_chw,
        (1, 2, 0),
    )

    rgba_hwc = np.clip(
        rgba_hwc,
        0.0,
        1.0,
    )

    imageio.imwrite(
        path,
        (rgba_hwc * 255.0 + 0.5).astype(np.uint8),
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = ArgumentParser(
        description=(
            "Test gradients through a two-slice neural scene."
        )
    )

    # Provides --source_path, --model_path, --resolution, etc.
    lp = ModelParams(parser)

    parser.add_argument(
        "--camera_index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--surface_template",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--volume_template",
        type=int,
        default=0,
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
        raise RuntimeError(
            "CUDA is required by the current 3DGS setup."
        )

    device = torch.device("cuda")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Using device:", device)

    # --------------------------------------------------------
    # Load FNO metadata.
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

    print("FNO rows:", len(dataset.df))
    print(dataset.df["render_mode"].value_counts())

    # --------------------------------------------------------
    # Load FNO models.
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

    if surface_dim != volume_dim:
        raise RuntimeError(
            "Surface and volume latent dimensions differ."
        )

    if surface_dim != dataset.latent_dim:
        raise RuntimeError(
            f"Checkpoint latent_dim={surface_dim}, "
            f"dataset latent_dim={dataset.latent_dim}"
        )

    # --------------------------------------------------------
    # Select fixed metadata templates.
    # --------------------------------------------------------
    surface_rows = dataset.df[
        dataset.df["render_mode"].astype(str) == "surface"
    ].reset_index(drop=True)

    volume_rows = dataset.df[
        dataset.df["render_mode"].astype(str) == "volume"
    ].reset_index(drop=True)

    if not 0 <= args.surface_template < len(surface_rows):
        raise IndexError(
            f"Invalid surface template index: "
            f"{args.surface_template}"
        )

    if not 0 <= args.volume_template < len(volume_rows):
        raise IndexError(
            f"Invalid volume template index: "
            f"{args.volume_template}"
        )

    surface_template = surface_rows.iloc[
        args.surface_template
    ]

    volume_template = volume_rows.iloc[
        args.volume_template
    ]

    print(
        "Surface template sample:",
        int(surface_template["sample_id"]),
    )

    print(
        "Volume template sample:",
        int(volume_template["sample_id"]),
    )

    # --------------------------------------------------------
    # Load one 3DGS scene camera.
    #
    # GaussianModel is only needed because the existing Scene
    # class requires it. The Gaussian renderer is not used.
    # --------------------------------------------------------
    scene_args = lp.extract(args)

    if not scene_args.model_path:
        scene_args.model_path = str(
            REPO_DIR / "output" / "gradient_scene_test"
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
        resolution_scales=[1.0],
    )

    cameras = scene.getTrainCameras(
        scale=1.0,
    )

    if len(cameras) == 0:
        raise RuntimeError(
            "No training cameras were loaded."
        )

    if not 0 <= args.camera_index < len(cameras):
        raise IndexError(
            f"camera_index={args.camera_index} is invalid. "
            f"Available cameras: {len(cameras)}"
        )

    viewpoint_camera = cameras[
        args.camera_index
    ]

    # --------------------------------------------------------
    # Compute scene center from point cloud.
    # --------------------------------------------------------
    xyz = gaussian_model.get_xyz.detach()

    scene_center = xyz.median(
        dim=0
    ).values.to(device)

    print(
        "Scene center:",
        scene_center.detach().cpu().numpy(),
    )

    # --------------------------------------------------------
    # Create parameter builder and neural renderer.
    # --------------------------------------------------------
    build_param_vector = make_parameter_builder(
        dataset,
        device,
    )

    renderer = NeuralSceneRenderer(
        surface_model=surface_model,
        volume_model=volume_model,
        dataset=dataset,
        build_param_vector=build_param_vector,
        fno_radius=FNO_RADIUS,
        canonical_patch_size=64,
    ).to(device)

    # --------------------------------------------------------
    # Create two neural slices.
    #
    # Stored order:
    #   slice 1 = back
    #   slice 2 = front
    # --------------------------------------------------------
    surface_slice = NeuralSlice(
        mode="surface",
        center=scene_center + torch.tensor(
            [0.0, 0.0, 0.0],
            dtype=torch.float32,
            device=device,
        ),
        world_size=1.0,
        template_row=surface_template,
        param_mean=surface_mean,
        param_std=surface_std,
    )
    
    volume_slice = NeuralSlice(
        mode="volume",
        center=scene_center + torch.tensor(
            [0.35, 0.0, 0.25],
            dtype=torch.float32,
            device=device,
        ),
        world_size=0.8,
        template_row=volume_template,
        param_mean=volume_mean,
        param_std=volume_std,
    )

    neural_scene = NeuralScene(
        slices=[
            surface_slice,
            volume_slice,
        ]
    ).to(device)

    # --------------------------------------------------------
    # Initial forward pass.
    # --------------------------------------------------------
    initial_composite, initial_info = renderer(
        viewpoint_camera,
        neural_scene,
    )

    if initial_composite.ndim != 4:
        raise RuntimeError(
            f"Expected composite [1,4,H,W], "
            f"got {initial_composite.shape}"
        )

    print(
        "Initial composite shape:",
        tuple(initial_composite.shape),
    )

    # --------------------------------------------------------
    # Check gradients directly.
    #
    # A sum loss is only a smoke test. It verifies that the
    # differentiable placement path reaches slice parameters.
    # --------------------------------------------------------
    gradient_test_loss = initial_composite.mean()

    gradient_test_loss.backward()

    print("\nGradient check:")

    for index, neural_slice in enumerate(
        neural_scene.slices
    ):
        print(
            f"slice {index} mode={neural_slice.mode}"
        )

        print(
            "  center.grad:",
            neural_slice.center.grad,
        )

        print(
            "  raw_world_size.grad:",
            neural_slice.raw_world_size.grad,
        )

        if neural_slice.center.grad is None:
            raise RuntimeError(
                f"No center gradient for slice {index}."
            )

        if neural_slice.raw_world_size.grad is None:
            raise RuntimeError(
                f"No size gradient for slice {index}."
            )

    # Clear smoke-test gradients before optimization.
    for parameter in neural_scene.parameters():
        parameter.grad = None

    # --------------------------------------------------------
    # Create a synthetic target from a known placement.
    #
    # This target is made using the same frozen FNO outputs
    # and renderer. It avoids needing Blender or a real target
    # scene for this optimization test.
    # --------------------------------------------------------
    target_surface = NeuralSlice(
        mode="surface",
        center=scene_center + torch.tensor(
            [0.10, 0.05, 0.0],
            dtype=torch.float32,
            device=device,
        ),
        world_size=1.10,
        template_row=surface_template,
        param_mean=surface_mean,
        param_std=surface_std,
    )
    
    target_volume = NeuralSlice(
        mode="volume",
        center=scene_center + torch.tensor(
            [0.45, 0.05, 0.30],
            dtype=torch.float32,
            device=device,
        ),
        world_size=0.70,
        template_row=volume_template,
        param_mean=volume_mean,
        param_std=volume_std,
    )

    target_scene = NeuralScene(
        slices=[
            target_surface,
            target_volume,
        ]
    ).to(device)

    with torch.no_grad():
        target_composite, _ = renderer(
            viewpoint_camera,
            target_scene,
        )

    target_composite = target_composite.detach()

    # --------------------------------------------------------
    # Optimize only the test scene slices.
    # The target scene is fixed and is not optimized.
    # --------------------------------------------------------
    optimizer = torch.optim.Adam(
        neural_scene.parameters(),
        lr=LEARNING_RATE,
    )

    print("\nStarting optimization...")

    for step in range(OPTIMIZATION_STEPS):
        predicted_composite, _ = renderer(
            viewpoint_camera,
            neural_scene,
        )

        loss = F.mse_loss(
            predicted_composite,
            target_composite,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()
        optimizer.step()

        if step % 50 == 0 or \
           step == OPTIMIZATION_STEPS - 1:

            print(
                f"step={step:04d} "
                f"loss={loss.item():.8f} "
                f"surface_center="
                f"{neural_scene.slices[0].center.detach().cpu().numpy()} "
                f"surface_size="
                f"{neural_scene.slices[0].world_size.item():.4f} "
                f"volume_center="
                f"{neural_scene.slices[1].center.detach().cpu().numpy()} "
                f"volume_size="
                f"{neural_scene.slices[1].world_size.item():.4f}"
            )

    # --------------------------------------------------------
    # Final forward pass.
    # --------------------------------------------------------
    with torch.no_grad():
        final_composite, _ = renderer(
            viewpoint_camera,
            neural_scene,
        )

    initial_np = initial_composite[0].detach().cpu().numpy()
    target_np = target_composite[0].detach().cpu().numpy()
    final_np = final_composite[0].detach().cpu().numpy()

    # --------------------------------------------------------
    # Save images.
    # --------------------------------------------------------
    save_rgb_chw(
        initial_np[:3],
        output_dir / "initial_rgb.png",
    )

    save_rgb_chw(
        target_np[:3],
        output_dir / "target_rgb.png",
    )

    save_rgb_chw(
        final_np[:3],
        output_dir / "final_rgb.png",
    )

    save_alpha_hw(
        initial_np[3],
        output_dir / "initial_alpha.png",
    )

    save_alpha_hw(
        target_np[3],
        output_dir / "target_alpha.png",
    )

    save_alpha_hw(
        final_np[3],
        output_dir / "final_alpha.png",
    )

    save_rgba_chw(
        final_np,
        output_dir / "final_rgba.png",
    )

    print("\nSaved outputs to:", output_dir)


if __name__ == "__main__":
    main()