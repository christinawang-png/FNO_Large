#!/usr/bin/env python

import sys
import math
import random
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
    camera_to_slice_pose,
    project_world_point,
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
    REPO_DIR / "four_slice_real_scene_optimization_outputs"
)

IMG_SIZE = (64, 64)
FNO_RADIUS = 2.2

OPTIMIZATION_STEPS = 1000
LEARNING_RATE = 1e-2

# Regularization prevents slices from immediately shrinking away
# because the real scene contains a large background.
POSITION_REG_WEIGHT = 1e-5
SIZE_REG_WEIGHT = 1e-3

# Default four slice configuration.
#
# The first slice is treated as the back layer and the last as
# the front layer.
SLICE_MODES = [
    "surface",
    "volume",
    "surface",
    "volume",
]

SLICE_TEMPLATE_INDICES = [
    0,
    0,
    1,
    1,
]

# Offsets from the point-cloud median center.
INITIAL_SLICE_OFFSETS = [
    [0.00, 0.00, 0.00],
    [0.35, 0.00, 0.25],
    [-0.35, 0.10, 0.15],
    [0.00, -0.35, 0.40],
]

INITIAL_SLICE_SIZES = [
    1.00,
    0.80,
    0.75,
    0.65,
]


# ============================================================
# CHECKPOINT HELPERS
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

def make_parameter_builder(dataset):
    """
    Build the callback used by NeuralSceneRenderer.

    The current FNO parameter construction is metadata-based.
    Therefore, shape/material/environment parameters come from
    a fixed template row. Camera phi/theta are replaced per view.
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

        # The current dataset builder is NumPy/metadata based,
        # so camera values are converted to Python floats.
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

        normalized_np = (
            raw_np - param_mean
        ) / param_std

        return torch.from_numpy(
            normalized_np.astype(np.float32)
        ).unsqueeze(0).to(device)

    return build_fno_param_vector


# ============================================================
# LOSSES
# ============================================================

def image_loss(predicted_rgba, target_rgb):
    """
    Loss against a real RGB target.

    predicted_rgba:
        [1,4,H,W]

    target_rgb:
        [1,3,H,W]

    The FNO color channels are premultiplied, so the RGB part
    can be compared directly when using a black background.
    """
    predicted_rgb = predicted_rgba[:, :3]

    loss_l1 = F.l1_loss(
        predicted_rgb,
        target_rgb,
    )

    loss_mse = F.mse_loss(
        predicted_rgb,
        target_rgb,
    )

    return 0.5 * loss_l1 + 0.5 * loss_mse


def parameter_regularization(
    neural_scene,
    initial_centers,
    initial_sizes,
):
    """
    Weak regularization against disappearing or drifting slices.
    """
    loss = torch.zeros(
        (),
        dtype=torch.float32,
        device=initial_centers[0].device,
    )

    for index, neural_slice in enumerate(
        neural_scene.slices
    ):
        center = neural_slice.center
        size = neural_slice.world_size

        center0 = initial_centers[index]
        size0 = initial_sizes[index]

        center_loss = F.mse_loss(
            center,
            center0,
        )

        size_loss = (
            (size - size0) / size0
        ) ** 2

        loss = loss + (
            POSITION_REG_WEIGHT * center_loss
            + SIZE_REG_WEIGHT * size_loss
        )

    return loss


# ============================================================
# SAVING
# ============================================================

def save_chw_rgb(rgb_chw, path):
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
            "Optimize four neural FNO slices against real "
            "3DGS scene camera images."
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
        help=(
            "Number of cameras used for optimization. "
            "One camera is sampled per iteration."
        ),
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=OPTIMIZATION_STEPS,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
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
    print("Output directory:", output_dir)

    # --------------------------------------------------------
    # Load FNO metadata.
    # --------------------------------------------------------
    fno_dataset = PlaneDatasetParamsToPremultRGBA(
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

    print("FNO rows:", len(fno_dataset.df))
    print(fno_dataset.df["render_mode"].value_counts())

    # --------------------------------------------------------
    # Load surface and volume FNOs.
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

    if surface_dim != fno_dataset.latent_dim:
        raise RuntimeError(
            f"FNO checkpoint dimension {surface_dim} does not "
            f"match dataset dimension {fno_dataset.latent_dim}."
        )

    # --------------------------------------------------------
    # Select template rows.
    # --------------------------------------------------------
    surface_rows = fno_dataset.df[
        fno_dataset.df["render_mode"].astype(str) == "surface"
    ].reset_index(drop=True)

    volume_rows = fno_dataset.df[
        fno_dataset.df["render_mode"].astype(str) == "volume"
    ].reset_index(drop=True)

    if len(surface_rows) == 0:
        raise RuntimeError("No surface rows found.")

    if len(volume_rows) == 0:
        raise RuntimeError("No volume rows found.")

    template_rows = []

    for mode, index in zip(
        SLICE_MODES,
        SLICE_TEMPLATE_INDICES,
    ):
        rows = (
            surface_rows
            if mode == "surface"
            else volume_rows
        )

        if index < 0 or index >= len(rows):
            raise IndexError(
                f"Template index {index} invalid for mode {mode}."
            )

        template_rows.append(
            rows.iloc[index]
        )

    print(
        "Template sample IDs:",
        [
            int(row["sample_id"])
            for row in template_rows
        ],
    )

    # --------------------------------------------------------
    # Load 3DGS scene for cameras and point cloud center.
    # --------------------------------------------------------
    scene_args = lp.extract(args)

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
        resolution_scales=[float(args.resolution)],
    )

    cameras = scene.getTrainCameras(
        scale=float(args.resolution),
    )

    if len(cameras) == 0:
        raise RuntimeError(
            "No training cameras found."
        )

    if args.camera_start < 0 or \
       args.camera_start >= len(cameras):
        raise IndexError(
            f"camera_start={args.camera_start} is invalid. "
            f"Number of cameras: {len(cameras)}."
        )

    camera_end = min(
        args.camera_start + args.num_cameras,
        len(cameras),
    )

    optimization_cameras = cameras[
        args.camera_start:camera_end
    ]

    print(
        "Using cameras:",
        args.camera_start,
        "to",
        camera_end - 1,
    )

    # --------------------------------------------------------
    # Scene center from point cloud.
    # --------------------------------------------------------
    scene_center = gaussian_model.get_xyz.detach().median(
        dim=0
    ).values.to(device)

    print(
        "Scene center:",
        scene_center.detach().cpu().numpy(),
    )

    # --------------------------------------------------------
    # Build neural renderer.
    # --------------------------------------------------------
    build_param_vector = make_parameter_builder(
        fno_dataset
    )

    renderer = NeuralSceneRenderer(
        surface_model=surface_model,
        volume_model=volume_model,
        dataset=fno_dataset,
        build_param_vector=build_param_vector,
        fno_radius=FNO_RADIUS,
        canonical_patch_size=64,
    ).to(device)

    # --------------------------------------------------------
    # Create four learnable slices.
    #
    # The slices are ordered back-to-front.
    # --------------------------------------------------------
    initial_centers = []
    initial_sizes = []
    neural_slices = []

    for mode, offset, size, template_row, mean, std in zip(
        SLICE_MODES,
        INITIAL_SLICE_OFFSETS,
        INITIAL_SLICE_SIZES,
        template_rows,
        [
            surface_mean,
            volume_mean,
            surface_mean,
            volume_mean,
        ],
        [
            surface_std,
            volume_std,
            surface_std,
            volume_std,
        ],
    ):
        offset_tensor = torch.tensor(
            offset,
            dtype=torch.float32,
            device=device,
        )

        center = scene_center + offset_tensor

        neural_slice = NeuralSlice(
            mode=mode,
            center=center,
            world_size=size,
            template_row=template_row,
            param_mean=mean,
            param_std=std,
        )

        neural_slices.append(neural_slice)
        initial_centers.append(center.detach().clone())
        initial_sizes.append(
            torch.tensor(
                size,
                dtype=torch.float32,
                device=device,
            )
        )

    neural_scene = NeuralScene(
        slices=neural_slices
    ).to(device)

    # --------------------------------------------------------
    # Initial preview on the first camera.
    # --------------------------------------------------------
    first_camera = optimization_cameras[0]

    with torch.no_grad():
        initial_rgba, _ = renderer(
            first_camera,
            neural_scene,
        )

    initial_rgba = initial_rgba.detach()

    # --------------------------------------------------------
    # Optimizer.
    #
    # Only NeuralSlice parameters are included.
    # FNO weights remain frozen.
    # --------------------------------------------------------
    optimizer = torch.optim.Adam(
        neural_scene.parameters(),
        lr=args.lr,
    )

    print("\nStarting real-scene optimization...")
    print(
        "Optimized parameters per slice:",
        "center[3] + uniform world_size",
    )

    # --------------------------------------------------------
    # Optimization loop.
    # --------------------------------------------------------
    for iteration in range(args.iterations):
        viewpoint_camera = random.choice(
            optimization_cameras
        )

        predicted_rgba, diagnostics = renderer(
            viewpoint_camera,
            neural_scene,
        )

        target_rgb = viewpoint_camera.original_image.detach()

        # Ensure target has a batch dimension.
        if target_rgb.ndim == 3:
            target_rgb = target_rgb.unsqueeze(0)

        target_rgb = target_rgb.to(
            device=device,
            dtype=predicted_rgba.dtype,
        )

        predicted_h, predicted_w = predicted_rgba.shape[-2:]
        target_h, target_w = target_rgb.shape[-2:]

        if (
            predicted_h != target_h
            or predicted_w != target_w
        ):
            raise RuntimeError(
                "Prediction and target dimensions differ: "
                f"prediction={predicted_rgba.shape}, "
                f"target={target_rgb.shape}"
            )

        loss_image = image_loss(
            predicted_rgba,
            target_rgb,
        )

        loss_reg = parameter_regularization(
            neural_scene,
            initial_centers,
            initial_sizes,
        )

        loss = loss_image + loss_reg

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()
        optimizer.step()

        if iteration % 25 == 0 or \
           iteration == args.iterations - 1:

            print(
                f"iter={iteration:05d} "
                f"loss={loss.item():.8f} "
                f"image={loss_image.item():.8f} "
                f"reg={loss_reg.item():.8f}"
            )

            for slice_index, neural_slice in enumerate(
                neural_scene.slices
            ):
                center_np = (
                    neural_slice.center
                    .detach()
                    .cpu()
                    .numpy()
                )

                size_value = (
                    neural_slice.world_size
                    .detach()
                    .cpu()
                    .item()
                )

                print(
                    f"  slice={slice_index} "
                    f"mode={neural_slice.mode} "
                    f"center={center_np} "
                    f"size={size_value:.5f}"
                )

    # --------------------------------------------------------
    # Final preview.
    # --------------------------------------------------------
    with torch.no_grad():
        final_rgba, _ = renderer(
            first_camera,
            neural_scene,
        )

    final_rgba = final_rgba.detach().clamp(
        0.0,
        1.0,
    )

    initial_np = initial_rgba[0].cpu().numpy()
    final_np = final_rgba[0].cpu().numpy()

    # Save initial/final RGB and alpha.
    save_rgba_chw(
        initial_np[:3],
        output_dir / "initial_rgb.png",
    )

    save_alpha_hw(
        initial_np[3],
        output_dir / "initial_alpha.png",
    )

    save_rgba_chw(
        final_np[:3],
        output_dir / "final_rgb.png",
    )

    save_alpha_hw(
        final_np[3],
        output_dir / "final_alpha.png",
    )

    save_rgba_chw(
        final_np,
        output_dir / "final_rgba.png",
    )

    # Save target from first camera.
    target = first_camera.original_image.detach()

    save_rgba_chw(
        target,
        output_dir / "target_rgb.png",
    )

    # Save side-by-side target and final output.
    target_np = target.cpu().numpy()
    target_hwc = np.transpose(
        target_np,
        (1, 2, 0),
    )

    final_hwc = np.transpose(
        final_np[:3],
        (1, 2, 0),
    )

    target_hwc = np.clip(
        target_hwc,
        0.0,
        1.0,
    )

    final_hwc = np.clip(
        final_hwc,
        0.0,
        1.0,
    )

    if target_hwc.shape == final_hwc.shape:
        comparison = np.concatenate(
            [
                target_hwc,
                final_hwc,
            ],
            axis=1,
        )

        imageio.imwrite(
            output_dir / "target_vs_final.png",
            (comparison * 255.0 + 0.5).astype(np.uint8),
        )

    # --------------------------------------------------------
    # Save optimized parameters.
    # --------------------------------------------------------
    optimized_parameters = []

    for index, neural_slice in enumerate(
        neural_scene.slices
    ):
        optimized_parameters.append(
            {
                "slice_index": index,
                "mode": neural_slice.mode,
                "center": neural_slice.center.detach()
                .cpu()
                .numpy()
                .tolist(),
                "world_size": neural_slice.world_size.detach()
                .cpu()
                .item(),
                "template_sample_id": int(
                    template_rows[index]["sample_id"]
                ),
            }
        )

    torch.save(
        {
            "optimized_slices": optimized_parameters,
            "scene_center": scene_center.detach()
            .cpu()
            .numpy(),
            "fno_radius": FNO_RADIUS,
            "camera_index": args.camera_start,
        },
        output_dir / "optimized_slices.pt",
    )

    print("\nSaved outputs to:", output_dir)
    print(
        "Saved optimized slice parameters to:",
        output_dir / "optimized_slices.pt",
    )


if __name__ == "__main__":
    main()