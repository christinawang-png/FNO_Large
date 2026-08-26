#!/usr/bin/env python

import sys
import math
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch


# ============================================================
# PATH SETUP
# ============================================================

REPO_DIR = Path(__file__).resolve().parent
FNO_LARGE_DIR = REPO_DIR.parent

sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(FNO_LARGE_DIR))


# ============================================================
# 3DGS IMPORTS
# ============================================================

from scene import Scene, GaussianModel
from arguments import ModelParams


# ============================================================
# FNO IMPORTS
# ============================================================

from train_premult_single_mode import (
    PlaneDatasetParamsToPremultRGBA,
    FNOPlusResNetSingle,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = FNO_LARGE_DIR / "plane_dataset_4"

RENDERS_DIR = BASE_DIR / "renders"
ALPHA_DIR = BASE_DIR / "hard_alpha"

IMG_META_CSV = RENDERS_DIR / "metadata_images_all_sharded.csv"
VOL_META_CSV = BASE_DIR / "metadata_volumes.csv"
ALPHA_META_CSV = ALPHA_DIR / "metadata_alpha_all.csv"

DEFAULT_SURFACE_CHECKPOINT = (
    FNO_LARGE_DIR / "fno_premult_surface_final.pt"
)

DEFAULT_VOLUME_CHECKPOINT = (
    FNO_LARGE_DIR / "fno_premult_volume_epoch025.pt"
)

OUTPUT_DIR = REPO_DIR / "neural_renderer_scene_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (64, 64)

# This must match the radius used during FNO training.
FNO_RADIUS = 2.2


# ============================================================
# CAMERA HELPERS
# ============================================================

def camera_forward_world(viewpoint_camera):
    """
    Return the camera's world-space forward direction.

    For this 3DGS matrix convention, local +Z is the viewing
    direction. This sign was verified by your alignment test.
    """
    view_inv = torch.linalg.inv(
        viewpoint_camera.world_view_transform
    )

    forward_world = view_inv[2, :3]

    return forward_world / torch.linalg.norm(
        forward_world
    ).clamp_min(1e-8)


def camera_to_slice_pose(viewpoint_camera, slice_center):
    """
    Convert a 3DGS camera position into spherical coordinates
    relative to a neural slice center.

    Returns:
        phi
        theta
        actual_radius
        relative_camera_position
    """
    camera_center = viewpoint_camera.camera_center

    relative = camera_center - slice_center

    actual_radius = torch.linalg.norm(relative).clamp_min(1e-8)

    x = relative[0]
    y = relative[1]
    z = relative[2]

    phi = torch.acos(
        torch.clamp(z / actual_radius, -1.0, 1.0)
    )

    theta = torch.atan2(y, x)
    theta = torch.remainder(theta, 2.0 * math.pi)

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


def alignment_to_center(viewpoint_camera, target_center):
    """
    Compare the actual camera viewing direction to the direction
    from the camera toward target_center.
    """
    camera_center = viewpoint_camera.camera_center

    toward_target = target_center - camera_center
    toward_target = toward_target / torch.linalg.norm(
        toward_target
    ).clamp_min(1e-8)

    forward = camera_forward_world(viewpoint_camera)

    dot = torch.dot(forward, toward_target)
    dot = torch.clamp(dot, -1.0, 1.0)

    angle_degrees = torch.rad2deg(torch.acos(dot))

    return dot, angle_degrees


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


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

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


# ============================================================
# PREVIEW
# ============================================================

def save_prediction_preview(
    rgba,
    output_path,
    mode,
    camera_index,
    sample_id,
    actual_radius,
    phi,
    theta,
    alignment_angle,
):
    """
    Save the FNO output as a 2x2 diagnostic image.
    """
    rgba = np.asarray(rgba, dtype=np.float32)

    if rgba.shape != (4, 64, 64):
        raise RuntimeError(
            f"Expected RGBA shape (4,64,64), got {rgba.shape}"
        )

    rgba = np.clip(rgba, 0.0, 1.0)

    premult_color = rgba[:3]
    alpha = rgba[3]

    color_hw = np.transpose(
        premult_color,
        (1, 2, 0),
    )

    # Premultiplied color over black.
    overlay_black = color_hw

    # Useful only for visualization.
    white = np.ones_like(color_hw)
    alpha_3 = alpha[..., None]
    overlay_white = color_hw + (
        1.0 - alpha_3
    ) * white

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8, 8),
    )

    axes[0, 0].imshow(
        np.clip(color_hw, 0.0, 1.0)
    )
    axes[0, 0].set_title("Predicted premultiplied RGB")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(
        alpha,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    axes[0, 1].set_title("Predicted alpha")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(
        np.clip(overlay_black, 0.0, 1.0)
    )
    axes[1, 0].set_title("Overlay over black")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(
        np.clip(overlay_white, 0.0, 1.0)
    )
    axes[1, 1].set_title("Overlay over white")
    axes[1, 1].axis("off")

    fig.suptitle(
        f"{mode} | camera={camera_index} | sample={sample_id}\n"
        f"actual radius={actual_radius:.4f} | "
        f"phi={math.degrees(phi):.2f}° | "
        f"theta={math.degrees(theta):.2f}° | "
        f"target angle={alignment_angle:.2f}°",
        fontsize=10,
    )

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
    parser = ArgumentParser(
        description=(
            "Run a pretrained FNO using cameras loaded from a "
            "3DGS scene."
        )
    )

    lp = ModelParams(parser)

    parser.add_argument(
        "--mode",
        choices=["surface", "volume"],
        default="surface",
    )

    parser.add_argument(
        "--camera_index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--num_cameras",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--template_index",
        type=int,
        default=0,
        help=(
            "Index inside the selected surface/volume rows "
            "from the FNO dataset."
        ),
    )

    parser.add_argument(
        "--surface_checkpoint",
        type=str,
        default=str(DEFAULT_SURFACE_CHECKPOINT),
    )

    parser.add_argument(
        "--volume_checkpoint",
        type=str,
        default=str(DEFAULT_VOLUME_CHECKPOINT),
    )

    parser.add_argument(
        "--fno_radius",
        type=float,
        default=FNO_RADIUS,
        help=(
            "Canonical radius passed to the FNO. "
            "Must match the FNO training radius."
        ),
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if device.type != "cuda":
        raise RuntimeError(
            "This 3DGS Scene/GaussianModel setup expects CUDA."
        )

    print("Using device:", device)
    print("Mode:", args.mode)
    print("FNO radius:", args.fno_radius)

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

    print("FNO dataset rows:", len(dataset.df))
    print(dataset.df["render_mode"].value_counts())

    if "render_mode" not in dataset.df.columns:
        raise RuntimeError(
            "render_mode is missing from the FNO metadata."
        )

    # --------------------------------------------------------
    # Load only the requested FNO model.
    # --------------------------------------------------------
    if args.mode == "surface":
        checkpoint_path = args.surface_checkpoint
    else:
        checkpoint_path = args.volume_checkpoint

    model, param_mean, param_std, latent_dim = \
        load_fno_checkpoint(
            checkpoint_path,
            device,
        )

    if latent_dim != dataset.latent_dim:
        raise RuntimeError(
            f"Checkpoint latent_dim={latent_dim}, "
            f"dataset latent_dim={dataset.latent_dim}"
        )

    # --------------------------------------------------------
    # Choose a metadata template for this mode.
    # --------------------------------------------------------
    mode_rows = dataset.df[
        dataset.df["render_mode"].astype(str) == args.mode
    ].reset_index(drop=True)

    if len(mode_rows) == 0:
        raise RuntimeError(
            f"No metadata rows found for mode={args.mode}"
        )

    if args.template_index < 0 or \
       args.template_index >= len(mode_rows):
        raise IndexError(
            f"template_index={args.template_index} is invalid. "
            f"Available rows: {len(mode_rows)}"
        )

    template_row = mode_rows.iloc[
        args.template_index
    ]

    template_sample_id = int(
        template_row["sample_id"]
    )

    print(
        f"Using template index={args.template_index}, "
        f"sample_id={template_sample_id}, "
        f"mode={args.mode}"
    )

    # --------------------------------------------------------
    # Load the 3DGS scene only for cameras.
    #
    # The existing Scene class still initializes a GaussianModel,
    # but this test does not use Gaussian rendering.
    # --------------------------------------------------------
    scene_args = lp.extract(args)

    if not scene_args.model_path:
        scene_args.model_path = str(
            REPO_DIR / "output" / "neural_camera_test"
        )

    Path(scene_args.model_path).mkdir(
        parents=True,
        exist_ok=True,
    )

    gaussian_model = GaussianModel(
        args.sh_degree,
        getattr(args, "optimizer_type", "default"),
    )

    scene = Scene(
        scene_args,
        gaussian_model,
        shuffle=False,
        resolution_scales=[1.0],
    )

    # Compute a different center for each loaded scene.
    scene_center = compute_scene_median_center(
        gaussian_model
    ).to(device)

    print(
        "Computed scene median center:",
        scene_center.detach().cpu().numpy(),
    )

    cameras = scene.getTrainCameras(scale=1.0)

    if len(cameras) == 0:
        raise RuntimeError(
            "No training cameras were loaded."
        )

    if args.camera_index < 0 or \
       args.camera_index >= len(cameras):
        raise IndexError(
            f"camera_index={args.camera_index} is invalid. "
            f"Available cameras: {len(cameras)}"
        )

    first_camera = args.camera_index
    last_camera = min(
        first_camera + args.num_cameras,
        len(cameras),
    )

    # --------------------------------------------------------
    # Render with the FNO for selected 3DGS cameras.
    # --------------------------------------------------------
    with torch.no_grad():
        for camera_index in range(
            first_camera,
            last_camera,
        ):
            viewpoint_camera = cameras[camera_index]

            phi, theta, actual_radius, relative = \
                camera_to_slice_pose(
                    viewpoint_camera,
                    scene_center,
                )

            # Reconstruct relative camera position to verify
            # the spherical conversion.
            reconstructed = reconstruct_camera_position(
                phi,
                theta,
                actual_radius,
            )

            reconstruction_error = torch.linalg.norm(
                reconstructed - relative
            ).item()

            # Check whether this real camera is looking at the
            # computed scene center.
            dot, alignment_angle = alignment_to_center(
                viewpoint_camera,
                scene_center,
            )

            # Build the exact normalized FNO input.
            param_vec = build_fno_param_vector(
                dataset=dataset,
                template_row=template_row,
                mode=args.mode,
                phi=phi,
                theta=theta,
                fno_radius=args.fno_radius,
                param_mean=param_mean,
                param_std=param_std,
                device=device,
            )

            prediction = model(param_vec)

            if prediction.shape != (1, 4, 64, 64):
                raise RuntimeError(
                    f"Unexpected FNO output shape: "
                    f"{prediction.shape}"
                )

            rgba_np = prediction[0].cpu().numpy()

            sample_id = int(template_row["sample_id"])

            output_path = OUTPUT_DIR / (
                f"{args.mode}_camera_{camera_index:04d}.png"
            )

            save_prediction_preview(
                rgba=rgba_np,
                output_path=output_path,
                mode=args.mode,
                camera_index=camera_index,
                sample_id=sample_id,
                actual_radius=float(
                    actual_radius.detach().cpu().item()
                ),
                phi=float(
                    phi.detach().cpu().item()
                ),
                theta=float(
                    theta.detach().cpu().item()
                ),
                alignment_angle=float(
                    alignment_angle.detach().cpu().item()
                ),
            )

            print(
                f"camera={camera_index} "
                f"name={viewpoint_camera.image_name} "
                f"actual_radius="
                f"{actual_radius.item():.6f} "
                f"phi={phi.item():.6f} "
                f"theta={theta.item():.6f} "
                f"reconstruction_error="
                f"{reconstruction_error:.8e} "
                f"target_alignment="
                f"{dot.item():.6f} "
                f"angle_to_target="
                f"{alignment_angle.item():.3f}° "
                f"FoVy={viewpoint_camera.FoVy:.6f}"
            )

            print("Saved:", output_path)

    print("Done.")
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()