#!/usr/bin/env python

import os
import sys
import math
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import imageio.v2 as imageio

import torch
import torch.nn.functional as F


# ============================================================
# PATH SETUP
# ============================================================

REPO_DIR = Path(__file__).resolve().parent
FNO_ROOT = REPO_DIR.parent

# Make FNO_Large and the 3DGS repository importable.
sys.path.insert(0, str(FNO_ROOT))
sys.path.insert(0, str(REPO_DIR))


# ============================================================
# IMPORTS
# ============================================================

from train_premult_single_mode import (
    PlaneDatasetParamsToPremultRGBA,
    FNOPlusResNetSingle,
)

from scene import Scene, GaussianModel
from arguments import ModelParams


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = FNO_ROOT / "plane_dataset_4"

RENDERS_DIR = BASE_DIR / "renders"
ALPHA_DIR = BASE_DIR / "hard_alpha"

IMG_META_CSV = RENDERS_DIR / "metadata_images_all_sharded.csv"
VOL_META_CSV = BASE_DIR / "metadata_volumes.csv"
ALPHA_META_CSV = ALPHA_DIR / "metadata_alpha_all.csv"

DEFAULT_SURFACE_CHECKPOINT = (
    FNO_ROOT / "fno_premult_surface_final.pt"
)

DEFAULT_VOLUME_CHECKPOINT = (
    FNO_ROOT / "fno_premult_volume_epoch025.pt"
)

DEFAULT_OUTPUT_DIR = (
    REPO_DIR / "neural_slice_placement_outputs"
)

IMG_SIZE = (64, 64)

# This must match the fixed radius used to train the FNO.
DEFAULT_FNO_RADIUS = 2.2


# ============================================================
# CAMERA / SCENE HELPERS
# ============================================================

def compute_scene_median_center(gaussian_model):
    """
    Compute a robust scene center from the initialized point cloud.

    This center is scene-specific. It is only used as the initial
    center for the test neural slice.
    """
    xyz = gaussian_model.get_xyz.detach()

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise RuntimeError(
            f"Expected point cloud shape [N,3], got {tuple(xyz.shape)}"
        )

    return xyz.median(dim=0).values


def camera_to_slice_pose(viewpoint_camera, slice_center):
    """
    Compute the camera position relative to a neural slice.

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
    Reconstruct a relative camera position from spherical coordinates.
    """
    return torch.stack([
        radius * torch.sin(phi) * torch.cos(theta),
        radius * torch.sin(phi) * torch.sin(theta),
        radius * torch.cos(phi),
    ])


def project_world_point(viewpoint_camera, point_world):
    """
    Project one world-space point into the 3DGS camera image.

    Important:
    The 3DGS camera matrices use a row-vector convention, so the
    homogeneous point is multiplied on the left:

        clip = point_h @ full_proj_transform
    """
    matrix = viewpoint_camera.full_proj_transform
    device = matrix.device
    dtype = matrix.dtype

    point_world = point_world.to(device=device, dtype=dtype)

    point_h = torch.cat([
        point_world,
        torch.ones(1, device=device, dtype=dtype),
    ])

    clip = point_h @ matrix

    if torch.abs(clip[3]) < 1e-8:
        raise RuntimeError(
            "Point projection produced an invalid homogeneous coordinate."
        )

    ndc = clip[:3] / clip[3]

    width = float(viewpoint_camera.image_width)
    height = float(viewpoint_camera.image_height)

    pixel_x = (ndc[0] + 1.0) * 0.5 * width
    pixel_y = (1.0 - ndc[1]) * 0.5 * height

    return pixel_x, pixel_y, ndc[2]


def projected_patch_bbox(
    viewpoint_camera,
    slice_center,
    world_width,
    world_height,
):
    """
    Project an unrotated rectangular slice into the camera image.

    Current assumptions:
        - slice local X is world X
        - slice local Y is world Y
        - no slice rotation
        - rectangle lies in the world XY plane

    Returns:
        center_x
        center_y
        width_px
        height_px
        corners_px
    """
    device = viewpoint_camera.camera_center.device
    dtype = viewpoint_camera.camera_center.dtype

    half_width = 0.5 * world_width
    half_height = 0.5 * world_height

    offsets = torch.tensor(
        [
            [-half_width, -half_height, 0.0],
            [ half_width, -half_height, 0.0],
            [ half_width,  half_height, 0.0],
            [-half_width,  half_height, 0.0],
        ],
        dtype=dtype,
        device=device,
    )

    corners_world = slice_center[None, :] + offsets

    projected = []
    for corner in corners_world:
        px, py, depth = project_world_point(
            viewpoint_camera,
            corner,
        )
        projected.append(torch.stack([px, py, depth]))

    projected = torch.stack(projected, dim=0)

    center_x, center_y, center_depth = project_world_point(
        viewpoint_camera,
        slice_center,
    )

    return {
        "center_x": center_x,
        "center_y": center_y,
        "width_px": (
            projected[:, 0].max() - projected[:, 0].min()
        ).clamp_min(1.0),
        "height_px": (
            projected[:, 1].max() - projected[:, 1].min()
        ).clamp_min(1.0),
        "corners_px": projected,
        "center_depth": center_depth,
    }


def resize_neural_patch(rgba, width_px, height_px):
    """
    Resize a premultiplied RGBA tensor.

    Input:
        rgba: [1,4,64,64]

    Output:
        [1,4,new_height,new_width]
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
    Paste a patch into a larger RGBA canvas.

    Both tensors use:
        [1,4,H,W]

    The patch is clipped if it extends outside the canvas.
    """
    _, _, canvas_h, canvas_w = canvas.shape
    _, _, patch_h, patch_w = patch.shape

    cx = int(round(float(center_x)))
    cy = int(round(float(center_y)))

    x0 = cx - patch_w // 2
    y0 = cy - patch_h // 2
    x1 = x0 + patch_w
    y1 = y0 + patch_h

    canvas_x0 = max(0, x0)
    canvas_y0 = max(0, y0)
    canvas_x1 = min(canvas_w, x1)
    canvas_y1 = min(canvas_h, y1)

    if canvas_x0 >= canvas_x1 or canvas_y0 >= canvas_y1:
        return canvas

    patch_x0 = canvas_x0 - x0
    patch_y0 = canvas_y0 - y0
    patch_x1 = patch_x0 + (canvas_x1 - canvas_x0)
    patch_y1 = patch_y0 + (canvas_y1 - canvas_y0)

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


# ============================================================
# FNO HELPERS
# ============================================================

def load_fno_checkpoint(checkpoint_path, device):
    """
    Load a single-mode FNO checkpoint and its normalization stats.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    print("Loading FNO checkpoint:", checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state = checkpoint.get("model_state", checkpoint)
    state = dict(state)
    state.pop("_metadata", None)

    if "latent_dim" not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain latent_dim."
        )

    if "param_mean" not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain param_mean."
        )

    if "param_std" not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain param_std."
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
            f"param_mean length {len(param_mean)} "
            f"does not match latent_dim {latent_dim}."
        )

    if len(param_std) != latent_dim:
        raise RuntimeError(
            f"param_std length {len(param_std)} "
            f"does not match latent_dim {latent_dim}."
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
    Build the exact normalized input vector used during FNO training.

    The template row provides:
        - sample/shape identity
        - material values
        - environment SH values

    The current camera replaces:
        - phi
        - theta

    The canonical FNO radius is supplied separately.
    """
    if mode not in ("surface", "volume"):
        raise ValueError(f"Invalid mode: {mode}")

    row = template_row.copy()

    row["phi"] = float(phi.detach().cpu().item())
    row["theta"] = float(theta.detach().cpu().item())
    row["radius"] = float(fno_radius)
    row["render_mode"] = mode

    if mode == "volume":
        # These were zeroed during volume training.
        row["metallic"] = 0.0
        row["roughness"] = 0.0
        row["specular"] = 0.0

    raw_np = dataset._build_param_vector_np(row)

    if raw_np.shape[0] != param_mean.shape[0]:
        raise RuntimeError(
            f"Parameter dimension mismatch: "
            f"constructed {raw_np.shape[0]}, "
            f"checkpoint expects {param_mean.shape[0]}."
        )

    normalized_np = (
        raw_np - param_mean
    ) / param_std

    return torch.from_numpy(
        normalized_np.astype(np.float32)
    ).unsqueeze(0).to(device)


# ============================================================
# IMAGE SAVING
# ============================================================

def save_rgb_tensor(rgb_chw, path):
    """
    Save RGB tensor/array with shape [3,H,W].
    """
    if torch.is_tensor(rgb_chw):
        rgb_chw = rgb_chw.detach().cpu().numpy()

    rgb_hwc = np.transpose(rgb_chw, (1, 2, 0))
    rgb_hwc = np.clip(rgb_hwc, 0.0, 1.0)

    imageio.imwrite(
        path,
        (rgb_hwc * 255.0 + 0.5).astype(np.uint8),
    )


def save_rgba_tensor(rgba_chw, path):
    """
    Save premultiplied RGBA with shape [4,H,W].
    """
    if torch.is_tensor(rgba_chw):
        rgba_chw = rgba_chw.detach().cpu().numpy()

    rgba_hwc = np.transpose(rgba_chw, (1, 2, 0))
    rgba_hwc = np.clip(rgba_hwc, 0.0, 1.0)

    imageio.imwrite(
        path,
        (rgba_hwc * 255.0 + 0.5).astype(np.uint8),
    )


def save_alpha_tensor(alpha_hw, path):
    """
    Save alpha as grayscale RGB.
    """
    if torch.is_tensor(alpha_hw):
        alpha_hw = alpha_hw.detach().cpu().numpy()

    alpha_hw = np.clip(alpha_hw, 0.0, 1.0)
    alpha_rgb = np.repeat(alpha_hw[..., None], 3, axis=2)

    imageio.imwrite(
        path,
        (alpha_rgb * 255.0 + 0.5).astype(np.uint8),
    )


def camera_forward_world(viewpoint_camera):
    view_inv = torch.linalg.inv(
        viewpoint_camera.world_view_transform
    )

    forward = view_inv[2, :3]

    return forward / torch.linalg.norm(
        forward
    ).clamp_min(1e-8
    )


def alignment_to_center(viewpoint_camera, target_center):
    camera_center = viewpoint_camera.camera_center

    toward_target = target_center - camera_center
    toward_target = toward_target / torch.linalg.norm(
        toward_target
    ).clamp_min(1e-8)

    forward = camera_forward_world(viewpoint_camera)

    dot = torch.dot(forward, toward_target)
    dot = dot.clamp(-1.0, 1.0)

    angle_deg = torch.rad2deg(torch.acos(dot))

    return dot, angle_deg

# ============================================================
# MAIN
# ============================================================

def main():
    parser = ArgumentParser(
        description=(
            "Run a pretrained FNO on cameras from a 3DGS scene, "
            "then resize and place the neural slice in the scene image."
        )
    )

    # Adds --source_path, --model_path, --resolution, etc.
    lp = ModelParams(parser)

    parser.add_argument(
        "--mode",
        choices=["surface", "volume"],
        default="surface",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Single-mode FNO checkpoint.",
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
        help="Index inside the selected mode's FNO metadata rows.",
    )

    parser.add_argument(
        "--fno_radius",
        type=float,
        default=DEFAULT_FNO_RADIUS,
        help="Canonical radius used during FNO training.",
    )

    parser.add_argument(
        "--slice_width",
        type=float,
        default=1.0,
        help="Neural slice width in scene/world units.",
    )

    parser.add_argument(
        "--slice_height",
        type=float,
        default=1.0,
        help="Neural slice height in scene/world units.",
    )

    parser.add_argument(
        "--slice_dx",
        type=float,
        default=0.0,
        help="Slice X offset from the scene median.",
    )

    parser.add_argument(
        "--slice_dy",
        type=float,
        default=0.0,
        help="Slice Y offset from the scene median.",
    )

    parser.add_argument(
        "--slice_dz",
        type=float,
        default=0.0,
        help="Slice Z offset from the scene median.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required by the current 3DGS Scene/GaussianModel setup."
        )

    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Using device:", device)
    print("Mode:", args.mode)
    print("FNO radius:", args.fno_radius)

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

    if "render_mode" not in fno_dataset.df.columns:
        raise RuntimeError(
            "render_mode is missing from FNO metadata."
        )

    print("FNO rows:", len(fno_dataset.df))
    print(fno_dataset.df["render_mode"].value_counts())

    # --------------------------------------------------------
    # Load selected FNO checkpoint.
    # --------------------------------------------------------
    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint)
    elif args.mode == "surface":
        checkpoint_path = DEFAULT_SURFACE_CHECKPOINT
    else:
        checkpoint_path = DEFAULT_VOLUME_CHECKPOINT

    model, param_mean, param_std, latent_dim = \
        load_fno_checkpoint(
            checkpoint_path,
            device,
        )

    if latent_dim != fno_dataset.latent_dim:
        raise RuntimeError(
            f"Checkpoint latent_dim={latent_dim}, "
            f"dataset latent_dim={fno_dataset.latent_dim}"
        )

    # --------------------------------------------------------
    # Select one metadata template for the requested mode.
    # --------------------------------------------------------
    mode_rows = fno_dataset.df[
        fno_dataset.df["render_mode"].astype(str) == args.mode
    ].reset_index(drop=True)

    if len(mode_rows) == 0:
        raise RuntimeError(
            f"No rows found for mode={args.mode}."
        )

    if args.template_index < 0 or \
       args.template_index >= len(mode_rows):
        raise IndexError(
            f"template_index={args.template_index} is invalid. "
            f"Available rows: {len(mode_rows)}."
        )

    template_row = mode_rows.iloc[args.template_index]
    template_sample_id = int(template_row["sample_id"])

    print(
        "Template index:",
        args.template_index,
        "sample_id:",
        template_sample_id,
    )

    # --------------------------------------------------------
    # Load the 3DGS scene only for camera information.
    #
    # GaussianModel is still required by the original Scene
    # implementation, but it is not used for neural rendering.
    # --------------------------------------------------------
    scene_args = lp.extract(args)

    if not scene_args.model_path:
        scene_args.model_path = str(
            REPO_DIR / "output" / "neural_slice_test"
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

    cameras = scene.getTrainCameras(scale=1.0)

    if len(cameras) == 0:
        raise RuntimeError(
            "No training cameras were loaded."
        )

    if args.camera_index < 0 or \
       args.camera_index >= len(cameras):
        raise IndexError(
            f"camera_index={args.camera_index} is invalid. "
            f"Available cameras: {len(cameras)}."
        )

    # --------------------------------------------------------
    # Scene-specific slice center.
    # --------------------------------------------------------
    scene_center = compute_scene_median_center(
        gaussian_model
    )

    slice_offset = torch.tensor(
        [
            args.slice_dx,
            args.slice_dy,
            args.slice_dz,
        ],
        dtype=scene_center.dtype,
        device=device,
    )

    slice_center = scene_center.to(device) + slice_offset

    print(
        "Scene median center:",
        scene_center.detach().cpu().numpy(),
    )

    print(
        "Neural slice center:",
        slice_center.detach().cpu().numpy(),
    )

    # --------------------------------------------------------
    # Render selected cameras.
    # --------------------------------------------------------
    first_camera = args.camera_index
    last_camera = min(
        first_camera + args.num_cameras,
        len(cameras),
    )

    with torch.no_grad():
        for camera_index in range(first_camera, last_camera):
            viewpoint_camera = cameras[camera_index]

            # ------------------------------------------------
            # Camera pose relative to slice.
            # ------------------------------------------------
            phi, theta, actual_radius, relative = \
                camera_to_slice_pose(
                    viewpoint_camera,
                    slice_center,
                )

            reconstructed_relative = \
                reconstruct_camera_position(
                    phi,
                    theta,
                    actual_radius,
                )

            reconstruction_error = torch.linalg.norm(
                reconstructed_relative - relative
            ).item()

            # ------------------------------------------------
            # Build FNO input.
            # ------------------------------------------------
            param_vec = build_fno_param_vector(
                dataset=fno_dataset,
                template_row=template_row,
                mode=args.mode,
                phi=phi,
                theta=theta,
                fno_radius=args.fno_radius,
                param_mean=param_mean,
                param_std=param_std,
                device=device,
            )

            prediction = model(param_vec).clamp(0.0, 1.0)

            if prediction.shape != (1, 4, 64, 64):
                raise RuntimeError(
                    f"Expected FNO output [1,4,64,64], "
                    f"got {tuple(prediction.shape)}"
                )

            # ------------------------------------------------
            # Project slice rectangle into the real camera.
            # ------------------------------------------------
            bbox = projected_patch_bbox(
                viewpoint_camera=viewpoint_camera,
                slice_center=slice_center,
                world_width=args.slice_width,
                world_height=args.slice_height,
            )

            projected_width = float(
                bbox["width_px"].detach().cpu().item()
            )

            projected_height = float(
                bbox["height_px"].detach().cpu().item()
            )

            projected_center_x = float(
                bbox["center_x"].detach().cpu().item()
            )

            projected_center_y = float(
                bbox["center_y"].detach().cpu().item()
            )

            # ------------------------------------------------
            # Resize and place the neural patch.
            # ------------------------------------------------
            canvas_h = int(viewpoint_camera.image_height)
            canvas_w = int(viewpoint_camera.image_width)

            canvas = torch.zeros(
                1,
                4,
                canvas_h,
                canvas_w,
                dtype=prediction.dtype,
                device=device,
            )

            resized_patch = resize_neural_patch(
                prediction,
                width_px=projected_width,
                height_px=projected_height,
            )

            patch_h = resized_patch.shape[-2]
            patch_w = resized_patch.shape[-1]

            max_patch_dim = 4 * max(canvas_h, canvas_w)

            if patch_h > max_patch_dim or \
               patch_w > max_patch_dim:
                print(
                    f"[WARN] Projected patch {patch_w}x{patch_h} "
                    f"is too large; skipping camera {camera_index}."
                )
                continue

            canvas = paste_patch(
                canvas=canvas,
                patch=resized_patch,
                center_x=projected_center_x,
                center_y=projected_center_y,
            )

            # ------------------------------------------------
            # Save neural output.
            # ------------------------------------------------
            canvas_np = canvas[0].detach().cpu().numpy()

            placed_rgba_path = output_dir / (
                f"placed_{args.mode}_camera_{camera_index:04d}.png"
            )

            placed_rgb_path = output_dir / (
                f"placed_{args.mode}_rgb_camera_{camera_index:04d}.png"
            )

            placed_alpha_path = output_dir / (
                f"placed_{args.mode}_alpha_camera_{camera_index:04d}.png"
            )

            save_rgba_tensor(
                canvas_np,
                placed_rgba_path,
            )

            save_rgb_tensor(
                canvas_np[:3],
                placed_rgb_path,
            )

            save_alpha_tensor(
                canvas_np[3],
                placed_alpha_path,
            )

            # ------------------------------------------------
            # Save the actual scene target image.
            # ------------------------------------------------
            target = viewpoint_camera.original_image

            target_path = output_dir / (
                f"target_camera_{camera_index:04d}.png"
            )

            save_rgb_tensor(
                target,
                target_path,
            )

            # ------------------------------------------------
            # Save target / placed side-by-side.
            # ------------------------------------------------
            target_np = target.detach().cpu().numpy()
            target_hwc = np.transpose(target_np, (1, 2, 0))
            target_hwc = np.clip(target_hwc, 0.0, 1.0)

            placed_hwc = np.transpose(
                canvas_np[:3],
                (1, 2, 0),
            )
            placed_hwc = np.clip(placed_hwc, 0.0, 1.0)

            if target_hwc.shape != placed_hwc.shape:
                print(
                    f"[WARN] Target/output shape mismatch: "
                    f"{target_hwc.shape} vs {placed_hwc.shape}"
                )
            else:
                comparison = np.concatenate(
                    [
                        target_hwc,
                        placed_hwc,
                    ],
                    axis=1,
                )

                comparison_path = output_dir / (
                    f"comparison_{args.mode}_camera_{camera_index:04d}.png"
                )

                imageio.imwrite(
                    comparison_path,
                    (comparison * 255.0 + 0.5).astype(np.uint8),
                )

            # ------------------------------------------------
            # Diagnostics.
            # ------------------------------------------------
            print(
                f"camera={camera_index} "
                f"name={viewpoint_camera.image_name} "
                f"actual_radius={actual_radius.item():.6f} "
                f"phi={phi.item():.6f} "
                f"theta={theta.item():.6f} "
                f"reconstruction_error="
                f"{reconstruction_error:.8e} "
                f"projected_center=("
                f"{projected_center_x:.2f},"
                f"{projected_center_y:.2f}) "
                f"projected_size=("
                f"{projected_width:.2f}x"
                f"{projected_height:.2f}) "
                f"FoVy={viewpoint_camera.FoVy:.6f}"
            )

            print("Saved:", placed_rgba_path)
            print("Saved:", placed_rgb_path)
            print("Saved:", placed_alpha_path)
            print("Saved:", target_path)
            
            dot, angle_deg = alignment_to_center(
                viewpoint_camera,
                slice_center,
            )
            
            print(
                f"camera={camera_index} "
                f"target_alignment={dot.item():.6f} "
                f"angle_to_slice_center={angle_deg.item():.3f} degrees"
            )

    print("Done.")
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()