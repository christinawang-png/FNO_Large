#!/usr/bin/env python

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

# Make both repositories importable.
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


# ============================================================
# FNO DATASET PATHS
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
    REPO_DIR / "two_neural_slice_outputs"
)

IMG_SIZE = (64, 64)


# ============================================================
# SCENE / CAMERA HELPERS
# ============================================================

def compute_scene_median_center(gaussian_model):
    """
    Compute a robust, scene-specific center from the point cloud.

    This is only the default center for the two test slices.
    """
    xyz = gaussian_model.get_xyz.detach()

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise RuntimeError(
            f"Expected point cloud [N,3], got {tuple(xyz.shape)}"
        )

    return xyz.median(dim=0).values


def camera_to_slice_pose(viewpoint_camera, slice_center):
    """
    Calculate the spherical camera pose relative to a slice center.

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
    Reconstruct relative camera position from spherical coordinates.
    """
    return torch.stack([
        radius * torch.sin(phi) * torch.cos(theta),
        radius * torch.sin(phi) * torch.sin(theta),
        radius * torch.cos(phi),
    ])


def project_world_point(viewpoint_camera, point_world):
    """
    Project a world-space point into a 3DGS camera.

    3DGS uses row-vector matrix multiplication:
        clip = point_h @ full_proj_transform
    """
    matrix = viewpoint_camera.full_proj_transform
    device = matrix.device
    dtype = matrix.dtype

    point_world = point_world.to(
        device=device,
        dtype=dtype,
    )

    point_h = torch.cat([
        point_world,
        torch.ones(1, device=device, dtype=dtype),
    ])

    clip = point_h @ matrix

    if torch.abs(clip[3]) < 1e-8:
        raise RuntimeError(
            "Invalid homogeneous coordinate during projection."
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
    Project an unrotated rectangular slice.

    Current slice orientation:
        local X = world X
        local Y = world Y
        local Z = world Z

    Rotation support can be added later.
    """
    device = viewpoint_camera.camera_center.device
    dtype = viewpoint_camera.camera_center.dtype

    hx = 0.5 * float(world_width)
    hy = 0.5 * float(world_height)

    offsets = torch.tensor(
        [
            [-hx, -hy, 0.0],
            [ hx, -hy, 0.0],
            [ hx,  hy, 0.0],
            [-hx,  hy, 0.0],
        ],
        dtype=dtype,
        device=device,
    )

    corners_world = slice_center[None, :] + offsets

    projected_corners = []

    for corner in corners_world:
        px, py, depth = project_world_point(
            viewpoint_camera,
            corner,
        )
        projected_corners.append(
            torch.stack([px, py, depth])
        )

    projected_corners = torch.stack(
        projected_corners,
        dim=0,
    )

    center_x, center_y, center_depth = project_world_point(
        viewpoint_camera,
        slice_center,
    )

    width_px = (
        projected_corners[:, 0].max()
        - projected_corners[:, 0].min()
    ).clamp_min(1.0)

    height_px = (
        projected_corners[:, 1].max()
        - projected_corners[:, 1].min()
    ).clamp_min(1.0)

    return {
        "center_x": center_x,
        "center_y": center_y,
        "center_depth": center_depth,
        "width_px": width_px,
        "height_px": height_px,
        "corners_px": projected_corners,
    }


# ============================================================
# PATCH OPERATIONS
# ============================================================

def resize_neural_patch(
    rgba,
    width_px,
    height_px,
):
    """
    Resize premultiplied RGBA.

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


def paste_patch(
    canvas,
    patch,
    center_x,
    center_y,
):
    """
    Paste a patch into a larger RGBA canvas.

    Both tensors:
        [1,4,H,W]

    This is appropriate for a fixed visualization test.
    It is not differentiable with respect to placement.
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


def alpha_over(
    back,
    front,
):
    """
    Composite premultiplied RGBA front over back.

    Inputs:
        back/front: [1,4,H,W]

    Returns:
        [1,4,H,W]
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


def composite_layers_back_to_front(
    layers,
    device,
):
    """
    Composite layers in back-to-front order.

    Each layer must be:
        [1,4,H,W]

    The first item is treated as the back layer.
    """
    if len(layers) == 0:
        raise ValueError("No layers were provided.")

    output = torch.zeros_like(layers[0])

    for layer in layers:
        output = alpha_over(
            back=output,
            front=layer,
        )

    return output


# ============================================================
# FNO HELPERS
# ============================================================

def load_fno_checkpoint(
    checkpoint_path,
    device,
):
    """
    Load a single-mode FNO and its saved normalization statistics.
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

    state = checkpoint.get(
        "model_state",
        checkpoint,
    )

    state = dict(state)
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
    Rebuild the exact normalized input vector used during training.

    The template row supplies:
        - shape identity/control points
        - sigma
        - material values
        - environment SH values

    Camera values are replaced by the current camera.
    """
    row = template_row.copy()

    row["phi"] = float(
        phi.detach().cpu().item()
    )

    row["theta"] = float(
        theta.detach().cpu().item()
    )

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
# IMAGE SAVING
# ============================================================

def chw_to_hwc(array_chw):
    return np.transpose(array_chw, (1, 2, 0))


def save_rgb_chw(rgb_chw, path):
    """
    Save [3,H,W] RGB as uint8 PNG.
    """
    if torch.is_tensor(rgb_chw):
        rgb_chw = rgb_chw.detach().cpu().numpy()

    rgb_hwc = chw_to_hwc(rgb_chw)
    rgb_hwc = np.clip(rgb_hwc, 0.0, 1.0)

    imageio.imwrite(
        path,
        (rgb_hwc * 255.0 + 0.5).astype(np.uint8),
    )


def save_rgba_chw(rgba_chw, path):
    """
    Save [4,H,W] RGBA as uint8 PNG.

    The RGB channels are premultiplied.
    """
    if torch.is_tensor(rgba_chw):
        rgba_chw = rgba_chw.detach().cpu().numpy()

    rgba_hwc = chw_to_hwc(rgba_chw)
    rgba_hwc = np.clip(rgba_hwc, 0.0, 1.0)

    imageio.imwrite(
        path,
        (rgba_hwc * 255.0 + 0.5).astype(np.uint8),
    )


def save_alpha_chw(alpha_chw, path):
    """
    Save alpha as grayscale RGB.
    """
    if torch.is_tensor(alpha_chw):
        alpha_chw = alpha_chw.detach().cpu().numpy()

    if alpha_chw.ndim == 3:
        alpha_hw = alpha_chw[0]
    else:
        alpha_hw = alpha_chw

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


# ============================================================
# MAIN
# ============================================================

def main():
    parser = ArgumentParser(
        description=(
            "Render two neural FNO slices using cameras from "
            "a 3DGS scene and alpha-composite them."
        )
    )

    # Adds --source_path, --model_path, --resolution, etc.
    lp = ModelParams(parser)

    parser.add_argument(
        "--mode1",
        choices=["surface", "volume"],
        default="surface",
        help="Mode of slice 1.",
    )

    parser.add_argument(
        "--mode2",
        choices=["surface", "volume"],
        default="volume",
        help="Mode of slice 2.",
    )

    parser.add_argument(
        "--checkpoint1",
        type=str,
        default=None,
        help="Optional checkpoint for slice 1.",
    )

    parser.add_argument(
        "--checkpoint2",
        type=str,
        default=None,
        help="Optional checkpoint for slice 2.",
    )

    parser.add_argument(
        "--template1",
        type=int,
        default=0,
        help="FNO metadata template index for slice 1.",
    )

    parser.add_argument(
        "--template2",
        type=int,
        default=1,
        help="FNO metadata template index for slice 2.",
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
        "--fno_radius",
        type=float,
        default=2.2,
    )

    # Slice 1 offset and size.
    parser.add_argument("--slice1_dx", type=float, default=0.0)
    parser.add_argument("--slice1_dy", type=float, default=0.0)
    parser.add_argument("--slice1_dz", type=float, default=0.0)
    parser.add_argument("--slice1_width", type=float, default=1.0)
    parser.add_argument("--slice1_height", type=float, default=1.0)

    # Slice 2 offset and size.
    parser.add_argument("--slice2_dx", type=float, default=0.35)
    parser.add_argument("--slice2_dy", type=float, default=0.0)
    parser.add_argument("--slice2_dz", type=float, default=0.25)
    parser.add_argument("--slice2_width", type=float, default=0.8)
    parser.add_argument("--slice2_height", type=float, default=0.8)

    parser.add_argument(
        "--order",
        type=str,
        default="slice1,slice2",
        choices=["slice1,slice2", "slice2,slice1"],
        help=(
            "Back-to-front compositing order. "
            "The first listed slice is behind."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required by the current 3DGS code."
        )

    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Using device:", device)
    print("Slice 1 mode:", args.mode1)
    print("Slice 2 mode:", args.mode2)
    print("FNO radius:", args.fno_radius)
    print("Composite order:", args.order)

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
    # Load one checkpoint per mode.
    # --------------------------------------------------------
    if args.checkpoint1 is not None:
        checkpoint1 = Path(args.checkpoint1)
    elif args.mode1 == "surface":
        checkpoint1 = SURFACE_CHECKPOINT
    else:
        checkpoint1 = VOLUME_CHECKPOINT

    if args.checkpoint2 is not None:
        checkpoint2 = Path(args.checkpoint2)
    elif args.mode2 == "surface":
        checkpoint2 = SURFACE_CHECKPOINT
    else:
        checkpoint2 = VOLUME_CHECKPOINT

    model1, mean1, std1, dim1 = load_fno_checkpoint(
        checkpoint1,
        device,
    )

    model2, mean2, std2, dim2 = load_fno_checkpoint(
        checkpoint2,
        device,
    )

    if dim1 != fno_dataset.latent_dim:
        raise RuntimeError(
            f"Slice 1 checkpoint latent_dim={dim1}, "
            f"dataset latent_dim={fno_dataset.latent_dim}"
        )

    if dim2 != fno_dataset.latent_dim:
        raise RuntimeError(
            f"Slice 2 checkpoint latent_dim={dim2}, "
            f"dataset latent_dim={fno_dataset.latent_dim}"
        )

    # --------------------------------------------------------
    # Select template rows.
    # --------------------------------------------------------
    rows1 = fno_dataset.df[
        fno_dataset.df["render_mode"].astype(str) == args.mode1
    ].reset_index(drop=True)

    rows2 = fno_dataset.df[
        fno_dataset.df["render_mode"].astype(str) == args.mode2
    ].reset_index(drop=True)

    if len(rows1) == 0:
        raise RuntimeError(
            f"No rows found for mode1={args.mode1}"
        )

    if len(rows2) == 0:
        raise RuntimeError(
            f"No rows found for mode2={args.mode2}"
        )

    if not 0 <= args.template1 < len(rows1):
        raise IndexError(
            f"template1={args.template1} is out of range."
        )

    if not 0 <= args.template2 < len(rows2):
        raise IndexError(
            f"template2={args.template2} is out of range."
        )

    template1 = rows1.iloc[args.template1]
    template2 = rows2.iloc[args.template2]

    print(
        "Slice 1 template sample_id:",
        int(template1["sample_id"]),
    )

    print(
        "Slice 2 template sample_id:",
        int(template2["sample_id"]),
    )

    # --------------------------------------------------------
    # Load scene for cameras and point-cloud coordinates.
    # --------------------------------------------------------
    scene_args = lp.extract(args)

    if not scene_args.model_path:
        scene_args.model_path = str(
            REPO_DIR / "output" / "two_slice_scene_test"
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

    if not 0 <= args.camera_index < len(cameras):
        raise IndexError(
            f"camera_index={args.camera_index} is invalid. "
            f"Available cameras: {len(cameras)}"
        )

    # Compute scene-specific default center.
    scene_center = compute_scene_median_center(
        gaussian_model
    ).to(device)

    slice1_center = scene_center + torch.tensor(
        [
            args.slice1_dx,
            args.slice1_dy,
            args.slice1_dz,
        ],
        dtype=torch.float32,
        device=device,
    )

    slice2_center = scene_center + torch.tensor(
        [
            args.slice2_dx,
            args.slice2_dy,
            args.slice2_dz,
        ],
        dtype=torch.float32,
        device=device,
    )

    print(
        "Scene median center:",
        scene_center.detach().cpu().numpy(),
    )

    print(
        "Slice 1 center:",
        slice1_center.detach().cpu().numpy(),
    )

    print(
        "Slice 2 center:",
        slice2_center.detach().cpu().numpy(),
    )

    # --------------------------------------------------------
    # Create ordered slice descriptions.
    # --------------------------------------------------------
    slice_specs = {
        "slice1": {
            "mode": args.mode1,
            "model": model1,
            "param_mean": mean1,
            "param_std": std1,
            "template": template1,
            "center": slice1_center,
            "width": args.slice1_width,
            "height": args.slice1_height,
        },
        "slice2": {
            "mode": args.mode2,
            "model": model2,
            "param_mean": mean2,
            "param_std": std2,
            "template": template2,
            "center": slice2_center,
            "width": args.slice2_width,
            "height": args.slice2_height,
        },
    }

    order_names = args.order.split(",")

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

            canvas_h = int(viewpoint_camera.image_height)
            canvas_w = int(viewpoint_camera.image_width)

            individual_layers = {}

            for slice_name in ["slice1", "slice2"]:
                spec = slice_specs[slice_name]

                # --------------------------------------------
                # Camera pose relative to this slice.
                # --------------------------------------------
                phi, theta, actual_radius, relative = \
                    camera_to_slice_pose(
                        viewpoint_camera,
                        spec["center"],
                    )

                reconstructed = reconstruct_camera_position(
                    phi,
                    theta,
                    actual_radius,
                )

                reconstruction_error = torch.linalg.norm(
                    reconstructed - relative
                ).item()

                # --------------------------------------------
                # Build FNO input.
                # --------------------------------------------
                param_vec = build_fno_param_vector(
                    dataset=fno_dataset,
                    template_row=spec["template"],
                    mode=spec["mode"],
                    phi=phi,
                    theta=theta,
                    fno_radius=args.fno_radius,
                    param_mean=spec["param_mean"],
                    param_std=spec["param_std"],
                    device=device,
                )

                prediction = spec["model"](param_vec)
                prediction = prediction.clamp(0.0, 1.0)

                if prediction.shape != (1, 4, 64, 64):
                    raise RuntimeError(
                        f"{slice_name} produced unexpected shape: "
                        f"{tuple(prediction.shape)}"
                    )

                # --------------------------------------------
                # Project slice and determine screen footprint.
                # --------------------------------------------
                bbox = projected_patch_bbox(
                    viewpoint_camera=viewpoint_camera,
                    slice_center=spec["center"],
                    world_width=spec["width"],
                    world_height=spec["height"],
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

                resized_patch = resize_neural_patch(
                    prediction,
                    width_px=projected_width,
                    height_px=projected_height,
                )

                # --------------------------------------------
                # Place this layer on a full-size transparent
                # canvas.
                # --------------------------------------------
                layer_canvas = torch.zeros(
                    1,
                    4,
                    canvas_h,
                    canvas_w,
                    dtype=prediction.dtype,
                    device=device,
                )

                layer_canvas = paste_patch(
                    canvas=layer_canvas,
                    patch=resized_patch,
                    center_x=projected_center_x,
                    center_y=projected_center_y,
                )

                individual_layers[slice_name] = layer_canvas

                # Save individual layer.
                layer_np = layer_canvas[0].detach().cpu().numpy()

                layer_rgba_path = output_dir / (
                    f"{slice_name}_{spec['mode']}_"
                    f"camera_{camera_index:04d}.png"
                )

                layer_alpha_path = output_dir / (
                    f"{slice_name}_{spec['mode']}_alpha_"
                    f"camera_{camera_index:04d}.png"
                )

                save_rgba_chw(
                    layer_np,
                    layer_rgba_path,
                )

                save_alpha_chw(
                    layer_np[3],
                    layer_alpha_path,
                )

                print(
                    f"{slice_name} camera={camera_index} "
                    f"mode={spec['mode']} "
                    f"actual_radius={actual_radius.item():.6f} "
                    f"phi={phi.item():.6f} "
                    f"theta={theta.item():.6f} "
                    f"reconstruction_error="
                    f"{reconstruction_error:.8e} "
                    f"center=("
                    f"{projected_center_x:.2f},"
                    f"{projected_center_y:.2f}) "
                    f"size=("
                    f"{projected_width:.2f}x"
                    f"{projected_height:.2f})"
                )

            # ------------------------------------------------
            # Composite layers.
            # ------------------------------------------------
            layers_back_to_front = [
                individual_layers[name]
                for name in order_names
            ]

            composite = composite_layers_back_to_front(
                layers=layers_back_to_front,
                device=device,
            )

            composite_np = composite[0].detach().cpu().numpy()

            composite_rgb_path = output_dir / (
                f"composite_rgb_camera_{camera_index:04d}.png"
            )

            composite_rgba_path = output_dir / (
                f"composite_rgba_camera_{camera_index:04d}.png"
            )

            composite_alpha_path = output_dir / (
                f"composite_alpha_camera_{camera_index:04d}.png"
            )

            save_rgb_chw(
                composite_np[:3],
                composite_rgb_path,
            )

            save_rgba_chw(
                composite_np,
                composite_rgba_path,
            )

            save_alpha_chw(
                composite_np[3],
                composite_alpha_path,
            )

            # ------------------------------------------------
            # Save target image from the 3DGS scene.
            # ------------------------------------------------
            target = viewpoint_camera.original_image.detach()

            target_path = output_dir / (
                f"target_camera_{camera_index:04d}.png"
            )

            save_rgb_chw(
                target,
                target_path,
            )

            # ------------------------------------------------
            # Save target vs composite side-by-side.
            # ------------------------------------------------
            target_np = target.cpu().numpy()
            target_hwc = np.transpose(
                target_np,
                (1, 2, 0),
            )
            target_hwc = np.clip(
                target_hwc,
                0.0,
                1.0,
            )

            composite_hwc = np.transpose(
                composite_np[:3],
                (1, 2, 0),
            )
            composite_hwc = np.clip(
                composite_hwc,
                0.0,
                1.0,
            )

            if target_hwc.shape == composite_hwc.shape:
                comparison = np.concatenate(
                    [
                        target_hwc,
                        composite_hwc,
                    ],
                    axis=1,
                )

                comparison_path = output_dir / (
                    f"comparison_camera_{camera_index:04d}.png"
                )

                imageio.imwrite(
                    comparison_path,
                    (comparison * 255.0 + 0.5).astype(np.uint8),
                )
            else:
                print(
                    "[WARN] Target/composite shape mismatch:",
                    target_hwc.shape,
                    composite_hwc.shape,
                )

            print(
                "Saved composite:",
                composite_rgb_path,
            )

    print("Done.")
    print("Outputs written to:", output_dir)


if __name__ == "__main__":
    main()