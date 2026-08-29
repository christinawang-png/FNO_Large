#!/usr/bin/env python

import sys
import math
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import imageio.v2 as imageio
import torch


# ============================================================
# PATH SETUP
# ============================================================

IMG_SIZE = (64, 64)

REPO_DIR = Path(__file__).resolve().parent
FNO_ROOT = REPO_DIR.parent

# Make the FNO files and local 3DGS repository importable.
sys.path.insert(0, str(FNO_ROOT))
sys.path.insert(0, str(REPO_DIR))


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
    REPO_DIR / "real_scene_uniform_placement_outputs"
)

DEFAULT_FNO_RADIUS = 2.2


# ============================================================
# CAMERA HELPERS
# ============================================================

def camera_to_slice_pose(viewpoint_camera, slice_center):
    """
    Calculate spherical coordinates of the camera relative to
    the neural slice center.

    Returns:
        phi
        theta
        actual_radius
        relative_camera_position
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
    Project a world-space point into the 3DGS camera.

    The 3DGS code uses row-vector multiplication:

        clip = point_h @ full_proj_transform
    """
    matrix = viewpoint_camera.full_proj_transform

    point_world = point_world.to(
        device=matrix.device,
        dtype=matrix.dtype,
    )

    point_h = torch.cat([
        point_world,
        torch.ones(
            1,
            device=matrix.device,
            dtype=matrix.dtype,
        ),
    ])

    clip = point_h @ matrix

    if torch.abs(clip[3]) < 1e-8:
        raise RuntimeError(
            "Invalid homogeneous coordinate during projection."
        )

    ndc = clip[:3] / clip[3]

    image_width = float(viewpoint_camera.image_width)
    image_height = float(viewpoint_camera.image_height)

    pixel_x = (
        (ndc[0] + 1.0)
        * 0.5
        * image_width
    )

    pixel_y = (
        (1.0 - ndc[1])
        * 0.5
        * image_height
    )

    return pixel_x, pixel_y, ndc[2]


def compute_scene_median_center(gaussian_model):
    """
    Compute a scene-specific center from the initialized
    point cloud.
    """
    xyz = gaussian_model.get_xyz.detach()

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise RuntimeError(
            f"Expected point cloud shape [N,3], got {xyz.shape}"
        )

    return xyz.median(dim=0).values


# ============================================================
# FNO HELPERS
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
    Build the same normalized parameter vector used during
    single-mode FNO training.

    The template supplies shape, material, and SH values.
    Camera phi/theta/radius are replaced here.
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
        row["metallic"] = 0.0
        row["roughness"] = 0.0
        row["specular"] = 0.0

    raw_np = dataset._build_param_vector_np(row)

    if raw_np.shape[0] != param_mean.shape[0]:
        raise RuntimeError(
            f"Parameter dimension mismatch: "
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
# DIFFERENTIABLE UNIFORM PLACEMENT
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
    Uniformly scale and place a patch.

    patch:
        [B,C,H,W]

    patch_size:
        Equal output width and height in pixels.

    This uses grid_sample, so center and size remain
    differentiable if they are tensors requiring gradients.
    """
    if patch.ndim != 4:
        raise ValueError(
            f"Expected patch [B,C,H,W], got {patch.shape}"
        )

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

    canvas_w_minus_one = float(
        max(canvas_width - 1, 1)
    )

    canvas_h_minus_one = float(
        max(canvas_height - 1, 1)
    )

    center_x_norm = (
        2.0 * center_x / canvas_w_minus_one - 1.0
    )

    center_y_norm = (
        2.0 * center_y / canvas_h_minus_one - 1.0
    )

    # Same scale in both directions.
    scale_x = canvas_w_minus_one / patch_size
    scale_y = canvas_h_minus_one / patch_size

    theta = torch.zeros(
        batch_size,
        2,
        3,
        dtype=dtype,
        device=device,
    )

    theta[:, 0, 0] = scale_x
    theta[:, 1, 1] = scale_y

    theta[:, 0, 2] = (
        -scale_x * center_x_norm
    )

    theta[:, 1, 2] = (
        -scale_y * center_y_norm
    )

    grid = torch.nn.functional.affine_grid(
        theta,
        size=(
            batch_size,
            patch.shape[1],
            canvas_height,
            canvas_width,
        ),
        align_corners=True,
    )

    canvas = torch.nn.functional.grid_sample(
        patch,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    return canvas


# ============================================================
# IMAGE SAVING
# ============================================================

def save_chw_rgb(array_chw, path):
    if torch.is_tensor(array_chw):
        array_chw = array_chw.detach().cpu().numpy()

    image_hwc = np.transpose(
        array_chw,
        (1, 2, 0),
    )

    image_hwc = np.clip(
        image_hwc,
        0.0,
        1.0,
    )

    imageio.imwrite(
        path,
        (image_hwc * 255.0 + 0.5).astype(np.uint8),
    )


def save_chw_rgba(array_chw, path):
    if torch.is_tensor(array_chw):
        array_chw = array_chw.detach().cpu().numpy()

    image_hwc = np.transpose(
        array_chw,
        (1, 2, 0),
    )

    image_hwc = np.clip(
        image_hwc,
        0.0,
        1.0,
    )

    imageio.imwrite(
        path,
        (image_hwc * 255.0 + 0.5).astype(np.uint8),
    )


def save_alpha(array_hw, path):
    if torch.is_tensor(array_hw):
        array_hw = array_hw.detach().cpu().numpy()

    array_hw = np.clip(
        array_hw,
        0.0,
        1.0,
    )

    alpha_rgb = np.repeat(
        array_hw[..., None],
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
            "Run a pretrained FNO on one 3DGS camera and "
            "place the output using one uniform scale."
        )
    )

    # Provides --source_path, --model_path, --resolution, etc.
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
    )

    parser.add_argument(
        "--camera_index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--template_index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--fno_radius",
        type=float,
        default=DEFAULT_FNO_RADIUS,
    )

    parser.add_argument(
        "--slice_width",
        type=float,
        default=1.0,
        help=(
            "World-space width of the slice. "
            "The output patch remains square."
        ),
    )

    parser.add_argument(
        "--slice_dx",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--slice_dy",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--slice_dz",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--scale_multiplier",
        type=float,
        default=1.0,
        help="Extra multiplier for the uniform projected patch size.",
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
    print("Mode:", args.mode)
    print("FNO radius:", args.fno_radius)

    # --------------------------------------------------------
    # Load FNO dataset metadata.
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

    print("FNO dataset rows:", len(fno_dataset.df))
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
    # Select one template row for this mode.
    # --------------------------------------------------------
    mode_rows = fno_dataset.df[
        fno_dataset.df["render_mode"].astype(str) == args.mode
    ].reset_index(drop=True)

    if len(mode_rows) == 0:
        raise RuntimeError(
            f"No rows found for mode={args.mode}"
        )

    if not 0 <= args.template_index < len(mode_rows):
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
        "Template index:",
        args.template_index,
        "sample_id:",
        template_sample_id,
    )

    # --------------------------------------------------------
    # Load the 3DGS scene only to obtain cameras and point
    # cloud coordinates.
    # --------------------------------------------------------
    scene_args = lp.extract(args)

    if not scene_args.model_path:
        scene_args.model_path = str(
            REPO_DIR / "output" / "uniform_neural_test"
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
        scale=1.0
    )

    if len(cameras) == 0:
        raise RuntimeError(
            "No training cameras were loaded."
        )

    if not 0 <= args.camera_index < len(cameras):
        raise IndexError(
            f"camera_index={args.camera_index} is invalid. "
            f"Available cameras: {len(cameras)}."
        )

    viewpoint_camera = cameras[
        args.camera_index
    ]

    # --------------------------------------------------------
    # Compute scene-specific slice center.
    # --------------------------------------------------------
    scene_center = compute_scene_median_center(
        gaussian_model
    ).to(device)

    slice_offset = torch.tensor(
        [
            args.slice_dx,
            args.slice_dy,
            args.slice_dz,
        ],
        dtype=torch.float32,
        device=device,
    )

    slice_center = scene_center + slice_offset

    print(
        "Scene median center:",
        scene_center.detach().cpu().numpy(),
    )

    print(
        "Slice center:",
        slice_center.detach().cpu().numpy(),
    )

    # --------------------------------------------------------
    # Compute current camera pose relative to slice.
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Build FNO input with current camera angles.
    # --------------------------------------------------------
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

    if param_vec.shape != (1, latent_dim):
        raise RuntimeError(
            f"Unexpected parameter vector shape: {param_vec.shape}"
        )

    # --------------------------------------------------------
    # Run FNO.
    # --------------------------------------------------------
    with torch.no_grad():
        prediction = model(param_vec).clamp(
            0.0,
            1.0,
        )

    if prediction.shape != (1, 4, 64, 64):
        raise RuntimeError(
            f"Expected FNO output [1,4,64,64], "
            f"got {tuple(prediction.shape)}"
        )

    print(
        "FNO output range:",
        float(prediction.min()),
        float(prediction.max()),
    )

    # --------------------------------------------------------
    # Project slice center into the camera.
    # --------------------------------------------------------
    center_x, center_y, center_depth = \
        project_world_point(
            viewpoint_camera,
            slice_center,
        )

    # --------------------------------------------------------
    # Compute ONE uniform patch size.
    #
    # The FNO was trained at 64x64 and canonical radius 2.2.
    # The canvas may be larger than 64x64, so scale from the
    # vertical canvas resolution.
    # --------------------------------------------------------
    canvas_h = int(
        viewpoint_camera.image_height
    )

    canvas_w = int(
        viewpoint_camera.image_width
    )

    resolution_factor = (
        float(canvas_h) / float(IMG_SIZE[0])
    )

    uniform_scale = (
        float(args.slice_width)
        * float(args.fno_radius)
        / actual_radius
    )

    patch_size = (
        float(IMG_SIZE[0])
        * resolution_factor
        * uniform_scale
        * float(args.scale_multiplier)
    )

    patch_size = max(
        2.0,
        patch_size,
    )

    print(
        "Projected center:",
        float(center_x.detach().cpu().item()),
        float(center_y.detach().cpu().item()),
    )

    print(
        "Actual camera-to-slice radius:",
        float(actual_radius.detach().cpu().item()),
    )

    print(
        "Uniform patch size:",
        patch_size,
        "x",
        patch_size,
    )

    # --------------------------------------------------------
    # Differentiable uniform placement.
    #
    # center_x, center_y and patch_size are equal in both axes.
    # --------------------------------------------------------
    placed = place_patch_uniform(
        patch=prediction,
        canvas_height=canvas_h,
        canvas_width=canvas_w,
        center_x=center_x,
        center_y=center_y,
        patch_size=patch_size,
    )

    placed = placed.clamp(
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Save FNO patch.
    # --------------------------------------------------------
    prediction_np = prediction[0].detach().cpu().numpy()

    patch_rgba_path = output_dir / (
        f"fno_patch_{args.mode}_camera_"
        f"{args.camera_index:04d}.png"
    )

    patch_rgb_path = output_dir / (
        f"fno_patch_rgb_{args.mode}_camera_"
        f"{args.camera_index:04d}.png"
    )

    patch_alpha_path = output_dir / (
        f"fno_patch_alpha_{args.mode}_camera_"
        f"{args.camera_index:04d}.png"
    )

    save_chw_rgba(
        prediction_np,
        patch_rgba_path,
    )

    save_chw_rgb(
        prediction_np[:3],
        patch_rgb_path,
    )

    save_alpha(
        prediction_np[3],
        patch_alpha_path,
    )

    # --------------------------------------------------------
    # Save placed result.
    # --------------------------------------------------------
    placed_np = placed[0].detach().cpu().numpy()

    placed_rgba_path = output_dir / (
        f"placed_rgba_{args.mode}_camera_"
        f"{args.camera_index:04d}.png"
    )

    placed_rgb_path = output_dir / (
        f"placed_rgb_{args.mode}_camera_"
        f"{args.camera_index:04d}.png"
    )

    placed_alpha_path = output_dir / (
        f"placed_alpha_{args.mode}_camera_"
        f"{args.camera_index:04d}.png"
    )

    save_chw_rgba(
        placed_np,
        placed_rgba_path,
    )

    save_chw_rgb(
        placed_np[:3],
        placed_rgb_path,
    )

    save_alpha(
        placed_np[3],
        placed_alpha_path,
    )

    # --------------------------------------------------------
    # Save target image for reference.
    # --------------------------------------------------------
    target = viewpoint_camera.original_image.detach()

    target_path = output_dir / (
        f"target_camera_{args.camera_index:04d}.png"
    )

    save_chw_rgb(
        target,
        target_path,
    )

    # --------------------------------------------------------
    # Save target and placed output side-by-side.
    # --------------------------------------------------------
    target_np = target.cpu().numpy()
    target_hwc = np.transpose(
        target_np,
        (1, 2, 0),
    )

    placed_hwc = np.transpose(
        placed_np[:3],
        (1, 2, 0),
    )

    target_hwc = np.clip(
        target_hwc,
        0.0,
        1.0,
    )

    placed_hwc = np.clip(
        placed_hwc,
        0.0,
        1.0,
    )

    if target_hwc.shape == placed_hwc.shape:
        comparison = np.concatenate(
            [
                target_hwc,
                placed_hwc,
            ],
            axis=1,
        )

        comparison_path = output_dir / (
            f"comparison_{args.mode}_camera_"
            f"{args.camera_index:04d}.png"
        )

        imageio.imwrite(
            comparison_path,
            (comparison * 255.0 + 0.5).astype(np.uint8),
        )

        print("Saved comparison:", comparison_path)

    # --------------------------------------------------------
    # Diagnostics.
    # --------------------------------------------------------
    print(
        "Camera:",
        viewpoint_camera.image_name,
    )

    print(
        "phi:",
        float(phi.detach().cpu().item()),
    )

    print(
        "theta:",
        float(theta.detach().cpu().item()),
    )

    print(
        "reconstruction_error:",
        reconstruction_error,
    )

    print(
        "Camera image size:",
        canvas_w,
        "x",
        canvas_h,
    )

    print("Saved:", patch_rgba_path)
    print("Saved:", placed_rgba_path)
    print("Saved:", placed_rgb_path)
    print("Saved:", placed_alpha_path)
    print("Saved:", target_path)

    print("Done.")
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()