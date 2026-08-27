#!/usr/bin/env python

import sys
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

from neural_patch_utils import (
    compute_scene_median_center,
    camera_to_slice_pose,
    projected_patch_bbox,
    place_patch_differentiable,
    load_fno_checkpoint,
    build_fno_param_vector,
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
    REPO_DIR / "real_scene_differentiable_outputs"
)

IMG_SIZE = (64, 64)
DEFAULT_FNO_RADIUS = 2.2


# ============================================================
# SAVING HELPERS
# ============================================================

def save_chw_rgb(array_chw, path):
    """
    Save [3,H,W] RGB data as uint8 PNG.
    """
    if torch.is_tensor(array_chw):
        array_chw = array_chw.detach().cpu().numpy()

    image_hwc = np.transpose(array_chw, (1, 2, 0))
    image_hwc = np.clip(image_hwc, 0.0, 1.0)

    imageio.imwrite(
        path,
        (image_hwc * 255.0 + 0.5).astype(np.uint8),
    )


def save_chw_rgba(array_chw, path):
    """
    Save [4,H,W] RGBA data as uint8 PNG.

    The RGB channels are premultiplied.
    """
    if torch.is_tensor(array_chw):
        array_chw = array_chw.detach().cpu().numpy()

    image_hwc = np.transpose(array_chw, (1, 2, 0))
    image_hwc = np.clip(image_hwc, 0.0, 1.0)

    imageio.imwrite(
        path,
        (image_hwc * 255.0 + 0.5).astype(np.uint8),
    )


def save_alpha(array_hw, path):
    """
    Save [H,W] alpha as grayscale RGB.
    """
    if torch.is_tensor(array_hw):
        array_hw = array_hw.detach().cpu().numpy()

    array_hw = np.clip(array_hw, 0.0, 1.0)
    image_rgb = np.repeat(array_hw[..., None], 3, axis=2)

    imageio.imwrite(
        path,
        (image_rgb * 255.0 + 0.5).astype(np.uint8),
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = ArgumentParser(
        description=(
            "Run a pretrained FNO using one camera from a 3DGS "
            "scene, then differentiably place the neural patch."
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
        help="Single-mode FNO checkpoint.",
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
        help="World-space width of the neural slice.",
    )

    parser.add_argument(
        "--slice_height",
        type=float,
        default=1.0,
        help="World-space height of the neural slice.",
    )

    parser.add_argument(
        "--slice_dx",
        type=float,
        default=0.0,
        help="Slice X offset from scene median.",
    )

    parser.add_argument(
        "--slice_dy",
        type=float,
        default=0.0,
        help="Slice Y offset from scene median.",
    )

    parser.add_argument(
        "--slice_dz",
        type=float,
        default=0.0,
        help="Slice Z offset from scene median.",
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
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Using device:", device)
    print("Mode:", args.mode)
    print("FNO radius:", args.fno_radius)

    # --------------------------------------------------------
    # Select checkpoint.
    # --------------------------------------------------------
    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint)
    elif args.mode == "surface":
        checkpoint_path = DEFAULT_SURFACE_CHECKPOINT
    else:
        checkpoint_path = DEFAULT_VOLUME_CHECKPOINT

    # --------------------------------------------------------
    # Load FNO metadata.
    #
    # This gives us the exact training parameter construction
    # and normalization context.
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
    # Load FNO model and checkpoint normalization.
    # --------------------------------------------------------
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
    #
    # This is the neural slice content. Camera parameters
    # will be replaced below.
    # --------------------------------------------------------
    mode_rows = fno_dataset.df[
        fno_dataset.df["render_mode"].astype(str) == args.mode
    ].reset_index(drop=True)

    if len(mode_rows) == 0:
        raise RuntimeError(
            f"No FNO metadata rows found for mode={args.mode}."
        )

    if not 0 <= args.template_index < len(mode_rows):
        raise IndexError(
            f"template_index={args.template_index} is out of range. "
            f"Available rows: {len(mode_rows)}."
        )

    template_row = mode_rows.iloc[args.template_index]
    template_sample_id = int(template_row["sample_id"])

    print(
        "Template index:",
        args.template_index,
        "template sample_id:",
        template_sample_id,
    )

    # --------------------------------------------------------
    # Load 3DGS scene only for cameras and point coordinates.
    #
    # The Gaussian model is not used to render this image.
    # The existing Scene implementation still requires it.
    # --------------------------------------------------------
    scene_args = lp.extract(args)

    if not scene_args.model_path:
        scene_args.model_path = str(
            REPO_DIR / "output" / "real_scene_neural_test"
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
            f"Available cameras: {len(cameras)}."
        )

    viewpoint_camera = cameras[args.camera_index]

    # --------------------------------------------------------
    # Compute scene-specific center and neural slice center.
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
        "Neural slice center:",
        slice_center.detach().cpu().numpy(),
    )

    # --------------------------------------------------------
    # Compute camera direction relative to the slice.
    #
    # actual_radius is retained for diagnostics. The FNO
    # receives the canonical radius args.fno_radius.
    # --------------------------------------------------------
    phi, theta, actual_radius, relative = \
        camera_to_slice_pose(
            viewpoint_camera,
            slice_center,
        )

    # --------------------------------------------------------
    # Build the exact normalized FNO parameter vector.
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
            f"Unexpected parameter-vector shape: {param_vec.shape}"
        )

    # --------------------------------------------------------
    # FNO inference.
    # --------------------------------------------------------
    with torch.no_grad():
        prediction = model(param_vec).clamp(0.0, 1.0)

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
    # Project the slice into the actual camera.
    # --------------------------------------------------------
    bbox = projected_patch_bbox(
        viewpoint_camera=viewpoint_camera,
        slice_center=slice_center,
        world_width=args.slice_width,
        world_height=args.slice_height,
    )

    center_x = bbox["center_x"]
    center_y = bbox["center_y"]
    width_px = bbox["width_px"]
    height_px = bbox["height_px"]

    print(
        "Projected center:",
        float(center_x.detach().cpu().item()),
        float(center_y.detach().cpu().item()),
    )

    print(
        "Projected size:",
        float(width_px.detach().cpu().item()),
        float(height_px.detach().cpu().item()),
    )

    # --------------------------------------------------------
    # Differentiably place patch on full camera canvas.
    #
    # No optimization is performed here. This tests the
    # real-scene forward path using differentiable placement.
    # --------------------------------------------------------
    canvas_h = int(viewpoint_camera.image_height)
    canvas_w = int(viewpoint_camera.image_width)

    placed = place_patch_differentiable(
        patch=prediction,
        canvas_height=canvas_h,
        canvas_width=canvas_w,
        center_x=center_x,
        center_y=center_y,
        patch_width=width_px,
        patch_height=height_px,
    )

    placed = placed.clamp(0.0, 1.0)
    placed_np = placed[0].detach().cpu().numpy()

    # --------------------------------------------------------
    # Save FNO patch.
    # --------------------------------------------------------
    patch_np = prediction[0].detach().cpu().numpy()

    patch_rgba_path = output_dir / (
        f"fno_patch_{args.mode}_camera_{args.camera_index:04d}.png"
    )

    patch_rgb_path = output_dir / (
        f"fno_patch_rgb_{args.mode}_camera_{args.camera_index:04d}.png"
    )

    patch_alpha_path = output_dir / (
        f"fno_patch_alpha_{args.mode}_camera_{args.camera_index:04d}.png"
    )

    save_chw_rgba(
        patch_np,
        patch_rgba_path,
    )

    save_chw_rgb(
        patch_np[:3],
        patch_rgb_path,
    )

    save_alpha(
        patch_np[3],
        patch_alpha_path,
    )

    # --------------------------------------------------------
    # Save placed output.
    # --------------------------------------------------------
    placed_rgba_path = output_dir / (
        f"placed_rgba_{args.mode}_camera_{args.camera_index:04d}.png"
    )

    placed_rgb_path = output_dir / (
        f"placed_rgb_{args.mode}_camera_{args.camera_index:04d}.png"
    )

    placed_alpha_path = output_dir / (
        f"placed_alpha_{args.mode}_camera_{args.camera_index:04d}.png"
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
    # Save original scene target image for reference.
    #
    # This is not expected to match the FNO patch because
    # the selected FNO template is arbitrary.
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
    # Save side-by-side target and placed FNO output.
    # --------------------------------------------------------
    target_np = target.cpu().numpy()
    target_hwc = np.transpose(target_np, (1, 2, 0))
    target_hwc = np.clip(target_hwc, 0.0, 1.0)

    placed_hwc = np.transpose(
        placed_np[:3],
        (1, 2, 0),
    )
    placed_hwc = np.clip(placed_hwc, 0.0, 1.0)

    if target_hwc.shape == placed_hwc.shape:
        comparison = np.concatenate(
            [
                target_hwc,
                placed_hwc,
            ],
            axis=1,
        )

        comparison_path = output_dir / (
            f"comparison_{args.mode}_camera_{args.camera_index:04d}.png"
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
        "Actual camera-to-slice radius:",
        float(actual_radius.detach().cpu().item()),
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
        "Camera image size:",
        canvas_w,
        "x",
        canvas_h,
    )

    print("Saved patch:", patch_rgba_path)
    print("Saved placed RGBA:", placed_rgba_path)
    print("Saved placed RGB:", placed_rgb_path)
    print("Saved placed alpha:", placed_alpha_path)
    print("Saved target:", target_path)

    print("Done.")
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()