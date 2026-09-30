#!/usr/bin/env python
"""
Evaluate a fixed voxel neural scene on 3DGS training cameras.

Outputs:
    output_dir/
    ├── per_camera_metrics.csv
    ├── summary.json
    └── previews/                  # optional
        ├── camera_0000_prediction.png
        ├── camera_0000_target.png
        └── camera_0000_comparison.png
"""

from __future__ import annotations

import json
import math
import sys
from argparse import ArgumentParser
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim


# ============================================================
# PATH SETUP
# ============================================================

REPO_DIR = Path(__file__).resolve().parent
FNO_ROOT = REPO_DIR.parent

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

if str(FNO_ROOT) not in sys.path:
    sys.path.insert(0, str(FNO_ROOT))


# ============================================================
# FIXED VOXEL CHECKPOINT RECONSTRUCTION
# ============================================================

def rebuild_fixed_voxel_scene_from_checkpoint(
    checkpoint_data,
    device,
    optimize_sh=False,
):
    """
    Rebuild exact voxel topology from checkpoint block_specs, then restore
    current learned scene parameters from scene_state.
    """
    from fixed_voxel_parameters import FixedVoxelBlock
    from fixed_voxel_scene import (
        FixedVoxelScene,
        module_name_from_key,
        voxel_key,
    )

    required = {
        "scene_state",
        "block_specs",
        "root_voxel_size",
    }

    missing = required - set(checkpoint_data.keys())

    if missing:
        raise KeyError(
            f"Checkpoint is missing required keys: {sorted(missing)}"
        )

    scene_state = checkpoint_data["scene_state"]
    blocks = []

    for spec in checkpoint_data["block_specs"]:
        level = int(spec["level"])
        grid_index = tuple(
            int(value)
            for value in spec["grid_index"]
        )

        parent_key = spec.get("parent_key")

        if parent_key is not None:
            parent_key = tuple(
                int(value)
                for value in parent_key
            )

        logical_key = voxel_key(level, grid_index)
        module_name = module_name_from_key(logical_key)
        prefix = f"blocks.{module_name}."

        block = FixedVoxelBlock(
            center=scene_state[prefix + "center"],
            world_size=float(
                scene_state[prefix + "world_size"]
            ),
            ctrl_values=scene_state[prefix + "initial_ctrl"],
            sigma=float(
                scene_state[prefix + "initial_sigma"]
            ),
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
            metallic=float(
                scene_state[prefix + "metallic_value"]
            ),
            specular=float(
                scene_state[prefix + "specular_value"]
            ),
            level=level,
            grid_index=grid_index,
            parent_key=parent_key,
            # Exact current values are restored immediately below.
            initial_alive_probability=0.99,
            initial_volume_probability=0.5,
            optimize_environment=bool(optimize_sh),
        ).to(device)

        blocks.append(block)

    global_sh_bound = float(
        checkpoint_data.get(
            "metadata",
            {},
        ).get(
            "global_sh_bound",
            0.005,
        )
    )

    voxel_scene = FixedVoxelScene(
        blocks=blocks,
        initial_global_sh=scene_state[
            "lighting.initial_global_sh"
        ],
        optimize_sh=bool(optimize_sh),
        global_sh_bound=global_sh_bound,
        root_voxel_size=float(
            checkpoint_data["root_voxel_size"]
        ),
    ).to(device)

    missing_keys, unexpected_keys = voxel_scene.load_state_dict(
        scene_state,
        strict=False,
    )

    if missing_keys:
        print("[WARN] Missing scene keys during evaluation:")
        for key in missing_keys:
            print("  ", key)

    if unexpected_keys:
        print("[WARN] Unexpected scene keys during evaluation:")
        for key in unexpected_keys:
            print("  ", key)

    return voxel_scene


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    predicted_rgb,
    target_rgb,
    l1_weight=0.4,
    mse_weight=0.4,
    ssim_weight=0.2,
):
    """
    Inputs:
        predicted_rgb: [1, 3, H, W], float in [0, 1]
        target_rgb:    [1, 3, H, W], float in [0, 1]

    Returns a Python dictionary.
    """
    predicted_rgb = predicted_rgb.clamp(0.0, 1.0)
    target_rgb = target_rgb.clamp(0.0, 1.0)

    l1_value = F.l1_loss(
        predicted_rgb,
        target_rgb,
    )

    mse_value = F.mse_loss(
        predicted_rgb,
        target_rgb,
    )

    ssim_value = ssim(
        predicted_rgb,
        target_rgb,
        data_range=1.0,
        size_average=True,
    )

    psnr_value = -10.0 * torch.log10(
        mse_value.clamp_min(1e-10)
    )

    mixed_loss = (
        float(l1_weight) * l1_value
        + float(mse_weight) * mse_value
        + float(ssim_weight) * (1.0 - ssim_value)
    )

    return {
        "l1": float(l1_value.item()),
        "mse": float(mse_value.item()),
        "psnr": float(psnr_value.item()),
        "ssim": float(ssim_value.item()),
        "mixed_loss": float(mixed_loss.item()),
    }


# ============================================================
# IMAGE SAVING
# ============================================================

def save_rgb_tensor(rgb_chw, path):
    """Save RGB [3,H,W] float tensor/array as PNG."""
    if torch.is_tensor(rgb_chw):
        rgb_chw = rgb_chw.detach().cpu().numpy()

    rgb_hwc = np.transpose(rgb_chw, (1, 2, 0))
    rgb_hwc = np.clip(rgb_hwc, 0.0, 1.0)

    imageio.imwrite(
        path,
        (rgb_hwc * 255.0 + 0.5).astype(np.uint8),
    )


def save_preview(
    prediction_rgb,
    target_rgb,
    output_dir,
    camera_index,
    camera_name,
):
    """
    Save prediction, target, and side-by-side comparison.
    """
    safe_name = str(camera_name).replace("/", "_").replace(" ", "_")

    prediction_path = output_dir / (
        f"camera_{camera_index:04d}_{safe_name}_prediction.png"
    )
    target_path = output_dir / (
        f"camera_{camera_index:04d}_{safe_name}_target.png"
    )
    comparison_path = output_dir / (
        f"camera_{camera_index:04d}_{safe_name}_comparison.png"
    )

    save_rgb_tensor(prediction_rgb, prediction_path)
    save_rgb_tensor(target_rgb, target_path)

    prediction_hwc = np.transpose(
        prediction_rgb.detach().cpu().numpy(),
        (1, 2, 0),
    )

    target_hwc = np.transpose(
        target_rgb.detach().cpu().numpy(),
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
        comparison_path,
        (comparison * 255.0 + 0.5).astype(np.uint8),
    )


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = ArgumentParser(
        description=(
            "Evaluate a fixed voxel neural-scene checkpoint over actual "
            "3DGS training cameras."
        )
    )

    parser.add_argument(
        "--scene_checkpoint",
        type=Path,
        required=True,
        help="Checkpoint from optimize_fixed_voxel_scene.py",
    )

    parser.add_argument(
        "--surface_checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--volume_checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--camera_start",
        type=int,
        default=0,
        help="First training camera index to evaluate.",
    )

    parser.add_argument(
        "--num_cameras",
        type=int,
        default=0,
        help="Number of cameras to evaluate. Use 0 for all remaining cameras.",
    )

    parser.add_argument(
        "--fno_batch_size",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--use_tile_renderer",
        action="store_true",
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
        "--camera_depth_sign",
        type=float,
        default=1.0,
        help=(
            "Must match the depth sign used during fixed voxel optimization."
        ),
    )

    parser.add_argument(
        "--l1_weight",
        type=float,
        default=0.4,
    )

    parser.add_argument(
        "--mse_weight",
        type=float,
        default=0.4,
    )

    parser.add_argument(
        "--ssim_weight",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--save_previews",
        type=int,
        default=0,
        help=(
            "Save this many evenly spaced prediction/target comparison PNGs. "
            "Use 0 to skip PNG previews."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./fixed_voxel_evaluation"),
    )

    return parser


# ============================================================
# MAIN
# ============================================================

def main():
    parser = parse_args()

    # Registers --source_path, --model_path, --resolution, etc.
    from arguments import ModelParams
    from scene import GaussianModel, Scene

    lp = ModelParams(parser)
    args = parser.parse_args()

    weights_total = (
        args.l1_weight
        + args.mse_weight
        + args.ssim_weight
    )

    if abs(weights_total - 1.0) > 1e-6:
        raise ValueError(
            "l1_weight + mse_weight + ssim_weight must equal 1.0, "
            f"got {weights_total:.6f}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluator.")

    if not args.scene_checkpoint.is_file():
        raise FileNotFoundError(
            f"Scene checkpoint not found:\n  {args.scene_checkpoint}"
        )

    device = torch.device("cuda")

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Using device:", device)
    print("Scene checkpoint:", args.scene_checkpoint)
    print("Output directory:", args.output_dir)

    # Import optimizer module to reuse FNO checkpoint loading and
    # premultiplied-RGBA compositing background behavior.
    import optimize_fixed_voxel_scene as optimizer_module
    from fixed_voxel_renderer import FixedVoxelRenderer

    # --------------------------------------------------------
    # Load frozen FNOs.
    # --------------------------------------------------------

    surface_bundle = optimizer_module.load_fno_checkpoint(
        args.surface_checkpoint,
        device=device,
        expected_mode="surface",
    )

    volume_bundle = optimizer_module.load_fno_checkpoint(
        args.volume_checkpoint,
        device=device,
        expected_mode="volume",
    )

    # --------------------------------------------------------
    # Restore voxel scene.
    # --------------------------------------------------------

    checkpoint_data = torch.load(
        args.scene_checkpoint,
        map_location=device,
        weights_only=False,
    )

    optimize_sh = bool(
        checkpoint_data.get(
            "metadata",
            {},
        ).get(
            "optimize_sh",
            False,
        )
    )

    voxel_scene = rebuild_fixed_voxel_scene_from_checkpoint(
        checkpoint_data=checkpoint_data,
        device=device,
        optimize_sh=optimize_sh,
    )

    voxel_scene.eval()

    fno_radius = float(
        checkpoint_data.get(
            "metadata",
            {},
        ).get(
            "fno_radius",
            2.2,
        )
    )

    renderer = FixedVoxelRenderer(
        surface_bundle=surface_bundle,
        volume_bundle=volume_bundle,
        fno_radius=fno_radius,
        fno_batch_size=args.fno_batch_size,
        flip_projection_y=args.flip_projection_y,
        flip_fno_vertical=args.flip_fno_vertical,
        use_tile_renderer=args.use_tile_renderer,
        alive_cull_threshold=0.0,
        use_fno_activation_checkpointing=False,
        camera_depth_sign=args.camera_depth_sign,
    ).to(device)

    renderer.eval()

    print(f"Rebuilt fixed voxel scene: {len(voxel_scene)} blocks")

    # --------------------------------------------------------
    # Load 3DGS training cameras.
    # --------------------------------------------------------

    scene_args = lp.extract(args)

    gaussian_model = GaussianModel(args.sh_degree)

    gs_scene = Scene(
        scene_args,
        gaussian_model,
        shuffle=False,
        resolution_scales=[1.0],
    )

    all_cameras = gs_scene.getTrainCameras(scale=1.0)

    if args.camera_start < 0 or args.camera_start >= len(all_cameras):
        raise ValueError(
            f"camera_start={args.camera_start} is invalid for "
            f"{len(all_cameras)} cameras."
        )

    cameras = all_cameras[args.camera_start:]

    if args.num_cameras > 0:
        cameras = cameras[:args.num_cameras]

    if not cameras:
        raise RuntimeError("No cameras selected for evaluation.")

    print(f"Evaluating {len(cameras)} training cameras.")

    # Select evenly spaced preview indices.
    if args.save_previews > 0:
        preview_count = min(args.save_previews, len(cameras))

        preview_indices = set(
            np.linspace(
                0,
                len(cameras) - 1,
                num=preview_count,
                dtype=np.int64,
            ).tolist()
        )
    else:
        preview_indices = set()

    # --------------------------------------------------------
    # Evaluate all selected cameras.
    # --------------------------------------------------------

    rows = []

    with torch.no_grad():
        for local_index, camera in enumerate(cameras):
            predicted_rgba, _ = renderer(
                camera,
                voxel_scene,
            )

            predicted_rgba = predicted_rgba.clamp(0.0, 1.0)

            predicted_rgb = optimizer_module.visible_rgb_from_rgba(
                predicted_rgba
            ).clamp(0.0, 1.0)

            target_rgb = camera.original_image.detach()

            if target_rgb.ndim == 3:
                target_rgb_batch = target_rgb.unsqueeze(0)
            else:
                target_rgb_batch = target_rgb

            target_rgb_batch = target_rgb_batch.to(
                device=device,
                dtype=predicted_rgb.dtype,
            ).clamp(0.0, 1.0)

            metrics = compute_metrics(
                predicted_rgb=predicted_rgb,
                target_rgb=target_rgb_batch,
                l1_weight=args.l1_weight,
                mse_weight=args.mse_weight,
                ssim_weight=args.ssim_weight,
            )

            row = {
                "camera_index": args.camera_start + local_index,
                "camera_name": str(camera.image_name),
                "width": int(camera.image_width),
                "height": int(camera.image_height),
                **metrics,
            }

            rows.append(row)

            if local_index in preview_indices:
                preview_dir = args.output_dir / "previews"
                preview_dir.mkdir(parents=True, exist_ok=True)

                save_preview(
                    prediction_rgb=predicted_rgb[0],
                    target_rgb=target_rgb_batch[0],
                    output_dir=preview_dir,
                    camera_index=args.camera_start + local_index,
                    camera_name=camera.image_name,
                )

            print(
                f"[{local_index + 1:04d}/{len(cameras)}] "
                f"camera={camera.image_name} "
                f"L1={metrics['l1']:.6f} "
                f"MSE={metrics['mse']:.6f} "
                f"PSNR={metrics['psnr']:.3f} "
                f"SSIM={metrics['ssim']:.5f}"
            )

    # --------------------------------------------------------
    # Save results.
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(rows)

    per_camera_csv = args.output_dir / "per_camera_metrics.csv"
    metrics_df.to_csv(per_camera_csv, index=False)

    summary = {
        "scene_checkpoint": str(args.scene_checkpoint),
        "surface_checkpoint": str(args.surface_checkpoint),
        "volume_checkpoint": str(args.volume_checkpoint),
        "num_cameras": int(len(metrics_df)),
        "mean_l1": float(metrics_df["l1"].mean()),
        "mean_mse": float(metrics_df["mse"].mean()),
        "mean_psnr": float(metrics_df["psnr"].mean()),
        "mean_ssim": float(metrics_df["ssim"].mean()),
        "mean_mixed_loss": float(metrics_df["mixed_loss"].mean()),
        "median_psnr": float(metrics_df["psnr"].median()),
        "median_ssim": float(metrics_df["ssim"].median()),
        "camera_depth_sign": float(args.camera_depth_sign),
        "flip_projection_y": bool(args.flip_projection_y),
        "flip_fno_vertical": bool(args.flip_fno_vertical),
        "use_tile_renderer": bool(args.use_tile_renderer),
    }

    summary_path = args.output_dir / "summary.json"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("Evaluation complete")
    print(f"Cameras:         {summary['num_cameras']}")
    print(f"Mean L1:         {summary['mean_l1']:.8f}")
    print(f"Mean MSE:        {summary['mean_mse']:.8f}")
    print(f"Mean PSNR:       {summary['mean_psnr']:.4f} dB")
    print(f"Mean SSIM:       {summary['mean_ssim']:.6f}")
    print(f"Mean mixed loss: {summary['mean_mixed_loss']:.8f}")
    print("Per-camera CSV:", per_camera_csv)
    print("Summary JSON:   ", summary_path)
    print("=" * 72)


if __name__ == "__main__":
    main()