#!/usr/bin/env python

import sys
import math
from pathlib import Path
from argparse import ArgumentParser

import imageio.v2 as imageio
import numpy as np
import torch


# ============================================================
# PATH SETUP
# ============================================================

REPO_DIR = Path(__file__).resolve().parent
FNO_ROOT = REPO_DIR.parent

sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(FNO_ROOT))


# ============================================================
# IMPORT EXISTING PROJECT COMPONENTS
# ============================================================

from scene import Scene, GaussianModel
from arguments import ModelParams

# Your current scene optimizer must provide these.
from optimize_neural_scene_random_init import (
    FNO_RADIUS,
    NUM_GLOBAL_ENVS,
    SH_ORDER,
    sh_for_global_env,
    load_fno_checkpoint,
    rebuild_neural_scene_from_checkpoint,
    NeuralSceneRenderer,
)


# ============================================================
# DEFAULT PATHS
# ============================================================

SURFACE_CHECKPOINT = (
    FNO_ROOT / "fno_premult_surface_epoch128_color.pt"
)

VOLUME_CHECKPOINT = (
    FNO_ROOT / "fno_premult_volume_epoch016_color.pt"
)


# ============================================================
# HELPERS
# ============================================================

def build_sh_bank(device):
    """
    Recreate the same procedural SH environment bank used while
    generating Blender render data.

    Returns:
        [NUM_GLOBAL_ENVS, 27] for SH order 2.
    """
    rows = []

    for env_id in range(NUM_GLOBAL_ENVS):
        coeffs = sh_for_global_env(
            env_id,
            order=SH_ORDER,
        )  # [9,3]

        rows.append(coeffs.reshape(-1))

    return torch.tensor(
        np.stack(rows, axis=0),
        dtype=torch.float32,
        device=device,
    )


def rgba_to_visible_rgb(rgba, background_rgb):
    """
    rgba:
        [1,4,H,W], premultiplied RGB + alpha.

    Returns:
        float NumPy image [H,W,3] in [0,1].
    """
    background = torch.tensor(
        background_rgb,
        dtype=rgba.dtype,
        device=rgba.device,
    ).view(1, 3, 1, 1)

    visible = (
        rgba[:, :3]
        + (1.0 - rgba[:, 3:4]) * background
    )

    image = (
        visible[0]
        .detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    return np.clip(image, 0.0, 1.0)


def alpha_to_rgb(alpha):
    """
    alpha:
        [H,W] float array.

    Returns:
        grayscale RGB [H,W,3].
    """
    alpha = np.clip(alpha, 0.0, 1.0)

    return np.repeat(
        alpha[..., None],
        3,
        axis=2,
    )


def smoothstep(t):
    """
    Smooth one-way interpolation from 0 to 1.
    """
    return t * t * (3.0 - 2.0 * t)


def loop_parameter(t):
    """
    Smooth A -> B -> A loop.
    """
    return (
        0.5
        - 0.5 * math.cos(2.0 * math.pi * t)
    )


def make_renderer(
    surface_model,
    volume_model,
    surface_mean,
    surface_std,
    volume_mean,
    volume_std,
    args,
    device,
):
    max_patches_per_tile = (
        None
        if args.max_patches_per_tile <= 0
        else args.max_patches_per_tile
    )

    return NeuralSceneRenderer(
        surface_model=surface_model,
        volume_model=volume_model,
        surface_mean=surface_mean,
        surface_std=surface_std,
        volume_mean=volume_mean,
        volume_std=volume_std,
        fno_radius=FNO_RADIUS,
        fno_batch_size=args.fno_batch_size,
        placement_batch_size=args.placement_batch_size,
        flip_projection_y=args.flip_projection_y,
        flip_fno_vertical=args.flip_fno_vertical,
        use_tile_renderer=args.use_tile_renderer,
        use_visibility_culling=args.use_visibility_culling,
        tile_size=args.tile_size,
        tile_roi_margin=args.tile_roi_margin,
        max_patches_per_tile=max_patches_per_tile,
        use_fno_activation_checkpointing=False,
        max_active_slices_per_camera=(
            args.max_active_slices_per_camera
        ),
        min_projected_patch_size=(
            args.min_projected_patch_size
        ),
        active_slice_margin=args.active_slice_margin,
    ).to(device)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = ArgumentParser(
        description=(
            "Render a fixed-camera neural-scene video while "
            "interpolating the shared global SH environment."
        )
    )

    lp = ModelParams(parser)

    parser.add_argument(
        "--scene_checkpoint",
        type=str,
        required=True,
        help="Saved neural-scene checkpoint.",
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
        "--camera_index",
        type=int,
        default=0,
        help="Fixed COLMAP/3DGS training-camera index.",
    )

    parser.add_argument(
        "--env_a",
        type=int,
        default=0,
        help="Starting procedural environment ID.",
    )

    parser.add_argument(
        "--env_b",
        type=int,
        default=64,
        help="Ending procedural environment ID.",
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--loop",
        action="store_true",
        help="Render env A -> B -> A.",
    )

    parser.add_argument(
        "--background",
        choices=["black", "white"],
        default="black",
    )

    parser.add_argument(
        "--keep_global_residual",
        action="store_true",
        help=(
            "Keep the learned checkpoint global-SH residual. "
            "By default it is reset to zero so the requested "
            "environment IDs control the global illumination."
        ),
    )

    parser.add_argument(
        "--fno_batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--placement_batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--use_tile_renderer",
        action="store_true",
    )

    parser.add_argument(
        "--tile_size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--tile_roi_margin",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--max_patches_per_tile",
        type=int,
        default=0,
        help="Use 0 for no tile patch cap.",
    )

    parser.add_argument(
        "--use_visibility_culling",
        action="store_true",
    )

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
        "--max_active_slices_per_camera",
        type=int,
        default=0,
        help="Use 0 for no active-slice cap.",
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
        "--save_alpha_video",
        action="store_true",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="neural_scene_videos",
    )

    parser.add_argument(
        "--output_name",
        type=str,
        default="environment_interpolation.mp4",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required by the current scene/FNO renderer."
        )

    if not (
        0 <= args.env_a < NUM_GLOBAL_ENVS
    ):
        raise ValueError(
            f"env_a must be in [0,{NUM_GLOBAL_ENVS - 1}]"
        )

    if not (
        0 <= args.env_b < NUM_GLOBAL_ENVS
    ):
        raise ValueError(
            f"env_b must be in [0,{NUM_GLOBAL_ENVS - 1}]"
        )

    device = torch.device("cuda")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_video = output_dir / args.output_name

    alpha_video = output_dir / (
        Path(args.output_name).stem
        + "_alpha.mp4"
    )

    background_rgb = (
        [1.0, 1.0, 1.0]
        if args.background == "white"
        else [0.0, 0.0, 0.0]
    )

    print("Using device:", device)
    print("Scene checkpoint:", args.scene_checkpoint)
    print("Fixed camera index:", args.camera_index)
    print(
        f"Environment interpolation: "
        f"{args.env_a} -> {args.env_b}"
    )
    print("Output:", output_video)

    # ========================================================
    # Load frozen FNO models
    # ========================================================

    surface_model, surface_mean, surface_std, surface_dim = (
        load_fno_checkpoint(
            args.surface_checkpoint,
            device,
        )
    )

    volume_model, volume_mean, volume_std, volume_dim = (
        load_fno_checkpoint(
            args.volume_checkpoint,
            device,
        )
    )

    if surface_dim != volume_dim:
        raise RuntimeError(
            f"Surface/volume latent mismatch: "
            f"{surface_dim} vs {volume_dim}"
        )

    # ========================================================
    # Load neural-scene checkpoint
    # ========================================================

    checkpoint = torch.load(
        args.scene_checkpoint,
        map_location=device,
        weights_only=False,
    )

    optimize_sh = bool(
        checkpoint.get(
            "metadata",
            {},
        ).get(
            "optimize_sh",
            False,
        )
    )

    neural_scene = rebuild_neural_scene_from_checkpoint(
        checkpoint=checkpoint,
        device=device,
        optimize_sh=optimize_sh,
    )

    print(
        "Loaded neural slices:",
        len(neural_scene.slices),
    )

    # Reset the learned global lighting residual unless explicitly
    # preserving it. This makes the requested SH-bank IDs define
    # the global environment directly.
    if not args.keep_global_residual:
        with torch.no_grad():
            neural_scene.lighting.raw_global_sh_delta.zero_()

    # ========================================================
    # Create renderer
    # ========================================================

    renderer = make_renderer(
        surface_model=surface_model,
        volume_model=volume_model,
        surface_mean=surface_mean,
        surface_std=surface_std,
        volume_mean=volume_mean,
        volume_std=volume_std,
        args=args,
        device=device,
    )

    print(
        "Renderer:",
        "tile-binned"
        if args.use_tile_renderer
        else "reference ROI",
    )

    # ========================================================
    # Load 3DGS source scene for fixed camera
    # ========================================================

    scene_args = lp.extract(args)

    if not scene_args.model_path:
        scene_args.model_path = str(
            REPO_DIR / "output" / "environment_video_scene"
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

    if (
        args.camera_index < 0
        or args.camera_index >= len(cameras)
    ):
        raise IndexError(
            f"camera_index={args.camera_index} invalid; "
            f"available={len(cameras)}"
        )

    fixed_camera = cameras[args.camera_index]

    print(
        "Fixed camera:",
        fixed_camera.image_name,
    )

    print(
        "Output resolution:",
        fixed_camera.image_width,
        "x",
        fixed_camera.image_height,
    )

    # ========================================================
    # Build procedural environment SH bank
    # ========================================================

    sh_bank = build_sh_bank(device)

    env_a = sh_bank[args.env_a]
    env_b = sh_bank[args.env_b]

    # ========================================================
    # Video writers
    # ========================================================

    rgb_writer = imageio.get_writer(
        str(output_video),
        fps=args.fps,
        codec="libx264",
        quality=8,
    )

    alpha_writer = None

    if args.save_alpha_video:
        alpha_writer = imageio.get_writer(
            str(alpha_video),
            fps=args.fps,
            codec="libx264",
            quality=8,
        )

    # ========================================================
    # Render frames
    # ========================================================

    print(
        f"Rendering {args.num_frames} frame(s) "
        f"at {args.fps} FPS..."
    )

    try:
        with torch.no_grad():
            for frame_index in range(args.num_frames):
                t = frame_index / max(
                    args.num_frames - 1,
                    1,
                )

                if args.loop:
                    u = loop_parameter(t)
                else:
                    u = smoothstep(t)

                # Interpolate between valid procedural SH vectors.
                frame_global_sh = (
                    (1.0 - u) * env_a
                    + u * env_b
                )

                # Update only global environment. Slice local SH
                # residuals remain from the loaded neural scene.
                neural_scene.lighting.initial_global_sh.copy_(
                    frame_global_sh
                )

                rgba, _ = renderer(
                    fixed_camera,
                    neural_scene,
                )

                rgb = rgba_to_visible_rgb(
                    rgba,
                    background_rgb,
                )

                rgb_writer.append_data(
                    (
                        rgb * 255.0 + 0.5
                    ).astype(np.uint8)
                )

                if alpha_writer is not None:
                    alpha = (
                        rgba[0, 3]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    alpha_writer.append_data(
                        (
                            alpha_to_rgb(alpha)
                            * 255.0
                            + 0.5
                        ).astype(np.uint8)
                    )

                if (
                    frame_index % 10 == 0
                    or frame_index == args.num_frames - 1
                ):
                    print(
                        f"Rendered frame "
                        f"{frame_index + 1}/"
                        f"{args.num_frames} "
                        f"(env blend={u:.3f})"
                    )

    finally:
        rgb_writer.close()

        if alpha_writer is not None:
            alpha_writer.close()

    print("Saved RGB video:", output_video)

    if alpha_writer is not None:
        print("Saved alpha video:", alpha_video)


if __name__ == "__main__":
    main()