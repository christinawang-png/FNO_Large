#!/usr/bin/env python

import sys
import math
from pathlib import Path
from argparse import ArgumentParser
from types import SimpleNamespace

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
# IMPORT YOUR EXISTING SCENE-OPTIMIZER COMPONENTS
# ============================================================

# Change this import only if your main optimizer file has a
# different filename.
from optimize_neural_scene_random_init import (
    FNO_RADIUS,
    load_fno_checkpoint,
    rebuild_neural_scene_from_checkpoint,
    NeuralSceneRenderer,
)

from scene import Scene, GaussianModel
from arguments import ModelParams


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
# CAMERA INTERPOLATION
# ============================================================

def smoothstep(t):
    """
    Smooth 0 -> 1 interpolation with slower motion near endpoints.
    """
    return t * t * (3.0 - 2.0 * t)


def orthonormalize_rotation(matrix):
    """
    Project a 3x3 matrix to the closest rotation matrix using SVD.
    """
    u, _, vh = torch.linalg.svd(matrix)
    rot = u @ vh

    # Prevent reflection if determinant becomes negative.
    if torch.det(rot) < 0:
        u = u.clone()
        u[:, -1] *= -1.0
        rot = u @ vh

    return rot


def interpolate_camera(
    camera_a,
    camera_b,
    t,
):
    """
    Interpolate two 3DGS camera poses.

    Uses:
      - linear interpolation for camera center/translation
      - linear interpolation followed by SVD orthonormalization
        for camera rotation

    The 3DGS code uses row-vector transforms:
        point_row @ world_view_transform
    """
    device = camera_a.world_view_transform.device
    dtype = camera_a.world_view_transform.dtype

    t_tensor = torch.tensor(
        float(t),
        device=device,
        dtype=dtype,
    )

    # In row-vector convention, inverse(view) maps camera-space
    # row coordinates back to world-space row coordinates.
    c2w_a = torch.linalg.inv(
        camera_a.world_view_transform
    )

    c2w_b = torch.linalg.inv(
        camera_b.world_view_transform
    )

    rotation_a = c2w_a[:3, :3]
    rotation_b = c2w_b[:3, :3]

    position_a = c2w_a[3, :3]
    position_b = c2w_b[3, :3]

    # Approximate rotation interpolation.
    rotation_linear = (
        (1.0 - t_tensor) * rotation_a
        + t_tensor * rotation_b
    )

    rotation = orthonormalize_rotation(
        rotation_linear
    )

    position = (
        (1.0 - t_tensor) * position_a
        + t_tensor * position_b
    )

    c2w = torch.eye(
        4,
        dtype=dtype,
        device=device,
    )

    c2w[:3, :3] = rotation
    c2w[3, :3] = position

    world_view_transform = torch.linalg.inv(c2w)

    # Assume intrinsic camera settings match between training cameras.
    projection_matrix = camera_a.projection_matrix

    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(
            projection_matrix.unsqueeze(0)
        )
    ).squeeze(0)

    camera_center = c2w[3, :3]

    # The neural renderer only needs these fields.
    return SimpleNamespace(
        image_width=int(camera_a.image_width),
        image_height=int(camera_a.image_height),
        world_view_transform=world_view_transform,
        projection_matrix=projection_matrix,
        full_proj_transform=full_proj_transform,
        camera_center=camera_center,
        FoVx=camera_a.FoVx,
        FoVy=camera_a.FoVy,
        image_name=f"interp_{t:.5f}",
    )


# ============================================================
# VIDEO HELPERS
# ============================================================

def rgba_to_visible_rgb(
    rgba,
    background_rgb,
):
    """
    rgba:
        [1,4,H,W], premultiplied RGB + alpha.

    Returns:
        [H,W,3], uint8-ready float image in [0,1].
    """
    background = torch.tensor(
        background_rgb,
        dtype=rgba.dtype,
        device=rgba.device,
    ).view(1, 3, 1, 1)

    visible_rgb = (
        rgba[:, :3]
        + (1.0 - rgba[:, 3:4]) * background
    )

    image = (
        visible_rgb[0]
        .detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    return np.clip(image, 0.0, 1.0)


def alpha_to_rgb(alpha):
    """
    alpha:
        [H,W] float.

    Returns grayscale RGB [H,W,3].
    """
    alpha = np.clip(alpha, 0.0, 1.0)

    return np.repeat(
        alpha[..., None],
        3,
        axis=2,
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
            "Render a neural-scene video by interpolating "
            "between two COLMAP/3DGS cameras."
        )
    )

    # Standard 3DGS source/model/resolution arguments.
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
        "--camera_a",
        type=int,
        default=0,
        help="First training-camera index.",
    )

    parser.add_argument(
        "--camera_b",
        type=int,
        default=10,
        help="Second training-camera index.",
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
        help="Render camera A -> B -> A instead of A -> B.",
    )

    parser.add_argument(
        "--background",
        choices=["black", "white"],
        default="black",
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
        default="camera_interpolation.mp4",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This script requires CUDA because the current "
            "3DGS camera/scene code uses CUDA tensors."
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

    if args.background == "white":
        background_rgb = [1.0, 1.0, 1.0]
    else:
        background_rgb = [0.0, 0.0, 0.0]

    print("Using device:", device)
    print("Scene checkpoint:", args.scene_checkpoint)
    print("Output video:", output_video)

    # ========================================================
    # Load frozen FNOs
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
    # Rebuild neural scene from checkpoint
    # ========================================================

    scene_checkpoint = torch.load(
        args.scene_checkpoint,
        map_location=device,
        weights_only=False,
    )

    optimize_sh = bool(
        scene_checkpoint.get(
            "metadata",
            {},
        ).get(
            "optimize_sh",
            False,
        )
    )

    neural_scene = rebuild_neural_scene_from_checkpoint(
        checkpoint=scene_checkpoint,
        device=device,
        optimize_sh=optimize_sh,
    )

    print(
        "Loaded neural slices:",
        len(neural_scene.slices),
    )

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
    # Load source 3DGS scene only for cameras
    # ========================================================

    scene_args = lp.extract(args)

    if not scene_args.model_path:
        scene_args.model_path = str(
            REPO_DIR / "output" / "video_scene"
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

    if (
        args.camera_a < 0
        or args.camera_a >= len(cameras)
    ):
        raise IndexError(
            f"camera_a={args.camera_a} invalid; "
            f"available cameras={len(cameras)}"
        )

    if (
        args.camera_b < 0
        or args.camera_b >= len(cameras)
    ):
        raise IndexError(
            f"camera_b={args.camera_b} invalid; "
            f"available cameras={len(cameras)}"
        )

    camera_a = cameras[args.camera_a]
    camera_b = cameras[args.camera_b]

    print(
        "Camera A:",
        args.camera_a,
        camera_a.image_name,
    )

    print(
        "Camera B:",
        args.camera_b,
        camera_b.image_name,
    )

    if (
        camera_a.image_width != camera_b.image_width
        or camera_a.image_height != camera_b.image_height
    ):
        print(
            "[WARN] Camera image sizes differ. "
            f"Using Camera A output resolution: "
            f"{camera_a.image_width}x{camera_a.image_height}"
        )

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
        f"Rendering {args.num_frames} frames "
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
                    # A -> B -> A.
                    interp_t = (
                        0.5
                        - 0.5
                        * math.cos(
                            2.0 * math.pi * t
                        )
                    )
                else:
                    # A -> B.
                    interp_t = smoothstep(t)
                    
                    interpolated_camera = interpolate_camera(
                        camera_a,
                        camera_b,
                        interp_t,
                    )

                rgba, _ = renderer(
                    interpolated_camera,
                    neural_scene,
                )

                rgb = rgba_to_visible_rgb(
                    rgba,
                    background_rgb=background_rgb,
                )

                rgb_writer.append_data(
                    (rgb * 255.0 + 0.5).astype(np.uint8)
                )
                
                if (frame_index % 10 == 0):
                    print(
                        f"frame={frame_index} "
                        f"t={interp_t:.3f} "
                        f"camera_center="
                        f"{interpolated_camera.camera_center.detach().cpu().numpy()}"
                    )
                

                if alpha_writer is not None:
                    alpha = (
                        rgba[0, 3]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    alpha_rgb = alpha_to_rgb(alpha)

                    alpha_writer.append_data(
                        (
                            alpha_rgb * 255.0
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
                        f"{args.num_frames}"
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