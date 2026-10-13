#!/usr/bin/env python
"""
Render a looped camera interpolation video for a FixedVoxelScene.

Path:

    training camera A -> training camera B -> training camera A

Camera position is linearly interpolated and orientation uses quaternion
SLERP. Every output frame has the same requested resolution.

Requirements:
    scipy
    imageio-ffmpeg
"""

from __future__ import annotations

import importlib
import math
import sys
from argparse import ArgumentParser
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp


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
# MINIMAL SYNTHETIC CAMERA
# ============================================================

class InterpolatedMiniCam:
    """
    Camera interface expected by FixedVoxelRenderer.
    """

    def __init__(
        self,
        image_width,
        image_height,
        fov_x,
        fov_y,
        world_view_transform,
        full_proj_transform,
        image_name,
    ):
        self.image_width = int(image_width)
        self.image_height = int(image_height)

        self.FoVx = float(fov_x)
        self.FoVy = float(fov_y)

        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform

        self.camera_center = torch.inverse(
            self.world_view_transform
        )[3, :3]

        self.image_name = str(image_name)


# ============================================================
# FIXED VOXEL CHECKPOINT RECONSTRUCTION
# ============================================================

def rebuild_fixed_voxel_scene_from_checkpoint(
    checkpoint_data,
    device,
    optimize_sh=False,
):
    """
    Reconstruct fixed voxel topology from block_specs, then restore exact
    saved tensors from scene_state.
    """
    from fixed_voxel_parameters import FixedVoxelBlock
    from fixed_voxel_scene import (
        FixedVoxelScene,
        module_name_from_key,
        voxel_key,
    )

    required_keys = {
        "scene_state",
        "block_specs",
        "root_voxel_size",
    }

    missing = required_keys - set(checkpoint_data.keys())

    if missing:
        raise KeyError(
            "Checkpoint missing required keys: "
            f"{sorted(missing)}"
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

        key = voxel_key(level, grid_index)
        module_name = module_name_from_key(key)
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
            0.05,
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
        print("[RESUME] Missing scene keys:")
        for key in missing_keys:
            print("  ", key)

    if unexpected_keys:
        print("[RESUME] Unexpected scene keys:")
        for key in unexpected_keys:
            print("  ", key)

    return voxel_scene


# ============================================================
# CAMERA POSE CONVERSION
# ============================================================

def camera_to_standard_pose(camera):
    """
    Convert a Graphdeco/3DGS Camera into standard column-vector camera pose.

    Returns:
        position: [3] NumPy array
        rotation_camera_to_world: [3,3] NumPy array
    """
    world_view_row = camera.world_view_transform.detach().cpu()

    # 3DGS stores the transpose for row-vector multiplication.
    camera_to_world_row = torch.inverse(world_view_row)

    # Convert to standard column-vector camera-to-world convention.
    camera_to_world_column = camera_to_world_row.transpose(0, 1)

    position = camera_to_world_column[:3, 3].numpy()
    rotation_camera_to_world = (
        camera_to_world_column[:3, :3].numpy()
    )

    return position, rotation_camera_to_world


def make_interpolated_camera(
    position,
    rotation_camera_to_world,
    fov_x,
    fov_y,
    image_width,
    image_height,
    znear,
    zfar,
    device,
    image_name,
):
    """
    Create a 3DGS row-vector-compatible synthetic camera.
    """
    from utils.graphics_utils import getProjectionMatrix

    position = torch.as_tensor(
        position,
        dtype=torch.float32,
        device=device,
    ).reshape(3)

    rotation_camera_to_world = torch.as_tensor(
        rotation_camera_to_world,
        dtype=torch.float32,
        device=device,
    ).reshape(3, 3)

    # Standard column-vector camera-to-world transform.
    camera_to_world_column = torch.eye(
        4,
        dtype=torch.float32,
        device=device,
    )

    camera_to_world_column[:3, :3] = rotation_camera_to_world
    camera_to_world_column[:3, 3] = position

    # Standard column-vector world-to-camera transform.
    world_to_camera_column = torch.inverse(
        camera_to_world_column
    )

    # Graphdeco camera matrix convention uses row vectors.
    world_view_transform = world_to_camera_column.transpose(0, 1)

    projection_matrix = getProjectionMatrix(
        znear=float(znear),
        zfar=float(zfar),
        fovX=float(fov_x),
        fovY=float(fov_y),
    ).transpose(0, 1).to(device)

    full_proj_transform = (
        world_view_transform.unsqueeze(0)
        .bmm(projection_matrix.unsqueeze(0))
        .squeeze(0)
    )

    return InterpolatedMiniCam(
        image_width=image_width,
        image_height=image_height,
        fov_x=fov_x,
        fov_y=fov_y,
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj_transform,
        image_name=image_name,
    )


def interpolate_camera_poses(
    camera_a,
    camera_b,
    frames_per_leg,
):
    """
    Build one loop:

        A -> B -> A

    Uses endpoint=False on each leg so the first frame is not duplicated
    at the leg boundary or loop boundary.
    """
    if frames_per_leg < 2:
        raise ValueError("--frames_per_leg must be at least 2.")

    position_a, rotation_a = camera_to_standard_pose(camera_a)
    position_b, rotation_b = camera_to_standard_pose(camera_b)

    rotations = Rotation.from_matrix(
        np.stack(
            [rotation_a, rotation_b],
            axis=0,
        )
    )

    slerp = Slerp(
        times=np.array([0.0, 1.0]),
        rotations=rotations,
    )

    # A -> B, excluding B because B starts the next leg.
    forward_t = np.linspace(
        0.0,
        1.0,
        num=frames_per_leg,
        endpoint=False,
        dtype=np.float64,
    )

    # B -> A, excluding A because it is the next loop's first frame.
    backward_t = np.linspace(
        1.0,
        0.0,
        num=frames_per_leg,
        endpoint=False,
        dtype=np.float64,
    )

    interpolation_values = np.concatenate(
        [forward_t, backward_t],
        axis=0,
    )

    interpolated_rotations = slerp(
        interpolation_values
    ).as_matrix()

    poses = []

    for frame_index, t in enumerate(interpolation_values):
        position = (
            (1.0 - t) * position_a
            + t * position_b
        )

        fov_x = (
            (1.0 - t) * float(camera_a.FoVx)
            + t * float(camera_b.FoVx)
        )

        fov_y = (
            (1.0 - t) * float(camera_a.FoVy)
            + t * float(camera_b.FoVy)
        )

        poses.append(
            {
                "t": float(t),
                "position": position,
                "rotation_camera_to_world": interpolated_rotations[
                    frame_index
                ],
                "fov_x": float(fov_x),
                "fov_y": float(fov_y),
            }
        )

    return poses


# ============================================================
# VIDEO HELPERS
# ============================================================

def frame_from_rgba(
    predicted_rgba,
    visible_rgb_from_rgba_fn,
):
    """
    Convert [1,4,H,W] premultiplied RGBA to uint8 HWC RGB.
    """
    rgb = visible_rgb_from_rgba_fn(
        predicted_rgba
    )[0]

    rgb_np = rgb.detach().cpu().numpy()
    rgb_np = np.transpose(rgb_np, (1, 2, 0))
    rgb_np = np.clip(rgb_np, 0.0, 1.0)

    return (rgb_np * 255.0 + 0.5).astype(np.uint8)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = ArgumentParser(
        description=(
            "Render a fixed voxel checkpoint by smoothly interpolating "
            "between two real 3DGS training cameras and looping back."
        )
    )

    # ModelParams below adds --source_path and --model_path.
    parser.add_argument(
        "--scene_checkpoint",
        type=Path,
        required=True,
        help="Fixed voxel checkpoint saved by optimize_fixed_voxel_scene.py",
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
        "--camera_a",
        type=int,
        required=True,
        help="First real training camera index.",
    )

    parser.add_argument(
        "--camera_b",
        type=int,
        required=True,
        help="Second real training camera index.",
    )

    parser.add_argument(
        "--frames_per_leg",
        type=int,
        default=60,
        help=(
            "Frames in A->B and frames in B->A. "
            "Total video frames = 2 * frames_per_leg."
        ),
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--znear",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--zfar",
        type=float,
        default=100.0,
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
            "Must match the camera-depth sign used in fixed voxel training. "
            "Usually +1 or -1."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./fixed_voxel_camera_loop.mp4"),
    )

    return parser


# ============================================================
# MAIN
# ============================================================

def main():
    parser = parse_args()

    from arguments import ModelParams
    from scene import GaussianModel, Scene

    lp = ModelParams(parser)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    if not args.scene_checkpoint.is_file():
        raise FileNotFoundError(
            f"Scene checkpoint not found:\n  {args.scene_checkpoint}"
        )

    if args.frames_per_leg < 2:
        raise ValueError(
            "--frames_per_leg must be at least 2."
        )

    device = torch.device("cuda")

    # H.264 prefers even dimensions.
    output_width = int(args.width) - (int(args.width) % 2)
    output_height = int(args.height) - (int(args.height) % 2)

    # Import the optimizer module only to reuse its FNO loader and
    # background-compositing function.
    optimizer_module = importlib.import_module(
        "optimize_fixed_voxel_scene"
    )

    from fixed_voxel_renderer import FixedVoxelRenderer

    print("Using device:", device)
    print("Fixed voxel checkpoint:", args.scene_checkpoint)

    # --------------------------------------------------------
    # Load frozen FNO models.
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
    # Load fixed voxel checkpoint.
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

    print(
        f"Rebuilt fixed voxel scene with {len(voxel_scene)} blocks."
    )

    # --------------------------------------------------------
    # Load actual training cameras.
    # --------------------------------------------------------

    scene_args = lp.extract(args)

    gaussian_model = GaussianModel(args.sh_degree)

    gs_scene = Scene(
        scene_args,
        gaussian_model,
        shuffle=False,
        resolution_scales=[1.0],
    )

    cameras = gs_scene.getTrainCameras(scale=1.0)

    if not (0 <= args.camera_a < len(cameras)):
        raise ValueError(
            f"camera_a={args.camera_a} invalid for {len(cameras)} cameras."
        )

    if not (0 <= args.camera_b < len(cameras)):
        raise ValueError(
            f"camera_b={args.camera_b} invalid for {len(cameras)} cameras."
        )

    camera_a = cameras[args.camera_a]
    camera_b = cameras[args.camera_b]

    print(
        f"Looping camera {args.camera_a} "
        f"('{camera_a.image_name}') -> "
        f"{args.camera_b} "
        f"('{camera_b.image_name}') -> "
        f"{args.camera_a}"
    )

    poses = interpolate_camera_poses(
        camera_a=camera_a,
        camera_b=camera_b,
        frames_per_leg=args.frames_per_leg,
    )

    # --------------------------------------------------------
    # Build synthetic interpolated cameras.
    # --------------------------------------------------------

    interpolation_cameras = []

    for frame_index, pose in enumerate(poses):
        interpolation_camera = make_interpolated_camera(
            position=pose["position"],
            rotation_camera_to_world=pose[
                "rotation_camera_to_world"
            ],
            fov_x=pose["fov_x"],
            fov_y=pose["fov_y"],
            image_width=output_width,
            image_height=output_height,
            znear=args.znear,
            zfar=args.zfar,
            device=device,
            image_name=f"fixed_interp_{frame_index:04d}",
        )

        interpolation_cameras.append(interpolation_camera)

    # --------------------------------------------------------
    # Render MP4.
    # --------------------------------------------------------

    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        str(args.output),
        fps=int(args.fps),
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )

    try:
        with torch.no_grad():
            for frame_index, camera in enumerate(
                interpolation_cameras
            ):
                predicted_rgba, _ = renderer(
                    camera,
                    voxel_scene,
                )

                frame = frame_from_rgba(
                    predicted_rgba.clamp(0.0, 1.0),
                    optimizer_module.visible_rgb_from_rgba,
                )

                writer.append_data(frame)

                print(
                    f"[{frame_index + 1:03d}/"
                    f"{len(interpolation_cameras)}] "
                    f"t={poses[frame_index]['t']:.3f}"
                )

    finally:
        writer.close()

    print("Done.")
    print("Wrote:", args.output)


if __name__ == "__main__":
    main()