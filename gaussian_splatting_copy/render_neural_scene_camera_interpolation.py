#!/usr/bin/env python
"""
Render a neural-scene checkpoint while smoothly interpolating between two
actual 3DGS training camera poses.

Camera interpolation:
    - position: linear interpolation;
    - rotation: quaternion SLERP;
    - FoV: linear interpolation.

The script works with either:

    --renderer_module optimize_neural_scene
or:
    --renderer_module optimize_neural_scene_softmode
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
# SYNTHETIC INTERPOLATED CAMERA
# ============================================================

class InterpolatedMiniCam:
    """
    Minimal camera interface expected by NeuralSceneRenderer.
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
            world_view_transform
        )[3, :3]

        self.image_name = str(image_name)


def make_interpolated_camera(
    position,
    rotation_camera_to_world,
    fov_x,
    fov_y,
    image_width,
    image_height,
    znear=0.01,
    zfar=100.0,
    device="cuda",
    image_name="interpolated_camera",
):
    """
    Construct a 3DGS-compatible camera from a world-space camera center and
    a standard 3x3 camera-to-world rotation matrix.

    Parameters
    ----------
    position:
        Camera center in world coordinates, shape [3].

    rotation_camera_to_world:
        Standard column-vector camera-to-world rotation matrix, shape [3,3].
        It maps local camera axes into world coordinates.

    Notes
    -----
    The Graphdeco camera implementation stores transforms transposed because
    it uses row-vector multiplication:

        point_h @ camera.full_proj_transform
    """
    from utils.graphics_utils import getProjectionMatrix

    device = torch.device(device)

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

    # 3DGS uses transposed matrices / row-vector convention.
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


# ============================================================
# CAMERA CONVERSION / INTERPOLATION
# ============================================================

def camera_to_standard_pose(camera):
    """
    Convert a 3DGS training camera into:

        position: [3] NumPy array
        rotation_camera_to_world: [3,3] NumPy array

    `camera.world_view_transform` is the transpose of the standard
    column-vector world-to-camera matrix. Its inverse is therefore the
    row-vector camera-to-world transform.

    We transpose that result to get standard column-vector C2W form.
    """
    world_view_row = camera.world_view_transform.detach().cpu()

    # Row-vector camera-to-world transform.
    camera_to_world_row = torch.inverse(world_view_row)

    # Standard column-vector camera-to-world transform.
    camera_to_world_column = camera_to_world_row.transpose(0, 1)

    position = camera_to_world_column[:3, 3].numpy()
    rotation_camera_to_world = (
        camera_to_world_column[:3, :3].numpy()
    )

    return position, rotation_camera_to_world


def interpolate_camera_poses(
    camera_a,
    camera_b,
    num_frames,
):
    """
    Interpolate two actual training-camera poses.

    Returns a list of dictionaries containing:
        position
        rotation_camera_to_world
        fov_x
        fov_y
        interpolation_t
    """
    if num_frames < 2:
        raise ValueError("num_frames must be at least 2.")

    position_a, rotation_a = camera_to_standard_pose(camera_a)
    position_b, rotation_b = camera_to_standard_pose(camera_b)

    # Quaternion SLERP for orientation.
    rotations = Rotation.from_matrix(
        np.stack([rotation_a, rotation_b], axis=0)
    )

    slerp = Slerp(
        times=np.array([0.0, 1.0]),
        rotations=rotations,
    )

    interpolation_values = np.linspace(
        0.0,
        1.0,
        num=int(num_frames),
        endpoint=True,
        dtype=np.float64,
    )

    interpolated_rotations = slerp(
        interpolation_values
    ).as_matrix()

    poses = []

    for frame_index, t in enumerate(interpolation_values):
        # Smooth camera-center interpolation.
        position = (
            (1.0 - t) * position_a
            + t * position_b
        )

        # FOV interpolation.
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
    Convert renderer output [1,4,H,W] to one uint8 RGB frame.
    """
    rgb = visible_rgb_from_rgba_fn(
        predicted_rgba
    )[0]

    rgb_np = rgb.detach().cpu().numpy()
    rgb_np = np.transpose(rgb_np, (1, 2, 0))
    rgb_np = np.clip(rgb_np, 0.0, 1.0)

    return (rgb_np * 255.0 + 0.5).astype(np.uint8)


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = ArgumentParser(
        description=(
            "Render a smooth interpolation between two real 3DGS "
            "training cameras."
        )
    )

    # ModelParams adds --source_path and --model_path later in main.
    parser.add_argument(
        "--scene_checkpoint",
        type=Path,
        required=True,
        help="Neural scene checkpoint to render.",
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
        "--renderer_module",
        type=str,
        default="optimize_neural_scene",
        help=(
            "Use optimize_neural_scene or "
            "optimize_neural_scene_softmode."
        ),
    )

    parser.add_argument(
        "--camera_a",
        type=int,
        required=True,
        help="First training camera index.",
    )

    parser.add_argument(
        "--camera_b",
        type=int,
        required=True,
        help="Second training camera index.",
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=120,
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
        help=(
            "Fixed output width. Camera poses are interpolated, while "
            "all output frames use this resolution."
        ),
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
        default=16,
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
        "--output",
        type=Path,
        default=Path("./neural_scene_camera_interpolation.mp4"),
    )

    return parser


# ============================================================
# MAIN
# ============================================================

def main():
    parser = parse_args()

    # ModelParams registers source_path/model_path and related 3DGS flags.
    from arguments import ModelParams
    lp = ModelParams(parser)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for neural-scene rendering.")

    if not args.scene_checkpoint.is_file():
        raise FileNotFoundError(
            f"Scene checkpoint not found:\n  {args.scene_checkpoint}"
        )

    if args.num_frames < 2:
        raise ValueError("--num_frames must be at least 2.")

    if args.width < 2 or args.height < 2:
        raise ValueError("--width and --height must be at least 2.")

    device = torch.device("cuda")

    # H.264 works best with even dimensions.
    output_width = int(args.width) - (int(args.width) % 2)
    output_height = int(args.height) - (int(args.height) % 2)

    # Import matching renderer module.
    render_module = importlib.import_module(
        args.renderer_module
    )

    required_names = [
        "load_fno_checkpoint",
        "NeuralSceneRenderer",
        "rebuild_neural_scene_from_checkpoint",
        "visible_rgb_from_rgba",
    ]

    missing = [
        name
        for name in required_names
        if not hasattr(render_module, name)
    ]

    if missing:
        raise RuntimeError(
            f"Renderer module '{args.renderer_module}' is missing:\n"
            f"{missing}"
        )

    from scene import GaussianModel, Scene

    print("Using device:", device)
    print("Renderer module:", args.renderer_module)

    # --------------------------------------------------------
    # Frozen FNO models.
    # --------------------------------------------------------

    surface_bundle = render_module.load_fno_checkpoint(
        args.surface_checkpoint,
        device=device,
        expected_mode="surface",
    )

    volume_bundle = render_module.load_fno_checkpoint(
        args.volume_checkpoint,
        device=device,
        expected_mode="volume",
    )

    # --------------------------------------------------------
    # Neural scene checkpoint.
    # --------------------------------------------------------

    scene_checkpoint = torch.load(
        args.scene_checkpoint,
        map_location=device,
        weights_only=False,
    )

    optimize_sh = bool(
        scene_checkpoint.get("metadata", {}).get(
            "optimize_sh",
            False,
        )
    )

    neural_scene = render_module.rebuild_neural_scene_from_checkpoint(
        checkpoint_data=scene_checkpoint,
        device=device,
        optimize_sh=optimize_sh,
    )

    neural_scene.eval()

    fno_radius = float(
        scene_checkpoint.get("metadata", {}).get(
            "fno_radius",
            2.2,
        )
    )

    renderer = render_module.NeuralSceneRenderer(
        surface_bundle=surface_bundle,
        volume_bundle=volume_bundle,
        fno_radius=fno_radius,
        fno_batch_size=args.fno_batch_size,
        flip_projection_y=args.flip_projection_y,
        flip_fno_vertical=args.flip_fno_vertical,
        use_tile_renderer=args.use_tile_renderer,
        use_visibility_culling=False,
        use_fno_activation_checkpointing=False,
    ).to(device)

    renderer.eval()

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

    training_cameras = gs_scene.getTrainCameras(scale=1.0)

    if not (
        0 <= args.camera_a < len(training_cameras)
    ):
        raise ValueError(
            f"camera_a={args.camera_a} is invalid for "
            f"{len(training_cameras)} cameras."
        )

    if not (
        0 <= args.camera_b < len(training_cameras)
    ):
        raise ValueError(
            f"camera_b={args.camera_b} is invalid for "
            f"{len(training_cameras)} cameras."
        )

    camera_a = training_cameras[args.camera_a]
    camera_b = training_cameras[args.camera_b]

    print(
        f"Interpolating training cameras "
        f"{args.camera_a} ('{camera_a.image_name}') -> "
        f"{args.camera_b} ('{camera_b.image_name}')"
    )

    print(
        f"Camera A center: "
        f"{camera_a.camera_center.detach().cpu().numpy()}"
    )

    print(
        f"Camera B center: "
        f"{camera_b.camera_center.detach().cpu().numpy()}"
    )

    poses = interpolate_camera_poses(
        camera_a=camera_a,
        camera_b=camera_b,
        num_frames=args.num_frames,
    )

    # --------------------------------------------------------
    # Build synthetic interpolation cameras.
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
            image_name=f"interp_{frame_index:04d}",
        )

        interpolation_cameras.append(interpolation_camera)

    # --------------------------------------------------------
    # Render video.
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
                    neural_scene,
                )

                frame = frame_from_rgba(
                    predicted_rgba.clamp(0.0, 1.0),
                    render_module.visible_rgb_from_rgba,
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