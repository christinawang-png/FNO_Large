#!/usr/bin/env python
"""
Render a smooth synthetic camera-orbit video of a neural-scene checkpoint.

Unlike training-camera video rendering, this script creates a continuous
camera circle around the point-cloud center. Every camera points at the same
target, so the video does not jump between inconsistent training poses.

The selected renderer module must provide:

    load_fno_checkpoint(...)
    NeuralSceneRenderer
    rebuild_neural_scene_from_checkpoint(...)
    visible_rgb_from_rgba(...)

Examples:
    --renderer_module optimize_neural_scene
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
# SYNTHETIC CAMERA
# ============================================================

class OrbitMiniCam:
    """
    Minimal camera object compatible with NeuralSceneRenderer.

    The renderer needs:

        image_width
        image_height
        FoVx
        FoVy
        world_view_transform
        full_proj_transform
        camera_center
        image_name
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


def normalize(vector, eps=1e-8):
    return vector / vector.norm().clamp_min(eps)


def make_look_at_camera(
    position,
    target,
    fov_x,
    image_width,
    image_height,
    znear=0.01,
    zfar=100.0,
    device="cuda",
    image_name="synthetic_orbit",
):
    """
    Build a synthetic 3DGS-compatible camera.

    3DGS / COLMAP convention used here:

        camera +Z = forward direction
        camera +X = right
        camera +Y = down

    The camera's optical axis points from `position` toward `target`.
    """
    from utils.graphics_utils import getProjectionMatrix

    device = torch.device(device)

    position = torch.as_tensor(
        position,
        dtype=torch.float32,
        device=device,
    ).reshape(3)

    target = torch.as_tensor(
        target,
        dtype=torch.float32,
        device=device,
    ).reshape(3)

    # Camera forward: world direction from camera to target.
    forward = normalize(target - position)

    # Standard world-up. For elevation near 0/180 degrees this may become
    # nearly parallel to forward, so fall back to another axis if needed.
    world_up = torch.tensor(
        [0.0, 0.0, 1.0],
        dtype=torch.float32,
        device=device,
    )

    if torch.abs(torch.dot(forward, world_up)) > 0.98:
        world_up = torch.tensor(
            [0.0, 1.0, 0.0],
            dtype=torch.float32,
            device=device,
        )

    # Camera x/right and y/down axes.
    right = normalize(torch.cross(forward, world_up, dim=0))
    down = normalize(torch.cross(forward, right, dim=0))

    # World-to-camera rotation matrix. Rows correspond to camera axes:
    #
    # x_camera = dot(right,   p_world - position)
    # y_camera = dot(down,    p_world - position)
    # z_camera = dot(forward, p_world - position)
    rotation = torch.stack(
        [right, down, forward],
        dim=0,
    )

    translation = -rotation @ position

    # Standard column-vector world-to-camera matrix.
    #
    # The Graphdeco renderer stores the transpose because it uses row-vector
    # multiplication:
    #
    # point_h @ world_view_transform
    world_to_camera_column = torch.eye(
        4,
        dtype=torch.float32,
        device=device,
    )

    world_to_camera_column[:3, :3] = rotation
    world_to_camera_column[:3, 3] = translation

    world_view_transform = world_to_camera_column.transpose(0, 1)

    # Keep horizontal FoV fixed. Recompute vertical FoV for selected output
    # aspect ratio assuming square pixels.
    fov_x = float(fov_x)

    fov_y = 2.0 * math.atan(
        math.tan(0.5 * fov_x)
        * float(image_height)
        / float(image_width)
    )

    projection_matrix = getProjectionMatrix(
        znear=float(znear),
        zfar=float(zfar),
        fovX=fov_x,
        fovY=fov_y,
    ).transpose(0, 1).to(device)

    full_proj_transform = (
        world_view_transform.unsqueeze(0)
        .bmm(projection_matrix.unsqueeze(0))
        .squeeze(0)
    )

    return OrbitMiniCam(
        image_width=image_width,
        image_height=image_height,
        fov_x=fov_x,
        fov_y=fov_y,
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj_transform,
        image_name=image_name,
    )


# ============================================================
# FNO / CHECKPOINT HELPERS
# ============================================================

def frame_from_rgba(
    predicted_rgba,
    visible_rgb_from_rgba_fn,
):
    """
    Convert [1,4,H,W] premultiplied RGBA to uint8 HWC RGB frame.
    """
    rgb = visible_rgb_from_rgba_fn(
        predicted_rgba
    )[0]

    rgb_np = rgb.detach().cpu().numpy()
    rgb_np = np.transpose(rgb_np, (1, 2, 0))
    rgb_np = np.clip(rgb_np, 0.0, 1.0)

    return (rgb_np * 255.0 + 0.5).astype(np.uint8)


def parse_args():
    parser = ArgumentParser(
        description=(
            "Render a smooth synthetic orbit video from a neural-scene "
            "checkpoint."
        )
    )

    parser.add_argument(
        "--source_path",
        type=str,
        required=True,
        help="3DGS source/dataset path.",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="3DGS model path used to load point cloud/cameras.",
    )

    parser.add_argument(
        "--scene_checkpoint",
        type=Path,
        required=True,
        help="Saved neural-scene checkpoint.",
    )

    parser.add_argument(
        "--surface_checkpoint",
        type=Path,
        required=True,
        help="Frozen trained surface FNO checkpoint.",
    )

    parser.add_argument(
        "--volume_checkpoint",
        type=Path,
        required=True,
        help="Frozen trained volume FNO checkpoint.",
    )

    parser.add_argument(
        "--renderer_module",
        type=str,
        default="optimize_neural_scene",
        help=(
            "Optimizer module matching the scene checkpoint. "
            "Use optimize_neural_scene_softmode for soft-mode scenes."
        ),
    )

    parser.add_argument(
        "--radius",
        type=float,
        default=2.2,
        help=(
            "Synthetic orbit radius around the point-cloud center. "
            "Use a larger value if object is clipped."
        ),
    )

    parser.add_argument(
        "--elevation_deg",
        type=float,
        default=60.0,
        help=(
            "Camera phi/elevation in degrees. "
            "90 means an equatorial sideways orbit."
        ),
    )

    parser.add_argument(
        "--theta_start_deg",
        type=float,
        default=0.0,
        help="Starting azimuth in degrees.",
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=120,
        help="Number of frames in the full 360-degree orbit.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frames per second in output MP4.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Synthetic camera/output video width.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Synthetic camera/output video height.",
    )

    parser.add_argument(
        "--fov_x_deg",
        type=float,
        default=0.0,
        help=(
            "Horizontal field of view in degrees. "
            "Use 0 to copy FoVx from the first training camera."
        ),
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
        "--output",
        type=Path,
        default=Path("./neural_scene_synthetic_orbit.mp4"),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for synthetic neural-scene orbit rendering."
        )

    if not args.scene_checkpoint.is_file():
        raise FileNotFoundError(
            f"Neural-scene checkpoint not found:\n"
            f"  {args.scene_checkpoint}"
        )

    if args.radius <= 0.0:
        raise ValueError("--radius must be positive.")

    if args.num_frames < 2:
        raise ValueError("--num_frames must be at least 2.")

    if args.width < 2 or args.height < 2:
        raise ValueError("--width and --height must be at least 2.")

    device = torch.device("cuda")

    # H.264 prefers even output dimensions.
    output_width = int(args.width) - (int(args.width) % 2)
    output_height = int(args.height) - (int(args.height) % 2)

    print("Using device:", device)
    print("Renderer module:", args.renderer_module)
    print("Output resolution:", output_width, "x", output_height)

    # --------------------------------------------------------
    # Import matching optimizer/renderer module.
    # --------------------------------------------------------

    render_module = importlib.import_module(
        args.renderer_module
    )

    required_names = [
        "load_fno_checkpoint",
        "NeuralSceneRenderer",
        "rebuild_neural_scene_from_checkpoint",
        "visible_rgb_from_rgba",
    ]

    missing_names = [
        name
        for name in required_names
        if not hasattr(render_module, name)
    ]

    if missing_names:
        raise RuntimeError(
            f"Renderer module '{args.renderer_module}' is missing:\n"
            f"{missing_names}"
        )

    from arguments import ModelParams
    from scene import GaussianModel, Scene

    # --------------------------------------------------------
    # Load frozen trained FNOs.
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
    # Restore neural scene checkpoint.
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

    print(
        f"Rebuilt neural scene with {len(neural_scene.slices)} slices."
    )

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
    # Load 3DGS point cloud and one camera for FOV reference.
    # --------------------------------------------------------

    parser = ArgumentParser()
    lp = ModelParams(parser)

    scene_cli = [
        "--source_path",
        str(args.source_path),
        "--model_path",
        str(args.model_path),
    ]

    scene_args, _ = parser.parse_known_args(scene_cli)
    dataset_args = lp.extract(scene_args)

    gaussian_model = GaussianModel(
        getattr(dataset_args, "sh_degree", 3)
    )

    gs_scene = Scene(
        dataset_args,
        gaussian_model,
        shuffle=False,
        resolution_scales=[1.0],
    )

    training_cameras = gs_scene.getTrainCameras(scale=1.0)

    if not training_cameras:
        raise RuntimeError(
            "No training cameras found. Need one camera for reference FoV."
        )

    point_xyz = gaussian_model.get_xyz.detach()
    scene_center = point_xyz.median(dim=0).values.to(device)

    reference_camera = training_cameras[0]

    if args.fov_x_deg > 0.0:
        fov_x = math.radians(args.fov_x_deg)
    else:
        fov_x = float(reference_camera.FoVx)

    print(
        "Scene center:",
        scene_center.detach().cpu().numpy(),
    )

    print(
        f"Orbit: radius={args.radius:.4f}, "
        f"elevation={args.elevation_deg:.1f}°, "
        f"fov_x={math.degrees(fov_x):.2f}°"
    )

    # --------------------------------------------------------
    # Create synthetic camera positions.
    # --------------------------------------------------------

    phi = math.radians(args.elevation_deg)
    theta_start = math.radians(args.theta_start_deg)

    theta_values = np.linspace(
        theta_start,
        theta_start + 2.0 * math.pi,
        num=int(args.num_frames),
        endpoint=False,
        dtype=np.float64,
    )

    synthetic_cameras = []

    for frame_index, theta in enumerate(theta_values):
        position = scene_center + torch.tensor(
            [
                args.radius * math.sin(phi) * math.cos(theta),
                args.radius * math.sin(phi) * math.sin(theta),
                args.radius * math.cos(phi),
            ],
            dtype=torch.float32,
            device=device,
        )

        camera = make_look_at_camera(
            position=position,
            target=scene_center,
            fov_x=fov_x,
            image_width=output_width,
            image_height=output_height,
            znear=args.znear,
            zfar=args.zfar,
            device=device,
            image_name=f"orbit_{frame_index:04d}",
        )

        synthetic_cameras.append(camera)

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
            for frame_index, camera in enumerate(synthetic_cameras):
                predicted_rgba, _ = renderer(
                    camera,
                    neural_scene,
                )

                frame = frame_from_rgba(
                    predicted_rgba=predicted_rgba.clamp(0.0, 1.0),
                    visible_rgb_from_rgba_fn=(
                        render_module.visible_rgb_from_rgba
                    ),
                )

                writer.append_data(frame)

                theta_deg = math.degrees(theta_values[frame_index]) % 360.0

                print(
                    f"[{frame_index + 1:03d}/{len(synthetic_cameras)}] "
                    f"theta={theta_deg:6.1f}°"
                )

    finally:
        writer.close()

    print("Done.")
    print("Wrote video:", args.output)


if __name__ == "__main__":
    main()