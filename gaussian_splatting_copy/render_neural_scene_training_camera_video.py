#!/usr/bin/env python
"""
Render a saved moving-slice neural-scene checkpoint as an MP4 using actual
3DGS training-camera poses.

The output frames are ordered by camera azimuth around the point-cloud center.
This is not an interpolated orbit: each frame uses a real camera pose from the
training data.

Supports either:
    --renderer_module optimize_neural_scene
or:
    --renderer_module optimize_neural_scene_softmode

The chosen module must provide:
    load_fno_checkpoint(...)
    NeuralSceneRenderer
    rebuild_neural_scene_from_checkpoint(...)
    visible_rgb_from_rgba(...)
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
# HELPERS
# ============================================================

def camera_azimuth_about_center(camera, scene_center):
    """
    Compute the camera azimuth around a given scene center.

    The camera is sorted using this azimuth, so resulting video frames
    approximately travel around the scene in a horizontal orbit.
    """
    camera_center = camera.camera_center.detach()

    relative = camera_center - scene_center

    theta = torch.atan2(
        relative[1],
        relative[0],
    )

    theta = torch.remainder(
        theta,
        2.0 * math.pi,
    )

    return float(theta.cpu())


def frame_from_rgba(
    predicted_rgba,
    visible_rgb_from_rgba_fn,
    output_height,
    output_width,
):
    """
    Convert predicted premultiplied RGBA to one fixed-size uint8 RGB frame.

    Every video frame is resized to output_height/output_width because
    training cameras may have different source image resolutions.
    """
    rgb = visible_rgb_from_rgba_fn(
        predicted_rgba
    )

    # rgb shape: [1, 3, H, W]
    rgb = torch.nn.functional.interpolate(
        rgb,
        size=(int(output_height), int(output_width)),
        mode="bilinear",
        align_corners=False,
    )

    rgb_np = rgb[0].detach().cpu().numpy()
    rgb_np = np.transpose(rgb_np, (1, 2, 0))
    rgb_np = np.clip(rgb_np, 0.0, 1.0)

    return (rgb_np * 255.0 + 0.5).astype(np.uint8)


def parse_args():
    parser = ArgumentParser(
        description=(
            "Render a neural-scene checkpoint using sorted actual "
            "3DGS training camera poses."
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
        help="3DGS model path used to load cameras/point cloud.",
    )

    parser.add_argument(
        "--scene_checkpoint",
        type=Path,
        required=True,
        help=(
            "Neural-scene checkpoint, such as "
            "checkpoint_iter_001000.pt."
        ),
    )

    parser.add_argument(
        "--surface_checkpoint",
        type=Path,
        required=True,
        help="Trained frozen surface FNO checkpoint.",
    )

    parser.add_argument(
        "--volume_checkpoint",
        type=Path,
        required=True,
        help="Trained frozen volume FNO checkpoint.",
    )

    parser.add_argument(
        "--renderer_module",
        type=str,
        default="optimize_neural_scene",
        help=(
            "Python module containing the matching scene reconstruction "
            "and renderer code. Use optimize_neural_scene_softmode for "
            "soft-mode checkpoints."
        ),
    )

    parser.add_argument(
        "--camera_start",
        type=int,
        default=0,
        help="First training-camera index considered.",
    )

    parser.add_argument(
        "--num_cameras",
        type=int,
        default=0,
        help=(
            "Number of training cameras used. 0 means use all cameras "
            "after --camera_start."
        ),
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=120,
        help=(
            "Number of video frames. Cameras are sampled evenly from "
            "the sorted training-pose list."
        ),
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Output MP4 frames per second.",
    )

    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        help=(
            "Integer output upscaling factor. scale=2 makes a 32x32 "
            "render into a 64x64 frame, etc."
        ),
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
        default=Path("./neural_scene_training_camera_video.mp4"),
    )
    
    parser.add_argument(
        "--output_width",
        type=int,
        default=0,
        help=(
            "Fixed MP4 output width. Use 0 to use the first selected "
            "camera width times --scale."
        ),
    )
    
    parser.add_argument(
        "--output_height",
        type=int,
        default=0,
        help=(
            "Fixed MP4 output height. Use 0 to use the first selected "
            "camera height times --scale."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this renderer.")

    if not args.scene_checkpoint.is_file():
        raise FileNotFoundError(
            f"Neural scene checkpoint not found:\n"
            f"  {args.scene_checkpoint}"
        )

    device = torch.device("cuda")

    # Import the exact optimizer module corresponding to checkpoint format.
    #
    # Examples:
    #   optimize_neural_scene
    #   optimize_neural_scene_softmode
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

    # Imports must occur after REPO_DIR is on sys.path.
    from arguments import ModelParams
    from scene import GaussianModel, Scene

    print("Using device:", device)
    print("Renderer module:", args.renderer_module)
    print("Scene checkpoint:", args.scene_checkpoint)

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
    # Rebuild neural scene.
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
        f"Rebuilt neural scene: "
        f"{len(neural_scene.slices)} slices, "
        f"optimize_sh={optimize_sh}"
    )

    # --------------------------------------------------------
    # Build renderer.
    # --------------------------------------------------------

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
    # Load 3DGS scene and its training cameras.
    # --------------------------------------------------------

    parser = ArgumentParser()
    lp = ModelParams(parser)

    # Create an args namespace that ModelParams.extract() can read.
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

    all_cameras = gs_scene.getTrainCameras(scale=1.0)

    if args.camera_start < 0 or args.camera_start >= len(all_cameras):
        raise ValueError(
            f"camera_start={args.camera_start} is invalid for "
            f"{len(all_cameras)} training cameras."
        )

    available_cameras = all_cameras[args.camera_start:]

    if args.num_cameras > 0:
        available_cameras = available_cameras[:args.num_cameras]

    if not available_cameras:
        raise RuntimeError("No cameras selected.")

    # Use point-cloud median as a stable center for azimuth ordering.
    point_xyz = gaussian_model.get_xyz.detach()
    scene_center = point_xyz.median(dim=0).values.to(device)

    # Sort actual training poses by azimuth.
    ordered_cameras = sorted(
        available_cameras,
        key=lambda camera: camera_azimuth_about_center(
            camera,
            scene_center,
        ),
    )

    # Uniformly sample sorted training views for requested video length.
    if args.num_frames <= 0:
        selected_cameras = ordered_cameras
    else:
        selected_indices = np.linspace(
            0,
            len(ordered_cameras) - 1,
            num=min(args.num_frames, len(ordered_cameras)),
            dtype=np.int64,
        )

        selected_cameras = [
            ordered_cameras[index]
            for index in selected_indices
        ]

    print(
        f"Selected {len(selected_cameras)} real training camera poses "
        f"from {len(available_cameras)} available cameras."
    )
    
    # All MP4 frames must have exactly the same spatial resolution.
    first_camera = selected_cameras[0]
    
    output_width = (
        int(args.output_width)
        if args.output_width > 0
        else int(first_camera.image_width) * int(args.scale)
    )
    
    output_height = (
        int(args.output_height)
        if args.output_height > 0
        else int(first_camera.image_height) * int(args.scale)
    )
    
    # H.264 commonly prefers even dimensions.
    output_width -= output_width % 2
    output_height -= output_height % 2
    
    output_width = max(output_width, 2)
    output_height = max(output_height, 2)
    
    print(
        f"Video output resolution: "
        f"{output_width}x{output_height}"
    )

    # --------------------------------------------------------
    # Render MP4.
    # --------------------------------------------------------

    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        str(args.output),
        fps=args.fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )

    try:
        with torch.no_grad():
            for frame_index, camera in enumerate(selected_cameras):
                predicted_rgba, _ = renderer(
                    camera,
                    neural_scene,
                )

                frame = frame_from_rgba(
                    predicted_rgba=predicted_rgba.clamp(0.0, 1.0),
                    visible_rgb_from_rgba_fn=(
                        render_module.visible_rgb_from_rgba
                    ),
                    output_height=output_height,
                    output_width=output_width,
                )

                writer.append_data(frame)

                theta = camera_azimuth_about_center(
                    camera,
                    scene_center,
                )

                print(
                    f"[{frame_index + 1:03d}/{len(selected_cameras)}] "
                    f"camera={camera.image_name} "
                    f"azimuth={math.degrees(theta):.1f}°"
                )

    finally:
        writer.close()

    print("Done.")
    print("Video:", args.output)


if __name__ == "__main__":
    main()