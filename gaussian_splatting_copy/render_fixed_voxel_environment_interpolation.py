#!/usr/bin/env python
"""
Render a fixed voxel scene while interpolating lighting from the initial
SH-bank environment to the current optimized environment.

Default behavior:
    t = 0:
        checkpoint initial_global_sh
        + zero global SH delta
        + zero local SH deltas

    t = 1:
        checkpoint current global SH delta
        + checkpoint current local SH deltas

Thus the output visualizes how learned environment optimization changed
the fixed voxel scene lighting.

Use --camera_index to select a real 3DGS training camera.
"""

from __future__ import annotations

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
# CHECKPOINT RECONSTRUCTION
# ============================================================

def rebuild_fixed_voxel_scene_from_checkpoint(
    checkpoint_data,
    device,
    optimize_sh=True,
    default_local_sh_bound=0.05,
    default_global_sh_bound=0.05,
):
    """
    Rebuild fixed voxel topology and restore learned state tensors.

    The default bounds should match the optimizer used to create older
    checkpoints that did not save SH bounds explicitly.
    """
    from fixed_voxel_parameters import FixedVoxelBlock
    from fixed_voxel_scene import (
        FixedVoxelScene,
        module_name_from_key,
        voxel_key,
    )

    scene_state = checkpoint_data["scene_state"]
    block_specs = checkpoint_data["block_specs"]

    blocks = []

    for spec in block_specs:
        level = int(spec["level"])
        grid_index = tuple(
            int(value)
            for value in spec["grid_index"]
        )

        parent_key = spec.get("parent_key", None)

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
            local_sh_bound=float(
                spec.get(
                    "local_sh_bound",
                    default_local_sh_bound,
                )
            ),
        ).to(device)

        blocks.append(block)

    global_sh_bound = float(
        checkpoint_data.get(
            "metadata",
            {},
        ).get(
            "global_sh_bound",
            default_global_sh_bound,
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
        print("[WARN] Missing checkpoint state keys:")
        for key in missing_keys:
            print("  ", key)

    if unexpected_keys:
        print("[WARN] Unexpected checkpoint state keys:")
        for key in unexpected_keys:
            print("  ", key)

    return voxel_scene


# ============================================================
# IMAGE HELPERS
# ============================================================

def frame_from_rgba(predicted_rgba, visible_rgb_from_rgba_fn):
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
# CLI
# ============================================================

def parse_args():
    parser = ArgumentParser(
        description=(
            "Interpolate fixed voxel scene lighting from the initial "
            "SH-bank environment to the current optimized environment."
        )
    )

    parser.add_argument(
        "--scene_checkpoint",
        type=Path,
        required=True,
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
        "--camera_index",
        type=int,
        default=0,
        help="Real 3DGS training camera used for all video frames.",
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
        "--camera_depth_sign",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--global_sh_bound",
        type=float,
        default=0.05,
        help=(
            "Fallback global SH bound for old checkpoints that did not "
            "save this value."
        ),
    )

    parser.add_argument(
        "--local_sh_bound",
        type=float,
        default=0.05,
        help=(
            "Fallback local SH bound for old checkpoints that did not "
            "save this value."
        ),
    )

    parser.add_argument(
        "--reverse",
        action="store_true",
        help=(
            "Render current optimized environment -> initial environment "
            "instead of initial -> current."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./fixed_voxel_environment_interpolation.mp4"),
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

    if args.num_frames < 2:
        raise ValueError("--num_frames must be at least 2.")

    device = torch.device("cuda")

    # Import optimizer module only to reuse its FNO loader and RGB compositor.
    import optimize_fixed_voxel_scene as optimizer_module
    from fixed_voxel_renderer import FixedVoxelRenderer

    # --------------------------------------------------------
    # Load frozen FNO checkpoints.
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

    voxel_scene = rebuild_fixed_voxel_scene_from_checkpoint(
        checkpoint_data=checkpoint_data,
        device=device,
        optimize_sh=True,
        default_local_sh_bound=args.local_sh_bound,
        default_global_sh_bound=args.global_sh_bound,
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

    # --------------------------------------------------------
    # Load actual 3DGS training camera.
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

    if not (0 <= args.camera_index < len(cameras)):
        raise ValueError(
            f"camera_index={args.camera_index} is invalid for "
            f"{len(cameras)} cameras."
        )

    camera = cameras[args.camera_index]

    # --------------------------------------------------------
    # Save current optimized SH raw deltas, then interpolate.
    # --------------------------------------------------------

    saved_global_raw_delta = (
        voxel_scene.lighting.raw_global_sh_delta
        .detach()
        .clone()
    )

    saved_local_raw_deltas = {
        key: block.raw_local_sh_delta.detach().clone()
        for key, block in voxel_scene.iter_blocks_with_keys()
    }

    initial_env_id = checkpoint_data.get(
        "metadata",
        {},
    ).get(
        "initial_env_id",
        "unknown",
    )

    print("Fixed voxel checkpoint:", args.scene_checkpoint)
    print("Blocks:", len(voxel_scene))
    print("Camera:", args.camera_index, camera.image_name)
    print("Initial environment ID:", initial_env_id)
    print(
        "Global SH bound:",
        voxel_scene.lighting.bound,
    )
    print(
        "First local SH bound:",
        next(
            iter(voxel_scene.blocks.values())
        ).local_sh_bound
        if len(voxel_scene) > 0
        else "none",
    )

    # t=0 means initial environment, t=1 means learned environment.
    t_values = np.linspace(
        0.0,
        1.0,
        num=args.num_frames,
        endpoint=True,
        dtype=np.float32,
    )

    if args.reverse:
        t_values = t_values[::-1].copy()

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
            for frame_index, t in enumerate(t_values):
                # Interpolate raw deltas. This is a smooth visual
                # interpolation; due to tanh bounds, it is not strictly
                # linear in effective SH coefficient space.
                voxel_scene.lighting.raw_global_sh_delta.copy_(
                    float(t) * saved_global_raw_delta
                )

                for key, block in voxel_scene.iter_blocks_with_keys():
                    block.raw_local_sh_delta.copy_(
                        float(t) * saved_local_raw_deltas[key]
                    )

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
                    f"[{frame_index + 1:03d}/{len(t_values)}] "
                    f"environment interpolation t={float(t):.3f}"
                )

    finally:
        # Restore checkpoint lighting state in memory before exiting.
        with torch.no_grad():
            voxel_scene.lighting.raw_global_sh_delta.copy_(
                saved_global_raw_delta
            )

            for key, block in voxel_scene.iter_blocks_with_keys():
                block.raw_local_sh_delta.copy_(
                    saved_local_raw_deltas[key]
                )

        writer.close()

    print("Done.")
    print("Wrote:", args.output)


if __name__ == "__main__":
    main()