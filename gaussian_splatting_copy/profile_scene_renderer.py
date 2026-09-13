#!/usr/bin/env python

import sys
from pathlib import Path
from argparse import ArgumentParser

import torch
from torch.profiler import (
    profile,
    record_function,
    ProfilerActivity,
)

# ============================================================
# PATH SETUP
# ============================================================

REPO_DIR = Path(__file__).resolve().parent
FNO_ROOT = REPO_DIR.parent

sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(FNO_ROOT))

# ============================================================
# IMPORT YOUR EXISTING OPTIMIZER COMPONENTS
# ============================================================

# This assumes your main file is named:
# optimize_neural_scene_random_init.py
from optimize_neural_scene_random_init import (
    load_fno_checkpoint,
    NeuralSceneRenderer,
    rebuild_neural_scene_from_checkpoint,
    image_loss,
    FNO_RADIUS,
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
# MAIN
# ============================================================

def main():
    parser = ArgumentParser(
        description=(
            "Profile one neural-scene renderer forward/backward "
            "step for a saved scene checkpoint."
        )
    )

    # Adds --source_path, --model_path, --resolution, etc.
    lp = ModelParams(parser)

    parser.add_argument(
        "--scene_checkpoint",
        type=str,
        required=True,
        help=(
            "Saved neural-scene checkpoint, e.g. "
            "checkpoint_iter_000500.pt"
        ),
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
    )

    parser.add_argument(
        "--fno_batch_size",
        type=int,
        default=8,
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
        help="Use 0 for no cap.",
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
        "--forward_only",
        action="store_true",
        help=(
            "Profile only forward rendering. "
            "Default profiles forward + loss + backward."
        ),
    )

    parser.add_argument(
        "--profile_dir",
        type=str,
        default="renderer_profiles",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this profiler.")

    device = torch.device("cuda")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass

    profile_dir = Path(args.profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)

    print("Using device:", device)
    print("Loading scene checkpoint:", args.scene_checkpoint)

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
            f"FNO latent dimensions disagree: "
            f"surface={surface_dim}, volume={volume_dim}"
        )

    print("FNO latent dimension:", surface_dim)

    # ========================================================
    # Load neural-scene checkpoint and rebuild slice topology
    # ========================================================

    scene_checkpoint = torch.load(
        args.scene_checkpoint,
        map_location=device,
        weights_only=False,
    )

    checkpoint_metadata = scene_checkpoint.get(
        "metadata",
        {},
    )

    optimize_sh = bool(
        checkpoint_metadata.get(
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
        "Rebuilt neural scene with slices:",
        len(neural_scene.slices),
    )

    # ========================================================
    # Create renderer
    # ========================================================

    max_patches_per_tile = (
        None
        if args.max_patches_per_tile <= 0
        else args.max_patches_per_tile
    )

    renderer = NeuralSceneRenderer(
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
        tile_size=args.tile_size,
        tile_roi_margin=args.tile_roi_margin,
        max_patches_per_tile=max_patches_per_tile,
        use_visibility_culling=False,
        use_fno_activation_checkpointing=True,
    ).to(device)

    print(
        "Renderer mode:",
        "tile-binned"
        if args.use_tile_renderer
        else "reference ROI",
    )

    # ========================================================
    # Load one 3DGS camera
    # ========================================================

    scene_args = lp.extract(args)

    if not scene_args.model_path:
        scene_args.model_path = str(
            REPO_DIR / "output" / "profile_scene"
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

    if args.camera_index < 0 or args.camera_index >= len(cameras):
        raise IndexError(
            f"camera_index={args.camera_index} invalid; "
            f"available={len(cameras)}"
        )

    camera = cameras[args.camera_index]

    target_rgb = camera.original_image.detach()

    if target_rgb.ndim == 3:
        target_rgb = target_rgb.unsqueeze(0)

    target_rgb = target_rgb.to(device)

    print("Camera:", camera.image_name)
    print(
        "Image size:",
        camera.image_width,
        "x",
        camera.image_height,
    )

    # ========================================================
    # Warmup
    #
    # Do not include allocator/setup overhead in measurement.
    # ========================================================

    print("Warmup render...")

    for parameter in neural_scene.parameters():
        parameter.grad = None

    torch.cuda.empty_cache()

    with torch.no_grad():
        _warmup_output, _ = renderer(
            camera,
            neural_scene,
        )

    del _warmup_output

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # ========================================================
    # Profile one step
    # ========================================================

    print("Profiling one renderer step...")

    activities = [
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
    ]

    for parameter in neural_scene.parameters():
        parameter.grad = None

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with record_function("FULL_RENDER_FORWARD"):
            prediction_rgba, _ = renderer(
                camera,
                neural_scene,
            )

        if not args.forward_only:
            with record_function("LOSS_AND_BACKWARD"):
                loss = image_loss(
                    prediction_rgba,
                    target_rgb.to(
                        dtype=prediction_rgba.dtype
                    ),
                )

                loss.backward()

    torch.cuda.synchronize()

    peak_allocated_gib = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    peak_reserved_gib = (
        torch.cuda.max_memory_reserved()
        / (1024 ** 3)
    )

    print()
    print("=" * 72)
    print("PROFILE SUMMARY")
    print("=" * 72)

    if args.forward_only:
        print("Mode: forward only")
    else:
        print("Mode: forward + backward")

    print(
        f"Peak allocated GPU memory: "
        f"{peak_allocated_gib:.3f} GiB"
    )

    print(
        f"Peak reserved GPU memory:  "
        f"{peak_reserved_gib:.3f} GiB"
    )

    print()
    print("=" * 72)
    print("TOP CUDA OPERATIONS")
    print("=" * 72)

    cuda_table = prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=30,
    )

    print(cuda_table)

    print()
    print("=" * 72)
    print("TOP CPU OPERATIONS")
    print("=" * 72)

    cpu_table = prof.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=30,
    )

    print(cpu_table)

    # ========================================================
    # Save reports
    # ========================================================

    suffix = (
        "tile"
        if args.use_tile_renderer
        else "reference"
    )

    mode = (
        "forward"
        if args.forward_only
        else "forward_backward"
    )

    cuda_report_path = profile_dir / (
        f"profile_{suffix}_{mode}_cuda.txt"
    )

    cpu_report_path = profile_dir / (
        f"profile_{suffix}_{mode}_cpu.txt"
    )

    trace_path = profile_dir / (
        f"profile_{suffix}_{mode}_trace.json"
    )

    memory_path = profile_dir / (
        f"profile_{suffix}_{mode}_memory.txt"
    )

    cuda_report_path.write_text(cuda_table)
    cpu_report_path.write_text(cpu_table)

    memory_path.write_text(
        f"scene_checkpoint: {args.scene_checkpoint}\n"
        f"camera: {camera.image_name}\n"
        f"slice_count: {len(neural_scene.slices)}\n"
        f"renderer: {suffix}\n"
        f"mode: {mode}\n"
        f"fno_batch_size: {args.fno_batch_size}\n"
        f"placement_batch_size: {args.placement_batch_size}\n"
        f"tile_size: {args.tile_size}\n"
        f"peak_allocated_gib: {peak_allocated_gib:.6f}\n"
        f"peak_reserved_gib: {peak_reserved_gib:.6f}\n"
    )

    prof.export_chrome_trace(
        str(trace_path)
    )

    print()
    print("Saved CUDA report:", cuda_report_path)
    print("Saved CPU report:", cpu_report_path)
    print("Saved memory report:", memory_path)
    print("Saved Chrome trace:", trace_path)

    # Avoid retaining a large final graph after profiling.
    del prediction_rgba

    if not args.forward_only:
        del loss


if __name__ == "__main__":
    main()