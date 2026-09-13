#!/usr/bin/env python

import sys
import time
from pathlib import Path
from argparse import ArgumentParser

import torch


# ============================================================
# PATH SETUP
# ============================================================

REPO_DIR = Path(__file__).resolve().parent
FNO_ROOT = REPO_DIR.parent

sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(FNO_ROOT))


# ============================================================
# IMPORTS FROM YOUR MAIN OPTIMIZER
# ============================================================

# Change this only if your optimizer file has another name.
from optimize_neural_scene_random_init import (
    FNO_RADIUS,
    load_fno_checkpoint,
    rebuild_neural_scene_from_checkpoint,
    NeuralSceneRenderer,
    NeuralScene,
    camera_to_slice_pose,
    project_world_point,
    build_tensor_fno_vector,
    run_fno_in_chunks,
    image_loss,
)

from tiled_roi_renderer import (
    render_rois_to_tiled_canvas,
)

from scene import Scene, GaussianModel
from arguments import ModelParams


# ============================================================
# DEFAULT MODEL PATHS
# ============================================================

SURFACE_CHECKPOINT = (
    FNO_ROOT / "fno_premult_surface_epoch128_color.pt"
)

VOLUME_CHECKPOINT = (
    FNO_ROOT / "fno_premult_volume_epoch016_color.pt"
)


# ============================================================
# TIMING HELPERS
# ============================================================

def time_cuda_and_wall(fn):
    """
    Run fn() once and report:
      - wall time: Python + CPU + GPU + synchronization
      - CUDA event time: GPU work between events

    Returns:
        output, wall_ms, cuda_ms
    """
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    wall_start = time.perf_counter()

    start_event.record()
    output = fn()
    end_event.record()

    torch.cuda.synchronize()

    wall_end = time.perf_counter()

    wall_ms = (wall_end - wall_start) * 1000.0
    cuda_ms = start_event.elapsed_time(end_event)

    return output, wall_ms, cuda_ms


def memory_stats_gib():
    return {
        "allocated": (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        ),
        "reserved": (
            torch.cuda.max_memory_reserved()
            / (1024 ** 3)
        ),
    }


def print_measurement(name, wall_ms, cuda_ms, memory):
    print()
    print("=" * 72)
    print(name)
    print("=" * 72)
    print(f"Wall time:        {wall_ms:.2f} ms")
    print(f"CUDA time:        {cuda_ms:.2f} ms")
    print(
        f"CPU/Python gap:   "
        f"{wall_ms - cuda_ms:.2f} ms"
    )
    print(
        f"Peak allocated:   "
        f"{memory['allocated']:.3f} GiB"
    )
    print(
        f"Peak reserved:    "
        f"{memory['reserved']:.3f} GiB"
    )


# ============================================================
# RENDERER SETUP
# ============================================================

def build_renderer(
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
        tile_size=args.tile_size,
        tile_roi_margin=args.tile_roi_margin,
        max_patches_per_tile=max_patches_per_tile,
        use_visibility_culling=False,
        use_fno_activation_checkpointing=(
            not args.disable_fno_activation_checkpointing
        ),
    ).to(device)


# ============================================================
# PHASE 1: BUILD SLICE RECORDS
# ============================================================

def build_slice_records(
    renderer,
    camera,
    neural_scene,
):
    """
    Build all FNO input vectors and projected patch metadata.

    This is equivalent to Phase A in your batched renderer,
    but separated so it can be timed independently.
    """
    device = camera.camera_center.device
    canvas_h = int(camera.image_height)

    records = []

    for slice_index, neural_slice in enumerate(
        neural_scene.slices
    ):
        phi, theta, actual_radius, relative = (
            camera_to_slice_pose(
                camera,
                neural_slice.center,
            )
        )

        if neural_slice.mode == "surface":
            param_mean = renderer.surface_mean
            param_std = renderer.surface_std
        elif neural_slice.mode == "volume":
            param_mean = renderer.volume_mean
            param_std = renderer.volume_std
        else:
            raise ValueError(
                f"Unknown mode: {neural_slice.mode}"
            )

        param_vec = build_tensor_fno_vector(
            neural_slice=neural_slice,
            param_mean=param_mean,
            param_std=param_std,
            phi=phi,
            theta=theta,
            fno_radius=renderer.fno_radius,
            device=device,
            shared_sh=neural_scene.global_sh,
        )

        center_x, center_y, center_depth = (
            project_world_point(
                camera,
                neural_slice.center,
                flip_projection_y=renderer.flip_projection_y,
            )
        )

        patch_size = (
            float(canvas_h)
            * neural_slice.world_size
            * renderer.fno_radius
            / actual_radius
        )

        records.append(
            {
                "slice_index": slice_index,
                "mode": neural_slice.mode,
                "param_vec": param_vec,
                "center_x": center_x,
                "center_y": center_y,
                "patch_size": patch_size,
                "actual_radius": actual_radius,
                "center_depth": center_depth,
                "phi": phi,
                "theta": theta,
                "relative": relative,
            }
        )

    return records


# ============================================================
# PHASE 2: FNO ONLY
# ============================================================

def evaluate_fnos_from_records(
    renderer,
    records,
    use_activation_checkpointing,
):
    """
    Evaluate FNO patches only.

    Returns:
        all_patches: [N,4,64,64]
    """
    num_slices = len(records)

    surface_indices = [
        i
        for i, record in enumerate(records)
        if record["mode"] == "surface"
    ]

    volume_indices = [
        i
        for i, record in enumerate(records)
        if record["mode"] == "volume"
    ]

    patches_by_index = [None] * num_slices

    if surface_indices:
        surface_params = torch.cat(
            [
                records[i]["param_vec"]
                for i in surface_indices
            ],
            dim=0,
        )

        surface_patches = run_fno_in_chunks(
            model=renderer.surface_model,
            params=surface_params,
            batch_size=renderer.fno_batch_size,
            use_activation_checkpointing=(
                use_activation_checkpointing
            ),
        )

        if renderer.flip_fno_vertical:
            surface_patches = torch.flip(
                surface_patches,
                dims=[2],
            )

        for local_index, record_index in enumerate(
            surface_indices
        ):
            patches_by_index[record_index] = (
                surface_patches[
                    local_index:local_index + 1
                ]
            )

    if volume_indices:
        volume_params = torch.cat(
            [
                records[i]["param_vec"]
                for i in volume_indices
            ],
            dim=0,
        )

        volume_patches = run_fno_in_chunks(
            model=renderer.volume_model,
            params=volume_params,
            batch_size=renderer.fno_batch_size,
            use_activation_checkpointing=(
                use_activation_checkpointing
            ),
        )

        if renderer.flip_fno_vertical:
            volume_patches = torch.flip(
                volume_patches,
                dims=[2],
            )

        for local_index, record_index in enumerate(
            volume_indices
        ):
            patches_by_index[record_index] = (
                volume_patches[
                    local_index:local_index + 1
                ]
            )

    if any(patch is None for patch in patches_by_index):
        raise RuntimeError(
            "At least one slice did not receive an FNO patch."
        )

    return torch.cat(
        patches_by_index,
        dim=0,
    )


# ============================================================
# PHASE 3: RASTERIZATION ONLY
# ============================================================

def rasterize_from_records(
    renderer,
    camera,
    records,
    patches,
):
    """
    Rasterize already-created patches.

    The caller should pass detached patches if measuring only
    rasterization memory/time independently of FNO backward.
    """
    canvas_h = int(camera.image_height)
    canvas_w = int(camera.image_width)

    center_x = torch.stack(
        [
            record["center_x"]
            for record in records
        ],
        dim=0,
    )

    center_y = torch.stack(
        [
            record["center_y"]
            for record in records
        ],
        dim=0,
    )

    patch_size = torch.stack(
        [
            record["patch_size"]
            for record in records
        ],
        dim=0,
    )

    depths = torch.stack(
        [
            record["actual_radius"]
            for record in records
        ],
        dim=0,
    )

    if renderer.use_tile_renderer:
        return renderer.tile_renderer(
            patches=patches,
            center_x=center_x,
            center_y=center_y,
            patch_size=patch_size,
            depths=depths,
            image_height=canvas_h,
            image_width=canvas_w,
        )

    # Reference ROI renderer expects globally sorted far -> near.
    sort_indices = torch.argsort(
        depths.detach(),
        descending=True,
    )

    return render_rois_to_tiled_canvas(
        sorted_patches=patches[sort_indices],
        sorted_center_x=center_x[sort_indices],
        sorted_center_y=center_y[sort_indices],
        sorted_patch_sizes=patch_size[sort_indices],
        canvas_height=canvas_h,
        canvas_width=canvas_w,
        tile_size=128,
        margin_pixels=16,
        max_roi_side=512,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = ArgumentParser(
        description=(
            "Measure FNO-only, rasterizer-only, and full "
            "forward/backward costs for a neural-scene checkpoint."
        )
    )

    lp = ModelParams(parser)

    parser.add_argument(
        "--scene_checkpoint",
        type=str,
        required=True,
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
        "--disable_fno_activation_checkpointing",
        action="store_true",
    )

    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="component_renderer_benchmarks",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

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

    print("Using device:", device)

    # ========================================================
    # FNO models
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
            f"Latent dimension mismatch: "
            f"{surface_dim} vs {volume_dim}"
        )

    # ========================================================
    # Scene checkpoint
    # ========================================================

    checkpoint_data = torch.load(
        args.scene_checkpoint,
        map_location=device,
        weights_only=False,
    )

    optimize_sh = bool(
        checkpoint_data.get("metadata", {}).get(
            "optimize_sh",
            False,
        )
    )

    neural_scene = rebuild_neural_scene_from_checkpoint(
        checkpoint=checkpoint_data,
        device=device,
        optimize_sh=optimize_sh,
    )

    print(
        "Loaded slice count:",
        len(neural_scene.slices),
    )

    # ========================================================
    # Renderer
    # ========================================================

    renderer = build_renderer(
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
    # 3DGS camera
    # ========================================================

    scene_args = lp.extract(args)

    if not scene_args.model_path:
        scene_args.model_path = str(
            REPO_DIR / "output" / "component_benchmark_scene"
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

    if args.camera_index >= len(cameras):
        raise IndexError(
            f"camera_index={args.camera_index}, "
            f"available={len(cameras)}"
        )

    camera = cameras[args.camera_index]

    target_rgb = camera.original_image.detach()

    if target_rgb.ndim == 3:
        target_rgb = target_rgb.unsqueeze(0)

    target_rgb = target_rgb.to(device)

    print("Camera:", camera.image_name)
    print(
        "Resolution:",
        camera.image_width,
        "x",
        camera.image_height,
    )

    # ========================================================
    # Warmup
    # ========================================================

    print("Warmup...")

    for _ in range(args.warmup_steps):
        with torch.no_grad():
            warmup_output, _ = renderer(
                camera,
                neural_scene,
            )

        del warmup_output

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # ========================================================
    # Component A: record construction
    # ========================================================

    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        records, prep_wall_ms, prep_cuda_ms = (
            time_cuda_and_wall(
                lambda: build_slice_records(
                    renderer=renderer,
                    camera=camera,
                    neural_scene=neural_scene,
                )
            )
        )

    prep_memory = memory_stats_gib()

    print_measurement(
        "A. RECORD / VECTOR / PROJECTION PREPARATION",
        prep_wall_ms,
        prep_cuda_ms,
        prep_memory,
    )

    print("Record count:", len(records))

    # ========================================================
    # Component B: FNO forward only
    # ========================================================

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        patches, fno_wall_ms, fno_cuda_ms = (
            time_cuda_and_wall(
                lambda: evaluate_fnos_from_records(
                    renderer=renderer,
                    records=records,
                    use_activation_checkpointing=False,
                )
            )
        )

    fno_memory = memory_stats_gib()

    print_measurement(
        "B. FNO FORWARD ONLY",
        fno_wall_ms,
        fno_cuda_ms,
        fno_memory,
    )

    print("Patch tensor shape:", tuple(patches.shape))

    # ========================================================
    # Component C: rasterizer only
    #
    # Patches are detached so no FNO graph is retained.
    # ========================================================

    detached_patches = patches.detach()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        raster_rgba, raster_wall_ms, raster_cuda_ms = (
            time_cuda_and_wall(
                lambda: rasterize_from_records(
                    renderer=renderer,
                    camera=camera,
                    records=records,
                    patches=detached_patches,
                )
            )
        )

    raster_memory = memory_stats_gib()

    print_measurement(
        "C. RASTERIZER / PATCH COMPOSITING ONLY",
        raster_wall_ms,
        raster_cuda_ms,
        raster_memory,
    )

    # ========================================================
    # Component D: real full forward + backward
    # ========================================================

    for parameter in neural_scene.parameters():
        parameter.grad = None

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    def full_step():
        prediction_rgba, _ = renderer(
            camera,
            neural_scene,
        )

        loss = image_loss(
            prediction_rgba,
            target_rgb.to(
                dtype=prediction_rgba.dtype
            ),
        )

        loss.backward()

        return (
            prediction_rgba.detach(),
            loss.detach(),
        )

    (full_output, full_loss), full_wall_ms, full_cuda_ms = (
        time_cuda_and_wall(full_step)
    )

    full_memory = memory_stats_gib()

    print_measurement(
        "D. FULL FORWARD + LOSS + BACKWARD",
        full_wall_ms,
        full_cuda_ms,
        full_memory,
    )

    print("Full-step loss:", float(full_loss.cpu()))

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 72)
    print("COMPONENT SUMMARY")
    print("=" * 72)

    print(
        f"Record prep CUDA: {prep_cuda_ms:.2f} ms"
    )

    print(
        f"FNO-only CUDA:    {fno_cuda_ms:.2f} ms"
    )

    print(
        f"Raster-only CUDA: {raster_cuda_ms:.2f} ms"
    )

    print(
        f"Full CUDA:        {full_cuda_ms:.2f} ms"
    )

    print(
        f"Estimated backward/recompute CUDA cost: "
        f"{full_cuda_ms - fno_cuda_ms - raster_cuda_ms:.2f} ms"
    )

    print(
        "Note: this estimate is approximate because full "
        "forward/backward uses activation checkpointing while "
        "the isolated FNO-only measurement does not."
    )

    print()
    print(
        f"Peak FNO-only allocation: "
        f"{fno_memory['allocated']:.3f} GiB"
    )

    print(
        f"Peak raster-only allocation: "
        f"{raster_memory['allocated']:.3f} GiB"
    )

    print(
        f"Peak full-step allocation: "
        f"{full_memory['allocated']:.3f} GiB"
    )

    # ========================================================
    # Save report
    # ========================================================

    renderer_name = (
        "tile"
        if args.use_tile_renderer
        else "reference"
    )

    report_path = output_dir / (
        f"component_benchmark_{renderer_name}.txt"
    )

    report_path.write_text(
        "\n".join(
            [
                f"scene_checkpoint: {args.scene_checkpoint}",
                f"camera: {camera.image_name}",
                f"slice_count: {len(neural_scene.slices)}",
                f"renderer: {renderer_name}",
                (
                    "resolution: "
                    f"{camera.image_width}x{camera.image_height}"
                ),
                f"fno_batch_size: {args.fno_batch_size}",
                (
                    "activation_checkpointing_full_step: "
                    f"{not args.disable_fno_activation_checkpointing}"
                ),
                "",
                "[record_preparation]",
                f"wall_ms: {prep_wall_ms:.6f}",
                f"cuda_ms: {prep_cuda_ms:.6f}",
                (
                    "peak_allocated_gib: "
                    f"{prep_memory['allocated']:.6f}"
                ),
                "",
                "[fno_forward_only]",
                f"wall_ms: {fno_wall_ms:.6f}",
                f"cuda_ms: {fno_cuda_ms:.6f}",
                (
                    "peak_allocated_gib: "
                    f"{fno_memory['allocated']:.6f}"
                ),
                "",
                "[raster_only]",
                f"wall_ms: {raster_wall_ms:.6f}",
                f"cuda_ms: {raster_cuda_ms:.6f}",
                (
                    "peak_allocated_gib: "
                    f"{raster_memory['allocated']:.6f}"
                ),
                "",
                "[full_forward_backward]",
                f"wall_ms: {full_wall_ms:.6f}",
                f"cuda_ms: {full_cuda_ms:.6f}",
                f"loss: {float(full_loss.cpu()):.8f}",
                (
                    "peak_allocated_gib: "
                    f"{full_memory['allocated']:.6f}"
                ),
                (
                    "peak_reserved_gib: "
                    f"{full_memory['reserved']:.6f}"
                ),
            ]
        )
        + "\n"
    )

    print()
    print("Saved report:", report_path)

    del records
    del patches
    del detached_patches
    del raster_rgba
    del full_output
    del full_loss


if __name__ == "__main__":
    main()