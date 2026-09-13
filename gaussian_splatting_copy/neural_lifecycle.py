#!/usr/bin/env python

import copy
import math
from pathlib import Path

import torch
import torch.nn as nn


# ============================================================
# GRADIENT STATISTICS
# ============================================================

def make_gradient_stats(num_slices):
    """
    Create running gradient statistics.

    Call once before an optimization window.
    """
    return {
        "count": 0,
        "center_grad_norm": [0.0] * num_slices,
        "size_grad_norm": [0.0] * num_slices,
    }


def accumulate_gradient_stats(neural_scene, stats):
    """
    Call after loss.backward() and before optimizer.step().
    """
    if len(neural_scene.slices) != len(stats["center_grad_norm"]):
        raise RuntimeError(
            "Gradient-stat length does not match current slice count."
        )

    stats["count"] += 1

    for i, neural_slice in enumerate(neural_scene.slices):
        if neural_slice.center.grad is not None:
            stats["center_grad_norm"][i] += float(
                neural_slice.center.grad.detach().norm().item()
            )

        if neural_slice.raw_world_size.grad is not None:
            stats["size_grad_norm"][i] += float(
                neural_slice.raw_world_size.grad.detach().abs().item()
            )


def finalize_gradient_stats(stats):
    """
    Return average gradient norms.
    """
    count = max(stats["count"], 1)

    return {
        "center_grad_norm": [
            value / count
            for value in stats["center_grad_norm"]
        ],
        "size_grad_norm": [
            value / count
            for value in stats["size_grad_norm"]
        ],
    }


# ============================================================
# VISIBILITY / CONTRIBUTION STATISTICS
# ============================================================


@torch.no_grad()
def collect_batched_lifecycle_stats(
    cameras,
    neural_scene,
    fno_radius,
    min_patch_size_px=2.0,
    margin_px=64.0,
    flip_projection_y=False,
):
    """
    Cheap GPU-batched lifecycle statistics.

    No FNO render and no per-slice image render.

    Returns a list of dicts, one per stored slice.
    """
    slices = neural_scene.slices
    n = len(slices)

    device = slices[0].center.device
    dtype = slices[0].center.dtype

    centers = torch.stack(
        [s.center.detach() for s in slices],
        dim=0,
    )  # [N,3]

    sizes = torch.stack(
        [s.world_size.detach() for s in slices],
        dim=0,
    )  # [N]

    # Approximate current scalar opacity.
    opacities = torch.stack(
        [
            s.fno_values(
                shared_sh=neural_scene.global_sh
            )["opacity"].detach()
            for s in slices
        ],
        dim=0,
    )  # [N]

    visible_views = torch.zeros(
        n,
        device=device,
        dtype=torch.float32,
    )

    total_area = torch.zeros_like(visible_views)
    max_size = torch.zeros_like(visible_views)

    for camera in cameras:
        relative = (
            camera.camera_center[None, :] - centers
        )

        radius = torch.linalg.norm(
            relative,
            dim=1,
        ).clamp_min(1e-8)

        ones = torch.ones(
            n,
            1,
            device=device,
            dtype=dtype,
        )

        points_h = torch.cat(
            [centers, ones],
            dim=1,
        )

        clip = points_h @ camera.full_proj_transform
        ndc = clip[:, :3] / clip[:, 3:4].clamp_min(1e-8)

        image_w = float(camera.image_width)
        image_h = float(camera.image_height)

        pixel_x = (
            (ndc[:, 0] + 1.0)
            * 0.5
            * image_w
        )

        if flip_projection_y:
            pixel_y = (
                (ndc[:, 1] + 1.0)
                * 0.5
                * image_h
            )
        else:
            pixel_y = (
                (1.0 - ndc[:, 1])
                * 0.5
                * image_h
            )

        patch_size = (
            image_h
            * sizes
            * float(fno_radius)
            / radius
        )

        visible = (
            (ndc[:, 2] > 0.0)
            & torch.isfinite(patch_size)
            & (patch_size >= float(min_patch_size_px))
            & (pixel_x >= -float(margin_px))
            & (pixel_x <= image_w + float(margin_px))
            & (pixel_y >= -float(margin_px))
            & (pixel_y <= image_h + float(margin_px))
        )

        visible_f = visible.float()

        visible_views += visible_f
        total_area += visible_f * patch_size.square()
        max_size = torch.maximum(
            max_size,
            patch_size * visible_f,
        )

    mean_area = total_area / visible_views.clamp_min(1.0)

    # One CPU transfer for all slices, only at lifecycle intervals.
    visible_cpu = visible_views.cpu().tolist()
    mean_area_cpu = mean_area.cpu().tolist()
    max_size_cpu = max_size.cpu().tolist()
    opacity_cpu = opacities.cpu().tolist()

    return [
        {
            "visible_views": int(visible_cpu[i]),
            "mean_projected_area": float(mean_area_cpu[i]),
            "max_projected_size": float(max_size_cpu[i]),
            "opacity": float(opacity_cpu[i]),
        }
        for i in range(n)
    ]

# ============================================================
# PRUNING
# ============================================================


def prune_slices(neural_scene, prune_indices):
    """
    Remove selected slices from neural_scene.slices.

    Returns:
        kept_old_indices
    """
    prune_set = set(prune_indices)

    kept = []
    kept_old_indices = []

    for i, neural_slice in enumerate(neural_scene.slices):
        if i not in prune_set:
            kept.append(neural_slice)
            kept_old_indices.append(i)

    neural_scene.slices = nn.ModuleList(kept)

    return kept_old_indices
    

def choose_batched_prune_candidates(
    lifecycle_stats,
    max_candidates=16,
    min_visible_views=1,
):
    """
    Cheap ranking for possible pruning.

    Does NOT prune directly. It only identifies weak slices for
    optional expensive contribution testing.
    """
    scores = []

    for i, stats in enumerate(lifecycle_stats):
        if stats["visible_views"] < min_visible_views:
            score = -1.0
        else:
            score = (
                stats["opacity"]
                * stats["mean_projected_area"]
                * stats["visible_views"]
            )

        scores.append((score, i))

    scores.sort(key=lambda x: x[0])

    return [
        i
        for _, i in scores[:max_candidates]
    ]


# ============================================================
# SPLITTING
# ============================================================

def choose_batched_split_candidates(
    lifecycle_stats,
    gradient_summary,
    max_candidates=8,
    min_visible_views=1,
    min_projected_area=64.0,
):
    """
    Return high-value split candidates.
    """
    scored = []

    for i, stats in enumerate(lifecycle_stats):
        if stats["visible_views"] < min_visible_views:
            continue

        if stats["mean_projected_area"] < min_projected_area:
            continue

        grad_score = (
            gradient_summary["center_grad_norm"][i]
            + gradient_summary["size_grad_norm"][i]
        )

        score = (
            grad_score
            * stats["opacity"]
            * stats["mean_projected_area"]
        )

        scored.append((score, i))

    scored.sort(reverse=True)

    return [
        i
        for _, i in scored[:max_candidates]
    ]
    
    
def perturb_child_parameters(
    child,
    ctrl_noise=0.05,
    sigma_noise=0.10,
    color_noise=0.05,
    opacity_noise=0.05,
    roughness_noise=0.05,
):
    """
    Slightly perturb child raw parameters after splitting.

    Uses RGB material fields, not old hue/saturation fields.
    """
    with torch.no_grad():
        child.raw_ctrl.add_(
            ctrl_noise * torch.randn_like(child.raw_ctrl)
        )

        child.raw_sigma.add_(
            sigma_noise * torch.randn_like(child.raw_sigma)
        )

        child.raw_base_color_r.add_(
            color_noise * torch.randn_like(
                child.raw_base_color_r
            )
        )

        child.raw_base_color_g.add_(
            color_noise * torch.randn_like(
                child.raw_base_color_g
            )
        )

        child.raw_base_color_b.add_(
            color_noise * torch.randn_like(
                child.raw_base_color_b
            )
        )

        child.raw_opacity.add_(
            opacity_noise * torch.randn_like(
                child.raw_opacity
            )
        )

        if child.mode == "surface":
            child.raw_roughness.add_(
                roughness_noise * torch.randn_like(
                    child.raw_roughness
                )
            )

        if child.optimize_environment:
            child.raw_local_sh_delta.add_(
                0.01 * torch.randn_like(
                    child.raw_local_sh_delta
                )
            )


def split_slice(
    parent,
    offset_direction=None,
    child_scale=0.70,
    offset_fraction=0.25,
):
    """
    Replace one parent slice with two children.

    Children:
      - inherit mode, shape, material, local lighting parameters
      - get independent learnable tensors via deepcopy
      - move in opposite world directions
      - become smaller

    This does not perturb FNO parameters yet. That is intentional:
    first verify topology changes before adding more randomness.
    """
    child_a = copy.deepcopy(parent)
    child_b = copy.deepcopy(parent)

    device = parent.center.device
    dtype = parent.center.dtype

    if offset_direction is None:
        offset_direction = torch.randn(
            3,
            device=device,
            dtype=dtype,
        )

    offset_direction = offset_direction / (
        offset_direction.norm().clamp_min(1e-8)
    )

    parent_size = parent.world_size.detach()

    offset = (
        offset_fraction
        * parent_size
        * offset_direction
    )

    # Make each child independently learnable.
    child_a.center = nn.Parameter(
        (parent.center.detach() + offset).clone()
    )

    child_b.center = nn.Parameter(
        (parent.center.detach() - offset).clone()
    )

    # log(size * child_scale)
    child_a.raw_world_size = nn.Parameter(
        (
            parent.raw_world_size.detach()
            + math.log(child_scale)
        ).clone()
    )

    child_b.raw_world_size = nn.Parameter(
        (
            parent.raw_world_size.detach()
            + math.log(child_scale)
        ).clone()
    )
    
    # Reset regularization anchors for the new children.
    # Otherwise the children are penalized for being offset from
    # their parent immediately after splitting.
    child_a.initial_center.copy_(
        child_a.center.detach()
    )

    child_b.initial_center.copy_(
        child_b.center.detach()
    )

    child_a.initial_world_size.copy_(
        child_a.world_size.detach()
    )

    child_b.initial_world_size.copy_(
        child_b.world_size.detach()
    )
    
    perturb_child_parameters(child_a)
    perturb_child_parameters(child_b)

    return child_a, child_b


def split_top_slices(
    neural_scene,
    candidate_indices,
    gradient_stats,
    max_splits=1,
    max_total_slices=None,
    mode_selector=None,
):
    """
    Split preselected slice indices.

    candidate_indices should be ranked from strongest split
    candidate to weakest. These normally come from batched
    lifecycle statistics.

    Each split replaces one parent with two children, therefore
    each split adds one net slice.
    """
    old_slices = list(neural_scene.slices)
    old_count = len(old_slices)

    if old_count == 0:
        return []

    if (
        max_total_slices is not None
        and max_total_slices > 0
        and old_count >= max_total_slices
    ):
        return []

    if not candidate_indices:
        return []

    if max_total_slices is None or max_total_slices <= 0:
        available_splits = int(max_splits)
    else:
        available_splits = min(
            int(max_splits),
            int(max_total_slices) - old_count,
        )

    if available_splits <= 0:
        return []

    # Keep valid, unique candidate indices only.
    selected = []

    for index in candidate_indices:
        index = int(index)

        if index < 0 or index >= old_count:
            continue

        if index in selected:
            continue

        # Optional sanity: do not split completely inactive
        # gradients if gradient statistics are available.
        if gradient_stats is not None:
            center_grad = gradient_stats[
                "center_grad_norm"
            ][index]

            size_grad = gradient_stats[
                "size_grad_norm"
            ][index]

            if center_grad <= 0.0 and size_grad <= 0.0:
                continue

        selected.append(index)

        if len(selected) >= available_splits:
            break

    if not selected:
        return []

    selected_set = set(selected)

    print(
        "[split_top_slices] "
        f"old_count={old_count}, "
        f"selected={sorted(selected_set)}"
    )

    new_slices = []

    for parent_index, parent in enumerate(old_slices):
        if parent_index not in selected_set:
            new_slices.append(parent)
            continue

        child_a, child_b = split_slice(parent)

        if mode_selector is not None:
            chosen = mode_selector(
                parent_index=parent_index,
                child_a=child_a,
                child_b=child_b,
            )

            if (
                not isinstance(chosen, tuple)
                or len(chosen) != 2
            ):
                raise RuntimeError(
                    "mode_selector must return "
                    "(child_a, child_b)."
                )

            child_a, child_b = chosen

        new_slices.extend([
            child_a,
            child_b,
        ])

    expected_count = old_count + len(selected)

    if len(new_slices) != expected_count:
        raise RuntimeError(
            f"Split count mismatch: old={old_count}, "
            f"splits={len(selected)}, "
            f"expected={expected_count}, "
            f"got={len(new_slices)}"
        )

    neural_scene.slices = nn.ModuleList(new_slices)

    print(
        "[split_top_slices] "
        f"new_count={len(neural_scene.slices)}"
    )

    return selected

# ============================================================
# POINT-CLOUD INITIALIZATION
# ============================================================

@torch.no_grad()
def voxel_chunk_seeds(
    xyz,
    voxel_size=0.5,
    min_points=30,
    max_chunks=64,
):
    """
    Create initial chunk seeds from a point cloud.

    Returns a list of dictionaries:

        {
            "center": [3] tensor,
            "world_size": float,
            "num_points": int,
        }

    This is a simple voxel-grid initialization. It is not yet
    orientation-aware.
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(
            f"Expected xyz [N,3], got {tuple(xyz.shape)}"
        )

    voxel_index = torch.floor(
        xyz / float(voxel_size)
    ).to(torch.int64)

    unique_voxels, inverse, counts = torch.unique(
        voxel_index,
        dim=0,
        return_inverse=True,
        return_counts=True,
    )

    candidates = []

    # Make sure inverse is a simple [N] vector.
    inverse = inverse.reshape(-1)
    counts = counts.reshape(-1)

    for voxel_id in range(unique_voxels.shape[0]):
        count = int(counts[voxel_id].item())

        if count < min_points:
            continue

        # Explicit indices are safer than direct boolean indexing.
        point_indices = torch.nonzero(
            inverse == voxel_id,
            as_tuple=True,
        )[0]

        # Defensive guard: should agree with `count`, but avoid crash.
        if point_indices.numel() == 0:
            print(
                f"[WARN] voxel_id={voxel_id} has count={count} "
                "but no selected points; skipping."
            )
            continue

        points = xyz.index_select(
            0,
            point_indices,
        )

        center = points.median(
            dim=0,
        ).values

        candidates.append(
            {
                "center": center,
                "world_size": float(voxel_size),
                "num_points": count,
            }
        )

    # Favor dense chunks initially.
    candidates.sort(
        key=lambda x: x["num_points"],
        reverse=True,
    )
    
    if max_chunks is None or max_chunks <= 0:
        return candidates

    return candidates[:max_chunks]


# ============================================================
# CHECKPOINTING
# ============================================================

def serialize_slice(neural_slice):
    """
    Store RGB-conditioned slice construction metadata.

    State dict still stores exact raw parameters/buffers, but this
    metadata is useful for inspecting/reconstructing topology.
    """
    values = neural_slice.fno_values()

    return {
        "mode": neural_slice.mode,

        "center": neural_slice.center.detach().cpu(),

        "world_size": float(
            neural_slice.world_size.detach().cpu()
        ),

        "ctrl_values": values["ctrl"].detach().cpu(),

        "sigma": float(
            values["sigma"].detach().cpu()
        ),

        "base_color_r": float(
            values["base_color_r"].detach().cpu()
        ),

        "base_color_g": float(
            values["base_color_g"].detach().cpu()
        ),

        "base_color_b": float(
            values["base_color_b"].detach().cpu()
        ),

        "opacity": float(
            values["opacity"].detach().cpu()
        ),

        "roughness": float(
            values["roughness"].detach().cpu()
        ),

        "metallic": float(
            neural_slice.metallic_value.detach().cpu()
        ),

        "specular": float(
            neural_slice.specular_value.detach().cpu()
        ),

        "sh_values": neural_slice.initial_sh.detach().cpu(),

        "optimize_environment": bool(
            neural_slice.optimize_environment
        ),

        "local_sh_bound": float(
            neural_slice.local_sh_bound
        ),
    }


def save_neural_scene_checkpoint(
    path,
    neural_scene,
    optimizer,
    iteration,
    metadata=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    slice_specs = [
        serialize_slice(s)
        for s in neural_scene.slices
    ]

    torch.save(
        {
            "iteration": iteration,
            "scene_state": neural_scene.state_dict(),
            "optimizer_state": optimizer.state_dict(),

            "slice_specs": slice_specs,

            "initial_global_sh": (
                neural_scene.lighting.initial_global_sh
                .detach()
                .cpu()
            ),

            "metadata": metadata or {},
        },
        path,
    )
    


def load_neural_scene_checkpoint(
    path,
    neural_scene,
    optimizer=None,
    device="cuda",
):
    """
    Load state into an already-constructed neural_scene.

    Important:
      The scene must be rebuilt with the same number/order/types
      of slices before calling this function.
    """
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    neural_scene.load_state_dict(
        checkpoint["scene_state"]
    )

    if (
        optimizer is not None
        and "optimizer_state" in checkpoint
    ):
        optimizer.load_state_dict(
            checkpoint["optimizer_state"]
        )

    return checkpoint