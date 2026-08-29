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
def collect_slice_statistics(
    renderer,
    neural_scene,
    cameras,
):
    """
    Render each slice individually over selected cameras and collect:

      max_alpha:
          Maximum alpha observed over all cameras.

      mean_alpha_mass:
          Mean sum(alpha) over full output canvases.

      mean_patch_size:
          Mean rendered square patch size in pixels.

      visible_views:
          Number of views where max alpha exceeds threshold.

    This is a forward-only diagnostic/statistics pass.
    """
    num_slices = len(neural_scene.slices)

    stats = [
        {
            "max_alpha": 0.0,
            "alpha_mass_sum": 0.0,
            "patch_size_sum": 0.0,
            "visible_views": 0,
            "num_views": 0,
        }
        for _ in range(num_slices)
    ]

    for camera in cameras:
        shared_sh = neural_scene.global_sh

        for i, neural_slice in enumerate(neural_scene.slices):
            layer, info = renderer.render_slice(
                camera=camera,
                neural_slice=neural_slice,
                shared_sh=shared_sh,
            )

            alpha = layer[:, 3:4]

            max_alpha = float(alpha.max().item())
            alpha_mass = float(alpha.sum().item())
            patch_size = float(info["patch_size"].item())

            stats[i]["max_alpha"] = max(
                stats[i]["max_alpha"],
                max_alpha,
            )

            stats[i]["alpha_mass_sum"] += alpha_mass
            stats[i]["patch_size_sum"] += patch_size
            stats[i]["num_views"] += 1

            if max_alpha > 0.01:
                stats[i]["visible_views"] += 1

    for s in stats:
        n = max(s["num_views"], 1)

        s["mean_alpha_mass"] = (
            s["alpha_mass_sum"] / n
        )

        s["mean_patch_size"] = (
            s["patch_size_sum"] / n
        )

    return stats


# ============================================================
# PRUNING
# ============================================================

def choose_prune_indices(
    slice_stats,
    min_max_alpha=0.01,
    min_alpha_mass=5.0,
    min_visible_views=1,
    min_remaining_slices=1,
):
    """
    Decide which slices are safe to prune.

    Conservative initial rule:
      prune if a slice is effectively invisible in all views.
    """
    candidate_indices = []

    for i, s in enumerate(slice_stats):
        invisible = (
            s["max_alpha"] < min_max_alpha
            or s["mean_alpha_mass"] < min_alpha_mass
            or s["visible_views"] < min_visible_views
        )

        if invisible:
            candidate_indices.append(i)

    max_prunable = max(
        0,
        len(slice_stats) - min_remaining_slices,
    )

    return candidate_indices[:max_prunable]


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


# ============================================================
# SPLITTING
# ============================================================

def score_slices_for_split(
    slice_stats,
    gradient_stats,
    min_alpha_mass=10.0,
    min_patch_size=40.0,
):
    """
    Compute a simple split score.

    High score means:
      - visible enough to matter
      - large on screen
      - high position/size gradient
    """
    scores = []

    for i, s in enumerate(slice_stats):
        if (
            s["mean_alpha_mass"] < min_alpha_mass
            or s["mean_patch_size"] < min_patch_size
        ):
            scores.append(float("-inf"))
            continue

        grad_score = (
            gradient_stats["center_grad_norm"][i]
            + gradient_stats["size_grad_norm"][i]
        )

        score = (
            grad_score
            * math.sqrt(max(s["mean_alpha_mass"], 1.0))
            * math.sqrt(max(s["mean_patch_size"], 1.0))
        )

        scores.append(score)

    return scores
    
    
def perturb_child_parameters(
    child,
    ctrl_noise=0.05,
    sigma_noise=0.10,
    hue_noise=0.02,
    opacity_noise=0.05,
    roughness_noise=0.05,
):
    """
    Add small noise in RAW parameter space after a split.

    This breaks child symmetry while preserving the parent's
    overall appearance/shape.
    """
    with torch.no_grad():
        child.raw_ctrl.add_(
            ctrl_noise * torch.randn_like(child.raw_ctrl)
        )

        child.raw_sigma.add_(
            sigma_noise * torch.randn_like(child.raw_sigma)
        )

        child.raw_hue.add_(
            hue_noise * torch.randn_like(child.raw_hue)
        )

        child.raw_saturation.add_(
            0.05 * torch.randn_like(child.raw_saturation)
        )

        child.raw_opacity.add_(
            opacity_noise * torch.randn_like(child.raw_opacity)
        )

        if child.mode == "surface":
            child.raw_roughness.add_(
                roughness_noise
                * torch.randn_like(child.raw_roughness)
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
    slice_stats,
    gradient_stats,
    max_splits=1,
    max_total_slices=64,
    min_alpha_mass=10.0,
    min_patch_size=40.0,
    mode_selector=None,
):
    """
    Split highest-scoring parent slices.

    Each selected parent is replaced by two children.
    Every non-selected slice is preserved.

    Returns:
        selected: list of original parent indices that were split.
    """
    old_slices = list(neural_scene.slices)
    num_old_slices = len(old_slices)

    if num_old_slices == 0:
        return []

    if num_old_slices >= max_total_slices:
        return []

    scores = score_slices_for_split(
        slice_stats=slice_stats,
        gradient_stats=gradient_stats,
        min_alpha_mass=min_alpha_mass,
        min_patch_size=min_patch_size,
    )

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    # Each parent replaced by two children adds one net slice.
    available_splits = max_total_slices - num_old_slices
    num_to_select = min(
        int(max_splits),
        available_splits,
    )

    selected = []

    for index in ranked_indices:
        if len(selected) >= num_to_select:
            break

        score = scores[index]

        if not math.isfinite(score):
            continue

        if score <= 0.0:
            continue

        selected.append(index)

    if not selected:
        return []

    selected_set = set(selected)

    print(
        "[split_top_slices] "
        f"old_count={num_old_slices}, "
        f"selected={sorted(selected_set)}"
    )

    new_slices = []

    # IMPORTANT:
    # This loop preserves every non-selected old slice.
    for parent_index, parent in enumerate(old_slices):
        if parent_index not in selected_set:
            new_slices.append(parent)
            continue

        # Parent is selected: replace it with two children.
        child_a, child_b = split_slice(parent)

        if mode_selector is not None:
            selected_children = mode_selector(
                parent_index=parent_index,
                child_a=child_a,
                child_b=child_b,
            )

            if (
                not isinstance(selected_children, tuple)
                or len(selected_children) != 2
            ):
                raise RuntimeError(
                    "mode_selector must return "
                    "(child_a, child_b)."
                )

            child_a, child_b = selected_children

        new_slices.append(child_a)
        new_slices.append(child_b)

    expected_count = num_old_slices + len(selected)

    if len(new_slices) != expected_count:
        raise RuntimeError(
            "Split produced wrong number of slices: "
            f"old={num_old_slices}, "
            f"num_split={len(selected)}, "
            f"expected={expected_count}, "
            f"actual={len(new_slices)}"
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

    for voxel_id in range(unique_voxels.shape[0]):
        count = int(counts[voxel_id].item())

        if count < min_points:
            continue

        points = xyz[inverse == voxel_id]

        center = points.median(
            dim=0
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

    return candidates[:max_chunks]


# ============================================================
# CHECKPOINTING
# ============================================================

def save_neural_scene_checkpoint(
    path,
    neural_scene,
    optimizer,
    iteration,
    metadata=None,
):
    """
    Save model/optimizer state.

    `metadata` should contain structural information needed to
    reconstruct the same slice list before loading state_dict.
    """
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "iteration": iteration,
            "scene_state": neural_scene.state_dict(),
            "optimizer_state": optimizer.state_dict(),
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