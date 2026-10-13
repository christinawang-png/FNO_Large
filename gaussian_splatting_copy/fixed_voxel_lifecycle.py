#!/usr/bin/env python
"""
Lifecycle utilities for fixed voxel scenes.

This file does not render or alter optimization parameters directly.

It provides:
    - running gradient statistics;
    - dead-block candidate selection;
    - split candidate selection;
    - simple lifecycle summaries.

Topology operations themselves are in fixed_voxel_scene.py:

    scene.mark_dead(key)
    scene.remove_block(key)
    split_scene_block(scene, parent_key)

After any hard removal or split, rebuild the optimizer because the
scene ModuleDict topology has changed.
"""

from __future__ import annotations

import torch


# ============================================================
# GRADIENT STATISTICS
# ============================================================

def make_voxel_gradient_stats(voxel_scene):
    """
    Create a gradient-statistics object matching current voxel topology.

    Call once at initialization and again after any split/hard removal.
    """
    return {
        "count": 0,
        "block_grad_norm": {
            key: 0.0
            for key, _ in voxel_scene.iter_blocks_with_keys()
        },
        "shape_grad_norm": {
            key: 0.0
            for key, _ in voxel_scene.iter_blocks_with_keys()
        },
        "appearance_grad_norm": {
            key: 0.0
            for key, _ in voxel_scene.iter_blocks_with_keys()
        },
        "alive_grad_norm": {
            key: 0.0
            for key, _ in voxel_scene.iter_blocks_with_keys()
        },
        "mode_grad_norm": {
            key: 0.0
            for key, _ in voxel_scene.iter_blocks_with_keys()
        },
        "alive_grad_signed": {
            key: 0.0
            for key, _ in voxel_scene.iter_blocks_with_keys()
        },
    }


def _grad_norm(parameter):
    """Return scalar L2 norm of a parameter gradient, or zero."""
    if parameter.grad is None:
        return 0.0

    return float(parameter.grad.detach().norm().item())


def accumulate_voxel_gradient_stats(voxel_scene, stats):
    """
    Call after all loss.backward() calls and before optimizer.step().

    Since fixed voxels do not optimize center/size, splitting importance
    comes from shape, appearance, alive, and mode gradients.
    """
    current_keys = {
        key
        for key, _ in voxel_scene.iter_blocks_with_keys()
    }

    tracked_keys = set(stats["block_grad_norm"].keys())

    if current_keys != tracked_keys:
        raise RuntimeError(
            "Voxel topology changed but gradient stats were not rebuilt."
        )

    stats["count"] += 1

    for key, block in voxel_scene.iter_blocks_with_keys():
        shape_norm = (
            _grad_norm(block.raw_ctrl)
            + _grad_norm(block.raw_sigma)
        )

        appearance_norm = (
            _grad_norm(block.raw_base_color_r)
            + _grad_norm(block.raw_base_color_g)
            + _grad_norm(block.raw_base_color_b)
            + _grad_norm(block.raw_opacity)
            + _grad_norm(block.raw_roughness)
        )

        alive_norm = _grad_norm(block.raw_alive_logit)
        mode_norm = _grad_norm(block.raw_mode_logit)
        
        if block.raw_alive_logit.grad is None:
            alive_signed = 0.0
        else:
            alive_signed = float(
                block.raw_alive_logit.grad.detach().item()
            )

        # A combined score useful for general diagnostics.
        block_norm = (
            shape_norm
            + appearance_norm
            + alive_norm
            + mode_norm
        )

        stats["shape_grad_norm"][key] += shape_norm
        stats["appearance_grad_norm"][key] += appearance_norm
        stats["alive_grad_norm"][key] += alive_norm
        stats["mode_grad_norm"][key] += mode_norm
        stats["block_grad_norm"][key] += block_norm
        stats["alive_grad_signed"][key] += alive_signed


def finalize_voxel_gradient_stats(stats):
    """
    Convert accumulated sums into average gradients per optimization step.
    """
    count = max(int(stats["count"]), 1)

    output = {"count": int(stats["count"])}

    for stat_name in [
        "block_grad_norm",
        "shape_grad_norm",
        "appearance_grad_norm",
        "alive_grad_norm",
        "alive_grad_signed",
        "mode_grad_norm",
    ]:
        output[stat_name] = {
            key: value / count
            for key, value in stats[stat_name].items()
        }

    return output


# ============================================================
# BLOCK STATE SUMMARIES
# ============================================================

@torch.no_grad()
def collect_voxel_lifecycle_stats(voxel_scene):
    """
    Return detached state summaries for each block.

    This is cheap and does not render any cameras.
    """
    rows = []

    for key, block in voxel_scene.iter_blocks_with_keys():
        rows.append(
            {
                "key": key,
                "level": int(block.level),
                "grid_index": tuple(block.grid_index),
                "world_size": float(block.world_size.detach().cpu()),
                "alive_weight": float(block.alive_weight.detach().cpu()),
                "surface_weight": float(
                    block.surface_weight.detach().cpu()
                ),
                "volume_weight": float(
                    block.volume_weight.detach().cpu()
                ),
                "opacity": float(
                    block.common_values(
                        shared_sh=voxel_scene.global_sh
                    )["opacity"].detach().cpu()
                ),
            }
        )

    return rows


# ============================================================
# SOFT DEATH / HARD PRUNE CANDIDATES
# ============================================================

def choose_dead_block_keys(
    lifecycle_stats,
    max_candidates=4,
    alive_threshold=0.05,
):
    """
    Select blocks whose learned alive weight is very low.

    This function only returns keys. It does not alter scene topology.

    Suggested workflow:
        1. Mark weak blocks soft-dead through alive optimization.
        2. At a lifecycle interval, call this function.
        3. Optionally hard-remove selected blocks.
        4. Rebuild optimizer after hard removal.
    """
    dead = [
        row
        for row in lifecycle_stats
        if row["alive_weight"] <= float(alive_threshold)
    ]

    dead.sort(
        key=lambda row: row["alive_weight"]
    )

    return [
        row["key"]
        for row in dead[:int(max_candidates)]
    ]


# ============================================================
# SPLIT CANDIDATES
# ============================================================

def choose_voxel_split_candidates(
    lifecycle_stats,
    gradient_summary,
    max_candidates=8,
    min_alive_weight=0.25,
    min_world_size=0.0,
    min_shape_gradient=0.0,
    max_level=None,
):
    """
    Rank fixed voxel blocks for splitting.

    The score intentionally does NOT favor large blocks merely because
    they are large. It is based mainly on shape/appearance optimization
    pressure and current block importance.

    Score:
        (shape_grad + appearance_grad + mode_grad)
        * alive_weight
        * opacity

    Parameters
    ----------
    min_alive_weight:
        Do not split mostly-dead blocks.

    min_world_size:
        Optional world-space floor. Set 0 to allow all levels to split.
        Later, use this to prevent endless octree refinement.

    min_shape_gradient:
        Ignore blocks with negligible shape gradient.
    """
    rows_by_key = {
        row["key"]: row
        for row in lifecycle_stats
    }

    scored = []

    for key, row in rows_by_key.items():
        alive_weight = float(row["alive_weight"])
        world_size = float(row["world_size"])
        opacity = float(row["opacity"])
        level = int(row["level"])

        if alive_weight < float(min_alive_weight):
            continue

        if world_size <= float(min_world_size):
            continue

        if max_level is not None and level >= int(max_level):
            continue

        shape_grad = float(
            gradient_summary["shape_grad_norm"].get(key, 0.0)
        )

        appearance_grad = float(
            gradient_summary["appearance_grad_norm"].get(key, 0.0)
        )

        mode_grad = float(
            gradient_summary["mode_grad_norm"].get(key, 0.0)
        )

        if shape_grad < float(min_shape_gradient):
            continue

        # No direct size weighting here.
        score = (
            (shape_grad + appearance_grad + mode_grad)
            * alive_weight
            * max(opacity, 1e-6)
        )

        scored.append((score, key))

    scored.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        key
        for _, key in scored[:int(max_candidates)]
    ]
    

# ============================================================
# CHEAP ALIVE-GATE PRUNING
# ============================================================

def update_dead_streaks(
    lifecycle_stats,
    dead_streaks,
    alive_threshold=0.02,
):
    """
    Track how many consecutive lifecycle checks each block has remained
    nearly dead.

    Returns:
        updated_dead_streaks
    """
    current_keys = {
        row["key"]
        for row in lifecycle_stats
    }

    # Remove keys that no longer exist in the scene.
    dead_streaks = {
        key: count
        for key, count in dead_streaks.items()
        if key in current_keys
    }

    for row in lifecycle_stats:
        key = row["key"]
        alive = float(row["alive_weight"])

        if alive <= float(alive_threshold):
            dead_streaks[key] = dead_streaks.get(key, 0) + 1
        else:
            dead_streaks[key] = 0

    return dead_streaks


def choose_stably_dead_blocks(
    lifecycle_stats,
    dead_streaks,
    min_dead_intervals=2,
    max_prunes=2,
):
    """
    Select blocks that have remained nearly dead for multiple lifecycle
    intervals.

    No rendering or contribution calculation is performed.
    """
    stats_by_key = {
        row["key"]: row
        for row in lifecycle_stats
    }

    candidates = [
        key
        for key, streak in dead_streaks.items()
        if streak >= int(min_dead_intervals)
        and key in stats_by_key
    ]

    # Prune lowest alive weights first.
    candidates.sort(
        key=lambda key: stats_by_key[key]["alive_weight"]
    )

    return candidates[:int(max_prunes)]


# ============================================================
# DIAGNOSTIC PRINTING
# ============================================================

def print_voxel_lifecycle_summary(
    lifecycle_stats,
    gradient_summary=None,
    max_rows=16,
):
    """
    Print the most useful block state/gradient information.
    """
    rows = list(lifecycle_stats)

    if gradient_summary is not None:
        rows.sort(
            key=lambda row: gradient_summary[
                "block_grad_norm"
            ].get(row["key"], 0.0),
            reverse=True,
        )
    else:
        rows.sort(
            key=lambda row: row["alive_weight"],
            reverse=True,
        )

    print("[FIXED VOXEL LIFECYCLE]")

    for row in rows[:int(max_rows)]:
        key = row["key"]

        if gradient_summary is None:
            grad_text = ""
        else:
            grad_text = (
                f" grad={gradient_summary['block_grad_norm'].get(key, 0.0):.3e}"
                f" shape_grad={gradient_summary['shape_grad_norm'].get(key, 0.0):.3e}"
                f" mode_grad={gradient_summary['mode_grad_norm'].get(key, 0.0):.3e}"
            )

        print(
            f"  key={key} "
            f"size={row['world_size']:.4f} "
            f"alive={row['alive_weight']:.3f} "
            f"surface={row['surface_weight']:.3f} "
            f"volume={row['volume_weight']:.3f} "
            f"opacity={row['opacity']:.3f}"
            f"{grad_text}"
        )
        
        
def choose_neighbor_activation_candidates(
    lifecycle_stats,
    gradient_summary,
    max_candidates=8,
    min_alive_weight=0.50,
    min_gradient=0.0,
    alive_growth_score_weight=2.0,
):
    """
    Rank source blocks for fixed-grid neighbor activation.

    Growth is favored for blocks whose signed alive-logit gradient indicates
    that gradient descent wants the block to become MORE alive.

    Gradient descent update:

        raw_alive_logit -= learning_rate * gradient

    Therefore:

        gradient < 0:
            optimizer increases raw_alive_logit
            -> alive_weight increases
            -> useful growth signal

        gradient > 0:
            optimizer decreases raw_alive_logit
            -> alive_weight decreases
            -> likely excessive / dying block
            -> receives no alive-growth bonus

    Positive alive gradients are not hard-excluded, because a block may still
    have strong shape/appearance pressure. They simply rank lower than
    otherwise comparable blocks with negative alive gradients.

    Parameters
    ----------
    lifecycle_stats:
        Output from collect_voxel_lifecycle_stats().

    gradient_summary:
        Output from finalize_voxel_gradient_stats(). Must contain:

            shape_grad_norm
            appearance_grad_norm
            mode_grad_norm
            alive_grad_signed

    max_candidates:
        Maximum source-block keys returned.

    min_alive_weight:
        Ignore mostly dead blocks as sources.

    min_gradient:
        Minimum non-alive gradient score required for consideration.

    alive_growth_score_weight:
        Strength of bonus given to negative signed alive gradients.

    Returns
    -------
    list
        Ranked voxel keys, strongest first.
    """
    stats_by_key = {
        row["key"]: row
        for row in lifecycle_stats
    }

    scored = []

    for key, row in stats_by_key.items():
        alive_weight = float(row["alive_weight"])
        opacity = float(row["opacity"])

        if alive_weight < float(min_alive_weight):
            continue

        shape_grad = float(
            gradient_summary["shape_grad_norm"].get(key, 0.0)
        )

        appearance_grad = float(
            gradient_summary["appearance_grad_norm"].get(key, 0.0)
        )

        mode_grad = float(
            gradient_summary["mode_grad_norm"].get(key, 0.0)
        )

        alive_grad_signed = float(
            gradient_summary["alive_grad_signed"].get(key, 0.0)
        )

        # Only negative alive gradients signal that gradient descent wants
        # to increase block existence. Positive gradients get no bonus.
        alive_growth_preference = max(
            0.0,
            -alive_grad_signed,
        )

        non_alive_gradient = (
            shape_grad
            + appearance_grad
            + mode_grad
        )

        if non_alive_gradient < float(min_gradient):
            continue

        # Avoid opacity becoming a complete veto for faint but useful areas.
        # A low-opacity block gets downweighted, but still remains eligible.
        opacity_factor = max(opacity, 0.2)

        score = (
            non_alive_gradient
            + float(alive_growth_score_weight)
            * alive_growth_preference
        ) * alive_weight * opacity_factor

        scored.append(
            (
                score,
                key,
                {
                    "shape_grad": shape_grad,
                    "appearance_grad": appearance_grad,
                    "mode_grad": mode_grad,
                    "alive_grad_signed": alive_grad_signed,
                    "alive_growth_preference": alive_growth_preference,
                    "alive_weight": alive_weight,
                    "opacity": opacity,
                },
            )
        )

    scored.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    print("[NEIGHBOR GROW CANDIDATES]")

    for rank, (score, key, info) in enumerate(
        scored[:min(int(max_candidates), 8)]
    ):
        print(
            f"  rank={rank:02d} "
            f"key={key} "
            f"score={score:.4e} "
            f"alive={info['alive_weight']:.3f} "
            f"opacity={info['opacity']:.3f} "
            f"shape_grad={info['shape_grad']:.3e} "
            f"appearance_grad={info['appearance_grad']:.3e} "
            f"mode_grad={info['mode_grad']:.3e} "
            f"alive_grad_signed={info['alive_grad_signed']:+.3e}"
        )

    return [
        key
        for _, key, _ in scored[:int(max_candidates)]
    ]