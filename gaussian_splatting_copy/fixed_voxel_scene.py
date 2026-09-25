#!/usr/bin/env python
"""
Fixed voxel/octree scene management.

This file manages topology, not rendering:

    - FixedVoxelBlock placement is never optimized.
    - A block can be marked dead through its soft alive logit.
    - A block can split into 8 fixed octree children.
    - A block can activate an empty face-neighbor cell at the same level.
    - Blocks are stored in a ModuleDict, so their learnable parameters are
      registered correctly with PyTorch.

The renderer should later iterate over:

    scene.blocks.values()

and use each block's fixed:

    block.center
    block.world_size

for projection and depth sorting.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from fixed_voxel_parameters import (
    FixedVoxelBlock,
    NUM_SH_BASIS,
    NUM_SH_CHANNELS,
    NUM_SH_VALUES,
    child_center_offsets,
    child_grid_indices,
    probability_to_logit,
)


# ============================================================
# KEY / INDEX HELPERS
# ============================================================

VoxelKey = Tuple[int, int, int, int]
# (level, ix, iy, iz)


def voxel_key(level, grid_index) -> VoxelKey:
    """
    Canonical logical voxel key:

        (level, ix, iy, iz)
    """
    if len(grid_index) != 3:
        raise ValueError("grid_index must contain exactly three values.")

    return (
        int(level),
        int(grid_index[0]),
        int(grid_index[1]),
        int(grid_index[2]),
    )


def module_name_from_key(key: VoxelKey) -> str:
    """
    Convert a logical voxel key to a ModuleDict-safe name.

    Example:
        (0, -2, 4, 1)
    becomes:
        L0_Xm2_Y4_Z1

    Module names cannot safely contain dots, and using explicit axis labels
    makes checkpoints/debugging easier.
    """
    level, ix, iy, iz = key

    def signed_name(value):
        value = int(value)
        return f"m{-value}" if value < 0 else str(value)

    return (
        f"L{int(level)}"
        f"_X{signed_name(ix)}"
        f"_Y{signed_name(iy)}"
        f"_Z{signed_name(iz)}"
    )


def root_grid_index_from_center(center, root_voxel_size):
    """
    Infer a level-0 voxel index from a center and root voxel size.

    This matches the initialization convention used by:

        floor(xyz / voxel_size)

    in voxel_chunk_seeds().

    The root grid is anchored at world origin. If you later want a shifted
    root grid, add a root_origin argument and use:

        floor((center - root_origin) / root_voxel_size)
    """
    center = torch.as_tensor(center).reshape(-1)

    if center.numel() != 3:
        raise ValueError(
            f"Expected 3D center, got shape {tuple(center.shape)}"
        )

    root_voxel_size = float(root_voxel_size)

    if root_voxel_size <= 0.0:
        raise ValueError(
            f"root_voxel_size must be positive, got {root_voxel_size}"
        )

    indices = torch.floor(
        center / root_voxel_size
    ).to(torch.int64)

    return tuple(int(value) for value in indices.tolist())


def face_neighbor_grid_indices(grid_index):
    """
    Return six face-connected neighbors at the same octree level.

    Does not include edge/corner neighbors.
    """
    ix, iy, iz = grid_index

    return [
        (ix - 1, iy, iz),
        (ix + 1, iy, iz),
        (ix, iy - 1, iz),
        (ix, iy + 1, iz),
        (ix, iy, iz - 1),
        (ix, iy, iz + 1),
    ]


# ============================================================
# SHARED GLOBAL LIGHTING
# ============================================================

class FixedVoxelLighting(nn.Module):
    """
    Shared global order-2 RGB SH lighting.

    Effective lighting:

        global_sh =
            initial_global_sh
            + bound * tanh(raw_global_sh_delta)
    """

    def __init__(
        self,
        initial_global_sh,
        optimize_sh=False,
        bound=0.005,
    ):
        super().__init__()

        initial_global_sh = torch.as_tensor(
            initial_global_sh,
            dtype=torch.float32,
        )

        if initial_global_sh.numel() != NUM_SH_VALUES:
            raise ValueError(
                f"Expected {NUM_SH_VALUES} SH values, got "
                f"{initial_global_sh.numel()}."
            )

        initial_global_sh = initial_global_sh.reshape(
            NUM_SH_BASIS,
            NUM_SH_CHANNELS,
        )

        self.register_buffer(
            "initial_global_sh",
            initial_global_sh.clone(),
        )

        self.bound = float(bound)

        self.raw_global_sh_delta = nn.Parameter(
            torch.zeros_like(initial_global_sh),
            requires_grad=bool(optimize_sh),
        )

    @property
    def global_sh(self):
        return self.initial_global_sh + (
            self.bound * torch.tanh(self.raw_global_sh_delta)
        )


# ============================================================
# FIXED VOXEL SCENE
# ============================================================

class FixedVoxelScene(nn.Module):
    """
    A fixed-position hierarchical voxel scene.

    Spatial placement does not move after initialization. Blocks are indexed
    by an octree key:

        (level, ix, iy, iz)

    Important topology convention
    -----------------------------
    A parent block is removed from the active ModuleDict when split. Its
    eight children replace it. This avoids rendering both parent and children
    simultaneously.

    "Death" is soft by default:
        block.raw_alive_logit is pushed toward a negative value.

    A dead block can later be hard-removed by calling remove_block(), but
    keeping soft-dead blocks briefly may make optimization/topology changes
    smoother.
    """

    def __init__(
        self,
        blocks: Optional[Iterable[FixedVoxelBlock]] = None,
        initial_global_sh=None,
        optimize_sh=False,
        global_sh_bound=0.005,
        root_voxel_size=None,
    ):
        super().__init__()

        if initial_global_sh is None:
            initial_global_sh = torch.zeros(
                NUM_SH_BASIS,
                NUM_SH_CHANNELS,
                dtype=torch.float32,
            )

        self.blocks = nn.ModuleDict()

        self.lighting = FixedVoxelLighting(
            initial_global_sh=initial_global_sh,
            optimize_sh=optimize_sh,
            bound=global_sh_bound,
        )

        self.root_voxel_size = (
            None
            if root_voxel_size is None
            else float(root_voxel_size)
        )

        if blocks is not None:
            for block in blocks:
                self.add_block(block)

    @property
    def global_sh(self):
        return self.lighting.global_sh

    def __len__(self):
        return len(self.blocks)

    def logical_key_for_block(self, block: FixedVoxelBlock) -> VoxelKey:
        return voxel_key(
            level=block.level,
            grid_index=block.grid_index,
        )

    def module_name_for_block(self, block: FixedVoxelBlock) -> str:
        return module_name_from_key(
            self.logical_key_for_block(block)
        )

    def has_key(self, key: VoxelKey) -> bool:
        return module_name_from_key(key) in self.blocks

    def get_block(self, key: VoxelKey) -> FixedVoxelBlock:
        name = module_name_from_key(key)

        if name not in self.blocks:
            raise KeyError(f"No block exists for voxel key {key}.")

        return self.blocks[name]

    def iter_blocks_with_keys(self):
        """
        Yield:
            (logical_key, block)

        The logical key is reconstructed from metadata stored in each block.
        """
        for block in self.blocks.values():
            yield self.logical_key_for_block(block), block

    def add_block(
        self,
        block: FixedVoxelBlock,
        overwrite=False,
    ):
        """
        Register one block.

        Parameters
        ----------
        overwrite:
            False by default. Replacing an existing voxel accidentally is
            dangerous, so an explicit opt-in is required.
        """
        if not isinstance(block, FixedVoxelBlock):
            raise TypeError(
                "add_block expects a FixedVoxelBlock, got "
                f"{type(block)}"
            )

        key = self.logical_key_for_block(block)
        name = module_name_from_key(key)

        if name in self.blocks and not overwrite:
            raise KeyError(
                f"Voxel key {key} already exists. "
                "Use overwrite=True only deliberately."
            )

        self.blocks[name] = block

        return key

    def remove_block(self, key: VoxelKey):
        """
        Hard-remove a block from the scene.

        The optimizer must be rebuilt afterward, because ModuleDict topology
        has changed.
        """
        name = module_name_from_key(key)

        if name not in self.blocks:
            raise KeyError(
                f"Cannot remove absent voxel block {key}."
            )

        del self.blocks[name]

    def mark_dead(
        self,
        key: VoxelKey,
        alive_probability=1e-4,
    ):
        """
        Softly deactivate one block.

        This does not remove it from ModuleDict. The renderer should multiply
        the blended patch by block.alive_weight.

        For hard topology removal after a later prune stage, call:
            scene.remove_block(key)
        """
        block = self.get_block(key)

        with torch.no_grad():
            block.raw_alive_logit.copy_(
                probability_to_logit(alive_probability).to(
                    device=block.raw_alive_logit.device,
                    dtype=block.raw_alive_logit.dtype,
                )
            )

    def mark_alive(
        self,
        key: VoxelKey,
        alive_probability=0.99,
    ):
        """Softly reactivate an existing block."""
        block = self.get_block(key)

        with torch.no_grad():
            block.raw_alive_logit.copy_(
                probability_to_logit(alive_probability).to(
                    device=block.raw_alive_logit.device,
                    dtype=block.raw_alive_logit.dtype,
                )
            )

    def active_blocks(
        self,
        alive_threshold=1e-3,
    ) -> List[FixedVoxelBlock]:
        """
        Return blocks whose detached soft alive weight exceeds threshold.

        Use this only for discrete culling/topology logic. The differentiable
        renderer can simply include every block and multiply its patch by
        alive_weight.
        """
        active = []

        for block in self.blocks.values():
            if float(block.alive_weight.detach()) > alive_threshold:
                active.append(block)

        return active


# ============================================================
# BLOCK CREATION FROM INITIAL VOXEL SEEDS
# ============================================================

def make_blocks_from_voxel_seeds(
    seeds,
    root_voxel_size,
    init_parameter_fn,
    initial_global_sh,
    device,
    optimize_environment=False,
    initial_volume_probability=0.50,
    initial_alive_probability=0.99,
):
    """
    Convert output from voxel_chunk_seeds() into FixedVoxelBlock objects.

    Parameters
    ----------
    seeds:
        List returned by voxel_chunk_seeds(). Each item has at least:

            {
                "center": tensor [3],
                "world_size": float,
                "num_points": int,
                "color": tensor [3] or None,
            }

    root_voxel_size:
        Grid size used by voxel_chunk_seeds().

    init_parameter_fn:
        Callback:

            init_parameter_fn(seed) -> dict

        Required fields in returned dict:

            ctrl_values
            sigma
            base_color_r
            base_color_g
            base_color_b
            opacity
            roughness
            metallic
            specular

        This lets your optimizer decide whether initialization is random,
        point-color driven, or uses priors from a checkpoint.

    initial_global_sh:
        [9,3] or [27] initial SH lighting.

    Returns
    -------
    blocks:
        List[FixedVoxelBlock]
    """
    root_voxel_size = float(root_voxel_size)

    blocks = []
    used_keys = set()

    for seed_index, seed in enumerate(seeds):
        center = torch.as_tensor(
            seed["center"],
            dtype=torch.float32,
        )

        grid_index = root_grid_index_from_center(
            center=center,
            root_voxel_size=root_voxel_size,
        )

        key = voxel_key(
            level=0,
            grid_index=grid_index,
        )

        if key in used_keys:
            raise RuntimeError(
                f"Two initialization seeds map to the same root voxel key "
                f"{key}. Check voxel seed generation."
            )

        used_keys.add(key)

        params = init_parameter_fn(seed)

        required = {
            "ctrl_values",
            "sigma",
            "base_color_r",
            "base_color_g",
            "base_color_b",
            "opacity",
            "roughness",
            "metallic",
            "specular",
        }

        missing = required - set(params.keys())

        if missing:
            raise KeyError(
                f"Initialization callback for seed {seed_index} is missing "
                f"fields: {sorted(missing)}"
            )

        block = FixedVoxelBlock(
            center=center,
            world_size=float(seed["world_size"]),
            ctrl_values=params["ctrl_values"],
            sigma=params["sigma"],
            base_color_r=params["base_color_r"],
            base_color_g=params["base_color_g"],
            base_color_b=params["base_color_b"],
            opacity=params["opacity"],
            roughness=params["roughness"],
            sh_values=initial_global_sh,
            metallic=params["metallic"],
            specular=params["specular"],
            level=0,
            grid_index=grid_index,
            parent_key=None,
            initial_alive_probability=initial_alive_probability,
            initial_volume_probability=initial_volume_probability,
            optimize_environment=optimize_environment,
        ).to(device)

        blocks.append(block)

    return blocks


# ============================================================
# FIXED OCTREE SPLITTING
# ============================================================

def _physical_values_from_parent(
    parent: FixedVoxelBlock,
    shared_sh=None,
):
    """
    Extract detached physical parent values for child initialization.
    """
    values = parent.common_values(shared_sh=shared_sh)

    return {
        "ctrl_values": values["ctrl"].detach().clone(),
        "sigma": float(values["sigma"].detach()),
        "base_color_r": float(values["base_color_r"].detach()),
        "base_color_g": float(values["base_color_g"].detach()),
        "base_color_b": float(values["base_color_b"].detach()),
        "opacity": float(values["opacity"].detach()),
        "roughness": float(values["roughness"].detach()),
        "metallic": float(parent.metallic_value.detach()),
        "specular": float(parent.specular_value.detach()),
        "sh_values": values["sh"].detach().clone(),
        "alive_probability": float(parent.alive_weight.detach()),
        "volume_probability": float(parent.volume_weight.detach()),
    }


def split_block_into_children(
    parent: FixedVoxelBlock,
    shared_sh=None,
    child_alive_probability=0.99,
    perturb_controls=0.02,
    perturb_sigma_raw=0.03,
    perturb_color_raw=0.02,
    perturb_opacity_raw=0.02,
    perturb_mode_logit=0.10,
):
    """
    Create eight fixed octree children from one parent.

    Parent remains unchanged. FixedVoxelScene.split_block() is responsible
    for removing/deactivating the parent and inserting the children.

    Children inherit parent physical values, then receive small raw-space
    perturbations so optimization can specialize them.
    """
    device = parent.center.device
    dtype = parent.center.dtype

    parent_values = _physical_values_from_parent(
        parent,
        shared_sh=shared_sh,
    )

    child_size = 0.5 * parent.world_size.detach()

    offsets = child_center_offsets(
        world_size=parent.world_size.detach(),
        device=device,
        dtype=dtype,
    )

    grid_indices = child_grid_indices(parent.grid_index)
    parent_key = voxel_key(parent.level, parent.grid_index)

    children = []

    for offset, grid_index in zip(offsets, grid_indices):
        child = FixedVoxelBlock(
            center=parent.center.detach() + offset,
            world_size=float(child_size),
            ctrl_values=parent_values["ctrl_values"],
            sigma=parent_values["sigma"],
            base_color_r=parent_values["base_color_r"],
            base_color_g=parent_values["base_color_g"],
            base_color_b=parent_values["base_color_b"],
            opacity=parent_values["opacity"],
            roughness=parent_values["roughness"],
            sh_values=parent_values["sh_values"],
            metallic=parent_values["metallic"],
            specular=parent_values["specular"],
            level=parent.level + 1,
            grid_index=grid_index,
            parent_key=parent_key,
            initial_alive_probability=child_alive_probability,
            initial_volume_probability=parent_values[
                "volume_probability"
            ],
            optimize_environment=parent.optimize_environment,
            local_sh_bound=parent.local_sh_bound,
        ).to(device)

        with torch.no_grad():
            child.raw_ctrl.add_(
                perturb_controls * torch.randn_like(child.raw_ctrl)
            )

            child.raw_sigma.add_(
                perturb_sigma_raw * torch.randn_like(child.raw_sigma)
            )

            child.raw_base_color_r.add_(
                perturb_color_raw
                * torch.randn_like(child.raw_base_color_r)
            )

            child.raw_base_color_g.add_(
                perturb_color_raw
                * torch.randn_like(child.raw_base_color_g)
            )

            child.raw_base_color_b.add_(
                perturb_color_raw
                * torch.randn_like(child.raw_base_color_b)
            )

            child.raw_opacity.add_(
                perturb_opacity_raw
                * torch.randn_like(child.raw_opacity)
            )

            child.raw_roughness.add_(
                perturb_color_raw
                * torch.randn_like(child.raw_roughness)
            )

            child.raw_mode_logit.add_(
                perturb_mode_logit
                * torch.randn_like(child.raw_mode_logit)
            )

        children.append(child)

    return children


def split_scene_block(
    scene: FixedVoxelScene,
    parent_key: VoxelKey,
    remove_parent=True,
    **child_kwargs,
):
    """
    Replace one scene block with its eight octree children.

    Parameters
    ----------
    remove_parent:
        True:
            Remove parent from active topology immediately.

        False:
            Soft-kill parent and retain it in ModuleDict for inspection.
            This is less memory-efficient and requires the renderer to respect
            alive_weight.

    Returns
    -------
    child_keys:
        List of eight child voxel keys.

    Important:
        Rebuild your optimizer after calling this function because scene
        topology and Parameter objects have changed.
    """
    parent = scene.get_block(parent_key)

    children = split_block_into_children(
        parent=parent,
        shared_sh=scene.global_sh,
        **child_kwargs,
    )

    child_keys = [
        scene.logical_key_for_block(child)
        for child in children
    ]

    for child_key in child_keys:
        if scene.has_key(child_key):
            raise RuntimeError(
                f"Cannot split {parent_key}: child key {child_key} "
                "already exists."
            )

    if remove_parent:
        scene.remove_block(parent_key)
    else:
        scene.mark_dead(parent_key)

    for child in children:
        scene.add_block(child)

    print(
        "[split_scene_block] "
        f"parent={parent_key}, "
        f"children={child_keys}, "
        f"remove_parent={remove_parent}"
    )

    return child_keys


# ============================================================
# SAME-LEVEL NEIGHBOR ACTIVATION
# ============================================================


def available_face_neighbors(
    scene: FixedVoxelScene,
    source_key: VoxelKey,
):
    """
    Return empty same-level face-neighbor keys for one block.
    """
    source = scene.get_block(source_key)

    available = []

    for neighbor_index in face_neighbor_grid_indices(
        source.grid_index
    ):
        key = voxel_key(
            level=source.level,
            grid_index=neighbor_index,
        )

        if not scene.has_key(key):
            available.append(key)

    return available
    
def cell_is_occupied_at_any_level(
    scene: FixedVoxelScene,
    level: int,
    grid_index,
):
    """
    Return True if the requested octree cell overlaps an existing block at:
      - the same level;
      - any coarser ancestor level;
      - any finer descendant level.

    This prevents neighbor activation from creating overlapping blocks when
    the scene contains mixed octree levels.
    """
    level = int(level)
    ix, iy, iz = (int(v) for v in grid_index)

    # Same-level or coarser ancestor occupancy.
    for ancestor_level in range(level, -1, -1):
        scale = 2 ** (level - ancestor_level)

        ancestor_index = (
            ix // scale,
            iy // scale,
            iz // scale,
        )

        ancestor_key = voxel_key(
            ancestor_level,
            ancestor_index,
        )

        if scene.has_key(ancestor_key):
            return True

    # Finer descendant occupancy.
    for key, _ in scene.iter_blocks_with_keys():
        other_level, ox, oy, oz = key

        if other_level <= level:
            continue

        scale = 2 ** (other_level - level)

        parent_index_at_level = (
            ox // scale,
            oy // scale,
            oz // scale,
        )

        if parent_index_at_level == (ix, iy, iz):
            return True

    return False


def available_face_neighbors_nonoverlapping(
    scene: FixedVoxelScene,
    source_key: VoxelKey,
):
    """
    Return empty same-level face neighbors that do not overlap any active
    block at a coarser or finer octree level.
    """
    source = scene.get_block(source_key)

    available = []

    for neighbor_index in face_neighbor_grid_indices(
        source.grid_index
    ):
        if not cell_is_occupied_at_any_level(
            scene=scene,
            level=source.level,
            grid_index=neighbor_index,
        ):
            available.append(
                voxel_key(
                    source.level,
                    neighbor_index,
                )
            )

    return available

def activate_neighbor_from_source(
    scene: FixedVoxelScene,
    source_key: VoxelKey,
    neighbor_grid_index,
    alive_probability=0.99,
    perturb_controls=0.03,
    perturb_sigma_raw=0.05,
    perturb_color_raw=0.03,
    perturb_mode_logit=0.15,
):
    """
    Activate an empty same-level face-neighbor voxel by duplicating a source.

    This implements your "duplicate itself onto neighboring dead blocks"
    concept. In this first version, a missing voxel is considered empty/dead.

    If you later keep explicitly dead blocks in ModuleDict, you can instead
    overwrite/reinitialize such a block after checking alive_weight.

    Returns:
        new_key, new_block

    Important:
        Rebuild optimizer after activation.
    """
    source = scene.get_block(source_key)

    neighbor_grid_index = tuple(
        int(value)
        for value in neighbor_grid_index
    )

    target_key = voxel_key(
        level=source.level,
        grid_index=neighbor_grid_index,
    )

    if cell_is_occupied_at_any_level(
        scene=scene,
        level=source.level,
        grid_index=neighbor_grid_index,
    ):
        raise RuntimeError(
            f"Neighbor voxel {target_key} overlaps an existing block "
            "at the same, coarser, or finer octree level."
        )

    source_values = _physical_values_from_parent(
        source,
        shared_sh=scene.global_sh,
    )

    displacement_index = torch.tensor(
        [
            neighbor_grid_index[0] - source.grid_index[0],
            neighbor_grid_index[1] - source.grid_index[1],
            neighbor_grid_index[2] - source.grid_index[2],
        ],
        dtype=source.center.dtype,
        device=source.center.device,
    )

    # Same-level neighboring cell centers are separated by world_size.
    target_center = (
        source.center.detach()
        + source.world_size.detach() * displacement_index
    )

    new_block = FixedVoxelBlock(
        center=target_center,
        world_size=float(source.world_size.detach()),
        ctrl_values=source_values["ctrl_values"],
        sigma=source_values["sigma"],
        base_color_r=source_values["base_color_r"],
        base_color_g=source_values["base_color_g"],
        base_color_b=source_values["base_color_b"],
        opacity=source_values["opacity"],
        roughness=source_values["roughness"],
        sh_values=source_values["sh_values"],
        metallic=source_values["metallic"],
        specular=source_values["specular"],
        level=source.level,
        grid_index=neighbor_grid_index,
        parent_key=source_key,
        initial_alive_probability=alive_probability,
        initial_volume_probability=source_values[
            "volume_probability"
        ],
        optimize_environment=source.optimize_environment,
        local_sh_bound=source.local_sh_bound,
    ).to(source.center.device)

    with torch.no_grad():
        new_block.raw_ctrl.add_(
            perturb_controls * torch.randn_like(new_block.raw_ctrl)
        )

        new_block.raw_sigma.add_(
            perturb_sigma_raw * torch.randn_like(new_block.raw_sigma)
        )

        new_block.raw_base_color_r.add_(
            perturb_color_raw
            * torch.randn_like(new_block.raw_base_color_r)
        )

        new_block.raw_base_color_g.add_(
            perturb_color_raw
            * torch.randn_like(new_block.raw_base_color_g)
        )

        new_block.raw_base_color_b.add_(
            perturb_color_raw
            * torch.randn_like(new_block.raw_base_color_b)
        )

        new_block.raw_mode_logit.add_(
            perturb_mode_logit
            * torch.randn_like(new_block.raw_mode_logit)
        )

    scene.add_block(new_block)

    print(
        "[activate_neighbor_from_source] "
        f"source={source_key}, target={target_key}"
    )

    return target_key, new_block