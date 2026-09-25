#!/usr/bin/env python
"""
Renderer for a FixedVoxelScene.

Each FixedVoxelBlock has fixed center/world_size, but soft gates:

    alive_weight   = sigmoid(raw_alive_logit)
    volume_weight  = sigmoid(raw_mode_logit)
    surface_weight = 1 - volume_weight

For every visible block:

    surface_patch = frozen_surface_fno(surface_features)
    volume_patch  = frozen_volume_fno(volume_features)

    mixed_patch =
        surface_weight * surface_patch
        + volume_weight * volume_patch

    final_patch = alive_weight * mixed_patch

The patch is premultiplied RGBA, so multiplying all four channels by
alive_weight is valid.

The final per-camera render composites blocks far-to-near.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from fixed_voxel_parameters import build_tensor_fno_vector
from tiled_roi_renderer import render_rois_to_tiled_canvas
from tile_patch_renderer import TilePatchRenderer


# ============================================================
# CAMERA / PROJECTION HELPERS
# ============================================================

def camera_to_block_pose(camera, block_center):
    """
    Compute spherical camera direction relative to a fixed voxel center.

    This exactly matches the conditioning used by train_implicit.py:

        x = r sin(phi) cos(theta)
        y = r sin(phi) sin(theta)
        z = r cos(phi)

    Returns
    -------
    phi, theta, radius, relative
    """
    camera_center = camera.camera_center

    block_center = block_center.to(
        device=camera_center.device,
        dtype=camera_center.dtype,
    )

    relative = camera_center - block_center

    radius = torch.linalg.norm(
        relative
    ).clamp_min(1e-8)

    phi = torch.acos(
        torch.clamp(
            relative[2] / radius,
            -1.0,
            1.0,
        )
    )

    theta = torch.remainder(
        torch.atan2(relative[1], relative[0]),
        2.0 * math.pi,
    )

    return phi, theta, radius, relative


def project_fixed_block_center(
    camera,
    center_world,
    flip_projection_y=False,
):
    """
    Project one fixed voxel center with the 3DGS row-vector convention.

    Returns
    -------
    pixel_x, pixel_y, ndc_z, valid_w
    """
    projection = camera.full_proj_transform

    center_world = center_world.to(
        device=projection.device,
        dtype=projection.dtype,
    )

    point_h = torch.cat(
        [
            center_world,
            torch.ones(
                1,
                device=projection.device,
                dtype=projection.dtype,
            ),
        ]
    )

    # 3DGS convention: row-vector multiplication.
    clip = point_h @ projection

    clip_w = clip[3]
    valid_w = clip_w > 1e-8

    safe_w = torch.where(
        valid_w,
        clip_w,
        torch.ones_like(clip_w),
    )

    ndc = clip[:3] / safe_w

    pixel_x = (
        (ndc[0] + 1.0)
        * 0.5
        * float(camera.image_width)
    )

    if flip_projection_y:
        pixel_y = (
            (ndc[1] + 1.0)
            * 0.5
            * float(camera.image_height)
        )
    else:
        pixel_y = (
            (1.0 - ndc[1])
            * 0.5
            * float(camera.image_height)
        )

    return pixel_x, pixel_y, ndc[2], valid_w


# ============================================================
# FROZEN FNO EVALUATION
# ============================================================

def run_fno_in_chunks(
    model,
    params,
    batch_size,
    use_activation_checkpointing=True,
):
    """
    Run a frozen FNO on [N, D] vectors in chunks.

    The FNO parameters should have requires_grad=False, but gradients still
    flow through params into the fixed voxel's learnable conditioning fields.
    """
    if params.ndim != 2:
        raise ValueError(
            f"Expected params [N,D], got {tuple(params.shape)}"
        )

    if params.shape[0] == 0:
        raise RuntimeError("Cannot evaluate FNO on an empty parameter batch.")

    outputs = []

    for start in range(0, params.shape[0], int(batch_size)):
        end = min(start + int(batch_size), params.shape[0])

        parameter_chunk = params[start:end]

        if (
            use_activation_checkpointing
            and torch.is_grad_enabled()
        ):
            output_chunk = checkpoint(
                model,
                parameter_chunk,
                use_reentrant=False,
            )
        else:
            output_chunk = model(parameter_chunk)

        outputs.append(output_chunk)

    return torch.cat(outputs, dim=0)


# ============================================================
# FIXED VOXEL RENDERER
# ============================================================

class FixedVoxelRenderer(nn.Module):
    """
    Render a FixedVoxelScene through frozen surface and volume FNOs.

    Parameters
    ----------
    surface_bundle, volume_bundle:
        Dictionaries returned by a checkpoint loader, each containing:

            {
                "model": frozen FNO model,
                "param_mean": numpy/tensor normalization mean,
                "param_std": numpy/tensor normalization std,
            }

    fno_radius:
        Geometric projected-footprint scale only. It is not passed to the
        FNO because camera radius was not a training feature.

    alive_cull_threshold:
        Blocks with detached alive_weight below this threshold are skipped
        completely. Set to 0.0 to evaluate all blocks regardless of life.

    Notes
    -----
    Block locations and sizes are fixed. Therefore visibility culling is
    safely detached/discrete: there are no center/size gradients to lose.
    """

    def __init__(
        self,
        surface_bundle,
        volume_bundle,
        fno_radius=2.2,
        fno_batch_size=16,
        flip_projection_y=False,
        flip_fno_vertical=False,
        use_tile_renderer=True,
        tile_size=128,
        tile_roi_margin=16,
        max_patches_per_tile=None,
        min_projected_patch_size=2.0,
        active_slice_margin=64.0,
        alive_cull_threshold=1e-4,
        use_fno_activation_checkpointing=True,
    ):
        super().__init__()

        self.surface_model = surface_bundle["model"]
        self.volume_model = volume_bundle["model"]

        self.surface_mean = surface_bundle["param_mean"]
        self.surface_std = surface_bundle["param_std"]

        self.volume_mean = volume_bundle["param_mean"]
        self.volume_std = volume_bundle["param_std"]

        self.fno_radius = float(fno_radius)
        self.fno_batch_size = int(fno_batch_size)

        self.flip_projection_y = bool(flip_projection_y)
        self.flip_fno_vertical = bool(flip_fno_vertical)

        self.use_tile_renderer = bool(use_tile_renderer)

        self.min_projected_patch_size = float(
            min_projected_patch_size
        )

        self.active_slice_margin = float(active_slice_margin)

        self.alive_cull_threshold = float(
            alive_cull_threshold
        )

        self.use_fno_activation_checkpointing = bool(
            use_fno_activation_checkpointing
        )

        self.tile_renderer = TilePatchRenderer(
            tile_size=int(tile_size),
            roi_margin=int(tile_roi_margin),
            max_patches_per_tile=max_patches_per_tile,
        )

        for model in [self.surface_model, self.volume_model]:
            model.eval()

            for parameter in model.parameters():
                parameter.requires_grad_(False)

    # ========================================================
    # BLOCK CULLING / RECORD CONSTRUCTION
    # ========================================================

    def _block_is_potentially_visible(
        self,
        camera,
        block,
    ):
        """
        Discrete visibility test for a fixed block.

        Returns:
            visible, center_x, center_y, depth, patch_size, radius,
            phi, theta, relative
        """
        (
            phi,
            theta,
            radius,
            relative,
        ) = camera_to_block_pose(
            camera,
            block.center,
        )

        (
            center_x,
            center_y,
            ndc_depth,
            valid_w,
        ) = project_fixed_block_center(
            camera,
            block.center,
            flip_projection_y=self.flip_projection_y,
        )

        image_height = float(camera.image_height)
        image_width = float(camera.image_width)

        patch_size = (
            image_height
            * block.world_size
            * self.fno_radius
            / radius
        )

        # Fixed block center/size means this detached Python visibility
        # decision does not eliminate placement gradients.
        visible = (
            bool(valid_w.detach())
            and float(ndc_depth.detach()) > 0.0
            and bool(torch.isfinite(patch_size).detach())
            and float(patch_size.detach()) >= self.min_projected_patch_size
            and float(center_x.detach()) >= -self.active_slice_margin
            and float(center_x.detach())
            <= image_width + self.active_slice_margin
            and float(center_y.detach()) >= -self.active_slice_margin
            and float(center_y.detach())
            <= image_height + self.active_slice_margin
        )

        return (
            visible,
            center_x,
            center_y,
            ndc_depth,
            patch_size,
            radius,
            phi,
            theta,
            relative,
        )

    def _build_visible_records(
        self,
        camera,
        voxel_scene,
    ):
        """
        Build one differentiable render record for each visible/alive block.
        """
        records = []

        for key, block in voxel_scene.iter_blocks_with_keys():
            alive_weight = block.alive_weight

            # Discrete performance culling only. The alive state remains
            # differentiable for blocks above this threshold.
            if (
                float(alive_weight.detach())
                < self.alive_cull_threshold
            ):
                continue

            (
                visible,
                center_x,
                center_y,
                ndc_depth,
                patch_size,
                radius,
                phi,
                theta,
                relative,
            ) = self._block_is_potentially_visible(
                camera,
                block,
            )

            if not visible:
                continue

            surface_params = build_tensor_fno_vector(
                voxel_block=block,
                mode="surface",
                param_mean=self.surface_mean,
                param_std=self.surface_std,
                phi=phi,
                theta=theta,
                device=camera.camera_center.device,
                shared_sh=voxel_scene.global_sh,
            )

            volume_params = build_tensor_fno_vector(
                voxel_block=block,
                mode="volume",
                param_mean=self.volume_mean,
                param_std=self.volume_std,
                phi=phi,
                theta=theta,
                device=camera.camera_center.device,
                shared_sh=voxel_scene.global_sh,
            )

            records.append(
                {
                    "key": key,
                    "block": block,
                    "surface_params": surface_params,
                    "volume_params": volume_params,
                    "surface_weight": block.surface_weight,
                    "volume_weight": block.volume_weight,
                    "alive_weight": alive_weight,
                    "center_x": center_x,
                    "center_y": center_y,
                    "patch_size": patch_size,
                    # Larger radius = farther for compositing.
                    "depth": radius,
                    "ndc_depth": ndc_depth,
                    "phi": phi,
                    "theta": theta,
                    "relative": relative,
                }
            )

        return records

    # ========================================================
    # PATCH EVALUATION / BLENDING
    # ========================================================

    def _evaluate_blended_patches(self, records):
        """
        Evaluate both FNOs for all visible blocks and blend per block.

        Returns:
            patches [N,4,Hpatch,Wpatch]
        """
        if not records:
            raise RuntimeError(
                "Cannot evaluate FNO patches for zero records."
            )

        surface_params = torch.cat(
            [record["surface_params"] for record in records],
            dim=0,
        )

        volume_params = torch.cat(
            [record["volume_params"] for record in records],
            dim=0,
        )

        surface_patches = run_fno_in_chunks(
            model=self.surface_model,
            params=surface_params,
            batch_size=self.fno_batch_size,
            use_activation_checkpointing=(
                self.use_fno_activation_checkpointing
            ),
        )

        volume_patches = run_fno_in_chunks(
            model=self.volume_model,
            params=volume_params,
            batch_size=self.fno_batch_size,
            use_activation_checkpointing=(
                self.use_fno_activation_checkpointing
            ),
        )

        if self.flip_fno_vertical:
            surface_patches = torch.flip(
                surface_patches,
                dims=[2],
            )

            volume_patches = torch.flip(
                volume_patches,
                dims=[2],
            )

        surface_weights = torch.stack(
            [record["surface_weight"] for record in records],
            dim=0,
        ).view(-1, 1, 1, 1)

        volume_weights = torch.stack(
            [record["volume_weight"] for record in records],
            dim=0,
        ).view(-1, 1, 1, 1)

        alive_weights = torch.stack(
            [record["alive_weight"] for record in records],
            dim=0,
        ).view(-1, 1, 1, 1)

        # Convex blend of premultiplied RGBA is valid.
        mixed_patches = (
            surface_weights * surface_patches
            + volume_weights * volume_patches
        )

        # Soft occupancy scales both premultiplied RGB and alpha.
        final_patches = alive_weights * mixed_patches

        return final_patches

    # ========================================================
    # MAIN RENDER
    # ========================================================

    def render_scene(
        self,
        camera,
        voxel_scene,
        collect_diagnostics=False,
    ):
        """
        Render a FixedVoxelScene from one 3DGS camera.

        Returns:
            rgba: [1,4,H,W]
            diagnostics: list[dict]
        """
        canvas_height = int(camera.image_height)
        canvas_width = int(camera.image_width)
        device = camera.camera_center.device

        records = self._build_visible_records(
            camera=camera,
            voxel_scene=voxel_scene,
        )

        if not records:
            empty = torch.zeros(
                1,
                4,
                canvas_height,
                canvas_width,
                dtype=torch.float32,
                device=device,
            )

            return empty, []

        patches = self._evaluate_blended_patches(records)

        center_x = torch.stack(
            [record["center_x"] for record in records],
            dim=0,
        )

        center_y = torch.stack(
            [record["center_y"] for record in records],
            dim=0,
        )

        patch_sizes = torch.stack(
            [record["patch_size"] for record in records],
            dim=0,
        )

        depths = torch.stack(
            [record["depth"] for record in records],
            dim=0,
        )

        # Far -> near. Larger camera-center distance means farther.
        sort_indices = torch.argsort(
            depths.detach(),
            descending=True,
        )

        if self.use_tile_renderer:
            rgba = self.tile_renderer(
                patches=patches,
                center_x=center_x,
                center_y=center_y,
                patch_size=patch_sizes,
                depths=depths,
                image_height=canvas_height,
                image_width=canvas_width,
            )
        else:
            rgba = render_rois_to_tiled_canvas(
                sorted_patches=patches[sort_indices],
                sorted_center_x=center_x[sort_indices],
                sorted_center_y=center_y[sort_indices],
                sorted_patch_sizes=patch_sizes[sort_indices],
                canvas_height=canvas_height,
                canvas_width=canvas_width,
                tile_size=128,
                margin_pixels=16,
                max_roi_side=max(canvas_height, canvas_width),
            )

        diagnostics = []

        if collect_diagnostics:
            for record_index in sort_indices.detach().cpu().tolist():
                record = records[record_index]
                block = record["block"]

                diagnostics.append(
                    {
                        "key": record["key"],
                        "level": block.level,
                        "grid_index": block.grid_index,
                        "alive_weight": record["alive_weight"],
                        "surface_weight": record["surface_weight"],
                        "volume_weight": record["volume_weight"],
                        "center_x": record["center_x"],
                        "center_y": record["center_y"],
                        "patch_size": record["patch_size"],
                        "depth": record["depth"],
                        "ndc_depth": record["ndc_depth"],
                        "phi": record["phi"],
                        "theta": record["theta"],
                        "relative": record["relative"],
                    }
                )

        return rgba.clamp(0.0, 1.0), diagnostics

    def forward(
        self,
        camera,
        voxel_scene,
    ):
        return self.render_scene(
            camera=camera,
            voxel_scene=voxel_scene,
            collect_diagnostics=False,
        )