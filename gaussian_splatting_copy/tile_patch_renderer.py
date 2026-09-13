#!/usr/bin/env python

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TilePatchRenderer(nn.Module):
    """
    Tile-binned renderer for premultiplied RGBA patches.

    Inputs:
        patches:     [N, 4, H_patch, W_patch]
        center_x:    [N] projected pixel x positions
        center_y:    [N] projected pixel y positions
        patch_size:  [N] projected square side lengths in pixels
        depths:      [N] camera distances/depth values

    Output:
        [1, 4, image_height, image_width]

    Assumptions:
        - Patches contain premultiplied RGB plus alpha:
              [C_r, C_g, C_b, alpha]
        - Larger depth means farther away.
        - Patches are alpha-composited far -> near.
        - Tile assignment and depth ordering are discrete/detached.
        - Patch sampling inside every tile remains differentiable
          with respect to patch values, center_x, center_y, and patch_size.
    """

    def __init__(
        self,
        tile_size=128,
        roi_margin=16,
        max_patches_per_tile=None,
    ):
        super().__init__()

        self.tile_size = int(tile_size)
        self.roi_margin = int(roi_margin)

        if self.tile_size <= 0:
            raise ValueError("tile_size must be positive")

        self.max_patches_per_tile = max_patches_per_tile

    @staticmethod
    def composite_far_to_near(layers):
        """
        Composite premultiplied layers in far-to-near order.

        Args:
            layers: [K, 4, H, W]
                layers[0] is farthest.
                layers[K-1] is nearest.

        Returns:
            composite: [1, 4, H, W]
        """
        if layers.ndim != 4 or layers.shape[1] != 4:
            raise ValueError(
                f"Expected layers [K,4,H,W], got {tuple(layers.shape)}"
            )

        if layers.shape[0] == 0:
            raise ValueError("Cannot composite an empty layer batch")

        color = layers[:, :3]      # [K,3,H,W]
        alpha = layers[:, 3:4]     # [K,1,H,W]

        transmittance = 1.0 - alpha

        # Need weight_i = product of transmittance from all
        # slices nearer than slice i.
        #
        # Reverse into near -> far, cumulative-product, shift,
        # then flip back to original far -> near ordering.
        reverse_trans = transmittance.flip(0)

        cumulative_trans = torch.cumprod(
            reverse_trans,
            dim=0,
        )

        shifted_trans = torch.cat(
            [
                torch.ones_like(cumulative_trans[:1]),
                cumulative_trans[:-1],
            ],
            dim=0,
        )

        far_to_near_weights = shifted_trans.flip(0)

        output_color = torch.sum(
            color * far_to_near_weights,
            dim=0,
            keepdim=True,
        )

        output_alpha = 1.0 - torch.prod(
            transmittance,
            dim=0,
            keepdim=True,
        )

        return torch.cat(
            [output_color, output_alpha],
            dim=1,
        )

    @staticmethod
    def _tile_bounds(
        tile_y,
        tile_x,
        image_height,
        image_width,
        tile_size,
    ):
        """
        Returns full-image pixel bounds:

            x0, x1, y0, y1
        """
        x0 = tile_x * tile_size
        y0 = tile_y * tile_size

        x1 = min(x0 + tile_size, image_width)
        y1 = min(y0 + tile_size, image_height)

        return x0, x1, y0, y1

    @staticmethod
    def _empty_tile(
        tile_y,
        tile_x,
        image_height,
        image_width,
        tile_size,
        device,
        dtype,
    ):
        """
        Return a transparent premultiplied RGBA tile.
        """
        x0, x1, y0, y1 = TilePatchRenderer._tile_bounds(
            tile_y=tile_y,
            tile_x=tile_x,
            image_height=image_height,
            image_width=image_width,
            tile_size=tile_size,
        )

        return torch.zeros(
            1,
            4,
            y1 - y0,
            x1 - x0,
            dtype=dtype,
            device=device,
        )

    def build_tile_bins(
        self,
        center_x,
        center_y,
        patch_size,
        image_height,
        image_width,
    ):
        """
        Discretely bin patches into all overlapped screen tiles.

        Returns:
            tile_bins:
                dict mapping (tile_y, tile_x) -> Python list of
                original patch indices.

        This uses detached scalar values only for tile membership.
        Tile membership is intentionally non-differentiable, like
        Gaussian visibility/tile binning in 3DGS.
        """
        if center_x.ndim != 1:
            raise ValueError("center_x must have shape [N]")

        n = center_x.shape[0]

        if (
            center_y.shape[0] != n
            or patch_size.shape[0] != n
        ):
            raise ValueError(
                "center_x, center_y, and patch_size "
                "must have the same length"
            )

        num_tiles_y = (
            image_height + self.tile_size - 1
        ) // self.tile_size

        num_tiles_x = (
            image_width + self.tile_size - 1
        ) // self.tile_size

        tile_bins = {}

        # These CPU copies happen once per render preprocessing
        # stage. They should later be replaced with GPU/CUDA tile
        # binning if this renderer becomes the production path.
        center_x_cpu = center_x.detach().cpu()
        center_y_cpu = center_y.detach().cpu()
        patch_size_cpu = patch_size.detach().cpu()

        for index in range(n):
            cx = float(center_x_cpu[index])
            cy = float(center_y_cpu[index])
            size = max(2.0, float(patch_size_cpu[index]))

            half_extent = (
                0.5 * size
                + float(self.roi_margin)
            )

            roi_x0 = int(math.floor(cx - half_extent))
            roi_x1 = int(math.ceil(cx + half_extent))

            roi_y0 = int(math.floor(cy - half_extent))
            roi_y1 = int(math.ceil(cy + half_extent))

            # Completely outside image.
            if (
                roi_x1 <= 0
                or roi_x0 >= image_width
                or roi_y1 <= 0
                or roi_y0 >= image_height
            ):
                continue

            roi_x0 = max(0, roi_x0)
            roi_x1 = min(image_width, roi_x1)

            roi_y0 = max(0, roi_y0)
            roi_y1 = min(image_height, roi_y1)

            if roi_x0 >= roi_x1 or roi_y0 >= roi_y1:
                continue

            first_tile_x = max(
                0,
                roi_x0 // self.tile_size,
            )

            last_tile_x = min(
                num_tiles_x - 1,
                (roi_x1 - 1) // self.tile_size,
            )

            first_tile_y = max(
                0,
                roi_y0 // self.tile_size,
            )

            last_tile_y = min(
                num_tiles_y - 1,
                (roi_y1 - 1) // self.tile_size,
            )

            for tile_y in range(
                first_tile_y,
                last_tile_y + 1,
            ):
                for tile_x in range(
                    first_tile_x,
                    last_tile_x + 1,
                ):
                    key = (tile_y, tile_x)

                    if key not in tile_bins:
                        tile_bins[key] = []

                    tile_bins[key].append(index)

        return tile_bins

    @staticmethod
    def _sample_patches_on_tile(
        patches,
        center_x,
        center_y,
        patch_size,
        tile_x0,
        tile_x1,
        tile_y0,
        tile_y1,
        image_height,
        image_width,
    ):
        """
        Sample K patches only at pixels belonging to one tile.

        Args:
            patches:
                [K,4,H_patch,W_patch]

            center_x/y:
                [K], full-image pixel coordinates.

            patch_size:
                [K], projected square side in full-image pixels.

        Returns:
            sampled_layers:
                [K,4,tile_h,tile_w]
        """
        if patches.ndim != 4:
            raise ValueError(
                f"Expected patches [K,4,H,W], got {tuple(patches.shape)}"
            )

        k, channels, _, _ = patches.shape

        if channels != 4:
            raise ValueError(
                f"Expected RGBA patch channels=4, got {channels}"
            )

        if k == 0:
            raise ValueError(
                "Cannot sample zero patches on a tile"
            )

        device = patches.device
        dtype = patches.dtype

        tile_h = tile_y1 - tile_y0
        tile_w = tile_x1 - tile_x0

        canvas_wm1 = float(max(image_width - 1, 1))
        canvas_hm1 = float(max(image_height - 1, 1))

        # Full-image tile pixel coordinates.
        ys = torch.arange(
            tile_y0,
            tile_y1,
            device=device,
            dtype=dtype,
        )

        xs = torch.arange(
            tile_x0,
            tile_x1,
            device=device,
            dtype=dtype,
        )

        yy, xx = torch.meshgrid(
            ys,
            xs,
            indexing="ij",
        )

        # [tile_h, tile_w]
        full_x_norm = (
            2.0 * xx / canvas_wm1 - 1.0
        )

        full_y_norm = (
            2.0 * yy / canvas_hm1 - 1.0
        )

        # [K]
        center_x_norm = (
            2.0 * center_x / canvas_wm1 - 1.0
        )

        center_y_norm = (
            2.0 * center_y / canvas_hm1 - 1.0
        )

        patch_size = patch_size.clamp_min(2.0)

        scale_x = canvas_wm1 / patch_size
        scale_y = canvas_hm1 / patch_size

        # Broadcast:
        # [K,1,1] * ([1,H,W] - [K,1,1])
        patch_x = scale_x[:, None, None] * (
            full_x_norm[None, :, :]
            - center_x_norm[:, None, None]
        )

        patch_y = scale_y[:, None, None] * (
            full_y_norm[None, :, :]
            - center_y_norm[:, None, None]
        )

        grid = torch.stack(
            [patch_x, patch_y],
            dim=-1,
        )  # [K,tile_h,tile_w,2]

        return F.grid_sample(
            patches,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

    def forward(
        self,
        patches,
        center_x,
        center_y,
        patch_size,
        depths,
        image_height,
        image_width,
        return_tile_bins=False,
    ):
        """
        Render premultiplied patches into one tiled image.

        Args:
            patches:
                [N,4,64,64]

            center_x, center_y:
                [N] full-image pixel coordinates.

            patch_size:
                [N] projected square side in pixels.

            depths:
                [N], larger means farther away.

            image_height, image_width:
                Python integers.

        Returns:
            image:
                [1,4,image_height,image_width]

            tile_bins:
                Only returned if return_tile_bins=True.
        """
        if patches.ndim != 4:
            raise ValueError(
                f"Expected patches [N,4,H,W], got {tuple(patches.shape)}"
            )

        n, channels, _, _ = patches.shape

        if channels != 4:
            raise ValueError(
                f"Expected RGBA patches with 4 channels, got {channels}"
            )

        if (
            center_x.shape != (n,)
            or center_y.shape != (n,)
            or patch_size.shape != (n,)
            or depths.shape != (n,)
        ):
            raise ValueError(
                "center_x, center_y, patch_size, and depths "
                "must all have shape [N]"
            )

        if n == 0:
            empty = torch.zeros(
                1,
                4,
                image_height,
                image_width,
                device=patches.device,
                dtype=patches.dtype,
            )

            if return_tile_bins:
                return empty, {}

            return empty

        tile_bins = self.build_tile_bins(
            center_x=center_x,
            center_y=center_y,
            patch_size=patch_size,
            image_height=image_height,
            image_width=image_width,
        )

        num_tiles_y = (
            image_height + self.tile_size - 1
        ) // self.tile_size

        num_tiles_x = (
            image_width + self.tile_size - 1
        ) // self.tile_size

        tile_rows = []

        for tile_y in range(num_tiles_y):
            row_tiles = []

            for tile_x in range(num_tiles_x):
                key = (tile_y, tile_x)

                tile_x0, tile_x1, tile_y0, tile_y1 = (
                    self._tile_bounds(
                        tile_y=tile_y,
                        tile_x=tile_x,
                        image_height=image_height,
                        image_width=image_width,
                        tile_size=self.tile_size,
                    )
                )

                if key not in tile_bins:
                    row_tiles.append(
                        self._empty_tile(
                            tile_y=tile_y,
                            tile_x=tile_x,
                            image_height=image_height,
                            image_width=image_width,
                            tile_size=self.tile_size,
                            device=patches.device,
                            dtype=patches.dtype,
                        )
                    )
                    continue

                patch_indices = tile_bins[key]

                if (
                    self.max_patches_per_tile is not None
                    and len(patch_indices)
                    > self.max_patches_per_tile
                ):
                    # Keep nearest patches if capped.
                    # Sorting is detached/discrete.
                    ordered = sorted(
                        patch_indices,
                        key=lambda idx: float(
                            depths[idx].detach().cpu()
                        ),
                    )

                    patch_indices = ordered[
                        :self.max_patches_per_tile
                    ]

                index_tensor = torch.tensor(
                    patch_indices,
                    dtype=torch.long,
                    device=patches.device,
                )

                tile_patches = patches[index_tensor]
                tile_center_x = center_x[index_tensor]
                tile_center_y = center_y[index_tensor]
                tile_patch_size = patch_size[index_tensor]
                tile_depths = depths[index_tensor]

                # Far -> near. Larger depth means farther.
                # argsort keeps this GPU-side.
                order = torch.argsort(
                    tile_depths.detach(),
                    descending=True,
                )

                tile_patches = tile_patches[order]
                tile_center_x = tile_center_x[order]
                tile_center_y = tile_center_y[order]
                tile_patch_size = tile_patch_size[order]

                sampled_layers = self._sample_patches_on_tile(
                    patches=tile_patches,
                    center_x=tile_center_x,
                    center_y=tile_center_y,
                    patch_size=tile_patch_size,
                    tile_x0=tile_x0,
                    tile_x1=tile_x1,
                    tile_y0=tile_y0,
                    tile_y1=tile_y1,
                    image_height=image_height,
                    image_width=image_width,
                )

                row_tiles.append(
                    self.composite_far_to_near(
                        sampled_layers
                    )
                )

            tile_rows.append(
                torch.cat(
                    row_tiles,
                    dim=3,
                )
            )

        image = torch.cat(
            tile_rows,
            dim=2,
        )

        image = image.clamp(0.0, 1.0)

        if return_tile_bins:
            return image, tile_bins

        return image