#!/usr/bin/env python
import math

import torch
import torch.nn as nn

from neuralop.models import FNO


def make_fourier_pos_features(
    height: int,
    width: int,
    num_freqs: int = 4,
) -> torch.Tensor:
    """
    Returns fixed positional features with shape [C_pos, H, W].
    """
    ys = torch.linspace(-1.0, 1.0, height)
    xs = torch.linspace(-1.0, 1.0, width)

    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    features = [xx, yy]

    for frequency in range(1, num_freqs + 1):
        features.extend(
            [
                torch.sin(frequency * math.pi * xx),
                torch.cos(frequency * math.pi * xx),
                torch.sin(frequency * math.pi * yy),
                torch.cos(frequency * math.pi * yy),
            ]
        )

    return torch.stack(features, dim=0)


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.activation(self.conv1(x))
        x = self.conv2(x)

        return residual + x


class ImageRefiner(nn.Module):
    """
    Small residual CNN refinement head for FNO-produced RGBA images.
    """

    def __init__(
        self,
        channels: int = 4,
        hidden_channels: int = 32,
        num_blocks: int = 3,
    ):
        super().__init__()

        self.entry = nn.Conv2d(
            channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
        )

        self.blocks = nn.Sequential(
            *[
                ResBlock(hidden_channels)
                for _ in range(num_blocks)
            ]
        )

        self.exit = nn.Conv2d(
            hidden_channels,
            channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.entry(x)
        x = self.blocks(x)
        x = self.exit(x)

        return residual + x


class ImplicitFNOImageModel(nn.Module):
    """
    Single-branch FNO model for either:

        --mode surface
    or:
        --mode volume

    Input:
        conditioning vector [B, D]

    Output:
        premultiplied RGBA [B, 4, H, W]
    """

    def __init__(
        self,
        latent_dim: int,
        image_height: int = 32,
        image_width: int = 32,
        num_fourier_freqs: int = 4,
        projection_channels: int = 64,
        fno_hidden_channels: int = 96,
        fno_modes: int = 16,
        refiner_hidden_channels: int = 32,
        refiner_blocks: int = 3,
    ):
        super().__init__()

        self.latent_dim = int(latent_dim)
        self.image_height = int(image_height)
        self.image_width = int(image_width)

        pos_features = make_fourier_pos_features(
            height=image_height,
            width=image_width,
            num_freqs=num_fourier_freqs,
        )

        self.register_buffer("pos_features", pos_features)

        input_channels = self.latent_dim + pos_features.shape[0]

        self.input_projection = nn.Conv2d(
            input_channels,
            projection_channels,
            kernel_size=1,
        )

        self.fno = FNO(
            n_modes=(fno_modes, fno_modes),
            hidden_channels=fno_hidden_channels,
            in_channels=projection_channels,
            out_channels=4,
        )

        self.refiner = ImageRefiner(
            channels=4,
            hidden_channels=refiner_hidden_channels,
            num_blocks=refiner_blocks,
        )

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        batch_size, latent_dim = params.shape

        if latent_dim != self.latent_dim:
            raise ValueError(
                f"Expected latent dimension {self.latent_dim}, "
                f"received {latent_dim}."
            )

        params_grid = params.view(
            batch_size,
            latent_dim,
            1,
            1,
        ).expand(
            batch_size,
            latent_dim,
            self.image_height,
            self.image_width,
        )

        pos_features = self.pos_features.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
            -1,
        )

        x = torch.cat([params_grid, pos_features], dim=1)
        x = self.input_projection(x)

        x = self.fno(x)
        x = self.refiner(x)

        # Premultiplied RGB and alpha are both expected in [0, 1].
        return torch.sigmoid(x)