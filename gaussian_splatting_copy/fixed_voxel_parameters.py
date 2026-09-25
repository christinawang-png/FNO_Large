#!/usr/bin/env python
"""
Fixed hierarchical voxel-block parameters for the next neural scene pipeline.

Unlike LearnableNeuralSlice:
    - center and world_size are fixed buffers, not learnable parameters;
    - blocks have a soft alive/dead gate;
    - blocks have a soft surface/volume gate;
    - shape/material/light parameters remain learnable;
    - splitting/neighbor activation will be managed by a separate lifecycle
      file later.

The FNO feature layout must match train_implicit.py:

    ctrl_0_0_0 ... ctrl_1_1_1                  8
    sigma                                       1
    base_color_r, base_color_g, base_color_b    3
    metallic, roughness, specular               3
    opacity                                     1
    sin(phi), cos(phi), sin(theta), cos(theta)  4
    SH coefficients                             27
                                                ---
                                                47 total
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ============================================================
# TRAINING-DISTRIBUTION BOUNDS
# ============================================================

CTRL_LOW = -0.5
CTRL_HIGH = 0.5

SIGMA_LOW = 0.02
SIGMA_HIGH = 0.70

COLOR_LOW = 0.02
COLOR_HIGH = 1.00

OPACITY_LOW = 0.01
OPACITY_HIGH = 0.99

ROUGHNESS_LOW = 0.10
ROUGHNESS_HIGH = 0.90

NUM_CTRL_VALUES = 8

NUM_SH_BASIS = 9
NUM_SH_CHANNELS = 3
NUM_SH_VALUES = NUM_SH_BASIS * NUM_SH_CHANNELS

FNO_FEATURE_DIM = 47


# ============================================================
# NUMERICAL HELPERS
# ============================================================

def inverse_bounded(value, low, high, eps=0.02):
    """
    Convert a bounded physical value to a stable raw logit value.
    """
    value = float(value)
    low = float(low)
    high = float(high)

    if high <= low:
        raise ValueError(f"Expected high > low, got {low}, {high}")

    normalized = (value - low) / (high - low)
    normalized = min(max(normalized, eps), 1.0 - eps)

    return torch.logit(
        torch.tensor(normalized, dtype=torch.float32)
    )


def bounded(raw, low, high):
    """
    Map an unconstrained parameter to [low, high].
    """
    raw = raw.clamp(-12.0, 12.0)

    return float(low) + (
        float(high) - float(low)
    ) * torch.sigmoid(raw)


def probability_to_logit(probability, eps=1e-4):
    """
    Convert an initial [0, 1] probability to a finite logit.
    """
    probability = float(probability)
    probability = min(max(probability, eps), 1.0 - eps)

    return torch.logit(
        torch.tensor(probability, dtype=torch.float32)
    )


# ============================================================
# FIXED VOXEL BLOCK
# ============================================================

class FixedVoxelBlock(nn.Module):
    """
    One fixed-position voxel/octree block.

    Spatial data is fixed:

        center
        world_size
        level
        grid_index

    Learnable data:

        8 B-spline controls
        sigma
        base RGB
        opacity
        roughness
        local SH residual
        alive logit
        surface/volume mode logit

    Soft gates:

        alive_weight  = sigmoid(raw_alive_logit)
        volume_weight = sigmoid(raw_mode_logit)
        surface_weight = 1 - volume_weight

    A renderer should evaluate both frozen FNOs, blend their predicted
    premultiplied RGBA patches with the mode weights, then multiply the
    whole RGBA patch by alive_weight.
    """

    def __init__(
        self,
        center,
        world_size,
        ctrl_values,
        sigma,
        base_color_r,
        base_color_g,
        base_color_b,
        opacity,
        roughness,
        sh_values,
        metallic=0.0,
        specular=0.5,
        level=0,
        grid_index=None,
        parent_key=None,
        initial_alive_probability=0.99,
        initial_volume_probability=0.50,
        optimize_environment=True,
        local_sh_bound=0.005,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Fixed spatial hierarchy metadata
        # ----------------------------------------------------

        center = torch.as_tensor(
            center,
            dtype=torch.float32,
        ).reshape(-1)

        if center.numel() != 3:
            raise ValueError(
                f"center must have 3 values, got shape {tuple(center.shape)}"
            )

        world_size = float(world_size)

        if world_size <= 0.0:
            raise ValueError(
                f"world_size must be positive, got {world_size}"
            )

        self.register_buffer("center", center.clone())

        self.register_buffer(
            "world_size",
            torch.tensor(world_size, dtype=torch.float32),
        )

        self.level = int(level)

        if grid_index is None:
            grid_index = (-1, -1, -1)

        if len(grid_index) != 3:
            raise ValueError(
                "grid_index must contain exactly three integer values."
            )

        self.grid_index = tuple(int(value) for value in grid_index)
        self.parent_key = parent_key

        # ----------------------------------------------------
        # B-spline shape
        # ----------------------------------------------------

        ctrl_values = torch.as_tensor(
            ctrl_values,
            dtype=torch.float32,
        ).reshape(-1)

        if ctrl_values.numel() != NUM_CTRL_VALUES:
            raise ValueError(
                f"Expected {NUM_CTRL_VALUES} implicit B-spline controls, "
                f"got {ctrl_values.numel()}."
            )

        self.raw_ctrl = nn.Parameter(
            torch.stack(
                [
                    inverse_bounded(value, CTRL_LOW, CTRL_HIGH)
                    for value in ctrl_values
                ]
            )
        )

        self.raw_sigma = nn.Parameter(
            inverse_bounded(sigma, SIGMA_LOW, SIGMA_HIGH)
        )

        # ----------------------------------------------------
        # Appearance
        # ----------------------------------------------------

        self.raw_base_color_r = nn.Parameter(
            inverse_bounded(
                base_color_r,
                COLOR_LOW,
                COLOR_HIGH,
            )
        )

        self.raw_base_color_g = nn.Parameter(
            inverse_bounded(
                base_color_g,
                COLOR_LOW,
                COLOR_HIGH,
            )
        )

        self.raw_base_color_b = nn.Parameter(
            inverse_bounded(
                base_color_b,
                COLOR_LOW,
                COLOR_HIGH,
            )
        )

        self.raw_opacity = nn.Parameter(
            inverse_bounded(
                opacity,
                OPACITY_LOW,
                OPACITY_HIGH,
            )
        )

        self.raw_roughness = nn.Parameter(
            inverse_bounded(
                roughness,
                ROUGHNESS_LOW,
                ROUGHNESS_HIGH,
            )
        )

        # Keep surface material discrete/fixed as in the training set.
        self.register_buffer(
            "metallic_value",
            torch.tensor(float(metallic), dtype=torch.float32),
        )

        self.register_buffer(
            "specular_value",
            torch.tensor(float(specular), dtype=torch.float32),
        )

        # ----------------------------------------------------
        # Soft block existence and soft representation mode
        # ----------------------------------------------------

        self.raw_alive_logit = nn.Parameter(
            probability_to_logit(initial_alive_probability)
        )

        # Probability that this block uses volume mode.
        #
        # 0.0 -> surface
        # 1.0 -> volume
        self.raw_mode_logit = nn.Parameter(
            probability_to_logit(initial_volume_probability)
        )

        # ----------------------------------------------------
        # SH lighting
        # ----------------------------------------------------

        sh_values = torch.as_tensor(
            sh_values,
            dtype=torch.float32,
        )

        if sh_values.numel() != NUM_SH_VALUES:
            raise ValueError(
                "Expected order-2 RGB SH coefficients containing 27 values "
                f"(shape [9, 3] or [27]), got {tuple(sh_values.shape)}."
            )

        sh_values = sh_values.reshape(
            NUM_SH_BASIS,
            NUM_SH_CHANNELS,
        )

        self.register_buffer(
            "initial_sh",
            sh_values.clone(),
        )

        self.optimize_environment = bool(optimize_environment)
        self.local_sh_bound = float(local_sh_bound)

        self.raw_local_sh_delta = nn.Parameter(
            torch.zeros_like(sh_values),
            requires_grad=self.optimize_environment,
        )

        # ----------------------------------------------------
        # Initial anchors for non-spatial regularization
        # ----------------------------------------------------

        self.register_buffer(
            "initial_ctrl",
            ctrl_values.clone(),
        )

        self.register_buffer(
            "initial_sigma",
            torch.tensor(float(sigma), dtype=torch.float32),
        )

        self.register_buffer(
            "initial_base_color_r",
            torch.tensor(float(base_color_r), dtype=torch.float32),
        )

        self.register_buffer(
            "initial_base_color_g",
            torch.tensor(float(base_color_g), dtype=torch.float32),
        )

        self.register_buffer(
            "initial_base_color_b",
            torch.tensor(float(base_color_b), dtype=torch.float32),
        )

        self.register_buffer(
            "initial_opacity",
            torch.tensor(float(opacity), dtype=torch.float32),
        )

        self.register_buffer(
            "initial_roughness",
            torch.tensor(float(roughness), dtype=torch.float32),
        )

    # ========================================================
    # SOFT GATES
    # ========================================================

    @property
    def alive_weight(self):
        """
        Soft occupancy/existence weight in [0, 1].
        """
        return torch.sigmoid(self.raw_alive_logit)

    @property
    def volume_weight(self):
        """
        Soft probability of using the volume FNO.
        """
        return torch.sigmoid(self.raw_mode_logit)

    @property
    def surface_weight(self):
        """
        Soft probability of using the surface FNO.
        """
        return 1.0 - self.volume_weight

    @property
    def hard_mode(self):
        """
        Non-differentiable convenience label for inspection/export.
        """
        if float(self.volume_weight.detach()) >= 0.5:
            return "volume"

        return "surface"

    @property
    def is_effectively_alive(self):
        """
        Non-differentiable inspection helper.
        """
        return float(self.alive_weight.detach()) >= 0.5

    # ========================================================
    # LIGHTING
    # ========================================================

    @property
    def local_sh_delta(self):
        if not self.optimize_environment:
            return torch.zeros_like(self.raw_local_sh_delta)

        return self.local_sh_bound * torch.tanh(
            self.raw_local_sh_delta
        )

    def effective_sh(self, shared_sh=None):
        """
        Return [9, 3] SH values.
        """
        if shared_sh is None:
            base_sh = self.initial_sh
        else:
            base_sh = torch.as_tensor(
                shared_sh,
                dtype=self.center.dtype,
                device=self.center.device,
            )

            if base_sh.numel() != NUM_SH_VALUES:
                raise ValueError(
                    "shared_sh must contain 27 values / shape [9, 3], "
                    f"got {tuple(base_sh.shape)}."
                )

            base_sh = base_sh.reshape(
                NUM_SH_BASIS,
                NUM_SH_CHANNELS,
            )

        return base_sh + self.local_sh_delta

    # ========================================================
    # PHYSICAL VALUES
    # ========================================================

    def common_values(self, shared_sh=None):
        """
        Return values shared between the surface and volume conditionings.
        """
        return {
            "ctrl": bounded(
                self.raw_ctrl,
                CTRL_LOW,
                CTRL_HIGH,
            ),
            "sigma": bounded(
                self.raw_sigma,
                SIGMA_LOW,
                SIGMA_HIGH,
            ),
            "base_color_r": bounded(
                self.raw_base_color_r,
                COLOR_LOW,
                COLOR_HIGH,
            ),
            "base_color_g": bounded(
                self.raw_base_color_g,
                COLOR_LOW,
                COLOR_HIGH,
            ),
            "base_color_b": bounded(
                self.raw_base_color_b,
                COLOR_LOW,
                COLOR_HIGH,
            ),
            "opacity": bounded(
                self.raw_opacity,
                OPACITY_LOW,
                OPACITY_HIGH,
            ),
            "roughness": bounded(
                self.raw_roughness,
                ROUGHNESS_LOW,
                ROUGHNESS_HIGH,
            ),
            "sh": self.effective_sh(shared_sh),
        }

    def fno_values(self, mode, shared_sh=None):
        """
        Return the exact physical features required by either FNO.

        Parameters
        ----------
        mode:
            "surface" or "volume".

        For volume conditioning, metallic/roughness/specular are zero,
        matching the volume training metadata.
        """
        if mode not in {"surface", "volume"}:
            raise ValueError(f"Unknown FNO mode: {mode}")

        values = self.common_values(shared_sh=shared_sh)

        if mode == "surface":
            values["metallic"] = self.metallic_value
            values["specular"] = self.specular_value
        else:
            zero = torch.zeros_like(values["opacity"])

            values["metallic"] = zero
            values["roughness"] = zero
            values["specular"] = zero

        return values

    # ========================================================
    # REGULARIZATION
    # ========================================================

    def regularization_loss(
        self,
        shared_sh=None,
        alive_weight=1e-4,
        mode_entropy_weight=0.0,
    ):
        """
        Regularize learnable non-spatial parameters.

        Spatial center and size are fixed by design.

        Parameters
        ----------
        alive_weight:
            Weakly encourages unused blocks to become dead. This should
            normally be small, e.g. 1e-5 to 1e-4.

        mode_entropy_weight:
            Optional late-stage term that encourages surface/volume decisions
            to become hard. Keep zero initially. Turn on only after the scene
            has learned useful representations.
        """
        values = self.common_values(shared_sh=shared_sh)

        loss = torch.zeros(
            (),
            dtype=self.center.dtype,
            device=self.center.device,
        )

        loss = loss + 1e-5 * torch.mean(
            (values["ctrl"] - self.initial_ctrl).square()
        )

        loss = loss + 1e-5 * (
            values["sigma"] - self.initial_sigma
        ).square()

        loss = loss + 1e-5 * (
            values["base_color_r"] - self.initial_base_color_r
        ).square()

        loss = loss + 1e-5 * (
            values["base_color_g"] - self.initial_base_color_g
        ).square()

        loss = loss + 1e-5 * (
            values["base_color_b"] - self.initial_base_color_b
        ).square()

        loss = loss + 1e-5 * (
            values["opacity"] - self.initial_opacity
        ).square()

        loss = loss + 1e-5 * (
            values["roughness"] - self.initial_roughness
        ).square()

        loss = loss + 1e-2 * torch.mean(
            self.local_sh_delta.square()
        )

        # Weak sparsity pressure. The image reconstruction loss decides
        # whether blocks remain needed.
        loss = loss + float(alive_weight) * self.alive_weight

        if mode_entropy_weight > 0.0:
            p_volume = self.volume_weight.clamp(1e-6, 1.0 - 1e-6)

            entropy = -(
                p_volume * torch.log(p_volume)
                + (1.0 - p_volume) * torch.log(1.0 - p_volume)
            )

            loss = loss + float(mode_entropy_weight) * entropy

        return loss


# ============================================================
# FNO FEATURE CONSTRUCTION
# ============================================================

def build_tensor_fno_vector(
    voxel_block,
    mode,
    param_mean,
    param_std,
    phi,
    theta,
    device=None,
    shared_sh=None,
):
    """
    Build one normalized [1, 47] checkpoint conditioning vector.

    Parameters
    ----------
    voxel_block:
        FixedVoxelBlock.

    mode:
        "surface" or "volume", selecting the matching feature behavior.

    param_mean, param_std:
        Normalization arrays saved in the matching FNO checkpoint.

    phi, theta:
        Camera spherical angles in radians.
    """
    if mode not in {"surface", "volume"}:
        raise ValueError(f"mode must be surface or volume, got '{mode}'")

    if device is None:
        device = voxel_block.center.device

    phi = torch.as_tensor(
        phi,
        dtype=voxel_block.center.dtype,
        device=device,
    )

    theta = torch.as_tensor(
        theta,
        dtype=voxel_block.center.dtype,
        device=device,
    )

    values = voxel_block.fno_values(
        mode=mode,
        shared_sh=shared_sh,
    )

    scalars = []

    # Eight controls: ctrl_0_0_0 through ctrl_1_1_1.
    scalars.extend(values["ctrl"].reshape(-1).unbind())

    # Sigma.
    scalars.append(values["sigma"])

    # Base color.
    scalars.extend(
        [
            values["base_color_r"],
            values["base_color_g"],
            values["base_color_b"],
        ]
    )

    # Exact training order:
    #
    # metallic, roughness, specular, opacity
    scalars.extend(
        [
            values["metallic"],
            values["roughness"],
            values["specular"],
            values["opacity"],
        ]
    )

    # Camera direction.
    scalars.extend(
        [
            torch.sin(phi),
            torch.cos(phi),
            torch.sin(theta),
            torch.cos(theta),
        ]
    )

    # Order-2 RGB SH: [9,3] -> [27].
    scalars.extend(values["sh"].reshape(-1).unbind())

    raw = torch.stack(scalars)

    if raw.numel() != FNO_FEATURE_DIM:
        raise RuntimeError(
            f"Constructed {raw.numel()} features, expected "
            f"{FNO_FEATURE_DIM}."
        )

    param_mean = torch.as_tensor(
        param_mean,
        dtype=raw.dtype,
        device=device,
    ).reshape(-1)

    param_std = torch.as_tensor(
        param_std,
        dtype=raw.dtype,
        device=device,
    ).reshape(-1).clamp_min(1e-5)

    if param_mean.numel() != FNO_FEATURE_DIM:
        raise RuntimeError(
            f"Checkpoint mean has {param_mean.numel()} values; expected "
            f"{FNO_FEATURE_DIM}."
        )

    if param_std.numel() != FNO_FEATURE_DIM:
        raise RuntimeError(
            f"Checkpoint std has {param_std.numel()} values; expected "
            f"{FNO_FEATURE_DIM}."
        )

    return ((raw - param_mean) / param_std).unsqueeze(0)


# ============================================================
# HIERARCHY HELPERS
# ============================================================

def child_center_offsets(world_size, device, dtype):
    """
    Return the eight local child-center offsets for splitting a cubic block.

    Parent side length: s
    Child side length:  s / 2
    Child center offsets from parent: ±s/4 along every axis.
    """
    world_size = torch.as_tensor(
        world_size,
        dtype=dtype,
        device=device,
    )

    quarter = 0.25 * world_size

    signs = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0,  1.0],
            [-1.0,  1.0, -1.0],
            [-1.0,  1.0,  1.0],
            [ 1.0, -1.0, -1.0],
            [ 1.0, -1.0,  1.0],
            [ 1.0,  1.0, -1.0],
            [ 1.0,  1.0,  1.0],
        ],
        dtype=dtype,
        device=device,
    )

    return quarter * signs


def child_grid_indices(parent_grid_index):
    """
    Return eight integer octree child indices.

    Parent (ix, iy, iz) maps to children:

        (2*ix + dx, 2*iy + dy, 2*iz + dz)

    for dx/dy/dz in {0, 1}.
    """
    ix, iy, iz = parent_grid_index

    return [
        (
            2 * ix + dx,
            2 * iy + dy,
            2 * iz + dz,
        )
        for dx in (0, 1)
        for dy in (0, 1)
        for dz in (0, 1)
    ]