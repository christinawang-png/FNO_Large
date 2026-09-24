#!/usr/bin/env python
"""
Learnable local scene parameters for the implicit B-spline neural renderer.

This file is compatible with the current separately trained surface and
volume FNO checkpoints from train_implicit.py.

Checkpoint feature order:

    ctrl_0_0_0 ... ctrl_1_1_1                  8
    sigma                                       1
    base_color_r, base_color_g, base_color_b    3
    metallic, roughness, specular               3
    opacity                                     1
    sin(phi), cos(phi), sin(theta), cos(theta)  4
    SH coefficients, order-2 RGB                27
                                                ---
                                                47 total

There is no camera radius feature and no is_volume feature because:
  - radius was constant during training;
  - surface and volume use separate trained models.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ============================================================
# TRAINING-DISTRIBUTION BOUNDS
# ============================================================

# Four-level training controls were:
#
# [-0.5, -0.1667, 0.1667, 0.5]
#
# The learnable renderer permits continuous values in the same full range.
CTRL_LOW = -0.5
CTRL_HIGH = 0.5

# Gaussian thickness in normalized world-space units.
SIGMA_LOW = 0.02
SIGMA_HIGH = 0.70

# Renderer sampled opacity uniformly in this range.
OPACITY_LOW = 0.01
OPACITY_HIGH = 0.99

# Surface-only roughness range.
ROUGHNESS_LOW = 0.10
ROUGHNESS_HIGH = 0.90

# Base color range.
COLOR_LOW = 0.02
COLOR_HIGH = 1.00

# Current SH lighting uses order 2:
#
# 1 + 3 + 5 = 9 SH basis coefficients, each with RGB.
NUM_SH_BASIS = 9
NUM_SH_CHANNELS = 3
NUM_SH_VALUES = NUM_SH_BASIS * NUM_SH_CHANNELS

NUM_CTRL_VALUES = 8
FNO_FEATURE_DIM = 47


# ============================================================
# BOUNDED PARAMETERIZATION
# ============================================================

def inverse_bounded(value, low, high, eps=0.02):
    """
    Map a physical bounded scalar to a stable sigmoid-logit parameter.

    Values exactly at bounds are moved slightly inward so optimization begins
    outside sigmoid saturation.
    """
    value = float(value)
    low = float(low)
    high = float(high)

    if not high > low:
        raise ValueError(
            f"Expected high > low, got low={low}, high={high}"
        )

    normalized = (value - low) / (high - low)
    normalized = min(max(normalized, eps), 1.0 - eps)

    return torch.logit(
        torch.tensor(
            normalized,
            dtype=torch.float32,
        )
    )


def bounded(raw, low, high):
    """
    Map an unconstrained raw parameter into [low, high].

    The raw clamp prevents extremely saturated sigmoid values and keeps
    optimization numerically stable.
    """
    raw = raw.clamp(-12.0, 12.0)

    return float(low) + (
        float(high) - float(low)
    ) * torch.sigmoid(raw)


# ============================================================
# LEARNABLE SLICE
# ============================================================

class LearnableNeuralSlice(nn.Module):
    """
    One local learnable neural-rendering slice.

    The pretrained FNO network is frozen and lives outside this class.
    This class stores the learnable quantities used to construct its
    conditioning vector:

      - world-space placement and size;
      - 8 implicit B-spline corner controls;
      - sigma;
      - base RGB;
      - opacity;
      - roughness for surface slices;
      - optional local SH lighting residual.

    SH representation:

        effective_SH = shared_global_SH + local_SH_delta

    if shared_global_SH is supplied to fno_values(). Otherwise:

        effective_SH = initial_SH + local_SH_delta
    """

    def __init__(
        self,
        mode,
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
        optimize_environment=True,
        local_sh_bound=0.005,
    ):
        super().__init__()

        if mode not in {"surface", "volume"}:
            raise ValueError(
                f"mode must be 'surface' or 'volume', got '{mode}'"
            )

        self.mode = str(mode)
        self.local_sh_bound = float(local_sh_bound)
        self.optimize_environment = bool(optimize_environment)

        # ----------------------------------------------------
        # World-space placement
        # ----------------------------------------------------

        center = torch.as_tensor(
            center,
            dtype=torch.float32,
        ).reshape(-1)

        if center.numel() != 3:
            raise ValueError(
                f"center must contain 3 values, got shape {tuple(center.shape)}"
            )

        world_size = float(world_size)

        if world_size <= 0.0:
            raise ValueError(
                f"world_size must be positive, got {world_size}"
            )

        self.center = nn.Parameter(center.clone())

        # world_size = exp(raw_world_size), with a safety clamp in property.
        self.raw_world_size = nn.Parameter(
            torch.tensor(
                math.log(max(world_size, 1e-4)),
                dtype=torch.float32,
            )
        )

        # ----------------------------------------------------
        # Implicit B-spline shape parameters
        # ----------------------------------------------------

        ctrl_values = torch.as_tensor(
            ctrl_values,
            dtype=torch.float32,
        ).reshape(-1)

        if ctrl_values.numel() != NUM_CTRL_VALUES:
            raise ValueError(
                f"Expected {NUM_CTRL_VALUES} control values for a 2x2x2 "
                f"implicit B-spline, got {ctrl_values.numel()} values "
                f"with shape {tuple(ctrl_values.shape)}."
            )

        self.raw_ctrl = nn.Parameter(
            torch.stack(
                [
                    inverse_bounded(
                        value,
                        CTRL_LOW,
                        CTRL_HIGH,
                    )
                    for value in ctrl_values
                ]
            )
        )

        self.raw_sigma = nn.Parameter(
            inverse_bounded(
                sigma,
                SIGMA_LOW,
                SIGMA_HIGH,
            )
        )

        # ----------------------------------------------------
        # Appearance parameters
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

        # The dataset used metallic ∈ {0, 1}, and fixed specular=0.5.
        # Keep them fixed in this initial implementation.
        self.register_buffer(
            "metallic_value",
            torch.tensor(
                float(metallic),
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "specular_value",
            torch.tensor(
                float(specular),
                dtype=torch.float32,
            ),
        )

        # ----------------------------------------------------
        # SH environment / local lighting
        # ----------------------------------------------------

        sh_values = torch.as_tensor(
            sh_values,
            dtype=torch.float32,
        )

        if sh_values.numel() != NUM_SH_VALUES:
            raise ValueError(
                "Expected order-2 RGB SH values with either shape [9, 3] "
                f"or [27]; got shape {tuple(sh_values.shape)} with "
                f"{sh_values.numel()} values."
            )

        # Keep a consistent [9, 3] layout internally.
        sh_values = sh_values.reshape(
            NUM_SH_BASIS,
            NUM_SH_CHANNELS,
        )

        self.register_buffer(
            "initial_sh",
            sh_values.clone(),
        )

        self.raw_local_sh_delta = nn.Parameter(
            torch.zeros_like(sh_values),
            requires_grad=self.optimize_environment,
        )

        # ----------------------------------------------------
        # Initial-value regularization anchors
        # ----------------------------------------------------

        self.register_buffer(
            "initial_center",
            center.clone(),
        )

        self.register_buffer(
            "initial_world_size",
            torch.tensor(
                world_size,
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "initial_ctrl",
            ctrl_values.clone(),
        )

        self.register_buffer(
            "initial_sigma",
            torch.tensor(
                float(sigma),
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "initial_base_color_r",
            torch.tensor(
                float(base_color_r),
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "initial_base_color_g",
            torch.tensor(
                float(base_color_g),
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "initial_base_color_b",
            torch.tensor(
                float(base_color_b),
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "initial_opacity",
            torch.tensor(
                float(opacity),
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "initial_roughness",
            torch.tensor(
                float(roughness),
                dtype=torch.float32,
            ),
        )

    # ========================================================
    # DIFFERENTIABLE PROPERTIES
    # ========================================================

    @property
    def world_size(self):
        """
        Current scalar world-space slice size.

        Clamped to a practical range before exponentiation.
        """
        safe_raw_world_size = self.raw_world_size.clamp(
            min=math.log(0.05),
            max=math.log(5.0),
        )

        return torch.exp(safe_raw_world_size)

    @property
    def local_sh_delta(self):
        """
        Bounded local SH lighting perturbation.

        If environment optimization is disabled, return exact zero without
        depending on the raw parameter.
        """
        if not self.optimize_environment:
            return torch.zeros_like(self.raw_local_sh_delta)

        return self.local_sh_bound * torch.tanh(
            self.raw_local_sh_delta
        )

    # ========================================================
    # FNO CONDITIONING VALUES
    # ========================================================

    def fno_values(self, shared_sh=None):
        """
        Return differentiable values used by the pretrained FNO model.

        Returned keys match the current training feature schema.

        For volume mode:
          metallic = roughness = specular = 0

        because volume training stored those surface-only fields as zeros.
        """
        ctrl = bounded(
            self.raw_ctrl,
            CTRL_LOW,
            CTRL_HIGH,
        )

        sigma = bounded(
            self.raw_sigma,
            SIGMA_LOW,
            SIGMA_HIGH,
        )

        base_color_r = bounded(
            self.raw_base_color_r,
            COLOR_LOW,
            COLOR_HIGH,
        )

        base_color_g = bounded(
            self.raw_base_color_g,
            COLOR_LOW,
            COLOR_HIGH,
        )

        base_color_b = bounded(
            self.raw_base_color_b,
            COLOR_LOW,
            COLOR_HIGH,
        )

        opacity = bounded(
            self.raw_opacity,
            OPACITY_LOW,
            OPACITY_HIGH,
        )

        current_roughness = bounded(
            self.raw_roughness,
            ROUGHNESS_LOW,
            ROUGHNESS_HIGH,
        )

        if self.mode == "volume":
            metallic = torch.zeros_like(opacity)
            roughness_for_fno = torch.zeros_like(opacity)
            specular = torch.zeros_like(opacity)
        else:
            metallic = self.metallic_value
            roughness_for_fno = current_roughness
            specular = self.specular_value

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
                    "shared_sh must contain 27 values / have shape [9, 3], "
                    f"received {tuple(base_sh.shape)}."
                )

            base_sh = base_sh.reshape(
                NUM_SH_BASIS,
                NUM_SH_CHANNELS,
            )

        sh = base_sh + self.local_sh_delta

        return {
            "ctrl": ctrl,
            "sigma": sigma,
            "base_color_r": base_color_r,
            "base_color_g": base_color_g,
            "base_color_b": base_color_b,
            "metallic": metallic,
            "roughness": roughness_for_fno,
            "specular": specular,
            "opacity": opacity,
            "sh": sh,
        }

    # ========================================================
    # REGULARIZATION
    # ========================================================

    def regularization_loss(self, shared_sh=None):
        """
        Softly keep optimized slice parameters near their initialization.

        The FNO itself stays frozen; this regularizes only local scene
        conditioning and placement parameters.
        """
        values = self.fno_values(shared_sh=shared_sh)

        loss = torch.zeros(
            (),
            dtype=self.center.dtype,
            device=self.center.device,
        )

        # Placement and scale.
        loss = loss + 1e-5 * torch.mean(
            (self.center - self.initial_center).square()
        )

        relative_size_change = (
            self.world_size - self.initial_world_size
        ) / self.initial_world_size.clamp_min(1e-4)

        loss = loss + 1e-3 * relative_size_change.square()

        # Shape and volume thickness.
        loss = loss + 1e-5 * torch.mean(
            (values["ctrl"] - self.initial_ctrl).square()
        )

        loss = loss + 1e-5 * (
            values["sigma"] - self.initial_sigma
        ).square()

        # Appearance.
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

        if self.mode == "surface":
            loss = loss + 1e-5 * (
                values["roughness"] - self.initial_roughness
            ).square()

        # Local SH residual is intentionally tightly controlled.
        loss = loss + 1e-2 * torch.mean(
            self.local_sh_delta.square()
        )

        return loss


# ============================================================
# FNO FEATURE-VECTOR CONSTRUCTION
# ============================================================

def build_tensor_fno_vector(
    neural_slice,
    param_mean,
    param_std,
    phi,
    theta,
    device=None,
    shared_sh=None,
):
    """
    Construct one normalized FNO conditioning vector.

    Feature order must exactly match build_raw_feature_matrix() in
    implicit_dataset.py:

        ctrl_0_0_0 ... ctrl_1_1_1                 8
        sigma                                      1
        base_color_r, base_color_g, base_color_b   3
        metallic, roughness, specular              3
        opacity                                    1
        sin(phi), cos(phi), sin(theta), cos(theta) 4
        flattened SH [l,m,r/g/b]                   27

    Total: 47 scalar features.

    Parameters
    ----------
    neural_slice:
        LearnableNeuralSlice instance.

    param_mean, param_std:
        Arrays from the corresponding trained checkpoint.

    phi, theta:
        Scalar torch tensors in radians.

    device:
        Optional explicit device. Defaults to neural_slice.center.device.

    shared_sh:
        Optional scene/global SH tensor with shape [9,3] or [27].
    """
    if device is None:
        device = neural_slice.center.device

    values = neural_slice.fno_values(shared_sh=shared_sh)

    phi = torch.as_tensor(
        phi,
        dtype=neural_slice.center.dtype,
        device=device,
    )

    theta = torch.as_tensor(
        theta,
        dtype=neural_slice.center.dtype,
        device=device,
    )

    scalars = []

    # Eight control-grid values in z/y/x flattening order.
    scalars.extend(values["ctrl"].reshape(-1).unbind())

    # Shape thickness.
    scalars.append(values["sigma"])

    # Base RGB.
    scalars.extend(
        [
            values["base_color_r"],
            values["base_color_g"],
            values["base_color_b"],
        ]
    )

    # This exact order must match training:
    # metallic, roughness, specular, opacity.
    scalars.extend(
        [
            values["metallic"],
            values["roughness"],
            values["specular"],
            values["opacity"],
        ]
    )

    # Camera direction encoding.
    scalars.extend(
        [
            torch.sin(phi),
            torch.cos(phi),
            torch.sin(theta),
            torch.cos(theta),
        ]
    )

    # [9,3] -> [27] in the same row-major ordering as pandas SH columns:
    #
    # l0,m0 RGB; l1,m-1 RGB; l1,m0 RGB; ...
    scalars.extend(values["sh"].reshape(-1).unbind())

    raw = torch.stack(scalars)

    if raw.numel() != FNO_FEATURE_DIM:
        raise RuntimeError(
            f"Internal feature construction produced {raw.numel()} values; "
            f"expected {FNO_FEATURE_DIM}."
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

    if param_mean.numel() != raw.numel():
        raise RuntimeError(
            "FNO feature dimension mismatch:\n"
            f"  raw constructed features: {raw.numel()}\n"
            f"  checkpoint param_mean:    {param_mean.numel()}\n"
            f"  checkpoint param_std:     {param_std.numel()}\n"
            "Check that the correct surface/volume checkpoint is being "
            "used and that its training feature schema matches this file."
        )

    if param_std.numel() != raw.numel():
        raise RuntimeError(
            "FNO parameter-standard-deviation dimension mismatch:\n"
            f"  raw constructed features: {raw.numel()}\n"
            f"  checkpoint param_std:     {param_std.numel()}"
        )

    return ((raw - param_mean) / param_std).unsqueeze(0)