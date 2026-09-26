#!/usr/bin/env python
"""
Learnable moving neural slices with a soft surface/volume gate.

Compatible with the current separately trained implicit B-spline FNO models.

The old hard `mode` field remains for:
    - loading old checkpoints;
    - initializing raw_mode_logit;
    - lifecycle/checkpoint metadata.

The renderer should no longer use mode as a hard selection. Instead, it
evaluates both FNOs for every slice and blends their premultiplied RGBA
patches using:

    volume_weight = sigmoid(raw_mode_logit)
    surface_weight = 1 - volume_weight
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

OPACITY_LOW = 0.01
OPACITY_HIGH = 0.99

ROUGHNESS_LOW = 0.10
ROUGHNESS_HIGH = 0.90

COLOR_LOW = 0.02
COLOR_HIGH = 1.00

NUM_CTRL_VALUES = 8

NUM_SH_BASIS = 9
NUM_SH_CHANNELS = 3
NUM_SH_VALUES = NUM_SH_BASIS * NUM_SH_CHANNELS

FNO_FEATURE_DIM = 47


# ============================================================
# NUMERICAL HELPERS
# ============================================================

def inverse_bounded(value, low, high, eps=0.02):
    """Convert a bounded physical value to a stable raw sigmoid logit."""
    value = float(value)
    low = float(low)
    high = float(high)

    if high <= low:
        raise ValueError(
            f"Expected high > low, got low={low}, high={high}"
        )

    normalized = (value - low) / (high - low)
    normalized = min(max(normalized, eps), 1.0 - eps)

    return torch.logit(
        torch.tensor(normalized, dtype=torch.float32)
    )


def bounded(raw, low, high):
    """Map an unconstrained raw scalar/tensor into [low, high]."""
    raw = raw.clamp(-12.0, 12.0)

    return float(low) + (
        float(high) - float(low)
    ) * torch.sigmoid(raw)


def probability_to_logit(probability, eps=1e-4):
    """Convert an initial probability in [0,1] into a finite logit."""
    probability = float(probability)
    probability = min(max(probability, eps), 1.0 - eps)

    return torch.logit(
        torch.tensor(probability, dtype=torch.float32)
    )


# ============================================================
# LEARNABLE NEURAL SLICE
# ============================================================

class LearnableNeuralSlice(nn.Module):
    """
    One freely movable neural slice.

    FNO weights are frozen externally. This module owns scene parameters:
        center, world size, implicit controls, sigma, RGB, opacity,
        roughness, local SH, and soft surface/volume gate.
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

        # Legacy checkpoint/init label. The renderer should use the soft
        # mode gate below rather than this field for hard model selection.
        self.mode = str(mode)

        self.optimize_environment = bool(optimize_environment)
        self.local_sh_bound = float(local_sh_bound)

        # ----------------------------------------------------
        # Placement
        # ----------------------------------------------------

        center = torch.as_tensor(
            center,
            dtype=torch.float32,
        ).reshape(-1)

        if center.numel() != 3:
            raise ValueError(
                f"center must contain 3 values, got {tuple(center.shape)}"
            )

        world_size = float(world_size)

        if world_size <= 0.0:
            raise ValueError(
                f"world_size must be positive, got {world_size}"
            )

        self.center = nn.Parameter(center.clone())

        self.raw_world_size = nn.Parameter(
            torch.tensor(
                math.log(max(world_size, 1e-4)),
                dtype=torch.float32,
            )
        )

        # ----------------------------------------------------
        # Implicit shape
        # ----------------------------------------------------

        ctrl_values = torch.as_tensor(
            ctrl_values,
            dtype=torch.float32,
        ).reshape(-1)

        if ctrl_values.numel() != NUM_CTRL_VALUES:
            raise ValueError(
                f"Expected {NUM_CTRL_VALUES} control values, got "
                f"{ctrl_values.numel()}."
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
            inverse_bounded(
                sigma,
                SIGMA_LOW,
                SIGMA_HIGH,
            )
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

        self.register_buffer(
            "metallic_value",
            torch.tensor(float(metallic), dtype=torch.float32),
        )

        self.register_buffer(
            "specular_value",
            torch.tensor(float(specular), dtype=torch.float32),
        )

        # ----------------------------------------------------
        # Soft surface/volume mode gate
        #
        # This is new and absent from old hard-mode checkpoints.
        # The old mode determines a near-hard initial state.
        # ----------------------------------------------------

        initial_volume_probability = (
            0.95 if mode == "volume" else 0.05
        )

        self.raw_mode_logit = nn.Parameter(
            probability_to_logit(
                initial_volume_probability
            )
        )

        # ----------------------------------------------------
        # Lighting
        # ----------------------------------------------------

        sh_values = torch.as_tensor(
            sh_values,
            dtype=torch.float32,
        )

        if sh_values.numel() != NUM_SH_VALUES:
            raise ValueError(
                f"Expected {NUM_SH_VALUES} SH values, got "
                f"{sh_values.numel()}."
            )

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
        # Regularization anchors
        # ----------------------------------------------------

        self.register_buffer("initial_center", center.clone())

        self.register_buffer(
            "initial_world_size",
            torch.tensor(world_size, dtype=torch.float32),
        )

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
    # PROPERTIES
    # ========================================================

    @property
    def world_size(self):
        safe_raw_size = self.raw_world_size.clamp(
            min=math.log(0.05),
            max=math.log(5.0),
        )

        return torch.exp(safe_raw_size)

    @property
    def volume_weight(self):
        """Differentiable local probability of volume rendering."""
        return torch.sigmoid(self.raw_mode_logit)

    @property
    def surface_weight(self):
        """Differentiable local probability of surface rendering."""
        return 1.0 - self.volume_weight

    @property
    def hard_mode(self):
        """Convenience non-differentiable mode label for diagnostics."""
        if float(self.volume_weight.detach()) >= 0.5:
            return "volume"

        return "surface"

    @property
    def local_sh_delta(self):
        if not self.optimize_environment:
            return torch.zeros_like(self.raw_local_sh_delta)

        return self.local_sh_bound * torch.tanh(
            self.raw_local_sh_delta
        )

    # ========================================================
    # FNO VALUES
    # ========================================================

    def fno_values(
        self,
        mode=None,
        shared_sh=None,
    ):
        """
        Return physical FNO conditioning values.

        Parameters
        ----------
        mode:
            Explicit "surface" or "volume" request.

            If None, preserves old hard-mode behavior via self.mode.
            Lifecycle/checkpoint code may use this default.

        shared_sh:
            Optional current scene/global SH tensor.
        """
        if mode is None:
            mode = self.mode

        if mode not in {"surface", "volume"}:
            raise ValueError(
                f"mode must be surface or volume, got '{mode}'"
            )

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

        if mode == "surface":
            metallic = self.metallic_value
            roughness_for_fno = current_roughness
            specular = self.specular_value
        else:
            zero = torch.zeros_like(opacity)

            metallic = zero
            roughness_for_fno = zero
            specular = zero

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
                    f"Expected {NUM_SH_VALUES} shared SH values, got "
                    f"{base_sh.numel()}."
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

    def regularization_loss(
        self,
        shared_sh=None,
        mode_entropy_weight=0.0,
    ):
        """
        Per-slice regularization.

        mode_entropy_weight should initially remain 0.0. Later it may be
        increased slightly to encourage a near-binary surface/volume choice.
        """
        values = self.fno_values(
            shared_sh=shared_sh,
        )

        loss = torch.zeros(
            (),
            dtype=self.center.dtype,
            device=self.center.device,
        )

        loss = loss + 1e-5 * torch.mean(
            (self.center - self.initial_center).square()
        )

        relative_size_change = (
            self.world_size - self.initial_world_size
        ) / self.initial_world_size.clamp_min(1e-4)

        loss = loss + 1e-3 * relative_size_change.square()

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

        # This is harmless even for a legacy volume slice because mode=None
        # makes its roughness FNO value zero. The raw parameter itself can
        # still be anchored gently.
        loss = loss + 1e-5 * (
            bounded(
                self.raw_roughness,
                ROUGHNESS_LOW,
                ROUGHNESS_HIGH,
            ) - self.initial_roughness
        ).square()

        loss = loss + 1e-2 * torch.mean(
            self.local_sh_delta.square()
        )

        if mode_entropy_weight > 0.0:
            p_volume = self.volume_weight.clamp(
                1e-6,
                1.0 - 1e-6,
            )

            entropy = -(
                p_volume * torch.log(p_volume)
                + (1.0 - p_volume)
                * torch.log(1.0 - p_volume)
            )

            loss = loss + float(mode_entropy_weight) * entropy

        return loss


# ============================================================
# CHECKPOINT-CONDITIONING VECTOR
# ============================================================

def build_tensor_fno_vector(
    neural_slice,
    mode,
    param_mean,
    param_std,
    phi,
    theta,
    device=None,
    shared_sh=None,
):
    """
    Build normalized [1,47] input for either the surface or volume FNO.

    Feature order must match the trained checkpoints:

        8 ctrl
        sigma
        RGB
        metallic, roughness, specular
        opacity
        sin(phi), cos(phi), sin(theta), cos(theta)
        27 SH values
    """
    if mode not in {"surface", "volume"}:
        raise ValueError(
            f"mode must be surface or volume, got '{mode}'"
        )

    if device is None:
        device = neural_slice.center.device

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

    values = neural_slice.fno_values(
        mode=mode,
        shared_sh=shared_sh,
    )

    scalars = []

    scalars.extend(values["ctrl"].reshape(-1).unbind())

    scalars.append(values["sigma"])

    scalars.extend(
        [
            values["base_color_r"],
            values["base_color_g"],
            values["base_color_b"],
        ]
    )

    # Exact trained order.
    scalars.extend(
        [
            values["metallic"],
            values["roughness"],
            values["specular"],
            values["opacity"],
        ]
    )

    scalars.extend(
        [
            torch.sin(phi),
            torch.cos(phi),
            torch.sin(theta),
            torch.cos(theta),
        ]
    )

    scalars.extend(values["sh"].reshape(-1).unbind())

    raw = torch.stack(scalars)

    if raw.numel() != FNO_FEATURE_DIM:
        raise RuntimeError(
            f"Built {raw.numel()} FNO inputs, expected "
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

    if param_mean.numel() != raw.numel():
        raise RuntimeError(
            f"FNO mean mismatch: raw={raw.numel()}, "
            f"mean={param_mean.numel()}."
        )

    if param_std.numel() != raw.numel():
        raise RuntimeError(
            f"FNO std mismatch: raw={raw.numel()}, "
            f"std={param_std.numel()}."
        )

    return ((raw - param_mean) / param_std).unsqueeze(0)