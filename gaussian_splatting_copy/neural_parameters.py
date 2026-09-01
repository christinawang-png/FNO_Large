#!/usr/bin/env python

import math

import torch
import torch.nn as nn


CTRL_LOW = 0.1
CTRL_HIGH = 0.9

SIGMA_LOW = 0.02
SIGMA_HIGH = 0.70

SAT_LOW = 0.30
SAT_HIGH = 0.90

OPACITY_LOW = 0.10
OPACITY_HIGH = 1.00

ROUGHNESS_LOW = 0.10
ROUGHNESS_HIGH = 0.90


def inverse_bounded(value, low, high, eps=0.02):
    """
    Convert a bounded scalar to a sigmoid-logit parameter.
    Avoids initializing exactly at sigmoid saturation.
    """
    value = float(value)
    t = (value - low) / (high - low)
    t = min(max(t, eps), 1.0 - eps)

    return torch.logit(
        torch.tensor(t, dtype=torch.float32)
    )


def bounded(raw, low, high):
    raw = raw.clamp(-12.0, 12.0)
    return low + (high - low) * torch.sigmoid(raw)


class LearnableNeuralSlice(nn.Module):
    """
    Learnable neural slice.

    The FNO weights remain frozen. This class stores learnable
    scene and FNO input parameters.

    SH lighting is represented as:

        SH_slice = SH_global + local_delta

    where local_delta is bounded by local_sh_bound.
    """

    def __init__(
        self,
        mode,
        center,
        world_size,
        ctrl_values,
        sigma,
        hue,
        saturation,
        opacity,
        roughness,
        sh_values,
        metallic=0.0,
        specular=0.5,
        optimize_environment=True,
        local_sh_bound=0.005,
    ):
        super().__init__()

        if mode not in ("surface", "volume"):
            raise ValueError(f"Invalid mode: {mode}")

        self.mode = mode
        self.local_sh_bound = float(local_sh_bound)
        self.optimize_environment = optimize_environment

        # ----------------------------------------------------
        # Placement
        # ----------------------------------------------------
        center = torch.as_tensor(
            center,
            dtype=torch.float32,
        )

        self.center = nn.Parameter(
            center.clone()
        )

        self.raw_world_size = nn.Parameter(
            torch.tensor(
                math.log(max(float(world_size), 1e-4)),
                dtype=torch.float32,
            )
        )

        # ----------------------------------------------------
        # Shape parameters
        # ----------------------------------------------------
        ctrl_values = torch.as_tensor(
            ctrl_values,
            dtype=torch.float32,
        )

        self.raw_ctrl = nn.Parameter(
            torch.stack([
                inverse_bounded(
                    value,
                    CTRL_LOW,
                    CTRL_HIGH,
                )
                for value in ctrl_values
            ])
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
        self.raw_hue = nn.Parameter(
            torch.tensor(
                float(hue),
                dtype=torch.float32,
            )
        )

        self.raw_saturation = nn.Parameter(
            inverse_bounded(
                saturation,
                SAT_LOW,
                SAT_HIGH,
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

        # Metallic and specular are kept fixed because your
        # training distribution used discrete metallic and fixed
        # specular values.
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
        # Environment
        # ----------------------------------------------------
        sh_values = torch.as_tensor(
            sh_values,
            dtype=torch.float32,
        )

        self.register_buffer(
            "initial_sh",
            sh_values.clone(),
        )

        self.raw_local_sh_delta = nn.Parameter(
            torch.zeros_like(sh_values),
            requires_grad=optimize_environment,
        )

        # ----------------------------------------------------
        # Initial values for regularization
        # ----------------------------------------------------
        self.register_buffer(
            "initial_center",
            center.clone(),
        )

        self.register_buffer(
            "initial_world_size",
            torch.tensor(
                float(world_size),
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
            "initial_hue",
            torch.tensor(
                float(hue),
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "initial_saturation",
            torch.tensor(
                float(saturation),
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

    @property
    def world_size(self):
        safe_raw_size = self.raw_world_size.clamp(
            min=math.log(0.05),
            max=math.log(5.0),
        )
        
        return torch.exp(safe_raw_size)

    @property
    def local_sh_delta(self):
        """
        Bounded local lighting residual.
        """
        if not self.optimize_environment:
            return torch.zeros_like(
                self.raw_local_sh_delta
            )

        return self.local_sh_bound * torch.tanh(
            self.raw_local_sh_delta
        )

    def fno_values(self, shared_sh=None):
        """
        Return differentiable values in the FNO training ranges.

        If shared_sh is supplied:

            SH = shared_sh + local residual

        Otherwise:

            SH = initial SH + local residual
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

        hue = torch.remainder(
            self.raw_hue.clamp(-100.0, 100.0),
            1.0,
        )

        saturation = bounded(
            self.raw_saturation,
            SAT_LOW,
            SAT_HIGH,
        )

        opacity = bounded(
            self.raw_opacity,
            OPACITY_LOW,
            OPACITY_HIGH,
        )

        roughness = bounded(
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
            roughness_for_fno = roughness
            specular = self.specular_value

        if shared_sh is None:
            sh = self.initial_sh + self.local_sh_delta
        else:
            sh = shared_sh + self.local_sh_delta

        return {
            "ctrl": ctrl,
            "sigma": sigma,
            "hue": hue,
            "saturation": saturation,
            "metallic": metallic,
            "roughness": roughness_for_fno,
            "opacity": opacity,
            "specular": specular,
            "sh": sh,
        }

    def regularization_loss(self, shared_sh=None):
        """
        Per-slice regularization.

        The local SH residual is kept small. Other parameters are
        softly encouraged to stay near initialization.
        """
        values = self.fno_values(
            shared_sh=shared_sh
        )

        loss = torch.zeros(
            (),
            dtype=self.center.dtype,
            device=self.center.device,
        )

        loss = loss + 1e-5 * torch.mean(
            (self.center - self.initial_center) ** 2
        )

        loss = loss + 1e-3 * (
            (
                self.world_size
                - self.initial_world_size
            )
            / self.initial_world_size.clamp_min(1e-4)
        ) ** 2

        loss = loss + 1e-5 * torch.mean(
            (values["ctrl"] - self.initial_ctrl) ** 2
        )

        loss = loss + 1e-5 * (
            values["sigma"] - self.initial_sigma
        ) ** 2

        loss = loss + 1e-5 * (
            values["hue"] - self.initial_hue
        ) ** 2

        loss = loss + 1e-5 * (
            values["saturation"]
            - self.initial_saturation
        ) ** 2

        loss = loss + 1e-5 * (
            values["opacity"]
            - self.initial_opacity
        ) ** 2

        if self.mode == "surface":
            loss = loss + 1e-5 * (
                values["roughness"]
                - self.initial_roughness
            ) ** 2

        # Stronger penalty on local lighting residual.
        loss = loss + 1e-2 * torch.mean(
            self.local_sh_delta ** 2
        )

        return loss


def build_tensor_fno_vector(
    neural_slice,
    param_mean,
    param_std,
    phi,
    theta,
    fno_radius,
    device,
    shared_sh=None,
):
    """
    Build the normalized FNO input vector using torch only.

    Feature order must match training:

        ctrl
        sigma
        hue
        saturation
        metallic
        roughness
        opacity
        specular
        sin(phi)
        cos(phi)
        sin(theta)
        cos(theta)
        radius
        SH values
        is_volume
    """
    values = neural_slice.fno_values(
        shared_sh=shared_sh
    )

    scalars = []

    for value in values["ctrl"]:
        scalars.append(value)

    scalars.extend([
        values["sigma"],
        values["hue"],
        values["saturation"],
        values["metallic"],
        values["roughness"],
        values["opacity"],
        values["specular"],
        torch.sin(phi),
        torch.cos(phi),
        torch.sin(theta),
        torch.cos(theta),
        torch.as_tensor(
            float(fno_radius),
            dtype=phi.dtype,
            device=device,
        ),
    ])

    for sh_value in values["sh"]:
        scalars.append(sh_value)

    scalars.append(
        torch.as_tensor(
            1.0 if neural_slice.mode == "volume" else 0.0,
            dtype=phi.dtype,
            device=device,
        )
    )

    raw = torch.stack(scalars)

    param_mean = torch.as_tensor(
        param_mean,
        dtype=raw.dtype,
        device=device,
    )

    param_std = torch.as_tensor(
        param_std,
        dtype=raw.dtype,
        device=device,
    ).clamp_min(1e-5)

    if raw.numel() != param_mean.numel():
        raise RuntimeError(
            f"FNO vector dimension mismatch: "
            f"raw={raw.numel()}, "
            f"mean={param_mean.numel()}"
        )

    return (
        (raw - param_mean) / param_std
    ).unsqueeze(0)