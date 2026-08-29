#!/usr/bin/env python

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Differentiable placement
# ============================================================

def place_patch_uniform(
    patch,
    canvas_height,
    canvas_width,
    center_x,
    center_y,
    patch_size,
):
    """
    Uniformly resize and place a patch into a full canvas.

    patch:
        [B,C,H,W]

    center_x, center_y:
        Output-canvas pixel coordinates.

    patch_size:
        Same output width and height in pixels.
    """
    if patch.ndim != 4:
        raise ValueError(
            f"Expected patch [B,C,H,W], got {patch.shape}"
        )

    batch_size = patch.shape[0]
    dtype = patch.dtype
    device = patch.device

    center_x = torch.as_tensor(
        center_x,
        dtype=dtype,
        device=device,
    ).reshape(-1)

    center_y = torch.as_tensor(
        center_y,
        dtype=dtype,
        device=device,
    ).reshape(-1)

    patch_size = torch.as_tensor(
        patch_size,
        dtype=dtype,
        device=device,
    ).reshape(-1)

    if center_x.numel() == 1:
        center_x = center_x.expand(batch_size)

    if center_y.numel() == 1:
        center_y = center_y.expand(batch_size)

    if patch_size.numel() == 1:
        patch_size = patch_size.expand(batch_size)

    patch_size = patch_size.clamp_min(2.0)

    canvas_wm1 = float(max(canvas_width - 1, 1))
    canvas_hm1 = float(max(canvas_height - 1, 1))

    center_x_norm = (
        2.0 * center_x / canvas_wm1 - 1.0
    )

    center_y_norm = (
        2.0 * center_y / canvas_hm1 - 1.0
    )

    scale_x = canvas_wm1 / patch_size
    scale_y = canvas_hm1 / patch_size

    theta = torch.zeros(
        batch_size,
        2,
        3,
        dtype=dtype,
        device=device,
    )

    theta[:, 0, 0] = scale_x
    theta[:, 1, 1] = scale_y

    theta[:, 0, 2] = -scale_x * center_x_norm
    theta[:, 1, 2] = -scale_y * center_y_norm

    grid = F.affine_grid(
        theta,
        size=(
            batch_size,
            patch.shape[1],
            canvas_height,
            canvas_width,
        ),
        align_corners=True,
    )

    return F.grid_sample(
        patch,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


# ============================================================
# Premultiplied compositing
# ============================================================

def alpha_over(back, front):
    """
    Composite premultiplied RGBA front over back.

    Inputs:
        back/front: [B,4,H,W]
    """
    back_color = back[:, :3]
    back_alpha = back[:, 3:4]

    front_color = front[:, :3]
    front_alpha = front[:, 3:4]

    output_color = (
        front_color
        + (1.0 - front_alpha) * back_color
    )

    output_alpha = (
        front_alpha
        + (1.0 - front_alpha) * back_alpha
    )

    return torch.cat(
        [output_color, output_alpha],
        dim=1,
    )


# ============================================================
# Projection helpers
# ============================================================

def project_world_point(viewpoint_camera, point_world):
    """
    Project a world-space point into a 3DGS camera.

    Uses the row-vector convention from the 3DGS code:
        clip = point_h @ full_proj_transform
    """
    matrix = viewpoint_camera.full_proj_transform

    point_world = point_world.to(
        device=matrix.device,
        dtype=matrix.dtype,
    )

    point_h = torch.cat([
        point_world,
        torch.ones(
            1,
            dtype=matrix.dtype,
            device=matrix.device,
        ),
    ])

    clip = point_h @ matrix

    w = clip[3].clamp_min(1e-8)
    ndc = clip[:3] / w

    width = float(viewpoint_camera.image_width)
    height = float(viewpoint_camera.image_height)

    pixel_x = (ndc[0] + 1.0) * 0.5 * width
    pixel_y = (1.0 - ndc[1]) * 0.5 * height

    return pixel_x, pixel_y, ndc[2]


def camera_to_slice_pose(viewpoint_camera, slice_center):
    """
    Compute camera position relative to a slice.

    Returns:
        phi
        theta
        actual_radius
        relative_position
    """
    camera_center = viewpoint_camera.camera_center

    slice_center = slice_center.to(
        device=camera_center.device,
        dtype=camera_center.dtype,
    )

    relative = camera_center - slice_center

    actual_radius = torch.linalg.norm(
        relative
    ).clamp_min(1e-8)

    x = relative[0]
    y = relative[1]
    z = relative[2]

    phi = torch.acos(
        torch.clamp(
            z / actual_radius,
            -1.0,
            1.0,
        )
    )

    theta = torch.atan2(y, x)
    theta = torch.remainder(
        theta,
        2.0 * math.pi,
    )

    return phi, theta, actual_radius, relative


# ============================================================
# Learnable neural slice
# ============================================================

class NeuralSlice(nn.Module):
    """
    One learnable neural slice.

    Initially, only center and world_size are learnable.
    The FNO template parameters remain fixed.

    mode:
        "surface" or "volume"

    center:
        Initial world-space center, shape [3].

    world_size:
        Initial uniform world-space size.

    template_row:
        Pandas row containing the FNO shape/material/environment
        parameters.

    param_mean/std:
        Checkpoint normalization values.
    """
    def __init__(
        self,
        mode,
        center,
        world_size,
        template_row,
        param_mean,
        param_std,
    ):
        super().__init__()

        if mode not in ("surface", "volume"):
            raise ValueError(
                f"Invalid mode: {mode}"
            )

        center = torch.as_tensor(
            center,
            dtype=torch.float32,
        )

        if center.shape != (3,):
            raise ValueError(
                f"center must have shape [3], got {center.shape}"
            )

        self.mode = mode
        self.template_row = template_row.copy()

        # Learnable world position.
        self.center = nn.Parameter(
            center.clone()
        )

        # Log parameter guarantees positive size.
        self.raw_world_size = nn.Parameter(
            torch.tensor(
                math.log(float(world_size)),
                dtype=torch.float32,
            )
        )

        # Stored as non-learnable metadata.
        self.param_mean = param_mean
        self.param_std = param_std

    @property
    def world_size(self):
        return torch.exp(
            self.raw_world_size
        ).clamp_min(1e-4)

    def extra_repr(self):
        return (
            f"mode={self.mode}, "
            f"world_size={self.world_size.item():.4f}"
        )


# ============================================================
# Neural scene
# ============================================================

class NeuralScene(nn.Module):
    """
    Collection of NeuralSlice objects.

    Slice order is initially explicit and fixed:
        first slice = back
        last slice = front
    """
    def __init__(self, slices=None):
        super().__init__()

        if slices is None:
            slices = []

        self.slices = nn.ModuleList(slices)

    def add_slice(self, neural_slice):
        if not isinstance(neural_slice, NeuralSlice):
            raise TypeError(
                "Expected a NeuralSlice."
            )

        self.slices.append(neural_slice)

    def __len__(self):
        return len(self.slices)


# ============================================================
# Neural scene renderer
# ============================================================

class NeuralSceneRenderer(nn.Module):
    """
    Renders multiple neural slices into one full-size canvas.

    The FNO models should already be loaded and frozen.

    build_param_vector must have the signature:

        build_param_vector(
            dataset=...,
            template_row=...,
            mode=...,
            phi=...,
            theta=...,
            fno_radius=...,
            param_mean=...,
            param_std=...,
            device=...,
        )

    This matches the parameter-builder used in your current
    scene test scripts.
    """
    def __init__(
        self,
        surface_model,
        volume_model,
        dataset,
        build_param_vector,
        fno_radius=2.2,
        canonical_patch_size=64,
    ):
        super().__init__()

        self.surface_model = surface_model
        self.volume_model = volume_model
        self.dataset = dataset
        self.build_param_vector = build_param_vector

        self.fno_radius = float(fno_radius)
        self.canonical_patch_size = int(
            canonical_patch_size
        )

        # The pretrained models are not optimized by this module.
        self.surface_model.eval()
        self.volume_model.eval()

        for parameter in self.surface_model.parameters():
            parameter.requires_grad_(False)

        for parameter in self.volume_model.parameters():
            parameter.requires_grad_(False)

    def render_slice(
        self,
        viewpoint_camera,
        neural_slice,
    ):
        """
        Render one slice onto a full camera-sized canvas.

        Returns:
            layer: [1,4,H,W]
            diagnostics: dictionary
        """
        device = viewpoint_camera.camera_center.device

        slice_center = neural_slice.center.to(device)

        canvas_h = int(
            viewpoint_camera.image_height
        )

        canvas_w = int(
            viewpoint_camera.image_width
        )

        # Camera pose relative to this slice.
        phi, theta, actual_radius, relative = \
            camera_to_slice_pose(
                viewpoint_camera,
                slice_center,
            )

        # Build normalized FNO input.
        #
        # Note: your current builder converts phi/theta to
        # Python floats, so FNO appearance is not currently
        # differentiable through camera angle. Placement remains
        # differentiable through center and size.
        param_vec = self.build_param_vector(
            dataset=self.dataset,
            template_row=neural_slice.template_row,
            mode=neural_slice.mode,
            phi=phi,
            theta=theta,
            fno_radius=self.fno_radius,
            param_mean=neural_slice.param_mean,
            param_std=neural_slice.param_std,
            device=device,
        )

        if neural_slice.mode == "surface":
            patch = self.surface_model(param_vec)
        else:
            patch = self.volume_model(param_vec)

        patch = patch.clamp(
            0.0,
            1.0,
        )

        if patch.shape[1:] != (4, 64, 64):
            raise RuntimeError(
                f"Expected FNO output [B,4,64,64], "
                f"got {tuple(patch.shape)}"
            )

        # Project slice center into camera.
        center_x, center_y, center_depth = \
            project_world_point(
                viewpoint_camera,
                slice_center,
            )

        # Uniform scene-to-image scale.
        #
        # The canonical FNO patch represents a unit-sized
        # slice at fno_radius. Actual world distance changes
        # only the output footprint.
        patch_size = (
            float(canvas_h)
            * neural_slice.world_size
            * self.fno_radius
            / actual_radius
        )

        layer = place_patch_uniform(
            patch=patch,
            canvas_height=canvas_h,
            canvas_width=canvas_w,
            center_x=center_x,
            center_y=center_y,
            patch_size=patch_size,
        )

        layer = layer.clamp(
            0.0,
            1.0,
        )

        diagnostics = {
            "phi": phi,
            "theta": theta,
            "actual_radius": actual_radius,
            "relative": relative,
            "center_x": center_x,
            "center_y": center_y,
            "center_depth": center_depth,
            "patch_size": patch_size,
        }

        return layer, diagnostics

    def forward(
        self,
        viewpoint_camera,
        neural_scene,
    ):
        """
        Render all slices.

        Slices are composited in the order stored in
        neural_scene.slices:
            first = back
            last = front

        Returns:
            composite: [1,4,H,W]
            diagnostics: list of dictionaries
        """
        if len(neural_scene) == 0:
            raise RuntimeError(
                "NeuralScene contains no slices."
            )

        layers = []
        diagnostics = []

        for neural_slice in neural_scene.slices:
            layer, info = self.render_slice(
                viewpoint_camera,
                neural_slice,
            )

            layers.append(layer)
            diagnostics.append(info)

        composite = torch.zeros_like(
            layers[0]
        )

        for layer in layers:
            composite = alpha_over(
                back=composite,
                front=layer,
            )

        return composite.clamp(0.0, 1.0), diagnostics