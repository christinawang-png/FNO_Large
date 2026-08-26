# neural_renderer.py

import math
import torch
import torch.nn as nn


FNO_RADIUS = 2.2


def camera_to_slice_pose(viewpoint_camera, slice_center):
    """
    Convert a 3DGS camera position into spherical angles relative
    to the neural slice center.

    Returns:
        phi: spherical polar angle
        theta: azimuth angle
        actual_radius: real camera-to-slice distance
    """
    cam_center = viewpoint_camera.camera_center
    relative = cam_center - slice_center

    actual_radius = torch.linalg.norm(relative).clamp_min(1e-8)

    x = relative[0]
    y = relative[1]
    z = relative[2]

    phi = torch.acos(
        torch.clamp(z / actual_radius, -1.0, 1.0)
    )

    theta = torch.atan2(y, x)
    theta = torch.remainder(theta, 2.0 * math.pi)

    return phi, theta, actual_radius


def reconstruct_camera_position(phi, theta, radius):
    """
    Optional test helper.

    Reconstruct a relative camera position from spherical coordinates.
    """
    return torch.stack([
        radius * torch.sin(phi) * torch.cos(theta),
        radius * torch.sin(phi) * torch.sin(theta),
        radius * torch.cos(phi),
    ])


class NeuralSliceRenderer(nn.Module):
    """
    First prototype:
        3DGS camera + fixed neural slice parameters
        -> FNO
        -> premultiplied RGBA

    No optimization or multiple-layer compositing yet.
    """

    def __init__(
        self,
        surface_model,
        volume_model,
        build_param_vector,
        device,
    ):
        super().__init__()

        self.surface_model = surface_model
        self.volume_model = volume_model

        # This function will construct the normalized FNO vector.
        self.build_param_vector = build_param_vector

        self.device = device

        self.surface_model.eval()
        self.volume_model.eval()

    def forward(self, viewpoint_camera, slice_params):
        """
        viewpoint_camera:
            One 3DGS Camera object.

        slice_params:
            Dictionary or object containing the neural slice parameters,
            including:
                center
                mode
                shape/material/environment parameters

        Returns:
            dictionary containing:
                rgba
                phi
                theta
                actual_radius
        """

        slice_center = slice_params["center"].to(self.device)

        # Camera position relative to this slice.
        phi, theta, actual_radius = camera_to_slice_pose(
            viewpoint_camera,
            slice_center,
        )

        # The FNO was trained with a fixed radius of 2.2.
        fno_radius = torch.tensor(
            FNO_RADIUS,
            dtype=torch.float32,
            device=self.device,
        )

        # Build the exact normalized vector expected by the checkpoint.
        param_vec = self.build_param_vector(
            slice_params=slice_params,
            phi=phi,
            theta=theta,
            radius=fno_radius,
        )

        if param_vec.ndim == 1:
            param_vec = param_vec.unsqueeze(0)

        mode = slice_params["mode"]

        if mode == "surface":
            rgba = self.surface_model(param_vec)
        elif mode == "volume":
            rgba = self.volume_model(param_vec)
        else:
            raise ValueError(f"Unknown slice mode: {mode}")

        return {
            "rgba": rgba,
            "phi": phi,
            "theta": theta,
            "actual_radius": actual_radius,
        }