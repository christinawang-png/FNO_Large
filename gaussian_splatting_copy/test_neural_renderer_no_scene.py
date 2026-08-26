#!/usr/bin/env python

import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch


# ============================================================
# PATH SETUP
# ============================================================

REPO_DIR = Path(__file__).resolve().parent
FNO_LARGE_DIR = REPO_DIR.parent

sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(FNO_LARGE_DIR))


# ============================================================
# IMPORTS
# ============================================================

from train_premult_single_mode import FNOPlusResNetSingle
from neural_renderer import NeuralSliceRenderer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = FNO_LARGE_DIR / "plane_dataset_4"

RENDERS_DIR = BASE_DIR / "renders"
VOL_META_CSV = BASE_DIR / "metadata_volumes.csv"
IMG_META_CSV = RENDERS_DIR / "metadata_images_all_sharded.csv"

SURFACE_CHECKPOINT = FNO_LARGE_DIR / "fno_premult_surface_final.pt"
VOLUME_CHECKPOINT = FNO_LARGE_DIR / "fno_premult_volume_epoch025.pt"

OUTPUT_DIR = REPO_DIR / "no_scene_test_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (64, 64)

# This must match the radius used during FNO training.
FNO_RADIUS = 2.2

# Use the scene origin as the neural slice center for this test.
SLICE_CENTER = torch.tensor(
    [0.0, 0.0, 0.0],
    dtype=torch.float32,
)

# Select template rows from metadata.
# These are row indices in the surface/volume filtered metadata.
SURFACE_TEMPLATE_INDEX = 1000
VOLUME_TEMPLATE_INDEX = 1000

# Several synthetic camera viewpoints.
TEST_POSES = [
    (math.radians(35.0), math.radians(0.0)),
    (math.radians(55.0), math.radians(45.0)),
    (math.radians(90.0), math.radians(135.0)),
    (math.radians(125.0), math.radians(225.0)),
    (math.radians(150.0), math.radians(315.0)),
]


# ============================================================
# SYNTHETIC CAMERA
# ============================================================

class SyntheticCamera:
    """
    Minimal camera object needed by NeuralSliceRenderer.

    NeuralSliceRenderer only needs:
        camera_center

    The center uses the same spherical convention as Blender:
        x = R sin(phi) cos(theta)
        y = R sin(phi) sin(theta)
        z = R cos(phi)
    """

    def __init__(self, camera_center):
        self.camera_center = camera_center


def make_synthetic_camera(phi, theta, radius, slice_center):
    direction = torch.tensor(
        [
            math.sin(phi) * math.cos(theta),
            math.sin(phi) * math.sin(theta),
            math.cos(phi),
        ],
        dtype=torch.float32,
        device=slice_center.device,
    )

    camera_center = slice_center + radius * direction
    return SyntheticCamera(camera_center)


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(checkpoint_path, device):
    print("Loading checkpoint:", checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state = checkpoint.get("model_state", checkpoint)
    state.pop("_metadata", None)

    latent_dim = int(checkpoint["latent_dim"])

    param_mean = np.asarray(
        checkpoint["param_mean"],
        dtype=np.float32,
    )

    param_std = np.asarray(
        checkpoint["param_std"],
        dtype=np.float32,
    )

    model = FNOPlusResNetSingle(
        latent_dim=latent_dim,
        img_size=IMG_SIZE,
    ).to(device)

    model.load_state_dict(state)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    print("  latent_dim:", latent_dim)
    print("  loaded successfully")

    return model, param_mean, param_std, latent_dim


# ============================================================
# PARAMETER VECTOR BUILDER
# ============================================================

def make_parameter_builder(
    df_img,
    df_vol,
    surface_stats,
    volume_stats,
    device,
):
    """
    Build a callback for NeuralSliceRenderer.

    It reproduces the ordering used in:
        PlaneDatasetParamsToPremultRGBA._build_param_vector_np

    Camera values are replaced dynamically.
    Shape/material/environment values come from a template metadata row.
    """

    surface_mean, surface_std = surface_stats
    volume_mean, volume_std = volume_stats

    # Same control-point ordering used by the training dataset.
    ctrl_cols = [
        c for c in df_vol.columns
        if c.startswith("ctrl_")
    ]

    shape_meta = df_vol.set_index("sample_id").to_dict("index")

    # Match the SH ordering used by the image metadata.
    sh_cols = [
        c for c in df_img.columns
        if c.startswith("sh_l")
        and c.endswith(("_r", "_g", "_b"))
    ]

    print("Control columns:", ctrl_cols)
    print("Number of SH columns:", len(sh_cols))

    def build_fno_param_vector(
        slice_params,
        phi,
        theta,
        radius,
    ):
        mode = slice_params["mode"]

        if mode not in ("surface", "volume"):
            raise ValueError(
                f"Invalid mode: {mode}"
            )

        # Copy the selected image metadata row.
        row = slice_params["template_row"].copy()

        # Override camera-dependent fields.
        row["phi"] = float(
            phi.detach().cpu().item()
        )
        row["theta"] = float(
            theta.detach().cpu().item()
        )
        row["radius"] = float(
            radius.detach().cpu().item()
        )

        # Explicitly override mode.
        row["render_mode"] = mode

        # These are ignored by the volume model during training.
        if mode == "volume":
            row["metallic"] = 0.0
            row["roughness"] = 0.0
            row["specular"] = 0.0

        sample_id = int(row["sample_id"])

        if sample_id not in shape_meta:
            raise KeyError(
                f"sample_id={sample_id} not found in volume metadata"
            )

        shape_info = shape_meta[sample_id]

        # ----------------------------------------------------
        # Match _build_param_vector_np ordering exactly.
        # ----------------------------------------------------

        scalars = []

        # Shape: control points and sigma.
        for c in ctrl_cols:
            scalars.append(float(shape_info[c]))

        scalars.append(float(shape_info["sigma"]))

        # Material.
        hue = float(row["hue"])
        saturation = float(row["saturation"])
        metallic = float(row["metallic"])
        roughness = float(row["roughness"])
        opacity = float(row["opacity"])
        specular = float(row.get("specular", 0.5))

        is_volume = 1.0 if mode == "volume" else 0.0

        if is_volume > 0.5:
            metallic = 0.0
            roughness = 0.0
            specular = 0.0

        scalars.extend([
            hue,
            saturation,
            metallic,
            roughness,
            opacity,
            specular,
        ])

        # Camera.
        phi_value = float(
            phi.detach().cpu().item()
        )
        theta_value = float(
            theta.detach().cpu().item()
        )
        radius_value = float(
            radius.detach().cpu().item()
        )

        scalars.extend([
            math.sin(phi_value),
            math.cos(phi_value),
            math.sin(theta_value),
            math.cos(theta_value),
            radius_value,
        ])

        # Environment SH.
        for c in sh_cols:
            scalars.append(float(row[c]))

        # Mode indicator.
        scalars.append(is_volume)

        raw = np.asarray(
            scalars,
            dtype=np.float32,
        )

        if mode == "surface":
            param_mean = surface_mean
            param_std = surface_std
        else:
            param_mean = volume_mean
            param_std = volume_std

        if raw.shape[0] != param_mean.shape[0]:
            raise RuntimeError(
                f"{mode} parameter dimension mismatch: "
                f"constructed={raw.shape[0]}, "
                f"checkpoint={param_mean.shape[0]}"
            )

        normalized = (
            raw - param_mean
        ) / param_std

        return torch.from_numpy(
            normalized.astype(np.float32)
        ).to(device)

    return build_fno_param_vector


# ============================================================
# PREVIEW SAVING
# ============================================================

def save_preview(
    rgba_np,
    output_path,
    mode,
    pose_index,
    phi,
    theta,
    actual_radius,
):
    """
    Save:
        predicted premultiplied RGB
        predicted alpha
        overlay over black
        overlay over white
    """

    rgba_np = np.asarray(
        rgba_np,
        dtype=np.float32,
    )

    if rgba_np.shape != (4, 64, 64):
        raise RuntimeError(
            f"Expected shape (4,64,64), got {rgba_np.shape}"
        )

    rgba_np = np.clip(rgba_np, 0.0, 1.0)

    color = rgba_np[:3]       # premultiplied RGB
    alpha = rgba_np[3]        # [H,W]

    color_hw = np.transpose(
        color,
        (1, 2, 0),
    )

    alpha_3 = alpha[..., None]

    white = np.ones_like(color_hw)

    overlay_black = color_hw
    overlay_white = color_hw + (
        1.0 - alpha_3
    ) * white

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8, 8),
    )

    axes[0, 0].imshow(
        np.clip(color_hw, 0.0, 1.0)
    )
    axes[0, 0].set_title("Predicted premultiplied RGB")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(
        alpha,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    axes[0, 1].set_title("Predicted alpha")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(
        np.clip(overlay_black, 0.0, 1.0)
    )
    axes[1, 0].set_title("Overlay over black")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(
        np.clip(overlay_white, 0.0, 1.0)
    )
    axes[1, 1].set_title("Overlay over white")
    axes[1, 1].axis("off")

    fig.suptitle(
        f"{mode} | pose={pose_index} | "
        f"phi={math.degrees(phi):.1f}° | "
        f"theta={math.degrees(theta):.1f}° | "
        f"radius={actual_radius:.4f}",
        fontsize=10,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    if not torch.cuda.is_available():
        print("[WARN] CUDA is unavailable; using CPU.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # --------------------------------------------------------
    # Load only metadata.
    #
    # No Scene, GaussianModel, PLY, SPZ, or COLMAP scene is used.
    # --------------------------------------------------------
    df_img = pd.read_csv(
        IMG_META_CSV,
        low_memory=False,
    )

    df_vol = pd.read_csv(
        VOL_META_CSV,
    )

    if "render_mode" not in df_img.columns:
        raise RuntimeError(
            "render_mode is missing from image metadata"
        )

    print("Image metadata rows:", len(df_img))
    print(
        df_img["render_mode"].value_counts()
    )

    # --------------------------------------------------------
    # Load both FNOs.
    # --------------------------------------------------------
    surface_model, surface_mean, surface_std, surface_dim = \
        load_checkpoint(
            SURFACE_CHECKPOINT,
            device,
        )

    volume_model, volume_mean, volume_std, volume_dim = \
        load_checkpoint(
            VOLUME_CHECKPOINT,
            device,
        )

    if surface_dim != volume_dim:
        raise RuntimeError(
            f"Surface and volume dimensions differ: "
            f"{surface_dim} vs {volume_dim}"
        )

    # --------------------------------------------------------
    # Select template rows.
    # --------------------------------------------------------
    surface_rows = df_img[
        df_img["render_mode"].astype(str) == "surface"
    ].reset_index(drop=True)

    volume_rows = df_img[
        df_img["render_mode"].astype(str) == "volume"
    ].reset_index(drop=True)

    if len(surface_rows) == 0:
        raise RuntimeError("No surface rows found.")

    if len(volume_rows) == 0:
        raise RuntimeError("No volume rows found.")

    if SURFACE_TEMPLATE_INDEX >= len(surface_rows):
        raise IndexError(
            "SURFACE_TEMPLATE_INDEX is out of range"
        )

    if VOLUME_TEMPLATE_INDEX >= len(volume_rows):
        raise IndexError(
            "VOLUME_TEMPLATE_INDEX is out of range"
        )

    surface_template = surface_rows.iloc[
        SURFACE_TEMPLATE_INDEX
    ]

    volume_template = volume_rows.iloc[
        VOLUME_TEMPLATE_INDEX
    ]

    print(
        "Surface template sample_id:",
        int(surface_template["sample_id"]),
    )

    print(
        "Volume template sample_id:",
        int(volume_template["sample_id"]),
    )

    # --------------------------------------------------------
    # Create parameter builder.
    # --------------------------------------------------------
    build_param_vector = make_parameter_builder(
        df_img=df_img,
        df_vol=df_vol,
        surface_stats=(surface_mean, surface_std),
        volume_stats=(volume_mean, volume_std),
        device=device,
    )

    # --------------------------------------------------------
    # Create neural renderer.
    #
    # NeuralSliceRenderer uses FNO_RADIUS=2.2 internally.
    # --------------------------------------------------------
    renderer = NeuralSliceRenderer(
        surface_model=surface_model,
        volume_model=volume_model,
        build_param_vector=build_param_vector,
        device=device,
    )

    # --------------------------------------------------------
    # Test both model branches.
    # --------------------------------------------------------
    for mode, template_row in [
        ("surface", surface_template),
        ("volume", volume_template),
    ]:
        print("\nTesting mode:", mode)

        slice_params = {
            "center": SLICE_CENTER,
            "mode": mode,
            "template_row": template_row,
        }

        for pose_index, (phi, theta) in enumerate(TEST_POSES):
            camera = make_synthetic_camera(
                phi=phi,
                theta=theta,
                radius=FNO_RADIUS,
                slice_center=SLICE_CENTER,
            )

            with torch.no_grad():
                result = renderer(
                    camera,
                    slice_params,
                )

            rgba = result["rgba"]

            if rgba.ndim != 4:
                raise RuntimeError(
                    f"Expected [B,4,H,W], got {rgba.shape}"
                )

            rgba_np = rgba[0].detach().cpu().numpy()

            actual_radius = float(
                result["actual_radius"]
                .detach()
                .cpu()
                .item()
            )

            phi_out = float(
                result["phi"]
                .detach()
                .cpu()
                .item()
            )

            theta_out = float(
                result["theta"]
                .detach()
                .cpu()
                .item()
            )

            output_path = OUTPUT_DIR / (
                f"{mode}_pose_{pose_index:02d}.png"
            )

            save_preview(
                rgba_np=rgba_np,
                output_path=output_path,
                mode=mode,
                pose_index=pose_index,
                phi=phi_out,
                theta=theta_out,
                actual_radius=actual_radius,
            )

            print(
                f"Saved {output_path} | "
                f"phi={math.degrees(phi_out):.2f}° | "
                f"theta={math.degrees(theta_out):.2f}° | "
                f"radius={actual_radius:.6f}"
            )

    print("\nDone.")
    print("Outputs written to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()