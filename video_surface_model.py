#!/usr/bin/env python

import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch

from train_premult_single_mode import (
    PlaneDatasetParamsToPremultRGBA,
    FNOPlusResNetSingle,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path("./plane_dataset_4")

RENDERS_DIR = BASE_DIR / "renders"
ALPHA_DIR = BASE_DIR / "hard_alpha"

IMG_META_CSV = RENDERS_DIR / "metadata_images_all_sharded.csv"
VOL_META_CSV = BASE_DIR / "metadata_volumes.csv"
ALPHA_META_CSV = ALPHA_DIR / "metadata_alpha_all.csv"

CHECKPOINT_PATH = Path("fno_premult_surface_final.pt")

OUTPUT_DIR = BASE_DIR / "surface_videos"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "surface_interpolation.mp4"
OUTPUT_ALPHA_VIDEO = OUTPUT_DIR / "surface_interpolation_alpha.mp4"

IMG_SIZE = (64, 64)
NUM_FRAMES = 180
FPS = 30

# Select two surface examples as interpolation endpoints.
# These are positions among the surface-only rows, not sample IDs.
START_SURFACE_INDEX = 0
END_SURFACE_INDEX = 500

# If True, the animation goes start -> end -> start smoothly.
LOOP_BACK = True

# Batch size used during inference.
INFERENCE_BATCH_SIZE = 32

# Background used for the RGB video.
# Since C is premultiplied RGB, black is the most direct visualization.
VIDEO_BACKGROUND = np.array([0.0, 0.0, 0.0], dtype=np.float32)


# ============================================================
# HELPERS
# ============================================================

def tensor_or_array_to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def shortest_periodic_delta(a, b, period):
    """
    Shortest signed angular difference from a to b.
    """
    return (b - a + 0.5 * period) % period - 0.5 * period


def interpolate_periodic(a, b, u, period):
    delta = shortest_periodic_delta(a, b, period)
    return (a + u * delta) % period


def smooth_loop_parameter(t):
    """
    0 -> 1 -> 0 with zero velocity at both ends.
    """
    return 0.5 - 0.5 * math.cos(2.0 * math.pi * t)


def row_to_state(dataset, row, sh_cols):
    """
    Extract a raw, interpretable parameter state from one metadata row.

    This mirrors the parameter construction used by the dataset, but keeps
    the individual fields so they can be interpolated smoothly.
    """
    sample_id = int(row["sample_id"])
    shape_info = dataset.shape_meta[sample_id]

    state = {
        "ctrl": {
            c: float(shape_info[c])
            for c in dataset.ctrl_cols
        },
        "sigma": float(shape_info["sigma"]),

        "hue": float(row["hue"]),
        "saturation": float(row["saturation"]),
        "metallic": float(row["metallic"]),
        "roughness": float(row["roughness"]),
        "opacity": float(row["opacity"]),
        "specular": float(row.get("specular", 0.5)),

        "phi": float(row["phi"]),
        "theta": float(row["theta"]),
        "radius": float(row["radius"]),

        "sh": {
            c: float(row[c])
            for c in sh_cols
        },
    }

    return state


def interpolate_states(a, b, u, ctrl_cols, sh_cols):
    """
    Interpolate two raw parameter states.

    Hue and theta use shortest circular interpolation.
    """
    out = {
        "ctrl": {},
        "sh": {},
    }

    for c in ctrl_cols:
        out["ctrl"][c] = (
            (1.0 - u) * a["ctrl"][c]
            + u * b["ctrl"][c]
        )

    out["sigma"] = (1.0 - u) * a["sigma"] + u * b["sigma"]

    # Hue is cyclic in [0, 1].
    out["hue"] = interpolate_periodic(
        a["hue"], b["hue"], u, period=1.0
    )

    out["saturation"] = (
        (1.0 - u) * a["saturation"]
        + u * b["saturation"]
    )
    out["metallic"] = (
        (1.0 - u) * a["metallic"]
        + u * b["metallic"]
    )
    out["roughness"] = (
        (1.0 - u) * a["roughness"]
        + u * b["roughness"]
    )
    out["opacity"] = (
        (1.0 - u) * a["opacity"]
        + u * b["opacity"]
    )
    out["specular"] = (
        (1.0 - u) * a["specular"]
        + u * b["specular"]
    )

    out["phi"] = (
        (1.0 - u) * a["phi"]
        + u * b["phi"]
    )

    # Theta is circular in [0, 2*pi].
    out["theta"] = interpolate_periodic(
        a["theta"], b["theta"], u, period=2.0 * math.pi
    )

    out["radius"] = (
        (1.0 - u) * a["radius"]
        + u * b["radius"]
    )

    for c in sh_cols:
        out["sh"][c] = (
            (1.0 - u) * a["sh"][c]
            + u * b["sh"][c]
        )

    return out


def state_to_normalized_vector(
    state,
    dataset,
    param_mean,
    param_std,
    sh_cols,
):
    """
    Build the exact normalized parameter vector expected by the surface model.

    The final is_volume flag is always 0 because this is the surface model.
    """
    scalars = []

    # Shape parameters.
    for c in dataset.ctrl_cols:
        scalars.append(state["ctrl"][c])
    scalars.append(state["sigma"])

    # Surface material parameters.
    scalars.extend([
        state["hue"],
        state["saturation"],
        state["metallic"],
        state["roughness"],
        state["opacity"],
        state["specular"],
    ])

    # Camera parameters.
    phi = state["phi"]
    theta = state["theta"]

    scalars.extend([
        math.sin(phi),
        math.cos(phi),
        math.sin(theta),
        math.cos(theta),
        state["radius"],
    ])

    # Environment SH parameters.
    for c in sh_cols:
        scalars.append(state["sh"][c])

    # Surface mode indicator.
    scalars.append(0.0)

    raw = np.asarray(scalars, dtype=np.float32)

    if raw.shape[0] != param_mean.shape[0]:
        raise RuntimeError(
            f"Parameter dimension mismatch: constructed {raw.shape[0]}, "
            f"checkpoint expects {param_mean.shape[0]}"
        )

    normalized = (raw - param_mean) / param_std
    return normalized.astype(np.float32)


def make_model_frame(model_output, background):
    """
    Convert model output [4,H,W] = premultiplied RGB + alpha
    into an RGB uint8 video frame.
    """
    output = np.clip(model_output, 0.0, 1.0)

    C = output[:3]       # [3,H,W], premultiplied color
    alpha = output[3:4]  # [1,H,W]

    # Composite premultiplied color over background.
    bg = background.reshape(3, 1, 1)
    composite = C + (1.0 - alpha) * bg

    frame = np.transpose(composite, (1, 2, 0))
    frame = np.clip(frame, 0.0, 1.0)

    return (frame * 255.0 + 0.5).astype(np.uint8)


def make_alpha_frame(model_output):
    """
    Convert alpha channel to an RGB grayscale uint8 frame.
    """
    alpha = np.clip(model_output[3], 0.0, 1.0)
    frame = np.repeat(alpha[..., None], 3, axis=2)
    return (frame * 255.0 + 0.5).astype(np.uint8)


# ============================================================
# MAIN
# ============================================================

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Using device:", device)

    # --------------------------------------------------------
    # Load dataset metadata and normalization information.
    # --------------------------------------------------------
    dataset = PlaneDatasetParamsToPremultRGBA(
        base_dir=BASE_DIR,
        img_meta_csv=IMG_META_CSV,
        vol_meta_csv=VOL_META_CSV,
        renders_dir=RENDERS_DIR,
        alpha_dir=ALPHA_DIR,
        alpha_meta_csv=ALPHA_META_CSV,
        img_size=IMG_SIZE,
        use_sh=True,
        normalize_params=True,
    )

    if "render_mode" not in dataset.df.columns:
        raise RuntimeError("Dataset does not contain render_mode.")

    surface_df = dataset.df[
        dataset.df["render_mode"].astype(str) == "surface"
    ].reset_index(drop=True)

    if len(surface_df) == 0:
        raise RuntimeError("No surface rows found.")

    if START_SURFACE_INDEX >= len(surface_df):
        raise IndexError("START_SURFACE_INDEX is out of range.")

    if END_SURFACE_INDEX >= len(surface_df):
        raise IndexError("END_SURFACE_INDEX is out of range.")

    start_row = surface_df.iloc[START_SURFACE_INDEX]
    end_row = surface_df.iloc[END_SURFACE_INDEX]

    print(
        "Start:",
        START_SURFACE_INDEX,
        "sample_id=",
        int(start_row["sample_id"]),
    )
    print(
        "End:",
        END_SURFACE_INDEX,
        "sample_id=",
        int(end_row["sample_id"]),
    )

    # The SH column ordering must exactly match the training dataset.
    sh_cols = [
        c for c in dataset.df.columns
        if c.startswith("sh_l")
        and c.endswith(("_r", "_g", "_b"))
    ]

    start_state = row_to_state(dataset, start_row, sh_cols)
    end_state = row_to_state(dataset, end_row, sh_cols)

    # --------------------------------------------------------
    # Load checkpoint.
    # --------------------------------------------------------
    print("Loading checkpoint:", CHECKPOINT_PATH)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    checkpoint_state = checkpoint.get("model_state", checkpoint)
    checkpoint_state.pop("_metadata", None)

    checkpoint_latent_dim = int(
        checkpoint.get("latent_dim", dataset.latent_dim)
    )

    if checkpoint_latent_dim != dataset.latent_dim:
        raise RuntimeError(
            f"Checkpoint latent_dim={checkpoint_latent_dim}, "
            f"but dataset latent_dim={dataset.latent_dim}"
        )

    if "param_mean" in checkpoint:
        param_mean = tensor_or_array_to_numpy(
            checkpoint["param_mean"]
        ).astype(np.float32)
    else:
        param_mean = dataset.param_mean.astype(np.float32)

    if "param_std" in checkpoint:
        param_std = tensor_or_array_to_numpy(
            checkpoint["param_std"]
        ).astype(np.float32)
    else:
        param_std = dataset.param_std.astype(np.float32)

    model = FNOPlusResNetSingle(
        latent_dim=checkpoint_latent_dim,
        img_size=IMG_SIZE,
    ).to(device)

    model.load_state_dict(checkpoint_state)
    model.eval()

    print("Checkpoint loaded.")

    # --------------------------------------------------------
    # Prepare output writers.
    # --------------------------------------------------------
    writer = imageio.get_writer(
        str(OUTPUT_VIDEO),
        fps=FPS,
        codec="libx264",
        quality=8,
    )

    alpha_writer = imageio.get_writer(
        str(OUTPUT_ALPHA_VIDEO),
        fps=FPS,
        codec="libx264",
        quality=8,
    )

    try:
        with torch.no_grad():
            for frame_idx in range(NUM_FRAMES):
                t = frame_idx / max(NUM_FRAMES - 1, 1)

                if LOOP_BACK:
                    u = smooth_loop_parameter(t)
                else:
                    # Smooth one-way interpolation.
                    u = 0.5 - 0.5 * math.cos(math.pi * t)

                state = interpolate_states(
                    start_state,
                    end_state,
                    u,
                    dataset.ctrl_cols,
                    sh_cols,
                )

                param_np = state_to_normalized_vector(
                    state=state,
                    dataset=dataset,
                    param_mean=param_mean,
                    param_std=param_std,
                    sh_cols=sh_cols,
                )

                param_tensor = torch.from_numpy(
                    param_np
                ).float().unsqueeze(0).to(device)

                pred = model(param_tensor)
                pred_np = pred[0].detach().cpu().numpy()

                rgb_frame = make_model_frame(
                    pred_np,
                    VIDEO_BACKGROUND,
                )
                alpha_frame = make_alpha_frame(pred_np)

                writer.append_data(rgb_frame)
                alpha_writer.append_data(alpha_frame)

                if frame_idx % 10 == 0:
                    print(
                        f"Rendered frame {frame_idx + 1}/{NUM_FRAMES}"
                    )

    finally:
        writer.close()
        alpha_writer.close()

    print("Saved RGB video:", OUTPUT_VIDEO)
    print("Saved alpha video:", OUTPUT_ALPHA_VIDEO)


if __name__ == "__main__":
    main()