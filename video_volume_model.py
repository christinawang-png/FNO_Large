#!/usr/bin/env python

import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
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

CHECKPOINT_PATH = Path("fno_premult_volume_epoch025.pt")

OUTPUT_DIR = BASE_DIR / "volume_videos"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_RGB_VIDEO = OUTPUT_DIR / "volume_interpolation.mp4"
OUTPUT_ALPHA_VIDEO = OUTPUT_DIR / "volume_interpolation_alpha.mp4"
OUTPUT_SIDE_BY_SIDE_VIDEO = OUTPUT_DIR / "volume_interpolation_rgb_alpha_vertical.mp4"

IMG_SIZE = (64, 64)

NUM_FRAMES = 180
FPS = 30

# These are indices within the volume-only dataset.
START_VOLUME_INDEX = 0
END_VOLUME_INDEX = 1000

# Set these to None to use the sigma values from the endpoint rows.
# Otherwise, explicitly override both endpoint sigmas.
START_SIGMA = 0.02       # example: 0.02
END_SIGMA = 0.7         # example: 0.70

# Optional endpoint opacity overrides.
# These control the volume opacity/density input.
START_OPACITY = 0.8
END_OPACITY = 0.1

# Output background used when converting premultiplied color to RGB.
VIDEO_BACKGROUND = np.array(
    [0.0, 0.0, 0.0],
    dtype=np.float32,
)

# axis=0 means:
#   RGB on top
#   alpha on bottom
#
# axis=1 would place RGB and alpha side by side horizontally.
SIDE_BY_SIDE_AXIS = 0


# ============================================================
# HELPERS
# ============================================================

def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def shortest_periodic_delta(a, b, period):
    return (b - a + 0.5 * period) % period - 0.5 * period


def interpolate_periodic(a, b, u, period):
    delta = shortest_periodic_delta(a, b, period)
    return (a + u * delta) % period


def smooth_interpolation(t):
    """
    Smooth interpolation from 0 to 1 with zero velocity
    at the beginning and end.
    """
    return 0.5 - 0.5 * math.cos(math.pi * t)


def smooth_loop(t):
    """
    Smooth loop:
        start -> end -> start
    """
    return 0.5 - 0.5 * math.cos(2.0 * math.pi * t)


def get_sh_columns(dataset):
    """
    Match the SH column order used by the dataset's parameter builder.
    """
    return [
        c for c in dataset.df.columns
        if c.startswith("sh_l")
        and c.endswith(("_r", "_g", "_b"))
    ]


def row_to_state(dataset, row, sh_cols):
    """
    Extract raw, interpretable parameters from one metadata row.
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

        # Volume model does not physically use these.
        "metallic": 0.0,
        "roughness": 0.0,
        "specular": 0.0,

        "opacity": float(row["opacity"]),

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
    Interpolate two volume parameter states.
    """
    out = {
        "ctrl": {},
        "sh": {},
    }

    # Shape control points.
    for c in ctrl_cols:
        out["ctrl"][c] = (
            (1.0 - u) * a["ctrl"][c]
            + u * b["ctrl"][c]
        )

    # Explicitly interpolate sigma.
    out["sigma"] = (
        (1.0 - u) * a["sigma"]
        + u * b["sigma"]
    )

    # Color.
    out["hue"] = interpolate_periodic(
        a["hue"],
        b["hue"],
        u,
        period=1.0,
    )

    out["saturation"] = (
        (1.0 - u) * a["saturation"]
        + u * b["saturation"]
    )

    # Volume opacity/density control.
    out["opacity"] = (
        (1.0 - u) * a["opacity"]
        + u * b["opacity"]
    )

    # Camera angles.
    out["phi"] = (
        (1.0 - u) * a["phi"]
        + u * b["phi"]
    )

    out["theta"] = interpolate_periodic(
        a["theta"],
        b["theta"],
        u,
        period=2.0 * math.pi,
    )

    out["radius"] = (
        (1.0 - u) * a["radius"]
        + u * b["radius"]
    )

    # Environment SH.
    for c in sh_cols:
        out["sh"][c] = (
            (1.0 - u) * a["sh"][c]
            + u * b["sh"][c]
        )

    # Keep volume-only material values zero.
    out["metallic"] = 0.0
    out["roughness"] = 0.0
    out["specular"] = 0.0

    return out


def state_to_normalized_vector(
    state,
    dataset,
    param_mean,
    param_std,
    sh_cols,
):
    """
    Construct the parameter vector in exactly the same order
    as PlaneDatasetParamsToPremultRGBA.

    This is for a volume model, so:
        metallic = 0
        roughness = 0
        specular = 0
        is_volume = 1
    """
    scalars = []

    # Shape parameters.
    for c in dataset.ctrl_cols:
        scalars.append(state["ctrl"][c])

    scalars.append(state["sigma"])

    # Material parameters.
    scalars.extend([
        state["hue"],
        state["saturation"],
        0.0,                    # metallic
        0.0,                    # roughness
        state["opacity"],
        0.0,                    # specular
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

    # Volume indicator.
    scalars.append(1.0)

    raw = np.asarray(scalars, dtype=np.float32)

    if raw.shape[0] != len(param_mean):
        raise RuntimeError(
            f"Parameter dimension mismatch: "
            f"constructed {raw.shape[0]}, "
            f"expected {len(param_mean)}"
        )

    normalized = (raw - param_mean) / param_std
    return normalized.astype(np.float32)


def make_rgb_frame(model_output, background):
    """
    model_output: [4,H,W]
        channels 0:3 = premultiplied RGB
        channel 3   = alpha
    """
    output = np.clip(model_output, 0.0, 1.0)

    color = output[:3]
    alpha = output[3:4]

    bg = background.reshape(3, 1, 1)

    # Composite premultiplied color over background.
    rgb = color + (1.0 - alpha) * bg
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = np.clip(rgb, 0.0, 1.0)

    return (rgb * 255.0 + 0.5).astype(np.uint8)


def make_alpha_frame(model_output):
    """
    Convert alpha to grayscale RGB for video writing.
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
        raise RuntimeError(
            "Dataset is missing render_mode."
        )

    # Keep only volume rows.
    volume_mask = (
        dataset.df["render_mode"].astype(str) == "volume"
    )
    volume_df = dataset.df[volume_mask].reset_index(drop=True)

    if len(volume_df) == 0:
        raise RuntimeError("No volume rows found.")

    if START_VOLUME_INDEX >= len(volume_df):
        raise IndexError("START_VOLUME_INDEX is out of range.")

    if END_VOLUME_INDEX >= len(volume_df):
        raise IndexError("END_VOLUME_INDEX is out of range.")

    start_row = volume_df.iloc[START_VOLUME_INDEX]
    end_row = volume_df.iloc[END_VOLUME_INDEX]

    print(
        "Start volume index:",
        START_VOLUME_INDEX,
        "sample_id:",
        int(start_row["sample_id"]),
    )
    print(
        "End volume index:",
        END_VOLUME_INDEX,
        "sample_id:",
        int(end_row["sample_id"]),
    )

    # --------------------------------------------------------
    # Load checkpoint.
    # --------------------------------------------------------
    print("Loading checkpoint:", CHECKPOINT_PATH)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    checkpoint_state = checkpoint.get(
        "model_state",
        checkpoint,
    )
    checkpoint_state.pop("_metadata", None)

    checkpoint_mode = checkpoint.get("mode", "unknown")
    if checkpoint_mode != "unknown" and checkpoint_mode != "volume":
        print(
            f"[WARN] Checkpoint mode is {checkpoint_mode!r}, "
            "not 'volume'."
        )

    checkpoint_latent_dim = int(
        checkpoint.get(
            "latent_dim",
            dataset.latent_dim,
        )
    )

    if checkpoint_latent_dim != dataset.latent_dim:
        raise RuntimeError(
            f"Checkpoint latent_dim={checkpoint_latent_dim}, "
            f"but dataset latent_dim={dataset.latent_dim}"
        )

    if "param_mean" in checkpoint:
        param_mean = to_numpy(
            checkpoint["param_mean"]
        ).astype(np.float32)
    else:
        param_mean = dataset.param_mean.astype(np.float32)

    if "param_std" in checkpoint:
        param_std = to_numpy(
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
    # Build endpoint states.
    # --------------------------------------------------------
    sh_cols = get_sh_columns(dataset)

    start_state = row_to_state(
        dataset,
        start_row,
        sh_cols,
    )

    end_state = row_to_state(
        dataset,
        end_row,
        sh_cols,
    )

    # Override endpoint sigmas if requested.
    if START_SIGMA is not None:
        start_state["sigma"] = float(START_SIGMA)

    if END_SIGMA is not None:
        end_state["sigma"] = float(END_SIGMA)

    # Override endpoint opacity if requested.
    if START_OPACITY is not None:
        start_state["opacity"] = float(START_OPACITY)

    if END_OPACITY is not None:
        end_state["opacity"] = float(END_OPACITY)

    print(
        f"Sigma interpolation: "
        f"{start_state['sigma']:.6f} -> "
        f"{end_state['sigma']:.6f}"
    )

    print(
        f"Opacity interpolation: "
        f"{start_state['opacity']:.6f} -> "
        f"{end_state['opacity']:.6f}"
    )

    # --------------------------------------------------------
    # Create video writers.
    # --------------------------------------------------------
    rgb_writer = imageio.get_writer(
        str(OUTPUT_RGB_VIDEO),
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

    side_writer = imageio.get_writer(
        str(OUTPUT_SIDE_BY_SIDE_VIDEO),
        fps=FPS,
        codec="libx264",
        quality=8,
    )

    try:
        with torch.no_grad():
            for frame_idx in range(NUM_FRAMES):
                t = frame_idx / max(NUM_FRAMES - 1, 1)

                # Use this for start -> end.
                u = smooth_interpolation(t)

                # For a looping animation, replace the previous line with:
                # u = smooth_loop(t)

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

                prediction = model(param_tensor)
                prediction_np = (
                    prediction[0]
                    .detach()
                    .cpu()
                    .numpy()
                )

                rgb_frame = make_rgb_frame(
                    prediction_np,
                    VIDEO_BACKGROUND,
                )

                alpha_frame = make_alpha_frame(
                    prediction_np,
                )

                if SIDE_BY_SIDE_AXIS == 0:
                    combined_frame = np.concatenate(
                        [rgb_frame, alpha_frame],
                        axis=0,
                    )
                else:
                    combined_frame = np.concatenate(
                        [rgb_frame, alpha_frame],
                        axis=1,
                    )

                rgb_writer.append_data(rgb_frame)
                alpha_writer.append_data(alpha_frame)
                side_writer.append_data(combined_frame)

                if frame_idx % 10 == 0:
                    print(
                        f"Rendered frame "
                        f"{frame_idx + 1}/{NUM_FRAMES}"
                    )

    finally:
        rgb_writer.close()
        alpha_writer.close()
        side_writer.close()

    print("Saved RGB video:", OUTPUT_RGB_VIDEO)
    print("Saved alpha video:", OUTPUT_ALPHA_VIDEO)
    print(
        "Saved RGB/alpha video:",
        OUTPUT_SIDE_BY_SIDE_VIDEO,
    )


if __name__ == "__main__":
    main()