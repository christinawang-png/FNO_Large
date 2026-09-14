#!/usr/bin/env python

import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from train_premult_single_mode import FNOPlusResNetSingle


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_PATH = Path(
    "fno_premult_surface_epoch128_color.pt"
)

OUTPUT_DIR = Path("surface_videos_rgb")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_RGB_VIDEO = (
    OUTPUT_DIR / "surface_rgb_interpolation.mp4"
)

OUTPUT_ALPHA_VIDEO = (
    OUTPUT_DIR / "surface_alpha_interpolation.mp4"
)

OUTPUT_COMBINED_VIDEO = (
    OUTPUT_DIR / "surface_rgb_alpha_vertical.mp4"
)

IMG_SIZE = (64, 64)

NUM_FRAMES = 180
FPS = 30

# True: start -> end -> start.
# False: start -> end.
LOOP_BACK = True

RANDOM_SEED = 42

# FNO was trained with this canonical radius.
FNO_RADIUS = 2.2

# Composite predicted premultiplied output over this background.
VIDEO_BACKGROUND = np.array(
    [1.0, 1.0, 1.0],
    dtype=np.float32,
)

# ============================================================
# RANDOM PARAMETER RANGES
# Must match the new RGB dataset/model.
# ============================================================

CTRL_LEVELS = np.array(
    [0.1, 0.3, 0.5, 0.7, 0.9],
    dtype=np.float32,
)

SIGMA_VALUES = np.array(
    [0.02, 0.08, 0.20, 0.50, 0.70],
    dtype=np.float32,
)

COLOR_LOW = 0.02
COLOR_HIGH = 1.00

OPACITY_LOW = 0.10
OPACITY_HIGH = 1.00

ROUGHNESS_LOW = 0.10
ROUGHNESS_HIGH = 0.90

NUM_GLOBAL_ENVS = 128
SH_ORDER = 2


# ============================================================
# OPTIONAL MANUAL ENDPOINT OVERRIDES
#
# Set any item to None to use random initialization.
# RGB values should remain roughly in [0.02, 1.0].
# Metallic is trained only at 0 or 1. Keep it fixed across
# the video unless you intentionally want extrapolation.
# ============================================================

START_OVERRIDE = {
    "ctrl": [0.1, 0.2, 0.2, 0.1],   # low/slightly bent plane
    "sigma": 0.08,
    "base_rgb": [0.9, 0.15, 0.10],  # red
    "metallic": 1.0,
    "roughness": 0.25,
    "opacity": 0.98,
    "phi": math.radians(120),
    "theta": math.radians(220),
    "env_id": 10,
}

END_OVERRIDE = {
    "ctrl": [0.55, 0.02, 0.01, 0.02],  # mean = 0.15
    "sigma": 0.08,
    "base_rgb": [0.10, 0.35, 0.95],   # blue
    "metallic": 0.0,
    "roughness": 0.25,
    "opacity": 0.98,
    "phi": math.radians(120),
    "theta": math.radians(220),
    "env_id": 10,
}

# ============================================================
# ENVIRONMENT SH
# Must match Blender render-generation code.
# ============================================================

def sh_lm_list(order):
    pairs = []

    for l in range(order + 1):
        for m in range(-l, l + 1):
            pairs.append((l, m))

    return pairs


def sh_for_global_env(env_id, order=2):
    """
    Return SH coefficients with shape [9,3] for SH order 2.
    Matches the previous Blender generation function.
    """
    pairs = sh_lm_list(order)

    coeffs = np.zeros(
        (len(pairs), 3),
        dtype=np.float32,
    )

    u = env_id / max(
        1.0,
        float(NUM_GLOBAL_ENVS - 1),
    )

    t = 2.0 * math.pi * u

    r = 0.5 + 0.4 * math.sin(t)
    g = 0.5 + 0.4 * math.sin(
        t + 2.0 * math.pi / 3.0
    )
    b = 0.5 + 0.4 * math.sin(
        t + 4.0 * math.pi / 3.0
    )

    rgb = np.array(
        [r, g, b],
        dtype=np.float32,
    )

    gray = np.full(
        3,
        rgb.mean(),
        dtype=np.float32,
    )

    if u < 1.0 / 3.0:
        color_mix = 0.1
    elif u < 2.0 / 3.0:
        color_mix = 0.5
    else:
        color_mix = 1.0

    rgb_scale = (
        (1.0 - color_mix) * gray
        + color_mix * rgb
    )

    # l=0, m=0.
    coeffs[0, :] = rgb_scale * 0.4

    for idx, (l, m) in enumerate(pairs):
        if l != 1:
            continue

        if m == -1:
            coeffs[idx, :] = rgb_scale * (
                0.2 * math.sin(2.0 * math.pi * u)
            )

        elif m == 0:
            coeffs[idx, :] = rgb_scale * (
                0.2 * math.cos(2.0 * math.pi * u)
            )

        elif m == 1:
            coeffs[idx, :] = rgb_scale * (
                0.2 * math.sin(
                    2.0 * math.pi * u + 1.0
                )
            )

    for idx, (l, m) in enumerate(pairs):
        if l == 2 and m == 0:
            coeffs[idx, :] += rgb_scale * (
                0.05 * math.cos(4.0 * math.pi * u)
            )

    return coeffs


def build_sh_bank():
    """
    Returns:
        sh_bank: [128,27]
    """
    rows = []

    for env_id in range(NUM_GLOBAL_ENVS):
        coeffs = sh_for_global_env(
            env_id,
            order=SH_ORDER,
        )  # [9,3]

        rows.append(
            coeffs.reshape(-1)
        )

    return np.stack(
        rows,
        axis=0,
    ).astype(np.float32)


# ============================================================
# STATE GENERATION / INTERPOLATION
# ============================================================

def shortest_periodic_delta(a, b, period):
    return (
        (b - a + 0.5 * period) % period
        - 0.5 * period
    )


def interpolate_periodic(a, b, t, period):
    return (
        a
        + t * shortest_periodic_delta(a, b, period)
    ) % period


def smooth_parameter(t):
    """
    Smooth 0 -> 1 interpolation.
    """
    return 0.5 - 0.5 * math.cos(math.pi * t)


def smooth_loop_parameter(t):
    """
    Smooth 0 -> 1 -> 0 loop.
    """
    return 0.5 - 0.5 * math.cos(
        2.0 * math.pi * t
    )


def random_state(rng, sh_bank):
    """
    Random valid surface-model state.
    """
    env_id = int(
        rng.integers(
            0,
            NUM_GLOBAL_ENVS,
        )
    )

    return {
        "ctrl": rng.choice(
            CTRL_LEVELS,
            size=4,
            replace=True,
        ).astype(np.float32),

        "sigma": float(
            rng.choice(SIGMA_VALUES)
        ),

        "base_rgb": rng.uniform(
            COLOR_LOW,
            COLOR_HIGH,
            size=3,
        ).astype(np.float32),

        # Surface model only saw 0 / 1.
        "metallic": float(
            rng.choice([0.0, 1.0])
        ),

        "roughness": float(
            rng.uniform(
                ROUGHNESS_LOW,
                ROUGHNESS_HIGH,
            )
        ),

        "opacity": float(
            rng.uniform(
                OPACITY_LOW,
                OPACITY_HIGH,
            )
        ),

        "specular": 0.5,

        "phi": float(
            rng.uniform(0.0, math.pi)
        ),

        "theta": float(
            rng.uniform(
                0.0,
                2.0 * math.pi,
            )
        ),

        "radius": FNO_RADIUS,

        "env_id": env_id,
        "sh": sh_bank[env_id].copy(),
    }


def apply_overrides(state, override, sh_bank):
    """
    Apply non-None manual settings to one random state.
    """
    state = dict(state)

    if override["ctrl"] is not None:
        ctrl = np.asarray(
            override["ctrl"],
            dtype=np.float32,
        )

        if ctrl.shape != (4,):
            raise ValueError(
                "ctrl override must have exactly 4 values."
            )

        state["ctrl"] = np.clip(
            ctrl,
            COLOR_LOW,
            COLOR_HIGH,
        )

    if override["sigma"] is not None:
        state["sigma"] = float(
            np.clip(
                override["sigma"],
                SIGMA_VALUES.min(),
                SIGMA_VALUES.max(),
            )
        )

    if override["base_rgb"] is not None:
        rgb = np.asarray(
            override["base_rgb"],
            dtype=np.float32,
        )

        if rgb.shape != (3,):
            raise ValueError(
                "base_rgb override must have exactly 3 values."
            )

        state["base_rgb"] = np.clip(
            rgb,
            COLOR_LOW,
            COLOR_HIGH,
        )

    if override["metallic"] is not None:
        metallic = float(override["metallic"])

        if metallic not in (0.0, 1.0):
            raise ValueError(
                "metallic override should be exactly 0.0 or 1.0."
            )

        state["metallic"] = metallic

    if override["roughness"] is not None:
        state["roughness"] = float(
            np.clip(
                override["roughness"],
                ROUGHNESS_LOW,
                ROUGHNESS_HIGH,
            )
        )

    if override["opacity"] is not None:
        state["opacity"] = float(
            np.clip(
                override["opacity"],
                OPACITY_LOW,
                OPACITY_HIGH,
            )
        )

    if override["phi"] is not None:
        state["phi"] = float(
            np.clip(
                override["phi"],
                0.0,
                math.pi,
            )
        )

    if override["theta"] is not None:
        state["theta"] = float(
            override["theta"]
        ) % (2.0 * math.pi)

    if override["env_id"] is not None:
        env_id = int(
            np.clip(
                override["env_id"],
                0,
                NUM_GLOBAL_ENVS - 1,
            )
        )

        state["env_id"] = env_id
        state["sh"] = sh_bank[env_id].copy()

    return state


def interpolate_states(start, end, t):
    """
    Surface state interpolation.

    RGB is interpolated directly.
    Metallic is selected discretely because the surface FNO saw
    only 0 or 1 during training.
    """
    metallic = (
        start["metallic"]
        if t < 0.5
        else end["metallic"]
    )

    return {
        "ctrl": (
            (1.0 - t) * start["ctrl"]
            + t * end["ctrl"]
        ).astype(np.float32),

        "sigma": float(
            (1.0 - t) * start["sigma"]
            + t * end["sigma"]
        ),

        "base_rgb": (
            (1.0 - t) * start["base_rgb"]
            + t * end["base_rgb"]
        ).astype(np.float32),

        "metallic": float(metallic),

        "roughness": float(
            (1.0 - t) * start["roughness"]
            + t * end["roughness"]
        ),

        "opacity": float(
            (1.0 - t) * start["opacity"]
            + t * end["opacity"]
        ),

        "specular": 0.5,

        "phi": float(
            (1.0 - t) * start["phi"]
            + t * end["phi"]
        ),

        "theta": float(
            interpolate_periodic(
                start["theta"],
                end["theta"],
                t,
                2.0 * math.pi,
            )
        ),

        "radius": FNO_RADIUS,

        # Smooth SH interpolation is slightly off the discrete
        # environment bank manifold, but useful for videos.
        "sh": (
            (1.0 - t) * start["sh"]
            + t * end["sh"]
        ).astype(np.float32),
    }


# ============================================================
# MODEL INPUT
# ============================================================

def build_normalized_surface_vector(
    state,
    param_mean,
    param_std,
):
    """
    Build the RGB-conditioned, 45-dimensional surface input.

    Exact feature order:

        ctrl[0:4]
        sigma
        base_r, base_g, base_b
        metallic
        roughness
        opacity
        specular
        sin(phi), cos(phi)
        sin(theta), cos(theta)
        radius
        27 SH values
        is_volume = 0
    """
    ctrl = np.clip(
        state["ctrl"],
        COLOR_LOW,
        COLOR_HIGH,
    )

    rgb = np.clip(
        state["base_rgb"],
        COLOR_LOW,
        COLOR_HIGH,
    )

    raw = np.concatenate(
        [
            ctrl,
            np.array(
                [
                    state["sigma"],
                    rgb[0],
                    rgb[1],
                    rgb[2],
                    state["metallic"],
                    state["roughness"],
                    state["opacity"],
                    state["specular"],
                    math.sin(state["phi"]),
                    math.cos(state["phi"]),
                    math.sin(state["theta"]),
                    math.cos(state["theta"]),
                    state["radius"],
                ],
                dtype=np.float32,
            ),
            state["sh"].astype(np.float32),
            np.array([0.0], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)

    if raw.shape[0] != 45:
        raise RuntimeError(
            f"Expected 45 surface parameters, got {raw.shape[0]}"
        )

    if raw.shape[0] != param_mean.shape[0]:
        raise RuntimeError(
            f"Parameter dimension mismatch: "
            f"constructed={raw.shape[0]}, "
            f"checkpoint expects={param_mean.shape[0]}"
        )

    return (
        (raw - param_mean) / param_std
    ).astype(np.float32)


# ============================================================
# VIDEO FRAMES
# ============================================================

def make_rgb_frame(model_output):
    """
    model_output:
        [4,H,W] = premultiplied RGB + alpha.
    """
    output = np.clip(
        model_output,
        0.0,
        1.0,
    )

    premult_color = output[:3]
    alpha = output[3:4]

    background = VIDEO_BACKGROUND.reshape(
        3,
        1,
        1,
    )

    visible = premult_color + (
        1.0 - alpha
    ) * background

    frame = np.transpose(
        visible,
        (1, 2, 0),
    )

    return (
        np.clip(frame, 0.0, 1.0)
        * 255.0
        + 0.5
    ).astype(np.uint8)


def make_alpha_frame(model_output):
    alpha = np.clip(
        model_output[3],
        0.0,
        1.0,
    )

    alpha_rgb = np.repeat(
        alpha[..., None],
        3,
        axis=2,
    )

    return (
        alpha_rgb * 255.0 + 0.5
    ).astype(np.uint8)


# ============================================================
# MAIN
# ============================================================

def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    sh_bank = build_sh_bank()

    # --------------------------------------------------------
    # Sample endpoint states directly.
    # --------------------------------------------------------
    start_state = apply_overrides(
        random_state(rng, sh_bank),
        START_OVERRIDE,
        sh_bank,
    )

    end_state = apply_overrides(
        random_state(rng, sh_bank),
        END_OVERRIDE,
        sh_bank,
    )

    # To stay strictly in distribution for metallic, use one
    # material type across the video unless manually overridden.
    if (
        START_OVERRIDE["metallic"] is None
        and END_OVERRIDE["metallic"] is None
    ):
        end_state["metallic"] = start_state["metallic"]

    print("Start state:")
    print(
        "  ctrl:", start_state["ctrl"],
        "sigma:", start_state["sigma"],
        "rgb:", start_state["base_rgb"],
        "metallic:", start_state["metallic"],
        "roughness:", start_state["roughness"],
        "opacity:", start_state["opacity"],
        "env:", start_state["env_id"],
    )

    print("End state:")
    print(
        "  ctrl:", end_state["ctrl"],
        "sigma:", end_state["sigma"],
        "rgb:", end_state["base_rgb"],
        "metallic:", end_state["metallic"],
        "roughness:", end_state["roughness"],
        "opacity:", end_state["opacity"],
        "env:", end_state["env_id"],
    )

    # --------------------------------------------------------
    # Load RGB-conditioned checkpoint.
    # --------------------------------------------------------
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    state_dict = dict(
        checkpoint["model_state"]
    )

    state_dict.pop("_metadata", None)

    latent_dim = int(
        checkpoint["latent_dim"]
    )

    if latent_dim != 45:
        raise RuntimeError(
            f"Expected RGB-conditioned latent_dim=45, "
            f"got {latent_dim}"
        )

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

    model.load_state_dict(state_dict)
    model.eval()

    print("Loaded checkpoint:", CHECKPOINT_PATH)

    # --------------------------------------------------------
    # Video writers.
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

    combined_writer = imageio.get_writer(
        str(OUTPUT_COMBINED_VIDEO),
        fps=FPS,
        codec="libx264",
        quality=8,
    )

    try:
        with torch.no_grad():
            for frame_index in range(NUM_FRAMES):
                t = frame_index / max(
                    NUM_FRAMES - 1,
                    1,
                )

                if LOOP_BACK:
                    interpolation_t = smooth_loop_parameter(t)
                else:
                    interpolation_t = smooth_parameter(t)

                state = interpolate_states(
                    start_state,
                    end_state,
                    interpolation_t,
                )

                param_np = build_normalized_surface_vector(
                    state=state,
                    param_mean=param_mean,
                    param_std=param_std,
                )

                param_tensor = torch.from_numpy(
                    param_np
                ).unsqueeze(0).to(
                    device=device,
                    dtype=torch.float32,
                )

                prediction = model(param_tensor)

                prediction_np = (
                    prediction[0]
                    .detach()
                    .cpu()
                    .numpy()
                )

                rgb_frame = make_rgb_frame(
                    prediction_np
                )

                alpha_frame = make_alpha_frame(
                    prediction_np
                )

                # RGB on top; alpha below.
                combined_frame = np.concatenate(
                    [rgb_frame, alpha_frame],
                    axis=0,
                )

                rgb_writer.append_data(rgb_frame)
                alpha_writer.append_data(alpha_frame)
                combined_writer.append_data(combined_frame)

                if frame_index % 10 == 0:
                    print(
                        f"Rendered frame "
                        f"{frame_index + 1}/{NUM_FRAMES}"
                    )

    finally:
        rgb_writer.close()
        alpha_writer.close()
        combined_writer.close()

    print("Saved RGB video:", OUTPUT_RGB_VIDEO)
    print("Saved alpha video:", OUTPUT_ALPHA_VIDEO)
    print("Saved combined video:", OUTPUT_COMBINED_VIDEO)


if __name__ == "__main__":
    main()