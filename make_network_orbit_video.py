#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from implicit_dataset import (
    build_frame_table,
    build_raw_feature_matrix,
)
from implicit_model import ImplicitFNOImageModel


# ============================================================
# ARGUMENTS
# ============================================================

def parse_rgb(text: str):
    """
    Parse an RGB string such as:

        "0.8,0.2,0.1"
    """
    values = [float(value.strip()) for value in text.split(",")]

    if len(values) != 3:
        raise argparse.ArgumentTypeError(
            "RGB must contain exactly three comma-separated values, "
            'for example: "0.8,0.2,0.1"'
        )

    if any(value < 0.0 or value > 1.0 for value in values):
        raise argparse.ArgumentTypeError(
            "RGB values must be in the interval [0, 1]."
        )

    return tuple(values)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a camera-orbit MP4 using an implicit B-spline "
            "network checkpoint, without Blender."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help=(
            "Path to implicit_bspline_dataset containing volumes/, "
            "renders_balanced/, and hard_alpha_balanced/."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint to use, typically best.pt or latest.pt.",
    )

    parser.add_argument(
        "--mode",
        choices=["surface", "volume"],
        required=True,
        help="Must match the model checkpoint.",
    )

    parser.add_argument(
        "--geometry-id",
        type=int,
        default=None,
        help=(
            "A geometry_id from the training split. If omitted, one "
            "training geometry is selected randomly."
        ),
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help=(
            "Optional sigma override. It is useful for volume mode. "
            "For surface mode sigma does not alter the mesh, though the "
            "current trained surface network may still have seen it."
        ),
    )

    parser.add_argument(
        "--color",
        type=parse_rgb,
        default=(0.65, 0.35, 0.10),
        help='Base RGB color, e.g. "0.8,0.2,0.1".',
    )

    parser.add_argument(
        "--background",
        type=parse_rgb,
        default=(0.02, 0.02, 0.02),
        help='Background RGB color, e.g. "0.02,0.02,0.02".',
    )

    parser.add_argument(
        "--opacity",
        type=float,
        default=0.95,
        help="Target opacity conditioning value in [0, 1].",
    )

    parser.add_argument(
        "--metallic",
        type=float,
        default=0.0,
        help="Surface metallic value in [0, 1]. Ignored in volume mode.",
    )

    parser.add_argument(
        "--roughness",
        type=float,
        default=0.35,
        help="Surface roughness value in [0, 1]. Ignored in volume mode.",
    )

    parser.add_argument(
        "--specular",
        type=float,
        default=0.5,
        help="Surface specular value in [0, 1]. Ignored in volume mode.",
    )

    parser.add_argument(
        "--env-id",
        type=int,
        default=None,
        help=(
            "Optional environment ID. If omitted, reuse lighting from "
            "the selected training example."
        ),
    )

    parser.add_argument(
        "--elevation-deg",
        type=float,
        default=55.0,
        help="Camera elevation phi in degrees. 90 is a sideways/equatorial view.",
    )

    parser.add_argument(
        "--theta-start-deg",
        type=float,
        default=0.0,
        help="Starting azimuth theta in degrees.",
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        default=120,
        help="Number of frames in the full orbit.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frames per second in output MP4.",
    )

    parser.add_argument(
        "--scale",
        type=int,
        default=16,
        help=(
            "Integer upsampling factor for video display. "
            "32x32 prediction with scale=16 becomes 512x512 video."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./network_orbit.mp4"),
        help="Output MP4 filename.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed used only when geometry_id is omitted.",
    )

    return parser.parse_args()


# ============================================================
# CHECKPOINT / SPLIT UTILITIES
# ============================================================

def load_model_state(model, checkpoint):
    """
    Load model weights while ignoring legacy checkpoint metadata.
    """
    state = checkpoint.get("model_state", checkpoint).copy()
    state.pop("_metadata", None)
    model.load_state_dict(state, strict=True)


def checkpoint_config_value(checkpoint, name, default):
    config = checkpoint.get("config", {})

    if not isinstance(config, dict):
        return default

    return config.get(name, default)


def load_training_geometry_ids(split_json_path: Path):
    """
    Load the exact geometry-level split created during training.
    """
    if not split_json_path.is_file():
        raise FileNotFoundError(
            f"Could not find split file:\n  {split_json_path}\n\n"
            "Expected dataset_split_and_features.json next to the checkpoint."
        )

    with open(split_json_path, "r") as f:
        split_info = json.load(f)

    if "train_geometry_ids" not in split_info:
        raise KeyError(
            "dataset_split_and_features.json has no train_geometry_ids field."
        )

    return np.asarray(
        split_info["train_geometry_ids"],
        dtype=np.int64,
    )


# ============================================================
# FEATURE OVERRIDES
# ============================================================

def set_feature(feature_vector, feature_index, name, value):
    """
    Assign a feature if it exists in the trained checkpoint feature layout.

    This supports surface checkpoints that may omit sigma in a future
    cleaned version.
    """
    if name in feature_index:
        feature_vector[feature_index[name]] = float(value)


def set_sh_features_from_environment_row(
    feature_vector,
    feature_index,
    frame_table,
    sh_columns,
    env_id,
):
    """
    Copy SH coefficients for a requested environment ID from any metadata row.

    The renderer's SH values depend on env_id, not geometry, so any row
    with the requested environment is valid.
    """
    matching = frame_table[
        frame_table["env_id"].astype(int) == int(env_id)
    ]

    if matching.empty:
        available = sorted(
            frame_table["env_id"].astype(int).unique().tolist()
        )

        raise ValueError(
            f"env_id={env_id} is absent from loaded metadata. "
            f"Available IDs include: {available[:20]} ..."
        )

    env_row = matching.iloc[0]

    for column in sh_columns:
        set_feature(
            feature_vector,
            feature_index,
            column,
            float(env_row[column]),
        )


def make_frame_feature_vector(
    base_raw_features,
    feature_names,
    frame_table,
    reference_row,
    sh_columns,
    args,
    theta_radians,
):
    """
    Start from a valid training frame's raw conditioning vector, then override:

      - camera pose;
      - base color;
      - opacity;
      - material settings;
      - optional sigma;
      - optional environment SH coefficients.
    """
    vector = np.asarray(
        base_raw_features,
        dtype=np.float32,
    ).copy()

    feature_index = {
        name: index
        for index, name in enumerate(feature_names)
    }

    phi_radians = math.radians(args.elevation_deg)

    # Base color.
    set_feature(
        vector,
        feature_index,
        "base_color_r",
        args.color[0],
    )
    set_feature(
        vector,
        feature_index,
        "base_color_g",
        args.color[1],
    )
    set_feature(
        vector,
        feature_index,
        "base_color_b",
        args.color[2],
    )

    # Appearance / opacity.
    set_feature(vector, feature_index, "opacity", args.opacity)

    if args.mode == "surface":
        set_feature(
            vector,
            feature_index,
            "metallic",
            args.metallic,
        )
        set_feature(
            vector,
            feature_index,
            "roughness",
            args.roughness,
        )
        set_feature(
            vector,
            feature_index,
            "specular",
            args.specular,
        )
    else:
        # Surface-only values were zero during volume training.
        set_feature(vector, feature_index, "metallic", 0.0)
        set_feature(vector, feature_index, "roughness", 0.0)
        set_feature(vector, feature_index, "specular", 0.0)

    # Sigma controls volume thickness. Optional override is mostly a no-op
    # conceptually for a surface mesh, but is allowed for checkpoint parity.
    if args.sigma is not None:
        set_feature(vector, feature_index, "sigma", args.sigma)

    # Camera direction encoding.
    set_feature(
        vector,
        feature_index,
        "sin_phi",
        math.sin(phi_radians),
    )
    set_feature(
        vector,
        feature_index,
        "cos_phi",
        math.cos(phi_radians),
    )
    set_feature(
        vector,
        feature_index,
        "sin_theta",
        math.sin(theta_radians),
    )
    set_feature(
        vector,
        feature_index,
        "cos_theta",
        math.cos(theta_radians),
    )

    # Optional environment-lighting override.
    if args.env_id is not None and sh_columns:
        set_sh_features_from_environment_row(
            feature_vector=vector,
            feature_index=feature_index,
            frame_table=frame_table,
            sh_columns=sh_columns,
            env_id=args.env_id,
        )

    return vector


# ============================================================
# IMAGE / VIDEO UTILITIES
# ============================================================

def predicted_rgba_to_frame(
    predicted_rgba,
    background_rgb,
    scale,
):
    """
    Convert network output [4,H,W] premultiplied RGBA into a uint8 RGB frame.

    Predicted channels are:

        [alpha*R, alpha*G, alpha*B, alpha]

    Composite over requested background:

        output = premult_rgb + (1 - alpha) * background
    """
    predicted_rgba = np.asarray(predicted_rgba, dtype=np.float32)

    premult_rgb = np.clip(predicted_rgba[:3], 0.0, 1.0)
    alpha = np.clip(predicted_rgba[3:4], 0.0, 1.0)

    background = np.asarray(
        background_rgb,
        dtype=np.float32,
    ).reshape(3, 1, 1)

    composite = premult_rgb + (1.0 - alpha) * background
    composite = np.clip(composite, 0.0, 1.0)

    # CHW -> HWC.
    frame = np.transpose(composite, (1, 2, 0))

    # Nearest-neighbor display enlargement preserves the actual 32x32
    # output-grid appearance. Use scale=16 for 512x512 video.
    if scale > 1:
        frame = np.repeat(frame, scale, axis=0)
        frame = np.repeat(frame, scale, axis=1)

    return (frame * 255.0).round().astype(np.uint8)


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    if not (0.0 <= args.opacity <= 1.0):
        raise ValueError("--opacity must be in [0, 1].")

    for name, value in [
        ("metallic", args.metallic),
        ("roughness", args.roughness),
        ("specular", args.specular),
    ]:
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"--{name} must be in [0, 1].")

    if args.num_frames < 2:
        raise ValueError("--num-frames must be at least 2.")

    if args.scale < 1:
        raise ValueError("--scale must be at least 1.")

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {args.checkpoint}"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    checkpoint_mode = checkpoint_config_value(
        checkpoint,
        "mode",
        None,
    )

    if checkpoint_mode is not None and checkpoint_mode != args.mode:
        raise RuntimeError(
            f"Checkpoint was trained for mode='{checkpoint_mode}', "
            f"but --mode is '{args.mode}'."
        )

    dataset_root = args.dataset_root.resolve()

    volumes_dir = dataset_root / "volumes"
    renders_dir = dataset_root / "renders_balanced"
    alpha_dir = dataset_root / "hard_alpha_balanced"

    image_metadata_csv = (
        renders_dir / "metadata_images_all_sharded.csv"
    )
    alpha_metadata_csv = alpha_dir / "metadata_alpha_all.csv"
    volume_metadata_csv = volumes_dir / "metadata_volumes.csv"

    height = int(
        checkpoint_config_value(checkpoint, "height", 32)
    )
    width = int(
        checkpoint_config_value(checkpoint, "width", 32)
    )
    fno_modes = int(
        checkpoint_config_value(checkpoint, "fno_modes", 16)
    )
    use_sh = not bool(
        checkpoint_config_value(checkpoint, "no_sh", False)
    )

    print(f"Checkpoint image resolution: {width}x{height}")
    print(f"Checkpoint FNO modes: {fno_modes}")
    print(f"Checkpoint uses SH: {use_sh}")

    # --------------------------------------------------------
    # Rebuild feature table exactly as in training.
    # --------------------------------------------------------

    frame_table, ctrl_columns, sh_columns = build_frame_table(
        dataset_root=dataset_root,
        renders_dir=renders_dir,
        alpha_dir=alpha_dir,
        image_metadata_csv=image_metadata_csv,
        alpha_metadata_csv=alpha_metadata_csv,
        volume_metadata_csv=volume_metadata_csv,
        mode=args.mode,
        use_sh=use_sh,
    )

    raw_features, feature_names = build_raw_feature_matrix(
        df=frame_table,
        ctrl_columns=ctrl_columns,
        sh_columns=sh_columns,
        use_sh=use_sh,
    )

    checkpoint_features = checkpoint.get("feature_names")

    if checkpoint_features is not None and checkpoint_features != feature_names:
        raise RuntimeError(
            "Feature layout mismatch between checkpoint and current dataset.\n"
            f"Checkpoint features:\n{checkpoint_features}\n\n"
            f"Current features:\n{feature_names}"
        )

    param_mean = checkpoint.get("param_mean")
    param_std = checkpoint.get("param_std")

    if param_mean is None or param_std is None:
        raise RuntimeError(
            "Checkpoint is missing param_mean/param_std. "
            "Use a checkpoint saved by train_implicit.py."
        )

    param_mean = np.asarray(param_mean, dtype=np.float32)
    param_std = np.asarray(param_std, dtype=np.float32)

    latent_dim = raw_features.shape[1]

    if latent_dim != len(param_mean):
        raise RuntimeError(
            f"Data has latent_dim={latent_dim}, but checkpoint normalization "
            f"has {len(param_mean)} dimensions."
        )

    # --------------------------------------------------------
    # Build and load network.
    # --------------------------------------------------------

    model = ImplicitFNOImageModel(
        latent_dim=latent_dim,
        image_height=height,
        image_width=width,
        fno_modes=fno_modes,
    ).to(device)

    load_model_state(model, checkpoint)
    model.eval()

    # --------------------------------------------------------
    # Select a geometry from the exact training split.
    # --------------------------------------------------------

    split_json_path = (
        args.checkpoint.parent / "dataset_split_and_features.json"
    )

    train_geometry_ids = load_training_geometry_ids(
        split_json_path
    )

    available_training_geometry_ids = np.sort(
        np.intersect1d(
            train_geometry_ids,
            frame_table["geometry_id"].astype(np.int64).unique(),
        )
    )

    if len(available_training_geometry_ids) == 0:
        raise RuntimeError(
            "No geometry IDs from the saved training split appear in the "
            "currently loaded metadata."
        )

    if args.geometry_id is None:
        rng = np.random.default_rng(args.seed)
        geometry_id = int(
            rng.choice(available_training_geometry_ids)
        )
    else:
        geometry_id = int(args.geometry_id)

        if geometry_id not in set(
            available_training_geometry_ids.tolist()
        ):
            raise ValueError(
                f"geometry_id={geometry_id} is not in the training split. "
                "Use a geometry ID listed in dataset_split_and_features.json, "
                "or omit --geometry-id for a random training geometry."
            )

    geometry_rows = frame_table[
        frame_table["geometry_id"].astype(int) == geometry_id
    ].copy()

    if geometry_rows.empty:
        raise RuntimeError(
            f"No frames found for geometry_id={geometry_id}."
        )

    # Choose a reference frame. It provides the selected shape's controls,
    # sigma, and default lighting if --env-id is omitted.
    #
    # For volume, sigma can be chosen explicitly. Otherwise use first row.
    if args.sigma is not None:
        sigma_matches = geometry_rows[
            np.isclose(
                geometry_rows["sigma"].astype(float),
                float(args.sigma),
                atol=1e-6,
            )
        ]

        if not sigma_matches.empty:
            reference_row = sigma_matches.iloc[0]
        else:
            print(
                f"[WARN] geometry_id={geometry_id} does not have "
                f"stored sigma={args.sigma:.6f}; using its first row "
                "and overriding the sigma feature anyway."
            )
            reference_row = geometry_rows.iloc[0]
    else:
        reference_row = geometry_rows.iloc[0]

    reference_row_index = int(reference_row.name)
    base_raw_features = raw_features[reference_row_index]

    print("=" * 72)
    print("Network orbit-video configuration")
    print(f"Mode:                {args.mode}")
    print(f"Geometry ID:         {geometry_id}")
    print(f"Reference sample ID: {int(reference_row['sample_id'])}")
    print(f"Reference sigma:     {float(reference_row['sigma']):.4f}")
    print(f"Sigma override:      {args.sigma}")
    print(f"Color override:      {args.color}")
    print(f"Opacity override:    {args.opacity:.3f}")
    print(f"Environment override:{args.env_id}")
    print(f"Elevation phi:       {args.elevation_deg:.1f} degrees")
    print(f"Frames / FPS:        {args.num_frames} / {args.fps}")
    print(f"Output:              {args.output}")
    print("=" * 72)

    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Full closed orbit: endpoint=False avoids duplicating frame 0.
    theta_values = np.linspace(
        math.radians(args.theta_start_deg),
        math.radians(args.theta_start_deg) + 2.0 * math.pi,
        num=args.num_frames,
        endpoint=False,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Predict frames and encode MP4.
    # --------------------------------------------------------

    writer = imageio.get_writer(
        str(args.output),
        fps=args.fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )

    try:
        with torch.no_grad():
            for frame_index, theta in enumerate(theta_values):
                raw_vector = make_frame_feature_vector(
                    base_raw_features=base_raw_features,
                    feature_names=feature_names,
                    frame_table=frame_table,
                    reference_row=reference_row,
                    sh_columns=sh_columns,
                    args=args,
                    theta_radians=float(theta),
                )

                normalized_vector = (
                    raw_vector - param_mean
                ) / param_std

                parameter_tensor = torch.from_numpy(
                    normalized_vector.astype(np.float32)
                ).unsqueeze(0).to(device)

                predicted_rgba = model(parameter_tensor)[0]
                predicted_rgba_np = predicted_rgba.cpu().numpy()

                frame = predicted_rgba_to_frame(
                    predicted_rgba=predicted_rgba_np,
                    background_rgb=args.background,
                    scale=args.scale,
                )

                writer.append_data(frame)

                if (
                    frame_index == 0
                    or (frame_index + 1) % 20 == 0
                    or frame_index + 1 == args.num_frames
                ):
                    theta_degrees = math.degrees(theta) % 360.0

                    print(
                        f"Frame {frame_index + 1:03d}/{args.num_frames}: "
                        f"theta={theta_degrees:6.1f} degrees"
                    )

    finally:
        writer.close()

    print("=" * 72)
    print("Done.")
    print("Wrote MP4:", args.output)
    print("=" * 72)


if __name__ == "__main__":
    main()