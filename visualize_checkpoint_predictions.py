#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Allows running without a display/X server.

import matplotlib.pyplot as plt
import numpy as np
import torch

from implicit_dataset import (
    ImplicitRenderDataset,
    build_frame_table,
    build_raw_feature_matrix,
)
from implicit_model import ImplicitFNOImageModel


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize ground-truth versus checkpoint predictions for "
            "implicit B-spline surface or volume rendering."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help=(
            "Path to implicit_bspline_dataset, containing volumes/, "
            "renders_balanced/, and hard_alpha_balanced/."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint path, for example checkpoints_implicit/surface/best.pt",
    )

    parser.add_argument(
        "--mode",
        choices=["surface", "volume"],
        required=True,
        help="Must match the mode used to train the checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./prediction_visualizations"),
        help="Directory where PNG comparison figures are written.",
    )

    parser.add_argument(
        "--num-train",
        type=int,
        default=4,
        help="Number of random training examples to visualize.",
    )

    parser.add_argument(
        "--num-test",
        type=int,
        default=4,
        help="Number of random held-out test examples to visualize.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for choosing visualization examples.",
    )

    parser.add_argument(
        "--alpha-epsilon",
        type=float,
        default=1e-5,
        help="Threshold used only when unpremultiplying RGBA for display.",
    )

    return parser.parse_args()


# ============================================================
# CHECKPOINT UTILITIES
# ============================================================

def load_checkpoint_model_state(model, checkpoint):
    """
    Load model parameters while ignoring the legacy _metadata entry.

    Some prior checkpoints contain '_metadata' inside model_state. It is not
    an actual model parameter/buffer, so torch rejects it by default.
    """
    state = checkpoint.get("model_state", checkpoint).copy()
    state.pop("_metadata", None)

    model.load_state_dict(state, strict=True)


def checkpoint_config_value(checkpoint, name, default):
    """
    Read a value from checkpoint['config'] when available, otherwise use
    a default. This makes the script work with both newer and older
    checkpoints.
    """
    config = checkpoint.get("config", {})

    if not isinstance(config, dict):
        return default

    return config.get(name, default)


# ============================================================
# SPLIT UTILITIES
# ============================================================

def indices_from_geometry_ids(frame_table, geometry_ids):
    """
    Recover frame-table row indices for a stored list of geometry IDs.
    """
    geometry_ids = np.asarray(geometry_ids, dtype=np.int64)

    table_geometry_ids = frame_table["geometry_id"].to_numpy(
        dtype=np.int64
    )

    return np.flatnonzero(
        np.isin(table_geometry_ids, geometry_ids)
    ).astype(np.int64)


def load_saved_split_indices(
    split_json_path: Path,
    frame_table,
):
    """
    Restore the exact geometry-level train/val/test split created during
    training.

    The training script saved geometry IDs rather than fragile row indices,
    which is appropriate because metadata can be sorted/reloaded safely.
    """
    if not split_json_path.is_file():
        raise FileNotFoundError(
            f"Split information file not found:\n  {split_json_path}\n\n"
            "Expected it next to the checkpoint, for example:\n"
            "  checkpoints_implicit/surface/dataset_split_and_features.json"
        )

    with open(split_json_path, "r") as f:
        split_info = json.load(f)

    required_keys = {
        "train_geometry_ids",
        "val_geometry_ids",
        "test_geometry_ids",
    }

    missing = required_keys - set(split_info.keys())

    if missing:
        raise KeyError(
            f"Split JSON is missing keys: {sorted(missing)}"
        )

    return {
        "train": indices_from_geometry_ids(
            frame_table,
            split_info["train_geometry_ids"],
        ),
        "val": indices_from_geometry_ids(
            frame_table,
            split_info["val_geometry_ids"],
        ),
        "test": indices_from_geometry_ids(
            frame_table,
            split_info["test_geometry_ids"],
        ),
    }, split_info


# ============================================================
# DISPLAY UTILITIES
# ============================================================

def premult_rgba_to_display_rgb(
    rgba_chw: np.ndarray,
    alpha_epsilon: float = 1e-5,
) -> np.ndarray:
    """
    Convert [4, H, W] premultiplied RGBA to normal RGB [H, W, 3] for display.

    Since RGB targets are premultiplied:

        C = alpha * RGB

    recover RGB only where alpha is nontrivial:

        RGB = C / alpha

    Transparent pixels are set to black.
    """
    premult_rgb = rgba_chw[:3]
    alpha = rgba_chw[3:4]

    rgb = np.zeros_like(premult_rgb, dtype=np.float32)

    valid = alpha > alpha_epsilon
    rgb = np.where(
        valid,
        premult_rgb / np.maximum(alpha, alpha_epsilon),
        0.0,
    )

    rgb = np.clip(rgb, 0.0, 1.0)

    return np.transpose(rgb, (1, 2, 0))


def premult_rgb_on_black(
    rgba_chw: np.ndarray,
) -> np.ndarray:
    """
    Return premultiplied RGB directly as an RGB image.

    This is the appearance composited over black and is often the most useful
    representation for comparing a model's output.
    """
    rgb = np.clip(rgba_chw[:3], 0.0, 1.0)
    return np.transpose(rgb, (1, 2, 0))


def save_comparison_figure(
    target_rgba: np.ndarray,
    prediction_rgba: np.ndarray,
    metadata,
    output_path: Path,
    split_name: str,
    alpha_epsilon: float,
):
    """
    Save a six-panel comparison image.
    """
    target_rgb_black = premult_rgb_on_black(target_rgba)
    prediction_rgb_black = premult_rgb_on_black(prediction_rgba)

    target_alpha = np.clip(target_rgba[3], 0.0, 1.0)
    prediction_alpha = np.clip(prediction_rgba[3], 0.0, 1.0)

    alpha_error = np.abs(prediction_alpha - target_alpha)

    rgb_error = np.mean(
        np.abs(
            prediction_rgb_black - target_rgb_black
        ),
        axis=2,
    )

    sample_id = int(metadata["sample_id"])
    geometry_id = int(metadata["geometry_id"])
    sigma = float(metadata["sigma"])
    view_idx = int(metadata["view_idx"])

    opacity = float(metadata["opacity"])
    env_id = int(metadata["env_id"])

    figure, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(12, 8),
        constrained_layout=True,
    )

    axes[0, 0].imshow(target_rgb_black)
    axes[0, 0].set_title("Target RGB\n(composited on black)")

    axes[0, 1].imshow(prediction_rgb_black)
    axes[0, 1].set_title("Prediction RGB\n(composited on black)")

    axes[0, 2].imshow(
        rgb_error,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    axes[0, 2].set_title("Mean absolute RGB error")

    axes[1, 0].imshow(
        target_alpha,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1, 0].set_title("Target alpha")

    axes[1, 1].imshow(
        prediction_alpha,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1, 1].set_title("Predicted alpha")

    axes[1, 2].imshow(
        alpha_error,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1, 2].set_title("Absolute alpha error")

    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])

    figure.suptitle(
        f"{split_name.upper()} | mode={metadata['render_mode']} | "
        f"geometry={geometry_id} | sample={sample_id} | "
        f"sigma={sigma:.3f} | view={view_idx} | "
        f"opacity={opacity:.3f} | env={env_id}",
        fontsize=12,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


# ============================================================
# MODEL / DATA CONSTRUCTION
# ============================================================

def build_dataset_and_model(args, checkpoint, device):
    dataset_root = args.dataset_root.resolve()

    volumes_dir = dataset_root / "volumes"
    renders_dir = dataset_root / "renders_balanced"
    alpha_dir = dataset_root / "hard_alpha_balanced"

    image_metadata_csv = (
        renders_dir / "metadata_images_all_sharded.csv"
    )
    alpha_metadata_csv = alpha_dir / "metadata_alpha_all.csv"
    volume_metadata_csv = volumes_dir / "metadata_volumes.csv"

    checkpoint_config = checkpoint.get("config", {})

    checkpoint_mode = checkpoint_config.get("mode")

    if checkpoint_mode is not None and checkpoint_mode != args.mode:
        raise RuntimeError(
            f"Checkpoint mode is '{checkpoint_mode}', but you requested "
            f"mode '{args.mode}'."
        )

    height = int(
        checkpoint_config_value(
            checkpoint,
            "height",
            32,
        )
    )
    width = int(
        checkpoint_config_value(
            checkpoint,
            "width",
            32,
        )
    )

    use_sh = not bool(
        checkpoint_config_value(
            checkpoint,
            "no_sh",
            False,
        )
    )

    fno_modes = int(
        checkpoint_config_value(
            checkpoint,
            "fno_modes",
            16,
        )
    )

    print("Checkpoint/dataset configuration:")
    print(f"  mode:      {args.mode}")
    print(f"  resolution:{width}x{height}")
    print(f"  use SH:    {use_sh}")
    print(f"  FNO modes: {fno_modes}")

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

    if checkpoint_features is not None:
        if feature_names != checkpoint_features:
            raise RuntimeError(
                "Current dataset feature order does not match checkpoint.\n\n"
                f"Current features:\n{feature_names}\n\n"
                f"Checkpoint features:\n{checkpoint_features}"
            )

    param_mean = checkpoint.get("param_mean")
    param_std = checkpoint.get("param_std")

    if param_mean is None or param_std is None:
        raise RuntimeError(
            "Checkpoint does not contain param_mean and param_std. "
            "Use a checkpoint created by the new train_implicit.py."
        )

    param_mean = np.asarray(param_mean, dtype=np.float32)
    param_std = np.asarray(param_std, dtype=np.float32)

    if raw_features.shape[1] != len(param_mean):
        raise RuntimeError(
            f"Feature dimension mismatch: dataset has {raw_features.shape[1]}, "
            f"checkpoint normalization has {len(param_mean)}."
        )

    latent_dim = int(
        checkpoint.get(
            "latent_dim",
            raw_features.shape[1],
        )
    )

    if latent_dim != raw_features.shape[1]:
        raise RuntimeError(
            f"Checkpoint latent_dim={latent_dim}, but current data has "
            f"{raw_features.shape[1]} features."
        )

    model = ImplicitFNOImageModel(
        latent_dim=latent_dim,
        image_height=height,
        image_width=width,
        fno_modes=fno_modes,
    ).to(device)

    load_checkpoint_model_state(model, checkpoint)
    model.eval()

    return (
        frame_table,
        raw_features,
        param_mean,
        param_std,
        model,
        renders_dir,
        alpha_dir,
        height,
        width,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    (
        frame_table,
        raw_features,
        param_mean,
        param_std,
        model,
        renders_dir,
        alpha_dir,
        height,
        width,
    ) = build_dataset_and_model(
        args=args,
        checkpoint=checkpoint,
        device=device,
    )

    split_json_path = (
        args.checkpoint.parent / "dataset_split_and_features.json"
    )

    split_indices, split_info = load_saved_split_indices(
        split_json_path=split_json_path,
        frame_table=frame_table,
    )

    # Create datasets for train/test rows. They reuse memory-mapped shards.
    train_dataset = ImplicitRenderDataset(
        frame_table=frame_table,
        raw_features=raw_features,
        row_indices=split_indices["train"],
        param_mean=param_mean,
        param_std=param_std,
        renders_dir=renders_dir,
        alpha_dir=alpha_dir,
        image_height=height,
        image_width=width,
    )

    test_dataset = ImplicitRenderDataset(
        frame_table=frame_table,
        raw_features=raw_features,
        row_indices=split_indices["test"],
        param_mean=param_mean,
        param_std=param_std,
        renders_dir=renders_dir,
        alpha_dir=alpha_dir,
        image_height=height,
        image_width=width,
    )

    rng = np.random.default_rng(args.seed)

    requested = {
        "train": (train_dataset, args.num_train),
        "test": (test_dataset, args.num_test),
    }

    print("=" * 72)
    print("Creating checkpoint prediction comparisons")
    print(f"Output directory: {args.output_dir}")
    print("=" * 72)

    with torch.no_grad():
        for split_name, (dataset, num_examples) in requested.items():
            if num_examples <= 0:
                continue

            num_examples = min(num_examples, len(dataset))

            chosen_indices = rng.choice(
                len(dataset),
                size=num_examples,
                replace=False,
            )

            split_output_dir = args.output_dir / split_name
            split_output_dir.mkdir(parents=True, exist_ok=True)

            for output_index, dataset_index in enumerate(chosen_indices):
                (
                    parameter_vector,
                    target_rgba,
                    geometry_id,
                    sample_id,
                ) = dataset[int(dataset_index)]

                parameter_vector = parameter_vector.unsqueeze(0).to(device)

                prediction_rgba = model(parameter_vector)[0]

                target_np = target_rgba.cpu().numpy()
                prediction_np = prediction_rgba.cpu().numpy()

                # Recover original frame-table row index from this split dataset.
                frame_row_index = int(
                    dataset.row_indices[int(dataset_index)]
                )
                metadata = frame_table.iloc[frame_row_index]

                output_name = (
                    f"{split_name}_{output_index:03d}_"
                    f"geom{int(geometry_id):06d}_"
                    f"sample{int(sample_id):06d}_"
                    f"view{int(metadata['view_idx']):03d}.png"
                )

                output_path = split_output_dir / output_name

                save_comparison_figure(
                    target_rgba=target_np,
                    prediction_rgba=prediction_np,
                    metadata=metadata,
                    output_path=output_path,
                    split_name=split_name,
                    alpha_epsilon=args.alpha_epsilon,
                )

                print(
                    f"[{split_name}] "
                    f"{output_index + 1}/{num_examples}: "
                    f"{output_path.name}"
                )

    print("=" * 72)
    print("Done.")
    print(f"Figures written to: {args.output_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()