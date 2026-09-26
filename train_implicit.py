#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from implicit_dataset import (
    ImplicitRenderDataset,
    build_frame_table,
    build_raw_feature_matrix,
    split_indices_by_geometry,
    training_feature_statistics,
)
from implicit_model import ImplicitFNOImageModel


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# LOSS / METRICS
# ============================================================

def premultiplied_rgba_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    alpha_weight: float = 2.0,
    foreground_weight: float = 8.0,
) -> Dict[str, torch.Tensor]:
    """
    Foreground-weighted premultiplied RGBA MSE.

    Background pixels remain part of the loss, but pixels having target alpha
    near one receive greater weight. This prevents the trivial solution:
    predict fully transparent black everywhere.
    """
    predicted_rgb = prediction[:, :3]
    predicted_alpha = prediction[:, 3:4]

    target_rgb = target[:, :3]
    target_alpha = target[:, 3:4]

    # Shape [B, 1, H, W].
    #
    # alpha=0   -> weight 1
    # alpha=1   -> weight 1 + foreground_weight
    pixel_weight = 1.0 + foreground_weight * target_alpha

    rgb_squared_error = (predicted_rgb - target_rgb).square()
    alpha_squared_error = (predicted_alpha - target_alpha).square()

    # RGB has three channels, while alpha has one.
    rgb_loss = (pixel_weight * rgb_squared_error).mean()
    alpha_loss = (pixel_weight * alpha_squared_error).mean()

    total_loss = rgb_loss + alpha_weight * alpha_loss

    return {
        "total": total_loss,
        "rgb": rgb_loss,
        "alpha": alpha_loss,
    }
    

def premultiplied_rgba_l1_finetune_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    alpha_weight: float = 2.0,
    foreground_weight: float = 8.0,
    l1_weight: float = 0.75,
) -> Dict[str, torch.Tensor]:
    """
    Foreground-weighted mixed MSE/L1 loss for sharpening an already-trained
    premultiplied RGBA predictor.

    l1_weight=0.0 -> pure MSE
    l1_weight=1.0 -> pure L1
    """
    predicted_rgb = prediction[:, :3]
    predicted_alpha = prediction[:, 3:4]

    target_rgb = target[:, :3]
    target_alpha = target[:, 3:4]

    # Surface/volume foreground pixels are emphasized.
    pixel_weight = 1.0 + foreground_weight * target_alpha

    rgb_mse = (
        pixel_weight * (predicted_rgb - target_rgb).square()
    ).mean()

    alpha_mse = (
        pixel_weight * (predicted_alpha - target_alpha).square()
    ).mean()

    rgb_l1 = (
        pixel_weight * (predicted_rgb - target_rgb).abs()
    ).mean()

    alpha_l1 = (
        pixel_weight * (predicted_alpha - target_alpha).abs()
    ).mean()

    mse_loss = rgb_mse + alpha_weight * alpha_mse
    l1_loss = rgb_l1 + alpha_weight * alpha_l1

    total_loss = (
        (1.0 - l1_weight) * mse_loss
        + l1_weight * l1_loss
    )

    return {
        "total": total_loss,
        "rgb": (
            (1.0 - l1_weight) * rgb_mse
            + l1_weight * rgb_l1
        ),
        "alpha": (
            (1.0 - l1_weight) * alpha_mse
            + l1_weight * alpha_l1
        ),
        "mse": mse_loss,
        "l1": l1_loss,
    }

# ============================================================
# TRAIN / EVAL EPOCHS
# ============================================================

def run_epoch(
    model,
    loader,
    device,
    optimizer,
    scaler,
    loss_type,
    alpha_weight: float,
    foreground_weight: float,
    use_amp: bool,
    l1_weight:float,
):
    """
    If optimizer is None, run validation.
    Otherwise, run training.
    """
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_examples = 0
    total_loss = 0.0
    total_rgb_loss = 0.0
    total_alpha_loss = 0.0

    for params, target_rgba, _, _ in loader:
        params = params.to(device, non_blocking=True)
        target_rgba = target_rgba.to(device, non_blocking=True)

        batch_size = params.shape[0]

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                prediction = model(params)

                if loss_type == "mse":
                    losses = premultiplied_rgba_loss(
                        prediction=prediction,
                        target=target_rgba,
                        alpha_weight=alpha_weight,
                        foreground_weight=foreground_weight,
                    )
                elif loss_type == "mixed_l1":
                    losses = premultiplied_rgba_l1_finetune_loss(
                        prediction=prediction,
                        target=target_rgba,
                        alpha_weight=alpha_weight,
                        foreground_weight=foreground_weight,
                        l1_weight=l1_weight,
                    )
                else:
                    raise ValueError(f"Unknown loss_type: {loss_type}")

                loss = losses["total"]

            if is_training:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )
                scaler.step(optimizer)
                scaler.update()

        total_examples += batch_size
        total_loss += float(losses["total"].detach()) * batch_size
        total_rgb_loss += float(losses["rgb"].detach()) * batch_size
        total_alpha_loss += float(losses["alpha"].detach()) * batch_size

    if total_examples == 0:
        raise RuntimeError("Epoch received zero examples.")

    return {
        "loss": total_loss / total_examples,
        "rgb_loss": total_rgb_loss / total_examples,
        "alpha_loss": total_alpha_loss / total_examples,
    }


# ============================================================
# CHECKPOINTING
# ============================================================

def save_checkpoint(
    checkpoint_path: Path,
    model,
    optimizer,
    epoch: int,
    best_val_loss: float,
    feature_names,
    param_mean,
    param_std,
    config,
):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "feature_names": feature_names,
            "param_mean": np.asarray(param_mean, dtype=np.float32),
            "param_std": np.asarray(param_std, dtype=np.float32),
            "config": vars(config),
        },
        checkpoint_path,
    )

def load_model_state(model, checkpoint):
    """
    Load a checkpoint model state while ignoring legacy/non-parameter metadata.
    """
    state = checkpoint.get("model_state", checkpoint).copy()
    state.pop("_metadata", None)
    model.load_state_dict(state)

# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a surface-only or volume-only FNO RGBA predictor "
            "for the implicit B-spline dataset."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("./implicit_bspline_dataset"),
        help="Directory containing volumes/, renders_balanced/, etc.",
    )

    parser.add_argument(
        "--mode",
        choices=["surface", "volume"],
        required=True,
        help="Train one model for this render mode.",
    )

    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)

    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--alpha-weight", type=float, default=2.0)

    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("./checkpoints_implicit"),
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional checkpoint to resume.",
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save an epoch checkpoint every N epochs.",
    )

    parser.add_argument(
        "--no-sh",
        action="store_true",
        help="Disable SH environment-lighting coefficients as conditioning.",
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA mixed precision.",
    )

    parser.add_argument(
        "--fno-modes",
        type=int,
        default=16,
        help="Fourier modes per spatial axis. 16 is suitable for 32x32.",
    )
    
    parser.add_argument(
        "--foreground-weight",
        type=float,
        default=8.0,
        help=(
            "Extra loss weighting for pixels with nonzero target alpha. "
            "Prevents transparent-background collapse."
        ),
    )
    
    parser.add_argument(
        "--loss-type",
        choices=["mse", "mixed_l1"],
        default="mse",
        help="Use normal weighted MSE training or mixed MSE/L1 fine-tuning.",
    )
    
    parser.add_argument(
        "--l1-weight",
        type=float,
        default=0.75,
        help=(
            "L1 fraction for --loss-type mixed_l1. "
            "0.0 means pure MSE; 1.0 means pure L1."
        ),
    )
    
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help=(
            "When resuming, load model weights but start a fresh optimizer. "
            "Use this for fine-tuning with a new learning rate or loss."
        ),
    )
    
    parser.add_argument(
        "--reset-best",
        action="store_true",
        help=(
            "Reset best validation loss when resuming. Use this when changing "
            "the loss function, such as switching from MSE to mixed L1/MSE "
            "fine-tuning."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    set_seed(args.seed)

    dataset_root = args.dataset_root.resolve()

    volumes_dir = dataset_root / "volumes"
    renders_dir = dataset_root / "renders_balanced"
    alpha_dir = dataset_root / "hard_alpha_balanced"

    image_metadata_csv = (
        renders_dir / "metadata_images_all_sharded.csv"
    )
    alpha_metadata_csv = alpha_dir / "metadata_alpha_all.csv"
    volume_metadata_csv = volumes_dir / "metadata_volumes.csv"

    checkpoint_dir = args.checkpoint_dir / args.mode
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Implicit B-spline FNO training")
    print(f"Mode:          {args.mode}")
    print(f"Dataset root:  {dataset_root}")
    print(f"Resolution:    {args.width}x{args.height}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Epochs:        {args.epochs}")
    print(f"Use SH:        {not args.no_sh}")
    print("=" * 72)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

        torch.backends.cudnn.benchmark = True

    use_amp = device.type == "cuda" and not args.no_amp
    print("Mixed precision:", use_amp)

    # --------------------------------------------------------
    # Metadata and conditioning vectors
    # --------------------------------------------------------

    frame_table, ctrl_columns, sh_columns = build_frame_table(
        dataset_root=dataset_root,
        renders_dir=renders_dir,
        alpha_dir=alpha_dir,
        image_metadata_csv=image_metadata_csv,
        alpha_metadata_csv=alpha_metadata_csv,
        volume_metadata_csv=volume_metadata_csv,
        mode=args.mode,
        use_sh=not args.no_sh,
    )

    raw_features, feature_names = build_raw_feature_matrix(
        df=frame_table,
        ctrl_columns=ctrl_columns,
        sh_columns=sh_columns,
        use_sh=not args.no_sh,
    )

    split_indices = split_indices_by_geometry(
        df=frame_table,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    param_mean, param_std = training_feature_statistics(
        raw_features=raw_features,
        train_indices=split_indices["train"],
    )

    latent_dim = raw_features.shape[1]

    print(f"Latent dimension: {latent_dim}")
    print("Feature names:")

    for index, feature_name in enumerate(feature_names):
        print(f"  [{index:02d}] {feature_name}")

    # Save split and feature information for reproducibility.
    split_info_path = checkpoint_dir / "dataset_split_and_features.json"

    split_info = {
        "mode": args.mode,
        "seed": args.seed,
        "feature_names": feature_names,
        "num_train_frames": int(len(split_indices["train"])),
        "num_val_frames": int(len(split_indices["val"])),
        "num_test_frames": int(len(split_indices["test"])),
        "train_geometry_ids": sorted(
            frame_table.iloc[
                split_indices["train"]
            ]["geometry_id"].astype(int).unique().tolist()
        ),
        "val_geometry_ids": sorted(
            frame_table.iloc[
                split_indices["val"]
            ]["geometry_id"].astype(int).unique().tolist()
        ),
        "test_geometry_ids": sorted(
            frame_table.iloc[
                split_indices["test"]
            ]["geometry_id"].astype(int).unique().tolist()
        ),
    }

    with open(split_info_path, "w") as f:
        json.dump(split_info, f, indent=2)

    np.save(checkpoint_dir / "param_mean.npy", param_mean)
    np.save(checkpoint_dir / "param_std.npy", param_std)

    # --------------------------------------------------------
    # Dataset objects
    # --------------------------------------------------------

    train_dataset = ImplicitRenderDataset(
        frame_table=frame_table,
        raw_features=raw_features,
        row_indices=split_indices["train"],
        param_mean=param_mean,
        param_std=param_std,
        renders_dir=renders_dir,
        alpha_dir=alpha_dir,
        image_height=args.height,
        image_width=args.width,
    )

    val_dataset = ImplicitRenderDataset(
        frame_table=frame_table,
        raw_features=raw_features,
        row_indices=split_indices["val"],
        param_mean=param_mean,
        param_std=param_std,
        renders_dir=renders_dir,
        alpha_dir=alpha_dir,
        image_height=args.height,
        image_width=args.width,
    )

    test_dataset = ImplicitRenderDataset(
        frame_table=frame_table,
        raw_features=raw_features,
        row_indices=split_indices["test"],
        param_mean=param_mean,
        param_std=param_std,
        renders_dir=renders_dir,
        alpha_dir=alpha_dir,
        image_height=args.height,
        image_width=args.width,
    )

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }

    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    # --------------------------------------------------------
    # Model / optimizer
    # --------------------------------------------------------

    model = ImplicitFNOImageModel(
        latent_dim=latent_dim,
        image_height=args.height,
        image_width=args.width,
        fno_modes=args.fno_modes,
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Trainable parameters: {parameter_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    start_epoch = 0
    best_val_loss = float("inf")

    # --------------------------------------------------------
    # Optional checkpoint resume
    # --------------------------------------------------------

    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(
                f"Requested resume checkpoint does not exist: {args.resume}"
            )

        print("Resuming from:", args.resume)

        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )

        load_model_state(model, checkpoint)

        if "optimizer_state" in checkpoint and not args.reset_optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        else:
            print("Using a fresh optimizer for fine-tuning.")

        start_epoch = int(checkpoint.get("epoch", 0))
        if args.reset_best:
            best_val_loss = float("inf")
            print(
                "Resetting best validation loss because this run is using "
                "a new objective / fine-tuning stage."
            )
        else:
            best_val_loss = float(
                checkpoint.get("best_val_loss", float("inf"))
            )

        old_features = checkpoint.get("feature_names")

        if old_features is not None and old_features != feature_names:
            raise RuntimeError(
                "Checkpoint feature ordering differs from this dataset setup. "
                "Do not resume with incompatible conditioning features."
            )

        print(
            f"Resume successful. Starting at epoch {start_epoch + 1}; "
            f"best validation loss so far = {best_val_loss:.8f}"
        )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = []

    for epoch in range(start_epoch, args.epochs):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            alpha_weight=args.alpha_weight,
            foreground_weight=args.foreground_weight,
            use_amp=use_amp,
            l1_weight=args.l1_weight,
            loss_type=args.loss_type,
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            scaler=scaler,
            alpha_weight=args.alpha_weight,
            foreground_weight=args.foreground_weight,
            use_amp=use_amp,
            l1_weight=args.l1_weight,
            loss_type=args.loss_type,
        )

        epoch_number = epoch + 1

        print(
            f"[{args.mode}] "
            f"epoch {epoch_number:03d}/{args.epochs} | "
            f"train={train_metrics['loss']:.7f} "
            f"(rgb={train_metrics['rgb_loss']:.7f}, "
            f"alpha={train_metrics['alpha_loss']:.7f}) | "
            f"val={val_metrics['loss']:.7f} "
            f"(rgb={val_metrics['rgb_loss']:.7f}, "
            f"alpha={val_metrics['alpha_loss']:.7f})"
        )

        history.append(
            {
                "epoch": epoch_number,
                "train_loss": train_metrics["loss"],
                "train_rgb_loss": train_metrics["rgb_loss"],
                "train_alpha_loss": train_metrics["alpha_loss"],
                "val_loss": val_metrics["loss"],
                "val_rgb_loss": val_metrics["rgb_loss"],
                "val_alpha_loss": val_metrics["alpha_loss"],
            }
        )

        # Always update latest checkpoint.
        save_checkpoint(
            checkpoint_path=checkpoint_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch_number,
            best_val_loss=best_val_loss,
            feature_names=feature_names,
            param_mean=param_mean,
            param_std=param_std,
            config=args,
        )

        # Save best validation checkpoint.
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

            save_checkpoint(
                checkpoint_path=checkpoint_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch_number,
                best_val_loss=best_val_loss,
                feature_names=feature_names,
                param_mean=param_mean,
                param_std=param_std,
                config=args,
            )

            print(
                f"  Saved new best checkpoint "
                f"(val={best_val_loss:.7f})."
            )

        # Periodic archived checkpoint.
        if epoch_number % args.save_every == 0:
            save_checkpoint(
                checkpoint_path=(
                    checkpoint_dir / f"epoch_{epoch_number:03d}.pt"
                ),
                model=model,
                optimizer=optimizer,
                epoch=epoch_number,
                best_val_loss=best_val_loss,
                feature_names=feature_names,
                param_mean=param_mean,
                param_std=param_std,
                config=args,
            )

        # Save history after every epoch, so interruption preserves progress.
        history_path = checkpoint_dir / "history.csv"

        import pandas as pd
        pd.DataFrame(history).to_csv(history_path, index=False)

    # --------------------------------------------------------
    # Final held-out test evaluation
    # --------------------------------------------------------

    best_checkpoint_path = checkpoint_dir / "best.pt"

    if best_checkpoint_path.is_file():
        checkpoint = torch.load(
            best_checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        load_model_state(model, checkpoint)

    test_metrics = run_epoch(
        model=model,
        loader=test_loader,
        device=device,
        optimizer=None,
        scaler=scaler,
        alpha_weight=args.alpha_weight,
        foreground_weight=args.foreground_weight,
        use_amp=use_amp,
        l1_weight=args.l1_weight,
        loss_type=args.loss_type,
    )

    print("=" * 72)
    print(f"[{args.mode}] final held-out geometry test metrics")
    print(f"  total loss: {test_metrics['loss']:.8f}")
    print(f"  RGB loss:   {test_metrics['rgb_loss']:.8f}")
    print(f"  alpha loss: {test_metrics['alpha_loss']:.8f}")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()