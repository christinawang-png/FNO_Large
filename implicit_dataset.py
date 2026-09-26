#!/usr/bin/env python
"""
Dataset utilities for the implicit 2x2x2 B-spline rendering dataset.

Expected layout:

implicit_bspline_dataset/
├── volumes/
│   └── metadata_volumes.csv
├── renders_balanced/
│   ├── images_32x32_<job_id>_shard_0000.npy
│   └── metadata_images_all_sharded.csv
└── hard_alpha_balanced/
    ├── alpha_32x32_<job_id>_shard_0000.npy
    └── metadata_alpha_all.csv
"""

from __future__ import annotations

import re
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ============================================================
# METADATA / FEATURE HELPERS
# ============================================================

_CTRL_RE = re.compile(r"^ctrl_(\d+)_(\d+)_(\d+)$")
_SH_RE = re.compile(r"^sh_l(-?\d+)_m(-?\d+)_([rgb])$")


def control_column_sort_key(name: str) -> Tuple[int, int, int]:
    """
    Sort:
        ctrl_0_0_0, ctrl_0_0_1, ..., ctrl_1_1_1
    """
    match = _CTRL_RE.match(name)

    if match is None:
        raise ValueError(f"Not a valid control-grid column name: {name}")

    return tuple(int(value) for value in match.groups())


def sh_column_sort_key(name: str) -> Tuple[int, int, int]:
    """
    Deterministic SH column ordering:

        sh_l0_m0_r, sh_l0_m0_g, sh_l0_m0_b,
        sh_l1_m-1_r, ...
    """
    match = _SH_RE.match(name)

    if match is None:
        raise ValueError(f"Not a valid SH column name: {name}")

    l, m, channel = match.groups()
    channel_index = {"r": 0, "g": 1, "b": 2}[channel]

    return int(l), int(m), channel_index


def find_control_columns(df: pd.DataFrame) -> List[str]:
    columns = [column for column in df.columns if _CTRL_RE.match(column)]

    if len(columns) != 8:
        raise RuntimeError(
            "Expected exactly eight B-spline control-grid columns "
            f"ctrl_0_0_0 through ctrl_1_1_1, found {len(columns)}:\n"
            f"{columns}"
        )

    return sorted(columns, key=control_column_sort_key)


def find_sh_columns(df: pd.DataFrame) -> List[str]:
    columns = [column for column in df.columns if _SH_RE.match(column)]

    if not columns:
        raise RuntimeError(
            "No spherical-harmonic columns found. Expected columns such as "
            "'sh_l0_m0_r'."
        )

    return sorted(columns, key=sh_column_sort_key)


def shard_path_from_id(
    directory: Path,
    prefix: str,
    shard_id: str,
    height: int,
    width: int,
) -> Path:
    """
    Convert render metadata shard ID into a real NPY filename.

    Renderer metadata contains IDs such as:

        part1_1_123456_task0_0

    but the actual RGB file is:

        images_32x32_part1_1_123456_task0_shard_0000.npy

    and alpha file is:

        alpha_32x32_part1_1_123456_task0_shard_0000.npy

    The final underscore-delimited part of shard_id is the local shard index.
    """
    shard_id = str(shard_id)

    if "_" not in shard_id:
        raise ValueError(
            f"Unexpected shard_id '{shard_id}'. "
            "Expected '<job_id>_<local_shard_index>'."
        )

    job_id, local_shard_text = shard_id.rsplit("_", 1)

    try:
        local_shard_index = int(local_shard_text)
    except ValueError as exc:
        raise ValueError(
            f"Cannot parse local shard index from shard_id='{shard_id}'"
        ) from exc

    filename = (
        f"{prefix}_{width}x{height}_{job_id}_"
        f"shard_{local_shard_index:04d}.npy"
    )

    return directory / filename


# ============================================================
# FRAME TABLE CONSTRUCTION
# ============================================================

def build_frame_table(
    dataset_root: Path,
    renders_dir: Path,
    alpha_dir: Path,
    image_metadata_csv: Path,
    alpha_metadata_csv: Path,
    volume_metadata_csv: Path,
    mode: str,
    use_sh: bool = True,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Build one table with:

      - rendered-image metadata;
      - matching alpha metadata;
      - B-spline control-grid values;
      - sigma;
      - geometry ID;
      - deterministic feature columns.

    Parameters
    ----------
    mode:
        "surface" or "volume".
    """
    if mode not in {"surface", "volume"}:
        raise ValueError(f"mode must be surface or volume, got '{mode}'")

    for path in [
        image_metadata_csv,
        alpha_metadata_csv,
        volume_metadata_csv,
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"Required metadata file not found: {path}")

    print("Loading RGB metadata:", image_metadata_csv)
    df_img = pd.read_csv(image_metadata_csv, low_memory=False)

    print("Loading alpha metadata:", alpha_metadata_csv)
    df_alpha = pd.read_csv(alpha_metadata_csv, low_memory=False)

    print("Loading volume metadata:", volume_metadata_csv)
    df_vol = pd.read_csv(volume_metadata_csv, low_memory=False)

    required_img_columns = {
        "sample_id",
        "geometry_id",
        "render_mode",
        "shard_id",
        "idx_in_shard",
        "base_color_r",
        "base_color_g",
        "base_color_b",
        "opacity",
        "phi",
        "theta",
    }

    missing_img_columns = required_img_columns - set(df_img.columns)

    if missing_img_columns:
        raise KeyError(
            "RGB metadata is missing required columns:\n"
            f"{sorted(missing_img_columns)}"
        )

    required_alpha_columns = {
        "img_shard_id",
        "idx_in_img_shard",
        "alpha_shard_id",
        "idx_in_alpha_shard",
    }

    missing_alpha_columns = required_alpha_columns - set(df_alpha.columns)

    if missing_alpha_columns:
        raise KeyError(
            "Alpha metadata is missing required columns:\n"
            f"{sorted(missing_alpha_columns)}"
        )

    required_vol_columns = {"sample_id", "geometry_id", "sigma"}

    missing_vol_columns = required_vol_columns - set(df_vol.columns)

    if missing_vol_columns:
        raise KeyError(
            "Volume metadata is missing required columns:\n"
            f"{sorted(missing_vol_columns)}"
        )

    # Ensure merge keys are always strings/integers, not inferred mixed types.
    df_img["shard_id"] = df_img["shard_id"].astype(str)
    df_img["idx_in_shard"] = df_img["idx_in_shard"].astype(np.int64)

    df_alpha["img_shard_id"] = df_alpha["img_shard_id"].astype(str)
    df_alpha["idx_in_img_shard"] = df_alpha["idx_in_img_shard"].astype(
        np.int64
    )
    df_alpha["alpha_shard_id"] = df_alpha["alpha_shard_id"].astype(str)
    df_alpha["idx_in_alpha_shard"] = df_alpha["idx_in_alpha_shard"].astype(
        np.int64
    )

    # Retain only alpha-location columns before merging. This avoids duplicate
    # sample_id / geometry_id / sigma columns from alpha metadata.
    alpha_locator_columns = [
        "img_shard_id",
        "idx_in_img_shard",
        "alpha_shard_id",
        "idx_in_alpha_shard",
    ]

    df_alpha_locator = df_alpha[alpha_locator_columns].copy()

    # one_to_one catches duplicate rendering jobs that reused the same shard IDs.
    df = df_img.merge(
        df_alpha_locator,
        how="inner",
        left_on=["shard_id", "idx_in_shard"],
        right_on=["img_shard_id", "idx_in_img_shard"],
        validate="one_to_one",
    )

    if len(df) != len(df_img):
        print(
            f"[WARN] RGB rows before alpha join: {len(df_img):,}; "
            f"after alpha join: {len(df):,}. "
            "Some RGB frames have no matching alpha record."
        )

    # Keep only requested render mode.
    df = df[df["render_mode"].astype(str) == mode].copy()

    if df.empty:
        raise RuntimeError(
            f"No '{mode}' rows found after metadata join. "
            "Check render_mode values and merged CSV files."
        )

    ctrl_columns = find_control_columns(df_vol)

    # Pull controls from volume metadata. They are the authoritative source.
    vol_columns = [
        "sample_id",
        "geometry_id",
        "sigma",
        *ctrl_columns,
    ]

    df_vol_subset = df_vol[vol_columns].copy()
    df_vol_subset["sample_id"] = df_vol_subset["sample_id"].astype(np.int64)

    # RGB metadata also has geometry_id/sigma. Keep its fields, use volume
    # metadata only for control coefficients.
    df = df.merge(
        df_vol_subset[["sample_id", *ctrl_columns]],
        how="inner",
        on="sample_id",
        validate="many_to_one",
    )

    if df.empty:
        raise RuntimeError(
            "No rows remained after joining render metadata to volume metadata."
        )

    # Make basic numeric columns numeric and fill unavailable values.
    #
    # Volume frames have NaN metallic/roughness/specular by design.
    numeric_defaults = {
        "base_color_r": 0.0,
        "base_color_g": 0.0,
        "base_color_b": 0.0,
        "metallic": 0.0,
        "roughness": 0.0,
        "specular": 0.0,
        "opacity": 0.0,
        "phi": 0.0,
        "theta": 0.0,
        "radius": 0.0,
        "sigma": 0.0,
    }

    for column, default in numeric_defaults.items():
        if column not in df.columns:
            df[column] = default

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        ).fillna(default)

    for column in ctrl_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if df[ctrl_columns].isna().any().any():
        raise RuntimeError("NaN control coefficients found after metadata merge.")

    sh_columns: List[str] = []

    if use_sh:
        sh_columns = find_sh_columns(df)

        for column in sh_columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            ).fillna(0.0)

    # Stable ordering is important for deterministic splits and checkpoints.
    df = df.sort_values(
        by=["geometry_id", "sample_id", "view_idx", "render_mode"],
        kind="stable",
    ).reset_index(drop=True)

    print(
        f"Loaded {len(df):,} usable '{mode}' frames from "
        f"{df['geometry_id'].nunique():,} unique geometries."
    )

    return df, ctrl_columns, sh_columns


def build_raw_feature_matrix(
    df: pd.DataFrame,
    ctrl_columns: List[str],
    sh_columns: List[str],
    use_sh: bool,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build unnormalized conditioning vectors.

    Feature order:

      1. Eight implicit B-spline corner coefficients.
      2. Sigma.
      3. RGB base color.
      4. Metallic, roughness, specular.
      5. Opacity.
      6. sin(phi), cos(phi), sin(theta), cos(theta).
      7. SH coefficients, if enabled.

    Radius is omitted because your current renderer always uses one fixed
    camera radius. A constant feature provides no learning signal.
    """
    feature_blocks = []
    feature_names = []

    # Eight corner coefficients.
    feature_blocks.append(
        df[ctrl_columns].to_numpy(dtype=np.float32, copy=True)
    )
    feature_names.extend(ctrl_columns)

    # Gaussian thickness.
    feature_blocks.append(
        df[["sigma"]].to_numpy(dtype=np.float32, copy=True)
    )
    feature_names.append("sigma")

    # Base color.
    color_columns = [
        "base_color_r",
        "base_color_g",
        "base_color_b",
    ]
    feature_blocks.append(
        df[color_columns].to_numpy(dtype=np.float32, copy=True)
    )
    feature_names.extend(color_columns)

    # Surface-only material parameters are zero for volume examples.
    material_columns = [
        "metallic",
        "roughness",
        "specular",
    ]
    feature_blocks.append(
        df[material_columns].to_numpy(dtype=np.float32, copy=True)
    )
    feature_names.extend(material_columns)

    # Opacity is directly sampled for surfaces and converted into the
    # sigma-normalized density multiplier for volumes.
    feature_blocks.append(
        df[["opacity"]].to_numpy(dtype=np.float32, copy=True)
    )
    feature_names.append("opacity")

    phi = df["phi"].to_numpy(dtype=np.float32, copy=True)
    theta = df["theta"].to_numpy(dtype=np.float32, copy=True)

    camera_features = np.stack(
        [
            np.sin(phi),
            np.cos(phi),
            np.sin(theta),
            np.cos(theta),
        ],
        axis=1,
    ).astype(np.float32)

    feature_blocks.append(camera_features)
    feature_names.extend(
        [
            "sin_phi",
            "cos_phi",
            "sin_theta",
            "cos_theta",
        ]
    )

    if use_sh:
        feature_blocks.append(
            df[sh_columns].to_numpy(dtype=np.float32, copy=True)
        )
        feature_names.extend(sh_columns)

    raw_features = np.concatenate(feature_blocks, axis=1).astype(
        np.float32,
        copy=False,
    )

    if not np.all(np.isfinite(raw_features)):
        raise RuntimeError("Raw feature matrix contains NaN or Inf values.")

    return raw_features, feature_names


# ============================================================
# GEOMETRY-LEVEL SPLITTING
# ============================================================

def split_indices_by_geometry(
    df: pd.DataFrame,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Split by geometry_id, never by rendered frame.

    This avoids leakage of the same implicit B-spline shape across train,
    validation, and test through other camera views, materials, sigmas,
    or render modes.
    """
    total_fraction = train_fraction + val_fraction + test_fraction

    if not np.isclose(total_fraction, 1.0):
        raise ValueError(
            "train_fraction + val_fraction + test_fraction must equal 1. "
            f"Received {total_fraction}."
        )

    geometry_ids = np.sort(
        df["geometry_id"].astype(np.int64).unique()
    )

    if len(geometry_ids) < 3:
        raise RuntimeError(
            f"Need at least 3 geometries for train/val/test; "
            f"found {len(geometry_ids)}."
        )

    rng = np.random.default_rng(seed)
    shuffled_geometry_ids = geometry_ids.copy()
    rng.shuffle(shuffled_geometry_ids)

    n_geometries = len(shuffled_geometry_ids)

    n_train = int(round(train_fraction * n_geometries))
    n_val = int(round(val_fraction * n_geometries))

    # Guarantee at least one geometry in each split.
    n_train = max(1, min(n_train, n_geometries - 2))
    n_val = max(1, min(n_val, n_geometries - n_train - 1))

    train_geometries = set(shuffled_geometry_ids[:n_train].tolist())
    val_geometries = set(
        shuffled_geometry_ids[n_train:n_train + n_val].tolist()
    )
    test_geometries = set(
        shuffled_geometry_ids[n_train + n_val:].tolist()
    )

    geometry_values = df["geometry_id"].astype(np.int64).to_numpy()

    split_indices = {
        "train": np.flatnonzero(
            np.isin(geometry_values, list(train_geometries))
        ).astype(np.int64),
        "val": np.flatnonzero(
            np.isin(geometry_values, list(val_geometries))
        ).astype(np.int64),
        "test": np.flatnonzero(
            np.isin(geometry_values, list(test_geometries))
        ).astype(np.int64),
    }

    # Strong leakage check.
    if train_geometries & val_geometries:
        raise RuntimeError("Geometry overlap between train and validation.")

    if train_geometries & test_geometries:
        raise RuntimeError("Geometry overlap between train and test.")

    if val_geometries & test_geometries:
        raise RuntimeError("Geometry overlap between validation and test.")

    print("Geometry-level split:")
    print(
        f"  train: {len(train_geometries):,} geometries, "
        f"{len(split_indices['train']):,} frames"
    )
    print(
        f"  val:   {len(val_geometries):,} geometries, "
        f"{len(split_indices['val']):,} frames"
    )
    print(
        f"  test:  {len(test_geometries):,} geometries, "
        f"{len(split_indices['test']):,} frames"
    )

    return split_indices


def training_feature_statistics(
    raw_features: np.ndarray,
    train_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute normalization statistics using training examples only.
    """
    train_features = raw_features[train_indices]

    mean = train_features.mean(axis=0).astype(np.float32)
    std = train_features.std(axis=0).astype(np.float32)

    # Constant features have no useful variance. Keep their normalized
    # value at zero rather than dividing by zero.
    std[std < 1e-6] = 1.0

    return mean, std


# ============================================================
# MEMORY-MAPPED SHARD DATASET
# ============================================================

class ImplicitRenderDataset(Dataset):
    """
    Dataset returning:

        params:      [D]
        target_rgba: [4, H, W]
        geometry_id: scalar long tensor
        sample_id:   scalar long tensor

    Target is premultiplied RGBA:

        [alpha*R, alpha*G, alpha*B, alpha]

    RGB and alpha NPY shard files are opened using mmap_mode='r'.
    """

    def __init__(
        self,
        frame_table: pd.DataFrame,
        raw_features: np.ndarray,
        row_indices: np.ndarray,
        param_mean: np.ndarray,
        param_std: np.ndarray,
        renders_dir: Path,
        alpha_dir: Path,
        image_height: int = 32,
        image_width: int = 32,
    ):
        self.frame_table = frame_table
        self.raw_features = raw_features
        self.row_indices = np.asarray(row_indices, dtype=np.int64)

        self.param_mean = np.asarray(param_mean, dtype=np.float32)
        self.param_std = np.asarray(param_std, dtype=np.float32)

        self.renders_dir = Path(renders_dir)
        self.alpha_dir = Path(alpha_dir)

        self.image_height = int(image_height)
        self.image_width = int(image_width)

        # Convert frequently accessed columns into arrays once.
        self.shard_ids = self.frame_table["shard_id"].astype(str).to_numpy()
        self.idx_in_shard = self.frame_table["idx_in_shard"].to_numpy(
            dtype=np.int64
        )

        self.alpha_shard_ids = (
            self.frame_table["alpha_shard_id"].astype(str).to_numpy()
        )
        self.idx_in_alpha_shard = (
            self.frame_table["idx_in_alpha_shard"].to_numpy(dtype=np.int64)
        )

        self.geometry_ids = self.frame_table["geometry_id"].to_numpy(
            dtype=np.int64
        )
        self.sample_ids = self.frame_table["sample_id"].to_numpy(
            dtype=np.int64
        )

        # Lazy caches. Every DataLoader worker gets its own cache.
        self._rgb_shards: Dict[str, np.ndarray] = {}
        self._alpha_shards: Dict[str, np.ndarray] = {}

    def __getstate__(self):
        """
        Avoid pickling open memmap objects when DataLoader uses spawn workers.
        """
        state = self.__dict__.copy()
        state["_rgb_shards"] = {}
        state["_alpha_shards"] = {}
        return state

    def __len__(self) -> int:
        return len(self.row_indices)

    def _get_rgb_shard(self, shard_id: str) -> np.ndarray:
        if shard_id not in self._rgb_shards:
            path = shard_path_from_id(
                directory=self.renders_dir,
                prefix="images",
                shard_id=shard_id,
                height=self.image_height,
                width=self.image_width,
            )

            if not path.is_file():
                raise FileNotFoundError(
                    f"RGB shard not found for shard_id='{shard_id}':\n{path}"
                )

            self._rgb_shards[shard_id] = np.load(path, mmap_mode="r")

        return self._rgb_shards[shard_id]

    def _get_alpha_shard(self, shard_id: str) -> np.ndarray:
        if shard_id not in self._alpha_shards:
            path = shard_path_from_id(
                directory=self.alpha_dir,
                prefix="alpha",
                shard_id=shard_id,
                height=self.image_height,
                width=self.image_width,
            )

            if not path.is_file():
                raise FileNotFoundError(
                    f"Alpha shard not found for shard_id='{shard_id}':\n{path}"
                )

            self._alpha_shards[shard_id] = np.load(path, mmap_mode="r")

        return self._alpha_shards[shard_id]

    def __getitem__(self, index: int):
        row_index = int(self.row_indices[index])

        rgb_shard_id = self.shard_ids[row_index]
        rgb_index = int(self.idx_in_shard[row_index])

        alpha_shard_id = self.alpha_shard_ids[row_index]
        alpha_index = int(self.idx_in_alpha_shard[row_index])

        rgb_shard = self._get_rgb_shard(rgb_shard_id)
        alpha_shard = self._get_alpha_shard(alpha_shard_id)

        rgb = np.asarray(rgb_shard[rgb_index], dtype=np.float32)
        alpha = np.asarray(alpha_shard[alpha_index], dtype=np.float32)

        expected_rgb_shape = (
            3,
            self.image_height,
            self.image_width,
        )
        expected_alpha_shape = (
            self.image_height,
            self.image_width,
        )

        if rgb.shape != expected_rgb_shape:
            raise RuntimeError(
                f"RGB shape mismatch for shard={rgb_shard_id}, index={rgb_index}. "
                f"Expected {expected_rgb_shape}, got {rgb.shape}."
            )

        if alpha.shape != expected_alpha_shape:
            raise RuntimeError(
                f"Alpha shape mismatch for shard={alpha_shard_id}, "
                f"index={alpha_index}. Expected {expected_alpha_shape}, "
                f"got {alpha.shape}."
            )

        alpha = np.clip(alpha, 0.0, 1.0)
        rgb = np.clip(rgb, 0.0, 1.0)

        alpha_channel = alpha[None, :, :]
        premultiplied_rgb = rgb * alpha_channel

        target_rgba = np.concatenate(
            [premultiplied_rgb, alpha_channel],
            axis=0,
        ).astype(np.float32)

        params = (
            self.raw_features[row_index] - self.param_mean
        ) / self.param_std

        return (
            torch.from_numpy(params.astype(np.float32, copy=False)),
            torch.from_numpy(target_rgba),
            torch.tensor(self.geometry_ids[row_index], dtype=torch.long),
            torch.tensor(self.sample_ids[row_index], dtype=torch.long),
        )