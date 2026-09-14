#!/usr/bin/env python
import math
import sys
from pathlib import Path

import numpy as np
import openvdb as vdb
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DIR = PROJECT_ROOT / "implicit_bspline_dataset" / "volumes"

VOLUME_META_CSV = BASE_DIR / "metadata_volumes.csv"

# Individual VDB files are written here:
#
#   implicit_bspline_dataset/volumes/vdb/volume_000001.vdb
#
VDB_DIR = BASE_DIR / "vdb"
VDB_DIR.mkdir(parents=True, exist_ok=True)

# The generator/exporter keeps values within ±3 sigma:
#
#   V = exp(-0.5 * (SDF / sigma)^2)
#
# At 3 sigma, density is exp(-4.5) ≈ 0.0111.
BAND_SIGMAS = 3.0

# VDB grid name expected by Blender's Volume Info node.
VDB_GRID_NAME = "density"


# ============================================================
# PATH / SHARD LOADING
# ============================================================

def resolve_csv_path(path_from_csv, base_dir: Path) -> Path:
    """
    Resolve an absolute or relative path recorded in metadata CSV.

    Absolute paths are used directly. Relative paths are checked relative
    to BASE_DIR, then relative to BASE_DIR.parent.
    """
    path = Path(str(path_from_csv))

    if path.is_absolute():
        return path

    candidate = base_dir / path
    if candidate.is_file():
        return candidate

    candidate = base_dir.parent / path
    if candidate.is_file():
        return candidate

    return base_dir / path


def load_volume_from_shard(metadata_row):
    """
    Read exactly one volume from an NPY volume shard.

    Expected CSV fields:
        volume_shard_path
        volume_idx_in_shard

    The shard has shape:
        (num_volumes_in_shard, nx, ny, nz)

    mmap_mode='r' avoids loading the entire shard into RAM.
    """
    if "volume_shard_path" not in metadata_row:
        raise KeyError(
            "CSV is missing 'volume_shard_path'. "
            "Use the sharded-volume generation script."
        )

    if "volume_idx_in_shard" not in metadata_row:
        raise KeyError(
            "CSV is missing 'volume_idx_in_shard'. "
            "Use the sharded-volume generation script."
        )

    shard_path = resolve_csv_path(
        metadata_row["volume_shard_path"],
        BASE_DIR,
    )

    if not shard_path.is_file():
        raise FileNotFoundError(f"Volume shard not found: {shard_path}")

    index_in_shard = int(metadata_row["volume_idx_in_shard"])

    # Memory map: do not read the entire e.g. 128 MiB shard at once.
    shard = np.load(shard_path, mmap_mode="r")

    if shard.ndim != 4:
        raise ValueError(
            "Expected volume shard with shape "
            "(num_volumes, nx, ny, nz), "
            f"but got {shard.shape} in {shard_path}"
        )

    if not (0 <= index_in_shard < shard.shape[0]):
        raise IndexError(
            f"volume_idx_in_shard={index_in_shard} is invalid for "
            f"{shard_path}, whose shape is {shard.shape}"
        )

    # Copy only this specific volume into ordinary float32 memory.
    volume = np.asarray(shard[index_in_shard], dtype=np.float32)

    if volume.ndim != 3:
        raise ValueError(
            f"Expected extracted 3D volume, got shape {volume.shape}"
        )

    if not np.all(np.isfinite(volume)):
        raise ValueError(
            f"Volume contains NaN or Inf values: "
            f"{shard_path}, index {index_in_shard}"
        )

    return volume, shard_path, index_in_shard


# ============================================================
# VDB EXPORT
# ============================================================

def volume_to_vdb(volume, vdb_path, grid_name=VDB_GRID_NAME):
    """
    Convert one NumPy volume with axis order (x, y, z) into an OpenVDB
    density grid.

    Assumes samples at array indices:
        (0, 0, 0)       -> world (-0.5, -0.5, -0.5)
        (N-1, N-1, N-1) -> world (+0.5, +0.5, +0.5)

    This matches the implicit B-spline generator and marching-cubes mesh.
    """
    volume = np.asarray(volume, dtype=np.float32)

    if volume.ndim != 3:
        raise ValueError(
            f"Expected 3D volume, received shape {volume.shape}"
        )

    nx, ny, nz = volume.shape

    if nx < 2 or ny < 2 or nz < 2:
        raise ValueError(
            f"All volume dimensions must be >= 2, got {volume.shape}"
        )

    # The original generator uses a cubic [-0.5, 0.5]^3 grid.
    # This restriction can be removed later if you use anisotropic grids.
    if not (nx == ny == nz):
        raise ValueError(
            f"Expected a cubic volume, received shape {volume.shape}"
        )

    density_threshold = math.exp(-0.5 * BAND_SIGMAS**2)

    # Sparse VDB: only explicitly store density near the implicit surface.
    mask = volume >= density_threshold

    if not np.any(mask):
        print(f"  [SKIP] No density values >= {density_threshold:.6f}")
        return False

    active_indices = np.argwhere(mask)

    print(f"  volume shape: {volume.shape}")
    print(f"  density threshold: {density_threshold:.6f}")
    print(f"  active voxels: {len(active_indices):,} / {volume.size:,}")

    grid = vdb.FloatGrid()
    grid.name = grid_name

    try:
        grid.setGridClass(vdb.GridClass.FOG_VOLUME)
    except (AttributeError, TypeError):
        # Some OpenVDB Python bindings do not expose this interface.
        pass

    # Samples include both -0.5 and +0.5, so spacing is 1/(N-1).
    voxel_size = 1.0 / float(nx - 1)

    grid.transform = vdb.createLinearTransform(
        voxelSize=voxel_size,
    )

    # Array index (0,0,0) maps to world (-0.5,-0.5,-0.5).
    grid.transform.postTranslate((-0.5, -0.5, -0.5))

    accessor = grid.getAccessor()

    # NumPy axes are (x, y, z), which matches VDB integer coordinates
    # passed here as (i, j, k).
    for i, j, k in active_indices:
        accessor.setValueOn(
            (int(i), int(j), int(k)),
            float(volume[i, j, k]),
        )

    vdb_path = Path(vdb_path)
    vdb_path.parent.mkdir(parents=True, exist_ok=True)

    vdb.write(
        str(vdb_path),
        grids=[grid],
    )

    print(f"  wrote: {vdb_path}")
    return True


# ============================================================
# METADATA ROW SELECTION
# ============================================================

def parse_cli_args():
    """
    Optional usage:

      python export_to_vdb_shards.py

      python export_to_vdb_shards.py --start_id 1 --end_id 100

      python export_to_vdb_shards.py --task_id 0 --num_tasks 48

    The task mode is preferable for a Slurm array because every row is
    distributed approximately evenly, without assuming sample IDs are dense.
    """
    argv = sys.argv[1:]

    start_id = None
    end_id = None
    task_id = None
    num_tasks = None
    overwrite = False

    if "--start_id" in argv:
        start_id = int(argv[argv.index("--start_id") + 1])

    if "--end_id" in argv:
        end_id = int(argv[argv.index("--end_id") + 1])

    if "--task_id" in argv:
        task_id = int(argv[argv.index("--task_id") + 1])

    if "--num_tasks" in argv:
        num_tasks = int(argv[argv.index("--num_tasks") + 1])

    if "--overwrite" in argv:
        overwrite = True

    if (task_id is None) != (num_tasks is None):
        raise ValueError(
            "Use --task_id and --num_tasks together, or omit both."
        )

    if task_id is not None:
        if num_tasks <= 0:
            raise ValueError(
                f"num_tasks must be positive, received {num_tasks}"
            )

        if not (0 <= task_id < num_tasks):
            raise ValueError(
                f"task_id must lie in [0, {num_tasks - 1}], got {task_id}"
            )

    return start_id, end_id, task_id, num_tasks, overwrite


def select_rows(df, start_id, end_id, task_id, num_tasks):
    """Filter by sample IDs and/or distribute rows across Slurm tasks."""
    if start_id is not None:
        df = df[df["sample_id"] >= start_id]

    if end_id is not None:
        df = df[df["sample_id"] <= end_id]

    # Sort before slicing so task assignment is stable.
    df = df.sort_values("sample_id").reset_index(drop=True)

    if task_id is not None:
        df = df.iloc[task_id::num_tasks].copy()

    return df


# ============================================================
# MAIN
# ============================================================

def main():
    (
        start_id,
        end_id,
        task_id,
        num_tasks,
        overwrite,
    ) = parse_cli_args()

    if not VOLUME_META_CSV.is_file():
        raise FileNotFoundError(
            f"Metadata CSV not found: {VOLUME_META_CSV}"
        )

    df = pd.read_csv(VOLUME_META_CSV)

    required_columns = {
        "sample_id",
        "volume_shard_path",
        "volume_idx_in_shard",
        "sigma",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise KeyError(
            "Metadata CSV is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df = select_rows(
        df=df,
        start_id=start_id,
        end_id=end_id,
        task_id=task_id,
        num_tasks=num_tasks,
    )

    task_label = (
        f"{task_id}/{num_tasks - 1}"
        if task_id is not None
        else "single-process"
    )

    print("=" * 70)
    print("Exporting sharded NumPy volumes to VDB")
    print(f"Metadata: {VOLUME_META_CSV}")
    print(f"Output VDB directory: {VDB_DIR}")
    print(f"Rows assigned: {len(df):,}")
    print(f"Task: {task_label}")
    print(f"Overwrite existing VDBs: {overwrite}")
    print("=" * 70)

    successful = 0
    skipped_existing = 0
    failed = 0

    for row_number, (_, row) in enumerate(df.iterrows(), start=1):
        sample_id = int(row["sample_id"])
        sigma = float(row["sigma"])

        # Stable, single-file-per-volume naming.
        vdb_path = VDB_DIR / f"volume_{sample_id:06d}.vdb"

        if vdb_path.is_file() and not overwrite:
            skipped_existing += 1

            if row_number % 100 == 0:
                print(
                    f"[{row_number}/{len(df)}] "
                    f"sample_id={sample_id}: already exists; skipped"
                )

            continue

        try:
            volume, shard_path, index_in_shard = load_volume_from_shard(row)

            print(
                f"[{row_number}/{len(df)}] "
                f"sample_id={sample_id}, "
                f"sigma={sigma:.4f}, "
                f"shard={shard_path.name}[{index_in_shard}]"
            )

            wrote = volume_to_vdb(
                volume=volume,
                vdb_path=vdb_path,
                grid_name=VDB_GRID_NAME,
            )

            if wrote:
                successful += 1

        except Exception as exc:
            failed += 1
            print(
                f"[WARN] sample_id={sample_id}: "
                f"{type(exc).__name__}: {exc}"
            )

    print("=" * 70)
    print("VDB export complete.")
    print(f"Written: {successful:,}")
    print(f"Skipped existing: {skipped_existing:,}")
    print(f"Failed: {failed:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()