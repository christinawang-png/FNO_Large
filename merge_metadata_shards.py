#!/usr/bin/env python
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

RENDERS_DIR = (
    PROJECT_ROOT
    / "implicit_bspline_dataset"
    / "renders_balanced"
)

OUT_CSV = RENDERS_DIR / "metadata_images_all_sharded.csv"


# ============================================================
# MAIN
# ============================================================

def main():
    # Matches files produced by the balanced renderer, for example:
    #
    # metadata_part1_1_123456_task0_shard_0000.csv
    #
    # It does NOT match:
    #   metadata_images_all_sharded.csv
    #   metadata_alpha_*.csv  (those are in hard_alpha_balanced anyway)
    metadata_files = sorted(
        RENDERS_DIR.glob("metadata_*_shard_*.csv")
    )

    if not metadata_files:
        raise FileNotFoundError(
            f"No metadata_*_shard_*.csv files found in:\n"
            f"  {RENDERS_DIR}"
        )

    print(f"Found {len(metadata_files):,} RGB metadata shard CSV files.")

    dataframes = []

    for csv_path in metadata_files:
        print("Loading:", csv_path.name)

        dataframe = pd.read_csv(csv_path)

        # Retain provenance for debugging a particular shard later.
        dataframe["source_metadata_file"] = csv_path.name

        dataframes.append(dataframe)

    df_all = pd.concat(
        dataframes,
        ignore_index=True,
    )

    print(f"Total rows before sorting: {len(df_all):,}")

    # Sort for convenient inspection/training indexing.
    sort_columns = [
        column
        for column in [
            "sample_id",
            "render_mode",
            "view_idx",
            "shard_id",
            "idx_in_shard",
        ]
        if column in df_all.columns
    ]

    if sort_columns:
        df_all = df_all.sort_values(
            by=sort_columns,
            kind="stable",
        ).reset_index(drop=True)

    # A unique RGB frame should be identified by its shard filename ID
    # plus its array index inside that shard.
    duplicate_key_columns = [
        column
        for column in [
            "shard_id",
            "idx_in_shard",
        ]
        if column in df_all.columns
    ]

    if len(duplicate_key_columns) == 2:
        duplicate_mask = df_all.duplicated(
            subset=duplicate_key_columns,
            keep=False,
        )

        duplicate_count = int(duplicate_mask.sum())

        if duplicate_count > 0:
            duplicate_report = df_all.loc[
                duplicate_mask,
                duplicate_key_columns + [
                    "sample_id",
                    "render_mode",
                    "view_idx",
                    "source_metadata_file",
                ],
            ]

            report_path = RENDERS_DIR / "duplicate_image_metadata_rows.csv"

            duplicate_report.to_csv(
                report_path,
                index=False,
            )

            raise RuntimeError(
                f"Found {duplicate_count:,} duplicate image metadata rows "
                f"using keys {duplicate_key_columns}.\n"
                f"Report written to:\n"
                f"  {report_path}\n"
                "This usually means a rendering job was run twice using "
                "the same job_id/shard naming."
            )

    df_all.to_csv(
        OUT_CSV,
        index=False,
    )

    print("=" * 70)
    print(f"Wrote merged RGB metadata to: {OUT_CSV}")
    print(f"Rows written: {len(df_all):,}")
    print(f"Columns: {len(df_all.columns)}")
    print("=" * 70)


if __name__ == "__main__":
    main()