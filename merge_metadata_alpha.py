#!/usr/bin/env python
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

ALPHA_DIR = (
    PROJECT_ROOT
    / "implicit_bspline_dataset"
    / "hard_alpha_balanced"
)

OUT_CSV = ALPHA_DIR / "metadata_alpha_all.csv"


# ============================================================
# MAIN
# ============================================================

def main():
    # Matches files such as:
    #
    # metadata_alpha_part1_1_123456_task0_shard_0000.csv
    #
    # This does not match the merged output file:
    #
    # metadata_alpha_all.csv
    #
    alpha_metadata_files = sorted(
        ALPHA_DIR.glob("metadata_alpha_*_shard_*.csv")
    )

    if not alpha_metadata_files:
        raise FileNotFoundError(
            f"No metadata_alpha_*_shard_*.csv files found in:\n"
            f"  {ALPHA_DIR}"
        )

    print(f"Found {len(alpha_metadata_files):,} alpha metadata files.")

    dataframes = []

    for csv_path in alpha_metadata_files:
        print("Loading:", csv_path.name)

        dataframe = pd.read_csv(csv_path)

        # Useful provenance if an individual shard needs debugging later.
        dataframe["source_metadata_file"] = csv_path.name

        dataframes.append(dataframe)

    alpha_metadata = pd.concat(
        dataframes,
        ignore_index=True,
    )

    print(f"Total merged alpha rows: {len(alpha_metadata):,}")

    # Sort into a convenient deterministic order.
    sort_columns = [
        column
        for column in [
            "sample_id",
            "render_mode",
            "view_idx",
            "alpha_shard_id",
            "idx_in_alpha_shard",
        ]
        if column in alpha_metadata.columns
    ]

    if sort_columns:
        alpha_metadata = alpha_metadata.sort_values(
            sort_columns,
            kind="stable",
        ).reset_index(drop=True)

    # Detect accidental duplicate frames/shard records.
    duplicate_key_columns = [
        column
        for column in [
            "alpha_shard_id",
            "idx_in_alpha_shard",
        ]
        if column in alpha_metadata.columns
    ]

    if len(duplicate_key_columns) == 2:
        duplicate_mask = alpha_metadata.duplicated(
            subset=duplicate_key_columns,
            keep=False,
        )

        num_duplicate_rows = int(duplicate_mask.sum())

        if num_duplicate_rows > 0:
            duplicate_rows = alpha_metadata.loc[
                duplicate_mask,
                duplicate_key_columns + ["source_metadata_file"],
            ]

            duplicate_report_path = (
                ALPHA_DIR / "duplicate_alpha_metadata_rows.csv"
            )

            duplicate_rows.to_csv(
                duplicate_report_path,
                index=False,
            )

            raise RuntimeError(
                f"Found {num_duplicate_rows:,} duplicate alpha metadata rows "
                f"using keys {duplicate_key_columns}.\n"
                f"Duplicate report written to:\n"
                f"  {duplicate_report_path}\n"
                "Do not merge until duplicate/repeated rendering jobs are "
                "resolved."
            )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    alpha_metadata.to_csv(
        OUT_CSV,
        index=False,
    )

    print("=" * 70)
    print(f"Wrote merged alpha metadata: {OUT_CSV}")
    print(f"Rows written: {len(alpha_metadata):,}")
    print(f"Columns: {len(alpha_metadata.columns)}")
    print("=" * 70)


if __name__ == "__main__":
    main()