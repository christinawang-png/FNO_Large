#!/usr/bin/env python
import pandas as pd
from pathlib import Path

BASE_DIR  = Path("plane_dataset_4")
ALPHA_DIR = BASE_DIR / "hard_alpha"

OUT_CSV   = ALPHA_DIR / "metadata_alpha_all.csv"

def main():
    alpha_meta_files = sorted(ALPHA_DIR.glob("metadata_alpha_*_shard_*.csv"))
    if not alpha_meta_files:
        raise FileNotFoundError(f"No metadata_alpha_*_shard_*.csv in {ALPHA_DIR}")

    dfs = []
    for f in alpha_meta_files:
        print("Loading", f)
        df = pd.read_csv(f)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    print("Total rows:", len(df_all))

    df_all.to_csv(OUT_CSV, index=False)
    print("Wrote merged alpha metadata to", OUT_CSV)

if __name__ == "__main__":
    main()