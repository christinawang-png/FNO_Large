import glob
import os
from pathlib import Path
import pandas as pd
import numpy as np

base_dir = Path("/orcd/home/002/yuanxiuw/FNO_Large/plane_dataset_3")
vol_csv  = base_dir / "metadata_volumes.csv"
renders_dir = base_dir / "renders"

# 1) volume metadata
df_vol = pd.read_csv(vol_csv)
all_sids = np.sort(df_vol["sample_id"].unique())
print("Total shapes in volume metadata:", len(all_sids))

# 2) read *all* image csv shards
csv_files = glob.glob(str(renders_dir / "metadata_images_*.csv"))
print("Found image CSV shards:", csv_files)

dfs = [pd.read_csv(f) for f in csv_files]
df_img_all = pd.concat(dfs, ignore_index=True)
print("Total image rows across shards:", len(df_img_all))

# 3) per-shape counts across *all* shards
counts = df_img_all["sample_id"].value_counts().sort_index()

IMAGES_PER_SHAPE = 500  # whatever you planned

print("Unique sample_id in all images:", counts.index.nunique())
print("Per-shape image count stats:")
print("  min:", counts.min(), "max:", counts.max(), "mean:", counts.mean())

incomplete = counts[counts < IMAGES_PER_SHAPE]
print("Shapes with < IMAGES_PER_SHAPE images:", len(incomplete))
print(incomplete.head())

missing_sids = np.setdiff1d(all_sids, counts.index.values)
print("Shapes with 0 images (missing):", len(missing_sids))
print("First few missing:", missing_sids)