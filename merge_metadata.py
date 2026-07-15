import glob
import os
from pathlib import Path
import pandas as pd

# Adjust base_dir and renders_dir to match your dataset
base_dir   = Path("/orcd/home/002/yuanxiuw/FNO_Large/plane_dataset_3")
renders_dir= base_dir / "renders"

# 1) Load all CSV shards
csv_files = glob.glob(str(renders_dir / "metadata_images_*.csv"))
print("Found CSV shards:", csv_files)

dfs = [pd.read_csv(f) for f in csv_files]
df_all = pd.concat(dfs, ignore_index=True)
print("Total rows before cleaning:", len(df_all))

# 2) Drop rows whose image file does not exist
exists_mask = df_all["image_path"].apply(os.path.isfile)
print("Rows with existing images:", exists_mask.sum())
print("Rows with missing images:", (~exists_mask).sum())

df_all = df_all[exists_mask].copy()

# 3) Drop duplicate image_path rows, keeping the *last* occurrence
#    (last write in your rendering runs)
before_dups = len(df_all)
df_all = df_all.drop_duplicates(subset="image_path", keep="last")
after_dups = len(df_all)
print("Rows after dropping duplicates by image_path:",
      after_dups, "(removed", before_dups - after_dups, "duplicates)")

# 4) (Optional) sort by sample_id and image_path for sanity
df_all = df_all.sort_values(by=["sample_id", "image_path"]).reset_index(drop=True)

# 5) Write merged CSV
out_csv = renders_dir / "metadata_images_all.csv"
df_all.to_csv(out_csv, index=False)
print("Wrote cleaned metadata to:", out_csv)

# 6) Sanity check: compare against actual PNG count
import subprocess

png_count = int(
    subprocess.check_output(
        ["bash", "-lc", f"find '{renders_dir}' -type f -name '*.png' | wc -l"]
    ).decode("utf-8").strip()
)
print("PNG files on disk:", png_count)
print("Rows in cleaned CSV:", len(df_all))