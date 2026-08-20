import glob
import pandas as pd
from pathlib import Path

base_dir   = Path("/orcd/home/002/yuanxiuw/FNO_Large/plane_dataset_4")
renders_dir= base_dir / "renders"

# 1) Find only the per-shard CSVs, not the old metadata_images_all.csv
csv_files = glob.glob(str(renders_dir / "metadata_job0_shard_*.csv"))
print("Found shard CSVs:", len(csv_files))

dfs = [pd.read_csv(f) for f in csv_files]
df_all = pd.concat(dfs, ignore_index=True)
print("Total rows in all shards:", len(df_all))

# 2) (Optional) sort by sample_id / shard_id / idx_in_shard
df_all = df_all.sort_values(
    by=["sample_id", "shard_id", "idx_in_shard"]
).reset_index(drop=True)

# 3) Write new merged metadata
out_csv = renders_dir / "metadata_images_all_sharded.csv"
df_all.to_csv(out_csv, index=False)
print("Wrote merged shard metadata to:", out_csv)