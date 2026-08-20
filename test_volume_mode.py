#!/usr/bin/env python
import random
from pathlib import Path

import torch

from train_premult_split import PlaneDatasetParamsToPremultRGBA

def main():
    base_dir   = Path("./plane_dataset_4")
    renders_dir = base_dir / "renders"
    alpha_dir   = base_dir / "hard_alpha"

    img_meta_csv   = renders_dir / "metadata_images_all_sharded.csv"
    vol_meta_csv   = base_dir / "metadata_volumes.csv"
    alpha_meta_csv = alpha_dir / "metadata_alpha_all.csv"

    ds = PlaneDatasetParamsToPremultRGBA(
        base_dir=base_dir,
        img_meta_csv=img_meta_csv,
        vol_meta_csv=vol_meta_csv,
        renders_dir=renders_dir,
        alpha_dir=alpha_dir,
        alpha_meta_csv=alpha_meta_csv,
        img_size=(64, 64),
        use_sh=True,
        normalize_params=True,
    )

    print("Total rows in dataset:", len(ds.df))

    # 1) Check render_mode distribution in the merged dataframe
    if "render_mode" not in ds.df.columns:
        print("ERROR: 'render_mode' column missing from dataset.df!")
    else:
        print("render_mode value counts:")
        print(ds.df["render_mode"].value_counts())

    # 2) Sample a bunch of items and print is_volume flags
    print("\nSampling 20 dataset entries to inspect is_volume flags:")
    n = len(ds)
    for i in range(20):
        idx = random.randrange(n)
        _, _, is_volume = ds[idx]
        mode = ds.df.iloc[idx].get("render_mode", "surface")
        print(f"idx={idx:6d}  render_mode={mode:7s}  is_volume={is_volume.item():.1f}")

    # 3) Count how many volume/surface examples actually exist in the dataset
    n_surf = 0
    n_vol  = 0
    for i in range(min(5000, len(ds))):  # just scan up to 5000 for speed
        _, _, is_volume = ds[i]
        if is_volume.item() > 0.5:
            n_vol += 1
        else:
            n_surf += 1

    print(f"\nApprox counts in first {min(5000,len(ds))} samples:")
    print("  surface examples:", n_surf)
    print("  volume  examples:", n_vol)

    if n_vol == 0:
        print("WARNING: no volume examples found in sampled range; volume branch may not be trained.")

if __name__ == "__main__":
    main()