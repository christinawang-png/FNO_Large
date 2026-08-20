#!/usr/bin/env python
import numpy as np
import openvdb as vdb
from pathlib import Path
import pandas as pd
import math

BASE_DIR = Path("/orcd/home/002/yuanxiuw/FNO_Large/plane_dataset_4")
VOLUME_META_CSV = BASE_DIR / "metadata_volumes.csv"

def npy_to_vdb(npy_path: Path, vdb_path: Path, grid_name: str = "density"):
    """
    Convert a saved volume_XXXX.npy (float32 [nx,ny,nz]) into an OpenVDB file
    with a single FloatGrid named `grid_name`.
    """

    V = np.load(npy_path).astype(np.float32)
    
    # choose band half-thickness in units of sigma:
    k = 3.0  # keep points within ±3σ from the plane
    v_thr = math.exp(-0.5 * k * k)  # ~0.011 for k=3
    
    band = np.maximum(V - v_thr, 0.0)  # 0 outside band, positive inside
    nx, ny, nz = V.shape

    print(f"  volume shape = {V.shape}")

    # Create empty float grid; background=0
    grid = vdb.FloatGrid()     # or vdb.FloatGrid(0.0)
    grid.name = grid_name

    # Set voxel size so the volume roughly spans [0,1]^3
    # (you can tweak this later if desired)
    voxel_size = 1.0 / nx
    grid.transform = vdb.createLinearTransform(voxelSize=voxel_size)

    acc = grid.getAccessor()

    # Fill only nonzero voxels for sparsity
    # (this is simple but not the fastest; OK for moderate sizes)
    it = np.ndenumerate(V)
    for (i, j, k), val in it:
        if val != 0.0:
            acc.setValueOn((int(i), int(j), int(k)), float(val))

    # Write VDB file
    vdb.write(str(vdb_path), grids=[grid])
    print(f"  wrote {vdb_path}")

def main():
    df = pd.read_csv(VOLUME_META_CSV)

    for _, row in df.iterrows():
        vol_rel = row["volume_path"]  # e.g. "volume_0001.npy"
        vol_npy = BASE_DIR / vol_rel
        vol_vdb = vol_npy.with_suffix(".vdb")

        if not vol_npy.is_file():
            print(f"[WARN] NPY not found: {vol_npy}")
            continue

        #if vol_vdb.is_file():
            #print("Skipping (exists):", vol_vdb)
            #continue

        print("Converting", vol_npy, "->", vol_vdb)
        npy_to_vdb(vol_npy, vol_vdb)

if __name__ == "__main__":
    main()