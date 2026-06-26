#!/usr/bin/env python
"""
generate_bent_planes.py

1. Sample a grid of (p1, p2) bending parameters.
2. For each (p1, p2), define a scalar field V(x,y,z) = z - f(x,y)
   where f(x,y) is a quadratic "bent plane".
3. Extract an isosurface using marching cubes.
4. Save:
   - coeffs_XXXX.npy : here we just store [p1, p2] as a placeholder
   - volume_XXXX.npy : scalar field on the grid
   - mesh_XXXX.npz   : verts, faces arrays
   - metadata_volumes.csv describing everything (including p1, p2)
"""

import os
import csv
import numpy as np
from skimage.measure import marching_cubes
from pathlib import Path

# ==============================
# CONFIGURATION
# ==============================

# Directory where this Python file lives
PROJECT_ROOT = Path(__file__).resolve().parent

# Base output directory
OUTPUT_DIR = PROJECT_ROOT / "plane_dataset_1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Grid resolution for evaluating the scalar field
GRID_NX = 64
GRID_NY = 64
GRID_NZ = 64

# "Degrees" are no longer meaningful here, but we keep them
# for compatibility with the old metadata format.
DEG_X = 2
DEG_Y = 2
DEG_Z = 2

# Grid of bending parameters
NUM_P1 = 10      # number of values for p1
NUM_P2 = 10      # number of values for p2
P1_MIN, P1_MAX = -1.0, 1.0
P2_MIN, P2_MAX = -1.0, 1.0

# Marching cubes isovalue; we use 0 so V(x,y,z) = 0 => z = f(x,y)
ISOVALUE = 0.5

# Placeholder coeff stats (not really used now)
COEFF_MEAN = 0.0
COEFF_STD = 1.0

# ==============================
# BENT PLANE VOLUME
# ==============================

def volume_bent_plane(p1, p2, nx, ny, nz,
                      sigma=0.02,
                      use_signed_dist=True):
    """
    Create a scalar field for a bent plane.

    If use_signed_dist=True:
        V = d = Z - f(x,y)  (signed "distance" along z; zero-level is the surface)

    If use_signed_dist=False:
        V = density = exp(-0.5 * (d / sigma)^2)  (Gaussian "thickness" around the surface)
    """
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    zs = np.linspace(0.0, 1.0, nz)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")  # (nx, ny, nz)

    # Centered coordinates
    x0, y0 = 0.5, 0.5
    fx = X - x0
    fy = Y - y0

    # Quadratic surface:
    base = 0.3
    f_xy = base + p1 * (fx ** 2) + p2 * (fy ** 2)

    # Signed "distance" along z
    d = Z - f_xy

    if use_signed_dist:
        V = d
    else:
        sigma = float(sigma)
        d_pos = np.maximum(d, 0.0)   # only one side of the plane
        V = np.exp(-0.5 * (d_pos / sigma) ** 2)  # (0, 1]

    return V, xs, ys, zs

# ==============================
# MAIN
# ==============================

def main():
    metadata_path = OUTPUT_DIR / "metadata_volumes.csv"
    fieldnames = [
        "sample_id",
        "coeff_path",
        "volume_path",
        "mesh_path",
        "deg_x", "deg_y", "deg_z",
        "grid_nx", "grid_ny", "grid_nz",
        "coeff_mean", "coeff_std",
        "isovalue",
        "p1", "p2",
    ]

    p1_values = np.linspace(P1_MIN, P1_MAX, NUM_P1)
    p2_values = np.linspace(P2_MIN, P2_MAX, NUM_P2)

    with open(metadata_path, "w", newline="") as f_meta:
        writer = csv.DictWriter(f_meta, fieldnames=fieldnames)
        writer.writeheader()

        sample_id = 0
        for p1 in p1_values:
            for p2 in p2_values:
                sample_id += 1

                # 1) build bent-plane volume
                V, xs, ys, zs = volume_bent_plane(
                    p1, p2, GRID_NX, GRID_NY, GRID_NZ,
                    sigma=0.05,          # controls volumetric thickness
                    use_signed_dist=False,
                )

                # 2) marching cubes
                verts, faces, normals, values = marching_cubes(
                    V,
                    level=ISOVALUE,
                    spacing=(1.0 / GRID_NX, 1.0 / GRID_NY, 1.0 / GRID_NZ),
                )
                coeff_path = OUTPUT_DIR / f"coeffs_{sample_id:04d}.npy"
                vol_path   = OUTPUT_DIR / f"volume_{sample_id:04d}.npy"
                mesh_path  = OUTPUT_DIR / f"mesh_{sample_id:04d}.npz"

                # For this dataset, coeffs are just [p1, p2] as a placeholder.
                # Training will use p1, p2 from metadata, not this file.
                C = np.array([p1, p2], dtype=np.float32)
                np.save(coeff_path, C)

                np.save(vol_path, V)
                np.savez(
                    mesh_path,
                    verts=verts.astype(np.float32),
                    faces=faces.astype(np.int32),
                )

                writer.writerow({
                    "sample_id": sample_id,
                    "coeff_path": str(coeff_path),
                    "volume_path": str(vol_path),
                    "mesh_path": str(mesh_path),
                    "deg_x": DEG_X,
                    "deg_y": DEG_Y,
                    "deg_z": DEG_Z,
                    "grid_nx": GRID_NX,
                    "grid_ny": GRID_NY,
                    "grid_nz": GRID_NZ,
                    "coeff_mean": COEFF_MEAN,
                    "coeff_std": COEFF_STD,
                    "isovalue": ISOVALUE,
                    "p1": float(p1),
                    "p2": float(p2),
                })

                print(f"[{sample_id}] p1={p1:.3f}, p2={p2:.3f} saved coeffs, volume, mesh")

    print("Done. Data written to", OUTPUT_DIR)

if __name__ == "__main__":
    main()