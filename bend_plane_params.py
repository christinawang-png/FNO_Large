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
from scipy.interpolate import RectBivariateSpline

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

def make_bspline_heightfield(control_grid, xs, ys, kx=3, ky=3):
    """
    control_grid: (Ny_ctrl, Nx_ctrl) array of control point heights
    xs: 1D array of x positions in [0,1], length nx
    ys: 1D array of y positions in [0,1], length ny

    Returns f_xy: (nx, ny) B-spline surface evaluated on xs, ys.
    """
    Ny_ctrl, Nx_ctrl = control_grid.shape

    # parameter positions of control points (uniform in [0,1])
    x_ctrl = np.linspace(0.0, 1.0, Nx_ctrl)
    y_ctrl = np.linspace(0.0, 1.0, Ny_ctrl)

    # create spline
    spline = RectBivariateSpline(x_ctrl, y_ctrl, control_grid.T, kx=kx, ky=ky)
    # NOTE: RectBivariateSpline expects axes order (x, y) with z[x_idx, y_idx],
    # so we supply control_grid.T above.

    # evaluate on full grid
    f_xy = spline(xs, ys)  # shape (nx, ny)
    return f_xy

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

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    # ---- NEW: B-spline height field instead of quadratic ----
    # Build a small control grid of heights (e.g., 4x4) per (p1,p2).
    # Use p1,p2 to seed randomness so it's deterministic.
    rng = np.random.RandomState(seed=int((p1 + 2) * 1000 + (p2 + 2) * 2000))

    Ny_ctrl, Nx_ctrl = 4, 4
    # base height plus small random perturbations
    base = 0.3
    ctrl_amp = 0.15  # how wiggly the surface is
    control_grid = base + ctrl_amp * rng.randn(Ny_ctrl, Nx_ctrl).astype(np.float32)

    # optional: bias control grid slightly with p1,p2 to keep some global trend
    control_grid += 0.05 * p1 * np.linspace(-1, 1, Nx_ctrl)[None, :]
    control_grid += 0.05 * p2 * np.linspace(-1, 1, Ny_ctrl)[:, None]

    # evaluate B-spline surface on (xs, ys)
    f_xy_2d = make_bspline_heightfield(control_grid, xs, ys)  # (nx, ny)
    f_xy = f_xy_2d[:, :, None]  # broadcast over z
    # ---------------------------------------------------------

    # Signed "distance" along z
    d = Z - f_xy

    if use_signed_dist:
        V = d
    else:
        sigma = float(sigma)
        V = np.exp(-0.5 * (d / sigma) ** 2)

    return V, xs, ys, zs, control_grid

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
                V, xs, ys, zs, control_grid = volume_bent_plane(
                    p1, p2, GRID_NX, GRID_NY, GRID_NZ,
                    sigma=0.02,
                    use_signed_dist=True,   # or False if you want Gaussian
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
                # You can still store p1,p2, but also store the control grid as the true shape params
                C = control_grid.astype(np.float32)  # shape (Ny_ctrl, Nx_ctrl)
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