#!/usr/bin/env python
import os
import csv
import itertools
import numpy as np
from skimage.measure import marching_cubes
from pathlib import Path
from scipy.interpolate import RectBivariateSpline

# ==============================
# CONFIGURATION
# ==============================

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR   = PROJECT_ROOT / "plane_dataset_4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GRID_NX = 64
GRID_NY = 64
GRID_NZ = 64

DEG_X = 2
DEG_Y = 2
DEG_Z = 2

# B-spline control grid size
NY_CTRL = 2
NX_CTRL = 2

# control point height range and sampling
CTRL_MAX          = 0.4           # larger control range
NUM_CTRL_LEVELS   = 5              # e.g. [-0.25,-0.125,0,0.125,0.25]
BASE_HEIGHT       = 0.5            # vertical offset

# sigma values (thickness)
SIGMA_VALUES_LARGE = [0.02, 0.08, 0.2, 0.5, 0.7]

COEFF_MEAN  = 0.0
COEFF_STD   = 1.0

# ==============================
# B-SPLINE HEIGHTFIELD
# ==============================

def make_bspline_heightfield(control_grid, xs, ys, kx=1, ky=1):
    """
    control_grid: (Ny_ctrl, Nx_ctrl)
    xs: [nx] in [0,1]
    ys: [ny] in [0,1]
    returns f_xy: (nx, ny)
    """
    Ny_ctrl, Nx_ctrl = control_grid.shape
    x_ctrl = np.linspace(0.0, 1.0, Nx_ctrl)
    y_ctrl = np.linspace(0.0, 1.0, Ny_ctrl)

    spline = RectBivariateSpline(x_ctrl, y_ctrl, control_grid.T, kx=kx, ky=ky)
    f_xy = spline(xs, ys)  # (nx, ny)
    return f_xy

def volume_from_control_grid(control_grid, sigma, nx, ny, nz):
    """
    Given a control_grid (Ny_ctrl, Nx_ctrl) and sigma, build volume V(x,y,z):

        d(x,y,z)  = z - f(x,y)
        V(x,y,z)  = exp(-0.5 * (d / sigma)^2)

    Returns:
        V: (nx,ny,nz), xs, ys, zs
    """
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    zs = np.linspace(0.0, 1.0, nz)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    f_xy_2d = make_bspline_heightfield(control_grid, xs, ys)  # (nx,ny)
    f_xy    = f_xy_2d[:, :, None]                             # (nx,ny,1)

    d = Z - f_xy
    d_pos = np.maximum(d, 0.0)        # or np.minimum(d, 0.0) for the other side
    V = np.exp(-0.5 * (d_pos / sigma)**2)
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
        "sigma",
    ]
    # also flatten control grid into metadata for reference
    for j in range(NY_CTRL):
        for i in range(NX_CTRL):
            fieldnames.append(f"ctrl_{j}_{i}")

    # define discrete levels for each control point
    ctrl_levels = np.linspace(-CTRL_MAX, CTRL_MAX, NUM_CTRL_LEVELS, dtype=np.float32)

    with open(metadata_path, "w", newline="") as f_meta:
        writer = csv.DictWriter(f_meta, fieldnames=fieldnames)
        writer.writeheader()

        sample_id = 0

        # iterate over all control-grid combinations
        # each control point gets a value from ctrl_levels
        for ctrl_values in itertools.product(ctrl_levels, repeat=NY_CTRL * NX_CTRL):
            ctrl_array = np.array(ctrl_values, dtype=np.float32).reshape(NY_CTRL, NX_CTRL)
            # add base height
            control_grid = BASE_HEIGHT + ctrl_array

            for sigma in SIGMA_VALUES_LARGE:
                sample_id += 1

                # 1) build volume
                V, xs, ys, zs = volume_from_control_grid(
                    control_grid, sigma, GRID_NX, GRID_NY, GRID_NZ
                )

                vmin = float(V.min())
                vmax = float(V.max())

                if vmin == vmax:
                    print(f"[SKIP] sample_id={sample_id+1} sigma={sigma:.3f}: constant volume V={vmin:.3e}")
                    continue

                level = vmin + 0.5 * (vmax - vmin)

                verts, faces, normals, values = marching_cubes(
                    V,
                    level=level,
                    spacing=(1.0 / GRID_NX, 1.0 / GRID_NY, 1.0 / GRID_NZ),
                )

                coeff_path = OUTPUT_DIR / f"coeffs_{sample_id:04d}.npy"
                vol_path   = OUTPUT_DIR / f"volume_{sample_id:04d}.npy"
                mesh_path  = OUTPUT_DIR / f"mesh_{sample_id:04d}.npz"

                # store control grid as coefficients
                np.save(coeff_path, control_grid.astype(np.float32))
                np.save(vol_path, V.astype(np.float32))
                np.savez(
                    mesh_path,
                    verts=verts.astype(np.float32),
                    faces=faces.astype(np.int32),
                )

                row = {
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
                    "isovalue": level,
                    "sigma": float(sigma),
                }
                # flatten control grid into metadata
                for j in range(NY_CTRL):
                    for i in range(NX_CTRL):
                        row[f"ctrl_{j}_{i}"] = float(control_grid[j, i])

                writer.writerow(row)

                print(f"[{sample_id}] sigma={sigma:.3f} saved coeffs, volume, mesh")

    print("Done. Data written to", OUTPUT_DIR)

if __name__ == "__main__":
    main()