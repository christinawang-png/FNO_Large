#!/usr/bin/env python
"""
generate_bent_planes.py

1. Sample a grid of (p1, p2) bending parameters.
2. For each (p1, p2), define a scalar field V(x,y,z) via a B-spline heightfield
   z = f(x,y).
3. Optionally wrap that in a Gaussian (thickness controlled by sigma).
4. Extract an isosurface using marching cubes.
5. Save:
   - coeffs_XXXX.npy : here we store the B-spline control grid
   - volume_XXXX.npy : scalar field on the grid
   - mesh_XXXX.npz   : verts, faces arrays
   - metadata_volumes.csv describing everything (including p1, p2, sigma)
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

PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = PROJECT_ROOT / "plane_dataset_2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GRID_NX = 64
GRID_NY = 64
GRID_NZ = 64

# Kept only for metadata compatibility
DEG_X = 2
DEG_Y = 2
DEG_Z = 2

# Grid of bending parameters
NUM_P1 = 10
NUM_P2 = 10
P1_MIN, P1_MAX = -1.0, 1.0
P2_MIN, P2_MAX = -1.0, 1.0

# Marching cubes isovalue: for Gaussian field, pick e.g. 0.5
ISOVALUE = 0.5

COEFF_MEAN = 0.0
COEFF_STD = 1.0

# Thickness parameters (Gaussian sigma)
SIGMA_VALUES = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16]

# Control grid resolution for B-spline heightfield
NY_CTRL = 4
NX_CTRL = 4

# ==============================
# B-SPLINE HEIGHTFIELD
# ==============================

def make_bspline_heightfield(control_grid, xs, ys, kx=3, ky=3):
    """
    control_grid: (Ny_ctrl, Nx_ctrl) array of control point heights
    xs: 1D array of x positions in [0,1], length nx
    ys: 1D array of y positions in [0,1], length ny

    Returns f_xy: (nx, ny) B-spline surface evaluated on xs, ys.
    """
    Ny_ctrl, Nx_ctrl = control_grid.shape

    x_ctrl = np.linspace(0.0, 1.0, Nx_ctrl)
    y_ctrl = np.linspace(0.0, 1.0, Ny_ctrl)

    # RectBivariateSpline expects z[x_idx, y_idx], so we pass control_grid.T
    spline = RectBivariateSpline(x_ctrl, y_ctrl, control_grid.T, kx=kx, ky=ky)

    f_xy = spline(xs, ys)  # shape (nx, ny)
    return f_xy


def volume_bent_plane(p1, p2, nx, ny, nz,
                      sigma=0.02,
                      use_signed_dist=True):
    """
    Create a scalar field for a bent plane via a B-spline heightfield.

    If use_signed_dist=True:
        V = d = Z - f(x,y)  (signed "distance" along z; zero-level is the surface)

    If use_signed_dist=False:
        V = density = exp(-0.5 * (d / sigma)^2)  (Gaussian "thickness" around the surface)

    Returns:
        V         : (nx, ny, nz) scalar field
        xs, ys, zs
        control_grid : (NY_CTRL, NX_CTRL) control point heights
    """
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    zs = np.linspace(0.0, 1.0, nz)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    # B-spline heightfield control grid, deterministic from (p1,p2)
    rng = np.random.RandomState(
        seed=int((p1 + 2.0) * 1000 + (p2 + 2.0) * 2000)
    )

    base = 0.3
    ctrl_amp = 0.15
    control_grid = base + ctrl_amp * rng.randn(NY_CTRL, NX_CTRL).astype(np.float32)

    # Slight global trend from p1, p2
    control_grid += 0.05 * p1 * np.linspace(-1, 1, NX_CTRL)[None, :]
    control_grid += 0.05 * p2 * np.linspace(-1, 1, NY_CTRL)[:, None]

    # Evaluate B-spline z = f(x,y)
    f_xy_2d = make_bspline_heightfield(control_grid, xs, ys)  # (nx, ny)
    f_xy = f_xy_2d[:, :, None]  # broadcast along z

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
        "sigma",
    ]

    p1_values = np.linspace(P1_MIN, P1_MAX, NUM_P1)
    p2_values = np.linspace(P2_MIN, P2_MAX, NUM_P2)

    with open(metadata_path, "w", newline="") as f_meta:
        writer = csv.DictWriter(f_meta, fieldnames=fieldnames)
        writer.writeheader()

        sample_id = 0
        for p1 in p1_values:
            for p2 in p2_values:
                for sigma in SIGMA_VALUES:
                    sample_id += 1

                    V, xs, ys, zs, control_grid = volume_bent_plane(
                        p1, p2, GRID_NX, GRID_NY, GRID_NZ,
                        sigma=sigma,
                        use_signed_dist=False,   # Gaussian field
                    )

                    verts, faces, normals, values = marching_cubes(
                        V,
                        level=ISOVALUE,
                        spacing=(
                            1.0 / GRID_NX,
                            1.0 / GRID_NY,
                            1.0 / GRID_NZ,
                        ),
                    )

                    coeff_path = OUTPUT_DIR / f"coeffs_{sample_id:04d}.npy"
                    vol_path   = OUTPUT_DIR / f"volume_{sample_id:04d}.npy"
                    mesh_path  = OUTPUT_DIR / f"mesh_{sample_id:04d}.npz"

                    # store B-spline control grid as coefficients
                    np.save(coeff_path, control_grid.astype(np.float32))
                    np.save(vol_path, V.astype(np.float32))
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
                        "sigma": float(sigma),
                    })

                    print(
                        f"[{sample_id}] p1={p1:.3f}, p2={p2:.3f}, "
                        f"sigma={sigma:.4f} saved coeffs, volume, mesh"
                    )

    print("Done. Data written to", OUTPUT_DIR)


if __name__ == "__main__":
    main()