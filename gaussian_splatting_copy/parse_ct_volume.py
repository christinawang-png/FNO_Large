#!/usr/bin/env python
"""
Preprocess a headerless CT/scalar RAW volume into:

  1. A marching-cubes mesh.
  2. A fixed shared scalar/control grid.
  3. A list of active trilinear cells whose eight corner values contain
     an F=0 sign change.

Each active cell can later be rendered by your surface FNO using its eight
shared grid values as the local 2x2x2 trilinear control lattice.

IMPORTANT:
A .raw file has no metadata. You MUST configure:
  - RAW_SHAPE_ZYX
  - RAW_DTYPE
  - VOXEL_SPACING_ZYX
  - ISO_VALUE
  - whether foreground is >= ISO_VALUE or <= ISO_VALUE

Dependencies:
    pip install numpy scipy scikit-image
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt
from skimage.measure import marching_cubes

# ============================================================
# USER CONFIGURATION: CHANGE THESE FOR YOUR DATASET
# ============================================================

RAW_PATH = Path("./tooth_103x94x161_uint8.raw")
OUTPUT_DIR = Path("./ct_preprocessed")

# Raw storage shape. Most medical/scalar raw files are stored as [z, y, x].
#
# Example:
# RAW_SHAPE_ZYX = (300, 512, 512)
RAW_SHAPE_ZYX = (161, 94, 103)

VOLUME_ORIGIN_XYZ = (-51.5, -47.0, -80.5)


# Examples:
#   np.dtype("<u1")  # 8-bit unsigned
#   np.dtype("<u2")  # 16-bit unsigned little endian
#   np.dtype("<i2")  # 16-bit signed little endian
#   np.dtype(">u2")  # 16-bit unsigned big endian
RAW_DTYPE = np.dtype("<u1")

# Physical voxel spacing in the SAME order as RAW_SHAPE_ZYX:
#     (spacing_z, spacing_y, spacing_x)
#
# Example for a CT with anisotropic slices:
# VOXEL_SPACING_ZYX = (1.0, 0.7, 0.7)
VOXEL_SPACING_ZYX = (1.0, 1.0, 1.0)

# Isosurface threshold in the RAW intensity/value scale.
#
# For a binary segmentation volume this may be 0.5 or 1.
# For a CT volume it depends on its scaling / stored HU values.
ISO_VALUE = 128.0

# True:
#     inside / foreground = volume >= ISO_VALUE
#
# False:
#     inside / foreground = volume <= ISO_VALUE
FOREGROUND_IS_GREATER_EQUAL = True

# ------------------------------------------------------------
# Fixed shared-grid resolution
# ------------------------------------------------------------
#
# Number of CELLS, not vertices.
#
# This creates:
#     (GRID_CELLS_Z + 1, GRID_CELLS_Y + 1, GRID_CELLS_X + 1)
#
# shared scalar/control vertices.
GRID_CELLS_Z = 32
GRID_CELLS_Y = 32
GRID_CELLS_X = 32

# ------------------------------------------------------------
# Canonical scene scaling
# ------------------------------------------------------------
#
# The mesh and grid are centered and uniformly scaled so the largest scene
# side becomes this many world units.
#
# Your original FNO training cube has side length 1.0. A useful first test is
# to choose a cell size near 1.0 after normalization, but that can create many
# cells for a full CT volume. Start smaller/coarser if needed.
TARGET_LONGEST_SIDE = 8.0

# ------------------------------------------------------------
# Control-value mapping
# ------------------------------------------------------------
#
# We map the world-space SDF to the FNO control range [-0.5, 0.5].
#
# A value at distance +CONTROL_BAND_WORLD maps to +0.5.
# A value at distance -CONTROL_BAND_WORLD maps to -0.5.
#
# Because the zero level set is unchanged by positive scaling, this controls
# only the field magnitude/conditioning range, not the extracted geometry.
#
# If None, choose 1.5 * the largest final fixed-cell side length.
CONTROL_BAND_WORLD = None

# Optional: quantize controls to the original four FNO training levels.
#
# Leave False for the most geometrically faithful CT approximation.
# Set True only if you specifically want to test in-distribution discrete
# control values. Quantization changes the reconstructed zero set.
QUANTIZE_TO_TRAINING_LEVELS = False

TRAINING_CONTROL_LEVELS = np.array(
    [-0.5, -1.0 / 6.0, 1.0 / 6.0, 0.5],
    dtype=np.float32,
)


# ============================================================
# RAW LOADING
# ============================================================

def load_raw_volume(
    raw_path: Path,
    shape_zyx,
    dtype: np.dtype,
) -> np.ndarray:
    """Load a headerless RAW file with explicit known shape and dtype."""
    raw_path = Path(raw_path)

    if not raw_path.is_file():
        raise FileNotFoundError(f"RAW file not found: {raw_path}")

    shape_zyx = tuple(int(v) for v in shape_zyx)
    expected_count = int(np.prod(shape_zyx))

    flat = np.fromfile(raw_path, dtype=dtype)

    if flat.size != expected_count:
        expected_bytes = expected_count * dtype.itemsize
        actual_bytes = raw_path.stat().st_size

        raise ValueError(
            "RAW file size does not match the configured shape/dtype.\n"
            f"  path:           {raw_path}\n"
            f"  configured:     shape={shape_zyx}, dtype={dtype}\n"
            f"  expected voxels:{expected_count:,}\n"
            f"  found voxels:   {flat.size:,}\n"
            f"  expected bytes: {expected_bytes:,}\n"
            f"  actual bytes:   {actual_bytes:,}\n"
            "Check RAW_SHAPE_ZYX, RAW_DTYPE, byte order, and whether the "
            "dataset contains a separate header file."
        )

    return flat.reshape(shape_zyx)


# ============================================================
# SDF / MESH
# ============================================================

def make_signed_distance(
    volume_zyx: np.ndarray,
    iso_value: float,
    spacing_zyx,
    foreground_is_greater_equal: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct a signed distance field in physical/world units.

    Convention:
        negative = inside / foreground
        positive = outside / background
        zero     = extracted boundary
    """
    if foreground_is_greater_equal:
        inside = volume_zyx >= float(iso_value)
    else:
        inside = volume_zyx <= float(iso_value)

    if not np.any(inside):
        raise RuntimeError(
            "No foreground voxels found. Check ISO_VALUE and "
            "FOREGROUND_IS_GREATER_EQUAL."
        )

    if np.all(inside):
        raise RuntimeError(
            "Every voxel is foreground. Check ISO_VALUE and "
            "FOREGROUND_IS_GREATER_EQUAL."
        )

    distance_to_inside = distance_transform_edt(
        ~inside,
        sampling=spacing_zyx,
    )

    distance_to_outside = distance_transform_edt(
        inside,
        sampling=spacing_zyx,
    )

    # Negative inside; positive outside.
    sdf_zyx = (
        distance_to_inside - distance_to_outside
    ).astype(np.float32)

    return sdf_zyx, inside


def extract_mesh_xyz(
    inside_zyx: np.ndarray,
    spacing_zyx,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract a mesh from a binary inside/outside mask.

    Input array order: Z, Y, X
    Output vertices: X, Y, Z
    """
    verts_zyx, faces, _, _ = marching_cubes(
        inside_zyx.astype(np.float32),
        level=0.5,
        spacing=tuple(float(v) for v in spacing_zyx),
    )

    # skimage returns coordinates in array-axis order: Z, Y, X.
    verts_xyz = verts_zyx[:, [2, 1, 0]].astype(np.float32)

    return verts_xyz, faces.astype(np.int32)


# ============================================================
# CANONICAL NORMALIZATION
# ============================================================

def normalize_geometry(
    mesh_verts_xyz: np.ndarray,
    sdf_zyx: np.ndarray,
    spacing_zyx,
    target_longest_side: float,
):
    """
    Center mesh/grid and uniformly scale them to a canonical world extent.

    Returns:
        normalized mesh vertices;
        normalized XYZ coordinate axes for the SDF grid;
        scale factor;
        original center.
    """
    nz, ny, nx = sdf_zyx.shape
    spacing_z, spacing_y, spacing_x = map(float, spacing_zyx)

    z_coords = np.arange(nz, dtype=np.float32) * spacing_z
    y_coords = np.arange(ny, dtype=np.float32) * spacing_y
    x_coords = np.arange(nx, dtype=np.float32) * spacing_x

    scene_min_xyz = np.array(
        [x_coords[0], y_coords[0], z_coords[0]],
        dtype=np.float32,
    )

    scene_max_xyz = np.array(
        [x_coords[-1], y_coords[-1], z_coords[-1]],
        dtype=np.float32,
    )

    scene_center_xyz = 0.5 * (scene_min_xyz + scene_max_xyz)
    scene_extent_xyz = scene_max_xyz - scene_min_xyz

    longest_side = float(scene_extent_xyz.max())

    if longest_side <= 0.0:
        raise RuntimeError("Invalid zero-sized volume extent.")

    scale = float(target_longest_side) / longest_side

    normalized_mesh = (
        mesh_verts_xyz - scene_center_xyz[None, :]
    ) * scale

    normalized_x = (x_coords - scene_center_xyz[0]) * scale
    normalized_y = (y_coords - scene_center_xyz[1]) * scale
    normalized_z = (z_coords - scene_center_xyz[2]) * scale

    return (
        normalized_mesh.astype(np.float32),
        normalized_x.astype(np.float32),
        normalized_y.astype(np.float32),
        normalized_z.astype(np.float32),
        scale,
        scene_center_xyz.astype(np.float32),
    )


# ============================================================
# SHARED CONTROL GRID
# ============================================================

def quantize_controls(values: np.ndarray) -> np.ndarray:
    """Snap values to the four original dataset control levels."""
    distances = np.abs(
        values[..., None] - TRAINING_CONTROL_LEVELS[None, ...]
    )

    closest_indices = np.argmin(distances, axis=-1)

    return TRAINING_CONTROL_LEVELS[closest_indices]


def sample_shared_control_grid(
    sdf_zyx: np.ndarray,
    normalized_x: np.ndarray,
    normalized_y: np.ndarray,
    normalized_z: np.ndarray,
    cells_zyx,
    control_band_world: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample the SDF at fixed shared grid vertices.

    Returns:
        controls_zyx:
            shape [cells_z + 1, cells_y + 1, cells_x + 1], values in [-0.5, .5]

        grid_x, grid_y, grid_z:
            normalized world coordinate arrays for control vertices.
    """
    cells_z, cells_y, cells_x = map(int, cells_zyx)

    grid_z = np.linspace(
        normalized_z[0],
        normalized_z[-1],
        cells_z + 1,
        dtype=np.float32,
    )

    grid_y = np.linspace(
        normalized_y[0],
        normalized_y[-1],
        cells_y + 1,
        dtype=np.float32,
    )

    grid_x = np.linspace(
        normalized_x[0],
        normalized_x[-1],
        cells_x + 1,
        dtype=np.float32,
    )

    # The stored SDF uses Z,Y,X array order.
    sdf_interpolator = RegularGridInterpolator(
        points=(normalized_z, normalized_y, normalized_x),
        values=sdf_zyx,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    zz, yy, xx = np.meshgrid(
        grid_z,
        grid_y,
        grid_x,
        indexing="ij",
    )

    query_zyx = np.stack(
        [zz.reshape(-1), yy.reshape(-1), xx.reshape(-1)],
        axis=1,
    )

    sampled_sdf = sdf_interpolator(query_zyx).reshape(
        cells_z + 1,
        cells_y + 1,
        cells_x + 1,
    )

    # The source SDF units must be scaled by the same geometry-normalization
    # factor. Here normalized_x etc. already use that scale, but sdf_zyx
    # itself remains in original physical units. We infer the scale ratio.
    original_x_span = float(normalized_x[-1] - normalized_x[0])
    # normalized span is already in final units; use coordinate relation below.
    # SDF scaling is handled outside this function by the caller.

    controls = np.clip(
        sampled_sdf / (2.0 * float(control_band_world)),
        -0.5,
        0.5,
    ).astype(np.float32)

    return controls, grid_x, grid_y, grid_z


def active_cells_from_shared_grid(
    control_grid_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find cells containing a sign change among their eight shared corners.

    Returns:
        active_cell_indices_zyx: [N, 3]
        active_cell_controls:    [N, 8]

    The control order is intentionally compatible with your training code:

        ctrl_0_0_0, ctrl_0_0_1, ctrl_0_1_0, ctrl_0_1_1,
        ctrl_1_0_0, ctrl_1_0_1, ctrl_1_1_0, ctrl_1_1_1

    where array axes are [z, y, x].
    """
    nz = control_grid_zyx.shape[0] - 1
    ny = control_grid_zyx.shape[1] - 1
    nx = control_grid_zyx.shape[2] - 1

    active_indices = []
    active_controls = []

    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                corners = control_grid_zyx[
                    z:z + 2,
                    y:y + 2,
                    x:x + 2,
                ]

                if float(corners.min()) < 0.0 < float(corners.max()):
                    active_indices.append((z, y, x))
                    active_controls.append(corners.reshape(-1))

    if not active_indices:
        return (
            np.empty((0, 3), dtype=np.int32),
            np.empty((0, 8), dtype=np.float32),
        )

    return (
        np.asarray(active_indices, dtype=np.int32),
        np.asarray(active_controls, dtype=np.float32),
    )


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("RAW CT / scalar volume preprocessing")
    print("RAW path:", RAW_PATH)
    print("RAW shape [Z,Y,X]:", RAW_SHAPE_ZYX)
    print("RAW dtype:", RAW_DTYPE)
    print("Voxel spacing [Z,Y,X]:", VOXEL_SPACING_ZYX)
    print("Iso value:", ISO_VALUE)
    print("=" * 72)

    volume_zyx = load_raw_volume(
        raw_path=RAW_PATH,
        shape_zyx=RAW_SHAPE_ZYX,
        dtype=RAW_DTYPE,
    )

    print(
        "Volume range:",
        float(volume_zyx.min()),
        "to",
        float(volume_zyx.max()),
    )
    
    print(
        "Unique volume values:",
        np.unique(volume_zyx)[:],
    )
    
    print(
        "Percentiles:",
        np.percentile(
            volume_zyx,
            [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100],
        ),
    )
    
    histogram, bin_edges = np.histogram(
        volume_zyx,
        bins=256,
        range=(0, 256),
    )
    
    print("Most frequent intensity values:")
    for index in np.argsort(histogram)[-20:][::-1]:
        print(
            f"  value={index:3d}, "
            f"count={histogram[index]:,}"
        )

    sdf_zyx_original_units, inside = make_signed_distance(
        volume_zyx=volume_zyx,
        iso_value=ISO_VALUE,
        spacing_zyx=VOXEL_SPACING_ZYX,
        foreground_is_greater_equal=FOREGROUND_IS_GREATER_EQUAL,
    )

    print(
        "Foreground fraction:",
        f"{float(inside.mean()):.6f}",
    )

    mesh_verts_xyz, mesh_faces = extract_mesh_xyz(
        inside_zyx=inside,
        spacing_zyx=VOXEL_SPACING_ZYX,
    )

    (
        mesh_verts_normalized_xyz,
        normalized_x,
        normalized_y,
        normalized_z,
        normalization_scale,
        original_center_xyz,
    ) = normalize_geometry(
        mesh_verts_xyz=mesh_verts_xyz,
        sdf_zyx=sdf_zyx_original_units,
        spacing_zyx=VOXEL_SPACING_ZYX,
        target_longest_side=TARGET_LONGEST_SIDE,
    )

    # The SDF is in original physical units, so apply the same world scaling
    # used for vertices/coordinates.
    sdf_zyx = (
        sdf_zyx_original_units * normalization_scale
    ).astype(np.float32)

    final_extent_xyz = np.array(
        [
            normalized_x[-1] - normalized_x[0],
            normalized_y[-1] - normalized_y[0],
            normalized_z[-1] - normalized_z[0],
        ],
        dtype=np.float32,
    )

    cells_zyx = (
        GRID_CELLS_Z,
        GRID_CELLS_Y,
        GRID_CELLS_X,
    )

    cell_size_xyz = final_extent_xyz / np.array(
        [GRID_CELLS_X, GRID_CELLS_Y, GRID_CELLS_Z],
        dtype=np.float32,
    )

    if CONTROL_BAND_WORLD is None:
        control_band_world = 1.5 * float(cell_size_xyz.max())
    else:
        control_band_world = float(CONTROL_BAND_WORLD)

    print("Normalized scene extent XYZ:", final_extent_xyz)
    print("Approx. cell size XYZ:", cell_size_xyz)
    print("Control SDF band:", control_band_world)

    controls_zyx, grid_x, grid_y, grid_z = sample_shared_control_grid(
        sdf_zyx=sdf_zyx,
        normalized_x=normalized_x,
        normalized_y=normalized_y,
        normalized_z=normalized_z,
        cells_zyx=cells_zyx,
        control_band_world=control_band_world,
    )

    if QUANTIZE_TO_TRAINING_LEVELS:
        controls_zyx = quantize_controls(controls_zyx).astype(np.float32)
        print("Controls quantized to original four training levels.")

    active_cell_indices_zyx, active_cell_controls = (
        active_cells_from_shared_grid(controls_zyx)
    )

    mesh_path = OUTPUT_DIR / "ct_marching_cubes_mesh.npz"
    grid_path = OUTPUT_DIR / "ct_shared_control_grid.npz"

    np.savez_compressed(
        mesh_path,
        verts_xyz=mesh_verts_normalized_xyz,
        faces=mesh_faces,
        original_center_xyz=original_center_xyz,
        normalization_scale=np.float32(normalization_scale),
    )

    np.savez_compressed(
        grid_path,
        # Shared scalar/control vertices, indexed [z, y, x].
        control_grid_zyx=controls_zyx,

        # Normalized world locations of shared grid vertices.
        grid_x=grid_x,
        grid_y=grid_y,
        grid_z=grid_z,

        # Only these cells contain a surface under the trilinear model.
        active_cell_indices_zyx=active_cell_indices_zyx,
        active_cell_controls=active_cell_controls,

        # Reproducibility / coordinate metadata.
        cells_zyx=np.asarray(cells_zyx, dtype=np.int32),
        target_longest_side=np.float32(TARGET_LONGEST_SIDE),
        control_band_world=np.float32(control_band_world),
        iso_value=np.float32(ISO_VALUE),
        foreground_is_greater_equal=np.bool_(
            FOREGROUND_IS_GREATER_EQUAL
        ),
        normalization_scale=np.float32(normalization_scale),
        original_center_xyz=original_center_xyz,
    )

    print("=" * 72)
    print("Saved mesh:", mesh_path)
    print("  vertices:", f"{len(mesh_verts_normalized_xyz):,}")
    print("  faces:   ", f"{len(mesh_faces):,}")
    print("Saved shared control grid:", grid_path)
    print("  grid shape [Z,Y,X]:", controls_zyx.shape)
    print("  active surface cells:", f"{len(active_cell_indices_zyx):,}")
    print("=" * 72)


if __name__ == "__main__":
    main()