#!/bin/bash
#
# Replace these SBATCH directives with your cluster-required syntax.
#
#SBATCH --job-name=implicit_test_512
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/orcd/home/002/yuanxiuw/FNO_Large/logs/test_512_%j.out
#SBATCH --error=/orcd/home/002/yuanxiuw/FNO_Large/logs/test_512_%j.err

set -euo pipefail

# ============================================================
# USER PATHS
# ============================================================

PROJECT_DIR="/orcd/home/002/yuanxiuw/FNO_Large"

# The Blender script from the prior message.
RENDER_SCRIPT="${PROJECT_DIR}/blender_render_bernstein.py"

# Your Blender executable. Replace if needed.
BLENDER_BIN="/orcd/home/002/yuanxiuw/blender-5.1.0-linux-x64/blender"

# ============================================================
# TEST PARAMETERS
# ============================================================

# The test renderer will deterministically select this many geometry IDs.
NUM_TEST_GEOMETRIES=10

# Unique name incorporated into:
# - temporary rendering directory;
# - image shard filenames;
# - metadata filenames.
JOB_ID="test512_${SLURM_JOB_ID}"

# ============================================================
# VALIDATION / LOGGING
# ============================================================

if [[ ! -f "${RENDER_SCRIPT}" ]]; then
    echo "ERROR: Render script not found:"
    echo "  ${RENDER_SCRIPT}"
    exit 1
fi

if ! command -v "${BLENDER_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Blender executable not found:"
    echo "  ${BLENDER_BIN}"
    exit 1
fi

echo "============================================================"
echo "Implicit B-spline balanced renderer: 512-sample test"
echo "Slurm job ID:       ${SLURM_JOB_ID}"
echo "Host:               $(hostname)"
echo "Start:              $(date)"
echo "Blender:            $(command -v "${BLENDER_BIN}")"
echo "Render script:      ${RENDER_SCRIPT}"
echo "Test geometries:    ${NUM_TEST_GEOMETRIES}"
echo "Expected volumes:   ${NUM_TEST_GEOMETRIES} x 5 = 50"
echo "Expected outputs:   50 x 16 views x 2 modes = 1600"
echo "============================================================"

cd "${PROJECT_DIR}"

# ============================================================
# RUN
# ============================================================

time "${BLENDER_BIN}" \
    -b --factory-startup -t 1 \
    -P "${RENDER_SCRIPT}" \
    -- \
    --task_id 0 \
    --num_tasks 1 \
    --job_id "${JOB_ID}" \
    --max_geometries "${NUM_TEST_GEOMETRIES}"

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"