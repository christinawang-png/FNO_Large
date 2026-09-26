#!/bin/bash
#
# Replace the SBATCH directives below with your cluster-specific settings.
#
#SBATCH --job-name=implicit_balanced_render
#SBATCH --array=0-47%48
#SBATCH --time=12:00:00
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --output=/orcd/home/002/yuanxiuw/FNO_Large/logs/render_%A_%a.out
#SBATCH --error=/orcd/home/002/yuanxiuw/FNO_Large/logs/render_%A_%a.err

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
# ARRAY SETTINGS
# ============================================================

NUM_TASKS=48
TASK_ID="${SLURM_ARRAY_TASK_ID}"

# A unique identifier used by the Blender script for:
# - temporary PNG directory,
# - output shard filenames,
# - metadata shard filenames.
JOB_ID="balanced_${SLURM_ARRAY_JOB_ID}_${TASK_ID}"

# ============================================================
# VALIDATION
# ============================================================

mkdir -p "${PROJECT_DIR}/logs"

if [[ ! -f "${RENDER_SCRIPT}" ]]; then
    echo "ERROR: Render script does not exist: ${RENDER_SCRIPT}"
    exit 1
fi

if ! command -v "${BLENDER_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Blender executable was not found: ${BLENDER_BIN}"
    exit 1
fi

echo "============================================================"
echo "Balanced implicit B-spline dataset render"
echo "Slurm job ID:       ${SLURM_ARRAY_JOB_ID}"
echo "Slurm array task:   ${TASK_ID}"
echo "Node:               $(hostname)"
echo "Start time:         $(date)"
echo "Blender:            $(command -v "${BLENDER_BIN}")"
echo "Render script:      ${RENDER_SCRIPT}"
echo "Unique job ID:      ${JOB_ID}"
echo "============================================================"

# ============================================================
# RUN BLENDER
# ============================================================

cd "${PROJECT_DIR}"

"${BLENDER_BIN}" \
    -b \
    -P "${RENDER_SCRIPT}" \
    -- \
    --task_id "${TASK_ID}" \
    --num_tasks "${NUM_TASKS}" \
    --job_id "${JOB_ID}"

echo "============================================================"
echo "Finished successfully: $(date)"
echo "============================================================"