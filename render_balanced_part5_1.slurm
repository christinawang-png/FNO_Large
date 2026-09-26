#!/bin/bash
#
# Replace these SBATCH directives with your cluster-specific account,
# partition, QOS, or node-request syntax as needed.
#
#SBATCH --job-name=implicit_bal_5_1
#SBATCH --time=07:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --output=/orcd/home/002/yuanxiuw/FNO_Large/logs/balanced_part5_1_%j.out
#SBATCH --error=/orcd/home/002/yuanxiuw/FNO_Large/logs/balanced_part5_1_%j.err

set -euo pipefail

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR="/orcd/home/002/yuanxiuw/FNO_Large"

# Change this only if your current renderer has a different name.
RENDER_SCRIPT="${PROJECT_DIR}/blender_render_bernstein.py"

BLENDER_BIN="/orcd/home/002/yuanxiuw/blender-5.1.0-linux-x64/blender"
LOG_DIR="${PROJECT_DIR}/logs"

mkdir -p "${LOG_DIR}"

# ============================================================
# TASK PARTITIONING
# ============================================================

# Number of Blender processes launched inside this Slurm allocation.
NUM_PROCESSES=24

# Must remain 144 because all three original parts together use:
#
#   Part 1: 0-47
#   Part 2: 48-95
#   Part 3: 96-143
#
TOTAL_GLOBAL_TASKS=288

# Part 2.2 owns global task IDs 72 through 95.
GLOBAL_TASK_OFFSET=192

PART_LABEL="part5_1"
SLURM_JOB_LABEL="${SLURM_JOB_ID:-manual}"

# ============================================================
# VALIDATION / STATUS
# ============================================================

if [[ ! -f "${RENDER_SCRIPT}" ]]; then
    echo "ERROR: Blender render script not found:"
    echo "  ${RENDER_SCRIPT}"
    exit 1
fi

if ! command -v "${BLENDER_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Blender executable not found:"
    echo "  ${BLENDER_BIN}"
    exit 1
fi

echo "============================================================"
echo "Balanced implicit B-spline renderer: Part 5.1"
echo "Slurm job ID:       ${SLURM_JOB_LABEL}"
echo "Node:               $(hostname)"
echo "Start time:         $(date)"
echo "Processes:          ${NUM_PROCESSES}"
echo "CPU allocation:     ${SLURM_CPUS_PER_TASK:-24}"
echo "Global task count:  ${TOTAL_GLOBAL_TASKS}"
echo "Task IDs:           192 through 215"
echo "Renderer:           ${RENDER_SCRIPT}"
echo "============================================================"

cd "${PROJECT_DIR}"

# ============================================================
# LAUNCH 24 BLENDER PROCESSES
# ============================================================

pids=()
failed=0

for LOCAL_TASK_ID in $(seq 0 $((NUM_PROCESSES - 1))); do
    GLOBAL_TASK_ID=$((GLOBAL_TASK_OFFSET + LOCAL_TASK_ID))

    UNIQUE_JOB_ID="${PART_LABEL}_${SLURM_JOB_LABEL}_task${GLOBAL_TASK_ID}"

    OUT_LOG="${LOG_DIR}/${PART_LABEL}_${SLURM_JOB_LABEL}_task${GLOBAL_TASK_ID}.out"
    ERR_LOG="${LOG_DIR}/${PART_LABEL}_${SLURM_JOB_LABEL}_task${GLOBAL_TASK_ID}.err"

    echo "Launching Blender global task ${GLOBAL_TASK_ID}"

    "${BLENDER_BIN}" \
        -b \
        -P "${RENDER_SCRIPT}" \
        -- \
        --task_id "${GLOBAL_TASK_ID}" \
        --num_tasks "${TOTAL_GLOBAL_TASKS}" \
        --job_id "${UNIQUE_JOB_ID}" \
        > "${OUT_LOG}" \
        2> "${ERR_LOG}" &

    pids+=("$!")
done

# ============================================================
# WAIT FOR ALL PROCESSES
# ============================================================

for PID in "${pids[@]}"; do
    if ! wait "${PID}"; then
        echo "ERROR: Blender process PID ${PID} failed."
        failed=1
    fi
done

echo "============================================================"
echo "Part 5.1 completed: $(date)"
echo "============================================================"

exit "${failed}"