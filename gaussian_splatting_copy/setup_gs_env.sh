#!/usr/bin/env bash

# Do not use "set -u" here because some cluster environments
# do not define every variable.
set -eo pipefail

# Resolve the repository location from this file itself.
export GS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# If your cluster requires a CUDA module, uncomment and adjust this.
# module load cuda/12.1

# Use the active conda environment's Python.
export PYTHON="$CONDA_PREFIX/bin/python"
if [ ! -x "$PYTHON" ]; then
    export PYTHON="$(command -v python)"
fi

echo "GS_ROOT=$GS_ROOT"
echo "Python=$PYTHON"

# ------------------------------------------------------------
# CUDA toolkit
# ------------------------------------------------------------

if [ -z "${CUDA_HOME:-}" ]; then
    if command -v nvcc >/dev/null 2>&1; then
        export CUDA_HOME="$(
            dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")"
        )"
    else
        echo "[WARN] nvcc not found; CUDA_HOME was not set."
    fi
fi

if [ -n "${CUDA_HOME:-}" ]; then
    export PATH="$CUDA_HOME/bin:$PATH"
fi

# ------------------------------------------------------------
# Make local 3DGS extensions importable.
# The parent directory of simple_knn/ must be included.
# ------------------------------------------------------------

export PYTHONPATH="$GS_ROOT:$GS_ROOT/submodules/simple-knn:$GS_ROOT/submodules/diff-gaussian-rasterization${PYTHONPATH:+:$PYTHONPATH}"

# ------------------------------------------------------------
# Runtime shared libraries
# ------------------------------------------------------------

TORCH_LIB_DIR="$("$PYTHON" - <<'PY'
import pathlib
import torch

torch_lib = pathlib.Path(torch.__file__).resolve().parent / "lib"
print(torch_lib)
PY
)"

echo "Torch library directory=$TORCH_LIB_DIR"

LIB_DIRS=(
    "$TORCH_LIB_DIR"
)

if [ -n "${CONDA_PREFIX:-}" ]; then
    LIB_DIRS+=("$CONDA_PREFIX/lib")
fi

if [ -n "${CUDA_HOME:-}" ]; then
    LIB_DIRS+=(
        "$CUDA_HOME/lib64"
        "$CUDA_HOME/targets/x86_64-linux/lib"
    )
fi

for d in "${LIB_DIRS[@]}"; do
    if [ -d "$d" ]; then
        export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
done

# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------

echo "CUDA_HOME=${CUDA_HOME:-<unset>}"
echo "PYTHONPATH=$PYTHONPATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

echo
echo "Testing simple_knn import..."

"$PYTHON" - <<'PY'
import sys
import torch

print("Python executable:", sys.executable)
print("Torch version:", torch.__version__)
print("Torch CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

from simple_knn._C import distCUDA2
print("simple_knn import: OK")

from diff_gaussian_rasterization import GaussianRasterizer
print("diff_gaussian_rasterization import: OK")
PY

echo
echo "3DGS environment is ready."