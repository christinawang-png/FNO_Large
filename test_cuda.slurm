#!/bin/bash
#SBATCH --job-name=test_torch_cuda
#SBATCH --nodes=1
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

# try WITHOUT loading any cuda module first
# module load cuda/13.0.1   # comment out for the first test

cd "$HOME/FNO_Large"

python - << 'EOF'
import os, torch
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.is_available()  ->", torch.cuda.is_available())
print("torch.cuda.device_count()  ->", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0 name:", torch.cuda.get_device_name(0))
EOF