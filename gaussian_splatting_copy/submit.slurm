#!/bin/bash
#SBATCH --job-name=neural_renderer_test
#SBATCH --nodes=1
#SBATCH --partition=seas_gpu
#SBATCH --cpus-per-task=1
#SBATCH --gpus=nvidia_h200:1
#SBATCH --mem=10G
#SBATCH --time=12:00:00
#SBATCH --mail-user=yuanxiuwang@fas.harvard.edu

cd "/n/home13/yuanxiuwang/FNO_Large/gaussian_splatting_copy/"
source "/n/helmod/apps/centos7/Core/Anaconda3/2021.05-jupyterood-fasrc01/x/etc/profile.d/conda.sh"
conda activate /n/home13/yuanxiuwang/.conda/envs/gaussian_splatting_cu124

module load gcc/12.2.0-fasrc01

source ./setup_gs_env.sh

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print("current device:", torch.cuda.current_device())
print("device name:", torch.cuda.get_device_name(0))

x = torch.zeros(1, device="cuda")
print("CUDA allocation test: OK", x.device)
PY

python optimize_neural_scene_random_init.py \
    --source_path /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/yuanxiuwang/treehill \
    --model_path ./output/dense_scene_test \
    --resolution 4 \
    --camera_start 0 \
    --num_cameras 141 \
    --voxel_size 0.1 \
    --min_points_per_voxel 1 \
    --mode_selection_cameras 10 \
    --warmup_iterations 300 \
    --split_interval 10 \
    --max_splits_per_update 10 \
    --prune_interval 100 \
    --contribution_prune_candidates 50 \
    --max_prunes_per_update 10 \
    --prune_selection_cameras 141 \
    --max_slices 20000 \
    --preview_views 2 \
    --fno_batch_size 2048 \
    --placement_batch_size 2 \
    --disable_initial_mode_selection \
    --checkpoint_interval 5 \
    --flip_projection_y \
    --max_initial_slices 0 \
    --iterations 260 \
    --resume_checkpoint /n/home13/yuanxiuwang/FNO_Large/gaussian_splatting_copy/neural_scene_random_init_outputs/checkpoint_iter_000200_final.pt