#!/bin/bash
#SBATCH --job-name=neural_renderer_test
#SBATCH --nodes=1
#SBATCH --partition=gpu,gpu_h200,seas_gpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=2:00:00
#SBATCH --mail-user=yuanxiuwang@fas.harvard.edu

cd "/n/home13/yuanxiuwang/FNO_Large/gaussian_splatting_copy/"
source "/n/helmod/apps/centos7/Core/Anaconda3/2021.05-jupyterood-fasrc01/x/etc/profile.d/conda.sh"
conda activate /n/home13/yuanxiuwang/.conda/envs/gaussian_splatting_cu124

module load gcc/12.2.0-fasrc01

source ./setup_gs_env.sh

python optimize_neural_scene_random_init.py \
    --source_path /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/yuanxiuwang/treehill \
    --model_path ./output/dense_scene_test \
    --resolution 4 \
    --camera_start 0 \
    --num_cameras 141 \
    --voxel_size 0.2 \
    --min_points_per_voxel 5 \
    --mode_selection_cameras 10 \
    --warmup_iterations 500 \
    --split_interval 10 \
    --max_splits_per_update 10 \
    --prune_interval 100 \
    --contribution_prune_candidates 50 \
    --max_prunes_per_update 10 \
    --prune_selection_cameras 141 \
    --max_slices 2000 \
    --preview_views 2 \
    --fno_batch_size 64 \
    --placement_batch_size 2 \
    --disable_initial_mode_selection \
    --checkpoint_interval 10 \
    --flip_projection_y \
    --max_initial_slices 64 \
    --iterations 200