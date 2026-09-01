#!/bin/bash
#SBATCH --job-name=neural_renderer_test
#SBATCH --nodes=1
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=6:00:00

cd "$HOME/FNO_Large/gaussian_splatting_copy"

source ./setup_gs_env.sh

python optimize_neural_scene_random_init.py \
    --source_path /home/yuanxiuw/orcd/pool/3dgs_datas/treehill \
    --model_path ./output/dense_scene_test \
    --resolution 4 \
    --camera_start 0 \
    --num_cameras 141 \
    --voxel_size 0.25 \
    --min_points_per_voxel 10 \
    --mode_selection_cameras 10 \
    --iterations 50 \
    --warmup_iterations 50 \
    --split_interval 10 \
    --max_splits_per_update 10 \
    --prune_interval 500 \
    --contribution_prune_candidates 50 \
    --max_prunes_per_update 10 \
    --prune_selection_cameras 141 \
    --max_slices 612 \
    --preview_views 2 \
    --fno_batch_size 64 \
    --placement_batch_size 12 \
    --disable_initial_mode_selection \
    --max_initial_slices 512 \
    --checkpoint_interval 10 \
    --flip_projection_y