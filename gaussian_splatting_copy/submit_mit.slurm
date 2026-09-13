#!/bin/bash
#SBATCH --job-name=neural_renderer_test_no_checkpoint
#SBATCH --nodes=1
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=2:00:00

cd "$HOME/FNO_Large/gaussian_splatting_copy"

source ./setup_gs_env.sh

python optimize_neural_scene_random_init.py \
    --source_path /home/yuanxiuw/orcd/pool/3dgs_datas/treehill \
    --model_path ./output/dense_scene_test \
    --resolution 4 \
    --camera_start 0 \
    --num_cameras 141 \
    --voxel_size 0.1 \
    --min_points_per_voxel 1 \
    --mode_selection_cameras 10 \
    --warmup_iterations 200 \
    --split_interval 10 \
    --max_splits_per_update 10 \
    --prune_interval 100 \
    --contribution_prune_candidates 50 \
    --max_prunes_per_update 10 \
    --prune_selection_cameras 141 \
    --max_slices 20000 \
    --preview_views 2 \
    --fno_batch_size 512 \
    --placement_batch_size 2 \
    --disable_initial_mode_selection \
    --checkpoint_interval 5 \
    --flip_projection_y \
    --max_initial_slices 0 \
    --iterations 10