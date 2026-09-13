#!/bin/bash
#SBATCH --job-name=neural_renderer_test_no_checkpoint
#SBATCH --nodes=1
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=6:00:00

cd "$HOME/FNO_Large/gaussian_splatting_copy"

source ./setup_gs_env.sh

python optimize_neural_scene_random_init.py \
    --source_path /home/yuanxiuw/orcd/pool/nerf_synthetic/hotdog \
    --model_path ./output/dense_scene_test \
    --resolution 4 \
    --camera_start 0 \
    --num_cameras 100 \
    --voxel_size 0.2 \
    --min_points_per_voxel 1 \
    --mode_selection_cameras 10 \
    --warmup_iterations 300 \
    --split_interval 10 \
    --max_splits_per_update 10 \
    --prune_interval 100 \
    --contribution_prune_candidates 50 \
    --max_prunes_per_update 10 \
    --prune_selection_cameras 100 \
    --max_slices 2000 \
    --preview_views 2 \
    --fno_batch_size 512 \
    --placement_batch_size 2 \
    --disable_initial_mode_selection \
    --checkpoint_interval 10 \
    --flip_projection_y \
    --max_initial_slices 0 \
    --iterations 240 \
    --resume_checkpoint /home/yuanxiuw/FNO_Large/gaussian_splatting_copy/neural_scene_random_init_outputs/checkpoint_iter_000120_final.pt