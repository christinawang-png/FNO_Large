#!/bin/bash
#SBATCH --job-name=neural_renderer_test_simple_fixed_voxels_splitting
#SBATCH --nodes=1
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=1:00:00

cd "$HOME/FNO_Large/gaussian_splatting_copy"

source ./setup_gs_env.sh

python optimize_fixed_voxel_scene.py \
    --source_path /home/yuanxiuw/orcd/pool/nerf_synthetic_processed/hotdog_black \
    --model_path ./output/scene_rgb_test_fixed_voxels \
    --iterations 2000 \
    --num_cameras 400 \
    --cameras_per_step 1 \
    --voxel_size 0.25 \
    --min_points_per_voxel 10 \
    --max_initial_blocks 128 \
    --fno_batch_size 128 \
    --checkpoint_interval 250 \
    --preview_views 3 \
    --output_dir fixed_voxel_static_128_splitting_prunning_neighbor \
    --flip_projection_y \
    --lifecycle_interval 200 \
    --max_blocks 256 \
    --max_voxel_level 1 \
    --max_splits_per_update 8 \
    --split_candidate_count 10 \
    --split_min_alive_weight 0.25 \
    --split_min_world_size 0 \
    --split_min_shape_gradient 0 \
    --neighbor_activation_interval 400 \
    --max_neighbor_activations_per_update 8 \
    --enable_hard_pruning \
    --dead_alive_threshold 0.02 \
    --dead_intervals_before_prune 2 \
    --max_prunes_per_update 5 \
    --neighbor_activation_interval 300 \
    --prune_interval 500 \
    --split_interval 700 \