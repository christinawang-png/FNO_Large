#!/bin/bash
#SBATCH --job-name=neural_renderer_test_simple_fixed_voxels_splitting
#SBATCH --nodes=1
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=6:00:00

cd "$HOME/FNO_Large/gaussian_splatting_copy"

source ./setup_gs_env.sh

python optimize_fixed_voxel_scene.py \
    --source_path /home/yuanxiuw/orcd/pool/nerf_synthetic_processed/hotdog_black \
    --model_path ./output/scene_rgb_test_fixed_voxels \
    --iterations 650 \
    --num_cameras 400 \
    --cameras_per_step 10 \
    --voxel_size 0.05 \
    --min_points_per_voxel 1 \
    --max_initial_blocks 0 \
    --fno_batch_size 2048 \
    --checkpoint_interval 5 \
    --preview_views 3 \
    --output_dir fixed_voxel_static_128_splitting_prunning_neighbor_hotdog \
    --flip_projection_y \
    --lifecycle_interval 20 \
    --max_blocks 0 \
    --max_voxel_level 2 \
    --max_splits_per_update 100 \
    --split_candidate_count 200 \
    --split_min_alive_weight 0.1 \
    --split_min_world_size 0 \
    --split_min_shape_gradient 0 \
    --neighbor_activation_interval 20 \
    --neighbor_candidate_count 500 \
    --max_neighbor_activations_per_update 100 \
    --enable_hard_pruning \
    --dead_alive_threshold=2e-2 \
    --dead_intervals_before_prune 2 \
    --max_prunes_per_update 2000 \
    --prune_interval 30 \
    --split_interval 0 \
    --initial_volume_probability 0.1 \
    --camera_depth_sign 1 \
    --optimize_sh \
    --resume_checkpoint fixed_voxel_static_128_splitting_prunning_neighbor_hotdog/checkpoint_iter_000610.pt