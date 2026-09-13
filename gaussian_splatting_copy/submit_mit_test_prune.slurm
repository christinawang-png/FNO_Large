#!/bin/bash
#SBATCH --job-name=neural_renderer_test_prune_split_test_sh
#SBATCH --nodes=1
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=1:00:00

cd "$HOME/FNO_Large/gaussian_splatting_copy"

source ./setup_gs_env.sh

python optimize_neural_scene_random_init.py \
    --source_path /home/yuanxiuw/orcd/pool/nerf_synthetic_processed/hotdog \
    --model_path ./output/scene_rgb_test_colmap \
    --output_dir ./output/scene_rgb_test_colmap \
    --resolution 2 \
    --camera_start 0 \
    --num_cameras 400 \
    --cameras_per_step 10 \
    --voxel_size 0.05 \
    --min_points_per_voxel 1 \
    --max_initial_slices 0 \
    --use_visibility_culling \
    --min_projected_patch_size 3 \
    --active_slice_margin 32 \
    --max_active_slices_per_camera 0 \
    --fno_batch_size 512 \
    --disable_initial_mode_selection \
    --split_interval 50 \
    --max_splits_per_update 50 \
    --max_slices 0 \
    --prune_interval 100 \
    --lifecycle_cameras 100 \
    --lifecycle_split_candidates 100 \
    --checkpoint_interval 5 \
    --preview_views 2 \
    --iterations 900 \
    --lifecycle_prune_candidates 50 \
    --max_prunes_per_update 20 \
    --contribution_threshold 0 \
    --prune_selection_cameras 10 \
    --flip_projection_y \
    --warmup_iterations 100 \
    --resume_checkpoint /home/yuanxiuw/FNO_Large/gaussian_splatting_copy/output/scene_rgb_test_colmap/checkpoint_iter_000880.pt \
    --optimize_sh