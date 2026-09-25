#!/bin/bash
#SBATCH --job-name=neural_renderer_test_prune_split_test_sh
#SBATCH --nodes=1
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=6:00:00

cd "$HOME/FNO_Large/gaussian_splatting_copy"

source ./setup_gs_env.sh
    
    
python optimize_neural_scene_random_init.py \
  --source_path /home/yuanxiuw/orcd/pool/nerf_synthetic_processed/hotdog_black \
  --model_path ./output/scene_rgb_test_colmap_black_split \
  --iterations 300 \
  --resolution 1 \
  --num_cameras 400 \
  --cameras_per_step 10 \
  --voxel_size 0.05 \
  --min_points_per_voxel 1 \
  --max_initial_slices 0 \
  --initial_mode random \
  --fno_batch_size 2048 \
  --checkpoint_interval 5 \
  --preview_views 2 \
  --output_dir ./test_neural_scene_split \
  --use_visibility_culling \
  --flip_projection_y \
  --max_slices 0 \
  --split_interval 10 \
  --max_splits_per_update 10 \
  --lifecycle_cameras 10 \
  --lifecycle_split_candidates 50 \
  --prune_interval 20 \
  --prune_selection_cameras 2 \
  --lifecycle_prune_candidates 10 \
  --max_prunes_per_update 10 \
  --contribution_threshold=-1e-06 \
  --optimize_sh \
  --child_selection_cameras 2 \
  --resume_checkpoint test_neural_scene_split/checkpoint_iter_000219_final.pt