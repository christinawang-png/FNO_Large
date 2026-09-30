#!/bin/bash
#SBATCH --job-name=neural_renderer_fixed_voxels_eval
#SBATCH --nodes=1
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=3:00:00

cd "$HOME/FNO_Large/gaussian_splatting_copy"

source ./setup_gs_env.sh

python evaluate_fixed_voxel_scene.py \
  --source_path /home/yuanxiuw/orcd/pool/nerf_synthetic_processed/hotdog_black \
  --model_path ./output/scene_rgb_test_fixed_voxels \
  --scene_checkpoint ./fixed_voxel_static_128_splitting_prunning_neighbor_hotdog/checkpoint_iter_000630.pt \
  --surface_checkpoint \
    /orcd/home/002/yuanxiuw/FNO_Large/checkpoints_implicit_modes16_l1/surface/best.pt \
  --volume_checkpoint \
    /orcd/home/002/yuanxiuw/FNO_Large/checkpoints_implicit_modes16_l1/volume/best.pt \
  --use_tile_renderer \
  --flip_projection_y \
  --camera_depth_sign 1 \
  --save_previews 8 \
  --output_dir fixed_voxel_eval_630