#!/bin/bash
#SBATCH --job-name=neural_renderer_fixed_voxels_video
#SBATCH --nodes=1
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=3:00:00

cd "$HOME/FNO_Large/gaussian_splatting_copy"

source ./setup_gs_env.sh

python render_fixed_voxel_camera_loop.py \
  --source_path /home/yuanxiuw/orcd/pool/nerf_synthetic_processed/hotdog_black \
  --model_path ./output/scene_rgb_test_fixed_voxels \
  --scene_checkpoint /home/yuanxiuw/orcd/pool/fixed_voxel_static_128_splitting_prunning_neighbor_hotdog_002/checkpoint_iter_000026.pt \
  --surface_checkpoint \
    /orcd/home/002/yuanxiuw/FNO_Large/checkpoints_implicit_modes16_l1/surface/best.pt \
  --volume_checkpoint \
    /orcd/home/002/yuanxiuw/FNO_Large/checkpoints_implicit_modes16_l1/volume/best.pt \
  --camera_a 0 \
  --camera_b 171 \
  --frames_per_leg 60 \
  --fps 24 \
  --width 794 \
  --height 794 \
  --use_tile_renderer \
  --flip_projection_y \
  --camera_depth_sign 1 \
  --output ./fixed_voxel_camera_loop0_171.mp4