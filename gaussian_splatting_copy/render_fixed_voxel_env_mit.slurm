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

python render_fixed_voxel_environment_interpolation.py \
  --source_path /home/yuanxiuw/orcd/pool/nerf_synthetic_processed/hotdog_black \
  --model_path ./output/scene_rgb_test_fixed_voxels \
  --scene_checkpoint \
    ./fixed_voxel_static_128_splitting_prunning_neighbor_hotdog/checkpoint_iter_000710_final.pt \
  --surface_checkpoint \
    /orcd/home/002/yuanxiuw/FNO_Large/checkpoints_implicit_modes16_l1/surface/best.pt \
  --volume_checkpoint \
    /orcd/home/002/yuanxiuw/FNO_Large/checkpoints_implicit_modes16_l1/volume/best.pt \
  --camera_index 0 \
  --num_frames 120 \
  --fps 24 \
  --flip_projection_y \
  --camera_depth_sign 1 \
  --global_sh_bound 0.05 \
  --local_sh_bound 0.05 \
  --output ./environment_initial_to_optimized.mp4