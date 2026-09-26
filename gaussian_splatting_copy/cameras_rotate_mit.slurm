#!/bin/bash
#SBATCH --job-name=camera_video
#SBATCH --nodes=1
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=2:00:00

cd "$HOME/FNO_Large/gaussian_splatting_copy"

source ./setup_gs_env.sh
  
 
python render_neural_scene_camera_interpolation.py \
  --source_path /orcd/home/002/yuanxiuw/orcd/pool/nerf_synthetic_processed/hotdog_black \
  --model_path ./output/scene_rgb_test_colmap_black_split_softmax \
  --scene_checkpoint /orcd/home/002/yuanxiuw/orcd/pool/test_neural_scene_split_softmax/checkpoint_iter_000149_final.pt \
  --surface_checkpoint /orcd/home/002/yuanxiuw/FNO_Large/checkpoints_implicit_modes16_l1/surface/best.pt \
  --volume_checkpoint /orcd/home/002/yuanxiuw/FNO_Large/checkpoints_implicit_modes16_l1/volume/best.pt \
  --renderer_module optimize_neural_scene_random_init \
  --camera_a 0 \
  --camera_b 100 \
  --num_frames 120 \
  --fps 24 \
  --width 786 \
  --height 786 \
  --use_tile_renderer \
  --flip_projection_y \
  --output ./camera_0_to_100.mp4