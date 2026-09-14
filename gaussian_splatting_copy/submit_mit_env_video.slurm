#!/bin/bash
#SBATCH --job-name=neural_renderer_test_env_video
#SBATCH --nodes=1
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=1:00:00

cd "$HOME/FNO_Large/gaussian_splatting_copy"

source ./setup_gs_env.sh


python render_neural_scene_env_video.py \
    --source_path /home/yuanxiuw/orcd/pool/nerf_synthetic_processed/hotdog \
    --model_path ./output/video_scene \
    --resolution 1 \
    --scene_checkpoint /home/yuanxiuw/FNO_Large/gaussian_splatting_copy/output/scene_rgb_test_colmap/checkpoint_iter_000899_final.pt \
    --camera_index 0 \
    --env_a 0 \
    --env_b 127 \
    --num_frames 200 \
    --fps 40 \
    --background white \
    --fno_batch_size 512 \
    --use_visibility_culling \
    --max_active_slices_per_camera 0 \
    --flip_projection_y \
    --save_alpha_video \
    --output_dir ./neural_scene_videos \
    --output_name fixed_camera_env.mp4 \
    --loop