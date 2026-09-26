#!/bin/bash
#SBATCH --job-name=camera_inspect
#SBATCH --nodes=1
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=0:10:00

cd "$HOME/FNO_Large/gaussian_splatting_copy"

source ./setup_gs_env.sh

python inspect_train_cameras.py --source_path /orcd/home/002/yuanxiuw/orcd/pool/nerf_synthetic_processed/hotdog_black --model_path ./output/scene_rgb_test_colmap_black_split_softmax --max_cameras 400