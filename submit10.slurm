#!/bin/bash
#SBATCH --job-name=shard_render_opacity
#SBATCH --nodes=1
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=1:00:00

cd "$HOME/FNO_Large"

python generate_opacity_from_images.py


wait