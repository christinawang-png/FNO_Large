#!/bin/bash
#SBATCH --job-name=test_train_surface
#SBATCH --nodes=1
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=6:00:00

cd "$HOME/FNO_Large"

python train_premult_single_mode.py --mode surface --resume fno_premult_surface_epoch120_color.pt

wait