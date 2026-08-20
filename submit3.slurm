#!/bin/bash
#SBATCH --job-name=test_train_volume
#SBATCH --nodes=1
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=20G
#SBATCH --time=6:00:00

cd "$HOME/FNO_Large"

python train_premult_single_mode.py --mode volume --resume fno_premult_volume_epoch025.pt

wait