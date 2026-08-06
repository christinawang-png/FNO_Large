#!/bin/bash
#SBATCH --job-name=test_train_small
#SBATCH --nodes=1
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=30G
#SBATCH --time=6:00:00

cd "$HOME/FNO_Large"

python train.py

wait