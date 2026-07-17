#!/bin/bash
#SBATCH --job-name=test_train_finetune
#SBATCH --nodes=1
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=20G
#SBATCH --time=6:00:00

cd "$HOME/FNO_Large"

python fine_tune_L1.py

wait