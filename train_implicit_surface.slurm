#!/bin/bash
#SBATCH --job-name=test_train_surface
#SBATCH --nodes=1
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=6:00:00

cd "$HOME/FNO_Large"

python train_implicit.py \
  --dataset-root /orcd/home/002/yuanxiuw/FNO_Large/implicit_bspline_dataset \
  --mode surface \
  --epochs 60 \
  --batch-size 2048 \
  --num-workers 8 \
  --foreground-weight 8.0 \
  --no-amp \
  --fno-modes 24 \
  --checkpoint-dir checkpoints_implicit_modes24