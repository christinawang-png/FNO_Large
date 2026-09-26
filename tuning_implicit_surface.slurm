#!/bin/bash
#SBATCH --job-name=finetune_surface
#SBATCH --nodes=1
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=3:00:00

python train_implicit.py \
  --dataset-root /orcd/home/002/yuanxiuw/FNO_Large/implicit_bspline_dataset \
  --mode surface \
  --epochs 140 \
  --batch-size 2048 \
  --num-workers 8 \
  --learning-rate 3e-5 \
  --loss-type mixed_l1 \
  --l1-weight 0.75 \
  --foreground-weight 8.0 \
  --fno-modes 16 \
  --no-amp \
  --resume checkpoints_implicit_modes16_l1/surface/best.pt \
  --checkpoint-dir checkpoints_implicit_modes16_l1