#!/bin/bash
#SBATCH --job-name=visualize_implicit
#SBATCH --nodes=1
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=0:10:00


python visualize_checkpoint_predictions.py \
  --dataset-root /orcd/home/002/yuanxiuw/FNO_Large/implicit_bspline_dataset \
  --checkpoint /orcd/home/002/yuanxiuw/FNO_Large/checkpoints_implicit_modes16_l1/surface/best.pt \
  --mode surface \
  --num-train 10 \
  --num-test 10 \
  --output-dir /orcd/home/002/yuanxiuw/FNO_Large/visualizations_surface_modes16_l1
 
python visualize_checkpoint_predictions.py \
  --dataset-root /orcd/home/002/yuanxiuw/FNO_Large/implicit_bspline_dataset \
  --checkpoint /orcd/home/002/yuanxiuw/FNO_Large/checkpoints_implicit_modes16_l1/volume/best.pt \
  --mode volume \
  --num-train 10 \
  --num-test 10 \
  --output-dir /orcd/home/002/yuanxiuw/FNO_Large/visualizations_volume_modes16_l1