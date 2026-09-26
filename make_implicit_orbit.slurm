#!/bin/bash
#SBATCH --job-name=make_video
#SBATCH --nodes=1
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=00:20:00

cd "$HOME"/ "FNO_Large"

python make_network_orbit_video.py \
  --dataset-root /orcd/home/002/yuanxiuw/FNO_Large/implicit_bspline_dataset \
  --checkpoint /orcd/home/002/yuanxiuw/FNO_Large/checkpoints_implicit_modes16_l1/volume/best.pt \
  --mode volume \
  --color "0.95, 0.3, 0.15" \
  --opacity 0.95 \
  --metallic 0.0 \
  --roughness 0.30 \
  --specular 0.5 \
  --sigma 0.08 \
  --elevation-deg 90 \
  --num-frames 120 \
  --geometry-id 789 \
  --fps 24 \
  --scale 16 \
  --env-id 0 \
  --output /orcd/home/002/yuanxiuw/FNO_Large/output/network_volume_orbit_789.mp4