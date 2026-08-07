#!/bin/bash
#SBATCH --job-name=test_video
#SBATCH --nodes=1
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=0:05:00

cd "$HOME/FNO_Large"

python wow_video.py

wait