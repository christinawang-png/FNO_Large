#!/bin/bash
#SBATCH --job-name=shard_render
#SBATCH --nodes=1
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=12:00:00

cd "$HOME"/ "FNO_Large"

python bend_plane_params.py