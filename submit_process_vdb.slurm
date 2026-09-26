#!/bin/bash
#SBATCH --job-name=shard_render
#SBATCH --nodes=1
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=12:00:00

cd "$HOME"/ "FNO_Large"

source /home/yuanxiuw/miniconda3/etc/profile.d/conda.sh

conda activate openvdb_env

python export_to_vdb.py