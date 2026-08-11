#!/bin/bash
#SBATCH --job-name=generate_transport_data_testing
#SBATCH --nodes=1
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=5G
#SBATCH --time=4:00:00

cd "$HOME/FNO_Large"

python generate_transport_data.py

wait