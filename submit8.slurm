#!/bin/bash
#SBATCH --job-name=generate_transport_data_testing
#SBATCH --nodes=1
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=00:30:00

cd "$HOME/FNO_Large"

python generate_transport_data.py

wait