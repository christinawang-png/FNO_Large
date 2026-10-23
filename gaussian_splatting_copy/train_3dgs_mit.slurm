#!/bin/bash
#SBATCH --job-name=neural_renderer_train_initialization
#SBATCH --nodes=1
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=5G
#SBATCH --time=00:30:00

cd "$HOME/FNO_Large/gaussian_splatting_copy"

source ./setup_gs_env.sh

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_GPUS=${SLURM_GPUS:-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
echo "Hostname: $(hostname)"

python train.py \
  -s /home/yuanxiuw/orcd/pool/nerf_synthetic_processed/hotdog_black \
  -m ./output/scene_rgb_test_fixed_voxels \
  --iterations 50000