#!/bin/bash
#SBATCH --job-name=neural_renderer_test
#SBATCH --nodes=1
#SBATCH --partition=gpu,seas_gpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=1:00:00
#SBATCH --mail-user=yuanxiuwang@fas.harvard.edu
#SBATCH --exclude=holygpu8a31105

cd "/n/home13/yuanxiuwang/FNO_Large/gaussian_splatting_copy/"
source "/n/helmod/apps/centos7/Core/Anaconda3/2021.05-jupyterood-fasrc01/x/etc/profile.d/conda.sh"
conda activate /n/home13/yuanxiuwang/.conda/envs/gaussian_splatting_cu124

module load gcc/12.2.0-fasrc01

source ./setup_gs_env.sh

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_GPUS=${SLURM_GPUS:-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
echo "Hostname: $(hostname)"

nvidia-smi

python train.py \
  -s /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/yuanxiuwang/nerf_synthetic_processed/hotdog_black \
  -m ./output/scene_rgb_test_fixed_voxels \
  --iterations 30000