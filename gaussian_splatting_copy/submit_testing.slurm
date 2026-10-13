#!/bin/bash
#SBATCH --job-name=neural_renderer_profile
#SBATCH --nodes=1
#SBATCH --partition=gpu,gpu_h200,seas_gpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=1:00:00
#SBATCH --mail-user=yuanxiuwang@fas.harvard.edu

cd "/n/home13/yuanxiuwang/FNO_Large/gaussian_splatting_copy/"
source "/n/helmod/apps/centos7/Core/Anaconda3/2021.05-jupyterood-fasrc01/x/etc/profile.d/conda.sh"
conda activate /n/home13/yuanxiuwang/.conda/envs/gaussian_splatting_cu124

module load gcc/12.2.0-fasrc01

source ./setup_gs_env.sh


python profile_scene_renderer.py \
    --source_path /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/yuanxiuwang/treehill \
    --model_path ./output/profile_scene \
    --resolution 4 \
    --scene_checkpoint /n/home13/yuanxiuwang/FNO_Large/gaussian_splatting_copy/neural_scene_random_init_outputs/checkpoint_iter_000080.pt \
    --camera_index 0 \
    --fno_batch_size 512 \
    --placement_batch_size 4 \
    --profile_dir ./renderer_profiles