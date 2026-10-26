#!/bin/bash
#SBATCH --job-name=neural_renderer_test_3dgs
#SBATCH --nodes=1
#SBATCH --partition=gpu,seas_gpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=20G
#SBATCH --time=12:00:00
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

python optimize_fixed_voxel_scene.py \
    --source_path /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/yuanxiuwang/nerf_synthetic_processed/hotdog_black \
    --model_path ./output/scene_rgb_test_fixed_voxels \
    --gaussian_model_path ./output/scene_rgb_test_fixed_voxels \
    --gaussian_iteration 50000 \
    --iterations 25 \
    --num_cameras 400 \
    --cameras_per_step 20 \
    --voxel_size 0.02 \
    --min_points_per_voxel 1 \
    --max_initial_blocks 0 \
    --fno_batch_size 512 \
    --checkpoint_interval 2 \
    --preview_views 3 \
    --output_dir /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/yuanxiuwang/fixed_voxel_static_128_splitting_prunning_neighbor_hotdog_3dgs \
    --flip_projection_y \
    --lifecycle_interval 10 \
    --max_blocks 0 \
    --max_voxel_level 2 \
    --max_splits_per_update 100 \
    --split_candidate_count 200 \
    --split_min_alive_weight 0.1 \
    --split_min_world_size 0 \
    --split_min_shape_gradient 0 \
    --neighbor_activation_interval 20 \
    --neighbor_candidate_count 5000 \
    --max_neighbor_activations_per_update 2000 \
    --enable_hard_pruning \
    --dead_alive_threshold=2e-2 \
    --dead_intervals_before_prune 2 \
    --max_prunes_per_update 2000 \
    --prune_interval 0 \
    --split_interval 0 \
    --initial_volume_probability 0.001 \
    --camera_depth_sign 1 \
    --optimize_sh \
    --use_tile_renderer \
    --surface_checkpoint=/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/yuanxiuwang/best_surface.pt \
    --volume_checkpoint=/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/yuanxiuwang/best_volume.pt \
    --resume_checkpoint /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/yuanxiuwang/fixed_voxel_static_128_splitting_prunning_neighbor_hotdog_3dgs/checkpoint_iter_000012.pt