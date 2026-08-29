#!/bin/bash
#SBATCH --job-name=neural_renderer_test
#SBATCH --nodes=1
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=01:00:00

cd "$HOME/FNO_Large/gaussian_splatting_copy"

source ./setup_gs_env.sh
    
python optimize_neural_scene_random_init.py \
    --source_path  /home/yuanxiuw/orcd/pool/3dgs_datas/treehill \
    --model_path ./output/random_init_scene \
    --resolution 4 \
    --camera_start 0 \
    --num_cameras 8 \
    --initial_num_slices 8 \
    --voxel_size 0.5 \
    --min_points_per_voxel 50 \
    --mode_selection_cameras 3 \
    --iterations 1000 \
    --lr 0.0001 \
    --warmup_iterations 300 \
    --prune_interval 0 \
    --split_interval 500 \
    --max_splits_per_update 1 \
    --max_slices 16 \
    --checkpoint_interval 500 \
    --optimize_sh