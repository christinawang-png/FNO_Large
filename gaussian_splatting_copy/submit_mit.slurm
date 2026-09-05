#!/bin/bash
#SBATCH --job-name=neural_renderer_test_no_checkpoint
#SBATCH --nodes=1
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=1:00:00

cd "$HOME/FNO_Large/gaussian_splatting_copy"

source ./setup_gs_env.sh

python optimize_neural_scene_random_init.py \
    --source_path /home/yuanxiuw/orcd/pool/3dgs_datas/treehill \
    --model_path ./output/tile_renderer_test \
    --resolution 4 \
    --num_cameras 2 \
    --max_initial_slices 64 \
    --disable_initial_mode_selection \
    --disable_child_mode_selection \
    --use_tile_renderer \
    --split_interval 0 \
    --prune_interval 0 \
    --iterations 200 \
    --fno_batch_size 8 \
    --checkpoint_interval 0 \
    --disable_fno_activation_checkpointing