#!/bin/bash
#SBATCH --job-name=neural_renderer_test_profile
#SBATCH --nodes=1
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=1:00:00

cd "$HOME/FNO_Large/gaussian_splatting_copy"

source ./setup_gs_env.sh
    
python benchmark_scene_renderer.py \
    --source_path /home/yuanxiuw/orcd/pool/3dgs_datas/treehill \
    --model_path ./output/benchmark_scene \
    --resolution 4 \
    --scene_checkpoint /home/yuanxiuw/FNO_Large/gaussian_splatting_copy/neural_scene_random_init_outputs/checkpoint_iter_000080.pt \
    --camera_index 0 \
    --fno_batch_size 512 \
    --placement_batch_size 4 \
    --output_dir ./lightweight_renderer_benchmarks