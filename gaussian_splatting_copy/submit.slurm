#!/bin/bash
#SBATCH --job-name=neural_renderer_test
#SBATCH --nodes=1
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=00:20:00

cd "$HOME/FNO_Large/gaussian_splatting_copy"

source ./setup_gs_env.sh

python test_neural_slice_placement.py \
    --source_path /home/yuanxiuw/orcd/pool/3dgs_datas/treehill \
    --model_path ./output/neural_slice_test \
    --mode surface \
    --checkpoint /orcd/home/002/yuanxiuw/FNO_Large/fno_premult_surface_final.pt \
    --camera_index 0 \
    --num_cameras 4 \
    --template_index 0 \
    --fno_radius 2.2 \
    --slice_width 0.5 \
    --slice_height 1.0 \
    --slice_dx 0.5 \
    --slice_dy 0.5 \
    --slice_dz 0.0 \
    --resolution 4