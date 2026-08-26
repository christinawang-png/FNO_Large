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

python test_two_neural_slices.py \
    --source_path /home/yuanxiuw/orcd/pool/3dgs_datas/treehill \
    --model_path ./output/two_slice_test \
    --mode1 surface \
    --mode2 volume \
    --checkpoint1 /home/yuanxiuw/FNO_Large/fno_premult_surface_final.pt \
    --checkpoint2 /home/yuanxiuw/FNO_Large/fno_premult_volume_epoch025.pt \
    --camera_index 0 \
    --num_cameras 4 \
    --template1 0 \
    --template2 1 \
    --fno_radius 2.2 \
    --slice1_dx 0.0 \
    --slice1_dy 0.0 \
    --slice1_dz 0.0 \
    --slice1_width 1.0 \
    --slice1_height 1.0 \
    --slice2_dx 0.35 \
    --slice2_dy 0.0 \
    --slice2_dz 0.25 \
    --slice2_width 0.8 \
    --slice2_height 0.8 \
    --order slice1,slice2 \
    --resolution 4