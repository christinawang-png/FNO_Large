#!/bin/bash
#SBATCH --job-name=test_render
#SBATCH --nodes=1
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=24
#SBATCH --mem=20G
#SBATCH --time=12:00:00

cd "$HOME"

blender-5.1.0-linux-x64/blender -b --factory-startup -P FNO_Large/blender_render_bernstein.py -- --start_id 1  --end_id 100 &
blender-5.1.0-linux-x64/blender -b --factory-startup -P FNO_Large/blender_render_bernstein.py -- --start_id 101 --end_id 200 &
blender-5.1.0-linux-x64/blender -b --factory-startup -P FNO_Large/blender_render_bernstein.py -- --start_id 201 --end_id 300 &
blender-5.1.0-linux-x64/blender -b --factory-startup -P FNO_Large/blender_render_bernstein.py -- --start_id 301 --end_id 400 &

wait