#!/bin/bash
#SBATCH --job-name=test_render
#SBATCH --nodes=1
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=24
#SBATCH --mem=20G
#SBATCH --time=12:00:00

cd "$HOME"

blender-5.1.0-linux-x64/blender -b --factory-startup -P FNO_Large/blender_render_bernstein.py -- --start_id 236 --end_id 300 &
blender-5.1.0-linux-x64/blender -b --factory-startup -P FNO_Large/blender_render_bernstein.py -- --start_id 301 --end_id 400 &
blender-5.1.0-linux-x64/blender -b --factory-startup -P FNO_Large/blender_render_bernstein.py -- --start_id 401 --end_id 500 &
blender-5.1.0-linux-x64/blender -b --factory-startup -P FNO_Large/blender_render_bernstein.py -- --start_id 501 --end_id 600 &

wait