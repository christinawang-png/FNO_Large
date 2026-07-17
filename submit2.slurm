#!/bin/bash
#SBATCH --job-name=test_render
#SBATCH --nodes=1
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=12:00:00

cd "$HOME"

# Re-render shapes 72–100
blender-5.1.0-linux-x64/blender -b --factory-startup -P FNO_Large/blender_render_bernstein.py --\
  --start_id 72 --end_id 100

# Re-render shapes 184–200
blender-5.1.0-linux-x64/blender -b --factory-startup -P FNO_Large/blender_render_bernstein.py --\
  --start_id 184 --end_id 200

wait