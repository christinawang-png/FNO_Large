#!/bin/bash
#SBATCH --job-name=blender_preview
#SBATCH --nodes=1
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=12:00:00

cd "$HOME"

blender-5.1.0-linux-x64/blender -b -P FNO_Large/blender_preview_surfaces.py -- --max_geometries 100