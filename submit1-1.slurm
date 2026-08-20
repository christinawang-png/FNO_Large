#!/bin/bash
#SBATCH --job-name=shard_render_fix
#SBATCH --nodes=1
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=3
#SBATCH --mem=10G
#SBATCH --time=12:00:00

cd "$HOME"

# fix A: shapes 446–460
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 446 --end_id 460 --shard_base 640 > log_fix_A.out 2>&1 &

# fix B: shapes 1226–1240
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1226 --end_id 1240 --shard_base 650 > log_fix_B.out 2>&1 &

# fix C: shapes 2591–2605
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2591 --end_id 2605 --shard_base 660 > log_fix_C.out 2>&1 &

wait