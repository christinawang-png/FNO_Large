#!/bin/bash
#SBATCH --job-name=shard_render_fix
#SBATCH --nodes=1
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=5
#SBATCH --mem=30G
#SBATCH --time=12:00:00

cd "$HOME"

# fix for job 25: shapes 1664–1695
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1664 --end_id 1695 --shard_base 500 > log_fix_25.out 2>&1 &

# fix for job 28: shapes 1842–1890
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1842 --end_id 1890 --shard_base 510 > log_fix_28.out 2>&1 &

# fix for job 36: shapes 2362–2410
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2362 --end_id 2410 --shard_base 520 > log_fix_36.out 2>&1 &

# fix for job 40: shapes 2622–2670
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2622 --end_id 2670 --shard_base 530 > log_fix_40.out 2>&1 &

# fix for job 44: shapes 2882–2930
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2882 --end_id 2930 --shard_base 540 > log_fix_44.out 2>&1 &

wait