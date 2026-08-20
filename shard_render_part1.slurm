#!/bin/bash
#SBATCH --job-name=shard_render_p1
#SBATCH --nodes=1
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=48
#SBATCH --mem=90G
#SBATCH --time=10:00:00

cd "$HOME"

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1 --end_id 22 --shard_base 0 > log_job00_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 67 --end_id 88 --shard_base 10 > log_job01_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 133 --end_id 154 --shard_base 20 > log_job02_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 199 --end_id 220 --shard_base 30 > log_job03_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 265 --end_id 286 --shard_base 40 > log_job04_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 331 --end_id 352 --shard_base 50 > log_job05_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 396 --end_id 417 --shard_base 60 > log_job06_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 461 --end_id 482 --shard_base 70 > log_job07_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 526 --end_id 547 --shard_base 80 > log_job08_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 591 --end_id 612 --shard_base 90 > log_job09_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 656 --end_id 677 --shard_base 100 > log_job10_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 721 --end_id 742 --shard_base 110 > log_job11_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 786 --end_id 807 --shard_base 120 > log_job12_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 851 --end_id 872 --shard_base 130 > log_job13_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 916 --end_id 937 --shard_base 140 > log_job14_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 981 --end_id 1002 --shard_base 150 > log_job15_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1046 --end_id 1067 --shard_base 160 > log_job16_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1111 --end_id 1132 --shard_base 170 > log_job17_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1176 --end_id 1197 --shard_base 180 > log_job18_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1241 --end_id 1262 --shard_base 190 > log_job19_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1306 --end_id 1327 --shard_base 200 > log_job20_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1371 --end_id 1392 --shard_base 210 > log_job21_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1436 --end_id 1457 --shard_base 220 > log_job22_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1501 --end_id 1522 --shard_base 230 > log_job23_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1566 --end_id 1587 --shard_base 240 > log_job24_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1631 --end_id 1652 --shard_base 250 > log_job25_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1696 --end_id 1717 --shard_base 260 > log_job26_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1761 --end_id 1782 --shard_base 270 > log_job27_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1826 --end_id 1847 --shard_base 280 > log_job28_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1891 --end_id 1912 --shard_base 290 > log_job29_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1956 --end_id 1977 --shard_base 300 > log_job30_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2021 --end_id 2042 --shard_base 310 > log_job31_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2086 --end_id 2107 --shard_base 320 > log_job32_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2151 --end_id 2172 --shard_base 330 > log_job33_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2216 --end_id 2237 --shard_base 340 > log_job34_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2281 --end_id 2302 --shard_base 350 > log_job35_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2346 --end_id 2367 --shard_base 360 > log_job36_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2411 --end_id 2432 --shard_base 370 > log_job37_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2476 --end_id 2497 --shard_base 380 > log_job38_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2541 --end_id 2562 --shard_base 390 > log_job39_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2606 --end_id 2627 --shard_base 400 > log_job40_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2671 --end_id 2692 --shard_base 410 > log_job41_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2736 --end_id 2757 --shard_base 420 > log_job42_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2801 --end_id 2822 --shard_base 430 > log_job43_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2866 --end_id 2887 --shard_base 440 > log_job44_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2931 --end_id 2952 --shard_base 450 > log_job45_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2996 --end_id 3017 --shard_base 460 > log_job46_part1.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 3061 --end_id 3082 --shard_base 470 > log_job47_part1.out 2>&1 &

wait
