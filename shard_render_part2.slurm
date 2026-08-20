#!/bin/bash
#SBATCH --job-name=shard_render_p2
#SBATCH --nodes=1
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=48
#SBATCH --mem=90G
#SBATCH --time=10:00:00

cd "$HOME"

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 23 --end_id 44 --shard_base 500 > log_job00_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 89 --end_id 110 --shard_base 510 > log_job01_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 155 --end_id 176 --shard_base 520 > log_job02_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 221 --end_id 242 --shard_base 530 > log_job03_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 287 --end_id 308 --shard_base 540 > log_job04_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 353 --end_id 374 --shard_base 550 > log_job05_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 418 --end_id 439 --shard_base 560 > log_job06_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 483 --end_id 504 --shard_base 570 > log_job07_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 548 --end_id 569 --shard_base 580 > log_job08_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 613 --end_id 634 --shard_base 590 > log_job09_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 678 --end_id 699 --shard_base 600 > log_job10_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 743 --end_id 764 --shard_base 610 > log_job11_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 808 --end_id 829 --shard_base 620 > log_job12_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 873 --end_id 894 --shard_base 630 > log_job13_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 938 --end_id 959 --shard_base 640 > log_job14_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1003 --end_id 1024 --shard_base 650 > log_job15_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1068 --end_id 1089 --shard_base 660 > log_job16_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1133 --end_id 1154 --shard_base 670 > log_job17_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1198 --end_id 1219 --shard_base 680 > log_job18_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1263 --end_id 1284 --shard_base 690 > log_job19_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1328 --end_id 1349 --shard_base 700 > log_job20_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1393 --end_id 1414 --shard_base 710 > log_job21_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1458 --end_id 1479 --shard_base 720 > log_job22_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1523 --end_id 1544 --shard_base 730 > log_job23_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1588 --end_id 1609 --shard_base 740 > log_job24_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1653 --end_id 1674 --shard_base 750 > log_job25_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1718 --end_id 1739 --shard_base 760 > log_job26_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1783 --end_id 1804 --shard_base 770 > log_job27_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1848 --end_id 1869 --shard_base 780 > log_job28_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1913 --end_id 1934 --shard_base 790 > log_job29_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1978 --end_id 1999 --shard_base 800 > log_job30_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2043 --end_id 2064 --shard_base 810 > log_job31_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2108 --end_id 2129 --shard_base 820 > log_job32_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2173 --end_id 2194 --shard_base 830 > log_job33_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2238 --end_id 2259 --shard_base 840 > log_job34_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2303 --end_id 2324 --shard_base 850 > log_job35_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2368 --end_id 2389 --shard_base 860 > log_job36_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2433 --end_id 2454 --shard_base 870 > log_job37_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2498 --end_id 2519 --shard_base 880 > log_job38_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2563 --end_id 2584 --shard_base 890 > log_job39_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2628 --end_id 2649 --shard_base 900 > log_job40_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2693 --end_id 2714 --shard_base 910 > log_job41_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2758 --end_id 2779 --shard_base 920 > log_job42_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2823 --end_id 2844 --shard_base 930 > log_job43_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2888 --end_id 2909 --shard_base 940 > log_job44_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2953 --end_id 2974 --shard_base 950 > log_job45_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 3018 --end_id 3039 --shard_base 960 > log_job46_part2.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 3083 --end_id 3104 --shard_base 970 > log_job47_part2.out 2>&1 &

wait
