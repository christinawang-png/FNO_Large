#!/bin/bash
#SBATCH --job-name=shard_render_p3
#SBATCH --nodes=1
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=48
#SBATCH --mem=90G
#SBATCH --time=10:00:00

cd "$HOME"

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 45 --end_id 66 --shard_base 1000 > log_job00_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 111 --end_id 132 --shard_base 1010 > log_job01_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 177 --end_id 198 --shard_base 1020 > log_job02_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 243 --end_id 264 --shard_base 1030 > log_job03_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 309 --end_id 330 --shard_base 1040 > log_job04_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 375 --end_id 395 --shard_base 1050 > log_job05_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 440 --end_id 460 --shard_base 1060 > log_job06_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 505 --end_id 525 --shard_base 1070 > log_job07_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 570 --end_id 590 --shard_base 1080 > log_job08_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 635 --end_id 655 --shard_base 1090 > log_job09_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 700 --end_id 720 --shard_base 1100 > log_job10_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 765 --end_id 785 --shard_base 1110 > log_job11_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 830 --end_id 850 --shard_base 1120 > log_job12_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 895 --end_id 915 --shard_base 1130 > log_job13_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 960 --end_id 980 --shard_base 1140 > log_job14_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1025 --end_id 1045 --shard_base 1150 > log_job15_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1090 --end_id 1110 --shard_base 1160 > log_job16_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1155 --end_id 1175 --shard_base 1170 > log_job17_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1220 --end_id 1240 --shard_base 1180 > log_job18_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1285 --end_id 1305 --shard_base 1190 > log_job19_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1350 --end_id 1370 --shard_base 1200 > log_job20_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1415 --end_id 1435 --shard_base 1210 > log_job21_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1480 --end_id 1500 --shard_base 1220 > log_job22_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1545 --end_id 1565 --shard_base 1230 > log_job23_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1610 --end_id 1630 --shard_base 1240 > log_job24_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1675 --end_id 1695 --shard_base 1250 > log_job25_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1740 --end_id 1760 --shard_base 1260 > log_job26_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1805 --end_id 1825 --shard_base 1270 > log_job27_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1870 --end_id 1890 --shard_base 1280 > log_job28_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 1935 --end_id 1955 --shard_base 1290 > log_job29_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2000 --end_id 2020 --shard_base 1300 > log_job30_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2065 --end_id 2085 --shard_base 1310 > log_job31_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2130 --end_id 2150 --shard_base 1320 > log_job32_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2195 --end_id 2215 --shard_base 1330 > log_job33_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2260 --end_id 2280 --shard_base 1340 > log_job34_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2325 --end_id 2345 --shard_base 1350 > log_job35_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2390 --end_id 2410 --shard_base 1360 > log_job36_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2455 --end_id 2475 --shard_base 1370 > log_job37_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2520 --end_id 2540 --shard_base 1380 > log_job38_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2585 --end_id 2605 --shard_base 1390 > log_job39_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2650 --end_id 2670 --shard_base 1400 > log_job40_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2715 --end_id 2735 --shard_base 1410 > log_job41_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2780 --end_id 2800 --shard_base 1420 > log_job42_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2845 --end_id 2865 --shard_base 1430 > log_job43_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2910 --end_id 2930 --shard_base 1440 > log_job44_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 2975 --end_id 2995 --shard_base 1450 > log_job45_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 3040 --end_id 3060 --shard_base 1460 > log_job46_part3.out 2>&1 &

blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_bernstein.py -- \
  --start_id 3105 --end_id 3125 --shard_base 1470 > log_job47_part3.out 2>&1 &

wait
