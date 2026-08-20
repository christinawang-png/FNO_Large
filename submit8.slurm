#!/bin/bash
#SBATCH --job-name=shard_render
#SBATCH --nodes=1
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=48
#SBATCH --mem=100G
#SBATCH --time=12:00:00

cd "$HOME"

# job 0: shapes 1–66
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 1   --end_id 66   --shard_base 0   > log_job00.out 2>&1 &

# job 1: shapes 67–132
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 67  --end_id 132  --shard_base 10  > log_job01.out 2>&1 &

# job 2: shapes 133–198
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 133 --end_id 198  --shard_base 20  > log_job02.out 2>&1 &

# job 3: shapes 199–264
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 199 --end_id 264  --shard_base 30  > log_job03.out 2>&1 &

# job 4: shapes 265–330
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 265 --end_id 330  --shard_base 40  > log_job04.out 2>&1 &

# job 5: shapes 331–395
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 331 --end_id 395  --shard_base 50  > log_job05.out 2>&1 &

# job 6: shapes 396–460
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 396 --end_id 460  --shard_base 60  > log_job06.out 2>&1 &

# job 7: shapes 461–525
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 461 --end_id 525  --shard_base 70  > log_job07.out 2>&1 &

# job 8: shapes 526–590
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 526 --end_id 590  --shard_base 80  > log_job08.out 2>&1 &

# job 9: shapes 591–655
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 591 --end_id 655  --shard_base 90  > log_job09.out 2>&1 &

# job 10: shapes 656–720
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 656 --end_id 720  --shard_base 100 > log_job10.out 2>&1 &

# job 11: shapes 721–785
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 721 --end_id 785  --shard_base 110 > log_job11.out 2>&1 &

# job 12: shapes 786–850
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 786 --end_id 850  --shard_base 120 > log_job12.out 2>&1 &

# job 13: shapes 851–915
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 851 --end_id 915  --shard_base 130 > log_job13.out 2>&1 &

# job 14: shapes 916–980
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 916 --end_id 980  --shard_base 140 > log_job14.out 2>&1 &

# job 15: shapes 981–1045
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 981 --end_id 1045 --shard_base 150 > log_job15.out 2>&1 &

# job 16: shapes 1046–1110
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 1046 --end_id 1110 --shard_base 160 > log_job16.out 2>&1 &

# job 17: shapes 1111–1175
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 1111 --end_id 1175 --shard_base 170 > log_job17.out 2>&1 &

# job 18: shapes 1176–1240
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 1176 --end_id 1240 --shard_base 180 > log_job18.out 2>&1 &

# job 19: shapes 1241–1305
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 1241 --end_id 1305 --shard_base 190 > log_job19.out 2>&1 &

# job 20: shapes 1306–1370
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 1306 --end_id 1370 --shard_base 200 > log_job20.out 2>&1 &

# job 21: shapes 1371–1435
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 1371 --end_id 1435 --shard_base 210 > log_job21.out 2>&1 &

# job 22: shapes 1436–1500
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 1436 --end_id 1500 --shard_base 220 > log_job22.out 2>&1 &

# job 23: shapes 1501–1565
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 1501 --end_id 1565 --shard_base 230 > log_job23.out 2>&1 &

# job 24: shapes 1566–1630
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 1566 --end_id 1630 --shard_base 240 > log_job24.out 2>&1 &

# job 25: shapes 1631–1695
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 1631 --end_id 1695 --shard_base 250 > log_job25.out 2>&1 &

# job 26: shapes 1696–1760
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 1696 --end_id 1760 --shard_base 260 > log_job26.out 2>&1 &

# job 27: shapes 1761–1825
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 1761 --end_id 1825 --shard_base 270 > log_job27.out 2>&1 &

# job 28: shapes 1826–1890
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 1826 --end_id 1890 --shard_base 280 > log_job28.out 2>&1 &

# job 29: shapes 1891–1955
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 1891 --end_id 1955 --shard_base 290 > log_job29.out 2>&1 &

# job 30: shapes 1956–2020
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 1956 --end_id 2020 --shard_base 300 > log_job30.out 2>&1 &

# job 31: shapes 2021–2085
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 2021 --end_id 2085 --shard_base 310 > log_job31.out 2>&1 &

# job 32: shapes 2086–2150
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 2086 --end_id 2150 --shard_base 320 > log_job32.out 2>&1 &

# job 33: shapes 2151–2215
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 2151 --end_id 2215 --shard_base 330 > log_job33.out 2>&1 &

# job 34: shapes 2216–2280
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 2216 --end_id 2280 --shard_base 340 > log_job34.out 2>&1 &

# job 35: shapes 2281–2345
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 2281 --end_id 2345 --shard_base 350 > log_job35.out 2>&1 &

# job 36: shapes 2346–2410
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 2346 --end_id 2410 --shard_base 360 > log_job36.out 2>&1 &

# job 37: shapes 2411–2475
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 2411 --end_id 2475 --shard_base 370 > log_job37.out 2>&1 &

# job 38: shapes 2476–2540
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 2476 --end_id 2540 --shard_base 380 > log_job38.out 2>&1 &

# job 39: shapes 2541–2605
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 2541 --end_id 2605 --shard_base 390 > log_job39.out 2>&1 &

# job 40: shapes 2606–2670
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 2606 --end_id 2670 --shard_base 400 > log_job40.out 2>&1 &

# job 41: shapes 2671–2735
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 2671 --end_id 2735 --shard_base 410 > log_job41.out 2>&1 &

# job 42: shapes 2736–2800
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 2736 --end_id 2800 --shard_base 420 > log_job42.out 2>&1 &

# job 43: shapes 2801–2865
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 2801 --end_id 2865 --shard_base 430 > log_job43.out 2>&1 &

# job 44: shapes 2866–2930
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 2866 --end_id 2930 --shard_base 440 > log_job44.out 2>&1 &

# job 45: shapes 2931–2995
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 2931 --end_id 2995 --shard_base 450 > log_job45.out 2>&1 &

# job 46: shapes 2996–3060
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 2996 --end_id 3060 --shard_base 460 > log_job46.out 2>&1 &

# job 47: shapes 3061–3125
blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \
  -P FNO_Large/blender_render_preview.py -- \
  --start_id 3061 --end_id 3125 --shard_base 470 > log_job47.out 2>&1 &

wait