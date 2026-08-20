#!/usr/bin/env python3
from pathlib import Path

# Original [start_id, end_id, shard_base] for 48 jobs
jobs = [
    (   1,   66,   0),
    (  67,  132,  10),
    ( 133,  198,  20),
    ( 199,  264,  30),
    ( 265,  330,  40),
    ( 331,  395,  50),
    ( 396,  460,  60),
    ( 461,  525,  70),
    ( 526,  590,  80),
    ( 591,  655,  90),
    ( 656,  720, 100),
    ( 721,  785, 110),
    ( 786,  850, 120),
    ( 851,  915, 130),
    ( 916,  980, 140),
    ( 981, 1045, 150),
    (1046, 1110, 160),
    (1111, 1175, 170),
    (1176, 1240, 180),
    (1241, 1305, 190),
    (1306, 1370, 200),
    (1371, 1435, 210),
    (1436, 1500, 220),
    (1501, 1565, 230),
    (1566, 1630, 240),
    (1631, 1695, 250),
    (1696, 1760, 260),
    (1761, 1825, 270),
    (1826, 1890, 280),
    (1891, 1955, 290),
    (1956, 2020, 300),
    (2021, 2085, 310),
    (2086, 2150, 320),
    (2151, 2215, 330),
    (2216, 2280, 340),
    (2281, 2345, 350),
    (2346, 2410, 360),
    (2411, 2475, 370),
    (2476, 2540, 380),
    (2541, 2605, 390),
    (2606, 2670, 400),
    (2671, 2735, 410),
    (2736, 2800, 420),
    (2801, 2865, 430),
    (2866, 2930, 440),
    (2931, 2995, 450),
    (2996, 3060, 460),
    (3061, 3125, 470),
]

header = """#!/bin/bash
#SBATCH --job-name=shard_render_PART
#SBATCH --nodes=1
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=48
#SBATCH --mem=70G
#SBATCH --time=12:00:00

cd "$HOME"

"""

blender_cmd = """blender-5.1.0-linux-x64/blender -b --factory-startup -t 1 \\
  -P FNO_Large/blender_render_bernstein.py -- \\
  --start_id {start} --end_id {end} --shard_base {base} > log_job{job:02d}_part{part}.out 2>&1 &

"""

out_dir = Path(".")
parts = 3
base_offsets = [0, 500, 1000]  # shard_base offset per part

for part in range(1, parts + 1):
    lines = [header.replace("PART", f"p{part}")]
    for job_id, (start, end, base) in enumerate(jobs):
        length = end - start + 1
        chunk = length // parts
        rem = length % parts

        # compute this part's subrange
        if part == 1:
            sub_start = start
            sub_end = sub_start + chunk - 1 + (1 if rem > 0 else 0)
        elif part == 2:
            sub_start = start + chunk + (1 if rem > 0 else 0)
            sub_end = sub_start + chunk - 1 + (1 if rem > 1 else 0)
        else:  # part == 3
            # whatever is left
            if rem == 0:
                sub_start = start + 2 * chunk
            elif rem == 1:
                sub_start = start + chunk + 1 + chunk
            else:  # rem == 2
                sub_start = start + chunk + 1 + chunk + 1
            sub_end = end

        part_base = base + base_offsets[part - 1]

        lines.append(blender_cmd.format(
            start=sub_start,
            end=sub_end,
            base=part_base,
            job=job_id,
            part=part,
        ))

    lines.append("wait\n")
    out_path = out_dir / f"shard_render_part{part}.slurm"
    out_path.write_text("".join(lines))
    print("Wrote", out_path)