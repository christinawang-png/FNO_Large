import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms

base_dir  = Path("/orcd/home/002/yuanxiuw/FNO_Large/plane_dataset_3")
csv_path  = base_dir / "renders" / "metadata_images_all.csv"
df        = pd.read_csv(csv_path)

img_size = (64, 64)
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
])

N = len(df)
shard_size = 50000
num_shards = (N + shard_size - 1) // shard_size

shard_ids = np.empty(N, dtype=np.int32)
idx_in_shard = np.empty(N, dtype=np.int32)

for s in range(num_shards):
    start = s * shard_size
    end   = min((s + 1) * shard_size, N)
    M = end - start
    X_shard = np.empty((M, 3, img_size[0], img_size[1]), dtype=np.float32)

    for local_i, i in enumerate(range(start, end)):
        row = df.iloc[i]
        img = Image.open(row["image_path"]).convert("RGB")
        img_t = transform(img)
        X_shard[local_i] = img_t.numpy()

        shard_ids[i]    = s
        idx_in_shard[i] = local_i

        if i % 2000 == 0:
            print(f"{i}/{N} packed into shard {s}")

    shard_path = base_dir / f"images_64x64_shard_{s:02d}.npy"
    np.save(shard_path, X_shard)
    print("Saved", shard_path)

# add shard info to CSV
df["shard_id"]      = shard_ids
df["idx_in_shard"]  = idx_in_shard
df.to_csv(csv_path, index=False)
print("Updated CSV with shard_id and idx_in_shard")