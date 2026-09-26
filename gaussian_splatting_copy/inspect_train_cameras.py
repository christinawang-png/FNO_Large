#!/usr/bin/env python
import sys
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parent
FNO_ROOT = REPO_DIR.parent

sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(FNO_ROOT))

from arguments import ModelParams
from scene import Scene, GaussianModel


def main():
    parser = ArgumentParser()
    lp = ModelParams(parser)

    parser.add_argument("--max_cameras", type=int, default=0)

    args = parser.parse_args()

    scene_args = lp.extract(args)

    gaussians = GaussianModel(args.sh_degree)

    scene = Scene(
        scene_args,
        gaussians,
        shuffle=False,
        resolution_scales=[1.0],
    )

    cameras = scene.getTrainCameras(scale=1.0)

    if args.max_cameras > 0:
        cameras = cameras[:args.max_cameras]

    centers = torch.stack(
        [camera.camera_center.detach().cpu() for camera in cameras],
        dim=0,
    )

    scene_center = gaussians.get_xyz.detach().median(
        dim=0
    ).values.cpu()

    relative = centers - scene_center[None, :]
    radii = torch.linalg.norm(relative, dim=1)

    print("Number of cameras:", len(cameras))
    print("Point-cloud median center:", scene_center.numpy())
    print("Camera center mean:", centers.mean(dim=0).numpy())
    print("Camera center median:", centers.median(dim=0).values.numpy())
    print(
        "Camera radius: "
        f"min={radii.min().item():.4f}, "
        f"median={radii.median().item():.4f}, "
        f"mean={radii.mean().item():.4f}, "
        f"max={radii.max().item():.4f}"
    )

    print("\nFirst camera centers:")
    for i, center in enumerate(centers[:10]):
        print(f"  {i:03d}: {center.numpy()}")


if __name__ == "__main__":
    main()