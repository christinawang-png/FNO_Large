from pathlib import Path

import math

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from neuralop.models import FNO  # must be installed

from torchvision.utils import save_image
import random

from torch.utils.data import Dataset
import torch.nn.functional as F
import os

class FaceTransportNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, num_layers=4):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.GELU())
            d = hidden
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, in_dim]
        return self.net(x)  # [B, out_dim]