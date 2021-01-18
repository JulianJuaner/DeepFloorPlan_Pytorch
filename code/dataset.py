import sys
import os
from abc import ABC

import torch
from .colormap import *
from torch.utils.data import Dataset


class FloorDataset(Dataset, ABC):
    def __init__(self, args=None, mode='train'):
        self.args = args
        self.mode = mode
        self.img_list = os.listdir(self.args.file_root)
