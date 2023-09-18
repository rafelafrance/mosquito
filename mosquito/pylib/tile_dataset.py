import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms

from .tile import Tile


class TileDataset(Dataset):
    def __init__(self, tiles: list[Tile], layers, target=None, augment=False):
        self.tiles = tiles
        self.layers = np.stack([io.imread(lay) for lay in layers], axis=2)
        self.target = io.imread(target) if target else None
        self.augment = augment

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        ...

    def get_tile(self):
        ...

    def transform(self, tile):
        """Perform the same transforms on the given images."""
        # if augment:
        #     xform += [
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomVerticalFlip(),
        #     ]
        #
        # xform += [
        #     transforms.ToTensor(),
        #     transforms.ConvertImageDtype(torch.float),
        #     transforms.Normalize(self.model.mean, self.model.std_dev),
        # ]
        #
        # return transforms.Compose(xform)
