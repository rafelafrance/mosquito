import numpy as np
from PIL import Image
from scipy import stats
from torch.utils.data import Dataset

from .tile import Tile

NA_LO = -3.0e38
NA_HI = 3.0e38


class TileDataset(Dataset):
    def __init__(self, tiles: list[Tile], layers, target, augment=False):
        self.layers = layers
        self.target = target
        self.augment = augment
        self.tiles = self.filter_tiles(tiles)

    def filter_tiles(self, tiles):
        """Remove tiles with NaNs."""
        new = []

        for tile in tiles:
            image = self.layers[:, tile.top : tile.bottom, tile.left : tile.right]

            if np.isnan(image).any():
                continue

            new.append(tile)
        return new

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        left, top, right, bottom = self.get_tile_coords(idx)

        image = self.layers[:, top:bottom, left:right]
        target = self.target[:, top:bottom, left:right]

        if self.augment:
            image, target = self.transform(image, target)

        return image, target, idx

    def get_tile_coords(self, idx):
        tile = self.tiles[idx]
        return tile.left, tile.top, tile.right, tile.bottom

    def pos_weight(self):
        pos, count = 0.0, 0.0
        for tile in self.tiles:
            count += (tile.bottom - tile.top) * (tile.right - tile.left)
            pos += self.target[:, tile.top : tile.bottom, tile.left : tile.right].sum()
        pos_wt = (count - pos) / pos if pos > 0.0 else 1.0
        return [pos_wt]

    @staticmethod
    def transform(image, target):
        change = False

        if (k := np.random.randint(4)) > 0:
            change = True
            image = np.rot90(image, k=k, axes=(1, 2))
            if target is not None:
                target = np.rot90(target, k=k, axes=(1, 2))

        if np.random.randint(2) > 0:
            change = True
            image = np.flip(image, axis=1)
            if target is not None:
                target = np.flip(target, axis=1)

        if np.random.randint(2) > 0:
            change = True
            image = np.flip(image, axis=2)
            if target is not None:
                target = np.flip(target, axis=2)

        if change:
            image = image.copy()
            if target is not None:
                target = target.copy()

        return image, target


def prepare_image(path, target=False, threshold=10_000, zscore=True):
    """Prepare the image data by clipping flagged Inf values and doing a z-norm."""
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as img:
        image = np.array(img)  # noqa

    if target:
        image[image > NA_HI] = 0.0
        image = np.expand_dims(image, axis=0)

    else:
        top = np.max(image)
        bot = np.min(image)

        unique = np.unique(image)

        if top > NA_HI:
            lo = max(unique[0], -threshold)
            hi = min(unique[-2], threshold)
            image = np.clip(image, lo, hi)

        elif bot < NA_LO:
            lo = max(unique[1], -threshold)
            hi = min(unique[-1], threshold)
            image = np.clip(image, lo, hi)

        if zscore:
            image = stats.zscore(image)

    return image
