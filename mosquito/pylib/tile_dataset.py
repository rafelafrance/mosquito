from torch.utils.data import Dataset

from .tile import Tile


class TileDataset(Dataset):
    def __init__(self, tiles: list[Tile], layers, target=None, augment=False):
        self.tiles = tiles
        self.layers = layers
        self.target = target
        self.augment = augment

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = self.tiles[idx]

        image = self.layers[:, tile.top : tile.bottom, tile.left : tile.right]

        target = None
        if self.target is not None:
            target = self.target[tile.top : tile.bottom, tile.left : tile.right]

        return image, target

    def transform(self, image, target):
        """Perform the transforms on the given images."""
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
