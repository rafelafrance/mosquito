import csv

import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms

from .stripe import Stripe


class LayerDataset(Dataset):
    def __init__(self, model, stripe_csv, layers, target=None, augment=False):
        self.model = model
        self.layers = np.stack([io.imread(lay) for lay in layers], axis=2)
        self.target = io.imread(target) if target else None
        self.transform = self.build_transforms(augment)

        with open(stripe_csv) as f:
            reader = csv.DictReader(f)
            self.stripes = [Stripe(**s) for s in reader]

    def __len__(self):
        ...

    def __getitem__(self, idx):
        ...

    def build_transforms(self, augment=False):
        """Build a pipeline of image transforms specific to the dataset."""
        xform = []

        if augment:
            xform += [
                transforms.AutoAugment(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]

        xform += [
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(self.model.mean, self.model.std_dev),
        ]

        return transforms.Compose(xform)

