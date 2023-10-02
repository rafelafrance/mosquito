import logging

import numpy as np
import torch
import torchvision.transforms as xform
from torch import nn
from torch.utils.data import DataLoader

from . import stripe
from . import tile
from .simple_unet import SimpleUNet
from .tile_dataset import prepare_image
from .tile_dataset import TileDataset


def predict(args):
    device = torch.device("cuda" if torch.has_cuda else "cpu")
    model = SimpleUNet()
    model.to(device)

    load_model_state(model, args.load_model)

    layers, target = get_images(args)

    infer_loader = get_infer_loader(args, layers, target)

    loss_fn = nn.BCELoss()

    logging.info("Testing started")

    model.eval()
    test_loss = one_epoch(model, device, infer_loader, loss_fn, args.image_dir)

    logging.info(f"Test: loss {test_loss:0.6f}")


def one_epoch(model, device, loader, loss_fn, image_dir):
    running_loss = 0.0

    for images, y_true, indexes in loader:
        images = images.to(device)
        y_true = y_true.to(device)

        y_pred = model(images)

        loss = loss_fn(y_pred, y_true)

        running_loss += loss.item()

        if image_dir:
            y_pred = model.zeros_and_ones(y_pred)

            y_pred = y_pred.detach().cpu()
            y_true = y_true.detach().cpu()

            for idx, true, pred in zip(indexes, y_true, y_pred):
                path = image_dir / f"test_true_{idx:04d}.jpg"
                to_image(path, true)

                path = image_dir / f"test_pred_{idx:04d}.jpg"
                to_image(path, pred)

    return running_loss / len(loader)


def to_image(path, data):
    data = torch.squeeze(data)
    data = (255.0 * data).type(torch.uint8).numpy()
    image = xform.ToPILImage()(data)
    image.save(path)
    image.close()


def get_images(args):
    logging.info("Reading image data")

    layers = [prepare_image(p) for p in args.layer_path]
    layers = np.stack(layers, axis=0)

    target = prepare_image(args.target_file, target=True)

    return layers, target


def get_infer_loader(args, layers, target):
    logging.info("Loading test data")
    stripes = stripe.read_stripes(args.stripe_csv, "test")
    tiles = tile.get_tiles(
        stripes, stride=args.test_stride, limits=target.shape[1:], size=args.tile_size
    )
    dataset = TileDataset(tiles, layers, target)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )


def load_model_state(model, load_model):
    model.state = torch.load(load_model)
    if model.state.get("model_state"):
        logging.info("Loading a model")
        model.load_state_dict(model.state["model_state"])
