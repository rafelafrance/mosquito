import logging

import numpy as np
import torch
import torchvision.transforms as xform
from torch.utils.data import DataLoader

from . import stripe
from . import tile
from .binary_jaccard_loss import BinaryJaccardLoss
from .simple_unet import SimpleUNet
from .tile_dataset import prepare_image
from .tile_dataset import TileDataset


def score(args):
    device = torch.device("cuda" if torch.has_cuda else "cpu")
    model = SimpleUNet()
    model.to(device)

    load_model_state(model, args.load_model)

    layers, target = get_images(args)

    test_loader = get_score_loader(args, layers, target)

    loss_fn = BinaryJaccardLoss()

    logging.info("Scoring started")

    model.eval()
    score_iou = one_epoch(model, device, test_loader, loss_fn, args.image_dir)

    best_iou = model.state.get("best_iou")
    logging.info(f"Score: IoU {score_iou:0.6f} Best: IoU {best_iou:0.6f}")


def one_epoch(model, device, loader, loss_fn, image_dir):
    running_iou = 0.0

    for images, y_true, indexes in loader:
        images = images.to(device)
        y_true = y_true.to(device)

        y_pred = model(images)

        loss = loss_fn(y_pred, y_true)

        running_iou += 1.0 - loss.item()

        if image_dir:
            y_pred = model.zeros_and_ones(y_pred)

            y_pred = y_pred.detach().cpu()
            y_true = y_true.detach().cpu()

            for idx, true, pred in zip(indexes, y_true, y_pred):
                path = image_dir / f"score_true_{idx:04d}.jpg"
                to_image(path, true)

                path = image_dir / f"score_pred_{idx:04d}.jpg"
                to_image(path, pred)

    return running_iou / len(loader)


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


def get_score_loader(args, layers, target):
    logging.info("Loading scoring data")
    stripes = stripe.read_stripes(args.stripe_csv, "test")
    tiles = tile.get_tiles(
        stripes, stride=args.score_stride, limits=target.shape[1:], size=args.tile_size
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
