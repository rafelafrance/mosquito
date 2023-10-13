import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from . import stripe
from . import tile
from .binary_jaccard_loss import BinaryJaccardLoss
from .simple_unet import SimpleUNet
from .tile_dataset import prepare_image
from .tile_dataset import TileDataset


@dataclass
class Stats:
    best: bool = False
    best_iou: float = 0.0
    train_iou: float = 0.0
    val_iou: float = 0.0

    def is_best(self):
        self.best = self.val_iou >= self.best_iou
        if self.best:
            self.best_iou = self.val_iou
        return self.best


def train(args):
    device = torch.device("cuda" if torch.has_cuda else "cpu")
    model = SimpleUNet()
    model.to(device)

    load_model_state(model, args.load_model)

    layers, target = get_images(args)

    train_loader = get_train_loader(args, layers, target)
    val_loader = get_val_loader(args, layers, target)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = BinaryJaccardLoss()

    logging.info("Training started")

    stats = Stats(best_iou=model.state.get("best_iou", 0.0))

    end_epoch, start_epoch = get_epoch_range(args, model)

    for epoch in range(start_epoch, end_epoch):
        model.train()
        stats.train_iou = one_epoch(model, device, train_loader, loss_fn, optimizer)

        model.eval()
        stats.val_iou = one_epoch(model, device, val_loader, loss_fn)

        save_checkpoint(model, optimizer, args.save_model, stats, epoch)
        log_stats(stats, epoch)

    return stats


def one_epoch(model, device, loader, loss_fn, optimizer=None):
    running_iou = 0.0

    for images, y_true, _ in loader:
        images = images.to(device)
        y_true = y_true.to(device)

        y_pred = model(images)

        loss = loss_fn(y_pred, y_true)

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_iou += 1.0 - loss.item()  # Reverse to put into bigger is better mode

    return running_iou / len(loader)


def get_images(args):
    logging.info("Reading image data")

    layers = [prepare_image(p) for p in args.layer_path]
    layers = np.stack(layers, axis=0)

    target = prepare_image(args.target_file, target=True) if args.target_file else None

    return layers, target


def get_epoch_range(args, model):
    start_epoch = model.state.get("epoch", 0) + 1
    end_epoch = start_epoch + args.epochs
    return end_epoch, start_epoch


def get_train_loader(args, layers, target):
    logging.info("Loading training data")
    stripes = stripe.read_stripes(args.stripe_csv, "train")
    tiles = tile.get_tiles(
        stripes, stride=args.train_stride, limits=target.shape[1:], size=args.tile_size
    )
    dataset = TileDataset(tiles, layers, target, augment=True)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True,
        pin_memory=True,
    )


def get_val_loader(args, layers, target):
    logging.info("Loading validation data")
    stripes = stripe.read_stripes(args.stripe_csv, "val")
    tiles = tile.get_tiles(
        stripes, stride=args.val_stride, limits=target.shape[1:], size=args.tile_size
    )
    dataset = TileDataset(tiles, layers, target)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )


def load_model_state(model, load_model):
    model.state = torch.load(load_model) if load_model else {}
    if model.state.get("model_state"):
        logging.info("Loading a model")
        model.load_state_dict(model.state["model_state"])


def save_checkpoint(model, optimizer, save_model, stats, epoch):
    if stats.is_best():
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_iou": stats.best_iou,
            },
            save_model,
        )


def log_stats(stats, epoch):
    logging.info(
        f"{epoch:4}: "
        f"Train: IoU {stats.train_iou:0.6f} "
        f"Valid: IoU {stats.val_iou:0.6f} "
        f"{'++' if stats.best else ''}"
    )
