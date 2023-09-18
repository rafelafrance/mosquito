import logging
from dataclasses import dataclass

import numpy as np
import torch
from skimage import io
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryJaccardIndex

from . import stripe
from . import tile
from .simple_unet import UNet
from .tile_dataset import TileDataset


@dataclass
class Stats:
    is_best: bool = False
    best_iou: float = 0.0
    best_loss: float = float("Inf")
    train_iou: float = 0.0
    train_loss: float = float("Inf")
    val_iou: float = 0.0
    val_loss: float = float("Inf")


def train(args):
    device = torch.device("cuda" if torch.has_cuda else "cpu")
    model = UNet()
    model.to(device)

    if args.load_model:
        load_model_state(model, args.load_model)

    logging.info("Reading image data data.")
    target = args.target_file
    layers = np.stack(
        [io.imread(lay) for lay in args.layer_dir if lay != target], axis=2
    )
    target = io.imread(target) if target else None

    train_loader = get_train_loader(args, layers, target)
    val_loader = get_val_loader(args, layers, target)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    all_epochs(args, model, device, train_loader, val_loader, loss_fn, optimizer)


def all_epochs(args, model, device, train_loader, val_loader, loss_fn, optimizer):
    logging.info("Training started.")

    stats = Stats(
        best_iou=model.state.get("iou", 0.0),
        best_loss=model.state.get("best_loss", float("Inf")),
    )

    jaccard = BinaryJaccardIndex()

    end_epoch, start_epoch = get_epoch_range(args, model)

    writer = SummaryWriter(args.log_dir)

    for epoch in range(start_epoch, end_epoch):
        model.train()
        stats.train_iou, stats.train_loss = one_epoch(
            model, device, train_loader, loss_fn, jaccard, optimizer
        )

        model.eval()
        stats.val_iou, stats.val_loss = one_epoch(
            model, device, val_loader, loss_fn, jaccard
        )

        save_checkpoint(model, optimizer, args.save_model, stats, epoch)
        log_stats(writer, stats, epoch)

    writer.close()
    return stats


def get_epoch_range(args, model):
    start_epoch = model.state.get("epoch", 0) + 1
    end_epoch = start_epoch + args.epochs
    return end_epoch, start_epoch


def one_epoch(model, device, loader, loss_fn, jaccard, optimizer=None):
    """Train or validate an epoch."""
    running_loss = 0.0
    running_iou = 0.0

    for images, y_true in loader:
        images = images.to(device)
        y_true = y_true.to(device)

        y_pred = model(images)

        loss = loss_fn(y_pred, y_true)

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        running_iou += jaccard(y_pred, y_true)

    return running_iou / len(loader), running_loss / len(loader)


def get_train_loader(args, layers, target):
    logging.info("Loading training data.")
    stripes = stripe.read_stripes(args.stripe_csv, "train")
    tiles = tile.get_tiles(stripes, stride=args.train_stride)
    dataset = TileDataset(tiles, layers, target, augment=True)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True,
        pin_memory=True,
        drop_last=len(dataset) % args.batch_size == 1,
    )


def get_val_loader(args, layers, target):
    logging.info("Loading validation data.")
    stripes = stripe.read_stripes(args.stripe_csv, "val")
    tiles = tile.get_tiles(stripes, stride=args.val_stride)
    dataset = TileDataset(tiles, layers, target)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )


def load_model_state(model, load_model):
    """Load a previous model."""
    model.state = torch.load(load_model) if load_model else {}
    if model.state.get("model_state"):
        logging.info("Loading a model.")
        model.load_state_dict(model.state["model_state"])


def save_checkpoint(model, optimizer, save_model, stats, epoch):
    """Save the model if it meets criteria for being the current best model."""
    stats.is_best = False
    if (stats.val_iou, -stats.val_loss) >= (stats.best_iou, -stats.best_loss):
        stats.is_best = True
        stats.best_iou = stats.val_iou
        stats.best_loss = stats.val_loss
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_loss": stats.best_loss,
                "iou": stats.best_iou,
            },
            save_model,
        )


def log_stats(writer, stats, epoch):
    """Log results of the epoch."""
    logging.info(
        f"{epoch:4}: "
        f"Train: loss {stats.train_loss:0.6f} IoU {stats.train_iou:0.6f} "
        f"Valid: loss {stats.val_loss:0.6f} IoU {stats.val_iou:0.6f}"
        f"{' ++' if stats.is_best else ''}"
    )
    writer.add_scalars(
        "Training vs. Validation",
        {
            "Training loss": stats.train_loss,
            "Training IoU": stats.train_iou,
            "Validation loss": stats.val_loss,
            "Validation IoU": stats.val_iou,
        },
        epoch,
    )
    writer.flush()
