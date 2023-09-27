import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryJaccardIndex

from . import stripe
from . import tile
from .iou_loss import IoULoss
from .simple_unet import SimpleUNet
from .tile_dataset import prepare_image
from .tile_dataset import TileDataset

# from torch import nn


@dataclass
class Stats:
    best: bool = False
    best_loss: float = float("Inf")
    train_loss: float = float("Inf")
    val_loss: float = float("Inf")
    best_iou: float = 0.0
    train_iou: float = 0.0
    val_iou: float = 0.0

    def is_best(self):
        self.best = False
        best = (-self.val_iou, self.val_loss) <= (-self.best_iou, self.best_loss)
        if best:
            self.best = True
            self.best_loss = self.val_loss
            self.best_iou = self.val_iou
        return best


def train(args):
    device = torch.device("cuda" if torch.has_cuda else "cpu")
    model = SimpleUNet()
    model.to(device)

    load_model_state(model, args.load_model)

    layers, target = get_images(args)

    train_loader = get_train_loader(args, layers, target)
    val_loader = get_val_loader(args, layers, target)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = get_loss_fn(train_loader.dataset, device)  # noqa
    metric = BinaryJaccardIndex().to(device)

    logging.info("Training started")

    stats = Stats(best_loss=model.state.get("best_loss", float("Inf")))

    end_epoch, start_epoch = get_epoch_range(args, model)

    writer = SummaryWriter(args.log_dir)

    for epoch in range(start_epoch, end_epoch):
        model.train()
        stats.train_loss, stats.train_iou = one_epoch(
            model, device, train_loader, loss_fn, metric, optimizer
        )

        model.eval()
        stats.val_loss, stats.val_iou = one_epoch(
            model, device, val_loader, loss_fn, metric
        )

        save_checkpoint(model, optimizer, args.save_model, stats, epoch)
        log_stats(writer, stats, epoch)

    writer.close()
    return stats


def one_epoch(model, device, loader, loss_fn, metric, optimizer=None):
    running_loss = 0.0
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

        y_pred = model.squash(y_pred)

        running_iou += metric(y_pred, y_true).item()
        running_loss += loss.item()

    return running_loss / len(loader), running_iou / len(loader)


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


def get_loss_fn(_dataset, _device):
    """Configure the loss_fn for model improvement."""
    # pos_weight = dataset.pos_weight()
    # pos_weight = torch.tensor(pos_weight, dtype=torch.float).to(device)
    # loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss_fn = IoULoss()
    return loss_fn


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
    """Load a previous model."""
    model.state = torch.load(load_model) if load_model else {}
    if model.state.get("model_state"):
        logging.info("Loading a model")
        model.load_state_dict(model.state["model_state"])


def save_checkpoint(model, optimizer, save_model, stats, epoch):
    """Save the model if it meets criteria for being the current best model."""
    if stats.is_best():
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_loss": stats.best_loss,
                "best_iou": stats.best_iou,
            },
            save_model,
        )


def log_stats(writer, stats, epoch):
    """Log results of the epoch."""
    logging.info(
        f"{epoch:4}: "
        f"Train: IoU {stats.train_iou:0.6f} loss {stats.train_loss:0.6f} "
        f"Valid: IoU {stats.val_iou:0.6f} loss {stats.val_loss:0.6f} "
        f"{'++' if stats.best else ''}"
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
