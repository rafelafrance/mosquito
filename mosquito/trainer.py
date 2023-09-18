#!/usr/bin/env python3
import argparse
import textwrap
from pathlib import Path

from pylib import log
from pylib import stripe
from pylib import tile
from pylib import trainer_engine


def main():
    log.started()

    args = parse_args()
    trainer_engine.train(args)

    test_stripes = stripe.read_stripes(args.stripe_csv, "test")
    test_tiles = tile.get_tiles(test_stripes)
    print(len(test_stripes))
    print(len(test_tiles))
    print()

    val_stripes = stripe.read_stripes(args.stripe_csv, "val")
    val_tiles = tile.get_tiles(val_stripes)
    print(len(val_stripes))
    print(len(val_tiles))
    print()

    train_stripes = stripe.read_stripes(args.stripe_csv, "train")
    train_tiles = tile.get_tiles(train_stripes, stride=192)
    print(len(train_stripes))
    print(len(train_tiles))
    print()

    log.finished()


def parse_args():
    description = """Train a model to find where mosquito larvae are likely to hatch."""
    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description), fromfile_prefix_chars="@"
    )

    arg_parser.add_argument(
        "--stripe-csv",
        type=Path,
        metavar="PATH",
        required=True,
        help="""Get image stripe data.""",
    )

    arg_parser.add_argument(
        "--layer-dir",
        type=Path,
        metavar="PATH",
        required=True,
        help="""Infer larval hatching areas from these images.""",
    )

    arg_parser.add_argument(
        "--target-file",
        type=Path,
        metavar="PATH",
        help="""The larval hatching area target results.""",
    )

    arg_parser.add_argument(
        "--save-model",
        type=Path,
        metavar="PATH",
        required=True,
        help="""Save best models to this path.""",
    )

    arg_parser.add_argument(
        "--load-model",
        type=Path,
        metavar="PATH",
        help="""Continue training with weights from this model.""",
    )

    arg_parser.add_argument(
        "--lr",
        type=float,
        metavar="FLOAT",
        default=0.001,
        help="""Initial learning rate. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--train-stride",
        type=int,
        metavar="INT",
        default=192,
        help="""Tile stride for training data. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--val-stride",
        type=int,
        metavar="INT",
        default=256,
        help="""Tile stride for validation data. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--tile-size",
        type=int,
        metavar="INT",
        default=tile.TILE_SIZE,
        help="""Tile height and width. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--batch-size",
        type=int,
        metavar="INT",
        default=16,
        help="""Input batch size. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--workers",
        type=int,
        metavar="INT",
        default=4,
        help="""Number of workers for loading data. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--epochs",
        type=int,
        metavar="INT",
        default=100,
        help="""How many epochs to train. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--limit",
        type=int,
        metavar="INT",
        help="""Limit the input to this many records.""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    main()
