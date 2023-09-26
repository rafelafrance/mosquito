#!/usr/bin/env python3
import argparse
import textwrap
from pathlib import Path

from pylib import log
from pylib import model_trainer
from pylib import tile


def main():
    log.started()

    args = parse_args()
    model_trainer.train(args)

    log.finished()


def parse_args():
    description = """Train a model to find where mosquito larvae are likely to hatch."""
    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description), fromfile_prefix_chars="@"
    )

    arg_parser.add_argument(
        "--stripe-csv",
        "--stripes",
        type=Path,
        metavar="PATH",
        required=True,
        help="""Get image stripe data.""",
    )

    arg_parser.add_argument(
        "--layer-path",
        "--lay",
        type=Path,
        action="append",
        metavar="PATH",
        required=True,
        help="""Input layer image.""",
    )

    arg_parser.add_argument(
        "--target-file",
        "--target",
        type=Path,
        metavar="PATH",
        required=True,
        help="""The larval hatching area target image.""",
    )

    arg_parser.add_argument(
        "--save-model",
        "--save",
        type=Path,
        metavar="PATH",
        required=True,
        help="""Save best models to this path.""",
    )

    arg_parser.add_argument(
        "--load-model",
        "--load",
        type=Path,
        metavar="PATH",
        help="""Continue training with weights from this model.""",
    )

    arg_parser.add_argument(
        "--lr",
        type=float,
        metavar="FLOAT",
        default=0.00001,
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
        "--size",
        type=int,
        metavar="INT",
        default=tile.TILE_SIZE,
        help="""Tile height and width. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--batch-size",
        "--batch",
        type=int,
        metavar="INT",
        default=8,
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
        "--log-dir",
        type=Path,
        metavar="DIR",
        help="""Save tensorboard logs to this directory.""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    main()
