#!/usr/bin/env python3
import argparse
import textwrap
from pathlib import Path

from pylib import log
from pylib import scorer
from pylib import tile


def main():
    log.started()

    args = parse_args()
    scorer.score(args)

    log.finished()


def parse_args():
    description = """
        Score a trained model on a holdout (test) dataset that predicts where
        mosquito larvae are likely to hatch.
        """
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
        "--load-model",
        "--load",
        type=Path,
        metavar="PATH",
        required=True,
        help="""Test this model.""",
    )

    arg_parser.add_argument(
        "--image-dir",
        type=Path,
        metavar="PATH",
        help="""Save image results to this directory.""",
    )

    arg_parser.add_argument(
        "--score-stride",
        "--stride",
        type=int,
        metavar="INT",
        default=tile.TILE_SIZE,
        help="""Tile stride for scoring data. (default: %(default)s)""",
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

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    main()
