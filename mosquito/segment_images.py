#!/usr/bin/env python3
import argparse
import csv
import logging
import sys
import textwrap
from itertools import cycle
from pathlib import Path

import numpy as np
import sklearn.model_selection as m_sel
from PIL import Image
from pylib import log
from pylib import stripe
from pylib import tile
from pylib.tile_dataset import NA_HI
from pylib.tile_dataset import NA_LO
from tqdm import tqdm


def main():
    log.started()

    args = parse_args()

    train_indexes, val_indexes, test_indexes = get_rows(args)
    train_groups, val_groups, test_groups = get_row_groups(
        train_indexes, val_indexes, test_indexes
    )
    write_stripes(args, train_groups, val_groups, test_groups)

    log.finished()


def get_rows(args):
    all_rows = list(range(args.rows // tile.TILE_SIZE))

    train_indexes, others = m_sel.train_test_split(
        all_rows, train_size=args.train_split, random_state=next(args.seed)
    )
    val_indexes, test_indexes = m_sel.train_test_split(
        others, test_size=args.test_split, random_state=next(args.seed)
    )

    train_indexes = sorted(train_indexes)
    val_indexes = sorted(val_indexes)
    test_indexes = sorted(test_indexes)

    logging.info(f"All rows {len(all_rows)}")
    logging.info(f"Training rows {len(train_indexes)}")
    logging.info(f"Validation rows {len(val_indexes)}")
    logging.info(f"Testing rows {len(test_indexes)}")

    return train_indexes, val_indexes, test_indexes


def group_rows(indexes):
    """Group adjacent indexes

    Now that the row indices are assigned to the datasets I will group rows that are
    adjacent into one big row. So if row 24 & 25 are both assigned to the "val" dataset
    I group them into one bigger row. The groups are given as ranges so the 24 & 25 case
    is written as (24, 26); remember that Python ranges are open at the top
    Just to be clear, an ungrouped row like 0 is written as (0, 1).

    I do this so that I can squeeze out more tiles from each stripe. If we allow tiles
    to overlap then grouped rows will allow for many more tiles by allowing the tiles
    to float vertically. I should be careful with un-augmented tiles (val/test), and
    limit their overlap. I don't have to use every possible tile in a dataset.
    """
    group_beg = indexes[0]
    group_end = group_beg + 1

    grouped = []

    i = None
    for i in indexes[1:]:
        if i == group_end:
            group_end = i + 1
        else:
            grouped.append((group_beg, group_end))
            group_beg = i
            group_end = i + 1

    grouped.append((group_beg, i + 1))
    return grouped


def get_row_groups(train_indexes, val_indexes, test_indexes):
    train_groups = group_rows(train_indexes)
    val_groups = group_rows(val_indexes)
    test_groups = group_rows(test_indexes)
    return train_groups, val_groups, test_groups


# #### Tiles with data


def has_data(args, row, col):
    """Does the tile contain data?"""
    a_tile = args.images[0][row : row + tile.TILE_SIZE, col : col + tile.TILE_SIZE]
    flag = ((a_tile > NA_LO) & (a_tile < NA_HI)).any()  # noqa
    return flag


def build_stripes(args, groups, pixel_stride, dataset):
    """Build stripes from rows.

    A lot of the potential tiles are completely blank and I don't want to train on
    them, so I'll keep a record of where the data is in each row.
    """
    stripes = []
    for beg, end in tqdm(groups):
        top = beg * tile.TILE_SIZE
        bot = end * tile.TILE_SIZE
        for row in range(top, bot, pixel_stride):
            beg = 999999
            end = -999999
            for col in range(0, args.cols, pixel_stride):
                if has_data(args, row, col):
                    beg = min(beg, col)
                    end = max(end, col)
            stripes.append(stripe.Stripe(dataset, row, beg, end))
    return stripes


def write_stripes(args, train_groups, val_groups, test_groups):
    train_stripes = build_stripes(args, train_groups, pixel_stride=8, dataset="train")
    logging.info(f"Training stripes {len(train_stripes)}")

    val_stripes = build_stripes(args, val_groups, pixel_stride=8, dataset="val")
    logging.info("Validation stripes {len(val_stripes)}")

    test_stripes = build_stripes(args, test_groups, pixel_stride=8, dataset="test")
    logging.info(f"Testing stripes {len(test_stripes)}")

    all_stripes = train_stripes + val_stripes + test_stripes

    # I'll write this out so that I don't have to do this calculation over and over
    with open(args.stripe_csv, "w") as f:
        writer = csv.DictWriter(f, all_stripes[0].__dict__.keys())
        writer.writeheader()
        for s in all_stripes:
            writer.writerow(s.__dict__)


def parse_args():
    description = """
        Slice the large TIFF files into tiles for training, validation,
        and testing datasets.

        NOTE: All TIFF files must have the same dimensions.

        I only have one large image (with 4 layers) for training, validation,
        and testing. The strategy is to pretend that I've got several images by slicing
        the large image into several smaller images. These slices are called "stripes".
        Stripes are then further divided into tiles which may or may not overlap.

        Dataset distribution strategy:
            1. Slice the images into rows of tile sized data. These are stripes.
            2. Randomly assign rows to three datasets (train, val, & test).
            3. Divide each stripe into potential tiles. I allow for more tiles than
               what is probably use to allow for flexibility in how to use the tiles.
            4. Get information on each stripe:
                a. Where does the data for the stripe begin and end. A large part of
                   the stripe is marked as a NaN and we will avoid those.
            5. Save the stripes to a file so they will remain the same between runs.
        """
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(description),
        fromfile_prefix_chars="@",
    )

    arg_parser.add_argument(
        "--stripe-csv",
        "--stripes",
        type=Path,
        metavar="PATH",
        required=True,
        help="""Output the image stripe data to this CSV file.""",
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
        help="""The larval hatching area target file.""",
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
        "--tile-stride",
        "--stride",
        type=int,
        metavar="INT",
        default=8,
        help="""Tile stride for validation data. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--train-split",
        "--train",
        type=float,
        metavar="FRACTION",
        default=0.6,
        help="""What fraction of records to use for training the model.
            (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--val-split",
        "--val",
        type=float,
        metavar="FRACTION",
        default=0.2,
        help="""What fraction of records to use for the validation. I.e. evaluating
            training progress at the end of each epoch. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--test-split",
        "--test",
        type=float,
        metavar="FRACTION",
        default=0.2,
        help="""What fraction of records to use for testing. I.e. the holdout
            data used to evaluate the model after training. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--seed",
        type=int,
        action="append",
        metavar="INT",
        help="""Random number seed.""",
    )

    args = arg_parser.parse_args()

    # Validate images have the same shape
    validate_images(args)

    # Turn random seeds into a cycle. I used multiple seeds, sorry
    args.seed = cycle(args.seed if args.seed else [None])

    # Make sure splits add to 1.0
    if 1.0 != args.train_split + args.val_split + args.test_split:
        sys.exit("Splits do no add to one.")

    return args


def validate_images(args):
    image = get_image(args.target_file)
    rows, cols = image.shape
    setattr(args, "target_image", rows)
    setattr(args, "rows", rows)
    setattr(args, "cols", cols)
    setattr(args, "layer_images", [])
    for path in args.layer_path:
        image = get_image(path)
        rows, cols = image.shape
        if rows != args.rows or cols != args.cols:
            sys.exit(f"{args.target_file} and {path} shapes do not match.")


def get_image(path):
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(path) as img:
            image = np.array(img)  # noqa
        return np.array(image)
    except:  # noqa
        sys.exit(f"Could not open {path}.")


if __name__ == "__main__":
    main()
