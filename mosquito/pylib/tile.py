from dataclasses import dataclass

import numpy as np

TILE_SIZE = 512


@dataclass
class Tile:
    left: int
    top: int
    right: int
    bottom: int


def get_tiles(stripes, stride, limits, size=TILE_SIZE) -> list[Tile]:
    height, width = limits

    tiles = []
    prev_row = -9999

    for stripe in stripes:
        # Honor strides on the row too
        if stripe.row < prev_row + stride:
            continue

        prev_row = stripe.row

        # Get tiles in the row by stride
        for left in range(stripe.beg, stripe.end, stride):
            right = left + size
            bottom = stripe.row + size

            if right >= width or bottom >= height:
                continue

            tile = Tile(left=left, top=stripe.row, right=right, bottom=bottom)
            tiles.append(tile)

    return tiles
