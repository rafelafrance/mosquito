from dataclasses import dataclass

TILE_SIZE = 512


@dataclass
class Tile:
    left: int
    top: int
    right: int
    bottom: int


def get_tiles(stripes, stride=TILE_SIZE) -> list[Tile]:
    tiles = []
    prev_row = -9999
    for stripe in stripes:
        # Honor strides on the row too
        if stripe.row < prev_row + stride:
            continue
        prev_row = stripe.row

        # Get tiles in the row by stride
        for left in range(stripe.beg, stripe.end, stride):
            tiles.append(
                Tile(
                    left=left,
                    top=stripe.row,
                    right=left + stride,
                    bottom=stripe.row + stride,
                )
            )

    return tiles
