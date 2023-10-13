import csv
from dataclasses import dataclass
from typing import Literal
from typing import TypeAlias

from .tile import TILE_SIZE

DatasetName: TypeAlias = Literal["train", "val", "test"]


@dataclass
class Stripe:
    """This class us used to mark where data is in a stripe."""

    dataset: DatasetName
    row: int  # Top pixel of stripe
    beg: int  # First column with data
    end: int  # Last column with data


def read_stripes(stripe_csv, dataset: DatasetName) -> list[Stripe]:
    stripes = []
    with open(stripe_csv) as f:
        reader = csv.DictReader(f)
        for row in [s for s in reader if s["dataset"] == dataset]:
            stripe = Stripe(
                dataset=row["dataset"],
                row=int(row["row"]),
                beg=int(row["beg"]),
                end=int(row["end"]),
            )
            stripes.append(stripe)
        return stripes


def filter_stripes(stripes, stride=TILE_SIZE):
    new = []
    prev = -999_999
    for s in stripes:
        if s.row < prev + stride:
            continue
        new.append(s)
        prev = s.row
    return new
