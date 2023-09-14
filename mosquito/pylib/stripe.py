from dataclasses import dataclass


@dataclass
class Stripe:
    dataset: str  # Train, val, test
    row: int  # Top pixel of stripe
    beg: int  # First column with data
    end: int  # Last column with data
