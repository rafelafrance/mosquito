from pathlib import Path

DATA_DIR = Path("..") / "data"
LAYER_DIR = DATA_DIR / "layers"

LAYERS = [
    LAYER_DIR / "dem.tif",
    LAYER_DIR / "fa.tif",
    LAYER_DIR / "slope.tif",
    LAYER_DIR / "wetness.tif",
]

TARGET = LAYER_DIR / "larv_spot_50m_correct.tif"

STRIPE_CSV = DATA_DIR / "stripes.csv"
