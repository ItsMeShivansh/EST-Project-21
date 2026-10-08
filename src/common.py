"""Shared paths, config loading and small helpers."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets"                  # one raw.nc per case (see datasets/README.md)
PROC = ROOT / "data" / "processed"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
for p in (PROC, FIG, TAB):
    p.mkdir(parents=True, exist_ok=True)

IGBP = {1: "Evergreen needleleaf forest", 2: "Evergreen broadleaf forest",
        3: "Deciduous needleleaf forest", 4: "Deciduous broadleaf forest",
        5: "Mixed forest", 6: "Closed shrubland", 7: "Open shrubland",
        8: "Woody savanna", 9: "Savanna", 10: "Grassland", 12: "Cropland",
        13: "Urban", 14: "Crop/natural mosaic", 16: "Barren", 17: "Water"}

# Coarse groups used for the ecosystem split
LC_GROUP = {1: "Forest", 2: "Forest", 3: "Forest", 4: "Forest", 5: "Forest",
            6: "Shrubland", 7: "Shrubland", 8: "Savanna", 9: "Savanna", 10: "Grassland"}


def load_config():
    with open(ROOT / "config" / "cases.yaml") as f:
        return yaml.safe_load(f)


def case_cfg(case_id):
    cfg = load_config()
    return cfg["cases"][case_id] | {"id": case_id}, cfg["common"]


def grid_coords(bbox, res):
    """Cell-centre coordinates aligned to the ERA5-Land grid (centres on multiples of res).
    lat is returned north -> south."""
    x0, y0, x1, y1 = bbox
    n_lon = int(round((x1 - x0) / res))
    n_lat = int(round((y1 - y0) / res))
    lon = np.round(x0 + res * np.arange(n_lon), 4)
    lat = np.round(y1 - res * np.arange(n_lat), 4)
    return lon, lat


def month_range(start, end):
    return [d.strftime("%Y-%m") for d in pd.date_range(start + "-01", end + "-01", freq="MS")]


def tslice(window):
    """Inclusive month window -> slice usable with ds.sel(time=...)."""
    return slice(pd.Timestamp(window[0] + "-01"),
                 pd.Timestamp(window[1] + "-01") + pd.offsets.MonthEnd(0))
