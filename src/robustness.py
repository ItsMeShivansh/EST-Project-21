"""Module F (light): dataset cross-checks and one threshold test."""
import json

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from . import sensitivity
from .common import PROC, TAB, case_cfg


def run(case_id, df=None):
    case, com = case_cfg(case_id)
    ds = xr.open_dataset(PROC / f"{case_id}.nc").load()
    if df is None:
        df = pd.read_csv(TAB / f"{case_id}_cells.csv")
    res = {}

    # ---- CHIRPS vs ERA5-Land monthly precipitation (all valid land cells, all months)
    a = ds["precip"].values.ravel()
    b = ds["tp_era5"].values.ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    rng = np.random.default_rng(0)
    idx = rng.choice(np.flatnonzero(ok), min(200_000, ok.sum()), replace=False)
    pd.DataFrame({"chirps": a[idx], "era5": b[idx]}).to_csv(
        TAB / f"{case_id}_F_precip_sample.csv", index=False)
    reg_c = ds["precip"].mean(["lat", "lon"]).values
    reg_e = ds["tp_era5"].mean(["lat", "lon"]).values
    res["precip"] = {"pixel_r": float(np.corrcoef(a[ok], b[ok])[0, 1]),
                     "regional_monthly_r": float(np.corrcoef(reg_c, reg_e)[0, 1]),
                     "bias_mm_month": float(np.mean(b[ok] - a[ok])),
                     "rel_bias_pct": float(100 * np.mean(b[ok] - a[ok]) / np.mean(a[ok]))}

    # ---- SPEI vs ERA5-Land soil moisture anomaly
    s, m = ds["spei03"].values.ravel(), ds["sm_z"].values.ravel()
    ok2 = np.isfinite(s) & np.isfinite(m)
    res["spei_vs_soil_moisture_r"] = float(np.corrcoef(s[ok2], m[ok2])[0, 1])

    # ---- MODIS active fire vs VIIRS, and burned area vs VIIRS
    v = df[df.valid]
    rho, p = stats.spearmanr(v.modis_fire_pixdays, v.viirs_count)
    has_v, has_b = v.viirs_count > 0, v.burned
    res["fire"] = {"modis_vs_viirs_spearman": float(rho), "p": float(p),
                   "burned_cells_with_viirs_pct": float(100 * (has_v & has_b).sum() / max(has_b.sum(), 1)),
                   "viirs_cells_burned_pct": float(100 * (has_v & has_b).sum() / max(has_v.sum(), 1))}

    # ---- threshold test: rerun Module C with the alternative event-1 threshold
    alt = case["event1_threshold_alt"]
    thr = -alt if case["first_extreme"] == "drought" else alt
    df2 = df.copy()
    df2["hit1_alt"] = df2.e1_sev >= thr
    r_alt = sensitivity.run(case_id, df2, hit_col="hit1_alt", tag="_alt")
    r_main = json.load(open(TAB / f"{case_id}_C_results.json"))
    res["threshold_test"] = {
        "main_threshold": case["event1_threshold"], "alt_threshold": alt,
        "main_diff": r_main["gpp_response"]["diff"], "main_p": r_main["gpp_response"]["mwu_p"],
        "alt_diff": r_alt["gpp_response"]["diff"], "alt_p": r_alt["gpp_response"]["mwu_p"],
        "same_sign": bool(np.sign(r_main["gpp_response"]["diff"]) == np.sign(r_alt["gpp_response"]["diff"])),
    }
    with open(TAB / f"{case_id}_F_results.json", "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(f"[F] {case_id}: precip r={res['precip']['pixel_r']:.2f}, "
          f"MODIS-VIIRS rho={rho:.2f}, threshold test same sign={res['threshold_test']['same_sign']}")
    return res
