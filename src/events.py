"""Module B: detect drought, heat and fire events; build the per-cell event table."""
import numpy as np
import pandas as pd
import xarray as xr

from .common import PROC, TAB, case_cfg, tslice


def heatwave_days(hot):
    """hot: bool (day, lat, lon). Count days belonging to runs of >= 3 hot days."""
    n = hot.shape[0]
    run = np.zeros(hot.shape[1:], int)
    out = np.zeros(hot.shape[1:], int)
    for t in range(n + 1):
        h = hot[t] if t < n else np.zeros_like(run, bool)
        ended = ~h & (run >= 3)
        out[ended] += run[ended]
        run = np.where(h, run + 1, 0)
    return out


def daily_heat(ds, d0, d1):
    """Heatwave days and peak daily Tmax anomaly (SD units) between two dates."""
    td = ds["tmax_day"].sel(day=slice(d0, d1))
    m = td["day"].dt.month.values
    p90 = ds["tmax_p90"].sel(month=xr.DataArray(m, dims="day")).values
    mu = ds["tmax_dmean"].sel(month=xr.DataArray(m, dims="day")).values
    sd = ds["tmax_dsd"].sel(month=xr.DataArray(m, dims="day")).values
    v = td.values
    hot = np.nan_to_num(v > p90)
    z = (v - mu) / sd
    return heatwave_days(hot.astype(bool)), np.nanmax(z, axis=0)


def window_mean(da, w):
    return da.sel(time=tslice(w)).mean("time").values


def run(case_id):
    case, com = case_cfg(case_id)
    ds = xr.open_dataset(PROC / f"{case_id}.nc").load()
    f = {}

    # ---- drought (SPEI)
    e1 = ds["spei12"].sel(time=tslice(case["event1"]))
    f["spei12_min_e1"] = e1.min("time").values
    f["drought_months_e1"] = (e1 <= -1.5).sum("time").values.astype(float)
    f["spei03_e2"] = window_mean(ds["spei03"], case["event2"])
    f["spei12_e2"] = window_mean(ds["spei12"], case["event2"])

    # ---- heat (ERA5-Land daily Tmax for heatwaves, monthly anomalies for intensity)
    e2d0 = pd.Timestamp(case["event2"][0] + "-01")
    e2d1 = pd.Timestamp(case["event2"][1] + "-01") + pd.offsets.MonthEnd(0)
    f["hw_days_e2"], f["tmax_peakz_e2"] = daily_heat(ds, e2d0, e2d1)
    if "event1_days" in case:
        f["hw_days_e1"], f["tmax_peakz_e1"] = daily_heat(ds, *case["event1_days"])
    for v in ["tmax_z", "vpd_z", "sm_z", "precip_z", "gpp_z", "ndvi_z"]:
        f[f"{v}_e2"] = window_mean(ds[v], case["event2"])
        f[f"{v}_fire"] = window_mean(ds[v], case["fire_window"])

    # vegetation state just before the fire window
    fw0 = pd.Timestamp(case["fire_window"][0] + "-01")
    f["ndvi_z_pre"] = ds["ndvi_z"].sel(time=slice(fw0 - pd.DateOffset(months=3),
                                                  fw0 - pd.DateOffset(days=1))).mean("time").values

    # ---- fire
    f["burn_frac"] = ds["burn"].sel(time=tslice(case["fire_window"])).sum("time").clip(max=1).values
    e1d0 = pd.Timestamp(case["event1"][0] + "-01")
    f["prior_burn"] = ds["burn"].sel(time=slice(e1d0 - pd.DateOffset(years=3),
                                                e1d0 - pd.DateOffset(days=1))
                                     ).sum("time").clip(max=1).values
    for v in ["modis_frp_max", "modis_fire_pixdays", "viirs_count", "viirs_frp_sum"]:
        f[v] = ds[v].values

    # ---- event-1 severity (bigger = more severe) and hit flag
    if case["first_extreme"] == "drought":
        sev = -f["spei12_min_e1"]
        thr = -case["event1_threshold"]
    elif case["first_extreme"] == "heat":
        sev = f["tmax_peakz_e1"]
        thr = case["event1_threshold"]
    else:
        raise NotImplementedError(case["first_extreme"])
    f["e1_sev"] = sev

    # ---- event-2 stress driver (bigger = more stress)
    f["e2_driver"] = f["tmax_z_e2"] if case["first_extreme"] == "drought" else -f["spei03_e2"]

    lat, lon = np.meshgrid(ds.lat.values, ds.lon.values, indexing="ij")
    df = pd.DataFrame({k: np.ravel(v) for k, v in f.items()})
    df["lat"], df["lon"] = lat.ravel(), lon.ravel()
    df["elev"] = ds["elev"].values.ravel()
    df["lc_group"] = ds["lc_group"].values.ravel()
    df["valid"] = ds["valid"].values.ravel()
    df["hit1"] = df["e1_sev"] >= thr
    df["burned"] = df["burn_frac"] >= com["burned_frac"]
    df["heavy_burn"] = df["burn_frac"] >= com["recovery_burned_frac"]
    df["unburned"] = df["burn_frac"] < com["unburned_frac"]
    df["clean"] = df["valid"] & (df["prior_burn"] < com["unburned_frac"]) & df["e1_sev"].notna()
    # secondary contrast for region-wide events: top vs bottom tercile of event-1 severity,
    # computed within each land-cover group so the contrast is not just a vegetation contrast
    q = df[df.clean].groupby("lc_group")["e1_sev"].quantile([1 / 3, 2 / 3]).unstack()
    lo = df.lc_group.map(q[1 / 3])
    hi = df.lc_group.map(q[2 / 3])
    df["tercile"] = np.select([df.e1_sev >= hi, df.e1_sev <= lo], ["top", "bottom"], "middle")
    df["group"] = np.select(
        [df.hit1 & df.burned, ~df.hit1 & df.burned, df.hit1 & df.unburned, ~df.hit1 & df.unburned],
        ["Double-hit", "Event 2 only", "Event 1 only", "Neither"], default="Partial burn")
    df["block"] = (np.floor(df.lat).astype(int).astype(str) + "_"
                   + np.floor(df.lon).astype(int).astype(str))  # 1 deg blocks for clustered SEs
    df.to_csv(TAB / f"{case_id}_cells.csv", index=False)

    c = df[df.clean]
    summary = {
        "cells_clean": len(c),
        "hit1": int(c.hit1.sum()),
        "burned": int(c.burned.sum()),
        **{f"group: {k}": int(v) for k, v in c.group.value_counts().items()},
    }
    pd.Series(summary).to_csv(TAB / f"{case_id}_event_counts.csv", header=["n"])
    print(f"[B] {case_id}: {summary}")
    return df
