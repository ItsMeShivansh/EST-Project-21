"""Module D: recovery after event 2, and the drought-memory lag plot. (H3, H4, H5)"""
import json
import warnings

import numpy as np
import pandas as pd
import ruptures as rpt
import xarray as xr
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.optimize import curve_fit

from .common import PROC, TAB, case_cfg, tslice

GROUPS = ["Double-hit", "Event 2 only", "Event 1 only", "Neither"]


def recovery_time(r, frac, k):
    """Months until r >= frac for k consecutive months. Returns (time, observed)."""
    ok = np.nan_to_num(r, nan=0) >= frac
    for t in range(len(ok) - k + 1):
        if ok[t:t + k].all():
            return t, True
    return len(ok), False


def fit_tau(r):
    t = np.arange(len(r))
    good = np.isfinite(r)
    if good.sum() < 12:
        return np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, _ = curve_fit(lambda t, a, tau: 1 - a * np.exp(-t / tau), t[good], r[good],
                             p0=[0.3, 12], bounds=([0, 0.5], [2, 240]), maxfev=4000)
        return p[1]
    except Exception:
        return np.nan


def state_shift(full_r, event_idx):
    """True if a breakpoint after the event leads to a lasting lower level (< 0.9, >= 24 months)."""
    x = pd.Series(full_r).interpolate().bfill().ffill().values
    if not np.isfinite(x).all():
        return False
    bk = rpt.Pelt(model="l2", min_size=12).fit(x).predict(pen=np.var(x) * 6)
    starts = [0] + bk[:-1]
    for s, e in zip(starts, bk):
        if s >= event_idx - 6 and e - s >= 24 and e == len(x) and x[s:e].mean() < 0.9:
            return True
    return False


def run(case_id, df=None):
    case, com = case_cfg(case_id)
    ds = xr.open_dataset(PROC / f"{case_id}.nc").load()
    if df is None:
        df = pd.read_csv(TAB / f"{case_id}_cells.csv")
    # burned groups use heavily burned cells only (see config: recovery_burned_frac)
    df = df.copy()
    df["group"] = np.select(
        [df.hit1 & df.heavy_burn, ~df.hit1 & df.heavy_burn, df.hit1 & df.unburned,
         ~df.hit1 & df.unburned],
        GROUPS, default="Other")
    cells = df[df.clean & df.group.isin(GROUPS)].copy()
    iy = ((ds.lat.values[0] - cells.lat) / 0.1).round().astype(int).values
    ix = ((cells.lon - ds.lon.values[0]) / 0.1).round().astype(int).values

    ratio = ds["ndvi_ratio"]
    rec = ratio.sel(time=slice(pd.Timestamp(case["recovery_start"] + "-01"), None)).values
    rec_gpp = ds["gpp_ratio"].sel(time=slice(pd.Timestamp(case["recovery_start"] + "-01"),
                                             None)).values
    full = ratio.values
    times = pd.DatetimeIndex(ratio.time.values)
    ev_idx = int(np.searchsorted(times, pd.Timestamp(case["recovery_start"] + "-01")))
    pre = ratio.sel(time=tslice(case["pre_window"])).mean("time").values
    dur = ratio.sel(time=tslice([case["fire_window"][0], case["recovery_start"]])).min("time").values
    post = ratio.sel(time=tslice(case["post_window"])).mean("time").values

    rows = []
    for (_, c), y, x in zip(cells.iterrows(), iy, ix):
        r = rec[:, y, x]
        tt, obs = recovery_time(r, com["recovery_fraction"], com["recovery_consecutive"])
        tg, obsg = recovery_time(rec_gpp[:, y, x], com["recovery_fraction"],
                                 com["recovery_consecutive"])
        burned = c.group in ("Double-hit", "Event 2 only")
        rows.append({
            "rec_months_gpp": tg, "recovered_gpp": obsg,
            "resistance": dur[y, x] / pre[y, x],
            "recovery": post[y, x] / dur[y, x],
            "resilience": post[y, x] / pre[y, x],
            "rec_months": tt, "recovered": obs,
            "tau": fit_tau(r) if burned else np.nan,
            "deficit": np.nansum(np.clip(1 - r, 0, None)),
            "last12": np.nanmean(r[-12:]),
            "shift": state_shift(full[:, y, x], ev_idx) if (burned and not obs) else False,
        })
    m = pd.concat([cells.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    m["status"] = np.select(
        [m.recovered, m["shift"], m.last12 >= 0.8],
        ["Recovered", "State shift", "Still recovering"], default="Not recovering")
    m.to_csv(TAB / f"{case_id}_D_recovery_cells.csv", index=False)

    # ---- group summaries
    summ = m.groupby("group").agg(
        n=("rec_months", "size"), resistance=("resistance", "median"),
        recovery=("recovery", "median"), resilience=("resilience", "median"),
        rec_months_median=("rec_months", "median"),
        rec_months_gpp_median=("rec_months_gpp", "median"),
        not_recovered_gpp_frac=("recovered_gpp", lambda s: 1 - s.mean()),
        tau_median=("tau", "median"),
        deficit_mean=("deficit", "mean"),
        not_recovered_frac=("recovered", lambda s: 1 - s.mean()),
        state_shift_frac=("shift", "mean")).reindex(GROUPS)
    summ.to_csv(TAB / f"{case_id}_D_recovery_summary.csv")

    # ---- Kaplan-Meier (median time to recovery) + log-rank tests
    km, tests = {}, {}
    curves = []
    for g in GROUPS:
        s = m[m.group == g]
        if len(s) < 5:
            continue
        k = KaplanMeierFitter().fit(s.rec_months, s.recovered, label=g)
        km[g] = float(k.median_survival_time_)
        sf = k.survival_function_.reset_index()
        sf.columns = ["months", "not_recovered"]
        curves.append(sf.assign(group=g))
    pd.concat(curves).to_csv(TAB / f"{case_id}_D_km_curves.csv", index=False)
    for a, b in [("Double-hit", "Event 2 only"), ("Event 1 only", "Neither")]:
        A, B = m[m.group == a], m[m.group == b]
        if len(A) >= 5 and len(B) >= 5:
            lr = logrank_test(A.rec_months, B.rec_months, A.recovered, B.recovered)
            tests[f"{a} vs {b}"] = {"p": float(lr.p_value), "stat": float(lr.test_statistic),
                                    "n": [len(A), len(B)]}

    # secondary contrast: heavily burned cells, top vs bottom tercile of event-1 severity
    burned = m[m.group.isin(["Double-hit", "Event 2 only"])]
    terc = {}
    for metric, ev in [("rec_months", "recovered"), ("rec_months_gpp", "recovered_gpp")]:
        A, B = burned[burned.tercile == "top"], burned[burned.tercile == "bottom"]
        if len(A) >= 5 and len(B) >= 5:
            lr = logrank_test(A[metric], B[metric], A[ev], B[ev])
            terc[metric] = {
                "n_top": len(A), "n_bottom": len(B), "p": float(lr.p_value),
                "median_top": float(KaplanMeierFitter().fit(A[metric], A[ev]).median_survival_time_),
                "median_bottom": float(KaplanMeierFitter().fit(B[metric], B[ev]).median_survival_time_),
                "not_recovered_top": float(1 - A[ev].mean()),
                "not_recovered_bottom": float(1 - B[ev].mean())}
            k = pd.concat([
                KaplanMeierFitter().fit(A[metric], A[ev]).survival_function_.reset_index()
                .set_axis(["months", "not_recovered"], axis=1).assign(group="Top third (most stressed)"),
                KaplanMeierFitter().fit(B[metric], B[ev]).survival_function_.reset_index()
                .set_axis(["months", "not_recovered"], axis=1).assign(group="Bottom third (least stressed)")])
            k.to_csv(TAB / f"{case_id}_D_km_terciles_{metric}.csv", index=False)

    # ---- mean NDVI-ratio trajectories per group (for the recovery-curve figure)
    traj = []
    t_all = ratio.sel(time=slice(pd.Timestamp(case["pre_window"][0] + "-01"), None))
    for g in GROUPS:
        s = m[m.group == g]
        if len(s) < 5:
            continue
        gy = ((ds.lat.values[0] - s.lat) / 0.1).round().astype(int).values
        gx = ((s.lon - ds.lon.values[0]) / 0.1).round().astype(int).values
        v = t_all.values[:, gy, gx]
        n = np.isfinite(v).sum(1)
        traj.append(pd.DataFrame({"time": t_all.time.values, "group": g,
                                  "mean": np.nanmean(v, 1),
                                  "se": np.nanstd(v, 1) / np.sqrt(np.maximum(n, 1))}))
    pd.concat(traj).to_csv(TAB / f"{case_id}_D_trajectories.csv", index=False)

    # ---- H5: recovery by ecosystem for burned cells
    m[m.group.isin(["Double-hit", "Event 2 only"])].groupby(["lc_group", "group"]).agg(
        n=("rec_months", "size"), rec_months_median=("rec_months", "median"),
        not_recovered_frac=("recovered", lambda s: 1 - s.mean())).to_csv(
        TAB / f"{case_id}_D_recovery_ecosystem.csv")

    # ---- H4: memory - correlation of SPEI-03 with GPP anomalies at lags 0-24 months
    v = df[df.valid]
    vy = ((ds.lat.values[0] - v.lat) / 0.1).round().astype(int).values
    vx = ((v.lon - ds.lon.values[0]) / 0.1).round().astype(int).values
    sp = ds["spei03"].values[:, vy, vx]
    gz = ds["gpp_z"].values[:, vy, vx]
    lag_rows = []
    for lag in range(25):
        a, b = sp[:len(sp) - lag], gz[lag:]
        am, bm = a - np.nanmean(a, 0), b - np.nanmean(b, 0)
        r = np.nansum(am * bm, 0) / np.sqrt(np.nansum(am ** 2, 0) * np.nansum(bm ** 2, 0))
        for g in sorted(v.lc_group.unique()):
            lag_rows.append({"lag": lag, "lc_group": g,
                             "r_median": float(np.nanmedian(r[(v.lc_group == g).values]))})
    pd.DataFrame(lag_rows).to_csv(TAB / f"{case_id}_D_lag_corr.csv", index=False)

    # mean GPP-ratio trajectories as well
    traj_g = []
    tg_all = ds["gpp_ratio"].sel(time=slice(pd.Timestamp(case["pre_window"][0] + "-01"), None))
    for g in GROUPS:
        s = m[m.group == g]
        if len(s) < 5:
            continue
        gy = ((ds.lat.values[0] - s.lat) / 0.1).round().astype(int).values
        gx = ((s.lon - ds.lon.values[0]) / 0.1).round().astype(int).values
        v = tg_all.values[:, gy, gx]
        n = np.isfinite(v).sum(1)
        traj_g.append(pd.DataFrame({"time": tg_all.time.values, "group": g,
                                    "mean": np.nanmean(v, 1),
                                    "se": np.nanstd(v, 1) / np.sqrt(np.maximum(n, 1))}))
    pd.concat(traj_g).to_csv(TAB / f"{case_id}_D_trajectories_gpp.csv", index=False)

    res = {"km_median_months": km, "logrank": tests, "terciles_burned": terc,
           "km_median_months_gpp": {
               g: float(KaplanMeierFitter().fit(m[m.group == g].rec_months_gpp,
                                                m[m.group == g].recovered_gpp).median_survival_time_)
               for g in GROUPS if (m.group == g).sum() >= 5},
           "status_share": m.groupby("group").status.value_counts(normalize=True)
           .unstack().reindex(GROUPS).to_dict(orient="index")}
    with open(TAB / f"{case_id}_D_results.json", "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(f"[D] {case_id}: KM median months {km}; log-rank {tests}")
    return m
