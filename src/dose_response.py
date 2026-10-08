"""Module E: dose-response of event-1 severity, and a threshold (breakpoint) test. (H2)"""
import json

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess

from .common import TAB, case_cfg


def segmented(x, y, n_grid=60):
    """Piecewise-linear fit with one breakpoint (grid search). Compared to a straight line by F-test."""
    lin = np.polyfit(x, y, 1)
    rss_lin = np.sum((y - np.polyval(lin, x)) ** 2)
    best = (np.inf, None, None)
    for bp in np.quantile(x, np.linspace(0.1, 0.9, n_grid)):
        X = np.column_stack([np.ones_like(x), x, np.clip(x - bp, 0, None)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        rss = np.sum((y - X @ beta) ** 2)
        if rss < best[0]:
            best = (rss, bp, beta)
    rss_seg, bp, beta = best
    n = len(x)
    F = ((rss_lin - rss_seg) / 2) / (rss_seg / (n - 4))
    return {"breakpoint": float(bp), "slope_before": float(beta[1]),
            "slope_after": float(beta[1] + beta[2]), "F": float(F),
            "p": float(1 - stats.f.cdf(F, 2, n - 4)), "n": int(n)}


def boot_breakpoint(x, y, n, rng):
    bps = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        bps.append(segmented(x[i], y[i], 30)["breakpoint"])
    return np.percentile(bps, [2.5, 97.5])


def binned(df, xcol, ycol, bins):
    b = pd.cut(df[xcol], bins)
    g = df.groupby(b, observed=True)[ycol]
    out = g.agg(["mean", "count", "sem"]).reset_index()
    out["x_mid"] = [iv.mid for iv in out[xcol]]
    return out.drop(columns=xcol)


def run(case_id, df=None, rec=None):
    case, com = case_cfg(case_id)
    rng = np.random.default_rng(com["seed"])
    if df is None:
        df = pd.read_csv(TAB / f"{case_id}_cells.csv")
    if rec is None:
        rec = pd.read_csv(TAB / f"{case_id}_D_recovery_cells.csv")
    res = {}
    edges = np.quantile(df.loc[df.clean, "e1_sev"].dropna(), np.linspace(0, 1, 11))
    edges = np.unique(edges)

    outcomes = {
        "gpp_response": (df[df.clean & df.unburned].dropna(subset=["gpp_z_e2"]), "gpp_z_e2"),
        "burn_probability": (df[df.clean & (df.burned | df.unburned)].assign(
            y=lambda d: d.burned.astype(float)), "y"),
        "recovery_months": (rec[rec.group.isin(["Double-hit", "Event 2 only"])], "rec_months"),
    }
    for name, (d, col) in outcomes.items():
        d = d.dropna(subset=["e1_sev", col])
        if len(d) < 30:
            continue
        x, y = d.e1_sev.values, d[col].values.astype(float)
        binned(d, "e1_sev", col, edges).to_csv(TAB / f"{case_id}_E_{name}_bins.csv", index=False)
        lw = lowess(y, x, frac=0.4, it=0, return_sorted=True)
        pd.DataFrame(lw, columns=["x", "y"]).to_csv(TAB / f"{case_id}_E_{name}_lowess.csv",
                                                    index=False)
        seg = segmented(x, y)
        seg["bp_ci"] = boot_breakpoint(x, y, 200, rng).tolist()
        seg["spearman_rho"], seg["spearman_p"] = stats.spearmanr(x, y)
        res[name] = seg
    with open(TAB / f"{case_id}_E_results.json", "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(f"[E] {case_id}: " + "; ".join(
        f"{k}: bp={v['breakpoint']:.2f} p={v['p']:.3g} rho={v['spearman_rho']:.2f}"
        for k, v in res.items()))
    return res
