"""Module C: is the response to event 2 stronger where event 1 hit first? (H1, H5)"""
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.neighbors import NearestNeighbors

from .common import TAB, case_cfg


# controls are matched on event-2 stress, elevation and location (nearby cells), within land cover
MATCH_ON = ["e2_driver", "elev", "lat", "lon"]


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return (gt - lt) / (len(a) * len(b))


def match_controls(df, treat_col, covars, caliper=0.25):
    """1:1 nearest-neighbour matching (with replacement) within land-cover group.
    Returns (treated, control) frames of equal length."""
    t_rows, c_rows = [], []
    for g, sub in df.groupby("lc_group"):
        t, c = sub[sub[treat_col]], sub[~sub[treat_col]]
        if len(t) == 0 or len(c) == 0:
            continue
        X = sub[covars]
        mu, sd = X.mean(), X.std().replace(0, 1)
        nn = NearestNeighbors(n_neighbors=1).fit(((c[covars] - mu) / sd).values)
        dist, idx = nn.kneighbors(((t[covars] - mu) / sd).values)
        ok = dist[:, 0] <= caliper * np.sqrt(len(covars))
        t_rows.append(t[ok])
        c_rows.append(c.iloc[idx[ok, 0]])
    if not t_rows:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(t_rows), pd.concat(c_rows)


def compare(t, c, col, n_boot, rng):
    d = t[col].values - c[col].values
    boots = [rng.choice(d, len(d)).mean() for _ in range(n_boot)]
    u = stats.mannwhitneyu(t[col], c[col])
    return {"n_pairs": len(d), "mean_hit": t[col].mean(), "mean_control": c[col].mean(),
            "diff": d.mean(), "ci_lo": np.percentile(boots, 2.5),
            "ci_hi": np.percentile(boots, 97.5), "mwu_p": u.pvalue,
            "cliffs_delta": cliffs_delta(t[col], c[col])}


def run(case_id, df=None, hit_col="hit1", tag=""):
    case, com = case_cfg(case_id)
    rng = np.random.default_rng(com["seed"])
    if df is None:
        df = pd.read_csv(TAB / f"{case_id}_cells.csv")
    res = {"case": case_id}

    # ---- H1: GPP response to event 2 in unburned cells, hit vs matched control
    u = df[df.clean & df.unburned].dropna(subset=["gpp_z_e2", "e2_driver", "elev"]).copy()
    u["hit"] = u[hit_col].astype(bool)
    t, c = match_controls(u, "hit", MATCH_ON)
    res["gpp_response"] = compare(t, c, "gpp_z_e2", com["bootstrap"], rng)
    res["ndvi_response"] = compare(t, c, "ndvi_z_e2", com["bootstrap"], rng)

    # sensitivity ratio S = response / driver where the driver is clearly anomalous
    s = u[u.e2_driver > 0.5].copy()
    s["S"] = s.gpp_z_e2 / s.e2_driver
    ts, cs = match_controls(s, "hit", MATCH_ON)
    res["sensitivity_S"] = compare(ts, cs, "S", com["bootstrap"], rng) if len(ts) > 5 else None

    # interaction model: does GPP fall faster per unit of event-2 stress in hit cells?
    u["hit_i"] = u.hit.astype(int)
    m = smf.ols("gpp_z_e2 ~ e2_driver * hit_i + C(lc_group) + elev", data=u).fit(
        cov_type="cluster", cov_kwds={"groups": u["block"]})
    res["interaction"] = {"coef_driver": m.params["e2_driver"],
                          "coef_hit": m.params["hit_i"],
                          "coef_driver_x_hit": m.params["e2_driver:hit_i"],
                          "p_driver_x_hit": m.pvalues["e2_driver:hit_i"],
                          "p_hit": m.pvalues["hit_i"], "n": int(m.nobs)}

    # ---- H5: ecosystem split
    eco = []
    for g in sorted(t.lc_group.unique()):
        tg, cg = t[t.lc_group == g], c[c.lc_group == g]
        if len(tg) >= 10:
            eco.append({"lc_group": g, **compare(tg, cg, "gpp_z_e2", com["bootstrap"], rng)})
    pd.DataFrame(eco).to_csv(TAB / f"{case_id}_C_ecosystem{tag}.csv", index=False)

    # ---- fire: logistic regression, burned vs unburned
    f = df[df.clean & (df.burned | df.unburned)].dropna(
        subset=["e1_sev", "vpd_z_fire", "sm_z_fire", "ndvi_z_pre", "elev"]).copy()
    f["y"] = f.burned.astype(int)
    # land-cover groups with almost no fire cause perfect separation; model where fire happens
    nb = f.groupby("lc_group").y.sum()
    f = f[f.lc_group.isin(nb[nb >= 20].index)]
    preds = ["e1_sev", "vpd_z_fire", "sm_z_fire", "ndvi_z_pre", "elev"]
    for p in preds:
        f[p + "_s"] = (f[p] - f[p].mean()) / f[p].std()
    if f.y.sum() >= 20:
        lc_term = " + C(lc_group)" if f.lc_group.nunique() > 1 else ""
        lg = smf.logit("y ~ " + " + ".join(p + "_s" for p in preds) + lc_term, data=f).fit(
            disp=0, cov_type="cluster", cov_kwds={"groups": f["block"]})
        ci = lg.conf_int()
        orr = pd.DataFrame({"predictor": preds,
                            "odds_ratio": np.exp(lg.params[[p + "_s" for p in preds]].values),
                            "ci_lo": np.exp(ci.loc[[p + "_s" for p in preds], 0].values),
                            "ci_hi": np.exp(ci.loc[[p + "_s" for p in preds], 1].values),
                            "p": lg.pvalues[[p + "_s" for p in preds]].values})
        orr.to_csv(TAB / f"{case_id}_C_fire_logit{tag}.csv", index=False)
        res["fire_logit"] = {"n": int(lg.nobs), "n_burned": int(f.y.sum()),
                             "lc_groups": sorted(f.lc_group.unique()),
                             "pseudo_r2": lg.prsquared,
                             "OR_e1_sev": float(orr.odds_ratio[0]), "p_e1_sev": float(orr.p[0])}
        # total effect of event 1: do not adjust for fire-season soil moisture / VPD / pre-fire
        # NDVI, which are pathways through which event 1 acts (mediators)
        tot = smf.logit("y ~ e1_sev_s + elev_s" + lc_term, data=f).fit(
            disp=0, cov_type="cluster", cov_kwds={"groups": f["block"]})
        ci_t = tot.conf_int().loc["e1_sev_s"]
        res["fire_logit_total"] = {"OR_e1_sev": float(np.exp(tot.params["e1_sev_s"])),
                                   "ci_lo": float(np.exp(ci_t[0])), "ci_hi": float(np.exp(ci_t[1])),
                                   "p_e1_sev": float(tot.pvalues["e1_sev_s"]),
                                   "pseudo_r2": float(tot.prsquared)}
        hit_rate = f.groupby(f[hit_col].astype(bool)).y.mean()
        res["burn_rate"] = {"hit": float(hit_rate.get(True, np.nan)),
                            "not_hit": float(hit_rate.get(False, np.nan))}

    pd.concat([t.assign(role="hit"), c.assign(role="control")]).to_csv(
        TAB / f"{case_id}_C_matched{tag}.csv", index=False)
    with open(TAB / f"{case_id}_C_results{tag}.json", "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    g = res["gpp_response"]
    print(f"[C] {case_id}{tag}: GPP_z hit {g['mean_hit']:.2f} vs control {g['mean_control']:.2f} "
          f"(diff {g['diff']:.2f} [{g['ci_lo']:.2f},{g['ci_hi']:.2f}], p={g['mwu_p']:.3g}, "
          f"n={g['n_pairs']})")
    return res


def run_terciles(case_id, df):
    """Secondary contrast: top vs bottom third of event-1 severity (middle third dropped)."""
    d = df[df.tercile != "middle"].copy()
    d["hit_top"] = d.tercile == "top"
    return run(case_id, d, hit_col="hit_top", tag="_terciles")
