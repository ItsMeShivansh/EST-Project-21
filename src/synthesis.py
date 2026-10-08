"""Module G: cross-case synthesis table and policy implications."""
import json

import numpy as np
import pandas as pd

from .common import TAB, load_config

POLICY = {
    "drought": "Use drought maps (SPEI) as an early fire-risk layer: prioritize fuel reduction and "
               "fire-readiness where prior drought was most severe.",
    "heat": "After a heatwave, treat the hardest-hit forests as high-risk for the rest of the "
            "season: monitor them and restrict activities that could start fires.",
    "fire": "Protect recovering burn scars during later droughts; prioritize replanting where "
            "recovery has stalled.",
    "rain": "After unusually wet seasons in arid regions, issue grass-fire warnings and burn fuel "
            "in advance.",
}


def run(case_ids):
    cfg = load_config()
    rows = []
    for cid in case_ids:
        case = cfg["cases"][cid]
        C = json.load(open(TAB / f"{cid}_C_results.json"))
        D = json.load(open(TAB / f"{cid}_D_results.json"))
        E = json.load(open(TAB / f"{cid}_E_results.json"))
        F = json.load(open(TAB / f"{cid}_F_results.json"))
        T = json.load(open(TAB / f"{cid}_C_results_terciles.json"))
        g = C["gpp_response"]
        km = D["km_median_months"]
        dh = D["status_share"].get("Double-hit") or {}
        e2 = D["status_share"].get("Event 2 only") or {}
        rows.append({
            "case": case["short"],
            "first_extreme": case["first_extreme"],
            "gpp_diff_hit_minus_control": g["diff"], "gpp_ci_lo": g["ci_lo"], "gpp_ci_hi": g["ci_hi"],
            "gpp_p": g["mwu_p"], "cliffs_delta": g["cliffs_delta"], "n_pairs": g["n_pairs"],
            "interaction_coef": C["interaction"]["coef_driver_x_hit"],
            "interaction_p": C["interaction"]["p_driver_x_hit"],
            "burn_OR_per_SD_e1": (C.get("fire_logit_total") or {}).get("OR_e1_sev", np.nan),
            "burn_OR_ci_lo": (C.get("fire_logit_total") or {}).get("ci_lo", np.nan),
            "burn_OR_ci_hi": (C.get("fire_logit_total") or {}).get("ci_hi", np.nan),
            "burn_OR_p": (C.get("fire_logit_total") or {}).get("p_e1_sev", np.nan),
            "burn_rate_top_third": (T.get("burn_rate") or {}).get("hit", np.nan),
            "burn_rate_bottom_third": (T.get("burn_rate") or {}).get("not_hit", np.nan),
            "gpp_diff_terciles": T["gpp_response"]["diff"],
            "gpp_terciles_ci_lo": T["gpp_response"]["ci_lo"],
            "gpp_terciles_ci_hi": T["gpp_response"]["ci_hi"],
            "km_ndvi_median_top_third": (D.get("terciles_burned") or {}).get("rec_months", {}).get("median_top", np.nan),
            "km_ndvi_median_bottom_third": (D.get("terciles_burned") or {}).get("rec_months", {}).get("median_bottom", np.nan),
            "km_terciles_p": (D.get("terciles_burned") or {}).get("rec_months", {}).get("p", np.nan),
            "km_gpp_median_double_hit": D.get("km_median_months_gpp", {}).get("Double-hit", np.nan),
            "burn_rate_hit": (C.get("burn_rate") or {}).get("hit", np.nan),
            "burn_rate_not_hit": (C.get("burn_rate") or {}).get("not_hit", np.nan),
            "km_median_double_hit": km.get("Double-hit", np.nan),
            "km_median_event2_only": km.get("Event 2 only", np.nan),
            "logrank_p_burned": D["logrank"].get("Double-hit vs Event 2 only", {}).get("p", np.nan),
            "not_recovered_double_hit": 1 - dh.get("Recovered", np.nan) if dh else np.nan,
            "not_recovered_event2_only": 1 - e2.get("Recovered", np.nan) if e2 else np.nan,
            "threshold_breakpoint_gpp": (E.get("gpp_response") or {}).get("breakpoint", np.nan),
            "threshold_p_gpp": (E.get("gpp_response") or {}).get("p", np.nan),
            "robust_to_threshold": F["threshold_test"]["same_sign"],
            "policy": POLICY[case["first_extreme"]],
        })
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "G_synthesis.csv", index=False)
    print("[G] synthesis:\n" + out.drop(columns="policy").T.to_string())
    return out
