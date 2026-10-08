"""All figures (1-15). Static PNGs for the slides and report."""
import json
import textwrap

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from .common import FIG, PROC, TAB, case_cfg, load_config, tslice  # noqa: E402

# ---- palette (validated reference palette; slots used in fixed order)
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, RED, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#4a3aa7"
NEUTRAL = "#b5b3ac"
GROUP_COLOR = {"Double-hit": ORANGE, "Event 2 only": BLUE, "Event 1 only": AQUA, "Neither": NEUTRAL}
GROUPS = list(GROUP_COLOR)
STATUS_COLOR = {"Recovered": "#0ca30c", "Still recovering": "#fab219",
                "Not recovering": "#ec835a", "State shift": "#d03b3b"}
DIVERGING = LinearSegmentedColormap.from_list(
    "div", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f3a3a2", "#e34948", "#8f1d1c"])
SEQ_BLUE = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#9ec5f4", "#2a78d6", "#0d366b"])
SEQ_RED = LinearSegmentedColormap.from_list("seqr", ["#f0efec", "#f3a3a2", "#e34948", "#8f1d1c"])

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2, "figure.dpi": 110,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig {name}")


def na_panel(ax, msg):
    ax.set_axis_off()
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, fontsize=10,
            color=INK2, wrap=True)


def log_ticks(ax, ticks=(0.25, 0.5, 1, 2, 4, 8)):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    lo, hi = ax.get_xlim()
    t = [v for v in ticks if lo <= v <= hi]
    ax.xaxis.set_major_locator(FixedLocator(t))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def load(case_id):
    ds = xr.open_dataset(PROC / f"{case_id}.nc")
    df = pd.read_csv(TAB / f"{case_id}_cells.csv")
    return ds, df


def geo_ax(fig, pos, bbox):
    ax = fig.add_subplot(*pos, projection=ccrs.PlateCarree())
    ax.set_extent([bbox[0], bbox[2], bbox[1], bbox[3]], ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6, edgecolor=INK2)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3, edgecolor=MUTED)
    return ax


def to_map(ds, df, col):
    grid = np.full((ds.sizes["lat"], ds.sizes["lon"]), np.nan)
    iy = ((ds.lat.values[0] - df.lat) / 0.1).round().astype(int).values
    ix = ((df.lon - ds.lon.values[0]) / 0.1).round().astype(int).values
    grid[iy, ix] = df[col].values
    return grid


def mesh(ds, ax, arr, **kw):
    return ax.pcolormesh(ds.lon, ds.lat, arr, transform=ccrs.PlateCarree(), shading="nearest", **kw)


def shade(ax, w, color, label=None):
    a = pd.Timestamp(w[0] + "-01")
    b = pd.Timestamp(w[1] + "-01") + pd.offsets.MonthEnd(0)
    ax.axvspan(a, b, color=color, alpha=0.18, lw=0, label=label)


# ===================================================================== 1-3 (method figures)
def fig01_concept():
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.axis("off")
    steps = [("Event 1", "drought / heat /\nfire / extreme rain", BLUE),
             ("Weakened\necosystem", "low reserves, stressed\ncanopy, more fuel", NEUTRAL),
             ("Event 2", "the next\ndisturbance", ORANGE),
             ("Amplified\ndamage", "bigger GPP / NDVI drop,\nmore burning", RED),
             ("Slow or no\nrecovery", "longer recovery,\npossible state shift", VIOLET)]
    for i, (t, s, c) in enumerate(steps):
        x = i * 2.2
        ax.add_patch(FancyBboxPatch((x, 0.35), 1.8, 1.2, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=c, ec="none", alpha=0.9))
        ax.text(x + 0.9, 1.08, t, ha="center", va="center", color="white", fontsize=11, weight="bold")
        ax.text(x + 0.9, 0.0, s, ha="center", va="center", color=INK2, fontsize=8.5)
        if i < len(steps) - 1:
            ax.annotate("", (x + 2.15, 0.95), (x + 1.85, 0.95),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.5))
    ax.text(0, 1.9, "Cascading climate stress: does one extreme increase vulnerability to the next?",
            fontsize=12, weight="bold", color=INK)
    ax.text(0, -0.55, "Test: compare cells hit by event 1 with matched cells that were not, "
                      "during event 2 and in the recovery afterwards.", fontsize=9, color=INK2)
    ax.set_xlim(-0.1, 10.9)
    ax.set_ylim(-0.8, 2.1)
    save(fig, "01_concept")


def fig02_regions(case_ids):
    cfg = load_config()
    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.LAND, fc="#efeee9")
    ax.add_feature(cfeature.COASTLINE, lw=0.4, ec=MUTED)
    colors = [ORANGE, BLUE, AQUA, YELLOW]
    for i, cid in enumerate(cfg["cases"]):
        c = cfg["cases"][cid]
        x0, y0, x1, y1 = c["bbox"]
        done = cid in case_ids
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, transform=ccrs.PlateCarree(),
                               fc=colors[i] if done else "none", ec=colors[i], lw=2, alpha=0.9))
        ax.text(x1 + 2, y1, f"Case {i + 1}: {c['short']}", transform=ccrs.PlateCarree(),
                fontsize=9, color=INK, weight="bold")
    ax.set_title("Case-study regions (filled = analysed for the mid-evaluation)")
    save(fig, "02_regions")


def fig03_flowchart():
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.axis("off")
    cols = [
        ("Datasets", BLUE, ["ERA5-Land (Tmax, VPD, soil moisture)", "SPEI-03 / SPEI-12", "CHIRPS rainfall",
                            "MODIS MCD64A1 burned area", "MODIS MOD14A1 + VIIRS fire", "MODIS GPP + NDVI",
                            "MODIS land cover, SRTM"]),
        ("A  Preprocessing", NEUTRAL, ["0.1° monthly grid (Earth Engine)", "QA + natural-vegetation mask",
                                       "z-anomalies vs 2001–2016", "NDVI seasonal ratio"]),
        ("B  Event detection", NEUTRAL, ["Drought: SPEI-12 ≤ −1.5", "Heat: Tmax > p90, ≥ 3 days",
                                         "Fire: burned fraction ≥ 30%", "Groups: double-hit, single-hit"]),
        ("Analysis", ORANGE, ["C  Sensitivity (matched controls)", "D  Recovery (KM, Lloret, shifts)",
                              "E  Dose–response / threshold", "F  Robustness + cross-checks"]),
        ("G  Synthesis", VIOLET, ["Cross-case effect sizes", "Policy implications"]),
    ]
    for i, (title, color, items) in enumerate(cols):
        x = i * 2.45
        ax.add_patch(FancyBboxPatch((x, 4.1), 2.1, 0.6, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc=color, ec="none"))
        ax.text(x + 1.05, 4.4, title, ha="center", va="center", color="white", weight="bold")
        for j, it in enumerate(items):
            y = 3.5 - j * 0.52
            ax.add_patch(FancyBboxPatch((x, y - 0.2), 2.1, 0.42, boxstyle="round,pad=0.01,rounding_size=0.06",
                                        fc="white", ec=AXIS, lw=0.8))
            ax.text(x + 1.05, y, it, ha="center", va="center", fontsize=7.6, color=INK)
        if i < len(cols) - 1:
            ax.annotate("", (x + 2.42, 4.4), (x + 2.12, 4.4),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.5))
    ax.set_xlim(-0.1, 12.2)
    ax.set_ylim(-0.3, 4.9)
    ax.set_title("Methods pipeline")
    save(fig, "03_flowchart")


# ===================================================================== 4-7 (data + events)
def fig04_timeline(case_ids):
    fig, axes = plt.subplots(len(case_ids), 1, figsize=(11, 1.9 * len(case_ids)), squeeze=False)
    for ax, cid in zip(axes[:, 0], case_ids):
        case, _ = case_cfg(cid)
        lanes = [("Event 1: " + case["first_extreme"], case["event1"], BLUE),
                 ("Fire season", case["fire_window"], RED),
                 ("Event 2", case["event2"], ORANGE),
                 ("Recovery", [case["recovery_start"], "2024-12"], AQUA)]
        for i, (lab, w, c) in enumerate(lanes):
            a = pd.Timestamp(w[0] + "-01")
            b = pd.Timestamp(w[1] + "-01") + pd.offsets.MonthEnd(0)
            ax.barh(i, (b - a).days, left=a, height=0.6, color=c)
            ax.text(a, i, f" {lab} ", va="center", ha="right", fontsize=8.5, color=INK2)
        ax.set_yticks([])
        ax.invert_yaxis()
        ax.set_xlim(pd.Timestamp(case["pre_window"][0] + "-01"), pd.Timestamp("2025-01-01"))
        ax.grid(axis="y", visible=False)
        ax.set_title(case["title"])
    fig.tight_layout()
    save(fig, "04_timelines")


def fig05_timeseries(case_id):
    ds, df = load(case_id)
    case, _ = case_cfg(case_id)
    v = ds["valid"].values
    series = [("SPEI-12", ds["spei12"]), ("CHIRPS rainfall (z)", ds["precip_z"]),
              ("ERA5-Land Tmax (z)", ds["tmax_z"]), ("MODIS GPP (z)", ds["gpp_z"]),
              ("MODIS NDVI (z)", ds["ndvi_z"])]
    fig, axes = plt.subplots(len(series), 1, figsize=(11, 8), sharex=True)
    for ax, (lab, da) in zip(axes, series):
        s = da.where(v).mean(["lat", "lon"]).to_series()
        s12 = s.rolling(3, center=True).mean()
        ax.fill_between(s.index, 0, s12, where=s12 >= 0, color=BLUE, alpha=0.35, lw=0)
        ax.fill_between(s.index, 0, s12, where=s12 < 0, color=RED, alpha=0.35, lw=0)
        ax.plot(s12.index, s12, color=INK2, lw=1)
        ax.axhline(0, color=AXIS, lw=0.8)
        shade(ax, case["event1"], BLUE)
        shade(ax, case["event2"], ORANGE)
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=9)
    axes[0].set_title(f"{case['title']}: regional means (3-month smoothed); "
                      "blue band = event 1, orange band = event 2")
    save(fig, f"05_timeseries_{case_id}")


def fig06_event_maps(case_id):
    ds, df = load(case_id)
    case, _ = case_cfg(case_id)
    d = df[df.valid]
    fig = plt.figure(figsize=(14, 4.6))
    if case["first_extreme"] == "drought":
        e1 = (to_map(ds, d, "spei12_min_e1"), "Event 1: minimum SPEI-12 (2017–19)", DIVERGING, (-3, 3))
        e2 = (to_map(ds, d, "hw_days_e2"), "Event 2: heatwave days (Dec–Feb)", SEQ_RED, (0, None))
    else:
        e1 = (to_map(ds, d, "tmax_peakz_e1"), "Event 1: peak daily Tmax anomaly (SD)", SEQ_RED, (0, None))
        e2 = (to_map(ds, d, "spei03_e2"), "Event 2: SPEI-03 (Aug–Sep 2021)", DIVERGING, (-3, 3))
    for i, (arr, title, cmap, (lo, hi)) in enumerate([e1, e2]):
        ax = geo_ax(fig, (1, 3, i + 1), case["bbox"])
        m = mesh(ds, ax, arr, cmap=cmap, vmin=lo, vmax=hi if hi else np.nanpercentile(arr, 98))
        fig.colorbar(m, ax=ax, shrink=0.75, pad=0.02)
        ax.set_title(title, fontsize=10)
    ax = geo_ax(fig, (1, 3, 3), case["bbox"])
    m = mesh(ds, ax, to_map(ds, d, "burn_frac"), cmap=SEQ_RED, vmin=0, vmax=1)
    vv = d[d.viirs_count > 0]
    ax.scatter(vv.lon, vv.lat, s=np.clip(vv.viirs_count / 20, 0.5, 12), c=INK, alpha=0.35, lw=0,
               transform=ccrs.PlateCarree(), label="VIIRS detections")
    fig.colorbar(m, ax=ax, shrink=0.75, pad=0.02)
    ax.set_title("Fire: MODIS burned fraction + VIIRS hotspots", fontsize=10)
    ax.legend(loc="lower left", fontsize=8)
    fig.suptitle(case["title"], x=0.01, ha="left", weight="bold")
    save(fig, f"06_event_maps_{case_id}")


def fig07_crosschecks(case_ids):
    fig, axes = plt.subplots(2, len(case_ids), figsize=(5.2 * len(case_ids), 9), squeeze=False)
    for j, cid in enumerate(case_ids):
        case, _ = case_cfg(cid)
        F = json.load(open(TAB / f"{cid}_F_results.json"))
        p = pd.read_csv(TAB / f"{cid}_F_precip_sample.csv")
        ax = axes[0, j]
        hb = ax.hexbin(p.chirps, p.era5, gridsize=50, bins="log", cmap=SEQ_BLUE, mincnt=1)
        mx = np.nanpercentile(p[["chirps", "era5"]].values, 99.5)
        ax.plot([0, mx], [0, mx], color=INK2, lw=1, ls="--")
        ax.set_xlim(0, mx)
        ax.set_ylim(0, mx)
        ax.set_xlabel("CHIRPS (mm/month)")
        ax.set_ylabel("ERA5-Land (mm/month)")
        ax.set_title(f"{case['short']}\nRainfall: r = {F['precip']['pixel_r']:.2f}, "
                     f"bias = {F['precip']['rel_bias_pct']:+.0f}%", fontsize=10)
        fig.colorbar(hb, ax=ax, label="cells (log)")
        _, df = load(cid)
        v = df[df.valid & ((df.viirs_count > 0) | (df.modis_fire_pixdays > 0))]
        ax = axes[1, j]
        ax.scatter(v.modis_fire_pixdays + 1, v.viirs_count + 1, s=8, color=BLUE, alpha=0.4, lw=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("MODIS MOD14A1 fire pixel-days + 1")
        ax.set_ylabel("VIIRS detections + 1")
        ax.set_title(f"Fire: Spearman ρ = {F['fire']['modis_vs_viirs_spearman']:.2f}; "
                     f"{F['fire']['burned_cells_with_viirs_pct']:.0f}% of burned cells have VIIRS hits",
                     fontsize=10)
    fig.tight_layout()
    save(fig, "07_crosschecks")


# ===================================================================== 8-9 (sensitivity)
def fig08_sensitivity(case_ids):
    fig, axes = plt.subplots(1, len(case_ids) + 1, figsize=(5 * (len(case_ids) + 1), 4.6),
                             gridspec_kw={"width_ratios": [1] * len(case_ids) + [1.3]})
    eco_rows = []
    for ax, cid in zip(axes, case_ids):
        case, _ = case_cfg(cid)
        m = pd.read_csv(TAB / f"{cid}_C_matched.csv")
        R = json.load(open(TAB / f"{cid}_C_results.json"))["gpp_response"]
        data = [m[m.role == "control"].gpp_z_e2, m[m.role == "hit"].gpp_z_e2]
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, lw=1.5), whiskerprops=dict(color=MUTED),
                        capprops=dict(color=MUTED))
        for patch, c in zip(bp["boxes"], [BLUE, ORANGE]):
            patch.set(fc=c, ec="none", alpha=0.85)
        ax.set_xticks([1, 2], ["Not hit by event 1\n(matched control)", "Hit by event 1"])
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_ylabel("GPP anomaly during event 2 (z)")
        ax.set_title(f"{case['short']}\nΔ = {R['diff']:+.2f} SD "
                     f"[{R['ci_lo']:+.2f}, {R['ci_hi']:+.2f}], p = {R['mwu_p']:.2g}, "
                     f"n = {R['n_pairs']}", fontsize=10)
        e = pd.read_csv(TAB / f"{cid}_C_ecosystem.csv")
        e["case"] = case["short"]
        eco_rows.append(e)
    ax = axes[-1]
    eco = pd.concat(eco_rows)
    labels = [f"{r.lc_group}\n({r.case.split(' (')[0]})" for r in eco.itertuples()]
    y = np.arange(len(eco))
    ax.errorbar(eco["diff"], y, xerr=[eco["diff"] - eco.ci_lo, eco.ci_hi - eco["diff"]], fmt="o",
                color=ORANGE, ecolor=MUTED, ms=7, capsize=0, lw=1.5)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Extra GPP loss in hit cells (hit − control, SD)")
    ax.set_title("By ecosystem type (mean, 95% CI)", fontsize=10)
    fig.tight_layout()
    save(fig, "08_sensitivity")


def fig09_fire_logit(case_ids):
    fig, axes = plt.subplots(1, len(case_ids), figsize=(5.5 * len(case_ids), 3.8), squeeze=False)
    names = {"e1_sev": "Event-1 severity: TOTAL effect", "e1_sev_med": "Event-1 severity (mechanism model)",
             "vpd_z_fire": "VPD anomaly (fire season)",
             "sm_z_fire": "Soil moisture anomaly", "ndvi_z_pre": "Pre-fire NDVI anomaly",
             "elev": "Elevation"}
    for ax, cid in zip(axes[0], case_ids):
        case, _ = case_cfg(cid)
        f = TAB / f"{cid}_C_fire_logit.csv"
        if not f.exists():
            na_panel(ax, f"{case['short']}\nToo few burned cells for a fire model")
            continue
        o = pd.read_csv(f)
        o["predictor"] = o.predictor.replace({"e1_sev": "e1_sev_med"})
        tot = json.load(open(TAB / f"{cid}_C_results.json"))["fire_logit_total"]
        o = pd.concat([pd.DataFrame([{"predictor": "e1_sev", "odds_ratio": tot["OR_e1_sev"],
                                      "ci_lo": tot["ci_lo"], "ci_hi": tot["ci_hi"],
                                      "p": tot["p_e1_sev"]}]), o], ignore_index=True)
        y = np.arange(len(o))
        cols = [ORANGE if p < 0.05 else NEUTRAL for p in o.p]
        ax.errorbar(o.odds_ratio, y, xerr=[o.odds_ratio - o.ci_lo, o.ci_hi - o.odds_ratio],
                    fmt="none", ecolor=MUTED, lw=1.5)
        ax.scatter(o.odds_ratio, y, c=cols, s=55, zorder=3)
        ax.axvline(1, color=INK2, lw=1)
        ax.set_xscale("log")
        log_ticks(ax)
        ax.set_yticks(y, [names[p] for p in o.predictor])
        ax.invert_yaxis()
        ax.set_xlabel("Odds ratio of burning, per 1 SD (log scale)")
        ax.axhline(0.5, color=AXIS, lw=0.8)
        ax.set_title(f"{case['short']}: odds of burning\n"
                     f"top row = total effect; rest = mechanism model; orange = p < 0.05",
                     fontsize=10)
    fig.tight_layout()
    save(fig, "09_fire_odds_ratios")


# ===================================================================== 10-13 (recovery)
def fig10_recovery_curves(case_ids):
    fig, axes = plt.subplots(len(case_ids), 2, figsize=(17, 3.8 * len(case_ids)), squeeze=False)
    for (ax, cid, var) in [(axes[i, j], cid, v) for i, cid in enumerate(case_ids)
                           for j, v in enumerate(["NDVI", "GPP"])]:
        case, com = case_cfg(cid)
        suffix = "" if var == "NDVI" else "_gpp"
        t = pd.read_csv(TAB / f"{cid}_D_trajectories{suffix}.csv", parse_dates=["time"])
        t = t[t.time >= pd.Timestamp(case["event1"][0] + "-01") - pd.DateOffset(years=2)]
        s = pd.read_csv(TAB / f"{cid}_D_recovery_summary.csv", index_col=0)
        for g in GROUPS:
            d = t[t.group == g].set_index("time")
            if d.empty:
                continue
            mm = d["mean"].rolling(3, center=True).mean()
            n = int(s.loc[g, "n"]) if g in s.index and np.isfinite(s.loc[g, "n"]) else 0
            ax.plot(mm.index, mm, color=GROUP_COLOR[g], label=f"{g} (n={n})")
            ax.fill_between(mm.index, mm - 1.96 * d.se, mm + 1.96 * d.se, color=GROUP_COLOR[g],
                            alpha=0.15, lw=0)
        ax.axhline(com["recovery_fraction"], color=INK2, ls="--", lw=1)
        ax.text(t.time.min(), com["recovery_fraction"] + 0.01,
                "recovery threshold (90% of normal)", fontsize=8, color=INK2)
        ax.axhline(1, color=AXIS, lw=0.8)
        shade(ax, case["event1"], BLUE)
        shade(ax, case["fire_window"], RED)
        ax.set_ylabel(f"{var} / seasonal normal")
        ax.set_title(f"{case['short']}: {var} by group (3-month mean, 95% CI; "
                     f"burned = ≥{int(com['recovery_burned_frac'] * 100)}% of cell)", fontsize=10)
        ax.legend(ncol=2, fontsize=8, loc="lower left")
    fig.tight_layout()
    save(fig, "10_recovery_curves")


def fig11_km(case_ids):
    fig, axes = plt.subplots(2, len(case_ids), figsize=(5.8 * len(case_ids), 8.6), squeeze=False)
    for ax, cid in zip(axes[1], case_ids):
        case, _ = case_cfg(cid)
        D = json.load(open(TAB / f"{cid}_D_results.json"))
        tr = D.get("terciles_burned", {})
        for metric, var, ls in [("rec_months", "NDVI", "-"), ("rec_months_gpp", "GPP", "--")]:
            f = TAB / f"{cid}_D_km_terciles_{metric}.csv"
            if not f.exists():
                continue
            k = pd.read_csv(f)
            for g, c in [("Top third (most stressed)", ORANGE), ("Bottom third (least stressed)", BLUE)]:
                d = k[k.group == g]
                ax.step(d.months, d.not_recovered, where="post", color=c, ls=ls, label=f"{var}: {g}")
        if not tr:
            na_panel(ax, f"{case['short']}\nToo few heavily burned cells\nto split by event-1 severity")
            continue
        p = tr.get("rec_months", {}).get("p")
        pg = tr.get("rec_months_gpp", {}).get("p")
        ax.set_xlabel("Months since end of event 2")
        ax.set_ylabel("Share of burned cells not yet recovered")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{case['short']}: by event-1 severity (burned cells)\n"
                     f"log-rank p: NDVI {p:.2g}, GPP {pg:.2g}" if p is not None and pg is not None
                     else case["short"], fontsize=10)
        ax.legend(fontsize=7.5)
    for ax, cid in zip(axes[0], case_ids):
        case, _ = case_cfg(cid)
        k = pd.read_csv(TAB / f"{cid}_D_km_curves.csv")
        D = json.load(open(TAB / f"{cid}_D_results.json"))
        for g in GROUPS:
            d = k[k.group == g]
            if not d.empty:
                med = D["km_median_months"].get(g)
                lab = f"{g} (median {med:.0f} mo)" if med is not None and np.isfinite(med) else f"{g} (median not reached)"
                ax.step(d.months, d.not_recovered, where="post", color=GROUP_COLOR[g], label=lab)
        lr = D["logrank"].get("Double-hit vs Event 2 only", {}).get("p")
        ax.set_xlabel("Months since end of event 2")
        ax.set_ylabel("Share of cells not yet recovered")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{case['short']}\nlog-rank (burned groups) p = {lr:.2g}" if lr is not None
                     else case["short"], fontsize=10)
        ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "11_kaplan_meier")


def fig12_recovery_maps(case_id):
    ds, _ = load(case_id)
    case, _ = case_cfg(case_id)
    rec = pd.read_csv(TAB / f"{case_id}_D_recovery_cells.csv")
    r = ds["ndvi_ratio"].where(ds["valid"])
    panels = [("Before (pre-event mean)", r.sel(time=tslice(case["pre_window"])).mean("time")),
              ("Worst (min. during event 2)",
               r.sel(time=tslice([case["fire_window"][0], case["recovery_start"]])).min("time")),
              ("Latest (2024 mean)", r.sel(time=tslice(["2024-01", "2024-12"])).mean("time"))]
    fig = plt.figure(figsize=(16, 4.4))
    for i, (t, a) in enumerate(panels):
        ax = geo_ax(fig, (1, 4, i + 1), case["bbox"])
        m = mesh(ds, ax, a.values, cmap=DIVERGING.reversed(), vmin=0.5, vmax=1.5)
        ax.set_title(t, fontsize=10)
    fig.colorbar(m, ax=fig.axes[:3], shrink=0.7, pad=0.01, label="NDVI / seasonal normal")
    ax = geo_ax(fig, (1, 4, 4), case["bbox"])
    b = rec[rec.group.isin(["Double-hit", "Event 2 only"])]
    for st, c in STATUS_COLOR.items():
        s = b[b.status == st]
        ax.scatter(s.lon, s.lat, s=3, marker="s", color=c, lw=0, transform=ccrs.PlateCarree(),
                   label=f"{st} ({len(s)})")
    ax.legend(loc="lower left", fontsize=7, markerscale=3)
    ax.set_title("Recovery status of burned cells", fontsize=10)
    fig.suptitle(case["title"], x=0.01, ha="left", weight="bold")
    save(fig, f"12_recovery_maps_{case_id}")


def fig13_lloret_lag(case_ids):
    fig, axes = plt.subplots(2, len(case_ids), figsize=(5.8 * len(case_ids), 8), squeeze=False)
    for j, cid in enumerate(case_ids):
        case, _ = case_cfg(cid)
        s = pd.read_csv(TAB / f"{cid}_D_recovery_summary.csv", index_col=0).reindex(GROUPS)
        ax = axes[0, j]
        idx = ["resistance", "recovery", "resilience"]
        x = np.arange(3)
        w = 0.2
        for i, g in enumerate(GROUPS):
            if g in s.index and np.isfinite(s.loc[g, "n"]):
                ax.bar(x + (i - 1.5) * w, s.loc[g, idx], w - 0.02, color=GROUP_COLOR[g], label=g)
        ax.axhline(1, color=INK2, lw=1)
        ax.set_xticks(x, ["Resistance\n(during / pre)", "Recovery\n(post / during)",
                          "Resilience\n(post / pre)"])
        ax.set_ylabel("Median Lloret index (NDVI)")
        ax.set_title(case["short"], fontsize=10)
        ax.legend(fontsize=8, ncol=2)
        lag = pd.read_csv(TAB / f"{cid}_D_lag_corr.csv")
        ax = axes[1, j]
        lc = sorted(lag.lc_group.unique())
        piv = lag.pivot(index="lc_group", columns="lag", values="r_median").reindex(lc)
        im = ax.imshow(piv.values, aspect="auto", cmap=DIVERGING.reversed(), vmin=-0.5, vmax=0.5)
        ax.set_yticks(range(len(lc)), lc)
        ax.set_xticks(range(0, 25, 3), range(0, 25, 3))
        ax.set_xlabel("Lag (months): SPEI-03 leads GPP anomaly")
        ax.grid(False)
        ax.set_title("Drought memory: median correlation", fontsize=10)
        fig.colorbar(im, ax=ax, label="r")
    fig.tight_layout()
    save(fig, "13_lloret_and_lag")


# ===================================================================== 14-15
def fig14_dose_response(case_ids):
    outs = [("gpp_response", "GPP anomaly in event 2 (z)\nunburned cells"),
            ("burn_probability", "Probability of burning"),
            ("recovery_months", "Months to recover\n(burned cells)")]
    fig, axes = plt.subplots(len(case_ids), 3, figsize=(15, 3.8 * len(case_ids)), squeeze=False)
    for i, cid in enumerate(case_ids):
        case, _ = case_cfg(cid)
        E = json.load(open(TAB / f"{cid}_E_results.json"))
        xl = "Event-1 severity: −min SPEI-12" if case["first_extreme"] == "drought" \
            else "Event-1 severity: peak Tmax anomaly (SD)"
        for j, (k, yl) in enumerate(outs):
            ax = axes[i, j]
            fb = TAB / f"{cid}_E_{k}_bins.csv"
            if not fb.exists():
                na_panel(ax, f"{case['short']}\nNot enough burned cells")
                continue
            b = pd.read_csv(fb)
            lw = pd.read_csv(TAB / f"{cid}_E_{k}_lowess.csv")
            ax.errorbar(b.x_mid, b["mean"], yerr=1.96 * b["sem"], fmt="o", color=ORANGE, ecolor=MUTED,
                        ms=7, capsize=0, label="decile mean ± 95% CI")
            ax.plot(lw.x, lw.y, color=BLUE, label="LOWESS")
            r = E[k]
            if r["p"] < 0.05:
                ax.axvline(r["breakpoint"], color=INK2, ls="--", lw=1)
                ax.text(r["breakpoint"], ax.get_ylim()[1], f" threshold {r['breakpoint']:.2f}",
                        fontsize=8, color=INK2, va="top")
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.set_title(f"{case['short']}: ρ = {r['spearman_rho']:.2f}", fontsize=10)
            if j == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "14_dose_response")


def fig15_synthesis():
    s = pd.read_csv(TAB / "G_synthesis.csv")
    fig, axes = plt.subplots(1, 3, figsize=(19, 3.4 + 0.6 * len(s)),
                             gridspec_kw={"wspace": 0.75})
    y = np.arange(len(s))

    def dots(ax, x, lo, hi, off, color, label):
        ax.errorbar(x, y + off, xerr=[x - lo, hi - x], fmt="o", color=color, ecolor=color, ms=9,
                    lw=2, capsize=0, label=label)

    ax = axes[0]
    dots(ax, s.gpp_diff_hit_minus_control, s.gpp_ci_lo, s.gpp_ci_hi, -0.12, ORANGE, "Hit vs not hit (main)")
    dots(ax, s.gpp_diff_terciles, s.gpp_terciles_ci_lo, s.gpp_terciles_ci_hi, 0.12, BLUE,
         "Top vs bottom third")
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlabel("GPP anomaly in event 2: stressed − control (SD)\n(left of 0 = more loss when pre-stressed)")
    ax.set_title("H1 Sensitivity to event 2", fontsize=10)
    ax = axes[1]
    dots(ax, s.burn_OR_per_SD_e1, s.burn_OR_ci_lo, s.burn_OR_ci_hi, 0, RED, "Odds ratio per SD")
    ax.axvline(1, color=INK2, lw=1)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 8)
    log_ticks(ax)
    for i, r in enumerate(s.itertuples()):
        if not np.isfinite(r.burn_OR_per_SD_e1):
            ax.text(0.55, i, "n/a: too few burned cells", va="center", fontsize=8, color=INK2)
            continue
        ax.text(r.burn_OR_ci_hi * 1.05, i, f" burned: {r.burn_rate_top_third:.0%} (top third)\n vs {r.burn_rate_bottom_third:.0%} (bottom third)", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("Odds of burning per 1 SD of event-1 severity (log)")
    ax.set_title("H1 Fire (total effect)", fontsize=10)
    ax = axes[2]
    ax.barh(y - 0.18, s.km_ndvi_median_top_third, 0.34, color=ORANGE, label="Top third (most stressed)")
    ax.barh(y + 0.18, s.km_ndvi_median_bottom_third, 0.34, color=BLUE, label="Bottom third")
    for i, r in enumerate(s.itertuples()):
        if not np.isfinite(r.km_terciles_p):
            ax.text(0.2, i, "n/a: too few burned cells", va="center", fontsize=8, color=INK2)
            continue
        ax.text(max(r.km_ndvi_median_top_third, r.km_ndvi_median_bottom_third) + 0.3, i,
                f"p = {r.km_terciles_p:.2g}", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("Median months for burned NDVI to recover")
    ax.set_title("H3 Recovery", fontsize=10)
    for ax in axes:
        ax.set_yticks(y, s.case)
        ax.set_ylim(len(s) - 0.5, -0.5)
    axes[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    axes[2].legend(fontsize=8, loc="lower right")
    fig.suptitle("Cross-case synthesis", x=0.01, ha="left", weight="bold")
    save(fig, "15_synthesis")

    fig, ax = plt.subplots(figsize=(12, 0.9 + 0.8 * len(s)))
    ax.axis("off")
    cell = [[r.case, "\n".join(textwrap.wrap(r.policy, 95))] for r in s.itertuples()]
    t = ax.table(cellText=cell, colLabels=["Case", "Policy implication"], loc="center",
                 cellLoc="left", colWidths=[0.22, 0.78])
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 2.6)
    for (r, c), cl in t.get_celld().items():
        cl.set_edgecolor(GRID)
        if r == 0:
            cl.set_facecolor(INK2)
            cl.get_text().set_color("white")
            cl.get_text().set_weight("bold")
    save(fig, "15b_policy_table")


def run(case_ids):
    fig01_concept()
    fig02_regions(case_ids)
    fig03_flowchart()
    fig04_timeline(case_ids)
    for cid in case_ids:
        fig05_timeseries(cid)
        fig06_event_maps(cid)
        fig12_recovery_maps(cid)
    fig07_crosschecks(case_ids)
    fig08_sensitivity(case_ids)
    fig09_fire_logit(case_ids)
    fig10_recovery_curves(case_ids)
    fig11_km(case_ids)
    fig13_lloret_lag(case_ids)
    fig14_dose_response(case_ids)
    fig15_synthesis()
