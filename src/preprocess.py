"""Module A: masking, standardized anomalies, NDVI seasonal ratio."""
import numpy as np
import xarray as xr

from .common import RAW, PROC, LC_GROUP, case_cfg, tslice

ANOM_VARS = ["gpp", "ndvi", "tmax", "vpd", "sm", "precip", "tp_era5"]


def lc_group_map(lc):
    out = np.full(lc.shape, "", dtype=object)
    for k, g in LC_GROUP.items():
        out[lc == k] = g
    return out


def run(case_id):
    case, com = case_cfg(case_id)
    ds = xr.open_dataset(RAW / case_id / "raw.nc").load()

    # ---- land-cover mask: natural vegetation, same coarse group in 2001, 2016 and 2019
    g01, g16, g19 = (lc_group_map(np.nan_to_num(ds[f"lc_{y}"].values).astype(int))
                     for y in (2001, 2016, 2019))
    stable = (g16 != "") & (g16 == g19) & (g16 == g01)
    natural = ds["natfrac"].values >= com["min_natfrac"]
    has_data = (np.isfinite(ds["tmax"]).all("time").values
                & (np.isfinite(ds["ndvi"]).sum("time").values > 0.8 * ds.sizes["time"])
                & np.isfinite(ds["gpp"]).any("time").values
                & np.isfinite(ds["spei12"]).any("time").values)
    valid = stable & natural & has_data

    # ---- standardized anomalies per calendar month vs. baseline
    base = ds.sel(time=tslice(com["baseline"]))
    out = xr.Dataset(coords=ds.coords)
    for v in ANOM_VARS:
        clim = base[v].groupby("time.month")
        mu, sd = clim.mean("time"), clim.std("time")
        z = (ds[v].groupby("time.month") - mu).groupby("time.month") / sd.where(sd > 0)
        out[f"{v}_z"] = z.drop_vars("month", errors="ignore")
        if v in ("ndvi", "gpp"):
            floor = 0.05 if v == "ndvi" else 0.2
            out[f"{v}_ratio"] = (ds[v].groupby("time.month") / mu.where(mu > floor)
                                 ).drop_vars("month", errors="ignore")
            out[f"{v}_clim"] = mu

    for v in ["gpp", "ndvi", "tmax", "vpd", "sm", "precip", "tp_era5", "spei03", "spei12", "burn"]:
        out[v] = ds[v]
    for v in ["elev", "natfrac", "modis_frp_max", "modis_fire_pixdays", "viirs_count",
              "viirs_frp_sum", "tmax_p90", "tmax_dmean", "tmax_dsd", "tmax_day", "lc_2016"]:
        out[v] = ds[v]
    out["valid"] = (("lat", "lon"), valid)
    out["lc_group"] = (("lat", "lon"), g16.astype(str))
    out.attrs = ds.attrs
    out.to_netcdf(PROC / f"{case_id}.nc")
    print(f"[A] {case_id}: {int(valid.sum())} valid cells of {valid.size}")
    return out
