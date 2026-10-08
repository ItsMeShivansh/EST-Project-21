"""Download every dataset for every case from Google Earth Engine.

All aggregation happens on Earth Engine's servers; we pull small numpy arrays on a
common 0.1 deg grid (aligned to the native ERA5-Land grid) with ee.data.computePixels.

Usage:
    EE_PROJECT=your-project-id .venv/bin/python gee/export_data.py [case_id ...]

Output: download cache in data/raw/<case>/<variable>_<chunk>.npy, assembled into
datasets/<case>/raw.nc (compressed)
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import requests
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, load_config, grid_coords, month_range  # noqa: E402

NODATA = -9999.0

ERA5 = "ECMWF/ERA5_LAND/DAILY_AGGR"
CHIRPS = "UCSB-CHG/CHIRPS/PENTAD"
GPP = "MODIS/061/MOD17A2HGF"
NDVI = "MODIS/061/MOD13Q1"
BURN = "MODIS/061/MCD64A1"
ACTIVE = "MODIS/061/MOD14A1"
LANDCOVER = "MODIS/061/MCD12Q1"
DEM = "USGS/SRTMGL1_003"
SPEI_CANDIDATES = ["CSIC/SPEI/2_11", "CSIC/SPEI/2_10", "CSIC/SPEI/2_9"]


# --------------------------------------------------------------------------- helpers
def grid_spec(bbox, res):
    lon, lat = grid_coords(bbox, res)
    x0 = float(lon[0] - res / 2)
    y1 = float(lat[0] + res / 2)
    return {
        "dimensions": {"width": len(lon), "height": len(lat)},
        "affineTransform": {"scaleX": res, "shearX": 0, "translateX": x0,
                            "shearY": 0, "scaleY": -res, "translateY": y1},
        "crsCode": "EPSG:4326",
    }, [res, 0, x0, 0, -res, y1]


def fetch(img, grid, retries=4):
    """Return array (bands, H, W) with NaN for no data."""
    img = ee.Image(img).unmask(NODATA).toFloat()
    for attempt in range(retries):
        try:
            arr = ee.data.computePixels({"expression": img, "fileFormat": "NUMPY_NDARRAY",
                                         "grid": grid})
            out = np.stack([arr[n] for n in arr.dtype.names]).astype("float32")
            out[out <= NODATA + 1] = np.nan
            return out
        except Exception as e:  # transient EE errors / timeouts
            if attempt == retries - 1:
                raise
            wait = 10 * (attempt + 1)
            print(f"    retry in {wait}s: {str(e)[:120]}")
            time.sleep(wait)


def month_bounds(ym):
    d0 = ee.Date(ym + "-01")
    return d0, d0.advance(1, "month")


def to_grid(img, native_proj, crs_transform, reducer, max_pixels=4096):
    """Aggregate a fine-resolution image onto the 0.1 deg grid."""
    return (ee.Image(img).setDefaultProjection(native_proj)
            .reduceResolution(reducer=reducer, maxPixels=max_pixels)
            .reproject(crs="EPSG:4326", crsTransform=crs_transform))


def es(t_c):
    """Saturation vapour pressure (kPa), Tetens. t_c: ee.Image in deg C."""
    return t_c.multiply(17.27).divide(t_c.add(237.3)).exp().multiply(0.6108)


# --------------------------------------------------------------------------- monthly variables
def era5_month(ym):
    d0, d1 = month_bounds(ym)
    c = ee.ImageCollection(ERA5).filterDate(d0, d1)
    tmax = c.select("temperature_2m_max").mean().subtract(273.15).rename("tmax")
    sm = c.select("volumetric_soil_water_layer_1").mean().rename("sm")
    tp = c.select("total_precipitation_sum").sum().multiply(1000).rename("tp_era5")

    def vpd(i):
        t = i.select("temperature_2m_max").subtract(273.15)
        td = i.select("dewpoint_temperature_2m").subtract(273.15)
        return es(t).subtract(es(td)).max(0)

    v = c.map(vpd).mean().rename("vpd")
    return ee.Image.cat([tmax, sm, tp, v])


def spei_month(ym, spei_id):
    d0, d1 = month_bounds(ym)
    img = ee.ImageCollection(spei_id).filterDate(d0, d1).first()
    img = ee.Image(ee.Algorithms.If(img, img, ee.Image.constant([NODATA, NODATA])
                                    .rename(["SPEI_03_month", "SPEI_12_month"])))
    return (img.select(["SPEI_03_month", "SPEI_12_month"], ["spei03", "spei12"])
            .resample("bilinear"))


def chirps_month(ym, crs_t):
    d0, d1 = month_bounds(ym)
    col = ee.ImageCollection(CHIRPS)
    s = col.filterDate(d0, d1).select("precipitation").sum()
    return to_grid(s, col.first().projection(), crs_t, ee.Reducer.mean(), 64).rename("precip")


def gpp_month(ym, crs_t):
    d0, d1 = month_bounds(ym)
    col = ee.ImageCollection(GPP)
    m = (col.filterDate(d0, d1).select("Gpp")
         .map(lambda i: i.updateMask(i.lt(30000)).multiply(0.0125))  # gC m-2 day-1
         .mean())
    return to_grid(m, col.first().projection(), crs_t, ee.Reducer.mean()).rename("gpp")


def ndvi_month(ym, crs_t):
    d0, d1 = month_bounds(ym)
    col = ee.ImageCollection(NDVI)
    m = (col.filterDate(d0, d1)
         .map(lambda i: i.select("NDVI").updateMask(i.select("SummaryQA").lte(1))
              .multiply(0.0001))
         .mean())
    return to_grid(m, col.first().select("NDVI").projection(), crs_t,
                   ee.Reducer.mean()).rename("ndvi")


def burn_month(ym, crs_t):
    d0, d1 = month_bounds(ym)
    col = ee.ImageCollection(BURN)
    b = col.filterDate(d0, d1).select("BurnDate").max().gt(0).unmask(0)
    return to_grid(b, col.first().projection(), crs_t, ee.Reducer.mean()).rename("burn")


MONTHLY = {
    "era5": (era5_month, ["tmax", "sm", "tp_era5", "vpd"]),
    "spei": (spei_month, ["spei03", "spei12"]),
    "chirps": (chirps_month, ["precip"]),
    "gpp": (gpp_month, ["gpp"]),
    "ndvi": (ndvi_month, ["ndvi"]),
    "burn": (burn_month, ["burn"]),
}


# --------------------------------------------------------------------------- static / event variables
def static_layers(crs_t, cfg_case):
    lc = ee.ImageCollection(LANDCOVER).select("LC_Type1")
    proj = lc.first().projection()
    out = []
    for yr in (2001, 2016, 2019):
        y = lc.filterDate(f"{yr}-01-01", f"{yr}-12-31").first()
        out.append(to_grid(y, proj, crs_t, ee.Reducer.mode(), 1024).rename(f"lc_{yr}"))
    y2016 = lc.filterDate("2016-01-01", "2016-12-31").first()
    nat = y2016.gte(1).And(y2016.lte(10))
    out.append(to_grid(nat, proj, crs_t, ee.Reducer.mean(), 1024).rename("natfrac"))
    dem = ee.Image(DEM).select("elevation")
    out.append(dem.reproject(crs="EPSG:4326", crsTransform=crs_t).rename("elev"))
    return ee.Image.cat(out)


def active_fire(crs_t, window):
    d0 = ee.Date(window[0] + "-01")
    d1 = ee.Date(window[1] + "-01").advance(1, "month")
    col = ee.ImageCollection(ACTIVE).filterDate(d0, d1)
    proj = col.first().select("MaxFRP").projection()
    frp = col.select("MaxFRP").max().multiply(0.1).unmask(0)
    days = col.select("FireMask").map(lambda i: i.gte(7)).sum().unmask(0)
    return ee.Image.cat([
        to_grid(frp, proj, crs_t, ee.Reducer.max(), 1024).rename("modis_frp_max"),
        # mean fire-days per 1 km pixel x number of 1 km pixels in the cell = fire pixel-days
        to_grid(days, proj, crs_t, ee.Reducer.mean(), 1024)
        .multiply(ee.Image.pixelArea().divide(proj.nominalScale().pow(2)))
        .rename("modis_fire_pixdays"),
    ])


def daily_tmax(window):
    d0 = ee.Date(window[0] + "-01")
    d1 = ee.Date(window[1] + "-01").advance(1, "month")
    c = ee.ImageCollection(ERA5).filterDate(d0, d1).select("temperature_2m_max")
    return c.toBands().subtract(273.15)


def tmax_climatology(baseline):
    c = (ee.ImageCollection(ERA5).filterDate(baseline[0] + "-01",
                                             ee.Date(baseline[1] + "-01").advance(1, "month"))
         .select("temperature_2m_max"))
    bands = []
    for m in range(1, 13):
        cm = c.filter(ee.Filter.calendarRange(m, m, "month"))
        k = cm.map(lambda i: i.subtract(273.15))
        bands += [k.reduce(ee.Reducer.percentile([90])).rename(f"p90_{m:02d}"),
                  k.mean().rename(f"mean_{m:02d}"),
                  k.reduce(ee.Reducer.stdDev()).rename(f"sd_{m:02d}")]
    return ee.Image.cat(bands)


# --------------------------------------------------------------------------- VIIRS (NASA FIRMS)
FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/data/country/viirs-snpp/{y}/viirs-snpp_{y}_{c}.csv"


def viirs_grid(case, lon, lat, res, out_dir):
    country = {"case1_australia": "Australia", "case2_pnw": "United_States",
               "case3_california": "United_States"}.get(case["id"])
    w0 = pd.Timestamp(case["fire_window"][0] + "-01")
    w1 = pd.Timestamp(case["fire_window"][1] + "-01") + pd.offsets.MonthEnd(1)
    x0, y0, x1, y1 = case["bbox"]
    frames = []
    for y in sorted({w0.year, w1.year}):
        f = out_dir / f"viirs_{country}_{y}.csv"
        if not f.exists():
            print(f"  downloading FIRMS VIIRS {country} {y}")
            r = requests.get(FIRMS_URL.format(y=y, c=country), timeout=600)
            r.raise_for_status()
            f.write_bytes(r.content)
        d = pd.read_csv(f, usecols=["latitude", "longitude", "acq_date", "frp", "confidence"])
        frames.append(d)
    d = pd.concat(frames)
    d["acq_date"] = pd.to_datetime(d["acq_date"])
    d = d[(d.acq_date >= w0) & (d.acq_date <= w1) & (d.confidence != "l")
          & d.longitude.between(x0, x1) & d.latitude.between(y0, y1)]
    ix = np.floor((d.longitude - (lon[0] - res / 2)) / res).astype(int)
    iy = np.floor(((lat[0] + res / 2) - d.latitude) / res).astype(int)
    ok = (ix >= 0) & (ix < len(lon)) & (iy >= 0) & (iy < len(lat))
    count = np.zeros((len(lat), len(lon)), "float32")
    frp_sum = np.zeros_like(count)
    np.add.at(count, (iy[ok], ix[ok]), 1)
    np.add.at(frp_sum, (iy[ok], ix[ok]), d.frp[ok])
    return count, frp_sum


# --------------------------------------------------------------------------- driver
def run_case(case_id, cfg, spei_id):
    case = cfg["cases"][case_id] | {"id": case_id}
    com = cfg["common"]
    res = com["grid_deg"]
    grid, crs_t = grid_spec(case["bbox"], res)
    lon, lat = grid_coords(case["bbox"], res)
    out = ROOT / "data" / "raw" / case_id
    out.mkdir(parents=True, exist_ok=True)
    months = month_range(com["start"], com["end"])
    years = sorted({m[:4] for m in months})
    print(f"== {case_id}: grid {len(lat)}x{len(lon)}, {len(months)} months")

    def job(name, yr):
        fn, bands = MONTHLY[name]
        f = out / f"{name}_{yr}.npy"
        yms = [m for m in months if m.startswith(yr)]
        if name == "spei":
            imgs = [fn(m, spei_id) for m in yms]
        elif name == "era5":
            imgs = [fn(m) for m in yms]
        else:
            imgs = [fn(m, crs_t) for m in yms]
        t = time.time()
        arr = fetch(ee.Image.cat(imgs), grid)  # (12*nb, H, W) month-major
        arr = arr.reshape(len(yms), len(bands), *arr.shape[1:])
        np.save(f, arr)
        print(f"  {name} {yr} ok ({time.time() - t:.0f}s)", flush=True)

    todo = [(n, y) for n in MONTHLY for y in years if not (out / f"{n}_{y}.npy").exists()]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for fut in [pool.submit(job, n, y) for n, y in todo]:
            fut.result()

    extras = {
        "static": lambda: static_layers(crs_t, case),
        "active_fire": lambda: active_fire(crs_t, case["fire_window"]),
        "tmax_clim": lambda: tmax_climatology(com["baseline"]),
        "tmax_daily": lambda: daily_tmax(case["daily_window"]),
    }
    for name, make in extras.items():
        f = out / f"{name}.npy"
        if not f.exists():
            t = time.time()
            np.save(f, fetch(make(), grid))
            print(f"  {name} ok ({time.time() - t:.0f}s)")

    f = out / "viirs.npy"
    if not f.exists():
        np.save(f, np.stack(viirs_grid(case, lon, lat, res, out)))
        print("  viirs ok")

    assemble(case_id, case, com, out, lon, lat, months, years)


def assemble(case_id, case, com, out, lon, lat, months, years):
    time_idx = pd.to_datetime([m + "-01" for m in months])
    data = {}
    for name, (_, bands) in MONTHLY.items():
        arr = np.concatenate([np.load(out / f"{name}_{y}.npy") for y in years])
        for i, b in enumerate(bands):
            data[b] = (("time", "lat", "lon"), arr[:, i])
    st = np.load(out / "static.npy")
    for i, b in enumerate(["lc_2001", "lc_2016", "lc_2019", "natfrac", "elev"]):
        data[b] = (("lat", "lon"), st[i])
    af = np.load(out / "active_fire.npy")
    data["modis_frp_max"] = (("lat", "lon"), af[0])
    data["modis_fire_pixdays"] = (("lat", "lon"), af[1])
    vi = np.load(out / "viirs.npy")
    data["viirs_count"] = (("lat", "lon"), vi[0])
    data["viirs_frp_sum"] = (("lat", "lon"), vi[1])
    cl = np.load(out / "tmax_clim.npy").reshape(12, 3, len(lat), len(lon))
    for i, b in enumerate(["tmax_p90", "tmax_dmean", "tmax_dsd"]):
        data[b] = (("month", "lat", "lon"), cl[:, i])
    dd = np.load(out / "tmax_daily.npy")
    d0 = pd.Timestamp(case["daily_window"][0] + "-01")
    days = pd.date_range(d0, periods=dd.shape[0], freq="D")
    data["tmax_day"] = (("day", "lat", "lon"), dd)

    ds = xr.Dataset(data, coords={"time": time_idx, "lat": lat, "lon": lon,
                                  "month": np.arange(1, 13), "day": days})
    ds.attrs.update(case=case_id, title=case["title"])
    dest = ROOT / "datasets" / case_id
    dest.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(dest / "raw.nc", encoding={v: {"zlib": True, "complevel": 6} for v in ds.data_vars})
    print(f"  wrote {dest / 'raw.nc'}")


def pick_spei():
    for sid in SPEI_CANDIDATES:
        try:
            last = ee.ImageCollection(sid).sort("system:time_start", False).first()
            d = ee.Date(last.get("system:time_start")).format("YYYY-MM").getInfo()
            print(f"SPEI: {sid} (last month {d})")
            return sid
        except Exception:
            continue
    raise RuntimeError("No SPEI collection available")


if __name__ == "__main__":
    project = os.environ.get("EE_PROJECT")
    if not project:
        sys.exit("Set EE_PROJECT to your Earth Engine-enabled Google Cloud project id")
    ee.Initialize(project=project)
    cfg = load_config()
    spei_id = pick_spei()
    for cid in (sys.argv[1:] or list(cfg["mid_eval_cases"])):
        run_case(cid, cfg, spei_id)
