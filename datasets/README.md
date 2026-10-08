# Datasets

One NetCDF file per case, holding every input variable on a common 0.1° grid.
`run_all.py` reads these files directly, so the analysis runs without Earth Engine access.

| File | Region | Grid | Size |
|---|---|---|---|
| `case1_australia/raw.nc` | SE Australia, 141–154°E, 28–39°S | 110 × 130 cells | 95 MB |
| `case2_pnw/raw.nc` | Pacific Northwest, 117–124°W, 42–49°N | 70 × 70 cells | 40 MB |

Monthly variables cover January 2001 to December 2024. The files are zlib-compressed (lossless).

## Sources

| Dataset | Earth Engine ID / source | Variables in the file |
|---|---|---|
| ERA5-Land | `ECMWF/ERA5_LAND/DAILY_AGGR` | monthly Tmax, VPD, soil moisture, precipitation; daily Tmax for the event window; Tmax climatology (2001–2016) |
| SPEI v2.11 | `CSIC/SPEI/2_11` | SPEI-03, SPEI-12 |
| CHIRPS | `UCSB-CHG/CHIRPS/PENTAD` | precipitation |
| MODIS GPP | `MODIS/061/MOD17A2HGF` | gross primary productivity |
| MODIS NDVI | `MODIS/061/MOD13Q1` | NDVI |
| MODIS burned area | `MODIS/061/MCD64A1` | monthly burned fraction |
| MODIS active fire | `MODIS/061/MOD14A1` | max fire radiative power, fire pixel-days |
| VIIRS | NASA FIRMS country archive (S-NPP, 375 m) | active-fire count and FRP per cell |
| MODIS land cover | `MODIS/061/MCD12Q1` | IGBP class (2001, 2016, 2019), natural-vegetation fraction |
| SRTM | `USGS/SRTMGL1_003` | elevation |

## Rebuilding the files

`gee/export_data.py` downloads everything again from Earth Engine and NASA FIRMS and writes
these files (needs an Earth Engine project; see the main README). Its download cache goes to
`data/raw/`, which is not part of the repository.
