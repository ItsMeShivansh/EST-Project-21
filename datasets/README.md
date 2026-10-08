# Datasets

One NetCDF file per case, holding every input variable on a common 0.1° grid.
`run_all.py` reads these files directly, so the analysis runs without Earth Engine access.

| File | Region | Grid | Size |
|---|---|---|---|
| `case1_australia/raw.nc` | SE Australia, 141–154°E, 28–39°S | 110 × 130 cells | 95 MB |
| `case2_pnw/raw.nc` | Pacific Northwest, 117–124°W, 42–49°N | 70 × 70 cells | 40 MB |

Monthly variables cover January 2001 to December 2024. The files are zlib-compressed (lossless).

## Sources

| Dataset | Earth Engine ID (what `export_data.py` downloads) | Original provider | Variables in the file |
|---|---|---|---|
| ERA5-Land | [`ECMWF/ERA5_LAND/DAILY_AGGR`](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_DAILY_AGGR) | [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land) | monthly Tmax, VPD, soil moisture, precipitation; daily Tmax for the event window; Tmax climatology (2001–2016) |
| SPEI v2.11 | [`CSIC/SPEI/2_11`](https://developers.google.com/earth-engine/datasets/catalog/CSIC_SPEI_2_11) | [SPEI Global Drought Database (CSIC)](https://spei.csic.es/database.html) | SPEI-03, SPEI-12 |
| CHIRPS | [`UCSB-CHG/CHIRPS/PENTAD`](https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_PENTAD) | [Climate Hazards Center, UCSB](https://www.chc.ucsb.edu/data/chirps) | precipitation |
| MODIS GPP | [`MODIS/061/MOD17A2HGF`](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD17A2HGF) | [NASA LP DAAC](https://lpdaac.usgs.gov/products/mod17a2hgfv061/) | gross primary productivity |
| MODIS NDVI | [`MODIS/061/MOD13Q1`](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13Q1) | [NASA LP DAAC](https://lpdaac.usgs.gov/products/mod13q1v061/) | NDVI |
| MODIS burned area | [`MODIS/061/MCD64A1`](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD64A1) | [NASA LP DAAC](https://lpdaac.usgs.gov/products/mcd64a1v061/) | monthly burned fraction |
| MODIS active fire | [`MODIS/061/MOD14A1`](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD14A1) | [NASA LP DAAC](https://lpdaac.usgs.gov/products/mod14a1v061/) | max fire radiative power, fire pixel-days |
| MODIS land cover | [`MODIS/061/MCD12Q1`](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD12Q1) | [NASA LP DAAC](https://lpdaac.usgs.gov/products/mcd12q1v061/) | IGBP class (2001, 2016, 2019), natural-vegetation fraction |
| SRTM | [`USGS/SRTMGL1_003`](https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003) | [NASA LP DAAC](https://lpdaac.usgs.gov/products/srtmgl1v003/) | elevation |
| VIIRS (S-NPP, 375 m) | not from Earth Engine; see below | [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/country/) | active-fire count and FRP per cell |

### VIIRS active-fire files (NASA FIRMS country archive)

`export_data.py` downloads these CSVs directly and counts the detections inside each case's grid:

| Case | File |
|---|---|
| Case 1 | [viirs-snpp_2019_Australia.csv](https://firms.modaps.eosdis.nasa.gov/data/country/viirs-snpp/2019/viirs-snpp_2019_Australia.csv) |
| Case 1 | [viirs-snpp_2020_Australia.csv](https://firms.modaps.eosdis.nasa.gov/data/country/viirs-snpp/2020/viirs-snpp_2020_Australia.csv) |
| Case 2 | [viirs-snpp_2021_United_States.csv](https://firms.modaps.eosdis.nasa.gov/data/country/viirs-snpp/2021/viirs-snpp_2021_United_States.csv) |

All datasets are free. Earth Engine needs a free noncommercial Google Cloud project; the Copernicus
Climate Data Store and NASA LP DAAC need free accounts for direct downloads; the FIRMS CSVs need no login.

## Rebuilding the files

`gee/export_data.py` downloads everything again from Earth Engine and NASA FIRMS and writes
these files (needs an Earth Engine project; see the main README). Its download cache goes to
`data/raw/`, which is not part of the repository.
