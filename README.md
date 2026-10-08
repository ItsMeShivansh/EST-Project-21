# Cascading Climate Stress (Project 21)

Does one climate extreme make ecosystems more vulnerable to the next one, and slower to recover?
Mid-evaluation scope: **Case 1** (SE Australia, drought → heat + fire) and **Case 2**
(Pacific NW, heat dome → drought + fire), all modules.

EST course project, IIIT Hyderabad. Team: Sahaj Chokshi (2024101031), Shivansh Santoki (2024101119),
Jainam Modi (2024101057), Fraz Uddin (2024201064).

- Presentation: [`presentation/Cascading_Climate_Stress_Mid-Evaluation.pdf`](presentation/Cascading_Climate_Stress_Mid-Evaluation.pdf)
- Results summary: [`RESULTS.md`](RESULTS.md)
- Datasets: [`datasets/`](datasets/README.md)

## Setup (once)

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Run

The input data is already in `datasets/`, so the whole analysis runs offline:

```bash
.venv/bin/python run_all.py          # Modules A–G + figures, about 20 seconds
```

Tables go to `outputs/tables/` and figures to `outputs/figures/`.

### Downloading the data again (optional)

`gee/export_data.py` rebuilds `datasets/<case>/raw.nc` from Earth Engine and NASA FIRMS.
You need a Google Cloud project registered for Earth Engine (free, noncommercial):
https://code.earthengine.google.com/register

```bash
.venv/bin/earthengine authenticate                                 # opens a browser; log in with Google
EE_PROJECT=<your-project-id> .venv/bin/python gee/export_data.py   # resumable
```

It caches every chunk in `data/raw/<case>/`, so if it is interrupted, rerun it and it continues
where it stopped.

## Layout

| Path | What |
|---|---|
| `datasets/<case>/raw.nc` | Input data for each case (see `datasets/README.md`) |
| `config/cases.yaml` | Regions, event dates, thresholds (edit here, not in code) |
| `gee/export_data.py` | Earth Engine + NASA FIRMS download → `datasets/<case>/raw.nc` |
| `src/preprocess.py` | **A**: land-cover mask, z-anomalies vs 2001–2016, NDVI seasonal ratio |
| `src/events.py` | **B**: drought / heatwave / fire detection; per-cell table `outputs/tables/<case>_cells.csv` |
| `src/sensitivity.py` | **C**: matched controls, Mann–Whitney, Cliff's δ, bootstrap CI, interaction model, fire logistic regression, ecosystem split |
| `src/recovery.py` | **D**: recovery time, Lloret indices, τ, deficit, Kaplan–Meier + log-rank, state shifts, lag correlation |
| `src/dose_response.py` | **E**: binned dose–response, LOWESS, segmented regression threshold |
| `src/robustness.py` | **F**: CHIRPS vs ERA5, SPEI vs soil moisture, MODIS vs VIIRS, alternative threshold |
| `src/synthesis.py` | **G**: cross-case table + policy implications |
| `src/plots.py` | Figures 1–15 → `outputs/figures/` |
| `outputs/` | Result tables and figures from the last run |
| `presentation/` | Mid-evaluation slides (PDF) |

## Data (all on a 0.1° monthly grid, 2001–2024)

| Dataset | Source | Variables |
|---|---|---|
| ERA5-Land | `ECMWF/ERA5_LAND/DAILY_AGGR` | Tmax (daily + monthly), VPD, soil moisture, precipitation |
| SPEI v2.11 | `CSIC/SPEI/2_11` | SPEI-03, SPEI-12 |
| CHIRPS | `UCSB-CHG/CHIRPS/PENTAD` | precipitation |
| MODIS | `MOD17A2HGF`, `MOD13Q1`, `MCD64A1`, `MOD14A1`, `MCD12Q1` | GPP, NDVI, burned area, FRP, land cover |
| VIIRS | NASA FIRMS country archive | active-fire detections, FRP |
| SRTM | `USGS/SRTMGL1_003` | elevation (matching covariate) |

## Key definitions

- **Hit by event 1**: Case 1 min SPEI-12 (2017–19) ≤ −1.5; Case 2 peak daily Tmax anomaly during the heat dome ≥ 2 SD.
- **Burned**: ≥ 30% of the cell burned in the fire window; **unburned**: < 2%. Cells burned in the 3 years before event 1 are excluded.
- **Groups**: Double-hit (event 1 + burned), Event 2 only (burned), Event 1 only (unburned), Neither.
- **Recovered**: NDVI ≥ 90% of its seasonal normal for 3 consecutive months.
- Standard errors in regressions are clustered by 1° blocks to account for spatial autocorrelation.
