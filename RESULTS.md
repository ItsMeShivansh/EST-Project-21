# Mid-evaluation results (Cases 1 and 2)

All numbers come from `outputs/tables/`; figures are in `outputs/figures/`. Rerun with `.venv/bin/python run_all.py`.

## Data checks (Module F)

| Check | Case 1 SE Australia | Case 2 Pacific NW |
|---|---|---|
| CHIRPS vs ERA5-Land monthly rain | r = 0.88, bias −2% | r = 0.93, bias +15% |
| MODIS vs VIIRS fire per cell | ρ = 0.76 | ρ = 0.63 |
| SPEI-03 vs ERA5 soil moisture | r = 0.65 | r = 0.49 |
| Burned area, fire season (MODIS) | 5.6 Mha in box | small, scattered |

## Results by hypothesis

| | Case 1: drought → heat + fire | Case 2: heat dome → drought + fire |
|---|---|---|
| Share of cells hit by event 1 | 74% (SPEI-12 ≤ −1.5) | 97% (≥ 2 SD) |
| **H1 GPP sensitivity** (hit − nearby matched control) | −0.01 SD [−0.03, 0.01], p = 0.08 → **not supported** | −0.27 SD [−0.38, −0.15], p < 0.001 → **suggestive** (sign flips at 3 SD) |
| H1 by ecosystem | Forest −0.37 SD (p < 0.001); grass/shrub none | Grassland −0.29 SD |
| **H1 fire** (total effect, per SD of event-1 severity) | **OR 2.4 [1.3, 4.4], p = 0.004**; forests: 66% vs 21% burned (top vs bottom third) | not testable (37 burned cells) |
| Fire mechanism | Effect runs through fire-season soil moisture (OR 0.34 per SD, p = 0.003) | — |
| **H2 dose–response** | Burn probability rises to SPEI ≈ −1.8 then levels off | GPP loss grows with heat severity (ρ = −0.33), steeper past ≈ 3.5 SD |
| **H3 recovery** (cells ≥ 60% burned) | NDVI median 6 months, GPP 8 months; all recovered by 2024; no drought effect (p = 0.34) | 17 cells: 71% not recovered by 2024, 59% state shift |

## Main observations

1. The drought → fire cascade is clear in Case 1 and works through drier soils and fuels.
2. A first extreme did not reliably make GPP more sensitive to the next one: no effect after drought, a modest but threshold-dependent effect after heat.
3. Recovery depends more on ecosystem type than on the first extreme: eucalypts resprouted within months (helped by wet 2020–22), burned Pacific NW conifers have not recovered.
4. Both first extremes were region-wide, so we use nearby matched controls, severity terciles and dose–response instead of a simple yes/no comparison.

## Changes made after seeing the data (all documented in code and config)

- Controls matched on location as well as elevation and event-2 stress.
- Fire model reports the total effect separately from the mechanism model.
- Recovery uses heavily burned cells (≥ 60%) and tracks GPP as well as NDVI.
- Case 2 alternative threshold set to 3 SD (at 1.5 SD every cell is hit).
- MODIS active-fire counts recomputed with a mean-based aggregation (the sum-based one undercounted).
