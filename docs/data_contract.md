# Data contract (Task 1.6)

> Partially filled from a live probe on 2026-06-03 (NO2). Finalised in Task 1.3/1.6.

Documents the actual shape of the data crossing each boundary, so Week 2 can build
the sampler against what *actually* exists rather than an assumed format.

## Optimeering — forward imbalance forecast

**SDK access pattern** (two steps):

```python
from optimeering import Configuration, OptimeeringClient, PredictionsApi
api = PredictionsApi(OptimeeringClient(configuration=Configuration(api_key=KEY)))

# 1. discover series (filter by area/product/statistic/unit_type/resolution)
series = api.list_series(area=["NO2"], product=["Imbalance"])   # -> PredictionsSeriesList
# 2. pull values for one or more series by integer id
data = api.retrieve_latest(series_id=[1834])                    # -> PredictionsDataList
# (also: api.retrieve(series_id=[...], start=..., end=...) for a historical window)
```

**Series metadata fields** (`list_series` items): `id` (int), `area`, `product`,
`statistic`, `unit_type`, `unit`, `resolution`, `description`, `version`,
`created_at`, `latest_event_time`.

**Vocabulary observed for area NO2** (`Imbalance` product, 8 series):

| field | values seen |
|-------|-------------|
| statistic   | `Quantile`, `Distribution`, `Point`, `Conditional_Index` |
| unit_type   | `Price_Spread` (EUR), `Direction` (N/A) |
| resolution  | `PT1H`, `PT15M` |
| unit        | `EUR` (for Price_Spread), `N/A` (Direction / Conditional_Index) |

Relevant series ids (NO2): Quantile/Price_Spread → **1834** (PT1H), **2036** (PT15M);
Point/Price_Spread → 1829 (PT1H), 2061 (PT15M).

**Quantiles available — DIFFERS BY RESOLUTION:**
- **`PT15M` (id 2036)**: full **9 quantiles — `1, 5, 10, 25, 50, 75, 90, 95, 99`**
  (P01/P05/P10/P25/P50/P75/P90/P95/P99). **P05 and P95 are present**, so a P95 IaR can be
  read directly; only the extreme tails beyond P01/P99 need extrapolation.
- **`PT1H` (id 1834)**: only **5 quantiles — `10, 25, 50, 75, 90`** (no P05/P95).

Since the MVP uses 15-minute MTUs, we use the PT15M series and get the full quantile set.
`value` is a dict keyed by quantile label, e.g.
`{"1": -37.7, "5": ..., "10": ..., "25": ..., "50": ..., "75": ..., "90": ..., "95": ..., "99": ...}`.

**Horizon:** the live forecast returns ~288 fifteen-minute points (~3 days ahead).

**Units / time:** values are **EUR**; timestamps are **UTC, ISO 8601**
(e.g. `2026-06-03T11:00:00+00:00`).

**`retrieve_latest` response shape:**

```
PredictionsDataList
└─ items: [ series ]
   └─ events: [ vintage ]          # created_at, event_time, is_simulated
      └─ predictions: [ point ]
         ├─ prediction_for         # target timestamp (UTC ISO 8601)
         └─ value: { "<quantile>": <EUR float>, ... }   # e.g. {"10": -7.85, "25": -2.73, "50": -1.29, "75": 4.44, "90": 7.63}
```

`event_time` / `created_at` per vintage map directly onto our `SimulationRun.vintage_ts`
for backtesting.

## Optimeering — historical actual imbalance prices

- Method: `api.retrieve(series_id=[...], start=..., end=...)` over a past window
  (`start`/`end` accept ISO 8601 datetimes or durations like `PT1H`, `-P1W1D`, `PT0S`=now).
- **Volume warning:** `retrieve` returns *every forecast vintage* in the window, not a
  single series — a 2-day NO2 window returned ~497k normalised rows (many vintages ×
  288 horizon points × 9 quantiles). Week-3 backfill must filter to the relevant vintage
  per settled period (using `vintage_ts`) rather than loading the whole window blindly.
- Which exact series represents *realised actual* imbalance price (vs forecast vintages):
  to confirm in Task 3.1.

## Flat-file inputs (DAM positions, generation forecasts, actual delivery)

CSV or Excel (`.csv` / `.xlsx`), **one row per MTU**, one file per series per portfolio.
The portfolio is supplied to the loader (not a column in the file).

| file (sample name)              | required columns          | units      | target table          |
|---------------------------------|---------------------------|------------|-----------------------|
| `dam_positions_*.csv`           | `timestamp`, `mwh`        | MWh / MTU  | `dam_positions`       |
| `generation_forecast_*.csv`     | `timestamp`, `forecast_mwh` | MWh / MTU | `generation_forecasts`|
| `actual_delivery_*.csv`         | `timestamp`, `actual_mwh` | MWh / MTU  | `actual_deliveries`   |
| `dam_price_*.csv`               | `timestamp`, `eur_per_mwh` | EUR/MWh   | `dam_prices`          |

**DAM (spot) price — keyed by price *area*, not portfolio.** Optimeering publishes
imbalance only as a `Price_Spread` vs. spot, so **Gross IaR** needs the absolute spot
price to reconstruct `imbalance_price = dam_price + spread`. The `dam_prices` table is
keyed by `(price_area, timestamp)` like the other price series, and is *source-agnostic*:
loaded from a flat file in the MVP, but a future ENTSO-E / Nord Pool / Volue price feed
could write to the same table with no downstream change. Loaded via
`load_dam_prices(session, price_area, path)`. **Spread IaR does not need this file.**

- **`timestamp`**: the MTU *start*, ISO 8601, parsed to **UTC** (e.g. `2026-06-03T11:00:00+00:00`).
- **Resolution**: 15 minutes (PT15M) to match the price forecast.
- **Validation** (loader rejects the file otherwise): required columns present, all
  timestamps parseable and unique, all values numeric/non-empty.
- **Idempotent loads**: by default a load replaces existing rows for that portfolio in
  that table, so re-running does not duplicate data.

**Sample/stub files**: `scripts/make_sample_data.py` generates the three files above for a
synthetic 100 MW NO2 wind portfolio (24h / 96 MTUs), with actual generation wobbling around
forecast so the imbalance is non-trivial. (`data/uploads/` is gitignored; regenerate via the
script.)
