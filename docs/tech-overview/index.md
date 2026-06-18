# Technical overview

## Tech stack

- Python 3.13, **uv** + **hatchling** (build), **Ruff** (lint/format), **pytest** (+ cov,
  + socket-blocking).
- numpy / pandas / scipy for the model; SQLAlchemy over SQLite for storage; Streamlit + plotly
  for the dashboard.
- No API key, no private SDK, no external feeds: all market data is synthetic.

## Architecture

The database is the integration hub; the Streamlit UI talks only to a thin read layer.

```
synthetic feeds  ->  Monte Carlo engine  ->  IaR / CIaR  ->  limits + alerts
(iar/ingestion/      (iar/simulation/         + backtest      + dashboard
 synthetic.py)        engine.py)              (iar/risk/)      (app/)
                              |                     |
                              +-------- SQLite ------+  <-- iar.service (read API) <-- app/data_source.py
```

## Synthetic feeds (iar/ingestion/synthetic.py)

Deterministic, market-like, rolling forward in time:

- imbalance-price spread forecast (per-MTU P01-P99 quantiles, heavy-tailed/skewed),
- day-ahead (spot) price (diurnal curve),
- realised imbalance price (DAM + a drawn spread),
- wind portfolio: replays real windsim daily capacity-factor shapes (`windsim_profiles.json`)
  for the position/forecast, with a realistic forecast error for actual delivery.

`iar/ingestion/clients.py` is a factory that returns these synthetic clients; there is no
real-feed code path in this demo.

## Internal algorithms

- Monte Carlo (independent price and imbalance draws), summed-quantile period IaR + CIaR.
- Backtest with as-of vintages and the Kupiec POF test (~5% target at P95).

See [methodology and limitations](../README.md) for the modelling assumptions (independence,
parametric error, illustrative figures) and the path to a production model.

## Project tree

```
iar/        package: db, ingestion (synthetic), simulation, risk, service.py, main.py, __about__.py
app/        dashboard.py (UI), data_source.py (read seam), bootstrap.py (self-seed + live tick)
scripts/    seed_synthetic_demo.py, run_iar.py, load_actuals.py, backfill_history.py, run_backtest.py, refresh.py
config/     app.toml, limits.toml          tests/   pytest suite          docs/   this documentation
```
