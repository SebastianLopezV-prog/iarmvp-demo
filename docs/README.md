# IaR MVP

Local proof-of-concept of an **Imbalance at Risk (IaR)** tool for a wind generation
portfolio. IaR estimates the worst-case imbalance settlement cost at a chosen
confidence level (e.g. P05 / 95%) over a forward horizon, via Monte Carlo simulation —
conceptually the VaR-equivalent for a portfolio's imbalance position in the
Nordic/European balancing market.

## Scope (MVP)

- DAM positions only; 2–3 users, one wind portfolio each, single price area (NO1/NO2/SE3).
- **Independence assumption**: imbalance price and portfolio imbalance are sampled
  independently. No copula / price↔position dependence (deliberate MVP simplification).
- Outputs: **Gross IaR**, **Spread IaR**, and **CIaR / Expected Shortfall**.
  Period IaR is the quantile of the *summed* P&L across MTUs (not the sum of per-MTU IaRs).

## Architecture (one line)

SQLite is the integration hub; backend logic lives in importable modules; the Streamlit
UI talks only to `iar/service.py`. See the architecture design doc for the full blueprint.

## Project layout

```
iar/         backend package (db, ingestion, simulation, risk, service)
app/         Streamlit dashboard (UI only)
scripts/     run_pipeline.py (headless ingest -> simulate -> store)
config/      app.toml, limits.toml
data/        uploads/ (input CSVs), cache/ (Optimeering responses), iar.db
tests/       pytest suite
docs/        this README, data_contract.md, assumptions.md
```

## Setup

```bash
# from the project root, with the venv active
pip install -e .            # install the iar package (editable) + dependencies
pip install -e ".[dev]"     # include dev tools (pytest)
pytest                      # run the test suite
```

The Optimeering API key must live in `.env` as `OPTIMEERING_API_KEY` (gitignored).
It is never printed, logged, or committed.

## Status

Week 1 in progress. Task 1.1 (repo + project skeleton) complete: package layout,
config, importable empty modules, and `pip install -e .` working.
```
```
