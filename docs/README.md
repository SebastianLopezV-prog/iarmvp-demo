# IaR MVP

Local proof-of-concept of an **Imbalance at Risk (IaR)** tool for a wind generation
portfolio. IaR estimates the worst-case imbalance settlement cost at a chosen
confidence level (e.g. P05 / 95%) over a forward horizon, via Monte Carlo simulation —
conceptually the VaR-equivalent for a portfolio's imbalance position in the
Nordic/European balancing market.

## Scope (MVP)

- DAM positions only; 2–3 users, one wind portfolio each, single price area (NO1/NO2/SE3).
- **Independence assumption**: imbalance price and portfolio imbalance are sampled
  independently. No copula / price↔position dependence (deliberate MVP simplification),
  but the random-draw step is a swappable seam so a copula can drop in later.
- Outputs: **Gross IaR**, **Spread IaR**, and **CIaR / Expected Shortfall**.
  Period IaR is the quantile of the *summed* P&L across MTUs (not the sum of per-MTU IaRs).

## Architecture (one line)

SQLite is the integration hub; backend logic lives in importable modules; the Streamlit
UI talks only to `iar/service.py`. See the architecture design doc for the full blueprint.

## Project layout

```
iar/
  db/          SQLAlchemy models + session (init_db, get_session)
  ingestion/   optimeering_client.py (live forecast), flatfile_loader.py (CSV/Excel)
  simulation/  imbalance_model.py (2.1), price_sampler.py (2.2), engine.py (2.3),
               persistence.py (2.4)
  risk/        alerts.py, backtest.py        (Week 3 — not yet built)
  service.py   backend interface for the UI (Week 3)
app/           Streamlit dashboard (UI only — skeleton)
scripts/       run_pipeline.py  (ingest -> store smoke path)
               run_iar.py       (run the Monte Carlo, print/store IaR)
               validate_engine.py (2.5 validation report)
               verify_all.py    (1.1 -> 2.x end-to-end health check)
               make_sample_data.py (stub NO2 wind CSVs)
config/        app.toml, limits.toml
data/          uploads/ (input CSVs), cache/ (Optimeering responses), iar.db
tests/         pytest suite (67 tests)
docs/          README.md, data_contract.md, assumptions.md, validation.md
```

## Setup

Windows, Python 3.13, venv at `venv/`.

```powershell
pip install -e ".[dev]"                       # install package (editable) + dev tools
.\venv\Scripts\python.exe -m pytest -q         # run the full suite (67 tests)
```

The Optimeering API key must live in `.env` as `OPTIMEERING_API_KEY` (gitignored).
It is never printed, logged, or committed. The client pins the production host
(`https://app.optimeering.com`).

## Run

```powershell
.\venv\Scripts\python.exe scripts\run_iar.py            # Monte Carlo -> Gross/Spread IaR + CIaR
.\venv\Scripts\python.exe scripts\run_iar.py --store    # ...and persist the run to data/iar.db
.\venv\Scripts\python.exe scripts\validate_engine.py    # engine validation report (8/8 checks)
.\venv\Scripts\python.exe scripts\verify_all.py         # end-to-end health check (1.1 -> 2.x)
.\venv\Scripts\python.exe scripts\run_pipeline.py       # ingestion smoke path
```

Engine sign convention: `cost = imbalance × price` (positive = cost/bad). IaR is the
upper-tail quantile of the summed cost; CIaR is the mean beyond it.

## Status

**Week 1 (ingestion + storage) and Week 2 (Monte Carlo engine) COMPLETE — 67 tests passing.**

- **Week 1 (1.1–1.5):** package skeleton; SQLite schema; Optimeering forecast client
  (live, cached, retry); flat-file loaders + validation; end-to-end ingest pipeline.
- **Week 2 (2.1–2.5):** parametric imbalance model; quantile price sampler (tails
  preserved, no Normal refit); independent Monte Carlo engine with the summed-quantile
  Gross/Spread IaR + CIaR and a swappable copula seam (no copula code); DB persistence
  of runs/results; validation (analytic Normal, 1/√N convergence, reproducibility,
  summed-quantile-vs-naive — see `docs/validation.md`).

**Open item — NO2 day-ahead/spot price source (blocks *real* Gross IaR; Spread IaR is fine).**
Optimeering's public catalogue (625 series, all areas) is balancing-market only — it
publishes the imbalance *spread* (`imbalance_price − spot`), not the spot price itself.
Spread IaR uses the spread directly; **Gross IaR needs the spot price added back**, which
is currently a **synthetic stub** (`dam_prices` table / `TODO(dam-source)`). A real feed
(Optimeering internal access via OAuth, ENTSO-E, or Nord Pool) writes the same table with
no downstream change. See `CLAUDE.md` for the full investigation.

**Next — Week 3 (`iar/risk/`):** backtesting (join settled cost to the IaR estimate whose
`vintage_ts` precedes it; Kupiec POF test), sigma recalibration, and alerts.

## Repo

Primary: `Volue/iarmvp` (GitHub). A local PostToolUse hook auto-commits after each
edit; pushes are manual.
