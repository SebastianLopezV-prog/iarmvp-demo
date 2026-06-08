# IaR MVP — Project Context

> This file is read automatically by Claude Code at the start of every session.
> It is the standing source of truth for scope, stack, and conventions. Read the
> reference docs below before writing code.

## What this is

A local proof-of-concept of an **Imbalance at Risk (IaR)** tool for a wind generation
portfolio. IaR estimates the worst-case imbalance settlement cost at a chosen confidence
level (e.g. the 95th percentile / P05 loss) over a forward horizon, via Monte Carlo
simulation. It is conceptually analogous to Value at Risk (VaR) in finance, applied to a
portfolio's imbalance position in the Nordic/European electricity balancing market.

Built by 2 junior data scientists who are new to Volue's systems. **Favour simplicity and
pragmatism — this is a PoC, not a production system.**

### Domain in one paragraph
A wind portfolio's market position is its Day-Ahead Market (DAM) sales (MWh per interval).
Actual delivery is metered generation. Portfolio imbalance = DAM position − actual delivery.
Imbalance settles at the TSO imbalance price, so unexpected imbalances create unplanned
costs/revenues. IaR puts a probabilistic number on that exposure.

## Reference docs (read these first)

- `IaR_MVP_architecture.pdf` — component design, data model, folder structure, and the
  seven key design decisions. This is the primary blueprint.
- `IaR_MVP_4_week_plan.pdf` — task breakdown, sequencing, dependencies, and risks.
- `iar_refined.py` — the existing Monte Carlo prototype to adapt into the engine
  (see the reuse rules below).

## Critical scope rules (do NOT exceed)

- **DAM positions only.** 2–3 users, one wind portfolio each, single price area per
  portfolio (NO1 / NO2 / SE3).
- **Independence assumption.** The Monte Carlo samples the imbalance price and the
  portfolio imbalance **independently**. This is an explicit, deliberate MVP simplification.
  Do **NOT** add copula or any price↔position dependence modelling.
- **Reusing `iar_refined.py`:** keep the per-scenario P&L construction, the
  summed-quantile period IaR (the quantile of summed P&L across MTUs — NOT the sum of
  per-MTU IaRs), the Gross vs Spread IaR distinction, and CIaR / Expected Shortfall.
  **REMOVE the t-copula entirely** — leave no half-wired dependence logic anywhere.
- **Out of scope:** correlation/dependence modelling, intraday or balancing-market
  positions, multi-area netting, live trading integration, regulatory reporting,
  cloud deployment.

### IaR definitions to implement
- **Gross IaR** = worst-case total settlement cost: position × imbalance price.
- **Spread IaR** = worst-case underperformance vs day-ahead: position × (imbalance price − DAM price).
- **CIaR / Expected Shortfall** = average loss across the worst scenarios beyond the IaR threshold.
- Period IaR is the quantile of the summed P&L across all MTUs in the horizon.

## Stack

- **Python only.**
- numpy, pandas, scipy (math + stats), SQLAlchemy over SQLite (storage),
  Streamlit + plotly (dashboard).
- **Optimeering** official Python SDK (`optimeering` package) for both
  (a) forward imbalance price forecasts with quantile/distribution statistics, and
  (b) historical actual imbalance prices (for backtesting), per price area.
- Flat files (CSV/Excel) are the ingestion source for DAM positions, generation
  forecasts, and actual delivery.

## Secrets

- `OPTIMEERING_API_KEY` lives in `.env` (already created and gitignored).
- Load it via `python-dotenv` (`load_dotenv()` then `os.getenv(...)`).
- The Optimeering client is configured as:
  `Configuration(api_key=os.getenv("OPTIMEERING_API_KEY"))` →
  `OptimeeringClient(configuration=configuration)`.
- **NEVER print, log, commit, or hardcode the key.** It only ever lives in `.env`.

## Architecture conventions

- **Database is the integration hub.** Components communicate through stored state in
  SQLite, not by calling each other directly. This keeps each module independently testable.
- **Backend separate from UI.** All calculation/logic lives in importable backend modules.
  The Streamlit UI calls only the functions exposed in `service.py` — it contains no
  simulation or DB logic itself.
- **Swappable imbalance-uncertainty model.** With no historical forecast-error data yet,
  model uncertainty on (DAM − forecast gen) parametrically (e.g. Gaussian/scaled-t with
  sigma as a configurable fraction of forecast/capacity). Keep it a single module so the
  backtest can recalibrate it.
- **Store summary results, not raw scenarios.** Persist IaR/CIaR by confidence + horizon;
  regenerate scenario vectors from a stored seed if ever needed.
- **Backtesting vintages.** Stamp each SimulationRun with a `vintage_ts` (as-of time of its
  inputs) so the backtest joins each settled period's realised cost to the IaR estimate
  whose vintage precedes it. Exceedance frequency checked with a Kupiec POF test (~5% at P95).

## Target folder structure (from the architecture doc)

```
iar_mvp/
├── config/        app.toml, limits.toml
├── data/          uploads/ (input CSVs), cache/ (Optimeering responses), iar.db
├── iar/
│   ├── db/        models.py (SQLAlchemy), session.py
│   ├── ingestion/ optimeering_client.py, flatfile_loader.py
│   ├── simulation/ imbalance_model.py, price_sampler.py, engine.py
│   ├── risk/      alerts.py, backtest.py
│   └── service.py  backend interface the UI calls
├── app/           dashboard.py (Streamlit, UI only)
├── scripts/       run_pipeline.py (headless ingest→simulate→store)
├── tests/
└── docs/          README.md, data_contract.md, assumptions.md
```

## Working style

- Work in **small, focused sessions** — one bounded objective at a time
  (e.g. "build the Optimeering client", then a fresh session for "build the DB schema").
  Don't attempt the whole MVP in one go.
- **Run and test your own code.** When something errors, read the error and fix it.
- **Review before committing.** Never stage `.env`. Confirm `.gitignore` is working before
  the first commit so the key never enters git history.

## Current status (Week 2 COMPLETE)

Environment: Windows, Python 3.13, venv at `venv/` (use `.\venv\Scripts\python.exe`).
Package installed editable (`pip install -e ".[dev]"`). **67 tests passing.**
(Note: the Read tool can't render PDFs here — `pdftoppm` is missing; extract PDF text with
`pypdf` instead.)

Run commands (Week 2):
```
.\venv\Scripts\python.exe scripts\run_iar.py            # run the MC, print Gross/Spread IaR+CIaR
.\venv\Scripts\python.exe scripts\run_iar.py --store    # ...and persist the run to data/iar.db
.\venv\Scripts\python.exe scripts\validate_engine.py    # 2.5 validation report (8/8 checks)
.\venv\Scripts\python.exe scripts\verify_all.py         # 1.1->2.x end-to-end health check
.\venv\Scripts\python.exe -m pytest -q                  # full suite (67 tests)
```

**Week 1 COMPLETE (1.1–1.5), all tested and verified against the live Optimeering API:**
- **1.1** repo + package skeleton; `pip install -e .` works.
- **1.2** SQLite schema — 11 SQLAlchemy models in `iar/db/models.py` + `init_db()`/session
  factory in `iar/db/session.py`. FK enforcement on; CHECK/UNIQUE constraints; index on
  `(portfolio_id, vintage_ts)`.
- **1.3** `iar/ingestion/optimeering_client.py` — `OptimeeringForecastClient` with
  `get_imbalance_price_forecast` + `get_historical_prices`; key auth from `.env`,
  transient-only retry/backoff, on-disk cache, responses normalised to flat dicts that map
  1:1 onto `ImbalancePriceForecast`. Verified live.
- **1.4** `iar/ingestion/flatfile_loader.py` — CSV/Excel loaders + validation + idempotent
  loads; `scripts/make_sample_data.py` generates stub NO2 wind CSVs.
- **1.5** `scripts/run_pipeline.py` — end-to-end: fetch real forecast → load stub positions →
  compute expected imbalance (DAM − forecast gen) → persist to `data/iar.db` → print summary.

Run commands:
```
.\venv\Scripts\python.exe scripts\make_sample_data.py   # (re)generate stub CSVs
.\venv\Scripts\python.exe scripts\run_pipeline.py        # end-to-end smoke path
.\venv\Scripts\python.exe -m pytest -q                   # test suite
```

**Key findings — READ `docs/data_contract.md` and `docs/assumptions.md`:**
- Optimeering is a two-step API: `list_series(...)` → ids, then
  `retrieve_latest(series_id=[...])` / `retrieve(..., start, end)`. Response nests
  items → events (vintages) → predictions (`prediction_for`, `value`).
- **Quantiles differ by resolution.** The **PT15M** Imbalance series (the one we use) gives the
  full 9 quantiles **incl. P05/P95** (`1,5,10,25,50,75,90,95,99`); the PT1H series gives only 5.
- **Imbalance is published as `Price_Spread` (EUR), not absolute price.** Spread IaR works
  directly; **Gross IaR needs absolute price = DAM price + spread**. Note: the DAM price term
  is *baked into* Optimeering's spread (it publishes `imbalance_price − spot`), so SPREAD IaR
  needs no separate DAM input; only GROSS does (to rebuild the absolute price). See the note
  block in `engine.py` at the cost computation.
- **Optimeering (public SDK) does NOT publish day-ahead/spot price.** Confirmed against the
  *full* catalogue this key can see — **625 series, 13 areas (DE/DK/FI/NO1-5/SE1-4), 17
  products — every one balancing-market** (FCR/aFRR/mFRR/Imbalance). The word "spot" appears
  only in imbalance descriptions as the spread baseline. So the spot price must come from
  elsewhere (see OPEN ITEM below).
- **Spot price ingestion is built and source-agnostic:** flat file `dam_price_*.csv` →
  `dam_prices` table (keyed by area) via `load_dam_prices(...)`. Currently fed by a SYNTHETIC
  stub (`make_sample_data.py`, `TODO(dam-source)`); a real feed writes the same table with no
  downstream change. **DAM is a cleared auction price → a deterministic engine input (not
  sampled); only the imbalance spread is simulated.**
- **Client host is pinned** to production (`DEFAULT_HOST = https://app.optimeering.com`) in
  `optimeering_client.py` (override via `OPTIMEERING_HOST`). We use the **public/external**
  `optimeering` SDK with **API-key** auth.
- Historical `retrieve` returns every forecast vintage in the window (2 days ≈ 500k rows /
  122 MB) — Week-3 backfill must filter by `vintage_ts`, not load whole windows.

**Week 2 progress (`iar_refined.py` provided; t-copula deliberately NOT carried over):**
- **2.1 DONE** `iar/simulation/imbalance_model.py` — parametric per-MTU imbalance
  distribution (Gaussian / scaled-t, sigma a configurable fraction of forecast/capacity);
  exposes `ppf` (the separable uniform-draw seam) + `sample`.
- **2.2 DONE** `iar/simulation/price_sampler.py` — `QuantilePriceSampler`, inverse-CDF
  sampling from the Optimeering quantiles; preserves asymmetry/tails (no Normal refit);
  linear/clamp tail policy.
- **2.3 DONE** `iar/simulation/engine.py` — `run_simulation`: independent price + position
  sampling, per-scenario **summed-quantile** Gross/Spread IaR + CIaR. The copula insertion
  point is a real swappable seam (`ScenarioDraw` Protocol; `IndependentDraw` default) — NO
  copula code, but a future copula is a clean drop-in. Validation tests live in
  `tests/test_engine.py` (analytic Normal, convergence, reproducibility, summed-quantile-vs-naive).
- **2.4 DONE** `iar/simulation/persistence.py` — `persist_report` writes one
  `SimulationRun` + a `gross` and `spread` `IaRResult`; stores seed/n_scenarios/vintage_ts
  for reproducibility & backtest joins (summaries only, not raw scenarios). Wired into
  `run_iar.py --store`. Tests in `tests/test_persistence.py`.
- **2.5 DONE** validation. Automated in `tests/test_engine.py`; human-readable report in
  `scripts/validate_engine.py` (8/8: analytic all-Normal IaR+CIaR, 1/sqrt(N) convergence,
  seed reproducibility, summed-quantile-vs-naive). Story written up in `docs/validation.md`.

Engine sign convention: `cost = imbalance × price` (positive = cost/bad); IaR is the
upper-tail quantile of summed cost, CIaR the mean beyond it.

**NEXT: Week 3** — risk/alerts + backtesting (`iar/risk/`). Backtest joins each settled
period's realised cost to the IaR estimate whose `vintage_ts` precedes it; Kupiec POF test
(~5% exceedances at P95); recalibrate the imbalance-model `sigma` against realised outcomes.

**Repo:** private GitHub `SebastianLopezV-prog/iarmvp`. A local PostToolUse hook
auto-commits after every Edit/Write (`.claude/settings.local.json`); pushes are manual.
