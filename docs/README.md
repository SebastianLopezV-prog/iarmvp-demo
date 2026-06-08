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
  ingestion/   optimeering_client.py (spread forecast, public SDK),
               markets_client.py (real DAM spot price, internal optipyclient SDK),
               flatfile_loader.py (CSV/Excel + DAM-price store)
  simulation/  imbalance_model.py (2.1), price_sampler.py (2.2), engine.py (2.3),
               persistence.py (2.4)
  risk/        alerts.py, backtest.py        (Week 3 — not yet built)
  service.py   backend interface for the UI (Week 3)
app/           Streamlit dashboard (UI only — skeleton)
scripts/       run_pipeline.py  (ingest -> store smoke path)
               run_iar.py       (run the Monte Carlo, print/store IaR)
               load_windsim_data.py (REAL portfolio data via the client CSV path)
               validate_engine.py (2.5 validation report)
               verify_all.py    (1.1 -> 2.x end-to-end health check)
               make_sample_data.py (offline stub NO2 wind CSVs)
config/        app.toml, limits.toml
data/          uploads/ (input CSVs), cache/ (Optimeering responses), iar.db
tests/         pytest suite (73 tests)
docs/          README.md, data_contract.md, assumptions.md, validation.md
```

## Setup

Windows, Python 3.13, venv at `venv/`.

```powershell
pip install -e ".[dev]"                       # install package (editable) + dev tools
.\venv\Scripts\python.exe -m pytest -q         # run the full suite (73 tests)
```

The Optimeering API key must live in `.env` as `OPTIMEERING_API_KEY` (gitignored).
It is never printed, logged, or committed. The client pins the production host
(`https://app.optimeering.com`).

### ⚠️ REQUIRED for real DAM (spot) prices — install the internal SDK wheel

The day-ahead **spot price** (needed for **Gross IaR**) comes from Optimeering's
**internal** SDK, **`optipyclient`**, which is **NOT on PyPI** — it is a **vendored wheel**
published on the **`Volue/sirius-prime` GitHub Releases** page. `pip install -e .` does
**not** install it, so you must add it manually:

1. Open `https://github.com/Volue/sirius-prime/releases` (requires Volue org access; the
   `Volue` org enforces SAML SSO — authorize your GitHub credential for it).
2. Download the latest **`optipyclient-*.whl`** asset.
3. Put it in **`vendor/`** (gitignored) and install into the venv:

   ```powershell
   .\venv\Scripts\python.exe -m pip install --force-reinstall vendor\optipyclient-*.whl
   ```

   *(Or, with the GitHub CLI authenticated:*
   `gh release download --repo Volue/sirius-prime -p "*.whl" -D vendor/ --clobber`*.)*

Auth reuses the same `OPTIMEERING_API_KEY` (no OAuth needed). **Without this wheel**, the
markets client raises a clear error and `run_iar.py` falls back to a flat `--dam-price`
stub (Spread IaR is unaffected; only real Gross IaR needs it).

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

**Week 1 (ingestion + storage) and Week 2 (Monte Carlo engine) COMPLETE — 73 tests passing.**

- **Week 1 (1.1–1.5):** package skeleton; SQLite schema; Optimeering forecast client
  (live, cached, retry); flat-file loaders + validation; end-to-end ingest pipeline.
- **Week 2 (2.1–2.5):** parametric imbalance model; quantile price sampler (tails
  preserved, no Normal refit); independent Monte Carlo engine with the summed-quantile
  Gross/Spread IaR + CIaR and a swappable copula seam (no copula code); DB persistence
  of runs/results; validation (analytic Normal, 1/√N convergence, reproducibility,
  summed-quantile-vs-naive — see `docs/validation.md`).

**DAM/spot price — RESOLVED.** The public SDK has no spot price, but the **internal**
`optipyclient` SDK's `MarketsApi` serves the **DAM cleared price** (the spot) — wired in via
`iar/ingestion/markets_client.py` and stored in the `dam_prices` table. `run_iar.py` now
computes **real Gross IaR** over the MTUs where both the live spread and the real DAM price
exist. *(Requires the vendored wheel — see Setup above.)*

**Remaining caveat:** the **portfolio** (positions/generation) and the imbalance-model
`sigma` are still stubs, so absolute euro figures stay illustrative until real portfolio
files are loaded and `sigma` is calibrated (Week 3).

**Next — Week 3 (`iar/risk/`):** backtesting (join settled cost to the IaR estimate whose
`vintage_ts` precedes it; Kupiec POF test), sigma recalibration, and alerts.

## Repo

Primary: `Volue/iarmvp` (GitHub). A local PostToolUse hook auto-commits after each
edit; pushes are manual.
