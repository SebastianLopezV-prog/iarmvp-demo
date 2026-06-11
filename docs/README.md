# IaR MVP

Local proof-of-concept of an Imbalance at Risk (IaR) tool for a wind generation
portfolio. IaR estimates the worst-case imbalance settlement cost at a chosen
confidence level (for example P05 / 95%) over a forward horizon, using Monte Carlo
simulation. It is the VaR-equivalent for a portfolio's imbalance position in the
Nordic and European balancing market.

## Scope (MVP)

- DAM positions only; 2 to 3 users, one wind portfolio each, single price area
  (NO1, NO2, or SE3).
- Independence assumption: the imbalance price and the portfolio imbalance are
  sampled independently. There is no copula or price/position dependence (a
  deliberate MVP simplification), but the random-draw step is a swappable seam so a
  copula can be added later.
- Outputs: Gross IaR, Spread IaR, and CIaR / Expected Shortfall. Period IaR is the
  quantile of the summed P&L across MTUs, not the sum of per-MTU IaRs.

## What you need to add (not in the repo)

Cloning gives you the code, but four things live outside git and you must add them
yourself. The tool degrades gracefully: with items 1 and 2 you get live Spread IaR
(Gross and portfolio stubbed); add items 3 and 4 to make everything real except the
imbalance sigma.

| # | What to add | Required for | If missing |
|---|-------------|--------------|------------|
| 1 | Python 3.11+ and a venv, then `pip install -e ".[dev]"` | everything | nothing runs |
| 2 | `.env` file with `OPTIMEERING_API_KEY=<key>` | any live data (spread, DAM) | live fetches fail |
| 3 | `optipyclient` wheel in `vendor/` (from `Volue/sirius-prime` Releases) | real DAM spot, so real Gross IaR | flat `--dam-price` stub (Spread IaR still works) |
| 4 | `windsim` (`pip install git+...sirius-imb-at-risk-mvp`) | real portfolio positions, generation, actuals | synthetic stub portfolio |

Items 3 and 4 are private Volue packages (not on PyPI) and require Volue GitHub org
access. The org enforces SAML SSO, so authorize your git credential once. Exact
install commands are in Setup below.

## Architecture (one line)

SQLite is the integration hub; backend logic lives in importable modules; the
Streamlit UI talks only to `iar/service.py`. See the architecture design doc for the
full blueprint.

## Project layout

```
iar/
  db/          SQLAlchemy models + session (init_db, get_session)
  ingestion/   optimeering_client.py (spread forecast, public SDK),
               markets_client.py (real DAM spot price, internal optipyclient SDK),
               flatfile_loader.py (CSV/Excel + DAM-price store)
  simulation/  imbalance_model.py (2.1), price_sampler.py (2.2), engine.py (2.3),
               persistence.py (2.4)
  risk/        realised_cost.py (3.1), replay.py (3.2), backtest.py (3.2/3.3),
               alerts.py (3.4), calibration.py (sigma calibration)
  service.py   frozen read API for the UI (3.5)
app/           Streamlit dashboard (UI only; skeleton for service.py, built in 4.1)
scripts/       run_pipeline.py       (ingest then store smoke path)
               run_iar.py            (run the Monte Carlo, print/store IaR + alerts)
               load_windsim_data.py  (real portfolio data via the client CSV path)
               load_actuals.py       (3.1, realised prices + realised cost)
               backfill_iar.py       (3.2, backfill day-ahead IaR vintages)
               run_backtest.py       (3.3, exceedances + Kupiec)
               calibrate_sigma.py    (sweep sigma against the backtest)
               seed_demo.py          (clean one-portfolio-per-area demo: NO1/NO2/SE3)
               validate_engine.py / verify_all.py / make_sample_data.py
config/        app.toml, limits.toml
data/          uploads/ (input CSVs), cache/ (Optimeering responses), iar.db
tests/         pytest suite (142 tests; hermetic, no key/network/wheel needed)
docs/          README.md, data_contract.md, assumptions.md, validation.md
```

## Setup

Developed on Windows, Python 3.13, with the venv at `venv/`.

```powershell
pip install -e ".[dev]"                       # install package (editable) + dev tools
.\venv\Scripts\python.exe -m pytest -q         # run the full suite (142 tests)
```

The test suite is self-contained: it needs no API key, network, vendor wheel, or
`windsim`, because all external SDKs are mocked or injected. A fresh clone plus
`pip install -e ".[dev]"` should pass immediately. Requires Python 3.11+ (for
`tomllib`); developed on 3.13.

The Optimeering API key must live in `.env` as `OPTIMEERING_API_KEY` (gitignored).
It is never printed, logged, or committed. The client pins the production host
(`https://app.optimeering.com`).

### Required for real DAM (spot) prices: install the internal SDK wheel

The day-ahead spot price (needed for Gross IaR) comes from Optimeering's internal
SDK, `optipyclient`, which is not on PyPI. It is a vendored wheel published on the
`Volue/sirius-prime` GitHub Releases page. `pip install -e .` does not install it, so
add it manually:

1. Open `https://github.com/Volue/sirius-prime/releases` (requires Volue org access;
   the `Volue` org enforces SAML SSO, so authorize your GitHub credential for it).
2. Download the latest `optipyclient-*.whl` asset.
3. Put it in `vendor/` (gitignored) and install it into the venv:

   ```powershell
   .\venv\Scripts\python.exe -m pip install --force-reinstall vendor\optipyclient-*.whl
   ```

   Or, with the GitHub CLI authenticated:
   `gh release download --repo Volue/sirius-prime -p "*.whl" -D vendor/ --clobber`

Auth reuses the same `OPTIMEERING_API_KEY` (no OAuth needed). Without this wheel, the
markets client raises a clear error and `run_iar.py` falls back to a flat
`--dam-price` stub. Spread IaR is unaffected; only real Gross IaR needs it.

### Optional: real portfolio data via `windsim`

Real positions, generation, and actuals come from Volue's wind-portfolio simulator,
`windsim` (in `Volue/sirius-imb-at-risk-mvp`, also private, not on PyPI). Install it
from the repo, then ingest a portfolio the client way (it generates the data, writes
the three upload CSVs, and loads them through the flat-file loaders into
`data/iar.db`):

```powershell
.\venv\Scripts\python.exe -m pip install "git+https://github.com/Volue/sirius-imb-at-risk-mvp.git"
.\venv\Scripts\python.exe scripts\load_windsim_data.py --windsim-portfolio north --area NO2
```

`run_iar.py` then uses these real positions automatically: it reads them from the DB
and simulates over the MTUs where the live spread, real DAM price, and real positions
overlap. Without it, `run_iar.py` falls back to a synthetic stub portfolio.

## Run

```powershell
.\venv\Scripts\python.exe scripts\load_windsim_data.py  # ingest real portfolio data (client CSV path)
.\venv\Scripts\python.exe scripts\run_iar.py            # Monte Carlo, prints Gross/Spread IaR + CIaR
.\venv\Scripts\python.exe scripts\run_iar.py --store    # ...and persist the run to data/iar.db
.\venv\Scripts\python.exe scripts\load_actuals.py       # realised prices + realised cost (3.1)
.\venv\Scripts\python.exe scripts\backfill_iar.py       # backfill day-ahead IaR vintages (3.2)
.\venv\Scripts\python.exe scripts\run_backtest.py       # exceedances + Kupiec test (3.3)
.\venv\Scripts\python.exe scripts\calibrate_sigma.py    # sweep sigma against the backtest
.\venv\Scripts\python.exe scripts\validate_engine.py    # engine validation report (8/8 checks)
.\venv\Scripts\python.exe scripts\verify_all.py         # end-to-end health check
```

Engine sign convention: `cost = imbalance * price` (positive means cost, i.e. bad).
IaR is the upper-tail quantile of the summed cost; CIaR is the mean beyond it.

## Dashboard

The Streamlit dashboard (`app/dashboard.py`) is built in Week 4.1 on top of
`service.py`, reading only through the service layer (no simulation or DB logic in the
UI). An earlier throwaway demo UI was removed; the rebuilt version replaces it.

## Status

Weeks 1, 2, and 3 are complete. 142 tests passing.

- Week 1 (1.1 to 1.5): package skeleton; SQLite schema; Optimeering forecast client
  (live, cached, retry); flat-file loaders + validation; end-to-end ingest pipeline.
- Week 2 (2.1 to 2.5): parametric imbalance model; quantile price sampler (tails
  preserved, no Normal refit); independent Monte Carlo engine with the summed-quantile
  Gross/Spread IaR + CIaR and a swappable copula seam (no copula code); DB persistence
  of runs and results; validation (analytic Normal, 1/sqrt(N) convergence,
  reproducibility, summed-quantile vs naive; see `docs/validation.md`).
- Week 3 (3.1 to 3.5): realised imbalance cost; IaR vintage replay and the comparison
  join; exceedance frequency and the Kupiec POF test; configurable per-portfolio
  euro-limits and alerts; the frozen `service.py` read API. Plus a sigma calibration
  routine that sweeps sigma against the backtest.

Data sources that are now real: the imbalance price spread (Optimeering public SDK),
the DAM cleared (spot) price (internal `optipyclient` SDK, requires the wheel), and
portfolio positions, generation, and actuals (windsim, via the client CSV path). The
only remaining parametric stub is the imbalance-model sigma (forecast-error size),
which the calibration routine can now tune against realised outcomes.

Next is Week 4: build the real dashboard (`app/dashboard.py`) on top of `service.py`,
the backtest view, end-to-end tests across NO1/NO2/SE3, documentation, and demo prep.

## Repo

Primary: `Volue/iarmvp` (GitHub). A backup remote `personal` points at
`SebastianLopezV-prog/iarmvp`. A local PostToolUse hook auto-commits after each edit;
pushes are manual.

---

# From MVP to production

This is a proof of concept, built quickly to prove the pipeline end-to-end. It is
deliberately not production software. This section lists every shortcut taken and
everything that would have to change to make it a real product. Read it before
quoting any IaR number to a customer.

## A. Shortcuts and simplifications we took

Modelling / quant

1. Independence assumption. Price and position are sampled independently; there is no
   copula. The background deck shows that ignoring the price/position link can
   understate IaR by about 70% for wind portfolios, which are short exactly when
   prices spike. The engine has a swappable `ScenarioDraw` seam where a copula would
   slot in, but no dependence code exists.
2. No cross-MTU price autocorrelation. Each MTU's price is drawn independently, then
   summed. Real imbalance prices persist across MTUs (system tightness lingers), which
   fattens the period tail. The summed-quantile aggregation is correct given
   independence, but independence across MTUs understates clustered stress (deck
   Challenge 3).
3. Sigma is a tuned knob, not measured. The imbalance forecast-error size has no
   historical error sample behind it. The calibration routine tunes it against the
   backtest scorecard (coarse, with few settled days), but it is not fitted to a real
   forecast-error distribution.
4. Symmetric, homoskedastic error. Imbalance is modelled as Normal or scaled-t around
   the expected value. Real wind errors are skewed, bounded (0 to capacity),
   heteroskedastic (larger when windy or ramping), and autocorrelated. None of that is
   captured.
5. Tails are extrapolated, not fitted. Beyond P01/P99 the price sampler uses a linear
   or clamp rule on 9 quantiles; there is no fat-tail (skew-t or Johnson SU) fit (deck
   Challenge 1).
6. Tier-1 positions only. DAM positions are treated as near-deterministic per MTU;
   there is no per-MTU position distribution (Tier 2) and no temporal position
   autocorrelation from a stochastic generation model (Tier 3).
7. DAM price is deterministic. We assume the intraday case where the spot price is
   known. Day-ahead Spread IaR, where DAM is itself a random variable jointly
   distributed with the imbalance price, is not modelled.
8. One IaR figure per run, for the remaining/next-day period. The engine emits a
   single period IaR, so only the remaining-day limit is actually evaluated; per-MTU
   and rolling-window limits exist in config but have no IaR series to check yet. No
   marginal IaR or diversification-ratio metrics.
9. Scope guardrails (per brief): single price area per portfolio, one-price
   settlement, no intraday or balancing positions, no multi-area netting, no UK
   cash-out.

Data

10. Synthetic and mismatched data. Positions come from the `windsim` simulator and are
    future-dated; realised imbalance prices are past-settled. The backtest overlap is
    therefore small and assembled by hand, so the Kupiec test and sigma calibration
    have very low statistical power. They are readouts, not verdicts.
11. Demo data in the UI. When the internal wheel is absent, the dashboard synthesises
    the realised imbalance price (DAM plus a sampled spread) and replays the live
    forecast curve as each day's day-ahead vintage. It is clearly tagged DEMO, but it
    is not real history.
12. DAM spot via a vendored wheel. Real spot comes from the internal `optipyclient`
    wheel (manual install), not a supported external feed (ENTSO-E or Nord Pool).
    There is no real position-ingestion or metered-actuals feed.
13. Forecast vintages are not stored. Replay uses fetched or synthesised vintages on
    the fly; we do not persist the `ImbalancePriceForecast` vintage history.

Engineering / platform

14. SQLite, single process, no migrations. The schema is created with `create_all`
    (no Alembic), with no concurrency, auth, or multi-tenancy.
15. The shown UI is throwaway. `beginner UI/dashboard.py` reaches into backend modules
    directly. The frozen `service.py` exists, but the real UI (`app/dashboard.py`)
    that should consume it is not built.
16. Batch, not real-time. Everything runs manually via scripts. There is no
    event-driven recompute on a new forecast, position, or MTU passage, no 15-minute
    cadence, and no webhooks or streaming (deck Challenge 4).
17. Calibrated sigma is not persisted. The recommendation is advisory; Apply in the UI
    is session-scoped. There is no model-params table or scheduled recalibration.
18. One basis persists at a time. `HistoricalPerformanceRecord` has no `iar_type`
    column, so a gross backtest and a spread backtest overwrite each other's rows.
19. Thin ops. No structured logging, metrics, or tracing; no CI; secrets in a local
    `.env`; host pinned in code; an auto-commit hook but manual pushes.

## B. What turning this into a real product requires

Quant model

- Implement the copula at the `ScenarioDraw` seam: price/position cross-dependence and
  cross-MTU autocorrelation, calibrated from history (historical, Gaussian, or
  Student-t). This is the biggest accuracy fix.
- Fit fat-tailed, skewed marginals (skew-t or Johnson SU) to the quantiles instead of
  linear extrapolation.
- Tier 2 and 3 positions: per-MTU position distributions with temporal autocorrelation
  from a stochastic generation model; fit sigma from real forecast-vs-actual error
  history (heteroskedastic, skewed, capacity-bounded).
- Stochastic DAM for day-ahead Spread IaR (a joint forecast of imbalance price and
  spot, using Volue DAM quantile forecasts).
- Emit per-MTU, rolling-window, and remaining-session IaR series so all limit types
  are live; add marginal IaR and the diversification ratio.
- Support two-price settlement, multiple areas with cross-area correlation, and more
  markets (DE and EU; UK cash-out as a separate mechanism).

Data and integration

- Real, supported feeds: DAM/spot (ENTSO-E or Nord Pool), metered actuals, and a
  position-ingestion API (push or poll) from portfolio systems, replacing `windsim`
  and the vendored wheel.
- Persist forecast vintages (`ImbalancePriceForecast`) so backtests replay true
  history; accumulate enough settled history for statistically powerful Kupiec,
  Christoffersen, or dynamic-quantile tests.

Platform and engineering

- Postgres with Alembic migrations, concurrency, and multi-tenant access with RBAC and
  auth.
- Build the real `app/dashboard.py` consuming only `service.py`; expose a REST or gRPC
  API plus webhooks for alerts; dashboards per the mockups (IaR-vs-limit curves, an
  MTU/hour heatmap, alerts before breach, a backtest panel).
- Real-time, event-driven recompute (new forecast, position update, or MTU passage) on
  a 15-minute cadence, with a scheduler and streaming.
- Persist calibrated model params plus scheduled recalibration jobs; optionally store
  raw scenarios for audit.
- Performance: quasi-Monte-Carlo or stratified sampling for stable P01/P05 at lower
  cost; profile and scale.
- Ops and observability: structured logging, metrics, tracing, and error tracking;
  CI/CD with the test suite, type-checking, and linting gated; a secrets manager (not
  `.env`); environment and config management; cloud deploy with monitoring and SLAs.

Validation and governance

- Long-window backtests; model-validation and governance documentation; auditable
  limit-breach trails; regulatory reporting; a sign-off process before numbers are
  used commercially.

In short: the plumbing is real and tested, but the numbers are illustrative. The
independence assumption, the parametric and symmetric error model, and the short
backtest window all push IaR optimistically low. Closing those (a copula, fitted
marginals, and real history) is the core of the production build.
