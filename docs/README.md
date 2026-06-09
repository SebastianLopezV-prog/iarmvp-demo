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

## What you need to add (NOT in the repo)

Cloning gives you the code, but four things live **outside git** — you must add them
yourself. The tool **degrades gracefully**: with just 1–2 you get live Spread IaR (Gross +
portfolio stubbed); add 3 and 4 to make everything real except the imbalance `sigma`.

| # | What to add | Required for | If missing |
|---|-------------|--------------|------------|
| 1 | **Python 3.13 + venv**, then `pip install -e ".[dev]"` | everything | nothing runs |
| 2 | **`.env`** file with `OPTIMEERING_API_KEY=<key>` | any live data (spread, DAM) | live fetches fail |
| 3 | **`optipyclient` wheel** in `vendor/` (from `Volue/sirius-prime` Releases) | real DAM spot → real **Gross IaR** | flat `--dam-price` stub (Spread IaR fine) |
| 4 | **`windsim`** (`pip install git+…/sirius-imb-at-risk-mvp`) | real portfolio positions/generation/actuals | synthetic stub portfolio |

Items 3–4 are **private Volue packages** (not on PyPI) and require **Volue GitHub org
access** (the org enforces SAML SSO — authorize your git credential once). Exact install
commands are in **Setup** below.

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
app/           Streamlit dashboard (UI only — skeleton for service.py)
beginner UI/   dashboard.py — demo risk dashboard (committed; needs the deps above)
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

### Optional — real portfolio data via `windsim`

Real positions/generation/actuals come from Volue's wind-portfolio simulator,
**`windsim`** (in `Volue/sirius-imb-at-risk-mvp`, also private — not on PyPI). Install it
from the repo, then ingest a portfolio **the client way** (it generates the data, writes
the three upload CSVs, and loads them through the flat-file loaders into `data/iar.db`):

```powershell
.\venv\Scripts\python.exe -m pip install "git+https://github.com/Volue/sirius-imb-at-risk-mvp.git"
.\venv\Scripts\python.exe scripts\load_windsim_data.py --windsim-portfolio north --area NO2
```

`run_iar.py` then uses these REAL positions automatically (it reads them from the DB and
simulates over the MTUs where the live spread, real DAM price, and real positions overlap).
Without it, `run_iar.py` falls back to a synthetic stub portfolio.

## Run

```powershell
.\venv\Scripts\python.exe scripts\load_windsim_data.py  # ingest REAL portfolio data (client CSV path)
.\venv\Scripts\python.exe scripts\run_iar.py            # Monte Carlo -> Gross/Spread IaR + CIaR
.\venv\Scripts\python.exe scripts\run_iar.py --store    # ...and persist the run to data/iar.db
.\venv\Scripts\python.exe scripts\validate_engine.py    # engine validation report (8/8 checks)
.\venv\Scripts\python.exe scripts\verify_all.py         # end-to-end health check (1.1 -> 2.x)
.\venv\Scripts\python.exe scripts\run_pipeline.py       # ingestion smoke path
```

Engine sign convention: `cost = imbalance × price` (positive = cost/bad). IaR is the
upper-tail quantile of the summed cost; CIaR is the mean beyond it.

## Dashboard (demo UI)

A simple Streamlit dashboard at **`beginner UI/dashboard.py`** visualizes one run:
headline Gross/Spread IaR + CIaR (shown as **P&L: negative = loss**), the scenario P&L
distribution with mean/IaR/CIaR markers, the live spread fan chart, and a **data-source
panel** (which inputs are real vs stub).

```powershell
.\venv\Scripts\python.exe -m streamlit run "beginner UI\dashboard.py"   # then open http://localhost:8501
```

It uses the **same real inputs as `run_iar.py`** (live spread; real DAM if the wheel is
installed; real positions if a portfolio is loaded), falling back to stubs otherwise. Drag
the **sigma** slider to see how forecast-error size drives the risk. (It reuses the backend
modules only — no logic lives in the UI.)

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

**Portfolio data — now REAL too.** Positions/generation/actuals can come from Volue's
`windsim` simulator, ingested through the **client CSV path** (`load_windsim_data.py` →
`data/uploads/*.csv` → flat-file loaders → `dam_prices`/positions tables). `run_iar.py`
picks them up from the DB automatically. So spread, DAM spot, **and** positions are now
real; the **only remaining stub is the imbalance-model `sigma`** (the forecast-error size),
which is calibrated against realised outcomes in Week 3.

**Next — Week 3 (`iar/risk/`):** backtesting (join settled cost to the IaR estimate whose
`vintage_ts` precedes it; Kupiec POF test), sigma recalibration, and alerts.

## Repo

Primary: `Volue/iarmvp` (GitHub). A local PostToolUse hook auto-commits after each
edit; pushes are manual.

---

# From MVP to production

This is a **proof of concept**, built fast to prove the pipeline end-to-end. It is
deliberately *not* production software. This section is the honest ledger: every
shortcut we took, and everything that would have to change to turn it into a real
product. Read it before quoting any IaR number to a customer.

## A. Shortcuts & simplifications we took (and why)

**Modelling / quant**
1. **Independence assumption (the big one).** Price and position are sampled
   **independently** — there is no copula. The background deck shows that ignoring
   the price↔position link can **understate IaR by ~70%** for wind portfolios (they
   are short exactly when prices spike). The engine has a swappable `ScenarioDraw`
   seam where a copula would slot in, but **no dependence code exists**.
2. **No cross-MTU price autocorrelation.** Each MTU's price is drawn independently,
   then summed. Real imbalance prices persist across MTUs (system tightness lingers),
   which fattens the *period* tail. Our summed-quantile is correct *given*
   independence, but independence across MTUs understates clustered stress (deck
   Challenge 3).
3. **Sigma is a tuned knob, not measured.** The imbalance forecast-error size has no
   historical error sample behind it. "Option 1" now *calibrates* it against the
   backtest scorecard (coarse, few settled days), but it is **not** fitted to a real
   forecast-error distribution ("Option 2").
4. **Symmetric, homoskedastic error.** Imbalance is modelled Normal/scaled-t around
   the expected value. Real wind errors are **skewed, bounded** (0…capacity),
   **heteroskedastic** (bigger when windy/ramping) and **autocorrelated** — none of
   that is captured.
5. **Tails are extrapolated, not fitted.** Beyond P01/P99 the price sampler uses a
   linear/clamp rule on 9 quantiles — no fat-tail (skew-t / Johnson SU) fit
   (deck Challenge 1).
6. **Tier-1 positions only.** DAM positions are treated as (near-)deterministic per
   MTU; no per-MTU position distribution (Tier 2) and no temporal position
   autocorrelation from a stochastic generation model (Tier 3).
7. **DAM price is deterministic.** We assume the intraday case where the spot price
   is known. Day-ahead Spread IaR (where DAM is itself a random variable, jointly
   distributed with the imbalance price) is not modelled.
8. **One IaR figure per run = remaining/next-day period.** The engine emits a single
   period IaR, so only the **remaining-day** limit is actually evaluated; per-MTU and
   rolling-window limits exist in config but have no IaR series to check yet. No
   **marginal IaR** or **diversification-ratio** product metrics.
9. **Scope guardrails** (per brief): single price area per portfolio, one-price
   settlement, no intraday/balancing positions, no multi-area netting, no UK cash-out.

**Data**
10. **Synthetic / mismatched data.** Positions come from the `windsim` simulator and
    are **future-dated**; realised imbalance prices are **past-settled**. Overlap for
    the backtest is therefore small and assembled by hand, so the **Kupiec test and
    sigma calibration have very low statistical power** — they are *readouts*, not
    verdicts.
11. **Demo data in the UI.** When the internal wheel is absent, the dashboard
    **synthesises** the realised imbalance price (DAM + a sampled spread) and replays
    the *live* forecast curve as each day's "day-ahead" vintage — clearly tagged DEMO,
    but not real history.
12. **DAM spot via a vendored wheel.** Real spot comes from the internal
    `optipyclient` wheel (manual install), not a supported external feed
    (ENTSO-E / Nord Pool). No real position-ingestion or metered-actuals feed exists.
13. **Forecast vintages aren't stored.** Replay synthesises/uses fetched vintages on
    the fly; we don't persist the `ImbalancePriceForecast` vintage history.

**Engineering / platform**
14. **SQLite, single-process, no migrations.** Schema is created with `create_all`
    (no Alembic), no concurrency, no auth, no multi-tenancy.
15. **The shown UI is throwaway.** `beginner UI/dashboard.py` reaches into backend
    modules directly. The frozen `service.py` exists, but the *real* UI
    (`app/dashboard.py`) that should consume it isn't built.
16. **Batch, not real-time.** Everything is run manually via scripts. There is no
    event-driven recompute on new forecast / position / MTU passage, no 15-minute
    cadence, no webhooks/streaming (deck Challenge 4).
17. **Calibrated sigma isn't persisted.** The recommendation is advisory; "Apply" in
    the UI is session-scoped. There is no model-params table or scheduled recalibration.
18. **One basis persists at a time.** `HistoricalPerformanceRecord` has no `iar_type`
    column, so a gross backtest and a spread backtest overwrite each other's rows.
19. **Thin ops.** No structured logging/metrics/tracing, no CI, secrets in a local
    `.env`, host pinned in code, auto-commit hook but manual pushes.

## B. What turning this into a real product requires

**Quant model**
- Implement the **copula** at the `ScenarioDraw` seam: price↔position cross-dependence
  **and** cross-MTU autocorrelation, calibrated from history (historical/Gaussian/
  Student-t). This is the single biggest accuracy fix.
- **Fit fat-tailed, skewed marginals** (skew-t / Johnson SU) to the quantiles instead
  of linear extrapolation.
- **Tier 2/3 positions:** per-MTU position distributions with temporal autocorrelation
  from a stochastic generation model; **fit sigma from real forecast-vs-actual error
  history** (heteroskedastic, skewed, capacity-bounded).
- **Stochastic DAM** for day-ahead Spread IaR (joint `(imbalance, spot)` forecast,
  using Volue DAM quantile forecasts).
- Emit **per-MTU / rolling-window / remaining-session IaR series** so all limit types
  are live; add **marginal IaR** and the **diversification ratio**.
- Support **two-price settlement**, **multiple areas + cross-area correlation**, more
  markets (DE/EU; UK cash-out as a separate mechanism).

**Data & integration**
- Real, supported feeds: **DAM/spot** (ENTSO-E / Nord Pool), **metered actuals**, and a
  **position-ingestion API** (push or poll) from portfolio systems — replacing
  `windsim` and the vendored wheel.
- **Persist forecast vintages** (`ImbalancePriceForecast`) so backtests replay true
  history; accumulate enough settled history for **statistically powerful** Kupiec /
  Christoffersen / dynamic-quantile tests.

**Platform & engineering**
- **Postgres + Alembic migrations**, concurrency, **multi-tenant + RBAC/auth**.
- Build the **real `app/dashboard.py`** consuming **only `service.py`**; expose a
  **REST/gRPC API + webhooks** for alerts; dashboards per the mockups (IaR-vs-limit
  curves, MTU/hour heatmap, alerts-before-breach, backtest panel).
- **Real-time, event-driven recompute** (new forecast / position update / MTU passage)
  on a 15-minute cadence; scheduler + streaming.
- **Persist calibrated model params** + scheduled recalibration jobs; optionally store
  raw scenarios for audit.
- **Performance:** quasi-Monte-Carlo / stratified sampling for stable P01/P05 at lower
  cost; profile and scale.
- **Ops/observability:** structured logging, metrics, tracing, error tracking; **CI/CD**
  with the test suite, type-checking and linting gated; **secrets manager** (not
  `.env`); environment/config management; cloud deploy + monitoring + SLAs.

**Validation & governance**
- Long-window backtests; model-validation and governance documentation; **auditable
  limit-breach trails**; regulatory reporting; sign-off process before numbers are
  used commercially.

> One-line takeaway: the **plumbing is real and tested**, but the **numbers are
> illustrative**. The independence assumption, the parametric/symmetric error model,
> and the short backtest window all push IaR **optimistically low** — closing those
> (copula + fitted marginals + real history) is the heart of the production build.
