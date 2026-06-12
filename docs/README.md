# IaR Demo — methodology and limitations

For what this is and how to run it, see the **[root README](../README.md)**. This document
covers the methodology, the deliberate simplifications, and what a production build would
require. Read it before quoting any IaR number.

This build is a synthetic demo: it runs on 100% fake, market-like data (no API key, no
private SDKs, no external feeds). The pipeline is real and tested; the numbers are
illustrative.

## What the engine computes

- **Gross IaR** — worst-case total settlement cost: position × imbalance price.
- **Spread IaR** — worst-case underperformance vs day-ahead: position × (imbalance price − DAM price).
- **CIaR / Expected Shortfall** — the average loss in the tail beyond IaR.
- **Period IaR** is the quantile of the *summed* P&L across all MTUs in the horizon, not the
  sum of per-MTU IaRs (summing per-MTU risks would assume every interval has its worst
  outcome simultaneously — the wrong probability level). The **diversification ratio** reports
  how much risk that joint view removes versus the naive sum.

Engine sign convention: `cost = imbalance × price` (positive = cost, i.e. bad). IaR is the
upper-tail quantile of the summed cost; CIaR is the mean beyond it.

Backtesting stamps each estimate with an as-of vintage, joins each settled period's realised
cost to the IaR that would have been quoted, and checks the exceedance rate against the ~5%
target with a Kupiec POF test.

## Synthetic data (this demo)

All feeds come from `iar/ingestion/synthetic.py`, deterministic and rolling forward in time:

- imbalance-price spread forecast — per-MTU P01-P99 quantiles, heavy-tailed and skewed;
- day-ahead (spot) price — diurnal curve with daily variation;
- realised imbalance price — DAM plus a drawn spread (so the backtest calibrates);
- wind portfolio — replays real windsim daily capacity-factor shapes
  (`windsim_profiles.json`), scaled and rolled forward, so positions/generation/actuals look
  like genuine wind data with no windsim install.

## Deliberate simplifications

Modelling:

1. **Independence.** Price and position, and prices across MTUs, are sampled independently;
   there is no copula. Ignoring the price/position link can understate IaR materially for
   wind (short exactly when prices spike). The engine has a swappable draw seam where a
   copula would slot in.
2. **No cross-MTU price autocorrelation.** Real imbalance prices persist across MTUs, which
   fattens the period tail; independence understates clustered stress.
3. **Sigma is tuned, not measured.** The imbalance forecast-error size is a parametric knob,
   not fitted to a real forecast-error distribution.
4. **Symmetric, homoskedastic error.** Real wind errors are skewed, capacity-bounded,
   heteroskedastic, and autocorrelated.
5. **Tails extrapolated, not fitted** beyond P01/P99 (no skew-t / Johnson SU fit).
6. **Tier-1 positions** only (deterministic per MTU; no per-MTU position distribution).
7. **Deterministic DAM price** (intraday case); day-ahead Spread IaR with a stochastic DAM
   is not modelled.
8. **One period IaR per run.** Per-MTU and rolling-window limits exist in config; the engine
   also reports per-MTU and rolling figures for the dashboard.

Scope guardrails: single price area per portfolio, one-price settlement, no intraday or
balancing positions, no multi-area netting, no UK cash-out.

## What a production build would require

- A copula at the draw seam: price/position cross-dependence and cross-MTU autocorrelation,
  calibrated from history (the biggest accuracy fix).
- Fitted fat-tailed, skewed marginals instead of linear extrapolation.
- Tier 2/3 positions and sigma fitted from real forecast-vs-actual history.
- Stochastic DAM for day-ahead Spread IaR.
- Real, supported feeds and a position-ingestion API (push or poll) from portfolio systems.
- Persisted forecast vintages and enough settled history for statistically powerful tests.
- Production platform: Postgres + migrations, auth/multi-tenancy, REST/webhooks, real-time
  event-driven recompute on a 15-minute cadence, observability, and CI/CD.
- Long-window backtests, model-validation/governance docs, and commercial sign-off before
  numbers are used with customers.

In short: the plumbing is real and tested, but the numbers are illustrative. The independence
assumption, the parametric symmetric error model, and (in the real product) a short backtest
window all push IaR optimistically low. Closing those — a copula, fitted marginals, and real
history — is the core of the production build.
