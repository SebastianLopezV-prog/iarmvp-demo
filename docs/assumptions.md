# Assumptions & limitations

Living log of every modelling simplification, kept honest for the demo. The MVP is a
PoC: several of these are deliberate scope decisions, not oversights.

## Deliberate MVP assumptions

1. **Independence.** Imbalance price and portfolio imbalance position are sampled
   independently. Real markets show price↔position dependence (a wind portfolio is short
   precisely when prices spike), which can understate true IaR by ~70% (deck, Challenge 2).
   This is sanctioned MVP scope and is surfaced via the backtest panel.
2. **Single-price settlement.** One imbalance price per MTU (Nordic/German convention).
3. **DAM positions only.** No intraday or balancing-market positions; single price area
   per portfolio.
4. **Parametric imbalance-uncertainty model.** With no historical forecast-error data yet,
   uncertainty on (DAM - forecast gen) is modelled parametrically (Gaussian/scaled-t, sigma
   a configurable fraction of forecast/capacity). Recalibrated against realised exceedances
   in the Week-3 backtest.

## Preserved from the full IaR vision

- Heavy tails on the **price** side: sampled from Optimeering's actual quantiles via
  inverse-CDF, not refitted to a Normal.
- **Period IaR = quantile of summed P&L** across MTUs (not the sum of per-MTU IaRs).
- Gross + Spread IaR and CIaR / Expected Shortfall are all produced.

## Open design points from the live API probe (2026-06-03, NO2)

These came out of a read-only probe of the Optimeering forecast for NO2 and need a
team decision before/while building the engine. See `data_contract.md` for detail.

1. **Quantile granularity differs by resolution (corrected 2026-06-03 after the
   client live-test).** The **PT15M** Imbalance series — the one the MVP uses — provides the
   **full 9 quantiles incl. P05/P95** (`1,5,10,25,50,75,90,95,99`), so a P95 IaR can be read
   directly; only the extreme tails beyond P01/P99 need extrapolation. (The earlier "only 5
   quantiles" note came from the coarser **PT1H** series, which does lack P05/P95.) Tail
   handling beyond P01/P99 in the price sampler (Task 2.2) remains a minor Week-2 decision.
2. **Imbalance is published as `Price_Spread` (EUR), not an absolute price** (for NO2 the
   `Imbalance` series are `Price_Spread` and `Direction`). This maps directly to
   **Spread IaR**. **Gross IaR** needs the *absolute* imbalance price = DAM price + spread,
   so it requires a DAM price input. Confirm with the team how Gross IaR is sourced.

## Out of MVP scope (roadmap toward "true IaR")

- Price↔position dependence (copula) and cross-MTU autocorrelation.
- Fat-tail modelling beyond the supplied quantiles.
- Multi-area netting, intraday/balancing positions, live trading integration,
  regulatory reporting, cloud deployment.
