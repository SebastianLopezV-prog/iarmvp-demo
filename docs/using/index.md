# How to use

The demo is a Streamlit dashboard, the "Command Centre". Launch it with
`streamlit run app/dashboard.py` (or open the hosted link); on first load it self-seeds its
synthetic database, then it is fast and ticks forward while open.

## The tabs

- **Command Centre** - period Gross/Spread IaR vs limits with status, peak-MTU IaR, an
  intraday per-MTU chart, and two heatmaps (forecast worst-case IaR; realised settled cost),
  plus a limit table and alert feed.
- **Risk Analytics** - risk-profile KPIs (IaR, Expected Shortfall/CIaR with tail ratio,
  peak-MTU IaR, diversification ratio), the IaR-over-time curve vs limit, a Gross-vs-Spread
  comparison, and a per-MTU risk-concentration curve.
- **Historical** - the backtest: realised cost vs the day-ahead IaR estimate, exceedances vs
  the ~5% target, and the Kupiec calibration verdict.
- **Usage** - a short methodology walkthrough.
- **Settings** - basis (Gross/Spread), confidence level, and editable euro risk limits.

## Reading the numbers

- **Gross IaR** - worst-case total settlement cost (position x imbalance price).
- **Spread IaR** - worst-case cost vs the day-ahead position (position x (imbalance - DAM price)).
- **CIaR / Expected Shortfall** - average loss in the tail beyond IaR.
- Period IaR is the quantile of the summed P&L across MTUs, not the sum of per-MTU IaRs.

The figures are illustrative (synthetic data); see [methodology](../README.md).
