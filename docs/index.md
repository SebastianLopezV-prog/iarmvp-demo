# Imbalance at Risk (IaR) - synthetic demo

A self-contained demo of an Imbalance at Risk tool for a wind portfolio: it estimates the
worst-case imbalance settlement cost at a chosen confidence level over a forward horizon, via
Monte Carlo simulation (the Value-at-Risk analogue for a portfolio's imbalance position).

This build runs on **100% synthetic, market-like data** - no API key, no private SDKs, no
external feeds. The pipeline is real and tested; the euro figures are illustrative.

See also the top-level [README](../README.md) for quick-start and the
[methodology / limitations](README.md) note.

## Documentation map (DIVIO)

- [How to use](using/index.md) - the dashboard and what each tab shows.
- [Technical overview](tech-overview/index.md) - architecture, stack, the synthetic feeds.
- [Developer getting started](development/index.md) - uv, pre-commit, tests.
- [Operational tasks](operations/index.md) - running, reseeding, deployment.

## License
Proprietary Volue license - see [LICENSE.md](../LICENSE.md).
