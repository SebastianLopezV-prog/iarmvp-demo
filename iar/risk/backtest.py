"""Backtesting (Tasks 3.2-3.3).

Task 3.1 (realised imbalance cost) is implemented in
:mod:`iar.risk.realised_cost`.

This module (Tasks 3.2-3.3, not yet built) will: backfill historical IaR
estimates so each settled period has an "as-of" vintage, join each period's
realised cost (from :mod:`iar.risk.realised_cost`) to the IaR estimate whose
``vintage_ts`` precedes it, compute exceedance frequency (~5% at P95), run a
Kupiec POF (chi-square) calibration test, and write
``HistoricalPerformanceRecord`` rows.
"""
