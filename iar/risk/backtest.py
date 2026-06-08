"""Backtesting (Tasks 3.1-3.3).

Placeholder skeleton (Task 1.1). Will compute realised imbalance cost
(actual imbalance x actual price), join each settled period to the IaR estimate
whose ``vintage_ts`` precedes it, compute exceedance frequency (~5% at P95),
and run a Kupiec POF (chi-square) calibration test. Writes
HistoricalPerformanceRecord rows.
"""
