"""Backend service interface (Task 3.5) — the only thing the UI talks to.

Placeholder skeleton (Task 1.1). Will expose a small set of read functions over
SQLAlchemy queries, returning tidy DataFrames/dicts and hiding all DB detail:

- ``get_latest_iar``
- ``get_iar_curve``
- ``get_alerts``
- ``get_backtest_summary``

The Streamlit dashboard imports only from here — it contains no simulation or
DB logic itself.
"""
