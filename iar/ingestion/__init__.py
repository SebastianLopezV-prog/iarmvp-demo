"""Ingestion layer — brings external data into the SQLite hub.

Modules
-------
- ``synthetic``       : synthetic market model + drop-in forecast/markets clients + wind generator.
- ``clients``         : factory returning the synthetic clients (this demo has no real-feed path).
- ``flatfile_loader`` : CSV/Excel loaders for DAM positions, generation forecasts, actuals.
"""
