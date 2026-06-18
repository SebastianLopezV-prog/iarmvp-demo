"""IaR MVP — Imbalance at Risk proof-of-concept package.

A local PoC that estimates worst-case imbalance settlement cost for a wind
portfolio via Monte Carlo simulation, under the MVP independence assumption
(price and imbalance position sampled independently).

Subpackages
-----------
- ``iar.db``         : SQLAlchemy models + session/engine management (the integration hub).
- ``iar.ingestion``  : synthetic market feeds (clients factory) and flat-file loaders.
- ``iar.simulation`` : imbalance-uncertainty model, price sampler, Monte Carlo engine.
- ``iar.risk``       : limits/alerts and backtesting (Kupiec POF).
- ``iar.service``    : the thin backend interface the Streamlit UI calls.

See ``docs/`` and the architecture design for the full blueprint.
"""

from iar.__about__ import __version__
