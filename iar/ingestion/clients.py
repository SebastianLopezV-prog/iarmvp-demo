"""Ingestion client factory: synthetic feeds by default (demo build).

The demo runs with NO real data, NO API key and NO extra downloads, so by default
this returns the synthetic market clients from :mod:`iar.ingestion.synthetic`. Set
the environment variable ``IAR_SYNTHETIC=0`` (or ``false``) to fall back to the real
Optimeering clients instead (requires the API key and, for markets, the vendored
``optipyclient`` wheel).

Pipeline scripts (``run_iar.py``, ``load_actuals.py``, ``backfill_history.py``) call
``get_forecast_client()`` / ``get_markets_client()`` rather than constructing a client
directly, so the data source is swappable in one place.
"""

from __future__ import annotations

import os
from typing import Any


def use_synthetic() -> bool:
    """True unless ``IAR_SYNTHETIC`` is explicitly set to a falsey value.

    Demo default is synthetic. ``IAR_SYNTHETIC`` in {0, false, no, off} opts back into
    the real Optimeering feeds.
    """
    val = os.getenv("IAR_SYNTHETIC")
    if val is None:
        return True
    return val.strip().lower() not in ("0", "false", "no", "off", "")


def get_forecast_client(**kwargs: Any):
    """Imbalance-spread forecast client (synthetic by default)."""
    if use_synthetic():
        from iar.ingestion.synthetic import SyntheticForecastClient
        return SyntheticForecastClient(**kwargs)
    from iar.ingestion.optimeering_client import OptimeeringForecastClient
    return OptimeeringForecastClient(**kwargs)


def get_markets_client(**kwargs: Any):
    """Markets client for DAM spot + realised imbalance price (synthetic by default)."""
    if use_synthetic():
        from iar.ingestion.synthetic import SyntheticMarketsClient
        return SyntheticMarketsClient(**kwargs)
    from iar.ingestion.markets_client import OptimeeringMarketsClient
    return OptimeeringMarketsClient(**kwargs)
