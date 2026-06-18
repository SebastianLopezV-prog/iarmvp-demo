"""Ingestion client factory for the demo: SYNTHETIC FEEDS ONLY.

This is a synthetic demo. There is **no real-feed code path and no API key** - the factory
always returns the synthetic market clients from :mod:`iar.ingestion.synthetic`. Pipeline
scripts call ``get_forecast_client()`` / ``get_markets_client()`` so the (synthetic) data
source is defined in one place.
"""

from __future__ import annotations

from typing import Any


def get_forecast_client(**kwargs: Any):
    """Imbalance-spread forecast client (synthetic)."""
    from iar.ingestion.synthetic import SyntheticForecastClient

    return SyntheticForecastClient(**kwargs)


def get_markets_client(**kwargs: Any):
    """Markets client for DAM spot + realised imbalance price (synthetic)."""
    from iar.ingestion.synthetic import SyntheticMarketsClient

    return SyntheticMarketsClient(**kwargs)
