"""Tests for the markets client (real DAM price via the internal optipyclient SDK).

The internal SDK is mocked via the ``_api`` injection seam, so these run without the
vendored wheel or network.
"""

from datetime import datetime, timezone

import pytest

from iar.ingestion.markets_client import OptimeeringMarketsClient


class FakeMarketsApi:
    """Stand-in for optipyclient.MarketsApi returning plain dicts (like .to_dict())."""

    def __init__(self, series_items, market_items):
        self._series_items = series_items
        self._market_items = market_items
        self.calls = {}

    def get_market_series(self, **kwargs):
        self.calls["get_market_series"] = kwargs
        return {"items": self._series_items}

    def get_market(self, **kwargs):
        self.calls["get_market"] = kwargs
        return {"items": self._market_items}


def _client(series_items, market_items):
    api = FakeMarketsApi(series_items, market_items)
    return OptimeeringMarketsClient(_api=api), api


SERIES = [
    {"id": 999, "market": "DAM", "series_type": "cleared price", "publisher": "ENTSOE"},
    {"id": 173, "market": "DAM", "series_type": "cleared price", "publisher": "Nordpool"},
]
DATAPOINTS = [
    {"start": datetime(2026, 6, 8, 12, 30, tzinfo=timezone.utc), "value": 80.29},
    {"start": datetime(2026, 6, 8, 12, 45, tzinfo=timezone.utc), "value": 84.89},
]
MARKET = [{"series_id": 173, "datapoints": DATAPOINTS}]


def test_resolve_prefers_nordpool_publisher():
    client, api = _client(SERIES, MARKET)
    client.get_dam_prices("NO2")
    assert api.calls["get_market"]["series_id"] == [173]  # Nordpool, not ENTSOE 999


def test_get_market_series_filters_dam_cleared_price():
    client, api = _client(SERIES, MARKET)
    client.get_dam_prices("NO2")
    kw = api.calls["get_market_series"]
    assert kw["area"] == ["NO2"]
    assert kw["market"] == ["DAM"]
    assert kw["series_type"] == ["cleared price"]


def test_get_dam_prices_normalises_records():
    client, _ = _client(SERIES, MARKET)
    recs = client.get_dam_prices("NO2")
    assert len(recs) == 2
    assert recs[0] == {
        "price_area": "NO2",
        "timestamp": "2026-06-08T12:30:00+00:00",
        "eur_per_mwh": 80.29,
    }
    # sorted by timestamp
    assert [r["timestamp"] for r in recs] == sorted(r["timestamp"] for r in recs)


def test_no_dam_series_raises():
    client, _ = _client([], MARKET)
    with pytest.raises(LookupError, match="cleared price"):
        client.get_dam_prices("NO2")
