"""Offline tests for the Optimeering client (Task 1.3).

These use a stubbed SDK layer so the suite stays deterministic and needs no
network or API key. A separate live check (see scripts/probe) exercises the
real API.
"""

from datetime import datetime, timezone

import pytest

from optimeering import ApiException
from iar.ingestion.optimeering_client import (
    MissingApiKeyError,
    OptimeeringForecastClient,
    _is_transient,
)

META = {
    "id": 2036,
    "area": "NO2",
    "statistic": "Quantile",
    "unit_type": "Price_Spread",
    "resolution": "PT15M",
}

# A retrieve_latest-shaped response (dict form, as .to_dict() would give).
SAMPLE_DATA = {
    "items": [
        {
            "events": [
                {
                    "event_time": datetime(2026, 6, 3, 10, 23, tzinfo=timezone.utc),
                    "predictions": [
                        {
                            "prediction_for": datetime(2026, 6, 3, 11, 0, tzinfo=timezone.utc),
                            "value": {"10": -7.85, "50": -1.29, "90": 7.63},
                        }
                    ],
                }
            ]
        }
    ]
}


def make_client(tmp_path):
    # load_env=False + explicit dummy key => no .env, no network on construction.
    return OptimeeringForecastClient(api_key="dummy", cache_dir=tmp_path, load_env=False)


def test_missing_key_raises():
    with pytest.raises(MissingApiKeyError):
        OptimeeringForecastClient(api_key=None, load_env=False)


def test_normalise_expands_quantiles(tmp_path):
    client = make_client(tmp_path)
    records = client._normalise(SAMPLE_DATA, META)
    assert len(records) == 3  # one row per quantile
    r = records[0]
    assert r["price_area"] == "NO2"
    assert r["statistic_type"] == "Quantile"
    assert r["timestamp"] == "2026-06-03T11:00:00+00:00"
    assert r["vintage_ts"] == "2026-06-03T10:23:00+00:00"
    assert {rec["quantile"] for rec in records} == {10.0, 50.0, 90.0}


def test_normalise_scalar_value(tmp_path):
    client = make_client(tmp_path)
    point = {
        "items": [
            {"events": [{"event_time": None, "predictions": [
                {"prediction_for": datetime(2026, 6, 3, 11, tzinfo=timezone.utc), "value": 42.0}
            ]}]}
        ]
    }
    records = client._normalise(point, {**META, "statistic": "Point"})
    assert len(records) == 1
    assert records[0]["quantile"] is None
    assert records[0]["value"] == 42.0


def test_forecast_uses_cache(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    monkeypatch.setattr(client, "_resolve_series", lambda *a, **k: META)

    calls = {"n": 0}

    def fake_retrieve_latest(series_id, max_event_time):
        calls["n"] += 1
        return SAMPLE_DATA

    monkeypatch.setattr(client, "_call_retrieve_latest", fake_retrieve_latest)

    first = client.get_imbalance_price_forecast("NO2", use_cache=True)
    second = client.get_imbalance_price_forecast("NO2", use_cache=True)
    assert first == second
    assert calls["n"] == 1  # second call served from disk cache


def test_is_transient_classification():
    e503 = ApiException(status=503)
    e400 = ApiException(status=400)
    assert _is_transient(e503) is True
    assert _is_transient(e400) is False
    assert _is_transient(TimeoutError()) is True
    assert _is_transient(ValueError("nope")) is False
