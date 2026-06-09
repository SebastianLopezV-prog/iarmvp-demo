"""Optimeering **markets** client — real day-ahead (spot) price via the INTERNAL SDK.

The public ``optimeering`` SDK only forecasts balancing markets and publishes the
imbalance as a *spread* (vs spot). The **internal** ``optipyclient`` SDK additionally
serves settled MARKET data through ``MarketsApi`` — including the day-ahead market
(``DAM``) **cleared price**, which IS the NO2 spot price we need for **Gross IaR**.

This replaces the old synthetic ``dam_price`` stub with the real source.

.. important::

    ``optipyclient`` is a **VENDORED wheel — NOT on PyPI.** It must be downloaded from the
    ``Volue/sirius-prime`` GitHub *Releases* page into ``vendor/`` and installed manually.
    See ``docs/README.md`` → **Setup**. The import below fails with a clear pointer if it
    isn't installed. Auth reuses ``OPTIMEERING_API_KEY``; the host is pinned to production.

Returns normalised records ``{"price_area", "timestamp", "eur_per_mwh"}`` — ready for
``flatfile_loader.store_dam_price_records(...)`` (the ``dam_prices`` table).
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

from iar.ingestion.optimeering_client import (
    DEFAULT_HOST,
    PROJECT_ROOT,
    MissingApiKeyError,
)

# Market + series-type names as published by Optimeering (confirmed via MarketsApi).
DAM_MARKET = "DAM"
CLEARED_PRICE = "cleared price"
PREFERRED_PUBLISHER = "Nordpool"

# Realised (settled) absolute imbalance price — the backtest input (Task 3.1).
IMBALANCE_MARKET = "Imbalance"
IMBALANCE_PRICE = "imbalance price"


class OptimeeringMarketsClient:
    """Thin wrapper over ``optipyclient.MarketsApi`` for the DAM cleared (spot) price."""

    def __init__(
        self,
        api_key: str | None = None,
        host: str | None = None,
        load_env: bool = True,
        _api: Any | None = None,  # injection seam for tests
    ) -> None:
        if _api is not None:
            self._api = _api
            self.host = host or DEFAULT_HOST
            return

        if load_env:
            load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
        key = api_key or os.getenv("OPTIMEERING_API_KEY")
        if not key:
            raise MissingApiKeyError(
                "OPTIMEERING_API_KEY not found. Add it to .env (never commit it)."
            )
        self.host = host or os.getenv("OPTIMEERING_HOST") or DEFAULT_HOST

        try:
            from optipyclient import Configuration, MarketsApi, OptimeeringClient
        except ImportError as exc:  # pragma: no cover - depends on local install
            raise ImportError(
                "optipyclient (the INTERNAL Optimeering SDK) is not installed. It is a "
                "vendored wheel from the Volue/sirius-prime Releases page — see "
                "docs/README.md 'Setup' for the install step. "
                f"(original error: {exc})"
            ) from exc

        self._api = MarketsApi(
            OptimeeringClient(configuration=Configuration(host=self.host, api_key=key))
        )

    # ------------------------------------------------------------------ #
    def _resolve_dam_series_id(self, area: str) -> int:
        """Find the DAM cleared-price series id for ``area`` (prefer Nordpool)."""
        resp = self._api.get_market_series(
            area=[area], market=[DAM_MARKET], series_type=[CLEARED_PRICE]
        )
        items = _items(resp)
        if not items:
            raise LookupError(
                f"No '{DAM_MARKET}' '{CLEARED_PRICE}' series found for area {area!r}."
            )
        preferred = [
            it for it in items
            if str(_get(it, "publisher", "")).lower() == PREFERRED_PUBLISHER.lower()
        ]
        chosen = (preferred or items)[0]
        return int(_get(chosen, "id"))

    def get_dam_prices(
        self, area: str, start: Any = "-P1D", end: Any = "P2D"
    ) -> list[dict[str, Any]]:
        """Return the DAM cleared (spot) price for ``area`` as normalised records.

        ``start``/``end`` accept ISO 8601 datetimes or durations (e.g. ``-P1D``, ``P2D``).
        Each record: ``{"price_area", "timestamp" (ISO 8601 UTC), "eur_per_mwh"}``.
        """
        sid = self._resolve_dam_series_id(area)
        data = self._api.get_market(series_id=[sid], start=start, end=end)
        records: list[dict[str, Any]] = []
        for series in _items(data):
            for dp in _get(series, "datapoints", []) or []:
                ts = _iso(_get(dp, "start"))
                val = _get(dp, "value")
                if ts is None or val is None:
                    continue
                records.append(
                    {"price_area": area, "timestamp": ts, "eur_per_mwh": float(val)}
                )
        records.sort(key=lambda r: r["timestamp"])
        return records


# --------------------------------------------------------------------------- #
# Tolerant helpers (handle SDK objects or plain dicts)
# --------------------------------------------------------------------------- #
def _items(resp: Any) -> list:
    d = resp.to_dict() if hasattr(resp, "to_dict") else resp
    if isinstance(d, dict):
        return d.get("items", []) or []
    return d or []


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
