"""Optimeering API client (Task 1.3).

A thin, testable wrapper around the official ``optimeering`` SDK that the rest of
the PoC uses to fetch imbalance price data. It hides the two-step SDK pattern
(``list_series`` -> ``retrieve*``) and returns **normalised plain-dict records**
so no other module depends on SDK objects.

Responsibilities
----------------
- **Auth**: load ``OPTIMEERING_API_KEY`` from ``.env`` (never logged/printed).
- **Two fetch methods**:
    * ``get_imbalance_price_forecast(area, ...)`` — forward (live) forecast.
    * ``get_historical_prices(area, ..., start, end)`` — an immutable past window
      (used for backfill / historical actuals in Week 3).
- **Retry/backoff**: only on *transient* failures (timeouts, 429/5xx); auth and
  bad-request errors fail fast.
- **Caching**: on-disk JSON cache under ``data/cache/``. Immutable results
  (historical windows, the series catalogue) are cached; the live forecast is
  not cached by default.

Normalised record (one per area/timestamp/quantile), JSON/DB friendly::

    {
        "price_area":     "NO2",
        "timestamp":      "2026-06-03T11:00:00+00:00",   # prediction_for (ISO 8601, UTC)
        "vintage_ts":     "2026-06-03T10:23:47+00:00",   # forecast event_time
        "statistic_type": "Quantile",
        "quantile":       10.0,        # None for non-quantile statistics
        "unit_type":      "Price_Spread",
        "resolution":     "PT15M",
        "value":          -7.85,       # EUR
    }

These keys map 1:1 onto ``iar.db.models.ImbalancePriceForecast``. Timestamps are
kept as ISO 8601 strings here (a transport format); the DB-insert step parses
them to timezone-aware datetimes.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from optimeering import (
    ApiException,
    Configuration,
    OptimeeringClient,
    PredictionsApi,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# Optimeering PRODUCTION host. The public ``optimeering`` SDK already defaults to this,
# but we pin it explicitly: the *internal* SDK defaults to **staging**, which Volue-group
# accounts cannot reach. Pinning prevents a silent switch to an inaccessible host.
# Override with the OPTIMEERING_HOST env var (or the ``host`` arg) if ever needed.
DEFAULT_HOST = "https://app.optimeering.com"

# Status codes worth retrying (transient server/throttling errors).
_TRANSIENT_STATUS = {429, 500, 502, 503, 504}


def _is_transient(exc: BaseException) -> bool:
    """True for errors worth retrying; auth/bad-request (4xx) fail fast."""
    if isinstance(exc, ApiException):
        return getattr(exc, "status", None) in _TRANSIENT_STATUS
    return isinstance(exc, (ConnectionError, TimeoutError))


# Shared retry policy: a few attempts with exponential backoff, transient-only.
_retry = retry(
    retry=retry_if_exception(_is_transient),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    reraise=True,
)


class MissingApiKeyError(RuntimeError):
    """Raised when OPTIMEERING_API_KEY is not available in the environment."""


class OptimeeringForecastClient:
    """Wrapper over Optimeering's PredictionsApi for imbalance price data."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        load_env: bool = True,
    ) -> None:
        if load_env:
            load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
        key = api_key or os.getenv("OPTIMEERING_API_KEY")
        if not key:
            raise MissingApiKeyError(
                "OPTIMEERING_API_KEY not found. Add it to .env (never commit it)."
            )
        # Build the SDK client. The key is held only inside the SDK config.
        self._api = PredictionsApi(OptimeeringClient(configuration=Configuration(api_key=key)))
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # On-disk cache helpers
    # ------------------------------------------------------------------ #
    def _cache_path(self, method: str, params: dict[str, Any]) -> Path:
        blob = json.dumps({"method": method, "params": params}, sort_keys=True, default=str)
        digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{method}_{digest}.json"

    def _cache_read(self, path: Path) -> Any | None:
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        return None

    def _cache_write(self, path: Path, payload: Any) -> None:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=0, default=str)

    # ------------------------------------------------------------------ #
    # Low-level SDK calls (wrapped with retry)
    # ------------------------------------------------------------------ #
    @_retry
    def _call_list_series(self, **filters: Any):
        return self._api.list_series(**filters)

    @_retry
    def _call_retrieve_latest(self, series_id: list[int], max_event_time: Any | None):
        return self._api.retrieve_latest(series_id=series_id, max_event_time=max_event_time)

    @_retry
    def _call_retrieve(self, series_id: list[int], start: Any, end: Any):
        return self._api.retrieve(series_id=series_id, start=start, end=end)

    # ------------------------------------------------------------------ #
    # Series catalogue
    # ------------------------------------------------------------------ #
    def list_series(self, use_cache: bool = True, **filters: Any) -> list[dict[str, Any]]:
        """Return series metadata dicts matching ``filters`` (area/product/...).

        The catalogue changes rarely, so it is cached by default.
        """
        # Drop None filters so the cache key is stable.
        clean = {k: v for k, v in filters.items() if v is not None}
        cache_path = self._cache_path("list_series", clean)
        if use_cache:
            cached = self._cache_read(cache_path)
            if cached is not None:
                return cached

        result = self._call_list_series(**clean)
        items = getattr(result, "items", result) or []
        series = [it.to_dict() for it in items]
        self._cache_write(cache_path, series)
        return series

    def _resolve_series(
        self,
        area: str,
        product: str,
        statistic: str,
        unit_type: str | None,
        resolution: str | None,
    ) -> dict[str, Any]:
        """Find the single series matching the given filters; error if 0 or >1."""
        matches = self.list_series(
            area=[area],
            product=[product],
            statistic=[statistic],
            unit_type=[unit_type] if unit_type else None,
            resolution=[resolution] if resolution else None,
        )
        if not matches:
            raise LookupError(
                f"No Optimeering series for area={area} product={product} "
                f"statistic={statistic} unit_type={unit_type} resolution={resolution}."
            )
        if len(matches) > 1:
            ids = [m.get("id") for m in matches]
            raise LookupError(
                f"Ambiguous series filter (matched ids {ids}); narrow resolution/unit_type."
            )
        return matches[0]

    # ------------------------------------------------------------------ #
    # Normalisation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalise(data: Any, meta: dict[str, Any]) -> list[dict[str, Any]]:
        """Flatten a PredictionsDataList into normalised records using series ``meta``."""
        records: list[dict[str, Any]] = []
        # NB: dict has a built-in `.items` method, so check dict first.
        if isinstance(data, dict):
            items = data.get("items", []) or []
        else:
            items = getattr(data, "items", []) or []

        for series in items:
            events = _attr(series, "events", [])
            for ev in events:
                vintage = _iso(_attr(ev, "event_time", None))
                for pred in _attr(ev, "predictions", []):
                    ts = _iso(_attr(pred, "prediction_for", None))
                    value = _attr(pred, "value", None)
                    base = {
                        "price_area": meta.get("area"),
                        "timestamp": ts,
                        "vintage_ts": vintage,
                        "statistic_type": meta.get("statistic"),
                        "unit_type": meta.get("unit_type"),
                        "resolution": meta.get("resolution"),
                    }
                    if isinstance(value, dict):
                        # Quantile/Distribution: one record per quantile level.
                        for q_label, q_val in value.items():
                            rec = dict(base)
                            rec["quantile"] = float(q_label)
                            rec["value"] = float(q_val)
                            records.append(rec)
                    elif value is not None:
                        rec = dict(base)
                        rec["quantile"] = None
                        rec["value"] = float(value)
                        records.append(rec)
        return records

    # ------------------------------------------------------------------ #
    # Public fetch methods
    # ------------------------------------------------------------------ #
    def get_imbalance_price_forecast(
        self,
        area: str,
        resolution: str = "PT15M",
        statistic: str = "Quantile",
        unit_type: str = "Price_Spread",
        product: str = "Imbalance",
        max_event_time: Any | None = None,
        use_cache: bool = False,
    ) -> list[dict[str, Any]]:
        """Fetch the latest (live) imbalance price forecast for ``area``.

        Returns normalised records. Not cached by default — this is the live
        forecast and is expected to change between calls.
        """
        meta = self._resolve_series(area, product, statistic, unit_type, resolution)
        params = {"series_id": meta["id"], "max_event_time": max_event_time}
        cache_path = self._cache_path("forecast_latest", params)
        if use_cache:
            cached = self._cache_read(cache_path)
            if cached is not None:
                return cached

        data = self._call_retrieve_latest(series_id=[meta["id"]], max_event_time=max_event_time)
        records = self._normalise(data, meta)
        if use_cache:
            self._cache_write(cache_path, records)
        return records

    def get_historical_prices(
        self,
        area: str,
        start: Any,
        end: Any,
        resolution: str = "PT15M",
        statistic: str = "Quantile",
        unit_type: str = "Price_Spread",
        product: str = "Imbalance",
        use_cache: bool = True,
    ) -> list[dict[str, Any]]:
        """Fetch a past window [start, end) for ``area``.

        Used for backfill / historical comparison in Week 3. A closed past window
        is immutable, so results are cached by default. ``start``/``end`` accept
        ISO 8601 datetimes or durations (e.g. ``-P7D``), per the SDK.

        NB: which exact series represents *realised actual* imbalance price is
        pinned down in Task 3.1; this method fetches whatever series matches the
        given filters over the window.
        """
        meta = self._resolve_series(area, product, statistic, unit_type, resolution)
        params = {"series_id": meta["id"], "start": start, "end": end}
        cache_path = self._cache_path("historical", params)
        if use_cache:
            cached = self._cache_read(cache_path)
            if cached is not None:
                return cached

        data = self._call_retrieve(series_id=[meta["id"]], start=start, end=end)
        records = self._normalise(data, meta)
        if use_cache:
            self._cache_write(cache_path, records)
        return records


# --------------------------------------------------------------------------- #
# Small helpers for tolerant attribute/timestamp handling
# --------------------------------------------------------------------------- #
def _attr(obj: Any, name: str, default: Any) -> Any:
    """Get ``name`` from an SDK object or a plain dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _iso(value: Any) -> str | None:
    """Normalise a datetime (or string) to an ISO 8601 string."""
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _ensure_list(x: Any) -> Iterable[Any]:
    return x if isinstance(x, (list, tuple)) else [x]
