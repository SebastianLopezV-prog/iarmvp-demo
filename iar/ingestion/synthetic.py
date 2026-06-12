"""Synthetic market feeds for the demo build (NO real data, NO API key, NO downloads).

This module replaces the three external live feeds the production MVP uses with a
self-contained, deterministic model that *behaves like* Nordic wind-market data:

* the day-ahead (DAM) spot price            -> :class:`SyntheticMarketsClient.get_dam_prices`
* the realised (settled) imbalance price    -> :class:`SyntheticMarketsClient.get_imbalance_prices`
* the forward imbalance-spread quantiles    -> :class:`SyntheticForecastClient.get_imbalance_price_forecast`

The two synthetic clients expose the **same method surface and record shapes** as the
real ``OptimeeringForecastClient`` / ``OptimeeringMarketsClient``, so the rest of the
pipeline (sampler, engine, persistence, backtest, dashboard) is unchanged. The
``iar.ingestion.clients`` factory selects these by default in the demo.

Design notes
------------
* **Deterministic.** Every value is derived from a hash of ``(area, kind, key)``, so a
  given timestamp returns the *same* DAM price / realised price across calls. This is
  essential: the forward run and the settled-price load must agree on the same MTU.
* **DAM spot** = a smooth diurnal curve (morning + evening peaks, night trough) in
  local Norwegian time, plus a per-day level shift and a weekend discount.
* **Imbalance spread** (``lambda - p``) is modelled as a split-scale Student-t: heavy
  tails (df=4), a wider *upper* scale (up-regulation price spikes cost more), a small
  positive evening location, and a per-day "system-stress" regime multiplier so some
  days are calm and a few are volatile. The forward forecast publishes the analytic
  quantiles of this distribution; the realised value is a single draw from it. Because
  both come from the same law, the backtest exceedance rate lands near 1 - confidence.
* **Realised imbalance price** = DAM spot + realised spread (absolute EUR/MWh).

Units are EUR/MWh. Timestamps are ISO 8601 UTC strings, matching the real clients.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

# Display/market timezone for the diurnal shape: the portfolio is Nordic, so the
# price curve peaks at local Norwegian wall-clock hours.
MARKET_TZ = "Europe/Oslo"
MTU = pd.Timedelta(minutes=15)

# PT15M imbalance series quantile levels (percent), matching the real Optimeering feed.
QUANTILE_LEVELS: tuple[int, ...] = (1, 5, 10, 25, 50, 75, 90, 95, 99)


# --------------------------------------------------------------------------- #
# Deterministic seeding
# --------------------------------------------------------------------------- #
def _seed(*parts: Any) -> int:
    """Stable 64-bit seed from the string form of ``parts`` (reproducible across runs)."""
    blob = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big")


def _rng(*parts: Any) -> np.random.Generator:
    return np.random.default_rng(_seed(*parts))


def _local_hour(ts: pd.Timestamp) -> float:
    """Hour-of-day (0-24, fractional) in Norwegian local time."""
    loc = ts.tz_convert(MARKET_TZ)
    return loc.hour + loc.minute / 60.0


# --------------------------------------------------------------------------- #
# Market model
# --------------------------------------------------------------------------- #
def _daily_regime(area: str, day: Any) -> float:
    """System-stress multiplier for a delivery day (~0.6 calm .. ~3 stressed).

    Most days sit near 1; a minority are volatile (heavier imbalance tails). This is
    what produces day-to-day variation and the occasional backtest exceedance.
    """
    r = _rng(area, "regime", day)
    base = float(np.exp(r.normal(0.0, 0.45)))  # lognormal, median 1
    if r.random() < 0.12:                       # ~1 day in 8 is a stress day
        base *= float(r.uniform(1.8, 3.0))
    return float(np.clip(base, 0.6, 4.0))


def dam_price(area: str, ts: pd.Timestamp) -> float:
    """Deterministic DAM (spot) cleared price in EUR/MWh for one MTU."""
    h = _local_hour(ts)
    shape = (
        38.0
        + 9.0 * np.exp(-((h - 8.0) ** 2) / (2 * 1.8 ** 2))     # morning peak
        + 13.0 * np.exp(-((h - 18.5) ** 2) / (2 * 2.2 ** 2))   # evening peak
        - 7.0 * np.exp(-((h - 3.0) ** 2) / (2 * 2.2 ** 2))     # night trough
    )
    day = ts.tz_convert(MARKET_TZ).date()
    level = float(_rng(area, "dam-level", day).normal(0.0, 7.0))     # per-day shift
    weekend = -5.0 if ts.tz_convert(MARKET_TZ).dayofweek >= 5 else 0.0
    noise = float(_rng(area, "dam-noise", ts.isoformat()).normal(0.0, 1.5))
    return float(max(shape + level + weekend + noise, -15.0))


def _spread_params(area: str, ts: pd.Timestamp) -> tuple[float, float, float, float]:
    """(location, lower-scale, upper-scale, dof) of the per-MTU spread distribution."""
    h = _local_hour(ts)
    day = ts.tz_convert(MARKET_TZ).date()
    regime = _daily_regime(area, day)
    base_scale = (
        5.5
        + 11.0 * np.exp(-((h - 18.5) ** 2) / (2 * 2.6 ** 2))   # evening volatility
        + 4.0 * np.exp(-((h - 7.5) ** 2) / (2 * 2.0 ** 2))     # morning ramp
    )
    scale = base_scale * regime
    loc = 1.0 + 2.5 * np.exp(-((h - 18.5) ** 2) / (2 * 3.0 ** 2))   # slight evening up-bias
    scale_up = scale * 1.7   # up-regulation spikes are larger (system short, costly)
    scale_dn = scale * 1.0
    return float(loc), float(scale_dn), float(scale_up), 4.0


def _spread_at(u, loc: float, scale_dn: float, scale_up: float, df: float):
    """Split-scale Student-t quantile/value at uniform ``u`` (scalar or array)."""
    z = student_t.ppf(u, df)
    scale = np.where(z >= 0.0, scale_up, scale_dn)
    return loc + scale * z


def spread_quantiles(area: str, ts: pd.Timestamp, levels=QUANTILE_LEVELS) -> dict[float, float]:
    """Analytic spread quantiles {percent_level: EUR} for the forward forecast."""
    loc, sd, su, df = _spread_params(area, ts)
    u = np.array([lv / 100.0 for lv in levels])
    vals = _spread_at(u, loc, sd, su, df)
    return {float(lv): float(v) for lv, v in zip(levels, vals)}


def realised_spread(area: str, ts: pd.Timestamp) -> float:
    """A single realised spread draw for one settled MTU (seeded by timestamp)."""
    loc, sd, su, df = _spread_params(area, ts)
    u = float(_rng(area, "spread-draw", ts.isoformat()).random())
    return float(_spread_at(u, loc, sd, su, df))


def realised_imbalance_price(area: str, ts: pd.Timestamp) -> float:
    """Realised absolute imbalance price = DAM spot + realised spread (EUR/MWh)."""
    return dam_price(area, ts) + realised_spread(area, ts)


# --------------------------------------------------------------------------- #
# Window / time helpers (accept ISO 8601 datetimes or simple durations like -P7D)
# --------------------------------------------------------------------------- #
_DUR = re.compile(r"^(-?)P(?:(\d+)D)?(?:T(\d+)H)?$")


def _parse_when(value: Any, now: pd.Timestamp) -> pd.Timestamp:
    """Resolve a datetime or an ISO-8601-ish duration (``-P7D``, ``P2D``, ``P0D``)."""
    if value is None:
        return now
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.to_datetime(value, utc=True)
    m = _DUR.match(str(value).strip())
    if m:
        sign = -1 if m.group(1) == "-" else 1
        days = int(m.group(2) or 0)
        hours = int(m.group(3) or 0)
        return now + sign * pd.Timedelta(days=days, hours=hours)
    return pd.to_datetime(value, utc=True)


def _window_index(start: Any, end: Any) -> pd.DatetimeIndex:
    """A 15-min UTC index over [start, end) resolving durations relative to *now*."""
    now = pd.Timestamp.now(tz="UTC").floor("15min")
    a = _parse_when(start, now).tz_convert("UTC").floor("15min")
    b = _parse_when(end, now).tz_convert("UTC").floor("15min")
    if b <= a:
        b = a + MTU
    return pd.date_range(a, b, freq="15min", inclusive="left")


# --------------------------------------------------------------------------- #
# Drop-in synthetic clients (same interface as the real Optimeering clients)
# --------------------------------------------------------------------------- #
DEFAULT_CAPACITY_MW = 100.0  # installed wind capacity for the synthetic portfolio

# Real windsim daily capacity-factor profiles (forecast / actual / cleared-bid), extracted
# once from the windsim DuckDB and shipped as a small committed file. This is the "windsim
# shape as the base"; the generator replays these and rolls them forward (no windsim install
# and no DuckDB needed on a host).
_PROFILES_PATH = Path(__file__).resolve().parent / "windsim_profiles.json"
_PROFILES_CACHE: list[dict[str, Any]] | None = None


def _load_profiles() -> list[dict[str, Any]]:
    global _PROFILES_CACHE
    if _PROFILES_CACHE is None:
        try:
            with _PROFILES_PATH.open("r", encoding="utf-8") as fh:
                _PROFILES_CACHE = json.load(fh).get("days", [])
        except Exception:
            _PROFILES_CACHE = []
    return _PROFILES_CACHE


def generate_wind_portfolio(
    area: str, start: Any, end: Any, capacity_mw: float = DEFAULT_CAPACITY_MW
) -> pd.DataFrame:
    """Synthetic wind portfolio over [start, end): DAM position, forecast, actual delivery.

    Replays REAL windsim daily capacity-factor profiles (``windsim_profiles.json``) so the
    shapes match windsim, while staying self-contained (no windsim install, no DuckDB) and
    rolling forward forever:

    * each delivery day deterministically picks a windsim profile (seeded by area + date)
      with a small per-day scale jitter, so days vary but always look like real wind;
    * **generation forecast** = windsim latest forecast, **actual delivery** = windsim
      actual, **DAM position** = windsim cleared bid (falling back to the forecast on the
      rare uncleared quarter so the committed position never reads exactly 0);
    * the forecast-vs-actual gap is windsim's own forecast error - the imbalance the IaR
      measures.

    All quantities are MWh per 15-min MTU. Deterministic in ``(area, day)``. Columns:
    ``timestamp`` (UTC), ``dam_mwh``, ``forecast_mwh``, ``actual_mwh``.
    """
    idx = _window_index(start, end)
    cap_mwh = capacity_mw * 0.25
    profiles = _load_profiles()
    rows: list[dict[str, Any]] = []
    if not profiles:  # defensive fallback: flat-ish profile (should not happen in the demo)
        for ts in idx:
            rows.append({"timestamp": ts, "dam_mwh": 0.4 * cap_mwh,
                         "forecast_mwh": 0.4 * cap_mwh, "actual_mwh": 0.4 * cap_mwh})
        return pd.DataFrame(rows, columns=["timestamp", "dam_mwh", "forecast_mwh", "actual_mwh"])

    n_days = len(profiles)
    local = idx.tz_convert(MARKET_TZ)
    for ts, lt in zip(idx, local):
        prof = profiles[_seed(area, "windsim-day", lt.date()) % n_days]
        q = lt.hour * 4 + lt.minute // 15          # quarter-of-day 0..95
        jit = 0.88 + 0.24 * (_rng(area, "windsim-jit", lt.date()).random())  # per-day scale
        fc = prof["forecast_cf"][q]
        ac = prof["actual_cf"][q]
        dam = prof["dam_cf"][q] if prof["dam_cf"][q] > 0.001 else fc   # never exactly 0
        rows.append({
            "timestamp": ts,
            "dam_mwh": float(np.clip(dam * jit, 0.0, 1.0) * cap_mwh),
            "forecast_mwh": float(np.clip(fc * jit, 0.0, 1.0) * cap_mwh),
            "actual_mwh": float(np.clip(ac * jit, 0.0, 1.0) * cap_mwh),
        })
    return pd.DataFrame(rows, columns=["timestamp", "dam_mwh", "forecast_mwh", "actual_mwh"])


def store_synthetic_portfolio(
    session: Any,
    area: str,
    start: Any = "-P31D",
    end: Any = "P3D",
    user: str = "Wind Co",
    portfolio_name: str | None = None,
    capacity_mw: float = DEFAULT_CAPACITY_MW,
    replace: bool = True,
) -> tuple[Any, int]:
    """Generate + persist the synthetic portfolio's positions/forecast/delivery.

    Writes the same three tables the windsim loader fills, so the rest of the pipeline
    (run_iar / backfill_history / realised cost) is unchanged. Returns ``(portfolio, n)``.
    """
    from iar.db.models import ActualDelivery, DAMPosition, GenerationForecast
    from iar.ingestion.flatfile_loader import get_or_create_portfolio

    pf = get_or_create_portfolio(session, user, portfolio_name or f"{area} Wind", area)
    df = generate_wind_portfolio(area, start, end, capacity_mw)
    if replace:
        for model in (DAMPosition, GenerationForecast, ActualDelivery):
            session.query(model).filter(model.portfolio_id == pf.portfolio_id).delete()
    for _, r in df.iterrows():
        ts = r["timestamp"].to_pydatetime()
        session.add(DAMPosition(portfolio_id=pf.portfolio_id, timestamp=ts, mwh=r["dam_mwh"]))
        session.add(GenerationForecast(portfolio_id=pf.portfolio_id, timestamp=ts,
                                       forecast_mwh=r["forecast_mwh"]))
        session.add(ActualDelivery(portfolio_id=pf.portfolio_id, timestamp=ts,
                                   actual_mwh=r["actual_mwh"]))
    session.flush()
    return pf, len(df)


class SyntheticForecastClient:
    """Synthetic stand-in for ``OptimeeringForecastClient`` (imbalance spread quantiles)."""

    #: How far ahead a published forecast vintage extends (Optimeering is ~36-48h).
    HORIZON_HOURS = 40

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # accept/ignore real-client args
        pass

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
        """Forward spread-quantile records from a vintage (now, or ``max_event_time``)."""
        now = pd.Timestamp.now(tz="UTC")
        vintage = pd.to_datetime(max_event_time, utc=True) if max_event_time else now
        start = vintage.floor("15min")
        idx = pd.date_range(start, start + pd.Timedelta(hours=self.HORIZON_HOURS),
                            freq="15min", inclusive="left")
        v_iso = vintage.isoformat()
        records: list[dict[str, Any]] = []
        for ts in idx:
            ts_iso = ts.isoformat()
            for lv, val in spread_quantiles(area, ts).items():
                records.append({
                    "price_area": area,
                    "timestamp": ts_iso,
                    "vintage_ts": v_iso,
                    "statistic_type": "Quantile",
                    "quantile": float(lv),
                    "unit_type": unit_type,
                    "resolution": resolution,
                    "value": float(val),
                })
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
        """Spread-quantile records over a past window (same shape as the live forecast)."""
        idx = _window_index(start, end)
        records: list[dict[str, Any]] = []
        for ts in idx:
            ts_iso = ts.isoformat()
            for lv, val in spread_quantiles(area, ts).items():
                records.append({
                    "price_area": area,
                    "timestamp": ts_iso,
                    "vintage_ts": ts_iso,
                    "statistic_type": "Quantile",
                    "quantile": float(lv),
                    "unit_type": unit_type,
                    "resolution": resolution,
                    "value": float(val),
                })
        return records


class SyntheticMarketsClient:
    """Synthetic stand-in for ``OptimeeringMarketsClient`` (DAM spot + realised imbalance)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # accept/ignore real-client args
        pass

    def get_dam_prices(
        self, area: str, start: Any = "-P1D", end: Any = "P2D", **_: Any
    ) -> list[dict[str, Any]]:
        """DAM cleared (spot) price records over [start, end)."""
        return [
            {"price_area": area, "timestamp": ts.isoformat(), "eur_per_mwh": dam_price(area, ts)}
            for ts in _window_index(start, end)
        ]

    def get_imbalance_prices(
        self, area: str, start: Any = "-P7D", end: Any = "P0D", **_: Any
    ) -> list[dict[str, Any]]:
        """Realised (settled) absolute imbalance price records over [start, end)."""
        return [
            {"price_area": area, "timestamp": ts.isoformat(),
             "eur_per_mwh": realised_imbalance_price(area, ts)}
            for ts in _window_index(start, end)
        ]
