"""Pluggable data sources for the dashboard (UI layer, Task 4.1).

The dashboard is **source-agnostic**: it never imports the live ingestion clients
and never talks to Optimeering / the markets SDK directly. It only ever talks to a
:class:`DataSource`. Two implementations ship today:

* :class:`ServiceDataSource` — the production path. Delegates to ``iar.service``
  (the frozen read API), which reads **only the SQLite database**. Live feeds reach
  that database via the backend scripts (``run_iar.py`` / ``backfill_iar.py`` / …),
  never via the UI. This keeps the architecture's "live feeds are not attached to
  the UI" rule intact.
* :class:`DemoDataSource` — a self-contained, deterministic, synthetic source.
  Needs no database, no API key and no network, so the dashboard is instantly
  demo-able. This is the "fake feed" you can swap in to make a demo product without
  changing a single line of UI code.

Both return the *same* plain-data shapes (tidy ``DataFrame`` / ``dict`` / scalar),
so swapping one for the other is a one-line change (see :func:`get_data_source`).
To add a third source later (e.g. a JSON snapshot for a hosted demo), implement the
same method surface.

Sign convention (inherited from the engine): IaR / CIaR / realised cost are in
**cost terms — positive = cost (bad)**, EUR.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# Reuse the *real* Kupiec test even for synthetic data, so the demo's backtest
# verdict is computed exactly the way the production path computes it.
from iar.risk.backtest import kupiec_pof

MTU_MINUTES = 15
MTUS_PER_DAY = 24 * 60 // MTU_MINUTES  # 96


# --------------------------------------------------------------------------- #
# The interface
# --------------------------------------------------------------------------- #
class DataSource(ABC):
    """Read surface the dashboard renders. Implementations return plain data only.

    Method contracts (all euro figures positive = cost):

    - :meth:`list_portfolios` → ``DataFrame[portfolio_id, name, price_area]``.
    - :meth:`overview` → KPI summary ``dict`` (or ``None`` when no data), shape::

          {
            "confidence", "vintage_ts", "run_ts", "n_scenarios", "horizon",
            "gross":  {"period_iar","ciar","peak_mtu_iar","limit","utilisation","severity"},
            "spread": {"period_iar","ciar","peak_mtu_iar","limit","utilisation","severity"},
            "overperformance_ratio",  # float | None
            "n_warnings", "n_breaches",
          }

      ``peak_mtu_iar`` / ``overperformance_ratio`` may be ``None`` when a source
      cannot supply them (the production engine emits one *period* IaR, not a
      per-MTU series — see ``docs/assumptions.md``).
    - :meth:`intraday` → ``DataFrame[timestamp, forecast_iar, realised_iar,
      position_mwh, mtu_limit, is_past]`` (empty when the source has no per-MTU series).
    - :meth:`heatmap` → tidy ``DataFrame[hour, quarter, iar]`` (empty if unavailable).
    - :meth:`limit_status` → ``DataFrame[label, iar_type, limit_type, current_iar,
      limit, utilisation, severity]`` (``severity`` ``None`` ⇒ within limit).
    - :meth:`alerts` → ``DataFrame[ts, title, body, severity]`` (newest first).
    - :meth:`iar_curve` → ``DataFrame[vintage_ts, iar_value, ciar_value, confidence]``.
    - :meth:`backtest` → summary ``dict`` incl. a ``"periods"`` ``DataFrame``.
    """

    #: Short human label shown in the UI source picker.
    label: str = "data source"

    @abstractmethod
    def list_portfolios(self) -> pd.DataFrame: ...

    @abstractmethod
    def overview(self, portfolio_id: int, *, confidence: float = 0.95) -> dict | None: ...

    @abstractmethod
    def intraday(self, portfolio_id: int, *, basis: str = "gross") -> pd.DataFrame: ...

    @abstractmethod
    def heatmap(self, portfolio_id: int, *, basis: str = "gross") -> pd.DataFrame: ...

    @abstractmethod
    def limit_status(self, portfolio_id: int) -> pd.DataFrame: ...

    @abstractmethod
    def alerts(self, portfolio_id: int) -> pd.DataFrame: ...

    @abstractmethod
    def iar_curve(self, portfolio_id: int, *, basis: str = "gross") -> pd.DataFrame: ...

    @abstractmethod
    def backtest(
        self, portfolio_id: int, *, basis: str = "gross", significance: float = 0.05
    ) -> dict: ...


# Empty-frame templates so the UI can render headers even with no rows.
_EMPTY_INTRADAY = pd.DataFrame(
    columns=["timestamp", "forecast_iar", "realised_iar", "position_mwh", "mtu_limit", "is_past"]
)
_EMPTY_HEATMAP = pd.DataFrame(columns=["hour", "quarter", "iar"])
_EMPTY_LIMITS = pd.DataFrame(
    columns=["label", "iar_type", "limit_type", "current_iar", "limit", "utilisation", "severity"]
)
_EMPTY_ALERTS = pd.DataFrame(columns=["ts", "title", "body", "severity"])
_EMPTY_CURVE = pd.DataFrame(columns=["vintage_ts", "iar_value", "ciar_value", "confidence"])


def _limit_label(iar_type: str, limit_type: str) -> str:
    """Pretty label for the limit-status table, e.g. ``Period Gross — Day``."""
    type_names = {
        "remaining_day": "Period {b} — Day",
        "rolling_window": "Rolling {b} window",
        "per_mtu": "Per-MTU {b} (peak)",
    }
    return type_names.get(limit_type, f"{{b}} {limit_type}").format(b=iar_type.capitalize())


# --------------------------------------------------------------------------- #
# Production source — delegates to iar.service (DB only, no live feeds)
# --------------------------------------------------------------------------- #
class ServiceDataSource(DataSource):
    """Reads the real pipeline output through ``iar.service`` (SQLite only).

    Per-MTU intraday and the risk heatmap come back empty because the MVP engine
    emits a single *period* IaR, not a per-MTU IaR series (descoped — see
    ``docs/assumptions.md``); the UI shows an honest placeholder for those panels.
    Everything else (KPIs, limit status, alerts, IaR curve, backtest) is real.
    """

    label = "Live (database)"

    def __init__(self) -> None:
        # Imported lazily so a missing/locked DB can't break ``import data_source``.
        from iar import service

        self._svc = service

    # -- portfolios -------------------------------------------------------- #
    def list_portfolios(self) -> pd.DataFrame:
        return self._svc.list_portfolios()

    # -- KPI overview ------------------------------------------------------ #
    def overview(self, portfolio_id: int, *, confidence: float = 0.95) -> dict | None:
        latest = self._svc.get_latest_iar(portfolio_id)
        if latest is None:
            return None
        limits = self._svc.get_limit_status(portfolio_id)  # remaining_day, gross+spread
        by_type = {row["iar_type"]: row for _, row in limits.iterrows()}

        def _basis(name: str) -> dict:
            pair = latest.get(name) or {}
            lim = by_type.get(name)
            return {
                "period_iar": pair.get("iar"),
                "ciar": pair.get("ciar"),
                "peak_mtu_iar": None,  # engine emits no per-MTU IaR series (MVP)
                "limit": (None if lim is None else lim["limit_value"]),
                "utilisation": (None if lim is None else lim["utilisation"]),
                "severity": (None if lim is None else lim["severity"]),
            }

        sev = limits["severity"] if not limits.empty else pd.Series(dtype=object)
        return {
            "confidence": latest.get("confidence"),
            "vintage_ts": latest.get("vintage_ts"),
            "run_ts": latest.get("run_ts"),
            "n_scenarios": latest.get("n_scenarios"),
            "horizon": latest.get("horizon"),
            "gross": _basis("gross"),
            "spread": _basis("spread"),
            "overperformance_ratio": None,
            "n_warnings": int((sev == "soft").sum()),
            "n_breaches": int((sev == "hard").sum()),
        }

    # -- per-MTU panels: not emitted by the MVP engine --------------------- #
    def intraday(self, portfolio_id: int, *, basis: str = "gross") -> pd.DataFrame:
        return _EMPTY_INTRADAY.copy()

    def heatmap(self, portfolio_id: int, *, basis: str = "gross") -> pd.DataFrame:
        return _EMPTY_HEATMAP.copy()

    # -- limits ------------------------------------------------------------ #
    def limit_status(self, portfolio_id: int) -> pd.DataFrame:
        raw = self._svc.get_limit_status(portfolio_id)
        if raw.empty:
            return _EMPTY_LIMITS.copy()
        return pd.DataFrame(
            {
                "label": [_limit_label(r["iar_type"], r["limit_type"]) for _, r in raw.iterrows()],
                "iar_type": raw["iar_type"],
                "limit_type": raw["limit_type"],
                "current_iar": raw["iar_value"],
                "limit": raw["limit_value"],
                "utilisation": raw["utilisation"],
                "severity": raw["severity"],
            }
        ).reset_index(drop=True)

    # -- alerts ------------------------------------------------------------ #
    def alerts(self, portfolio_id: int) -> pd.DataFrame:
        raw = self._svc.get_alerts(portfolio_id)
        if raw.empty:
            return _EMPTY_ALERTS.copy()
        rows = []
        for _, a in raw.iterrows():
            sev = a["severity"]
            verb = "Breach" if sev == "hard" else "Warning"
            rows.append(
                {
                    "ts": a["breach_ts"],
                    "title": f"{verb} — {str(a['iar_type']).capitalize()} {a['limit_type']}",
                    "body": (
                        f"IaR €{a['iar_value']:,.0f} vs limit €{a['limit_value']:,.0f}"
                        if pd.notna(a["iar_value"]) else f"Limit €{a['limit_value']:,.0f}"
                    ),
                    "severity": sev,
                }
            )
        return pd.DataFrame(rows, columns=["ts", "title", "body", "severity"])

    # -- IaR curve over vintages ------------------------------------------ #
    def iar_curve(self, portfolio_id: int, *, basis: str = "gross") -> pd.DataFrame:
        raw = self._svc.get_iar_curve(portfolio_id, basis)
        if raw.empty:
            return _EMPTY_CURVE.copy()
        return raw[["vintage_ts", "iar_value", "ciar_value", "confidence"]].reset_index(drop=True)

    # -- backtest ---------------------------------------------------------- #
    def backtest(
        self, portfolio_id: int, *, basis: str = "gross", significance: float = 0.05
    ) -> dict:
        return self._svc.get_backtest_summary(portfolio_id, basis, significance=significance)


# --------------------------------------------------------------------------- #
# Demo source — fully synthetic, deterministic, offline
# --------------------------------------------------------------------------- #
# Demo euro-limits (mirror config/limits.toml shape; tuned to look like the mockup).
_DEMO_LIMITS = {
    "gross": {"remaining_day": 50_000, "rolling_window": 10_000, "per_mtu": 3_000},
    "spread": {"remaining_day": 30_000, "rolling_window": 6_000, "per_mtu": 1_500},
}
_DEMO_PORTFOLIOS = [
    (1, "NO1 Wind", "NO1"),
    (2, "NO2 Wind", "NO2"),
    (3, "SE3 Wind", "SE3"),
]
# A fixed "as-of" so the demo is fully deterministic (no wall-clock dependence).
_DEMO_NOW = datetime(2026, 6, 11, 14, 15, tzinfo=timezone.utc)


class DemoDataSource(DataSource):
    """Synthetic, deterministic source — no DB, no key, no network.

    Everything is generated from a per-portfolio seed so a given portfolio always
    renders identically. Provides the *full* mockup surface (per-MTU intraday bars,
    risk heatmap, peak-MTU IaR, overperformance ratio) that the production engine
    does not yet emit — this is the "fake feed" for building a demo product.
    """

    label = "Demo (synthetic)"

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _rng(portfolio_id: int) -> np.random.Generator:
        return np.random.default_rng(1000 + portfolio_id)

    @staticmethod
    def _conf_scale(confidence: float) -> float:
        """Heavier confidence ⇒ a larger worst-case figure (rough z-ratio vs P95)."""
        from scipy.stats import norm

        return float(norm.ppf(confidence) / norm.ppf(0.95))

    def _mtu_profile(self, portfolio_id: int, basis: str, confidence: float):
        """Per-MTU forecast IaR + position for a day, peaking in the afternoon."""
        rng = self._rng(portfolio_id + (0 if basis == "gross" else 50))
        n = MTUS_PER_DAY
        hours = np.arange(n) * (MTU_MINUTES / 60.0)
        # Afternoon-peaking risk shape (evening wind ramp + price volatility).
        shape = 0.45 + 0.55 * np.exp(-((hours - 15.5) ** 2) / (2 * 3.2 ** 2))
        base_peak = (2_140 if basis == "gross" else 880) * self._conf_scale(confidence)
        noise = rng.normal(1.0, 0.10, n).clip(0.6, 1.5)
        forecast = base_peak * shape * noise
        # Net-short wind position (MWh per MTU), mildly time-varying.
        position = -(18 + 10 * shape + rng.normal(0, 2, n))
        return forecast, position

    # -- portfolios -------------------------------------------------------- #
    def list_portfolios(self) -> pd.DataFrame:
        return pd.DataFrame(_DEMO_PORTFOLIOS, columns=["portfolio_id", "name", "price_area"])

    # -- KPI overview ------------------------------------------------------ #
    def overview(self, portfolio_id: int, *, confidence: float = 0.95) -> dict | None:
        scale = self._conf_scale(confidence)
        out: dict = {
            "confidence": confidence,
            "vintage_ts": _DEMO_NOW - timedelta(hours=10),
            "run_ts": _DEMO_NOW - timedelta(minutes=8),
            "n_scenarios": 10_000,
            "horizon": "96xPT15M",
            "overperformance_ratio": round(0.72 + 0.0 * portfolio_id, 2),
        }
        rng = self._rng(portfolio_id)
        for basis in ("gross", "spread"):
            forecast, _ = self._mtu_profile(portfolio_id, basis, confidence)
            # Period IaR is NOT the sum of per-MTU IaRs (summed-quantile concept);
            # approximate it as a sub-additive aggregate for plausible demo numbers.
            period = float(forecast.sum() * (0.62 + 0.04 * rng.random()))
            peak = float(forecast.max())
            limit = _DEMO_LIMITS[basis]["remaining_day"]
            util = period / limit
            out[basis] = {
                "period_iar": period,
                "ciar": period * (1.18 + 0.05 * rng.random()),
                "peak_mtu_iar": peak,
                "limit": limit,
                "utilisation": util,
                "severity": _severity(util),
            }
        statuses = self.limit_status(portfolio_id)["severity"]
        out["n_warnings"] = int((statuses == "soft").sum())
        out["n_breaches"] = int((statuses == "hard").sum())
        return out

    # -- intraday ---------------------------------------------------------- #
    def intraday(self, portfolio_id: int, *, basis: str = "gross") -> pd.DataFrame:
        forecast, position = self._mtu_profile(portfolio_id, basis, 0.95)
        n = MTUS_PER_DAY
        start = _DEMO_NOW.replace(hour=0, minute=0, second=0, microsecond=0)
        ts = [start + timedelta(minutes=MTU_MINUTES * i) for i in range(n)]
        current_idx = int((_DEMO_NOW - start).total_seconds() // (MTU_MINUTES * 60))
        is_past = np.arange(n) < current_idx
        rng = self._rng(portfolio_id + 7)
        # Realised IaR only exists for elapsed MTUs; jitter around the forecast.
        realised = np.where(is_past, forecast * rng.normal(1.0, 0.18, n).clip(0.4, 1.8), np.nan)
        mtu_limit = _DEMO_LIMITS[basis]["per_mtu"]
        return pd.DataFrame(
            {
                "timestamp": ts,
                "forecast_iar": forecast,
                "realised_iar": realised,
                "position_mwh": position,
                "mtu_limit": float(mtu_limit),
                "is_past": is_past,
            }
        )

    # -- heatmap ----------------------------------------------------------- #
    def heatmap(self, portfolio_id: int, *, basis: str = "gross") -> pd.DataFrame:
        df = self.intraday(portfolio_id, basis=basis)
        df = df.assign(
            hour=[t.hour for t in df["timestamp"]],
            quarter=[t.minute for t in df["timestamp"]],
        )
        return df[["hour", "quarter", "forecast_iar"]].rename(columns={"forecast_iar": "iar"})

    # -- limits ------------------------------------------------------------ #
    def limit_status(self, portfolio_id: int) -> pd.DataFrame:
        rng = self._rng(portfolio_id + 3)
        rows = []
        for basis in ("gross", "spread"):
            forecast, _ = self._mtu_profile(portfolio_id, basis, 0.95)
            period = float(forecast.sum() * (0.62 + 0.04 * rng.random()))
            peak = float(forecast.max())
            # Rolling 4h window = worst contiguous 16-MTU sum (illustrative).
            window = float(pd.Series(forecast).rolling(16).sum().max())
            current = {"remaining_day": period, "per_mtu": peak, "rolling_window": window}
            for limit_type in ("remaining_day", "rolling_window", "per_mtu"):
                limit = _DEMO_LIMITS[basis][limit_type]
                cur = current[limit_type]
                util = cur / limit
                rows.append(
                    {
                        "label": _limit_label(basis, limit_type),
                        "iar_type": basis,
                        "limit_type": limit_type,
                        "current_iar": cur,
                        "limit": float(limit),
                        "utilisation": util,
                        "severity": _severity(util),
                    }
                )
        return pd.DataFrame(rows, columns=_EMPTY_LIMITS.columns)

    # -- alerts ------------------------------------------------------------ #
    def alerts(self, portfolio_id: int) -> pd.DataFrame:
        statuses = self.limit_status(portfolio_id)
        rows = []
        for _, r in statuses.iterrows():
            if r["severity"] is None:
                continue
            verb = "Breach" if r["severity"] == "hard" else "Warning"
            rows.append(
                {
                    "ts": _DEMO_NOW - timedelta(minutes=len(rows) * 12 + 3),
                    "title": f"{verb} threshold reached — {r['label']}",
                    "body": (
                        f"{r['label']} reached €{r['current_iar']:,.0f} "
                        f"({r['utilisation']:.0%} of €{r['limit']:,.0f} limit)."
                    ),
                    "severity": r["severity"],
                }
            )
        # A couple of informational feed entries to look alive.
        rows.append(
            {
                "ts": _DEMO_NOW - timedelta(minutes=45),
                "title": "Forecast update",
                "body": "New Optimeering imbalance forecast received; period IaR revised.",
                "severity": None,
            }
        )
        rows.append(
            {
                "ts": _DEMO_NOW - timedelta(minutes=70),
                "title": "Position update",
                "body": "Portfolio position updated for afternoon MTUs (+12 MWh long).",
                "severity": None,
            }
        )
        return pd.DataFrame(rows, columns=_EMPTY_ALERTS.columns).sort_values(
            "ts", ascending=False, ignore_index=True
        )

    # -- IaR curve over vintages ------------------------------------------ #
    def iar_curve(self, portfolio_id: int, *, basis: str = "gross") -> pd.DataFrame:
        rng = self._rng(portfolio_id + (11 if basis == "gross" else 13))
        n_days = 20
        anchor = self.overview(portfolio_id)[basis]["period_iar"]
        vintages = [_DEMO_NOW - timedelta(days=(n_days - i)) for i in range(n_days)]
        drift = np.cumsum(rng.normal(0, 0.05, n_days))
        iar = anchor * (0.85 + 0.30 * (drift - drift.min()) / (np.ptp(drift) + 1e-9))
        return pd.DataFrame(
            {
                "vintage_ts": pd.to_datetime(vintages, utc=True),
                "iar_value": iar,
                "ciar_value": iar * 1.2,
                "confidence": 0.95,
            }
        )

    # -- backtest ---------------------------------------------------------- #
    def backtest(
        self, portfolio_id: int, *, basis: str = "gross", significance: float = 0.05
    ) -> dict:
        rng = self._rng(portfolio_id + (21 if basis == "gross" else 23))
        confidence = 0.95
        n = 30
        anchor = self.overview(portfolio_id)[basis]["period_iar"]
        periods = []
        for i in range(n):
            estimate = anchor * (0.9 + 0.2 * rng.random())
            # Realised cost ~ below the IaR estimate most days; occasional breach.
            realised = estimate * (0.55 + 0.45 * rng.random())
            if rng.random() < 0.06:  # ~6% exceedance — close to the 5% target
                realised = estimate * (1.05 + 0.4 * rng.random())
            periods.append(
                {
                    "period": (_DEMO_NOW - timedelta(days=n - i)).strftime("%Y-%m-%d"),
                    "iar_estimate": estimate,
                    "realised_cost": realised,
                    "exceeded": realised > estimate,
                }
            )
        pdf = pd.DataFrame(periods, columns=["period", "iar_estimate", "realised_cost", "exceeded"])
        k = kupiec_pof(
            n_obs=len(pdf),
            n_exceedances=int(pdf["exceeded"].sum()),
            expected_rate=1.0 - confidence,
            significance=significance,
        )
        return {
            "portfolio_id": portfolio_id,
            "iar_type": basis,
            "confidence": confidence,
            "n_periods": int(len(pdf)),
            "n_exceedances": int(pdf["exceeded"].sum()),
            "observed_rate": k.observed_rate,
            "expected_rate": k.expected_rate,
            "kupiec_lr": k.lr_statistic,
            "kupiec_p_value": k.p_value,
            "well_calibrated": k.well_calibrated,
            "periods": pdf,
        }


# --------------------------------------------------------------------------- #
# Shared helpers + factory
# --------------------------------------------------------------------------- #
def _severity(utilisation: float, soft_ratio: float = 0.80) -> str | None:
    """Mirror ``iar.risk.alerts.classify_severity`` on a utilisation ratio."""
    if utilisation is None or not np.isfinite(utilisation):
        return None
    if utilisation > 1.0:
        return "hard"
    if utilisation > soft_ratio:
        return "soft"
    return None


def get_data_source(kind: str) -> DataSource:
    """Factory: ``"live"`` → :class:`ServiceDataSource`, else :class:`DemoDataSource`.

    This is the single swap point. Point the dashboard at a different source by
    changing the ``kind`` it requests — no UI code changes.
    """
    if kind == "live":
        return ServiceDataSource()
    return DemoDataSource()
