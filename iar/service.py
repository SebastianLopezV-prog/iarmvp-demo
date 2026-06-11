"""Backend service interface (Task 3.5) — the only module the UI should import.

A thin, frozen read API over the database and the risk modules. Every function
returns **plain data** (tidy ``DataFrame`` / ``dict`` / scalar), never ORM
objects, so the Streamlit UI stays dumb: it calls these and renders, with no DB
or simulation logic of its own (architecture: backend separated from UI).

Frozen surface
--------------
- :func:`list_portfolios`          — portfolios available to pick.
- :func:`get_portfolio`            — resolve one by area or id.
- :func:`get_latest_iar`           — newest run's Gross/Spread IaR + CIaR.
- :func:`get_iar_curve`            — IaR estimate over time (per stored vintage).
- :func:`get_alerts`               — persisted limit-breach alerts (3.4).
- :func:`get_backtest_summary`     — exceedance + Kupiec readout (3.1–3.3).

Sessions: each function opens its own session against the default database and
returns detached data, so callers never manage sessions. Tests (and any caller
with an isolated DB) may pass an explicit ``session=`` to reuse one instead.

Sign convention is the engine's throughout: IaR/CIaR and realised cost are in
**cost terms — positive = cost (bad)**.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Iterator

import pandas as pd
from sqlalchemy.orm import Session

from iar.db.models import IaRResult, Portfolio, SimulationRun
from iar.db.session import get_session, init_db
from iar.risk.alerts import check_run, classify_severity, load_alerts, load_limits
from iar.risk.backtest import run_backtest
from iar.risk.realised_cost import compute_realised_cost

__all__ = [
    "list_portfolios",
    "get_portfolio",
    "get_latest_iar",
    "get_iar_curve",
    "get_limit_status",
    "get_limit_overview",
    "get_intraday",
    "get_realised_intraday",
    "get_alerts",
    "get_backtest_summary",
]


def _latest_run(s: Session, portfolio_id: int) -> SimulationRun | None:
    """The portfolio's newest run (by run_ts then run_id), or ``None``."""
    return (
        s.query(SimulationRun)
        .filter_by(portfolio_id=portfolio_id)
        .order_by(SimulationRun.run_ts.desc(), SimulationRun.run_id.desc())
        .first()
    )


@contextmanager
def _scope(session: Session | None) -> Iterator[Session]:
    """Use the caller's session if given, else open one on the default DB."""
    if session is not None:
        yield session
    else:
        init_db()
        with get_session() as s:
            yield s


# --------------------------------------------------------------------------- #
# Portfolios
# --------------------------------------------------------------------------- #
def list_portfolios(*, session: Session | None = None) -> pd.DataFrame:
    """All portfolios as ``[portfolio_id, name, price_area]`` (by id)."""
    with _scope(session) as s:
        rows = s.query(Portfolio).order_by(Portfolio.portfolio_id).all()
        return pd.DataFrame(
            [{"portfolio_id": p.portfolio_id, "name": p.name, "price_area": p.price_area}
             for p in rows],
            columns=["portfolio_id", "name", "price_area"],
        )


def get_portfolio(
    area: str | None = None,
    portfolio_id: int | None = None,
    *,
    session: Session | None = None,
) -> dict | None:
    """Resolve one portfolio by ``portfolio_id`` or (latest in) ``area``.

    Returns ``{portfolio_id, name, price_area}`` or ``None`` if not found.
    """
    if portfolio_id is None and area is None:
        raise ValueError("pass either portfolio_id or area")
    with _scope(session) as s:
        q = s.query(Portfolio)
        if portfolio_id is not None:
            pf = q.filter_by(portfolio_id=portfolio_id).one_or_none()
        else:
            pf = (q.filter_by(price_area=area)
                  .order_by(Portfolio.portfolio_id.desc()).first())
        if pf is None:
            return None
        return {"portfolio_id": pf.portfolio_id, "name": pf.name, "price_area": pf.price_area}


# --------------------------------------------------------------------------- #
# IaR
# --------------------------------------------------------------------------- #
def get_latest_iar(portfolio_id: int, *, session: Session | None = None) -> dict | None:
    """Newest run's Gross/Spread IaR + CIaR for a portfolio (or ``None``).

    Shape::

        {
            "run_id", "run_ts", "vintage_ts", "horizon",
            "confidence", "n_scenarios",
            "gross":  {"iar", "ciar"},
            "spread": {"iar", "ciar"},
        }
    """
    with _scope(session) as s:
        run = (
            s.query(SimulationRun)
            .filter_by(portfolio_id=portfolio_id)
            .order_by(SimulationRun.run_ts.desc(), SimulationRun.run_id.desc())
            .first()
        )
        if run is None:
            return None
        by_type = {r.iar_type: r for r in run.results}

        def _pair(t: str) -> dict | None:
            r = by_type.get(t)
            return {"iar": r.iar_value, "ciar": r.ciar_value} if r else None

        any_result = next(iter(run.results), None)
        return {
            "portfolio_id": portfolio_id,
            "run_id": run.run_id,
            "run_ts": pd.to_datetime(run.run_ts, utc=True),
            "vintage_ts": pd.to_datetime(run.vintage_ts, utc=True),
            "horizon": any_result.horizon if any_result else None,
            "confidence": any_result.confidence if any_result else None,
            "n_scenarios": run.n_scenarios,
            "gross": _pair("gross"),
            "spread": _pair("spread"),
        }


def get_iar_curve(
    portfolio_id: int,
    iar_type: str = "gross",
    *,
    session: Session | None = None,
) -> pd.DataFrame:
    """IaR estimates over time for a portfolio + basis, one row per stored run.

    Columns ``[horizon, vintage_ts, run_ts, confidence, iar_value, ciar_value]``,
    ordered by ``vintage_ts`` — the time series the dashboard plots as the IaR
    curve / limit-tracking view.
    """
    if iar_type not in ("gross", "spread"):
        raise ValueError(f"iar_type must be 'gross' or 'spread', got {iar_type!r}")
    with _scope(session) as s:
        results = (
            s.query(IaRResult)
            .join(SimulationRun, IaRResult.run_id == SimulationRun.run_id)
            .filter(
                SimulationRun.portfolio_id == portfolio_id,
                IaRResult.iar_type == iar_type,
            )
            .order_by(SimulationRun.vintage_ts, SimulationRun.run_id)
            .all()
        )
        return pd.DataFrame(
            [
                {
                    "horizon": r.horizon,
                    "vintage_ts": pd.to_datetime(r.run.vintage_ts, utc=True),
                    "run_ts": pd.to_datetime(r.run.run_ts, utc=True),
                    "confidence": r.confidence,
                    "iar_value": r.iar_value,
                    "ciar_value": r.ciar_value,
                }
                for r in results
            ],
            columns=["horizon", "vintage_ts", "run_ts", "confidence", "iar_value", "ciar_value"],
        )


# --------------------------------------------------------------------------- #
# Alerts (3.4) and backtest (3.1–3.3)
# --------------------------------------------------------------------------- #
def get_limit_status(
    portfolio_id: int,
    limit_type: str = "remaining_day",
    *,
    session: Session | None = None,
) -> pd.DataFrame:
    """Latest run's IaR vs configured limits — current headroom + severity.

    Computed on read (no writes), so the dashboard can draw the IaR-vs-limit line
    and show 🟢/🟠/🔴 status without relying on persisted alerts. Columns
    ``[iar_type, limit_type, iar_value, limit_value, utilisation, severity]``;
    ``severity`` is ``None`` when within limit. Empty if no run or no limits config.
    """
    empty = pd.DataFrame(
        columns=["iar_type", "limit_type", "iar_value", "limit_value", "utilisation", "severity"]
    )
    with _scope(session) as s:
        run = (
            s.query(SimulationRun)
            .filter_by(portfolio_id=portfolio_id)
            .order_by(SimulationRun.run_ts.desc(), SimulationRun.run_id.desc())
            .first()
        )
        if run is None:
            return empty
        try:
            config = load_limits()
        except FileNotFoundError:
            return empty
        checks = check_run(run, config, limit_type)
        return pd.DataFrame(
            [
                {
                    "iar_type": c.iar_type,
                    "limit_type": c.limit_type,
                    "iar_value": c.iar_value,
                    "limit_value": c.limit_value,
                    "utilisation": c.utilisation,
                    "severity": c.severity,
                }
                for c in checks
            ],
            columns=["iar_type", "limit_type", "iar_value", "limit_value", "utilisation", "severity"],
        )


_INTRADAY_COLS = [
    "timestamp", "gross_iar", "gross_ciar", "spread_iar", "spread_ciar",
    "position_mwh", "expected_imbalance_mwh",
]
_LIMIT_OVERVIEW_COLS = [
    "iar_type", "limit_type", "iar_value", "limit_value", "utilisation", "severity",
]


def get_intraday(portfolio_id: int, *, session: Session | None = None) -> pd.DataFrame:
    """Per-MTU IaR series for the latest run (gross & spread), with positions.

    Columns ``[timestamp, gross_iar, gross_ciar, spread_iar, spread_ciar,
    position_mwh, expected_imbalance_mwh]``, one row per MTU — the series behind the
    dashboard's intraday bars and risk heatmap. Empty if the latest run stored no
    per-MTU detail (e.g. older runs predating the per-MTU read-off).
    """
    empty = pd.DataFrame(columns=_INTRADAY_COLS)
    with _scope(session) as s:
        run = _latest_run(s, portfolio_id)
        if run is None or not run.per_mtu_json:
            return empty
        data = json.loads(run.per_mtu_json)
        g, sp = data.get("gross", {}), data.get("spread", {})
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime(data["timestamps"], utc=True),
                "gross_iar": g.get("iar"),
                "gross_ciar": g.get("ciar"),
                "spread_iar": sp.get("iar"),
                "spread_ciar": sp.get("ciar"),
                "position_mwh": data.get("position_mwh"),
                "expected_imbalance_mwh": data.get("expected_imbalance_mwh"),
            },
            columns=_INTRADAY_COLS,
        )


def get_realised_intraday(
    portfolio_id: int,
    start=None,
    end=None,
    *,
    session: Session | None = None,
) -> pd.DataFrame:
    """Per-MTU **realised** imbalance cost over ``[start, end)`` (settled MTUs only).

    Columns ``[timestamp, realised_gross_cost, realised_spread_cost,
    imbalance_mwh]`` — the "what actually happened" half of the intraday view, used
    to fill elapsed MTUs so the heatmap is a full day (realised behind now, forecast
    ahead). Empty until realised prices have been loaded (``load_actuals.py``).
    """
    cols = ["timestamp", "realised_gross_cost", "realised_spread_cost", "imbalance_mwh"]
    with _scope(session) as s:
        df = compute_realised_cost(s, portfolio_id, start=start, end=end)
        if df.empty:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime(df["timestamp"], utc=True),
                "realised_gross_cost": df["gross_cost"],
                "realised_spread_cost": df["spread_cost"],
                "imbalance_mwh": df["imbalance_mwh"],
            },
            columns=cols,
        )


def get_limit_overview(portfolio_id: int, *, session: Session | None = None) -> pd.DataFrame:
    """Latest run's IaR vs limits across **all** limit types (day / rolling / per-MTU).

    Unlike :func:`get_limit_status` (one limit type, period IaR only), this compares
    the right current figure to each configured limit: the period IaR for
    ``remaining_day``, the rolling-window IaR for ``rolling_window`` and the peak-MTU
    IaR for ``per_mtu`` (the last two from the run's per-MTU detail). Columns
    ``[iar_type, limit_type, iar_value, limit_value, utilisation, severity]``; empty
    if no run or no limits config. Computed on read.
    """
    empty = pd.DataFrame(columns=_LIMIT_OVERVIEW_COLS)
    with _scope(session) as s:
        run = _latest_run(s, portfolio_id)
        if run is None:
            return empty
        try:
            config = load_limits()
        except FileNotFoundError:
            return empty
        period = {r.iar_type: r.iar_value for r in run.results}
        detail = json.loads(run.per_mtu_json) if run.per_mtu_json else {}
        name = run.portfolio.name
        rows = []
        for basis in ("gross", "spread"):
            block = detail.get(basis, {})
            current = {
                "remaining_day": period.get(basis),
                "rolling_window": block.get("rolling_iar"),
                "per_mtu": block.get("peak_iar"),
            }
            for limit_type in ("remaining_day", "rolling_window", "per_mtu"):
                cur = current[limit_type]
                limit = config.limit_for(name, basis, limit_type)
                if cur is None or limit is None:
                    continue
                rows.append(
                    {
                        "iar_type": basis,
                        "limit_type": limit_type,
                        "iar_value": cur,
                        "limit_value": limit,
                        "utilisation": (cur / limit if limit else float("nan")),
                        "severity": classify_severity(cur, limit),
                    }
                )
        return pd.DataFrame(rows, columns=_LIMIT_OVERVIEW_COLS)


def get_alerts(portfolio_id: int, *, session: Session | None = None) -> pd.DataFrame:
    """Persisted limit-breach alerts for a portfolio (newest first).

    Populated whenever a run is stored (``run_iar.py --store`` evaluates limits).
    For the *current* live status use :func:`get_limit_status` instead.
    """
    with _scope(session) as s:
        return load_alerts(s, portfolio_id)


def get_backtest_summary(
    portfolio_id: int,
    iar_type: str = "gross",
    *,
    significance: float = 0.05,
    session: Session | None = None,
) -> dict:
    """Exceedance + Kupiec calibration readout for a portfolio + basis.

    Computed on read (does not persist). Returns the scalar summary plus a
    ``"periods"`` DataFrame of per-period estimate-vs-realised outcomes::

        {... summary fields ..., "periods": DataFrame[period, iar_estimate,
                                                       realised_cost, exceeded]}
    """
    with _scope(session) as s:
        result = run_backtest(s, portfolio_id, iar_type, significance=significance, persist=False)
        summary = result.summary()
        summary["periods"] = result.as_frame()
        return summary
