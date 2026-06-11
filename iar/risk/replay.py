"""IaR vintage replay / backfill (Task 3.2).

Backtesting compares the IaR estimate that *would have been made at the time*
against the cost that actually occurred. That requires a history of stored
estimates, each stamped with the "as-of" time of its inputs (``vintage_ts``).
This module generates that history.

The replay model (MVP)
----------------------
- The **period** is a delivery day (UTC midnight to midnight) — the deck's
  "next trading day" horizon, the natural unit for daily limit governance.
- For each delivery day ``D`` we pick the **day-ahead vintage**: the most recent
  forecast whose event_time is at or before ``D``'s start. That is exactly the
  information set you'd have had before the day began — no look-ahead.
- We build the engine inputs for ``D``'s MTUs from that vintage's spread
  quantiles, the DAM (spot) prices, and the portfolio's positions/generation,
  run the Monte Carlo, and persist one ``SimulationRun`` (``vintage_ts`` = the
  chosen vintage, ``horizon`` = ``D`` as ISO date) plus its ``gross``/``spread``
  ``IaRResult`` rows.

The comparison join itself lives in :mod:`iar.risk.backtest`
(``estimate_for_period``): for a settled period it returns the estimate whose
vintage precedes it. Because vintages are monotonic and each day uses the latest
vintage before its start, "latest vintage ≤ period start" resolves to that day's
day-ahead estimate.

Heavy-window guard
------------------
Optimeering's historical ``retrieve`` returns *every* vintage in the window
(huge). This module operates on already-fetched, **grouped-by-vintage** records
and selects one vintage per day, so it never holds the full cross-product. The
fetching script is responsible for requesting a sane window.

Decoupled from the API: the core :func:`backfill_iar` takes plain data
structures (forecast records, price/position maps), mirroring how the rest of the
codebase keeps compute pure and pushes I/O to the edges. ``scripts/backfill_iar.py``
wires the real sources.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from iar.db.models import Portfolio, SimulationRun
from iar.simulation.engine import EngineConfig, IaRReport, run_simulation
from iar.simulation.imbalance_model import ImbalanceModel, ImbalanceModelConfig
from iar.simulation.persistence import persist_report
from iar.simulation.price_sampler import QuantilePriceSampler

# windsim/Optimeering quantities are MWh per 15-minute MTU.
MTU_HOURS = 0.25
DEFAULT_CAPACITY_MWH = 25.0  # 100 MW × 0.25 h — matches scripts/run_iar.py's stub basis


def _group_by_vintage(
    forecast_records: list[dict],
) -> dict[pd.Timestamp, dict[pd.Timestamp, dict[float, float]]]:
    """Group normalised forecast records into ``{vintage: {mtu_ts: {quantile: value}}}``.

    Records without a quantile (mean/distribution rows) or without a vintage are
    skipped — replay needs the quantile curve to sample the spread.
    """
    by_vintage: dict = defaultdict(lambda: defaultdict(dict))
    for r in forecast_records:
        q = r.get("quantile")
        v = r.get("vintage_ts")
        if q is None or v is None:
            continue
        vintage = pd.to_datetime(v, utc=True)
        ts = pd.to_datetime(r["timestamp"], utc=True)
        by_vintage[vintage][ts][float(q)] = float(r["value"])
    return by_vintage


def _delivery_days(by_vintage: dict) -> list[pd.Timestamp]:
    """Sorted unique delivery-day starts (UTC midnight) across all forecast MTUs."""
    days = set()
    for ts_map in by_vintage.values():
        for ts in ts_map:
            days.add(ts.normalize())
    return sorted(days)


def _pick_vintage(by_vintage: dict, day_start: pd.Timestamp, day_end: pd.Timestamp):
    """Most recent vintage at/before ``day_start`` that forecasts MTUs inside the day."""
    candidates = [
        v for v, ts_map in by_vintage.items()
        if v <= day_start and any(day_start <= ts < day_end for ts in ts_map)
    ]
    return max(candidates) if candidates else None


@dataclass
class PeriodEstimate:
    """One day-ahead IaR estimate: the period, its chosen vintage, and the report.

    Also carries the per-MTU inputs (``timestamps`` ISO strings, ``dam_pos``, ``gen``)
    so the backfill can persist the same per-MTU read-off the live run does — i.e. a
    backfilled *complete* delivery day yields a full-day intraday/heatmap.
    """

    day_start: pd.Timestamp
    day_end: pd.Timestamp
    horizon: str          # delivery day, ISO date (e.g. "2026-06-09")
    vintage: pd.Timestamp  # the day-ahead forecast event_time used
    n_mtus: int
    report: IaRReport
    timestamps: list[str]
    dam_pos: np.ndarray
    gen: np.ndarray


def estimate_periods(
    *,
    forecast_records: list[dict],
    dam_price_map: dict,
    position_map: dict,
    capacity_mwh: float = DEFAULT_CAPACITY_MWH,
    model_config: ImbalanceModelConfig | None = None,
    engine_config: EngineConfig | None = None,
) -> list[PeriodEstimate]:
    """Compute one day-ahead IaR estimate per delivery day — **pure, no DB**.

    Groups the forecast records by vintage, picks each day's day-ahead vintage,
    builds the engine inputs over that day's MTUs, and runs the Monte Carlo.
    Shared by :func:`backfill_iar` (which persists the reports) and the sigma
    calibration loop (which only reads the IaR values), so both stay consistent.
    """
    model_config = model_config or ImbalanceModelConfig()
    engine_config = engine_config or EngineConfig()

    # Normalise the lookup maps to UTC keys once.
    dam_by_ts = {pd.to_datetime(t, utc=True): float(p) for t, p in dam_price_map.items()}
    pos_by_ts = {
        pd.to_datetime(t, utc=True): (float(d), float(g))
        for t, (d, g) in position_map.items()
    }

    by_vintage = _group_by_vintage(forecast_records)
    estimates: list[PeriodEstimate] = []
    for day_start in _delivery_days(by_vintage):
        day_end = day_start + timedelta(days=1)
        vintage = _pick_vintage(by_vintage, day_start, day_end)
        if vintage is None:
            continue  # no information set preceding this day — cannot replay it

        ts_quantiles = by_vintage[vintage]
        # MTUs inside the day that we have a forecast, a DAM price, AND positions for.
        kept = sorted(
            ts for ts in ts_quantiles
            if day_start <= ts < day_end and ts in dam_by_ts and ts in pos_by_ts
        )
        if not kept:
            continue
        # Quantile levels common to every kept MTU.
        pct = sorted(set.intersection(*(set(ts_quantiles[ts]) for ts in kept)))
        if len(pct) < 2:
            continue  # need at least two levels to interpolate a marginal

        pct_arr = np.array(pct, dtype=float)
        spread = np.array([[ts_quantiles[ts][q] for q in pct] for ts in kept])
        dam_price = np.array([dam_by_ts[ts] for ts in kept])
        dam_pos = np.array([pos_by_ts[ts][0] for ts in kept])
        gen = np.array([pos_by_ts[ts][1] for ts in kept])

        price = QuantilePriceSampler.from_percentiles(pct_arr, spread)
        imb = ImbalanceModel.from_inputs(
            dam_pos, gen, capacity_mwh=capacity_mwh, config=model_config
        )
        report = run_simulation(price, imb, dam_price, engine_config)
        estimates.append(
            PeriodEstimate(
                day_start=day_start,
                day_end=day_end,
                horizon=day_start.date().isoformat(),
                vintage=vintage,
                n_mtus=len(kept),
                report=report,
            )
        )
    return estimates


def backfill_iar(
    session: Session,
    portfolio_id: int,
    *,
    forecast_records: list[dict],
    dam_price_map: dict,
    position_map: dict,
    capacity_mwh: float = DEFAULT_CAPACITY_MWH,
    model_config: ImbalanceModelConfig | None = None,
    engine_config: EngineConfig | None = None,
    replace: bool = True,
) -> list[SimulationRun]:
    """Backfill day-ahead IaR estimates over the window covered by the inputs.

    Thin DB wrapper over :func:`estimate_periods`: persists each period's report
    as a ``SimulationRun`` (stamped with the day-ahead ``vintage_ts``) plus its
    ``gross``/``spread`` ``IaRResult`` rows.

    Parameters
    ----------
    session:
        Active SQLAlchemy session (caller commits).
    portfolio_id:
        Portfolio to attach the runs to (must exist).
    forecast_records:
        Normalised imbalance-spread forecast records spanning multiple vintages
        (e.g. from ``OptimeeringForecastClient.get_historical_prices``). Each dict
        has ``vintage_ts``, ``timestamp``, ``quantile``, ``value``.
    dam_price_map:
        ``{timestamp -> dam_price}`` (EUR/MWh); tz-aware or ISO, normalised to UTC.
    position_map:
        ``{timestamp -> (dam_position_mwh, forecast_generation_mwh)}``.
    capacity_mwh:
        Per-MTU installed-capacity energy used as the imbalance sigma basis.
    model_config, engine_config:
        Imbalance-model and engine settings; sensible defaults if omitted.
    replace:
        If True (default), delete any existing run for the same (portfolio, day)
        before writing — makes backfill idempotent.

    Returns
    -------
    list[SimulationRun]
        One persisted run per delivery day that had a usable day-ahead vintage.
    """
    if session.get(Portfolio, portfolio_id) is None:
        raise ValueError(f"Portfolio id {portfolio_id} does not exist.")

    estimates = estimate_periods(
        forecast_records=forecast_records,
        dam_price_map=dam_price_map,
        position_map=position_map,
        capacity_mwh=capacity_mwh,
        model_config=model_config,
        engine_config=engine_config,
    )

    runs: list[SimulationRun] = []
    for est in estimates:
        if replace:
            existing = (
                session.query(SimulationRun)
                .filter(
                    SimulationRun.portfolio_id == portfolio_id,
                    SimulationRun.vintage_ts == est.vintage.to_pydatetime(),
                )
                .all()
            )
            for run in existing:
                if run.results and run.results[0].horizon == est.horizon:
                    session.delete(run)
            session.flush()

        run = persist_report(
            session,
            est.report,
            portfolio_id,
            vintage_ts=est.vintage.to_pydatetime(),
            horizon=est.horizon,
            extra_config={
                "replay": True,
                "period_start": est.day_start.isoformat(),
                "period_end": est.day_end.isoformat(),
                "n_mtus": est.n_mtus,
            },
        )
        runs.append(run)

    return runs
