"""Backtesting: the vintage comparison join (Task 3.2) and calibration (Task 3.3).

Task 3.1 (realised imbalance cost) lives in :mod:`iar.risk.realised_cost`; the
estimate backfill (the other half of 3.2) lives in :mod:`iar.risk.replay`.

This module defines the **comparison join** that ties the two together: given a
settled period, which stored IaR estimate should it be judged against? Per the
architecture's vintage rule, the answer is *the most recent estimate whose
``vintage_ts`` is at or before the period start* — i.e. an estimate that used no
information from inside the period (no look-ahead).

Task 3.3 (still to come) will build on this join: walk each settled period,
compare its realised cost (3.1) to the joined estimate, flag exceedances, and run
a Kupiec POF test for ~5% calibration at P95.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from sqlalchemy.orm import Session

from iar.db.models import IaRResult, SimulationRun


def estimate_for_period(
    session: Session,
    portfolio_id: int,
    period_start: datetime | str,
) -> SimulationRun | None:
    """Return the IaR estimate applicable to a period starting at ``period_start``.

    The chosen run is the one for ``portfolio_id`` with the **latest
    ``vintage_ts`` at or before ``period_start``** — the estimate that was current
    just before the period began. Returns ``None`` if no estimate precedes it.

    ``period_start`` is normalised to UTC; estimates are stored with UTC vintages,
    so the comparison is timezone-consistent.
    """
    ps = pd.to_datetime(period_start, utc=True).to_pydatetime()
    return (
        session.query(SimulationRun)
        .filter(
            SimulationRun.portfolio_id == portfolio_id,
            SimulationRun.vintage_ts <= ps,
        )
        .order_by(SimulationRun.vintage_ts.desc())
        .first()
    )


def iar_estimate_for_period(
    session: Session,
    portfolio_id: int,
    period_start: datetime | str,
    iar_type: str,
) -> IaRResult | None:
    """The ``gross``/``spread`` :class:`IaRResult` joined to ``period_start``.

    Convenience over :func:`estimate_for_period` that drills into the run and
    returns the single result of the requested ``iar_type`` (or ``None`` if no
    estimate precedes the period).
    """
    if iar_type not in ("gross", "spread"):
        raise ValueError(f"iar_type must be 'gross' or 'spread', got {iar_type!r}")
    run = estimate_for_period(session, portfolio_id, period_start)
    if run is None:
        return None
    for result in run.results:
        if result.iar_type == iar_type:
            return result
    return None
