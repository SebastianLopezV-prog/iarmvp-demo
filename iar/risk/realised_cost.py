"""Realised imbalance cost (Task 3.1).

The "what actually happened" counterpart to the simulated IaR. For each settled
MTU we have, after the fact:

- the **DAM position** the portfolio sold (MWh),
- the **actual delivery** that was metered (MWh),
- the **actual imbalance price** the TSO published (EUR/MWh, absolute),
- the **DAM (spot) price** for that MTU (EUR/MWh).

From these the realised settlement cost is reconstructed on the *same two bases*
the engine simulates, so the Week-3 backtest (3.3) can compare like-for-like:

    imbalance_t   = dam_position_t − actual_delivery_t          (MWh; >0 = short)
    gross_cost_t  = imbalance_t × actual_imbalance_price_t       (EUR)
    spread_cost_t = imbalance_t × (actual_imbalance_price_t − dam_price_t)  (EUR)

Sign convention matches :mod:`iar.simulation.engine`: **positive = net cost
(bad), negative = net revenue (good)**. A short position (imbalance > 0) settled
at a positive price costs money; a long position earns it; negative prices flip
both.

Period vs per-MTU. :func:`compute_realised_cost` returns the per-MTU series;
:func:`realised_period_cost` sums it over the window to give the single period
figure that is compared against the period IaR estimate. (IaR is a quantile of a
*distribution*; the realised cost is one draw from the world — the comparison is
"did the realised period cost exceed the estimate?".)

No new table. Realised cost is **derived on demand** from the stored inputs
(``dam_positions``, ``actual_deliveries``, ``actual_imbalance_prices``,
``dam_prices``) — consistent with the architecture's "store summaries, not
intermediate series" decision. Task 3.3 persists the per-period comparison into
``HistoricalPerformanceRecord``.

The series spans the MTUs where **all four** inputs exist (their intersection) —
the same alignment idiom ``scripts/run_iar.py`` uses across forecast/DAM/position.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from sqlalchemy.orm import Session

from iar.db.models import (
    ActualDelivery,
    ActualImbalancePrice,
    DAMPosition,
    DAMPrice,
    Portfolio,
)

# Columns of the per-MTU realised-cost frame (stable public contract).
COLUMNS = [
    "timestamp",
    "dam_position_mwh",
    "actual_mwh",
    "imbalance_mwh",
    "dam_price",
    "actual_imbalance_price",
    "spread",
    "gross_cost",
    "spread_cost",
]


def _ts_map(rows, value_attr: str) -> dict[pd.Timestamp, float]:
    """Map UTC-normalised timestamp -> value for a set of ORM rows.

    SQLite hands back naive datetimes; normalise to tz-aware UTC so series from
    different tables (and from the live API) align on identical keys.
    """
    return {
        pd.to_datetime(r.timestamp, utc=True): float(getattr(r, value_attr))
        for r in rows
    }


def compute_realised_cost(
    session: Session,
    portfolio_id: int,
    start: datetime | str | None = None,
    end: datetime | str | None = None,
) -> pd.DataFrame:
    """Per-MTU realised imbalance cost for ``portfolio_id`` (gross + spread).

    Parameters
    ----------
    session:
        Active SQLAlchemy session.
    portfolio_id:
        Portfolio whose positions/actuals to use (must exist).
    start, end:
        Optional half-open window ``[start, end)`` on the MTU timestamp. Accept
        tz-aware datetimes or ISO 8601 strings; naive values are treated as UTC.
        ``None`` means unbounded on that side.

    Returns
    -------
    pandas.DataFrame
        One row per settled MTU (sorted by timestamp) with the columns in
        :data:`COLUMNS`. Empty (with those columns) if there is no overlap.
    """
    portfolio = session.get(Portfolio, portfolio_id)
    if portfolio is None:
        raise ValueError(f"Portfolio id {portfolio_id} does not exist.")
    area = portfolio.price_area

    dam_pos = _ts_map(
        session.query(DAMPosition).filter_by(portfolio_id=portfolio_id), "mwh"
    )
    actual = _ts_map(
        session.query(ActualDelivery).filter_by(portfolio_id=portfolio_id), "actual_mwh"
    )
    imb_price = _ts_map(
        session.query(ActualImbalancePrice).filter_by(price_area=area), "price"
    )
    dam_price = _ts_map(session.query(DAMPrice).filter_by(price_area=area), "price")

    # Intersection: an MTU is settled only if every input is present.
    timestamps = set(dam_pos) & set(actual) & set(imb_price) & set(dam_price)

    lo = pd.to_datetime(start, utc=True) if start is not None else None
    hi = pd.to_datetime(end, utc=True) if end is not None else None
    timestamps = {
        t for t in timestamps
        if (lo is None or t >= lo) and (hi is None or t < hi)
    }

    rows = []
    for t in sorted(timestamps):
        pos = dam_pos[t]
        act = actual[t]
        lam = imb_price[t]   # absolute imbalance price
        spot = dam_price[t]
        imbalance = pos - act           # MWh; >0 = short
        spread = lam - spot             # imbalance price spread vs day-ahead
        rows.append(
            {
                "timestamp": t,
                "dam_position_mwh": pos,
                "actual_mwh": act,
                "imbalance_mwh": imbalance,
                "dam_price": spot,
                "actual_imbalance_price": lam,
                "spread": spread,
                "gross_cost": imbalance * lam,
                "spread_cost": imbalance * spread,
            }
        )

    return pd.DataFrame(rows, columns=COLUMNS)


def realised_period_cost(
    session: Session,
    portfolio_id: int,
    start: datetime | str | None = None,
    end: datetime | str | None = None,
) -> dict:
    """Sum the per-MTU realised cost over the window into one period figure.

    Returns a dict with the realised ``gross`` and ``spread`` period costs (EUR),
    the number of MTUs covered, and the actual time span of the data — the shape
    Task 3.3 joins against each period's IaR estimate.
    """
    df = compute_realised_cost(session, portfolio_id, start=start, end=end)
    if df.empty:
        return {
            "gross": 0.0,
            "spread": 0.0,
            "n_mtus": 0,
            "first_mtu": None,
            "last_mtu": None,
        }
    return {
        "gross": float(df["gross_cost"].sum()),
        "spread": float(df["spread_cost"].sum()),
        "n_mtus": int(len(df)),
        "first_mtu": df["timestamp"].iloc[0].to_pydatetime(),
        "last_mtu": df["timestamp"].iloc[-1].to_pydatetime(),
    }
