"""Flat-file loader for portfolio inputs (Task 1.4).

Reads CSV/Excel files for DAM positions, generation forecasts, and actual
delivery; validates the schema; parses timestamps to timezone-aware UTC at MTU
resolution; and writes rows into the SQLite tables.

Expected file format (one row per MTU)
--------------------------------------
- DAM positions:        columns ``timestamp``, ``mwh``
- Generation forecasts: columns ``timestamp``, ``forecast_mwh``
- Actual delivery:      columns ``timestamp``, ``actual_mwh``

``timestamp`` is the MTU *start*, ISO 8601, parsed to UTC. Values are MWh per MTU.
See ``docs/data_contract.md`` for the full contract.

Loads are **idempotent**: by default each loader replaces the existing rows for
that portfolio in that table, so re-running the pipeline does not duplicate data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Type

import pandas as pd
from sqlalchemy.orm import Session

from iar.db.models import (
    ActualDelivery,
    DAMPosition,
    DAMPrice,
    GenerationForecast,
    Portfolio,
    User,
)


class FileValidationError(ValueError):
    """Raised when an input file is missing, malformed, or fails validation."""


# --------------------------------------------------------------------------- #
# Portfolio helper
# --------------------------------------------------------------------------- #
def get_or_create_portfolio(
    session: Session,
    user_name: str,
    portfolio_name: str,
    price_area: str,
) -> Portfolio:
    """Return the portfolio for ``user_name``, creating user+portfolio if absent."""
    user = session.query(User).filter_by(name=user_name).one_or_none()
    if user is None:
        user = User(name=user_name)
        session.add(user)
        session.flush()
    portfolio = session.query(Portfolio).filter_by(user_id=user.user_id).one_or_none()
    if portfolio is None:
        portfolio = Portfolio(
            name=portfolio_name, price_area=price_area, user_id=user.user_id
        )
        session.add(portfolio)
        session.flush()
    return portfolio


# --------------------------------------------------------------------------- #
# Reading + validation
# --------------------------------------------------------------------------- #
def _read_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileValidationError(f"Input file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise FileValidationError(
            f"Unsupported file type '{suffix}' (expected .csv/.xlsx): {path}"
        )
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _validate(df: pd.DataFrame, value_col: str, path: Path) -> pd.DataFrame:
    """Validate required columns, parse timestamps to UTC, sanity-check values."""
    missing = {"timestamp", value_col} - set(df.columns)
    if missing:
        raise FileValidationError(
            f"{path.name}: missing required column(s) {sorted(missing)}; "
            f"found {list(df.columns)}."
        )

    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if ts.isna().any():
        bad = df.loc[ts.isna(), "timestamp"].head(3).tolist()
        raise FileValidationError(f"{path.name}: unparseable timestamp(s), e.g. {bad}.")
    if ts.duplicated().any():
        dupes = ts[ts.duplicated()].head(3).astype(str).tolist()
        raise FileValidationError(f"{path.name}: duplicate timestamp(s), e.g. {dupes}.")

    vals = pd.to_numeric(df[value_col], errors="coerce")
    if vals.isna().any():
        raise FileValidationError(f"{path.name}: non-numeric/empty values in '{value_col}'.")

    out = pd.DataFrame({"timestamp": ts, value_col: vals.astype(float)})
    return out.sort_values("timestamp").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Generic loader + typed wrappers
# --------------------------------------------------------------------------- #
def _load_timeseries(
    session: Session,
    portfolio_id: int,
    path: str | Path,
    value_col: str,
    model: Type,
    replace: bool,
) -> int:
    """Load a single-value time series file into ``model`` for ``portfolio_id``."""
    if session.get(Portfolio, portfolio_id) is None:
        raise FileValidationError(f"Portfolio id {portfolio_id} does not exist.")

    path = Path(path)
    df = _validate(_read_file(path), value_col, path)

    if replace:
        session.query(model).filter(model.portfolio_id == portfolio_id).delete()

    rows = [
        model(
            portfolio_id=portfolio_id,
            timestamp=ts.to_pydatetime(),
            **{value_col: float(val)},
        )
        for ts, val in zip(df["timestamp"], df[value_col])
    ]
    session.add_all(rows)
    session.flush()
    return len(rows)


def load_dam_positions(session, portfolio_id, path, replace: bool = True) -> int:
    """Load DAM positions (``timestamp``, ``mwh``)."""
    return _load_timeseries(session, portfolio_id, path, "mwh", DAMPosition, replace)


def load_generation_forecasts(session, portfolio_id, path, replace: bool = True) -> int:
    """Load generation forecasts (``timestamp``, ``forecast_mwh``)."""
    return _load_timeseries(
        session, portfolio_id, path, "forecast_mwh", GenerationForecast, replace
    )


def load_actual_delivery(session, portfolio_id, path, replace: bool = True) -> int:
    """Load actual delivery (``timestamp``, ``actual_mwh``)."""
    return _load_timeseries(
        session, portfolio_id, path, "actual_mwh", ActualDelivery, replace
    )


def load_dam_prices(
    session: Session,
    price_area: str,
    path: str | Path,
    replace: bool = True,
) -> int:
    """Load day-ahead (spot) prices (``timestamp``, ``eur_per_mwh``) for an area.

    Unlike the position/generation/actual series (which are per *portfolio*), the
    DAM price is a per *price-area* market series, so it is keyed by ``price_area``
    rather than ``portfolio_id`` — matching the other price tables. Needed to turn
    Optimeering's imbalance *spread* into an absolute price for Gross IaR.

    .. warning::

        TODO(dam-source): **STUB DATA SOURCE — NOT A REAL DAM PRICE FEED.**
        Today this is fed by ``scripts/make_sample_data.py``, which writes a
        *synthetic* diurnal price curve to ``dam_price_*.csv``. Those numbers are
        invented and must NOT be treated as real market prices — any Gross IaR
        computed from them is illustrative only.

        Optimeering does NOT publish NO2 day-ahead/spot (confirmed: the whole NO2
        catalogue is balancing-market only), so a real source must be wired in
        before Gross IaR is trustworthy. Likely candidates: a Volue Data &
        Forecasts / PowerTSM curve, or ENTSO-E / Nord Pool. Open question for
        Erik M re: whether Optimeering can provide/derive spot on our subscription.

        This table is source-agnostic: a real feed just needs to write
        (price_area, timestamp, EUR/MWh) rows here — no downstream change required.
    """
    path = Path(path)
    df = _validate(_read_file(path), "eur_per_mwh", path)

    if replace:
        session.query(DAMPrice).filter(DAMPrice.price_area == price_area).delete()

    rows = [
        DAMPrice(price_area=price_area, timestamp=ts.to_pydatetime(), price=float(val))
        for ts, val in zip(df["timestamp"], df["eur_per_mwh"])
    ]
    session.add_all(rows)
    session.flush()
    return len(rows)
