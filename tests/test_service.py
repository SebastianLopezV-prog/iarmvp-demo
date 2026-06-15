"""Backend service-interface tests (Task 3.5).

Exercises the frozen read API with an injected in-memory session, asserting it
returns plain data (DataFrames/dicts) and hides ORM detail.
"""

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from sqlalchemy.orm import sessionmaker

from iar.db.models import (
    ActualDelivery,
    ActualImbalancePrice,
    DAMPosition,
    DAMPrice,
    IaRResult,
    SimulationRun,
)
from iar.db.session import init_db, make_engine
from iar.ingestion.flatfile_loader import get_or_create_portfolio
from iar.risk.alerts import LimitConfig, evaluate_run
from iar import service

D1 = datetime(2026, 6, 8, tzinfo=timezone.utc)
D2 = datetime(2026, 6, 9, tzinfo=timezone.utc)


@pytest.fixture
def session():
    engine = make_engine(":memory:")
    init_db(engine)
    with sessionmaker(bind=engine, expire_on_commit=False, future=True)() as s:
        yield s


def _run(session, pid, day, gross_iar, spread_iar, *, replay=True):
    vintage = day - timedelta(hours=12)
    cfg = {"replay": replay,
           "period_start": day.isoformat(),
           "period_end": (day + timedelta(days=1)).isoformat(),
           "n_mtus": 1}
    run = SimulationRun(portfolio_id=pid, run_ts=vintage, vintage_ts=vintage,
                        n_scenarios=5000, seed=42, config_json=json.dumps(cfg))
    session.add(run)
    session.flush()
    h = day.date().isoformat()
    session.add(IaRResult(run_id=run.run_id, confidence=0.95, horizon=h,
                          iar_type="gross", iar_value=gross_iar, ciar_value=gross_iar + 100))
    session.add(IaRResult(run_id=run.run_id, confidence=0.95, horizon=h,
                          iar_type="spread", iar_value=spread_iar, ciar_value=spread_iar + 50))
    session.flush()
    return run


# --------------------------------------------------------------------------- #
# Portfolios
# --------------------------------------------------------------------------- #
def test_list_and_get_portfolio(session):
    pf = get_or_create_portfolio(session, "Wind Co", "NO2 Wind", "NO2")
    session.flush()
    df = service.list_portfolios(session=session)
    assert list(df.columns) == ["portfolio_id", "name", "price_area"]
    assert df.iloc[0]["name"] == "NO2 Wind"

    by_id = service.get_portfolio(portfolio_id=pf.portfolio_id, session=session)
    by_area = service.get_portfolio(area="NO2", session=session)
    assert by_id["price_area"] == "NO2"
    assert by_area["portfolio_id"] == pf.portfolio_id
    assert service.get_portfolio(area="SE3", session=session) is None


def test_get_portfolio_requires_an_argument(session):
    with pytest.raises(ValueError, match="portfolio_id or area"):
        service.get_portfolio(session=session)


# --------------------------------------------------------------------------- #
# IaR
# --------------------------------------------------------------------------- #
def test_get_latest_iar_returns_newest_run(session):
    pf = get_or_create_portfolio(session, "Wind Co", "NO2 Wind", "NO2")
    _run(session, pf.portfolio_id, D1, gross_iar=1000, spread_iar=600)
    _run(session, pf.portfolio_id, D2, gross_iar=2000, spread_iar=900)  # newer
    session.commit()

    latest = service.get_latest_iar(pf.portfolio_id, session=session)
    assert latest["horizon"] == "2026-06-09"
    assert latest["gross"] == {"iar": 2000, "ciar": 2100}
    assert latest["spread"]["iar"] == 900
    assert latest["confidence"] == 0.95
    assert isinstance(latest["vintage_ts"], pd.Timestamp)


def test_get_latest_iar_none_when_empty(session):
    pf = get_or_create_portfolio(session, "Wind Co", "NO2 Wind", "NO2")
    assert service.get_latest_iar(pf.portfolio_id, session=session) is None


def test_get_iar_curve_orders_by_vintage(session):
    pf = get_or_create_portfolio(session, "Wind Co", "NO2 Wind", "NO2")
    _run(session, pf.portfolio_id, D2, gross_iar=2000, spread_iar=900)
    _run(session, pf.portfolio_id, D1, gross_iar=1000, spread_iar=600)
    session.commit()

    curve = service.get_iar_curve(pf.portfolio_id, "gross", session=session)
    assert curve["iar_value"].tolist() == [1000, 2000]  # D1 then D2 by vintage
    assert curve["horizon"].tolist() == ["2026-06-08", "2026-06-09"]
    assert (curve["vintage_ts"].diff().dropna() > pd.Timedelta(0)).all()


def test_get_iar_curve_bad_type_raises(session):
    pf = get_or_create_portfolio(session, "Wind Co", "NO2 Wind", "NO2")
    with pytest.raises(ValueError, match="gross.*spread"):
        service.get_iar_curve(pf.portfolio_id, "net", session=session)


# --------------------------------------------------------------------------- #
# Alerts
# --------------------------------------------------------------------------- #
def test_get_alerts_returns_persisted_breaches(session):
    pf = get_or_create_portfolio(session, "Wind Co", "NO2 Wind", "NO2")
    run = _run(session, pf.portfolio_id, D1, gross_iar=60000, spread_iar=0)
    cfg = LimitConfig({"default": {"gross": {"remaining_day_eur": 50000}}})
    evaluate_run(session, run, cfg)
    session.commit()

    df = service.get_alerts(pf.portfolio_id, session=session)
    assert len(df) == 1
    assert df.iloc[0]["severity"] == "hard"
    assert df.iloc[0]["iar_type"] == "gross"


# --------------------------------------------------------------------------- #
# Limit status
# --------------------------------------------------------------------------- #
def test_get_limit_status_reports_severity_and_headroom(session):
    pf = get_or_create_portfolio(session, "Wind Co", "NO2 Wind", "NO2")
    # Gross 4800 breaches the default 4000 remaining-day limit; spread 1000 is within 2800.
    _run(session, pf.portfolio_id, D1, gross_iar=4800, spread_iar=1000)
    session.commit()
    df = service.get_limit_status(pf.portfolio_id, session=session)
    by_type = df.set_index("iar_type")
    assert by_type.loc["gross", "severity"] == "hard"
    assert by_type.loc["gross", "limit_value"] == 4000
    assert by_type.loc["gross", "utilisation"] == pytest.approx(1.2)
    assert pd.isna(by_type.loc["spread", "severity"])  # within limit -> no severity


def test_get_limit_status_empty_without_runs(session):
    pf = get_or_create_portfolio(session, "Wind Co", "NO2 Wind", "NO2")
    df = service.get_limit_status(pf.portfolio_id, session=session)
    assert df.empty
    assert list(df.columns) == [
        "iar_type", "limit_type", "iar_value", "limit_value", "utilisation", "severity"
    ]


# --------------------------------------------------------------------------- #
# Backtest summary
# --------------------------------------------------------------------------- #
def test_get_backtest_summary_shape_and_values(session):
    pf = get_or_create_portfolio(session, "Wind Co", "NO2 Wind", "NO2")
    pid = pf.portfolio_id
    _run(session, pid, D1, gross_iar=500, spread_iar=300)
    _run(session, pid, D2, gross_iar=500, spread_iar=300)
    # D1 realised gross = 20*100 = 2000 > 500 -> exceeded; D2 = 1*100 = 100 < 500.
    for day, pos in ((D1, 20.0), (D2, 1.0)):
        session.add(DAMPosition(portfolio_id=pid, timestamp=day, mwh=pos))
        session.add(ActualDelivery(portfolio_id=pid, timestamp=day, actual_mwh=0.0))
        session.add(ActualImbalancePrice(price_area="NO2", timestamp=day, price=100.0))
        session.add(DAMPrice(price_area="NO2", timestamp=day, price=40.0))
    session.commit()

    summary = service.get_backtest_summary(pid, "gross", session=session)
    assert summary["n_periods"] == 2
    assert summary["n_exceedances"] == 1
    assert summary["iar_type"] == "gross"
    assert isinstance(summary["periods"], pd.DataFrame)
    assert set(summary["periods"]["period"]) == {"2026-06-08", "2026-06-09"}
    # read-only: nothing persisted
    from iar.db.models import HistoricalPerformanceRecord
    assert session.query(HistoricalPerformanceRecord).count() == 0
