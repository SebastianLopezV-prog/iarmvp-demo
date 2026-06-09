"""IaR vintage replay + comparison join tests (Task 3.2)."""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.orm import sessionmaker

from iar.db.models import IaRResult, SimulationRun
from iar.db.session import init_db, make_engine
from iar.ingestion.flatfile_loader import get_or_create_portfolio
from iar.risk.backtest import estimate_for_period, iar_estimate_for_period
from iar.risk.replay import backfill_iar
from iar.simulation.engine import EngineConfig

# Two delivery days and their day-ahead vintages (noon the day before).
D1 = datetime(2026, 6, 8, tzinfo=timezone.utc)
D2 = datetime(2026, 6, 9, tzinfo=timezone.utc)
V0 = datetime(2026, 6, 7, 12, tzinfo=timezone.utc)   # day-ahead for D1
V1 = datetime(2026, 6, 8, 12, tzinfo=timezone.utc)   # day-ahead for D2

FAST = EngineConfig(n_scenarios=2000, confidence=0.95, seed=42)
PCT = [5.0, 50.0, 95.0]
SPREAD = {5.0: -20.0, 50.0: 0.0, 95.0: 30.0}


@pytest.fixture
def session():
    engine = make_engine(":memory:")
    init_db(engine)
    with sessionmaker(bind=engine, expire_on_commit=False, future=True)() as s:
        yield s


def _day_mtus(day_start: datetime, n: int = 4):
    return [day_start + timedelta(minutes=15 * i) for i in range(n)]


def _forecast_records(vintage, day_start, n=4):
    """Spread-quantile records for ``n`` MTUs of a day, under one vintage."""
    recs = []
    for ts in _day_mtus(day_start, n):
        for q in PCT:
            recs.append(
                {
                    "vintage_ts": vintage.isoformat(),
                    "timestamp": ts.isoformat(),
                    "quantile": q,
                    "value": SPREAD[q],
                }
            )
    return recs


def _maps(*day_starts, n=4, dam=40.0, pos=1.0, gen=0.0):
    dam_map, pos_map = {}, {}
    for d in day_starts:
        for ts in _day_mtus(d, n):
            dam_map[ts] = dam
            pos_map[ts] = (pos, gen)
    return dam_map, pos_map


def _two_day_setup(session):
    pf = get_or_create_portfolio(session, "Wind Co", "NO2 Wind", "NO2")
    records = _forecast_records(V0, D1) + _forecast_records(V1, D2)
    dam_map, pos_map = _maps(D1, D2)
    runs = backfill_iar(
        session, pf.portfolio_id,
        forecast_records=records, dam_price_map=dam_map, position_map=pos_map,
        engine_config=FAST,
    )
    session.commit()
    return pf, runs


# --------------------------------------------------------------------------- #
# Backfill
# --------------------------------------------------------------------------- #
def test_backfill_creates_one_run_per_day(session):
    pf, runs = _two_day_setup(session)
    assert len(runs) == 2
    assert session.query(SimulationRun).count() == 2
    assert session.query(IaRResult).count() == 4  # gross + spread per run


def test_backfill_stamps_dayahead_vintage_and_horizon(session):
    _pf, runs = _two_day_setup(session)
    by_horizon = {r.results[0].horizon: r for r in runs}
    assert set(by_horizon) == {"2026-06-08", "2026-06-09"}
    # Each day is stamped with the vintage that *preceded* it.
    assert by_horizon["2026-06-08"].vintage_ts.replace(tzinfo=timezone.utc) == V0
    assert by_horizon["2026-06-09"].vintage_ts.replace(tzinfo=timezone.utc) == V1


def test_backfill_records_period_bounds_in_config(session):
    import json
    _pf, runs = _two_day_setup(session)
    cfg = json.loads(runs[0].config_json)
    assert cfg["replay"] is True
    assert cfg["n_mtus"] == 4
    assert cfg["period_start"].startswith("2026-06-08")
    assert cfg["period_end"].startswith("2026-06-09")


def test_day_without_preceding_vintage_is_skipped(session):
    # The only vintage for D1 is an *intraday* one (06:00 on D1), which is after
    # D1's start -> no day-ahead information set -> the day is skipped.
    pf = get_or_create_portfolio(session, "Wind Co", "NO2 Wind", "NO2")
    intraday = D1 + timedelta(hours=6)
    records = _forecast_records(intraday, D1)
    dam_map, pos_map = _maps(D1)
    runs = backfill_iar(
        session, pf.portfolio_id,
        forecast_records=records, dam_price_map=dam_map, position_map=pos_map,
        engine_config=FAST,
    )
    assert runs == []


def test_mtus_without_price_or_position_are_dropped(session):
    pf = get_or_create_portfolio(session, "Wind Co", "NO2 Wind", "NO2")
    records = _forecast_records(V0, D1, n=4)
    dam_map, pos_map = _maps(D1, n=4)
    # Remove one MTU's DAM price -> only 3 MTUs usable.
    del dam_map[_day_mtus(D1, 4)[2]]
    runs = backfill_iar(
        session, pf.portfolio_id,
        forecast_records=records, dam_price_map=dam_map, position_map=pos_map,
        engine_config=FAST,
    )
    import json
    assert json.loads(runs[0].config_json)["n_mtus"] == 3


def test_backfill_is_idempotent(session):
    pf, _ = _two_day_setup(session)
    # Re-run with the same inputs and replace=True -> still 2 runs, not 4.
    records = _forecast_records(V0, D1) + _forecast_records(V1, D2)
    dam_map, pos_map = _maps(D1, D2)
    backfill_iar(
        session, pf.portfolio_id,
        forecast_records=records, dam_price_map=dam_map, position_map=pos_map,
        engine_config=FAST, replace=True,
    )
    session.commit()
    assert session.query(SimulationRun).count() == 2


def test_unknown_portfolio_raises(session):
    with pytest.raises(ValueError, match="does not exist"):
        backfill_iar(
            session, 999,
            forecast_records=[], dam_price_map={}, position_map={},
        )


def test_no_records_returns_empty(session):
    pf = get_or_create_portfolio(session, "Wind Co", "NO2 Wind", "NO2")
    assert backfill_iar(
        session, pf.portfolio_id,
        forecast_records=[], dam_price_map={}, position_map={},
    ) == []


# --------------------------------------------------------------------------- #
# Comparison join
# --------------------------------------------------------------------------- #
def test_join_picks_latest_vintage_at_or_before_period_start(session):
    pf, _ = _two_day_setup(session)
    # D1's period -> V0 estimate; D2's period -> V1 estimate.
    assert estimate_for_period(session, pf.portfolio_id, D1).vintage_ts.replace(
        tzinfo=timezone.utc
    ) == V0
    assert estimate_for_period(session, pf.portfolio_id, D2).vintage_ts.replace(
        tzinfo=timezone.utc
    ) == V1


def test_join_returns_none_before_first_vintage(session):
    pf, _ = _two_day_setup(session)
    assert estimate_for_period(session, pf.portfolio_id, V0 - timedelta(days=5)) is None


def test_join_accepts_iso_string(session):
    pf, _ = _two_day_setup(session)
    run = estimate_for_period(session, pf.portfolio_id, "2026-06-09T00:00:00+00:00")
    assert run.vintage_ts.replace(tzinfo=timezone.utc) == V1


def test_iar_estimate_for_period_returns_typed_result(session):
    pf, _ = _two_day_setup(session)
    gross = iar_estimate_for_period(session, pf.portfolio_id, D1, "gross")
    spread = iar_estimate_for_period(session, pf.portfolio_id, D1, "spread")
    assert gross.iar_type == "gross" and spread.iar_type == "spread"
    assert gross.iar_value != spread.iar_value


def test_iar_estimate_rejects_bad_type(session):
    pf, _ = _two_day_setup(session)
    with pytest.raises(ValueError, match="gross.*spread"):
        iar_estimate_for_period(session, pf.portfolio_id, D1, "net")
