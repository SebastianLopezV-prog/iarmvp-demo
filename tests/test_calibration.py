"""Sigma calibration tests (Option 1)."""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.orm import sessionmaker

from iar.db.models import ActualDelivery, ActualImbalancePrice, DAMPosition, DAMPrice
from iar.db.session import init_db, make_engine
from iar.ingestion.flatfile_loader import get_or_create_portfolio
from iar.risk.calibration import calibrate_sigma
from iar.simulation.engine import EngineConfig

FAST = EngineConfig(n_scenarios=4000, confidence=0.95, seed=42)
PCT = [5.0, 50.0, 95.0]
# A symmetric spread curve centred on 0 so the estimate's IaR is driven by sigma.
SPREAD = {5.0: -20.0, 50.0: 0.0, 95.0: 20.0}


@pytest.fixture
def session():
    engine = make_engine(":memory:")
    init_db(engine)
    with sessionmaker(bind=engine, expire_on_commit=False, future=True)() as s:
        yield s


def _day(i: int) -> datetime:
    return datetime(2026, 6, 8, tzinfo=timezone.utc) + timedelta(days=i)


def _forecast_records(days):
    """One MTU (00:00) per day, day-ahead vintage at noon the day before."""
    recs = []
    for d in days:
        vintage = (d - timedelta(hours=12)).isoformat()
        for q in PCT:
            recs.append({"vintage_ts": vintage, "timestamp": d.isoformat(),
                         "quantile": q, "value": SPREAD[q]})
    return recs


def _setup(session, realised_positions):
    """Estimates use a flat zero-imbalance position; realised cost is controlled
    separately via DB rows (dam_pos short at price 40 -> known gross cost)."""
    pf = get_or_create_portfolio(session, "Wind Co", "NO2 Wind", "NO2")
    pid = pf.portfolio_id
    days = [_day(i) for i in range(len(realised_positions))]

    # Estimate inputs (expected imbalance 0: dam_pos == gen).
    forecast_records = _forecast_records(days)
    dam_map = {d.isoformat(): 40.0 for d in days}
    pos_map = {d.isoformat(): (0.0, 0.0) for d in days}

    # Realised inputs in the DB: short `p` MWh @ 40 -> realised gross = p*40.
    for d, p in zip(days, realised_positions):
        session.add(DAMPosition(portfolio_id=pid, timestamp=d, mwh=p))
        session.add(ActualDelivery(portfolio_id=pid, timestamp=d, actual_mwh=0.0))
        session.add(ActualImbalancePrice(price_area="NO2", timestamp=d, price=40.0))
        session.add(DAMPrice(price_area="NO2", timestamp=d, price=40.0))
    session.flush()
    return pid, forecast_records, dam_map, pos_map


def test_exceedance_rate_decreases_with_sigma(session):
    # Four days with increasing realised cost (100, 200, 400, 800 EUR).
    pid, fr, dam, pos = _setup(session, [2.5, 5.0, 10.0, 20.0])
    res = calibrate_sigma(
        session, pid, forecast_records=fr, dam_price_map=dam, position_map=pos,
        engine_config=FAST, iar_type="gross",
    )
    assert res.n_periods == 4
    rates = [r for _s, r in res.grid if r is not None]
    # IaR grows with sigma -> fewer exceedances -> non-increasing rate.
    assert all(a >= b for a, b in zip(rates, rates[1:]))
    # Smallest sigma over-confident (breaches), largest conservative (none).
    assert rates[0] == 1.0
    assert rates[-1] == 0.0


def test_recommendation_is_closest_to_target(session):
    pid, fr, dam, pos = _setup(session, [2.5, 5.0, 10.0, 20.0])
    res = calibrate_sigma(
        session, pid, forecast_records=fr, dam_price_map=dam, position_map=pos,
        engine_config=FAST, iar_type="gross",
    )
    assert res.target_rate == pytest.approx(0.05)
    # The achieved rate must be the grid rate closest to the target.
    best = min((r for _s, r in res.grid if r is not None),
               key=lambda r: abs(r - res.target_rate))
    assert res.achieved_rate == pytest.approx(best)
    assert res.recommended_sigma_fraction in [s for s, _r in res.grid]


def test_higher_target_allows_smaller_sigma(session):
    pid, fr, dam, pos = _setup(session, [2.5, 5.0, 10.0, 20.0])
    low = calibrate_sigma(session, pid, forecast_records=fr, dam_price_map=dam,
                          position_map=pos, engine_config=FAST, target_rate=0.05)
    high = calibrate_sigma(session, pid, forecast_records=fr, dam_price_map=dam,
                           position_map=pos, engine_config=FAST, target_rate=0.50)
    # Tolerating more breaches (higher target) needs no more conservatism.
    assert high.recommended_sigma_fraction <= low.recommended_sigma_fraction


def test_no_settled_periods_returns_none(session):
    # Estimates exist but no realised prices in the DB -> nothing to calibrate.
    pf = get_or_create_portfolio(session, "Wind Co", "NO2 Wind", "NO2")
    days = [_day(0), _day(1)]
    fr = _forecast_records(days)
    dam = {d.isoformat(): 40.0 for d in days}
    pos = {d.isoformat(): (0.0, 0.0) for d in days}
    res = calibrate_sigma(session, pf.portfolio_id, forecast_records=fr,
                          dam_price_map=dam, position_map=pos, engine_config=FAST)
    assert res.n_periods == 0
    assert res.recommended_sigma_fraction is None
    assert "No settled periods" in res.note


def test_custom_grid_is_used(session):
    pid, fr, dam, pos = _setup(session, [5.0, 5.0])
    res = calibrate_sigma(session, pid, forecast_records=fr, dam_price_map=dam,
                          position_map=pos, engine_config=FAST,
                          sigma_grid=(0.05, 0.10, 0.20))
    assert [s for s, _r in res.grid] == [0.05, 0.10, 0.20]


def test_bad_iar_type_raises(session):
    pf = get_or_create_portfolio(session, "Wind Co", "NO2 Wind", "NO2")
    with pytest.raises(ValueError, match="gross.*spread"):
        calibrate_sigma(session, pf.portfolio_id, forecast_records=[],
                        dam_price_map={}, position_map={}, iar_type="net")
