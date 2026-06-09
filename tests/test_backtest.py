"""Realised-cost + calibration tests (Tasks 3.1 and 3.3).

Covers the realised imbalance cost computation (gross + spread, sign convention,
period sum, time-window filter, input intersection), the actual-imbalance-price
ingestion sinks, the markets-client imbalance-price fetch (mocked SDK), the Kupiec
POF test, and the end-to-end backtest (exceedance flagging + persistence).

The 3.2 vintage join (``estimate_for_period``) is tested in ``test_replay.py``.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.orm import sessionmaker

from iar.db.models import (
    ActualDelivery,
    ActualImbalancePrice,
    DAMPosition,
    DAMPrice,
    HistoricalPerformanceRecord,
    IaRResult,
    SimulationRun,
)
from iar.db.session import init_db, make_engine
from iar.ingestion.flatfile_loader import (
    get_or_create_portfolio,
    store_actual_imbalance_price_records,
)
from iar.ingestion.markets_client import OptimeeringMarketsClient
from iar.risk.backtest import (
    kupiec_pof,
    load_performance_records,
    run_backtest,
)
from iar.risk.realised_cost import compute_realised_cost, realised_period_cost

T0 = datetime(2026, 6, 8, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def session():
    engine = make_engine(":memory:")
    init_db(engine)
    with sessionmaker(bind=engine, expire_on_commit=False, future=True)() as s:
        yield s


def _mtu(i: int) -> datetime:
    return T0 + timedelta(minutes=15 * i)


def _seed_portfolio(session, area="NO2", *, positions, actuals, imb_prices, dam_prices):
    """Create a portfolio and insert aligned per-MTU inputs (lists, one per MTU)."""
    pf = get_or_create_portfolio(session, "Wind Co", f"{area} Wind", area)
    for i, (pos, act, lam, spot) in enumerate(
        zip(positions, actuals, imb_prices, dam_prices)
    ):
        ts = _mtu(i)
        session.add(DAMPosition(portfolio_id=pf.portfolio_id, timestamp=ts, mwh=pos))
        session.add(ActualDelivery(portfolio_id=pf.portfolio_id, timestamp=ts, actual_mwh=act))
        session.add(ActualImbalancePrice(price_area=area, timestamp=ts, price=lam))
        session.add(DAMPrice(price_area=area, timestamp=ts, price=spot))
    session.flush()
    return pf


# --------------------------------------------------------------------------- #
# Core computation
# --------------------------------------------------------------------------- #
def test_realised_cost_values_and_sign_convention(session):
    # MTU0: short 10 MWh (sold 50, delivered 40) at imbalance price 100, spot 40.
    #   imbalance = +10 ; gross = 10*100 = 1000 (cost, positive) ;
    #   spread = 10*(100-40) = 600.
    # MTU1: long 5 MWh (sold 30, delivered 35) at price 80, spot 50.
    #   imbalance = -5 ; gross = -5*80 = -400 (revenue, negative) ;
    #   spread = -5*(80-50) = -150.
    pf = _seed_portfolio(
        session,
        positions=[50.0, 30.0],
        actuals=[40.0, 35.0],
        imb_prices=[100.0, 80.0],
        dam_prices=[40.0, 50.0],
    )
    df = compute_realised_cost(session, pf.portfolio_id)

    assert list(df.columns)[:4] == ["timestamp", "dam_position_mwh", "actual_mwh", "imbalance_mwh"]
    assert df["imbalance_mwh"].tolist() == [10.0, -5.0]
    assert df["gross_cost"].tolist() == [1000.0, -400.0]
    assert df["spread_cost"].tolist() == [600.0, -150.0]
    # short at a positive price is a cost (>0); long is revenue (<0)
    assert df.loc[0, "gross_cost"] > 0
    assert df.loc[1, "gross_cost"] < 0


def test_period_cost_sums_mtus(session):
    pf = _seed_portfolio(
        session,
        positions=[50.0, 30.0],
        actuals=[40.0, 35.0],
        imb_prices=[100.0, 80.0],
        dam_prices=[40.0, 50.0],
    )
    period = realised_period_cost(session, pf.portfolio_id)
    assert period["gross"] == pytest.approx(1000.0 - 400.0)
    assert period["spread"] == pytest.approx(600.0 - 150.0)
    assert period["n_mtus"] == 2
    assert period["first_mtu"] == _mtu(0)
    assert period["last_mtu"] == _mtu(1)


def test_window_filter_is_half_open(session):
    pf = _seed_portfolio(
        session,
        positions=[10.0, 10.0, 10.0],
        actuals=[0.0, 0.0, 0.0],
        imb_prices=[100.0, 100.0, 100.0],
        dam_prices=[0.0, 0.0, 0.0],
    )
    # [MTU1, MTU2) -> only MTU1 included.
    df = compute_realised_cost(session, pf.portfolio_id, start=_mtu(1), end=_mtu(2))
    assert df["timestamp"].tolist() == [_mtu(1)]
    assert df["gross_cost"].tolist() == [1000.0]


def test_only_intersecting_mtus_are_returned(session):
    # 3 positions/actuals but a price gap at MTU1 -> that MTU is dropped.
    pf = get_or_create_portfolio(session, "Wind Co", "NO2 Wind", "NO2")
    for i in range(3):
        session.add(DAMPosition(portfolio_id=pf.portfolio_id, timestamp=_mtu(i), mwh=10.0))
        session.add(ActualDelivery(portfolio_id=pf.portfolio_id, timestamp=_mtu(i), actual_mwh=0.0))
        session.add(DAMPrice(price_area="NO2", timestamp=_mtu(i), price=0.0))
    # imbalance price only for MTU0 and MTU2
    session.add(ActualImbalancePrice(price_area="NO2", timestamp=_mtu(0), price=100.0))
    session.add(ActualImbalancePrice(price_area="NO2", timestamp=_mtu(2), price=100.0))
    session.flush()

    df = compute_realised_cost(session, pf.portfolio_id)
    assert df["timestamp"].tolist() == [_mtu(0), _mtu(2)]


def test_empty_when_no_data(session):
    pf = get_or_create_portfolio(session, "Wind Co", "NO2 Wind", "NO2")
    df = compute_realised_cost(session, pf.portfolio_id)
    assert df.empty
    assert list(df.columns)  # columns still present
    period = realised_period_cost(session, pf.portfolio_id)
    assert period == {"gross": 0.0, "spread": 0.0, "n_mtus": 0, "first_mtu": None, "last_mtu": None}


def test_unknown_portfolio_raises(session):
    with pytest.raises(ValueError, match="does not exist"):
        compute_realised_cost(session, 999)


def test_prices_are_area_scoped(session):
    # Imbalance prices exist for SE3 but the portfolio is NO2 -> no overlap.
    pf = get_or_create_portfolio(session, "Wind Co", "NO2 Wind", "NO2")
    session.add(DAMPosition(portfolio_id=pf.portfolio_id, timestamp=_mtu(0), mwh=10.0))
    session.add(ActualDelivery(portfolio_id=pf.portfolio_id, timestamp=_mtu(0), actual_mwh=0.0))
    session.add(DAMPrice(price_area="NO2", timestamp=_mtu(0), price=0.0))
    session.add(ActualImbalancePrice(price_area="SE3", timestamp=_mtu(0), price=100.0))
    session.flush()
    assert compute_realised_cost(session, pf.portfolio_id).empty


# --------------------------------------------------------------------------- #
# Ingestion sink
# --------------------------------------------------------------------------- #
def test_store_actual_imbalance_price_records_feeds_realised_cost(session):
    pf = get_or_create_portfolio(session, "Wind Co", "NO2 Wind", "NO2")
    session.add(DAMPosition(portfolio_id=pf.portfolio_id, timestamp=_mtu(0), mwh=20.0))
    session.add(ActualDelivery(portfolio_id=pf.portfolio_id, timestamp=_mtu(0), actual_mwh=5.0))
    session.add(DAMPrice(price_area="NO2", timestamp=_mtu(0), price=30.0))
    n = store_actual_imbalance_price_records(
        session, "NO2", [{"timestamp": _mtu(0).isoformat(), "eur_per_mwh": 90.0}]
    )
    session.flush()
    assert n == 1
    df = compute_realised_cost(session, pf.portfolio_id)
    # imbalance = 20-5 = 15 ; gross = 15*90 = 1350 ; spread = 15*(90-30)=900
    assert df.loc[0, "gross_cost"] == pytest.approx(1350.0)
    assert df.loc[0, "spread_cost"] == pytest.approx(900.0)


def test_store_actual_imbalance_prices_replace_dedupes(session):
    get_or_create_portfolio(session, "Wind Co", "NO2 Wind", "NO2")
    store_actual_imbalance_price_records(
        session, "NO2", [{"timestamp": _mtu(0).isoformat(), "eur_per_mwh": 10.0}]
    )
    # replace=True (default) wipes prior NO2 rows; last write wins on dup ts.
    store_actual_imbalance_price_records(
        session,
        "NO2",
        [
            {"timestamp": _mtu(0).isoformat(), "eur_per_mwh": 50.0},
            {"timestamp": _mtu(0).isoformat(), "eur_per_mwh": 55.0},
        ],
    )
    session.flush()
    rows = session.query(ActualImbalancePrice).filter_by(price_area="NO2").all()
    assert len(rows) == 1
    assert rows[0].price == 55.0


# --------------------------------------------------------------------------- #
# Markets client: realised imbalance price fetch (mocked SDK)
# --------------------------------------------------------------------------- #
class _FakeMarketsApi:
    def __init__(self, series_items, market_items):
        self._series_items = series_items
        self._market_items = market_items
        self.calls = {}

    def get_market_series(self, **kwargs):
        self.calls["get_market_series"] = kwargs
        return {"items": self._series_items}

    def get_market(self, **kwargs):
        self.calls["get_market"] = kwargs
        return {"items": self._market_items}


def test_get_imbalance_prices_filters_and_normalises():
    series = [{"id": 42, "market": "Imbalance", "series_type": "imbalance price", "publisher": "Nordpool"}]
    datapoints = [
        {"start": datetime(2026, 6, 8, 0, 0, tzinfo=timezone.utc), "value": 91.5},
        {"start": datetime(2026, 6, 8, 0, 15, tzinfo=timezone.utc), "value": 88.0},
    ]
    api = _FakeMarketsApi(series, [{"series_id": 42, "datapoints": datapoints}])
    client = OptimeeringMarketsClient(_api=api)

    recs = client.get_imbalance_prices("NO2")
    kw = api.calls["get_market_series"]
    assert kw["market"] == ["Imbalance"]
    assert kw["series_type"] == ["imbalance price"]
    assert recs[0] == {
        "price_area": "NO2",
        "timestamp": "2026-06-08T00:00:00+00:00",
        "eur_per_mwh": 91.5,
    }
    assert [r["timestamp"] for r in recs] == sorted(r["timestamp"] for r in recs)


def test_get_imbalance_prices_no_series_raises():
    api = _FakeMarketsApi([], [])
    client = OptimeeringMarketsClient(_api=api)
    with pytest.raises(LookupError, match="imbalance price"):
        client.get_imbalance_prices("NO2")


# --------------------------------------------------------------------------- #
# Kupiec POF test (Task 3.3) — pure, no DB
# --------------------------------------------------------------------------- #
def test_kupiec_perfectly_calibrated_is_zero():
    k = kupiec_pof(n_obs=100, n_exceedances=5, expected_rate=0.05)
    assert k.lr_statistic == pytest.approx(0.0, abs=1e-9)
    assert k.p_value == pytest.approx(1.0, abs=1e-6)
    assert k.reject_h0 is False
    assert k.well_calibrated is True


def test_kupiec_too_many_exceedances_rejects():
    # 25 breaches in 250 obs (10%) vs an expected 5% -> mis-calibrated.
    k = kupiec_pof(n_obs=250, n_exceedances=25, expected_rate=0.05)
    assert k.lr_statistic == pytest.approx(10.33, abs=0.2)
    assert k.p_value < 0.05
    assert k.reject_h0 is True
    assert k.well_calibrated is False


def test_kupiec_zero_exceedances_is_positive_stat():
    # Zero breaches over 100 obs is *too few* at 5% -> non-trivial statistic.
    k = kupiec_pof(n_obs=100, n_exceedances=0, expected_rate=0.05)
    assert k.lr_statistic > 0
    assert k.observed_rate == 0.0


def test_kupiec_no_observations_is_none_safe():
    k = kupiec_pof(n_obs=0, n_exceedances=0, expected_rate=0.05)
    assert k.lr_statistic is None and k.p_value is None
    assert k.reject_h0 is None and k.well_calibrated is None


def test_kupiec_bad_expected_rate_raises():
    with pytest.raises(ValueError, match="expected_rate"):
        kupiec_pof(n_obs=10, n_exceedances=1, expected_rate=1.5)


# --------------------------------------------------------------------------- #
# End-to-end backtest (Task 3.3) — exceedance flagging + persistence
# --------------------------------------------------------------------------- #
def _insert_replay_run(session, pid, day_start, gross_iar, spread_iar, confidence=0.95):
    """Insert a stored day-ahead estimate (run + gross/spread results) for a day."""
    day_end = day_start + timedelta(days=1)
    vintage = day_start - timedelta(hours=12)
    run = SimulationRun(
        portfolio_id=pid, run_ts=vintage, vintage_ts=vintage,
        n_scenarios=5000, seed=42,
        config_json=json.dumps({
            "replay": True,
            "period_start": day_start.isoformat(),
            "period_end": day_end.isoformat(),
            "n_mtus": 1,
        }),
    )
    session.add(run)
    session.flush()
    horizon = day_start.date().isoformat()
    session.add(IaRResult(run_id=run.run_id, confidence=confidence, horizon=horizon,
                          iar_type="gross", iar_value=gross_iar, ciar_value=gross_iar))
    session.add(IaRResult(run_id=run.run_id, confidence=confidence, horizon=horizon,
                          iar_type="spread", iar_value=spread_iar, ciar_value=spread_iar))
    session.flush()
    return horizon


def _insert_realised(session, pid, area, day_start, *, dam_pos, actual, imb_price, dam_price):
    """Insert one settled MTU at ``day_start`` producing a known realised cost."""
    session.add(DAMPosition(portfolio_id=pid, timestamp=day_start, mwh=dam_pos))
    session.add(ActualDelivery(portfolio_id=pid, timestamp=day_start, actual_mwh=actual))
    session.add(ActualImbalancePrice(price_area=area, timestamp=day_start, price=imb_price))
    session.add(DAMPrice(price_area=area, timestamp=day_start, price=dam_price))
    session.flush()


D1 = datetime(2026, 6, 8, tzinfo=timezone.utc)
D2 = datetime(2026, 6, 9, tzinfo=timezone.utc)
D3 = datetime(2026, 6, 10, tzinfo=timezone.utc)


def _three_day_backtest_setup(session):
    pf = get_or_create_portfolio(session, "Wind Co", "NO2 Wind", "NO2")
    pid = pf.portfolio_id
    # Estimates: gross IaR = 500 each day.
    _insert_replay_run(session, pid, D1, gross_iar=500.0, spread_iar=300.0)
    _insert_replay_run(session, pid, D2, gross_iar=500.0, spread_iar=300.0)
    _insert_replay_run(session, pid, D3, gross_iar=500.0, spread_iar=300.0)
    # D1 realised: short 20 MWh @ 100 -> gross 2000 > 500 -> EXCEEDED.
    _insert_realised(session, pid, "NO2", D1, dam_pos=20.0, actual=0.0, imb_price=100.0, dam_price=40.0)
    # D2 realised: short 1 MWh @ 100 -> gross 100 < 500 -> within.
    _insert_realised(session, pid, "NO2", D2, dam_pos=1.0, actual=0.0, imb_price=100.0, dam_price=40.0)
    # D3: no realised inputs -> not settled -> skipped.
    session.commit()
    return pid


def test_backtest_flags_exceedances_and_skips_unsettled(session):
    pid = _three_day_backtest_setup(session)
    res = run_backtest(session, pid, "gross")
    assert res.n_periods == 2          # D3 skipped (unsettled)
    assert res.n_exceedances == 1      # only D1
    by_period = {p.period: p for p in res.periods}
    assert by_period["2026-06-08"].exceeded is True
    assert by_period["2026-06-09"].exceeded is False
    assert by_period["2026-06-08"].realised_cost == pytest.approx(2000.0)


def test_backtest_kupiec_uses_estimate_confidence(session):
    pid = _three_day_backtest_setup(session)
    res = run_backtest(session, pid, "gross")
    assert res.confidence == 0.95
    assert res.kupiec.expected_rate == pytest.approx(0.05)
    assert res.kupiec.observed_rate == pytest.approx(0.5)  # 1 of 2
    assert res.kupiec.n_obs == 2


def test_backtest_persists_one_record_per_period(session):
    pid = _three_day_backtest_setup(session)
    run_backtest(session, pid, "gross", persist=True)
    session.commit()
    recs = session.query(HistoricalPerformanceRecord).filter_by(portfolio_id=pid).all()
    assert len(recs) == 2
    assert all(r.kupiec_stat is not None for r in recs)
    df = load_performance_records(session, pid)
    assert set(df["period"]) == {"2026-06-08", "2026-06-09"}
    assert bool(df.set_index("period").loc["2026-06-08", "exceeded"]) is True


def test_backtest_persist_is_idempotent(session):
    pid = _three_day_backtest_setup(session)
    run_backtest(session, pid, "gross", persist=True, replace=True)
    run_backtest(session, pid, "gross", persist=True, replace=True)
    session.commit()
    assert session.query(HistoricalPerformanceRecord).filter_by(portfolio_id=pid).count() == 2


def test_backtest_persist_false_writes_nothing(session):
    pid = _three_day_backtest_setup(session)
    run_backtest(session, pid, "gross", persist=False)
    assert session.query(HistoricalPerformanceRecord).filter_by(portfolio_id=pid).count() == 0


def test_backtest_spread_basis(session):
    pid = _three_day_backtest_setup(session)
    # D1 spread realised = 20*(100-40)=1200 > 300 -> exceeded; D2 = 1*60=60 < 300.
    res = run_backtest(session, pid, "spread")
    by_period = {p.period: p for p in res.periods}
    assert by_period["2026-06-08"].exceeded is True
    assert by_period["2026-06-08"].realised_cost == pytest.approx(1200.0)
    assert by_period["2026-06-09"].exceeded is False


def test_backtest_no_settled_periods_is_empty(session):
    pf = get_or_create_portfolio(session, "Wind Co", "NO2 Wind", "NO2")
    _insert_replay_run(session, pf.portfolio_id, D1, gross_iar=500.0, spread_iar=300.0)
    session.commit()
    res = run_backtest(session, pf.portfolio_id, "gross")
    assert res.n_periods == 0
    assert res.kupiec.well_calibrated is None
    assert res.as_frame().empty


def test_backtest_rejects_bad_iar_type(session):
    pf = get_or_create_portfolio(session, "Wind Co", "NO2 Wind", "NO2")
    with pytest.raises(ValueError, match="gross.*spread"):
        run_backtest(session, pf.portfolio_id, "net")
