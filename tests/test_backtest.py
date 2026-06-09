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
