"""Tests for the flat-file loader (Task 1.4)."""

import pandas as pd
import pytest
from sqlalchemy.orm import sessionmaker

from iar.db.models import ActualDelivery, DAMPosition, DAMPrice, GenerationForecast
from iar.db.session import init_db, make_engine
from iar.ingestion.flatfile_loader import (
    FileValidationError,
    get_or_create_portfolio,
    load_actual_delivery,
    load_dam_positions,
    load_dam_prices,
    load_generation_forecasts,
)


@pytest.fixture
def session():
    engine = make_engine(":memory:")
    init_db(engine)
    with sessionmaker(bind=engine, expire_on_commit=False, future=True)() as s:
        yield s


@pytest.fixture
def portfolio(session):
    return get_or_create_portfolio(session, "Alice", "NO2 Wind", "NO2")


def _write_csv(tmp_path, name, value_col):
    df = pd.DataFrame(
        {
            "timestamp": ["2026-06-03T00:00:00+00:00", "2026-06-03T00:15:00+00:00"],
            value_col: [10.0, 12.5],
        }
    )
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def test_load_three_series(session, portfolio, tmp_path):
    pid = portfolio.portfolio_id
    assert load_dam_positions(session, pid, _write_csv(tmp_path, "dam.csv", "mwh")) == 2
    assert (
        load_generation_forecasts(
            session, pid, _write_csv(tmp_path, "gen.csv", "forecast_mwh")
        )
        == 2
    )
    assert (
        load_actual_delivery(session, pid, _write_csv(tmp_path, "act.csv", "actual_mwh"))
        == 2
    )
    assert session.query(DAMPosition).count() == 2
    assert session.query(GenerationForecast).count() == 2
    assert session.query(ActualDelivery).count() == 2


def test_load_dam_prices(session, portfolio, tmp_path):
    # DAM price is keyed by area, not portfolio.
    path = _write_csv(tmp_path, "dam_price.csv", "eur_per_mwh")
    assert load_dam_prices(session, "NO2", path) == 2
    assert session.query(DAMPrice).count() == 2
    assert {r.price_area for r in session.query(DAMPrice)} == {"NO2"}


def test_load_dam_prices_is_idempotent(session, portfolio, tmp_path):
    path = _write_csv(tmp_path, "dam_price.csv", "eur_per_mwh")
    load_dam_prices(session, "NO2", path)
    load_dam_prices(session, "NO2", path)  # replace=True default
    assert session.query(DAMPrice).count() == 2  # not 4


def test_load_is_idempotent(session, portfolio, tmp_path):
    pid = portfolio.portfolio_id
    path = _write_csv(tmp_path, "dam.csv", "mwh")
    load_dam_positions(session, pid, path)
    load_dam_positions(session, pid, path)  # replace=True default
    assert session.query(DAMPosition).count() == 2  # not 4


def test_missing_column_rejected(session, portfolio, tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"timestamp": ["2026-06-03T00:00:00+00:00"], "wrong": [1.0]}).to_csv(
        bad, index=False
    )
    with pytest.raises(FileValidationError, match="missing required column"):
        load_dam_positions(session, portfolio.portfolio_id, bad)


def test_duplicate_timestamp_rejected(session, portfolio, tmp_path):
    bad = tmp_path / "dupe.csv"
    pd.DataFrame(
        {"timestamp": ["2026-06-03T00:00:00+00:00"] * 2, "mwh": [1.0, 2.0]}
    ).to_csv(bad, index=False)
    with pytest.raises(FileValidationError, match="duplicate"):
        load_dam_positions(session, portfolio.portfolio_id, bad)


def test_unknown_portfolio_rejected(session, tmp_path):
    with pytest.raises(FileValidationError, match="does not exist"):
        load_dam_positions(session, 999, _write_csv(tmp_path, "dam.csv", "mwh"))
