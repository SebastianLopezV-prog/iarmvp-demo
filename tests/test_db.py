"""Schema tests for Task 1.2: tables build, relationships and constraints hold."""

from datetime import UTC, datetime, timezone

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from iar.db.models import (
    Base,
    IaRResult,
    Portfolio,
    SimulationRun,
    User,
)
from iar.db.session import init_db, make_engine

EXPECTED_TABLES = {
    "users",
    "portfolios",
    "dam_positions",
    "generation_forecasts",
    "actual_deliveries",
    "imbalance_price_forecasts",
    "actual_imbalance_prices",
    "simulation_runs",
    "iar_results",
    "alerts",
    "historical_performance_records",
}


@pytest.fixture
def session():
    engine = make_engine(":memory:")
    init_db(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with factory() as s:
        yield s


def test_all_tables_created(session):
    tables = set(inspect(session.bind).get_table_names())
    assert EXPECTED_TABLES <= tables


def test_user_portfolio_run_result_chain(session):
    user = User(name="Alice")
    pf = Portfolio(name="NO2 Wind", price_area="NO2", user=user)
    run = SimulationRun(
        portfolio=pf,
        run_ts=datetime.now(UTC),
        vintage_ts=datetime.now(UTC),
        n_scenarios=5000,
        seed=42,
    )
    run.results.append(
        IaRResult(
            confidence=0.95, horizon="24h", iar_type="gross", iar_value=42800.0, ciar_value=51000.0
        )
    )
    session.add(user)
    session.commit()

    fetched = session.query(IaRResult).one()
    assert fetched.run.portfolio.user.name == "Alice"
    assert fetched.iar_value == 42800.0


def test_iar_type_check_constraint(session):
    pf = Portfolio(name="P", price_area="NO1", user=User(name="Bob"))
    run = SimulationRun(
        portfolio=pf,
        run_ts=datetime.now(UTC),
        vintage_ts=datetime.now(UTC),
        n_scenarios=10,
        seed=1,
    )
    run.results.append(IaRResult(confidence=0.95, horizon="24h", iar_type="bogus", iar_value=1.0))
    session.add(pf)
    with pytest.raises(IntegrityError):
        session.commit()


def test_price_area_check_constraint(session):
    session.add(Portfolio(name="P", price_area="XX9", user=User(name="Carol")))
    with pytest.raises(IntegrityError):
        session.commit()
