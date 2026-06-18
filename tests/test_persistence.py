"""Tests for simulation-result persistence (Task 2.4)."""

import json
from datetime import UTC, datetime, timezone

import numpy as np
import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from iar.db.models import IaRResult, SimulationRun
from iar.db.session import init_db, make_engine
from iar.ingestion.flatfile_loader import get_or_create_portfolio
from iar.simulation.engine import EngineConfig, run_simulation
from iar.simulation.imbalance_model import ImbalanceModel
from iar.simulation.persistence import persist_report
from iar.simulation.price_sampler import QuantilePriceSampler

VINTAGE = datetime(2026, 6, 8, 9, 0, tzinfo=UTC)


@pytest.fixture
def session():
    engine = make_engine(":memory:")
    init_db(engine)
    with sessionmaker(bind=engine, expire_on_commit=False, future=True)() as s:
        yield s


@pytest.fixture
def report():
    price = QuantilePriceSampler(np.array([0.05, 0.5, 0.95]), np.tile([-20.0, 0.0, 30.0], (4, 1)))
    imb = ImbalanceModel(np.full(4, 1.0), np.full(4, 3.0), dist="normal")
    return run_simulation(
        price,
        imb,
        dam_price=np.full(4, 40.0),
        config=EngineConfig(n_scenarios=5000, confidence=0.95, seed=42),
    )


def test_persist_creates_run_and_two_results(session, report):
    pf = get_or_create_portfolio(session, "Alice", "NO2 Wind", "NO2")
    run = persist_report(session, report, pf.portfolio_id, VINTAGE, horizon="4xPT15M")
    session.commit()

    assert session.query(SimulationRun).count() == 1
    assert run.run_id is not None
    assert run.n_scenarios == 5000
    assert run.seed == 42
    assert run.vintage_ts.replace(tzinfo=UTC) == VINTAGE

    results = session.query(IaRResult).filter_by(run_id=run.run_id).all()
    assert {r.iar_type for r in results} == {"gross", "spread"}
    for r in results:
        assert r.confidence == 0.95
        assert r.horizon == "4xPT15M"
        assert r.ciar_value is not None


def test_persisted_values_match_report(session, report):
    pf = get_or_create_portfolio(session, "Bob", "NO2 Wind", "NO2")
    run = persist_report(session, report, pf.portfolio_id, VINTAGE, horizon="h")
    session.commit()
    by_type = {r.iar_type: r for r in run.results}
    assert by_type["gross"].iar_value == pytest.approx(report.gross.iar)
    assert by_type["gross"].ciar_value == pytest.approx(report.gross.ciar)
    assert by_type["spread"].iar_value == pytest.approx(report.spread.iar)


def test_config_json_records_reproducibility_info(session, report):
    pf = get_or_create_portfolio(session, "Carol", "NO2 Wind", "NO2")
    run = persist_report(
        session,
        report,
        pf.portfolio_id,
        VINTAGE,
        horizon="h",
        extra_config={"area": "NO2", "dist": "normal"},
    )
    cfg = json.loads(run.config_json)
    assert cfg["seed"] == 42 and cfg["n_scenarios"] == 5000
    assert cfg["confidence"] == 0.95 and cfg["area"] == "NO2"


def test_seedless_report_rejected(session, report):
    pf = get_or_create_portfolio(session, "Dave", "NO2 Wind", "NO2")
    object.__setattr__(report, "seed", None)  # simulate a no-seed run
    with pytest.raises(ValueError, match="seed is None"):
        persist_report(session, report, pf.portfolio_id, VINTAGE, horizon="h")


def test_unknown_portfolio_violates_fk(session, report):
    # FK enforcement is on; a non-existent portfolio_id must fail.
    with pytest.raises(IntegrityError):
        persist_report(session, report, 999, VINTAGE, horizon="h")
        session.flush()


def test_multiple_runs_accumulate(session, report):
    pf = get_or_create_portfolio(session, "Erin", "NO2 Wind", "NO2")
    persist_report(session, report, pf.portfolio_id, VINTAGE, horizon="h")
    persist_report(session, report, pf.portfolio_id, VINTAGE, horizon="h")
    session.commit()
    assert session.query(SimulationRun).count() == 2
    assert session.query(IaRResult).count() == 4  # 2 per run
