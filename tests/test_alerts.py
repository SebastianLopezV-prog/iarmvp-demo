"""Limits & alerts tests (Task 3.4)."""

from datetime import datetime, timezone

import pytest
from sqlalchemy.orm import sessionmaker

from iar.db.models import Alert, IaRResult, SimulationRun
from iar.db.session import init_db, make_engine
from iar.ingestion.flatfile_loader import get_or_create_portfolio
from iar.risk.alerts import (
    LimitConfig,
    check_run,
    classify_severity,
    evaluate_latest,
    evaluate_run,
    load_alerts,
    load_limits,
)

VINTAGE = datetime(2026, 6, 8, 9, 0, tzinfo=timezone.utc)


@pytest.fixture
def session():
    engine = make_engine(":memory:")
    init_db(engine)
    with sessionmaker(bind=engine, expire_on_commit=False, future=True)() as s:
        yield s


def _run_with_iar(session, gross_iar, spread_iar, *, name="NO2 Wind", run_ts=VINTAGE):
    """Create a portfolio + a SimulationRun with gross/spread IaRResults."""
    pf = get_or_create_portfolio(session, f"user::{name}", name, "NO2")
    run = SimulationRun(
        portfolio_id=pf.portfolio_id, run_ts=run_ts, vintage_ts=run_ts,
        n_scenarios=5000, seed=42, config_json="{}",
    )
    session.add(run)
    session.flush()
    for iar_type, val in (("gross", gross_iar), ("spread", spread_iar)):
        session.add(IaRResult(run_id=run.run_id, confidence=0.95, horizon="2026-06-08",
                              iar_type=iar_type, iar_value=val, ciar_value=val))
    session.flush()
    return pf, run


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_load_limits_reads_defaults():
    cfg = load_limits()  # the real config/limits.toml
    assert cfg.limit_for("anything", "gross", "remaining_day") == 4000
    assert cfg.limit_for("anything", "spread", "per_mtu") == 250


def test_limit_for_override_beats_default():
    cfg = LimitConfig({
        "default": {"gross": {"remaining_day_eur": 50000}},
        "portfolio": {"NO2 Wind": {"gross": {"remaining_day_eur": 75000}}},
    })
    assert cfg.limit_for("NO2 Wind", "gross", "remaining_day") == 75000
    assert cfg.limit_for("Other", "gross", "remaining_day") == 50000


def test_limit_for_missing_returns_none():
    cfg = LimitConfig({"default": {"gross": {"remaining_day_eur": 50000}}})
    assert cfg.limit_for("x", "spread", "remaining_day") is None


def test_limit_for_bad_type_raises():
    cfg = LimitConfig({})
    with pytest.raises(ValueError, match="limit_type"):
        cfg.limit_for("x", "gross", "weekly")


# --------------------------------------------------------------------------- #
# Severity
# --------------------------------------------------------------------------- #
def test_classify_hard_soft_within():
    assert classify_severity(60000, 50000) == "hard"     # > L
    assert classify_severity(45000, 50000) == "soft"     # > 0.8L, <= L
    assert classify_severity(30000, 50000) is None        # <= 0.8L


def test_classify_negative_iar_never_breaches():
    # Worst case is still a net revenue -> no breach.
    assert classify_severity(-10000, 50000) is None


def test_classify_nonpositive_limit_is_none():
    assert classify_severity(10, 0) is None


# --------------------------------------------------------------------------- #
# check_run (pure)
# --------------------------------------------------------------------------- #
def test_check_run_reports_headroom_and_severity(session):
    cfg = LimitConfig({"default": {
        "gross": {"remaining_day_eur": 50000},
        "spread": {"remaining_day_eur": 40000},
    }})
    _pf, run = _run_with_iar(session, gross_iar=60000, spread_iar=10000)
    checks = {c.iar_type: c for c in check_run(run, cfg)}
    assert checks["gross"].severity == "hard"
    assert checks["gross"].utilisation == pytest.approx(1.2)
    assert checks["spread"].severity is None
    assert checks["spread"].utilisation == pytest.approx(0.25)


def test_check_run_skips_variants_without_limits(session):
    cfg = LimitConfig({"default": {"gross": {"remaining_day_eur": 50000}}})
    _pf, run = _run_with_iar(session, gross_iar=10, spread_iar=10)
    checks = check_run(run, cfg)
    assert {c.iar_type for c in checks} == {"gross"}  # spread has no limit


# --------------------------------------------------------------------------- #
# evaluate_run / evaluate_latest (persisting)
# --------------------------------------------------------------------------- #
def test_evaluate_run_persists_only_breaches(session):
    cfg = LimitConfig({"default": {
        "gross": {"remaining_day_eur": 50000},
        "spread": {"remaining_day_eur": 40000},
    }})
    _pf, run = _run_with_iar(session, gross_iar=60000, spread_iar=10000)
    alerts = evaluate_run(session, run, cfg, breach_ts=VINTAGE)
    session.commit()
    assert len(alerts) == 1  # only gross breaches
    a = session.query(Alert).one()
    assert a.severity == "hard"
    assert a.limit_type == "remaining_day"
    assert a.limit_value == 50000
    assert a.result.iar_type == "gross"


def test_evaluate_run_soft_warning(session):
    cfg = LimitConfig({"default": {"gross": {"remaining_day_eur": 50000}}})
    _pf, run = _run_with_iar(session, gross_iar=45000, spread_iar=0)
    alerts = evaluate_run(session, run, cfg)
    assert len(alerts) == 1 and alerts[0].severity == "soft"


def test_evaluate_run_is_idempotent(session):
    cfg = LimitConfig({"default": {"gross": {"remaining_day_eur": 50000}}})
    _pf, run = _run_with_iar(session, gross_iar=60000, spread_iar=0)
    evaluate_run(session, run, cfg)
    evaluate_run(session, run, cfg)  # replace=True wipes the prior alert first
    session.commit()
    assert session.query(Alert).count() == 1


def test_evaluate_run_persist_false_writes_nothing(session):
    cfg = LimitConfig({"default": {"gross": {"remaining_day_eur": 50000}}})
    _pf, run = _run_with_iar(session, gross_iar=60000, spread_iar=0)
    alerts = evaluate_run(session, run, cfg, persist=False)
    assert len(alerts) == 1
    assert session.query(Alert).count() == 0


def test_evaluate_latest_picks_newest_run(session):
    cfg = LimitConfig({"default": {"gross": {"remaining_day_eur": 50000}}})
    pf, _old = _run_with_iar(session, gross_iar=10, spread_iar=0,
                             run_ts=datetime(2026, 6, 7, tzinfo=timezone.utc))
    # newer run for the SAME portfolio that breaches
    run2 = SimulationRun(portfolio_id=pf.portfolio_id,
                         run_ts=datetime(2026, 6, 9, tzinfo=timezone.utc),
                         vintage_ts=datetime(2026, 6, 9, tzinfo=timezone.utc),
                         n_scenarios=5000, seed=42, config_json="{}")
    session.add(run2)
    session.flush()
    session.add(IaRResult(run_id=run2.run_id, confidence=0.95, horizon="2026-06-09",
                          iar_type="gross", iar_value=99000, ciar_value=99000))
    session.flush()
    alerts = evaluate_latest(session, pf.portfolio_id, cfg)
    assert len(alerts) == 1 and alerts[0].severity == "hard"
    assert alerts[0].result.iar_value == 99000


def test_evaluate_latest_no_runs_is_empty(session):
    pf = get_or_create_portfolio(session, "Empty", "NO2 Wind", "NO2")
    assert evaluate_latest(session, pf.portfolio_id) == []


def test_load_alerts_frame(session):
    cfg = LimitConfig({"default": {"gross": {"remaining_day_eur": 50000}}})
    pf, run = _run_with_iar(session, gross_iar=60000, spread_iar=0)
    evaluate_run(session, run, cfg, breach_ts=VINTAGE)
    session.commit()
    df = load_alerts(session, pf.portfolio_id)
    assert len(df) == 1
    assert df.iloc[0]["severity"] == "hard"
    assert df.iloc[0]["iar_type"] == "gross"
    assert df.iloc[0]["limit_value"] == 50000
