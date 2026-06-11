"""Persist Monte Carlo results to the database (Task 2.4).

The engine (2.3) is deliberately pure (no DB). This module is the bridge: it takes
an :class:`~iar.simulation.engine.IaRReport` and writes it to SQLite as one
``SimulationRun`` row plus two ``IaRResult`` rows (one ``gross``, one ``spread``).

Per the architecture, we **store summaries, not raw scenarios**: the per-scenario
cost vectors in the report are diagnostic only and are not persisted. The run's
``seed`` and ``n_scenarios`` are stored so the exact scenario set can be
regenerated on demand if ever needed.

The ``vintage_ts`` (as-of time of the inputs — e.g. the Optimeering forecast's
event_time) is stamped on the run so the Week-3 backtest can join each settled
period to the estimate whose vintage precedes it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
from sqlalchemy.orm import Session

from iar.db.models import IaRResult, SimulationRun
from iar.simulation.engine import IaRReport


def build_per_mtu(
    report: IaRReport,
    timestamps: list[str],
    dam_position,
    forecast_generation,
) -> dict:
    """Assemble the per-MTU read-off payload for ``persist_report(per_mtu=...)``.

    Shared by the live run (``run_iar.py``) and the backfill (``replay.py``) so both
    persist the same shape: per-MTU gross/spread IaR & CIaR series, the peak/rolling
    scalars, the per-MTU positions, and the (ISO) timestamps. Numbers only.
    """
    dam = np.asarray(dam_position, dtype=float)
    gen = np.asarray(forecast_generation, dtype=float)

    def _block(m) -> dict:
        return {
            "iar": m.iar_per_mtu.tolist(),
            "ciar": m.ciar_per_mtu.tolist(),
            "peak_iar": m.peak_mtu_iar,
            "rolling_iar": m.rolling_iar,
            "rolling_window": m.rolling_window,
        }

    return {
        "timestamps": list(timestamps),
        "position_mwh": dam.tolist(),
        "expected_imbalance_mwh": (dam - gen).tolist(),
        "gross": _block(report.gross),
        "spread": _block(report.spread),
    }


def persist_report(
    session: Session,
    report: IaRReport,
    portfolio_id: int,
    vintage_ts: datetime,
    horizon: str,
    run_ts: datetime | None = None,
    extra_config: dict | None = None,
    per_mtu: dict | None = None,
) -> SimulationRun:
    """Write an :class:`IaRReport` as a ``SimulationRun`` + two ``IaRResult`` rows.

    Parameters
    ----------
    session:
        Active SQLAlchemy session (caller commits).
    report:
        Engine output to persist.
    portfolio_id:
        The portfolio this run belongs to (must exist — FK enforced).
    vintage_ts:
        As-of time of the inputs (e.g. the forecast event_time), timezone-aware.
        Used for backtest vintage joins.
    horizon:
        Label for the simulated horizon (e.g. ``"288xPT15M"`` or ``"2026-06-08"``).
    run_ts:
        When the run was executed; defaults to now (UTC).
    extra_config:
        Optional extra key/values to fold into the stored ``config_json``.
    per_mtu:
        Optional per-MTU IaR detail (timestamps, per-MTU gross/spread IaR series,
        peak/rolling scalars, positions) serialised to ``SimulationRun.per_mtu_json``
        for the dashboard's intraday/heatmap/per-MTU panels. Numbers only — still a
        read-off, not raw scenarios.

    Returns
    -------
    SimulationRun
        The persisted run (with ``run_id`` populated and ``results`` attached).
    """
    if report.seed is None:
        raise ValueError(
            "report.seed is None; set a seed in EngineConfig before persisting "
            "(the seed is stored so the run can be reproduced)."
        )

    run_ts = run_ts or datetime.now(timezone.utc)
    config = {
        "confidence": report.confidence,
        "n_scenarios": report.n_scenarios,
        "seed": report.seed,
    }
    if extra_config:
        config.update(extra_config)

    run = SimulationRun(
        portfolio_id=portfolio_id,
        run_ts=run_ts,
        vintage_ts=vintage_ts,
        n_scenarios=report.n_scenarios,
        seed=report.seed,
        config_json=json.dumps(config, default=str),
        per_mtu_json=json.dumps(per_mtu, default=str) if per_mtu is not None else None,
    )
    session.add(run)
    session.flush()  # assign run.run_id before creating child rows

    for iar_type, measure in (("gross", report.gross), ("spread", report.spread)):
        session.add(
            IaRResult(
                run_id=run.run_id,
                confidence=report.confidence,
                horizon=horizon,
                iar_type=iar_type,
                iar_value=measure.iar,
                ciar_value=measure.ciar,
            )
        )
    session.flush()
    return run
