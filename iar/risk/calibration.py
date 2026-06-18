"""Sigma calibration (Option 1 — close the Week-3 backtest loop).

The imbalance model's ``sigma`` (how uncertain the generation forecast is) is the
last hand-set knob: there is no historical forecast-error sample, so it is a
guessed fraction of capacity. This module turns that guess into a **data-informed**
choice by reusing the backtest: it sweeps candidate ``sigma_fraction`` values,
re-derives the day-ahead IaR estimates at each, measures how often realised cost
would have **exceeded** the estimate, and recommends the sigma whose exceedance
rate is closest to the target (≈ ``1 − confidence``, i.e. ~5% at P95).

Why a sweep works: IaR grows monotonically with sigma, so the exceedance rate is
monotonically *decreasing* in sigma — too small ⇒ too many breaches (over-
confident), too large ⇒ none (over-conservative). The sweep makes that trade-off
explicit and picks the calibrated middle.

Honesty note: with only a handful of settled days the exceedance rate moves in
coarse steps (``k / N``) and can't land exactly on 5% — the recommendation is the
best available, and gets sharper as settled history accrues. This is the same
low-power caveat as the Kupiec test. It tunes sigma to fit the *scorecard*; it is
not a substitute for fitting sigma to a real forecast-error distribution (the
post-MVP upgrade, "Option 2").
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from sqlalchemy.orm import Session

from iar.db.models import Portfolio
from iar.risk.realised_cost import realised_period_cost
from iar.risk.replay import DEFAULT_CAPACITY_MWH, estimate_periods
from iar.simulation.engine import EngineConfig
from iar.simulation.imbalance_model import ImbalanceModelConfig

# Default sweep: 2%..40% of capacity in 2% steps.
DEFAULT_SIGMA_GRID = tuple(round(x, 2) for x in np.arange(0.02, 0.401, 0.02))


@dataclass
class CalibrationResult:
    """Outcome of a sigma sweep against the backtest."""

    iar_type: str
    target_rate: float
    n_periods: int
    recommended_sigma_fraction: float | None
    achieved_rate: float | None
    grid: list[tuple[float, float | None]]  # (sigma_fraction, exceedance_rate)
    note: str


def calibrate_sigma(
    session: Session,
    portfolio_id: int,
    *,
    forecast_records: list[dict],
    dam_price_map: dict,
    position_map: dict,
    capacity_mwh: float = DEFAULT_CAPACITY_MWH,
    engine_config: EngineConfig | None = None,
    base_model_config: ImbalanceModelConfig | None = None,
    iar_type: str = "gross",
    target_rate: float | None = None,
    sigma_grid: tuple[float, ...] | None = None,
) -> CalibrationResult:
    """Recommend the ``sigma_fraction`` that best calibrates the backtest.

    For each candidate sigma it re-derives the day-ahead estimates (via
    :func:`iar.risk.replay.estimate_periods`), compares each settled period's
    realised cost (3.1) to the estimate, and computes the exceedance rate. Returns
    the sigma whose rate is closest to ``target_rate`` (ties → the smaller sigma,
    least conservative). Pure read: it does **not** persist anything.

    Parameters mirror :func:`backfill_iar`; ``base_model_config`` supplies the
    distribution/scale settings whose ``sigma_fraction`` is overridden per candidate.
    """
    if iar_type not in ("gross", "spread"):
        raise ValueError(f"iar_type must be 'gross' or 'spread', got {iar_type!r}")
    if session.get(Portfolio, portfolio_id) is None:
        raise ValueError(f"Portfolio id {portfolio_id} does not exist.")

    engine_config = engine_config or EngineConfig()
    base_model_config = base_model_config or ImbalanceModelConfig()
    target_rate = target_rate if target_rate is not None else (1.0 - engine_config.confidence)
    grid = tuple(sigma_grid) if sigma_grid else DEFAULT_SIGMA_GRID

    grid_rates: list[tuple[float, float | None]] = []
    n_periods_seen = 0
    for sigma in grid:
        model_config = replace(base_model_config, sigma_fraction=sigma)
        estimates = estimate_periods(
            forecast_records=forecast_records,
            dam_price_map=dam_price_map,
            position_map=position_map,
            capacity_mwh=capacity_mwh,
            model_config=model_config,
            engine_config=engine_config,
        )
        settled = 0
        exceedances = 0
        for est in estimates:
            realised = realised_period_cost(
                session,
                portfolio_id,
                start=est.day_start.isoformat(),
                end=est.day_end.isoformat(),
            )
            if realised["n_mtus"] == 0:
                continue
            settled += 1
            iar_value = getattr(est.report, iar_type).iar
            if realised[iar_type] > iar_value:
                exceedances += 1
        n_periods_seen = max(n_periods_seen, settled)
        grid_rates.append((sigma, (exceedances / settled) if settled else None))

    # Pick the sigma whose rate is closest to target; ties -> smaller sigma.
    scored = [(s, r) for s, r in grid_rates if r is not None]
    if not scored:
        return CalibrationResult(
            iar_type=iar_type,
            target_rate=target_rate,
            n_periods=0,
            recommended_sigma_fraction=None,
            achieved_rate=None,
            grid=grid_rates,
            note="No settled periods to calibrate against — load realised prices "
            "(load_actuals.py) over a window that overlaps the positions.",
        )

    best_sigma, best_rate = min(scored, key=lambda sr: (abs(sr[1] - target_rate), sr[0]))
    note = (
        f"Calibrated on {n_periods_seen} settled period(s) — coarse with few periods "
        f"(rate moves in steps of 1/{n_periods_seen}); refine as history grows."
    )
    return CalibrationResult(
        iar_type=iar_type,
        target_rate=target_rate,
        n_periods=n_periods_seen,
        recommended_sigma_fraction=best_sigma,
        achieved_rate=best_rate,
        grid=grid_rates,
        note=note,
    )
