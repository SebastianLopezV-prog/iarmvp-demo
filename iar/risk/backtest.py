"""Backtesting: vintage join (Task 3.2) and exceedance/calibration (Task 3.3).

Task 3.1 (realised imbalance cost) lives in :mod:`iar.risk.realised_cost`; the
estimate backfill (the other half of 3.2) lives in :mod:`iar.risk.replay`.

The comparison join (3.2)
-------------------------
:func:`estimate_for_period` ties the two together: given a settled period, which
stored IaR estimate should it be judged against? Per the architecture's vintage
rule, the answer is *the most recent estimate whose ``vintage_ts`` is at or before
the period start* — an estimate that used no information from inside the period.

The calibration test (3.3)
--------------------------
:func:`run_backtest` walks each settled period, compares its realised cost (3.1)
to the joined day-ahead estimate, and flags an **exceedance** when the realised
cost is worse than the IaR estimate (``realised_cost > iar_value`` — recall the
engine's convention: positive = cost). For a well-calibrated model at confidence
``c`` the exceedance rate should be ≈ ``1 − c`` (≈5% at P95).

:func:`kupiec_pof` formalises that with the Kupiec Proportion-of-Failures
likelihood-ratio test: under H0 (correct calibration) the statistic is
χ²(1)-distributed, so a small p-value means the observed breach rate is
implausible under the model. Results are persisted one
``HistoricalPerformanceRecord`` per period (estimate, realised cost, exceedance,
and the series-level Kupiec statistic).

Power caveat: the MVP's backtest windows are short (a handful of settled days),
so the Kupiec test has very low power — treat it as a wiring-correct calibration
*readout*, not a verdict, until more settled history accrues.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
from scipy.stats import chi2
from sqlalchemy.orm import Session

from iar.db.models import HistoricalPerformanceRecord, IaRResult, SimulationRun
from iar.risk.realised_cost import realised_period_cost


# --------------------------------------------------------------------------- #
# Task 3.2 — the comparison join
# --------------------------------------------------------------------------- #
def estimate_for_period(
    session: Session,
    portfolio_id: int,
    period_start: datetime | str,
) -> SimulationRun | None:
    """Return the IaR estimate applicable to a period starting at ``period_start``.

    The chosen run is the one for ``portfolio_id`` with the **latest
    ``vintage_ts`` at or before ``period_start``** — the estimate that was current
    just before the period began. Returns ``None`` if no estimate precedes it.

    ``period_start`` is normalised to UTC; estimates are stored with UTC vintages,
    so the comparison is timezone-consistent.
    """
    ps = pd.to_datetime(period_start, utc=True).to_pydatetime()
    return (
        session.query(SimulationRun)
        .filter(
            SimulationRun.portfolio_id == portfolio_id,
            SimulationRun.vintage_ts <= ps,
        )
        .order_by(SimulationRun.vintage_ts.desc())
        .first()
    )


def iar_estimate_for_period(
    session: Session,
    portfolio_id: int,
    period_start: datetime | str,
    iar_type: str,
) -> IaRResult | None:
    """The ``gross``/``spread`` :class:`IaRResult` joined to ``period_start``.

    Convenience over :func:`estimate_for_period` that drills into the run and
    returns the single result of the requested ``iar_type`` (or ``None`` if no
    estimate precedes the period).
    """
    if iar_type not in ("gross", "spread"):
        raise ValueError(f"iar_type must be 'gross' or 'spread', got {iar_type!r}")
    run = estimate_for_period(session, portfolio_id, period_start)
    if run is None:
        return None
    for result in run.results:
        if result.iar_type == iar_type:
            return result
    return None


# --------------------------------------------------------------------------- #
# Task 3.3 — Kupiec POF test
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KupiecResult:
    """Kupiec Proportion-of-Failures likelihood-ratio test outcome."""

    n_obs: int
    n_exceedances: int
    observed_rate: float
    expected_rate: float
    lr_statistic: float | None  # χ²(1); None if no observations
    p_value: float | None
    reject_h0: bool | None      # True ⇒ calibration rejected at ``significance``
    significance: float

    @property
    def well_calibrated(self) -> bool | None:
        """True when H0 (correct calibration) is *not* rejected."""
        return None if self.reject_h0 is None else (not self.reject_h0)


def kupiec_pof(
    n_obs: int,
    n_exceedances: int,
    expected_rate: float,
    significance: float = 0.05,
) -> KupiecResult:
    """Kupiec POF test for whether ``n_exceedances`` is consistent with ``expected_rate``.

    The likelihood ratio is

        LR_POF = −2 · [ ln L(p) − ln L(π̂) ]

    with ``p`` the model's expected failure rate (= ``1 − confidence``) and
    ``π̂ = x / N`` the observed rate. Under H0 it is χ²(1)-distributed; H0 is
    rejected (model mis-calibrated) when the p-value falls below ``significance``.

    Numerically guarded for the ``x = 0`` / ``x = N`` corners (``0·ln 0 ≡ 0``).
    Returns a :class:`KupiecResult` with everything None-safe for ``n_obs == 0``.
    """
    if not (0.0 < expected_rate < 1.0):
        raise ValueError(f"expected_rate must be in (0, 1), got {expected_rate}")
    if n_obs == 0:
        return KupiecResult(0, 0, 0.0, expected_rate, None, None, None, significance)

    x, n = int(n_exceedances), int(n_obs)
    pi_hat = x / n

    def _term(count: int, prob: float) -> float:
        return 0.0 if count == 0 else count * math.log(prob)  # 0·ln0 ≡ 0

    log_l_null = _term(n - x, 1.0 - expected_rate) + _term(x, expected_rate)
    log_l_mle = _term(n - x, 1.0 - pi_hat) + _term(x, pi_hat)
    lr = max(-2.0 * (log_l_null - log_l_mle), 0.0)
    p_value = float(chi2.sf(lr, df=1))
    return KupiecResult(
        n_obs=n,
        n_exceedances=x,
        observed_rate=pi_hat,
        expected_rate=expected_rate,
        lr_statistic=lr,
        p_value=p_value,
        reject_h0=p_value < significance,
        significance=significance,
    )


# --------------------------------------------------------------------------- #
# Task 3.3 — the backtest itself
# --------------------------------------------------------------------------- #
@dataclass
class PeriodOutcome:
    """One backtested period: estimate vs realised, plus the exceedance flag."""

    period: str
    iar_estimate: float
    realised_cost: float
    exceeded: bool


@dataclass
class BacktestResult:
    """Full backtest readout for a portfolio on one IaR basis."""

    portfolio_id: int
    iar_type: str
    confidence: float | None
    kupiec: KupiecResult
    periods: list[PeriodOutcome] = field(default_factory=list)

    @property
    def n_periods(self) -> int:
        return len(self.periods)

    @property
    def n_exceedances(self) -> int:
        return sum(p.exceeded for p in self.periods)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "period": p.period,
                    "iar_estimate": p.iar_estimate,
                    "realised_cost": p.realised_cost,
                    "exceeded": p.exceeded,
                }
                for p in self.periods
            ],
            columns=["period", "iar_estimate", "realised_cost", "exceeded"],
        )

    def summary(self) -> dict:
        k = self.kupiec
        return {
            "portfolio_id": self.portfolio_id,
            "iar_type": self.iar_type,
            "confidence": self.confidence,
            "n_periods": self.n_periods,
            "n_exceedances": self.n_exceedances,
            "observed_rate": k.observed_rate,
            "expected_rate": k.expected_rate,
            "kupiec_lr": k.lr_statistic,
            "kupiec_p_value": k.p_value,
            "well_calibrated": k.well_calibrated,
        }


def _replay_periods(session: Session, portfolio_id: int) -> dict[str, tuple[str, str]]:
    """``{delivery_day -> (period_start, period_end)}`` from stored replay runs."""
    runs = (
        session.query(SimulationRun)
        .filter(SimulationRun.portfolio_id == portfolio_id)
        .all()
    )
    periods: dict[str, tuple[str, str]] = {}
    for run in runs:
        cfg = json.loads(run.config_json or "{}")
        ps, pe = cfg.get("period_start"), cfg.get("period_end")
        if not (ps and pe) or not run.results:
            continue
        periods[run.results[0].horizon] = (ps, pe)
    return periods


def run_backtest(
    session: Session,
    portfolio_id: int,
    iar_type: str = "gross",
    *,
    significance: float = 0.05,
    persist: bool = True,
    replace: bool = True,
) -> BacktestResult:
    """Compare realised cost to the day-ahead IaR estimate over all settled periods.

    For every replayed period (3.2) that now has realised cost (3.1), record
    whether the realised cost exceeded the estimate, then run the Kupiec POF test
    over the exceedance series. Optionally persist one
    ``HistoricalPerformanceRecord`` per period.

    Parameters
    ----------
    iar_type:
        ``"gross"`` or ``"spread"`` — which IaR basis to test.
    significance:
        Test level for the Kupiec verdict (default 0.05).
    persist:
        If True (default), write ``HistoricalPerformanceRecord`` rows (the
        series-level Kupiec statistic is stamped on each).
    replace:
        If True (default) and persisting, delete the portfolio's existing
        performance records first (idempotent re-runs). NB: the record schema has
        no ``iar_type`` column, so stored records reflect this call's basis.

    Returns
    -------
    BacktestResult
        Per-period outcomes + the Kupiec readout (always returned, even when no
        period has settled yet — in which case it is empty).
    """
    if iar_type not in ("gross", "spread"):
        raise ValueError(f"iar_type must be 'gross' or 'spread', got {iar_type!r}")

    periods = _replay_periods(session, portfolio_id)
    outcomes: list[PeriodOutcome] = []
    confidence: float | None = None

    for label, (ps, pe) in sorted(periods.items()):
        realised = realised_period_cost(session, portfolio_id, start=ps, end=pe)
        if realised["n_mtus"] == 0:
            continue  # period not settled yet — nothing to compare
        estimate = iar_estimate_for_period(session, portfolio_id, ps, iar_type)
        if estimate is None:
            continue
        confidence = confidence or estimate.confidence
        realised_cost = float(realised[iar_type])
        outcomes.append(
            PeriodOutcome(
                period=label,
                iar_estimate=float(estimate.iar_value),
                realised_cost=realised_cost,
                exceeded=realised_cost > estimate.iar_value,
            )
        )

    expected_rate = (1.0 - confidence) if confidence is not None else significance
    kupiec = kupiec_pof(
        n_obs=len(outcomes),
        n_exceedances=sum(o.exceeded for o in outcomes),
        expected_rate=expected_rate,
        significance=significance,
    )

    if persist:
        if replace:
            (
                session.query(HistoricalPerformanceRecord)
                .filter(HistoricalPerformanceRecord.portfolio_id == portfolio_id)
                .delete()
            )
            session.flush()
        for o in outcomes:
            session.add(
                HistoricalPerformanceRecord(
                    portfolio_id=portfolio_id,
                    period=o.period,
                    iar_estimate=o.iar_estimate,
                    realised_cost=o.realised_cost,
                    exceeded=o.exceeded,
                    kupiec_stat=kupiec.lr_statistic,
                )
            )
        session.flush()

    return BacktestResult(
        portfolio_id=portfolio_id,
        iar_type=iar_type,
        confidence=confidence,
        kupiec=kupiec,
        periods=outcomes,
    )


def load_performance_records(session: Session, portfolio_id: int) -> pd.DataFrame:
    """Stored ``HistoricalPerformanceRecord`` rows for a portfolio as a tidy frame."""
    rows = (
        session.query(HistoricalPerformanceRecord)
        .filter(HistoricalPerformanceRecord.portfolio_id == portfolio_id)
        .order_by(HistoricalPerformanceRecord.period)
        .all()
    )
    return pd.DataFrame(
        [
            {
                "period": r.period,
                "iar_estimate": r.iar_estimate,
                "realised_cost": r.realised_cost,
                "exceeded": r.exceeded,
                "kupiec_stat": r.kupiec_stat,
            }
            for r in rows
        ],
        columns=["period", "iar_estimate", "realised_cost", "exceeded", "kupiec_stat"],
    )
