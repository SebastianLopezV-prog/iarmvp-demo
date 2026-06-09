"""Limits & alerts (Task 3.4).

Evaluates a stored IaR estimate against configurable per-portfolio euro-limits and
raises ``Alert`` rows when it breaches them. Limits live in ``config/limits.toml``
(a ``default`` block plus optional per-portfolio overrides), set in EUR, per IaR
variant (``gross`` / ``spread``) and per limit type.

Severity (from the methodology doc / deck)
------------------------------------------
- **hard**: IaR > limit                — breach; immediate notification.
- **soft**: IaR > ``soft_ratio`` × limit (default 0.80) — advance warning.
- otherwise: within limit, no alert.

Sign convention follows the engine: IaR is in cost terms (**positive = cost**),
so a breach is simply ``iar_value > limit`` — a worst case that is still a net
revenue (negative IaR) never breaches.

Limit types & horizons
-----------------------
``limits.toml`` carries ``per_mtu``, ``rolling_window`` and ``remaining_day``
limits. The MVP engine emits a single **period** IaR per run (a remaining/next-day
figure), so :func:`evaluate_run` defaults to ``remaining_day`` and compares the
run's period IaR to that limit. Per-MTU and rolling-window evaluation will light
up unchanged once the engine also emits those IaR series (the deck's horizons) —
the config and code already accommodate them.

Pure vs persisting: :func:`classify_severity` and :func:`check_run` are pure
(no DB writes) so the UI can show limit *headroom* even when nothing breaches;
:func:`evaluate_run` / :func:`evaluate_latest` persist the breaches as ``Alert``s.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from iar.db.models import Alert, SimulationRun

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIMITS_PATH = PROJECT_ROOT / "config" / "limits.toml"
LIMIT_TYPES = ("per_mtu", "rolling_window", "remaining_day")
SOFT_RATIO = 0.80


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class LimitConfig:
    """Per-portfolio euro-limits with a ``default`` fallback (from ``limits.toml``)."""

    def __init__(self, data: dict):
        self._default: dict = data.get("default", {})
        self._portfolio: dict = data.get("portfolio", {})

    def limit_for(self, portfolio_name: str, variant: str, limit_type: str) -> float | None:
        """EUR limit for ``(portfolio, variant, limit_type)`` — override else default.

        Returns ``None`` if no limit is configured (caller skips that check).
        """
        if limit_type not in LIMIT_TYPES:
            raise ValueError(f"limit_type must be one of {LIMIT_TYPES}, got {limit_type!r}")
        key = f"{limit_type}_eur"
        value = self._default.get(variant, {}).get(key)
        override = self._portfolio.get(portfolio_name, {}).get(variant, {})
        if key in override:
            value = override[key]
        return float(value) if value is not None else None


def load_limits(path: str | Path = DEFAULT_LIMITS_PATH) -> LimitConfig:
    """Load ``limits.toml`` into a :class:`LimitConfig` (stdlib ``tomllib``)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Limits config not found: {p}")
    with p.open("rb") as fh:
        return LimitConfig(tomllib.load(fh))


# --------------------------------------------------------------------------- #
# Pure evaluation
# --------------------------------------------------------------------------- #
def classify_severity(
    iar_value: float, limit_value: float, soft_ratio: float = SOFT_RATIO
) -> str | None:
    """Return ``"hard"``, ``"soft"`` or ``None`` for an IaR vs its euro-limit."""
    if limit_value <= 0:
        return None
    if iar_value > limit_value:
        return "hard"
    if iar_value > soft_ratio * limit_value:
        return "soft"
    return None


@dataclass
class LimitCheck:
    """One IaR-vs-limit evaluation (severity ``None`` ⇒ within limit)."""

    result_id: int | None
    iar_type: str
    limit_type: str
    iar_value: float
    limit_value: float
    utilisation: float       # iar_value / limit_value (1.0 = at the limit)
    severity: str | None


def check_run(
    run: SimulationRun,
    config: LimitConfig | None = None,
    limit_type: str = "remaining_day",
    *,
    soft_ratio: float = SOFT_RATIO,
) -> list[LimitCheck]:
    """Evaluate every IaRResult of ``run`` against ``limit_type`` (no DB writes).

    Returns one :class:`LimitCheck` per result that has a configured limit —
    including within-limit ones (severity ``None``) so callers can show headroom.
    """
    if limit_type not in LIMIT_TYPES:
        raise ValueError(f"limit_type must be one of {LIMIT_TYPES}, got {limit_type!r}")
    config = config or load_limits()
    name = run.portfolio.name
    checks: list[LimitCheck] = []
    for result in run.results:
        limit = config.limit_for(name, result.iar_type, limit_type)
        if limit is None:
            continue
        checks.append(
            LimitCheck(
                result_id=result.result_id,
                iar_type=result.iar_type,
                limit_type=limit_type,
                iar_value=result.iar_value,
                limit_value=limit,
                utilisation=result.iar_value / limit if limit else float("nan"),
                severity=classify_severity(result.iar_value, limit, soft_ratio),
            )
        )
    return checks


# --------------------------------------------------------------------------- #
# Persisting evaluation
# --------------------------------------------------------------------------- #
def evaluate_run(
    session: Session,
    run: SimulationRun,
    config: LimitConfig | None = None,
    limit_type: str = "remaining_day",
    *,
    soft_ratio: float = SOFT_RATIO,
    breach_ts: datetime | None = None,
    persist: bool = True,
    replace: bool = True,
) -> list[Alert]:
    """Raise ``Alert`` rows for ``run``'s IaRResults that breach their limit.

    Parameters
    ----------
    limit_type:
        Which configured limit to compare against (default ``remaining_day``).
    breach_ts:
        Detection time stamped on the alerts (default now, UTC).
    persist:
        If True (default), write the alerts. ``replace`` first deletes existing
        alerts for this run's results, so re-evaluating is idempotent.
    """
    config = config or load_limits()
    breach_ts = breach_ts or datetime.now(timezone.utc)
    checks = check_run(run, config, limit_type, soft_ratio=soft_ratio)

    alerts = [
        Alert(
            portfolio_id=run.portfolio_id,
            result_id=c.result_id,
            limit_type=c.limit_type,
            limit_value=c.limit_value,
            breach_ts=breach_ts,
            severity=c.severity,
        )
        for c in checks
        if c.severity is not None
    ]

    if persist:
        if replace:
            result_ids = [r.result_id for r in run.results]
            if result_ids:
                session.query(Alert).filter(
                    Alert.result_id.in_(result_ids)
                ).delete(synchronize_session=False)
                session.flush()
        session.add_all(alerts)
        session.flush()
    return alerts


def _latest_run(session: Session, portfolio_id: int) -> SimulationRun | None:
    return (
        session.query(SimulationRun)
        .filter(SimulationRun.portfolio_id == portfolio_id)
        .order_by(SimulationRun.run_ts.desc(), SimulationRun.run_id.desc())
        .first()
    )


def evaluate_latest(
    session: Session,
    portfolio_id: int,
    config: LimitConfig | None = None,
    limit_type: str = "remaining_day",
    *,
    soft_ratio: float = SOFT_RATIO,
    breach_ts: datetime | None = None,
    persist: bool = True,
    replace: bool = True,
) -> list[Alert]:
    """Evaluate the portfolio's most recent run against limits (see :func:`evaluate_run`)."""
    run = _latest_run(session, portfolio_id)
    if run is None:
        return []
    return evaluate_run(
        session, run, config, limit_type,
        soft_ratio=soft_ratio, breach_ts=breach_ts, persist=persist, replace=replace,
    )


def load_alerts(session: Session, portfolio_id: int) -> pd.DataFrame:
    """Stored alerts for a portfolio as a tidy frame (newest first)."""
    rows = (
        session.query(Alert)
        .filter(Alert.portfolio_id == portfolio_id)
        .order_by(Alert.breach_ts.desc())
        .all()
    )
    return pd.DataFrame(
        [
            {
                "breach_ts": pd.to_datetime(a.breach_ts, utc=True),
                "iar_type": a.result.iar_type if a.result else None,
                "limit_type": a.limit_type,
                "iar_value": a.result.iar_value if a.result else None,
                "limit_value": a.limit_value,
                "severity": a.severity,
            }
            for a in rows
        ],
        columns=["breach_ts", "iar_type", "limit_type", "iar_value", "limit_value", "severity"],
    )
