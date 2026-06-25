"""Top up portfolio positions forward so live runs never go empty.

The forward IaR needs DAM positions covering the forecast window. Positions come from
windsim (or, on the synthetic demo, from the synthetic generator) and only ever extend to a
fixed date, so once "today" passes that date every run aborts with "No overlapping MTUs".
This script extends each area's positions out to ``today + --ahead-days`` whenever they would
run short within ``--buffer-days``. It is idempotent (a per-area seed keeps the regenerated
history stable) and safe to run on a schedule (e.g. weekly) or by hand.

Usage:
    python scripts/topup_positions.py                          # all default areas
    python scripts/topup_positions.py --areas NO2 SE1 --ahead-days 60
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta

import pandas as pd

from iar.db.models import DAMPosition, Portfolio
from iar.db.session import get_session, init_db

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Deterministic per-area seed so a regenerated window reproduces the same history.
SEED_MAP = {"NO2": 1, "SE1": 2, "SE2": 3, "SE3": 4, "SE4": 5, "NO1": 6, "NO3": 7, "NO4": 8, "NO5": 9, "FI": 10}
DEFAULT_START = "2026-05-10"


def _latest_position(area: str):
    """Return ``(latest_ts, earliest_ts, portfolio)`` for the most-recent portfolio in the
    area that has positions, or ``(None, None, None)``."""
    with get_session() as s:
        pfs = (
            s.query(Portfolio)
            .filter_by(price_area=area)
            .order_by(Portfolio.portfolio_id.desc())
            .all()
        )
        for pf in pfs:
            q = s.query(DAMPosition).filter_by(portfolio_id=pf.portfolio_id)
            ts = [r.timestamp for r in q]
            if ts:
                return max(ts), min(ts), pf
        return None, None, None


def _topup_synthetic(area: str, start: str, end: str) -> bool:
    """Demo path: regenerate synthetic positions for the area (no windsim/network)."""
    from iar.ingestion.synthetic import store_synthetic_portfolio

    with get_session() as s:
        pf, n = store_synthetic_portfolio(
            s, area, start=start, end=end, user=f"Wind Co {area}", portfolio_name=f"{area} Wind"
        )
        s.commit()
        print(f"  [{area}] synthetic positions regenerated to {end}: {n} MTUs (pf #{pf.portfolio_id})")
    return True


def _topup_windsim(area: str, start_date: str, days: int) -> bool:
    """Live path: regenerate windsim for [start_date .. start_date+days] and load it."""
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "scripts", "load_windsim_data.py"),
        "--area", area, "--user", f"Wind Co {area}",
        f"--start={start_date}", "--days", str(days),
        "--seed", str(SEED_MAP.get(area, 1)), "--regen",
    ]
    print(f"  [{area}] regenerating windsim {start_date} for {days} days...")
    r = subprocess.run(cmd, cwd=PROJECT_ROOT, env=dict(os.environ))
    return r.returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Extend portfolio positions forward.")
    ap.add_argument("--areas", nargs="+", default=["NO2", "SE1", "SE2", "SE3", "SE4"])
    ap.add_argument("--ahead-days", type=int, default=90, help="extend positions this far ahead")
    ap.add_argument(
        "--buffer-days", type=int, default=21, help="top up when positions run out within this"
    )
    ap.add_argument("--force", action="store_true", help="top up even if not short")
    args = ap.parse_args()
    init_db()

    # The synthetic generator (demo) is preferred when present; else windsim (live).
    try:
        import iar.ingestion.synthetic  # noqa: F401

        synthetic = True
    except ImportError:
        synthetic = False
    print(f"Position top-up | mode={'synthetic' if synthetic else 'windsim'} | areas={args.areas}")

    now = datetime.now(UTC).replace(tzinfo=None)
    target = now + timedelta(days=args.ahead_days)
    for area in args.areas:
        latest, earliest, pf = _latest_position(area)
        if latest is None:
            print(f"  [{area}] no positions yet - skipping (seed it first).")
            continue
        short_by = (latest - now).days
        if not args.force and short_by > args.buffer_days:
            print(f"  [{area}] OK - covered to {latest:%Y-%m-%d} ({short_by} days ahead).")
            continue
        print(f"  [{area}] covered only to {latest:%Y-%m-%d} ({short_by} days) - extending...")
        if synthetic:
            back = max(1, (now - (earliest or now)).days)
            _topup_synthetic(area, start=f"-P{back}D", end=f"P{args.ahead_days}D")
        else:
            start_date = (earliest or pd.Timestamp(DEFAULT_START)).strftime("%Y-%m-%d")
            days = (target - (earliest or now)).days + 1
            _topup_windsim(area, start_date, days)
    print("Done.")


if __name__ == "__main__":
    main()
