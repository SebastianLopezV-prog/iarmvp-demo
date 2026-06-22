"""Continuous refresh orchestrator — keeps the database current (run on a schedule).

This is the single entry point a scheduler (e.g. Windows Task Scheduler, every
15 min) calls so the dashboard never goes stale or drops a day. It is **idempotent**
and rolls forward automatically: each run re-estimates the *current* delivery day,
loads freshly-settled realised prices, and re-runs the backtest.

Per area, every cycle (``--mode fast``, the default):
  1. ``run_iar.py --store``  — live forward IaR for now→end-of-day (cheap; rolls into
     the new day automatically at midnight). This is the *forecast* (future) half of
     the intraday/heatmap.
  2. ``load_actuals.py``     — realised imbalance + DAM prices for recently-settled
     MTUs. This is the *realised* (past) half — what fills the heatmap's morning so
     there's no gap, plus what the backtest needs.
  3. ``run_backtest.py``     — recompute the Kupiec exceedance readout.
Then prune: keep every backfilled (day-ahead history) run but only the **newest live
run per portfolio**, so 15-min cycles don't accumulate thousands of rows.

``--mode full`` additionally runs ``backfill_iar.py`` over ``--backfill-window`` to
extend the historical IaR curve (heavier — schedule this hourly/daily, not every cycle).

Usage:
    python scripts/refresh.py                       # fast cycle, all areas
    python scripts/refresh.py --areas NO2            # one area
    python scripts/refresh.py --mode full --backfill-window=-P3D
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _run(script: str, *args: str) -> bool:
    """Run a pipeline script as a subprocess; return success (never raises)."""
    cmd = [PY, str(PROJECT_ROOT / "scripts" / script), *args]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode == 0


def prune_live_runs() -> int:
    """Keep all replay (backfill) runs + only the newest live run per portfolio.

    Live runs are the frequent ``run_iar.py`` cycles (no ``replay`` flag in their
    config); backfill runs (``replay: true``) are the historical curve and are kept.
    Returns the number of stale live runs deleted.
    """
    from iar.db.models import SimulationRun
    from iar.db.session import get_session, init_db

    init_db()
    deleted = 0
    with get_session() as s:
        runs = (
            s.query(SimulationRun)
            .order_by(SimulationRun.portfolio_id, SimulationRun.run_id.desc())
            .all()
        )
        by_pf: dict[int, list] = defaultdict(list)
        for r in runs:
            by_pf[r.portfolio_id].append(r)
        for rs in by_pf.values():
            live = [r for r in rs if not json.loads(r.config_json or "{}").get("replay")]
            for stale in live[1:]:  # rs is newest-first, so live[0] is the newest live run
                s.delete(stale)
                deleted += 1
        s.commit()
    return deleted


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Refresh the IaR database (scheduled).")
    ap.add_argument("--areas", nargs="+", default=["SE1", "SE2", "SE3", "SE4"])
    ap.add_argument(
        "--mode",
        choices=["fast", "full"],
        default="fast",
        help="fast = live run + actuals + backtest; full also backfills the curve",
    )
    ap.add_argument(
        "--actuals-window",
        default="-P2D",
        help="how far back to pull realised prices each cycle (default 2 days)",
    )
    ap.add_argument(
        "--backfill-window", default="-P3D", help="history window for --mode full backfill"
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    started = datetime.now(UTC)
    print(f"[refresh] {started.isoformat()}  mode={args.mode}  areas={args.areas}")

    for area in args.areas:
        print(f"--- {area}: live IaR ---")
        _run("run_iar.py", "--area", area, "--store")
        print(f"--- {area}: realised prices ---")
        _run("load_actuals.py", "--area", area, f"--start={args.actuals_window}", "--end=P0D")
        if args.mode == "full":
            print(f"--- {area}: backfill curve ---")
            _run("backfill_iar.py", "--area", area, f"--start={args.backfill_window}", "--end=P0D")

    for area in args.areas:
        print(f"--- {area}: backtest ---")
        _run("run_backtest.py", "--area", area, "--basis", "both")

    pruned = prune_live_runs()
    elapsed = (datetime.now(UTC) - started).total_seconds()
    print(f"[refresh] done in {elapsed:.0f}s; pruned {pruned} stale live run(s).")


if __name__ == "__main__":
    main()
