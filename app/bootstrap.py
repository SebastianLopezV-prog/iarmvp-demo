"""Boot-time data guard for the hosted / synthetic demo.

Makes the demo self-contained on a fresh host, where the database is absent (``iar.db``
is gitignored) and windsim is not installed:

* if the database has **no simulation runs**, build a full synthetic dataset
  (positions + day-ahead history + realised prices + a live run + backtest);
* if data exists but is **stale** and ``IAR_DEMO_AUTOREFRESH`` is set, run a fast
  forward refresh so the view always looks current.

All work uses the synthetic feeds (``IAR_SYNTHETIC=1``), so nothing external is needed.
Safe to call on every startup: the heavy path runs only when the database is empty.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Lighter than the manual seed so first page load on a cold host stays reasonable.
BOOT_SEED_DAYS = "21"
BOOT_SEED_SCENARIOS = "5000"


def _run(script: str, *args: str) -> None:
    env = dict(os.environ, IAR_SYNTHETIC="1")
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / script), *args],
        cwd=str(PROJECT_ROOT), env=env, check=False,
    )


def _latest_run_ts():
    from iar.db.models import SimulationRun
    from iar.db.session import get_session, init_db

    init_db()
    with get_session() as s:
        run = s.query(SimulationRun).order_by(SimulationRun.run_ts.desc()).first()
        return run.run_ts if run else None


def ensure_demo_data(stale_minutes: int = 90) -> str:
    """Seed on an empty DB; optionally fast-refresh a stale one. Returns a status string."""
    try:
        latest = _latest_run_ts()
    except Exception:
        latest = None

    if latest is None:
        _run("seed_synthetic_demo.py", "--area", "NO2",
             "--days", BOOT_SEED_DAYS, "--scenarios", BOOT_SEED_SCENARIOS)
        return "seeded"

    if os.getenv("IAR_DEMO_AUTOREFRESH"):
        lt = latest if latest.tzinfo else latest.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) - lt > timedelta(minutes=stale_minutes):
            _run("refresh.py", "--areas", "NO2")
            return "refreshed"
    return "ok"
