"""Boot-time and live-tick data guard for the hosted / synthetic demo.

Makes the demo self-contained and *living* on a host, where the database is absent
(``iar.db`` is gitignored), windsim is not installed, and there is no scheduled task:

* :func:`ensure_demo_data` - run once at startup. If the database has no simulation
  runs, it builds a full synthetic dataset (positions + day-ahead history + realised
  prices + a live run + backtest). This is the blocking first-load seed.
* :func:`maybe_advance` - call repeatedly from a Streamlit ``run_every`` fragment. When
  the latest run is older than the refresh interval, it runs a fast synthetic forward
  refresh so new MTUs settle, the heatmap fills in, the IaR curve gains points and the
  "as of" clock advances. A short lock prevents concurrent viewers from stampeding.

All work uses the synthetic feeds (``IAR_SYNTHETIC=1``), so nothing external is needed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# First-load seed: kept modest so a cold host paints reasonably quickly, while still
# showing a couple of weeks of backtest history.
BOOT_SEED_DAYS = "14"
BOOT_SEED_SCENARIOS = "3000"

# Live cadence: advance the synthetic data when the newest run is older than this. Ten
# minutes matches the real 15-min settlement rhythm without thrashing.
REFRESH_STALE_MINUTES = 10
_LOCK = PROJECT_ROOT / "data" / "refresh.lock"
_LOCK_TTL_SECONDS = 180


def _run(script: str, *args: str) -> None:
    env = dict(os.environ, IAR_SYNTHETIC="1")
    # Make the `iar` package importable in the subprocess (and its grandchildren, which
    # inherit this env). On a host there is no editable install: the dashboard imports
    # `iar` via a sys.path tweak, but a child process does not inherit that - so without
    # this the seed scripts fail with ModuleNotFoundError and the DB stays empty.
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + existing if existing else "")
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


def _age_minutes(ts) -> float:
    lt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - lt).total_seconds() / 60.0


def ensure_demo_data() -> str:
    """Seed the synthetic database if it is empty (blocking first-load). Returns status."""
    try:
        latest = _latest_run_ts()
    except Exception:
        latest = None
    if latest is None:
        _run("seed_synthetic_demo.py", "--area", "NO2",
             "--days", BOOT_SEED_DAYS, "--scenarios", BOOT_SEED_SCENARIOS)
        return "seeded"
    return "ok"


def maybe_advance() -> str:
    """Advance the synthetic data forward if it is stale. Safe to call every few minutes.

    Returns one of ``"seeded"`` / ``"advanced"`` / ``"fresh"`` / ``"locked"`` / ``"error"``.
    A small lock file debounces concurrent viewers so only one refresh runs at a time.
    """
    try:
        latest = _latest_run_ts()
    except Exception:
        return "error"

    if latest is None:
        return ensure_demo_data()
    if _age_minutes(latest) < REFRESH_STALE_MINUTES:
        return "fresh"

    # Debounce: skip if another viewer started a refresh in the last few minutes.
    try:
        if _LOCK.exists() and (time.time() - _LOCK.stat().st_mtime) < _LOCK_TTL_SECONDS:
            return "locked"
        _LOCK.write_text(datetime.now(timezone.utc).isoformat())
    except Exception:
        pass

    try:
        _run("refresh.py", "--areas", "NO2")
        return "advanced"
    finally:
        try:
            _LOCK.unlink()
        except Exception:
            pass
