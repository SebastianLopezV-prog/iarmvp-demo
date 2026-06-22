"""Seed the demo database entirely from SYNTHETIC feeds (no API key, no downloads).

This is the one command to (re)build a fully synthetic demo database. It:

  1. clears the price / forecast / simulation / backtest tables (keeping the windsim
     positions, generation and actual-delivery series already loaded);
  2. backfills ~N days of day-ahead IaR estimates from the synthetic spread forecast;
  3. loads synthetic DAM spot + realised imbalance prices and computes realised cost;
  4. runs one live forward IaR for the current day;
  5. runs the Kupiec backtest (Gross + Spread).

All feeds come from ``iar.ingestion.synthetic`` (this demo has no real-feed path), so
nothing external is contacted. Run it after copying the demo, or any time you want a
clean synthetic dataset.

Usage:
    python scripts/seed_synthetic_demo.py                # NO2, 30 days of history
    python scripts/seed_synthetic_demo.py --area NO2 --days 45
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _run(script: str, *args: str) -> None:
    """Run a pipeline script as a subprocess."""
    env = dict(os.environ)
    cmd = [PY, str(PROJECT_ROOT / "scripts" / script), *args]
    print(f"\n>>> {script} {' '.join(args)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    if result.returncode != 0:
        raise SystemExit(f"{script} failed (exit {result.returncode}).")


def ensure_positions(area: str, days: int, force: bool) -> None:
    """Ensure the area has a wind portfolio with positions.

    On a fresh host (no windsim, no shipped DB) there are no positions, so generate a
    synthetic wind portfolio. Where positions already exist (e.g. local windsim data),
    keep them unless ``force`` is set.
    """
    from iar.db.models import DAMPosition, Portfolio
    from iar.db.session import get_session, init_db
    from iar.ingestion.synthetic import store_synthetic_portfolio

    init_db()
    with get_session() as s:
        pf = (
            s.query(Portfolio)
            .filter_by(price_area=area)
            .order_by(Portfolio.portfolio_id.desc())
            .first()
        )
        has_positions = pf is not None and (
            s.query(DAMPosition).filter_by(portfolio_id=pf.portfolio_id).first() is not None
        )
        if has_positions and not force:
            print(f"  positions already present for {area} (keeping them).")
            return
        pf, n = store_synthetic_portfolio(
            s,
            area,
            start=f"-P{days + 1}D",
            end="P3D",
            user=f"Wind Co {area}",
            portfolio_name=f"{area} Wind",
        )
        s.commit()
        print(f"  generated synthetic wind portfolio '{pf.name}' (#{pf.portfolio_id}): {n} MTUs")


def clear_generated_tables() -> None:
    """Delete generated rows, keeping portfolios and the windsim position/delivery series."""
    from iar.db.models import (
        ActualImbalancePrice,
        Alert,
        DAMPrice,
        HistoricalPerformanceRecord,
        IaRResult,
        ImbalancePriceForecast,
        SimulationRun,
    )
    from iar.db.session import get_session, init_db

    init_db()
    with get_session() as s:
        # Children first, then parents, then the price/forecast feeds.
        for model in (
            Alert,
            IaRResult,
            HistoricalPerformanceRecord,
            SimulationRun,
            ImbalancePriceForecast,
            DAMPrice,
            ActualImbalancePrice,
        ):
            n = s.query(model).delete()
            print(f"  cleared {model.__name__}: {n} rows")
        s.commit()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Seed a fully synthetic demo database.")
    ap.add_argument(
        "--areas",
        nargs="+",
        default=["SE1", "SE2", "SE3", "SE4"],
        help="price areas to seed (default: the four Swedish bidding zones)",
    )
    ap.add_argument("--days", type=int, default=30, help="days of day-ahead history to backfill")
    ap.add_argument("--scenarios", type=int, default=10_000, help="MC scenarios per run")
    ap.add_argument(
        "--fresh-positions",
        action="store_true",
        help="regenerate synthetic positions even if some already exist",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    areas = list(args.areas)
    bar = "=" * 64
    print(bar)
    print(f"Seeding SYNTHETIC demo | areas={areas} days={args.days} scenarios={args.scenarios}")
    print(bar)

    print("\n[1/3] Ensuring a wind portfolio with positions for each area...")
    for area in areas:
        ensure_positions(area, args.days, args.fresh_positions)

    print("\n[2/3] Clearing generated price/run tables once (keeping positions)...")
    clear_generated_tables()

    print("\n[3/3] Per-area: backfill history, synthetic prices, live run, backtest...")
    for i, area in enumerate(areas, 1):
        print(f"\n--- [{i}/{len(areas)}] {area} ---")
        _run(
            "backfill_history.py",
            "--area", area, "--days", str(args.days), "--scenarios", str(args.scenarios),
        )
        _run("load_actuals.py", "--area", area, f"--start=-P{args.days + 1}D", "--end=P0D")
        _run("run_iar.py", "--area", area, "--scenarios", str(args.scenarios), "--store")
        _run("run_backtest.py", "--area", area, "--basis", "both")

    print(
        f"\n{bar}\nDone. The demo database is now fully synthetic "
        f"({len(areas)} zones: {', '.join(areas)}). "
        f"Launch the dashboard with scripts/run_dashboard.bat.\n{bar}"
    )


if __name__ == "__main__":
    main()
