"""Ingest a realistic wind portfolio from the windsim simulator — the CLIENT way.

Replaces the synthetic ``make_sample_data`` stubs with real, ground-truth data from
Volue's ``windsim`` (the wind-portfolio spot-market simulator in
``Volue/sirius-imb-at-risk-mvp``). Crucially it routes the data through the **same
flat-file CSV path a real customer uses**:

    windsim DuckDB  ->  data/uploads/*.csv (client upload format)  ->  load_* loaders  ->  iar.db

windsim quantities are per-quarter **MW**; our contract is **MWh per 15-min MTU**, so we
multiply by 0.25 h. Mapping:
    DAM position      <- bids.cleared_mw            (committed day-ahead volume)
    generation forecast <- latest_forecast_for_quarter, summed over parks
    actual delivery   <- actuals.actual_mw, summed over parks

windsim is a dev/data tool installed from the private repo (not on PyPI) — see
docs/README.md Setup. Dates are generated for the current horizon so they overlap the
live Optimeering spread + real DAM price used by ``run_iar.py``.

Usage:
    python scripts/load_windsim_data.py                 # north -> NO2, today..+3 days
    python scripts/load_windsim_data.py --windsim-portfolio offshore_west --area NO2
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from iar.db.session import DEFAULT_DB_PATH, get_session, init_db
from iar.ingestion.flatfile_loader import (
    get_or_create_portfolio,
    load_actual_delivery,
    load_dam_positions,
    load_generation_forecasts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOADS = PROJECT_ROOT / "data" / "uploads"
DEFAULT_DB = PROJECT_ROOT / "data" / "cache" / "windsim.duckdb"
MTU_HOURS = 0.25  # MW -> MWh per 15-min quarter


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Ingest a windsim portfolio via the client CSV path.")
    ap.add_argument("--windsim-portfolio", default="north", help="portfolio name in windsim")
    ap.add_argument("--area", default="NO2", help="our price area to file it under (NO1/NO2/SE3)")
    ap.add_argument("--user", default="Wind Co", help="our user/customer name")
    ap.add_argument("--portfolio-name", default=None, help="our portfolio name (default '<area> Wind')")
    ap.add_argument("--days", type=int, default=4, help="number of days from the start to generate")
    ap.add_argument("--start", default=None,
                    help="windsim start date YYYY-MM-DD (default today; use a past date for history)")
    ap.add_argument("--seed", type=int, default=1, help="windsim seed (deterministic)")
    ap.add_argument("--db", default=str(DEFAULT_DB), help="windsim DuckDB path")
    ap.add_argument("--regen", action="store_true", help="regenerate the windsim DB even if present")
    return ap.parse_args()


def ensure_windsim_db(db_path: Path, days: int, seed: int, regen: bool) -> None:
    """Generate the windsim DuckDB for today..today+days if missing (or --regen)."""
    if db_path.exists() and not regen:
        return
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    start = date.today()
    end = start + timedelta(days=days)
    print(f"Generating windsim data {start}..{end} (seed {seed}) -> {db_path}")
    subprocess.run(
        [sys.executable, "-m", "windsim.cli", "run", "--db", str(db_path),
         "--start", start.isoformat(), "--end", end.isoformat(), "--seed", str(seed)],
        check=True,
    )


def _quarter_mwh(frames: list[pd.DataFrame], mw_col: str) -> pd.Series:
    """Concat per-day frames, sum MW over parks per quarter, convert to MWh per MTU."""
    df = pd.concat(frames, ignore_index=True)
    s = df.groupby("delivery_start")[mw_col].sum().sort_index()
    return s * MTU_HOURS


def build_csvs(db_path: Path, windsim_pf: str, area: str) -> dict[str, Path]:
    """Read windsim, build the three client CSVs in data/uploads/, return their paths."""
    from windsim import Database

    db = Database(str(db_path))
    days = db.get_delivery_days(windsim_pf)
    if not days:
        raise SystemExit(f"windsim has no delivery days for portfolio {windsim_pf!r}.")

    bids = [db.get_bids(windsim_pf, d) for d in days]
    fcs = [db.latest_forecast_for_quarter(windsim_pf, d) for d in days]
    acts = [db.get_actuals(windsim_pf, d) for d in days]

    dam_pos = _quarter_mwh(bids, "cleared_mw")        # committed day-ahead volume (MWh)
    gen = _quarter_mwh(fcs, "forecast_mw")            # latest forecast, summed over parks
    act = _quarter_mwh(acts, "actual_mw")             # realised, summed over parks

    UPLOADS.mkdir(parents=True, exist_ok=True)

    def _write(series: pd.Series, value_col: str, name: str) -> Path:
        out = pd.DataFrame({
            "timestamp": [ts.isoformat() for ts in series.index],
            value_col: series.values.round(4),
        })
        path = UPLOADS / name
        out.to_csv(path, index=False)
        print(f"  wrote {path.name}  ({len(out)} rows)")
        return path

    return {
        "dam": _write(dam_pos, "mwh", f"dam_positions_{area}.csv"),
        "gen": _write(gen, "forecast_mwh", f"generation_forecast_{area}.csv"),
        "act": _write(act, "actual_mwh", f"actual_delivery_{area}.csv"),
    }


def main() -> None:
    args = parse_args()
    db_path = Path(args.db)
    portfolio_name = args.portfolio_name or f"{args.area} Wind"

    ensure_windsim_db(db_path, args.days, args.seed, args.regen)

    print(f"\nExporting windsim portfolio '{args.windsim_portfolio}' -> client CSVs ({args.area}):")
    files = build_csvs(db_path, args.windsim_portfolio, args.area)

    print("\nLoading the CSVs the client way (flat-file loaders -> iar.db):")
    init_db()
    with get_session() as s:
        pf = get_or_create_portfolio(s, args.user, portfolio_name, args.area)
        n_dam = load_dam_positions(s, pf.portfolio_id, files["dam"])
        n_gen = load_generation_forecasts(s, pf.portfolio_id, files["gen"])
        n_act = load_actual_delivery(s, pf.portfolio_id, files["act"])
        s.commit()
        print(f"  portfolio #{pf.portfolio_id} '{pf.name}' ({pf.price_area}) | "
              f"DAM_pos={n_dam}, forecast={n_gen}, actual={n_act} rows -> {DEFAULT_DB_PATH}")

    print("\nDone. Positions/generation/actuals now come from REAL windsim data "
          "(ingested via the client CSV path). [OK]")


if __name__ == "__main__":
    main()
