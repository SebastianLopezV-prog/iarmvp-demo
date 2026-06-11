"""Backfill day-ahead IaR estimates over a multi-week window, efficiently.

Unlike ``backfill_iar.py`` (which bulk-fetches every forecast vintage in the window,
roughly 90 to 180 MB per two days), this fetches **one day-ahead vintage per day** via
``retrieve_latest`` with ``max_event_time`` set to each day's start. Each call returns a
single small vintage, so a 30-day window stays light and fast.

It then feeds those per-day vintages into the same replay engine (``backfill_iar``), which
picks each delivery day's day-ahead vintage and persists a ``SimulationRun`` per day. Pair
with ``load_windsim_data.py --start ...`` (positions/delivery), ``load_actuals.py`` (realised
prices) and ``run_backtest.py`` to extend the Kupiec backtest history.

Usage:
    python scripts/backfill_history.py --area NO2 --days 30
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

import pandas as pd

from iar.db.models import DAMPosition, GenerationForecast, Portfolio
from iar.db.session import DEFAULT_DB_PATH, get_session, init_db
from iar.ingestion.clients import get_forecast_client, get_markets_client
from iar.risk.replay import MTU_HOURS, backfill_iar
from iar.simulation.engine import EngineConfig
from iar.simulation.imbalance_model import ImbalanceModelConfig


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Backfill day-ahead IaR estimates over many days.")
    ap.add_argument("--area", default="NO2", help="price area")
    ap.add_argument("--days", type=int, default=30, help="number of past days to backfill")
    ap.add_argument("--scenarios", type=int, default=10_000, help="MC scenarios per day")
    ap.add_argument("--confidence", type=float, default=0.95, help="confidence level")
    ap.add_argument("--capacity-mw", type=float, default=100.0, help="capacity (sigma basis)")
    ap.add_argument("--sigma-fraction", type=float, default=0.10, help="imbalance sigma fraction")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    return ap.parse_args()


def fetch_positions_map(area: str):
    """Return ({ts(UTC): (dam_pos_mwh, gen_mwh)}, portfolio_id, name) for the area's portfolio."""
    init_db()
    with get_session() as s:
        pf = (
            s.query(Portfolio).filter_by(price_area=area)
            .order_by(Portfolio.portfolio_id.desc()).first()
        )
        if pf is None:
            return {}, None, None
        dam = {r.timestamp: r.mwh for r in s.query(DAMPosition).filter_by(portfolio_id=pf.portfolio_id)}
        gen = {
            r.timestamp: r.forecast_mwh
            for r in s.query(GenerationForecast).filter_by(portfolio_id=pf.portfolio_id)
        }
        pos = {pd.to_datetime(t, utc=True): (dam[t], gen[t]) for t in (set(dam) & set(gen))}
        return pos, pf.portfolio_id, pf.name


def main() -> None:
    args = parse_args()

    pos_map, portfolio_id, pf_name = fetch_positions_map(args.area)
    if not pos_map:
        raise SystemExit(
            f"No portfolio with positions loaded for area {args.area!r}. "
            "Run scripts/load_windsim_data.py --start <past date> first."
        )

    # Real DAM (spot) price over the window (one historical time-series call).
    dam_recs = get_markets_client().get_dam_prices(
        args.area, start=f"-P{args.days + 1}D", end="P2D"
    )
    dam_map = {pd.to_datetime(r["timestamp"], utc=True): float(r["eur_per_mwh"]) for r in dam_recs}

    # One day-ahead forecast vintage per day (max_event_time = that day's UTC start).
    fc = get_forecast_client()
    now = datetime.now(timezone.utc)
    records: list[dict] = []
    for k in range(args.days, -1, -1):
        day_start = (now - timedelta(days=k)).replace(hour=0, minute=0, second=0, microsecond=0)
        recs = fc.get_imbalance_price_forecast(
            args.area, max_event_time=day_start.isoformat(), use_cache=True
        )
        records.extend(recs)
        vint = next((r["vintage_ts"] for r in recs if r.get("vintage_ts")), None)
        print(f"  day-ahead as of {day_start.date()}  vintage={vint}  records={len(recs)}")

    print(f"\nFetched {len(records)} forecast records across {args.days + 1} day-ahead vintages.")

    init_db()
    with get_session() as s:
        runs = backfill_iar(
            s, portfolio_id,
            forecast_records=records,
            dam_price_map=dam_map,
            position_map=pos_map,
            capacity_mwh=args.capacity_mw * MTU_HOURS,
            model_config=ImbalanceModelConfig(
                dist="normal", sigma_fraction=args.sigma_fraction, scale_basis="capacity"
            ),
            engine_config=EngineConfig(
                n_scenarios=args.scenarios, confidence=args.confidence, seed=args.seed
            ),
        )
        s.commit()
        print(f"[stored] {len(runs)} day-ahead IaR estimate(s) for '{pf_name}' "
              f"(#{portfolio_id}) -> {DEFAULT_DB_PATH}")


if __name__ == "__main__":
    main()
