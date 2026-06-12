"""Backfill historical day-ahead IaR estimates over a past window (Task 3.2).

Generates the "estimate-as-of" vintages the Week-3 backtest compares against. For
each delivery day in the window it picks the day-ahead forecast vintage, runs the
Monte Carlo, and persists a SimulationRun (stamped with that vintage) + IaRResults.

Sources (all REAL):
  * imbalance-spread forecast vintages -> Optimeering public SDK (get_historical_prices)
  * DAM (spot) price                   -> Optimeering internal SDK (get_dam_prices)
  * positions / generation             -> the portfolio loaded in the DB (windsim)

.. note::
    Optimeering's historical ``retrieve`` returns *every* vintage in the window
    (large). Keep the window modest (a few days). ``backfill_iar`` selects a single
    day-ahead vintage per day, so memory stays bounded regardless.

Usage:
    python scripts/backfill_iar.py --area NO2 --start -P5D --end P0D
    python scripts/backfill_iar.py --area NO2 --start 2026-06-01 --end 2026-06-07
"""

from __future__ import annotations

import argparse

import pandas as pd

from iar.db.models import DAMPosition, GenerationForecast, Portfolio
from iar.db.session import DEFAULT_DB_PATH, get_session, init_db
from iar.ingestion.clients import get_forecast_client, get_markets_client
from iar.risk.replay import MTU_HOURS, backfill_iar
from iar.simulation.engine import EngineConfig
from iar.simulation.imbalance_model import ImbalanceModelConfig


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Backfill historical day-ahead IaR estimates.")
    ap.add_argument("--area", default="NO2", help="price area (NO1/NO2/SE3)")
    ap.add_argument("--start", default="-P5D", help="window start (ISO datetime or duration)")
    ap.add_argument("--end", default="P0D", help="window end (ISO datetime or duration)")
    ap.add_argument("--scenarios", type=int, default=10_000, help="MC scenarios per run")
    ap.add_argument("--confidence", type=float, default=0.95, help="confidence level (0-1)")
    ap.add_argument("--capacity-mw", type=float, default=100.0, help="installed capacity (sigma basis)")
    ap.add_argument("--sigma-fraction", type=float, default=0.10, help="imbalance sigma fraction")
    ap.add_argument("--dist", choices=["normal", "student_t"], default="normal")
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
        pos = {
            pd.to_datetime(t, utc=True): (dam[t], gen[t]) for t in (set(dam) & set(gen))
        }
        return pos, pf.portfolio_id, pf.name


def main() -> None:
    args = parse_args()

    pos_map, portfolio_id, pf_name = fetch_positions_map(args.area)
    if not pos_map:
        raise SystemExit(
            f"No portfolio with positions+generation loaded for area {args.area!r}. "
            "Run scripts/load_windsim_data.py first."
        )

    forecast_records = get_forecast_client().get_historical_prices(
        args.area, start=args.start, end=args.end
    )
    dam_recs = get_markets_client().get_dam_prices(args.area, start=args.start, end=args.end)
    dam_map = {pd.to_datetime(r["timestamp"], utc=True): float(r["eur_per_mwh"]) for r in dam_recs}

    init_db()
    with get_session() as s:
        runs = backfill_iar(
            s, portfolio_id,
            forecast_records=forecast_records,
            dam_price_map=dam_map,
            position_map=pos_map,
            capacity_mwh=args.capacity_mw * MTU_HOURS,
            model_config=ImbalanceModelConfig(
                dist=args.dist, sigma_fraction=args.sigma_fraction, scale_basis="capacity"
            ),
            engine_config=EngineConfig(
                n_scenarios=args.scenarios, confidence=args.confidence, seed=args.seed
            ),
        )
        s.commit()

        bar = "=" * 64
        print(bar)
        print(f"IaR backfill | area={args.area} portfolio='{pf_name}' (#{portfolio_id}) "
              f"window {args.start}..{args.end}")
        print(f"backfilled {len(runs)} day-ahead estimate(s):")
        for run in runs:
            g = next(r for r in run.results if r.iar_type == "gross")
            sp = next(r for r in run.results if r.iar_type == "spread")
            print(f"  {run.results[0].horizon} | vintage {run.vintage_ts}  "
                  f"Gross IaR {g.iar_value:+,.0f}  Spread IaR {sp.iar_value:+,.0f} EUR")
        print(bar)
        print(f"[stored] -> {DEFAULT_DB_PATH}  (join via iar.risk.backtest.estimate_for_period)")


if __name__ == "__main__":
    main()
