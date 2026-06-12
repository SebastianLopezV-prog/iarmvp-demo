"""Calibrate the imbalance sigma against the backtest (Option 1, headless).

Sweeps candidate ``sigma_fraction`` values, re-derives the day-ahead IaR estimates
at each, and recommends the one whose realised exceedance rate is closest to the
target (~5% at P95). Read-only: prints a recommendation, persists nothing.

Prerequisites (same data the backtest needs):
  * positions/actuals    -> scripts/load_windsim_data.py
  * realised prices/cost -> scripts/load_actuals.py     (over a settled window)

Usage:
    python scripts/calibrate_sigma.py --area NO2 --start=-P7D --end=P0D
    python scripts/calibrate_sigma.py --area NO2 --basis spread --confidence 0.99
"""

from __future__ import annotations

import argparse

import pandas as pd

from iar.db.models import DAMPosition, GenerationForecast, Portfolio
from iar.db.session import get_session, init_db
from iar.ingestion.clients import get_forecast_client, get_markets_client
from iar.risk.calibration import calibrate_sigma
from iar.risk.replay import MTU_HOURS
from iar.simulation.engine import EngineConfig


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Calibrate imbalance sigma against the backtest.")
    ap.add_argument("--area", default="NO2", help="price area (NO1/NO2/SE3)")
    ap.add_argument("--start", default="-P7D", help="forecast/DAM window start (ISO or duration)")
    ap.add_argument("--end", default="P0D", help="forecast/DAM window end (ISO or duration)")
    ap.add_argument("--basis", choices=["gross", "spread"], default="gross")
    ap.add_argument("--confidence", type=float, default=0.95, help="confidence (sets target ~1-c)")
    ap.add_argument("--scenarios", type=int, default=5000, help="MC scenarios per candidate")
    ap.add_argument("--capacity-mw", type=float, default=100.0, help="capacity (sigma basis)")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def fetch_positions_map(area: str):
    init_db()
    with get_session() as s:
        pf = (s.query(Portfolio).filter_by(price_area=area)
              .order_by(Portfolio.portfolio_id.desc()).first())
        if pf is None:
            return {}, None, None
        dam = {r.timestamp: r.mwh for r in s.query(DAMPosition).filter_by(portfolio_id=pf.portfolio_id)}
        gen = {r.timestamp: r.forecast_mwh
               for r in s.query(GenerationForecast).filter_by(portfolio_id=pf.portfolio_id)}
        pos = {pd.to_datetime(t, utc=True): (dam[t], gen[t]) for t in (set(dam) & set(gen))}
        return pos, pf.portfolio_id, pf.name


def main() -> None:
    args = parse_args()
    pos_map, pid, pf_name = fetch_positions_map(args.area)
    if not pos_map:
        raise SystemExit(f"No portfolio with positions loaded for {args.area!r}. "
                         "Run scripts/load_windsim_data.py first.")

    forecast_records = get_forecast_client().get_historical_prices(
        args.area, start=args.start, end=args.end
    )
    dam_recs = get_markets_client().get_dam_prices(args.area, start=args.start, end=args.end)
    dam_map = {pd.to_datetime(r["timestamp"], utc=True): float(r["eur_per_mwh"]) for r in dam_recs}

    init_db()
    with get_session() as s:
        res = calibrate_sigma(
            s, pid,
            forecast_records=forecast_records, dam_price_map=dam_map, position_map=pos_map,
            capacity_mwh=args.capacity_mw * MTU_HOURS,
            engine_config=EngineConfig(n_scenarios=args.scenarios,
                                       confidence=args.confidence, seed=args.seed),
            iar_type=args.basis,
        )

    bar = "=" * 64
    print(bar)
    print(f"Sigma calibration | area={args.area} portfolio='{pf_name}' (#{pid}) basis={args.basis}")
    print(f"target exceedance rate ~ {res.target_rate:.0%}   settled periods: {res.n_periods}")
    print(bar)
    print("sigma_fraction -> exceedance rate")
    for sigma, rate in res.grid:
        marker = "  <-- recommended" if sigma == res.recommended_sigma_fraction else ""
        rate_s = "   n/a" if rate is None else f"{rate:6.0%}"
        print(f"  {sigma:5.0%}   {rate_s}{marker}")
    print(bar)
    if res.recommended_sigma_fraction is None:
        print(res.note)
    else:
        print(f"RECOMMENDED sigma_fraction = {res.recommended_sigma_fraction:.0%} "
              f"(achieved rate {res.achieved_rate:.0%})")
        print(res.note)
        print("Apply it: set [imbalance_model].sigma_fraction in config/app.toml "
              "(or the dashboard slider).")


if __name__ == "__main__":
    main()
