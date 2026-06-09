"""Run the IaR backtest and print the calibration readout (Task 3.3, headless).

Joins each settled period's realised cost (3.1) to the day-ahead IaR estimate
whose vintage preceded it (3.2), flags exceedances, and runs the Kupiec POF test.
Persists one HistoricalPerformanceRecord per period.

Prerequisites (the inputs this consumes must already be in the DB):
  * day-ahead estimates  -> scripts/backfill_iar.py     (3.2)
  * realised prices/cost -> scripts/load_actuals.py      (3.1)
  * positions/actuals    -> scripts/load_windsim_data.py

Usage:
    python scripts/run_backtest.py --area NO2
    python scripts/run_backtest.py --area NO2 --basis both --significance 0.05
    python scripts/run_backtest.py --area NO2 --basis spread --no-persist
"""

from __future__ import annotations

import argparse

from iar.db.models import Portfolio
from iar.db.session import DEFAULT_DB_PATH, get_session, init_db
from iar.risk.backtest import BacktestResult, run_backtest


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the IaR backtest (3.3) and print calibration.")
    ap.add_argument("--area", default="NO2", help="price area (NO1/NO2/SE3)")
    ap.add_argument("--basis", choices=["gross", "spread", "both"], default="gross",
                    help="IaR basis to test (default gross)")
    ap.add_argument("--significance", type=float, default=0.05,
                    help="Kupiec test significance level (default 0.05)")
    ap.add_argument("--no-persist", action="store_true",
                    help="compute only; do not write HistoricalPerformanceRecord rows")
    return ap.parse_args()


def _portfolio(area: str):
    with get_session() as s:
        pf = (s.query(Portfolio).filter_by(price_area=area)
              .order_by(Portfolio.portfolio_id.desc()).first())
        return (pf.portfolio_id, pf.name) if pf else (None, None)


def _verdict(res: BacktestResult) -> str:
    wc = res.kupiec.well_calibrated
    return "n/a" if wc is None else ("CALIBRATED" if wc else "MIS-CALIBRATED")


def _print_result(res: BacktestResult, persisted: bool) -> None:
    bar = "-" * 64
    k = res.kupiec
    conf = f"{res.confidence:.0%}" if res.confidence is not None else "n/a"
    print(bar)
    print(f"basis: {res.iar_type.upper()}  (confidence {conf})")
    if res.n_periods == 0:
        print("  no settled periods to compare yet — "
              "load realised prices (load_actuals.py) and backfill estimates (backfill_iar.py).")
        return

    df = res.as_frame().copy()
    df["exceeded"] = df["exceeded"].map(lambda v: "EXCEEDED" if v else "within")
    print(df.to_string(index=False,
                       formatters={"iar_estimate": "{:,.0f}".format,
                                   "realised_cost": "{:,.0f}".format}))
    print(bar)
    lr = f"{k.lr_statistic:.3f}" if k.lr_statistic is not None else "n/a"
    pv = f"{k.p_value:.3f}" if k.p_value is not None else "n/a"
    print(f"exceedances    : {res.n_exceedances} / {res.n_periods}  "
          f"(observed {k.observed_rate:.0%}, expected ~{k.expected_rate:.0%})")
    print(f"Kupiec POF     : LR={lr}  p-value={pv}  ->  {_verdict(res)} "
          f"(alpha={k.significance:.0%})")
    print(f"records        : {'persisted' if persisted else 'NOT persisted (--no-persist)'}")


def main() -> None:
    args = parse_args()
    init_db()

    pid, pf_name = _portfolio(args.area)
    if pid is None:
        raise SystemExit(
            f"No portfolio loaded for area {args.area!r}. Run scripts/load_windsim_data.py first."
        )

    bases = ["gross", "spread"] if args.basis == "both" else [args.basis]
    persist = not args.no_persist

    bar = "=" * 64
    print(bar)
    print(f"IaR backtest (3.3) | area={args.area} portfolio='{pf_name}' (#{pid})")
    print(bar)

    with get_session() as s:
        for i, basis in enumerate(bases):
            # When testing both bases, only the first persists (the record schema
            # has no iar_type column, so a second persist would overwrite the first).
            do_persist = persist and (i == 0)
            res = run_backtest(s, pid, basis, significance=args.significance, persist=do_persist)
            _print_result(res, persisted=do_persist)
        s.commit()

    print("=" * 64)
    print(f"stored -> {DEFAULT_DB_PATH}  "
          "(read back via iar.risk.backtest.load_performance_records)")
    if args.basis == "both" and persist:
        print("note: with --basis both, only GROSS records were persisted "
              "(HistoricalPerformanceRecord has no basis column).")


if __name__ == "__main__":
    main()
