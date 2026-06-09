"""Load realised prices and compute realised imbalance cost (Task 3.1, headless).

Makes the realised-cost path runnable from the command line with REAL data. It
fetches and stores the two settled price series the backtest needs, then computes
the realised settlement cost for the area's portfolio:

    Optimeering internal MarketsApi ──► DAMPrice              (spot, EUR/MWh)
                                   └──► ActualImbalancePrice  (realised, EUR/MWh)
    windsim (already loaded)        ──► DAMPosition + ActualDelivery
                                   ──► realised cost = imbalance × price  (3.1)

.. important::
    The realised price feed uses the **vendored optipyclient wheel** (internal
    SDK). If it isn't installed the fetch fails with a clear pointer — see
    docs/README.md "Setup". For an offline alternative, load CSVs with
    ``flatfile_loader.load_dam_prices`` / ``load_actual_imbalance_prices`` (the
    ``--from-csv`` flow below), or use the dashboard's demo populator.

Date alignment (read this):
    Realised imbalance prices only exist for **settled (past)** MTUs, whereas
    ``scripts/load_windsim_data.py`` generates positions for **today..+N days**.
    For a non-empty realised cost, generate windsim over a *past* window (so its
    MTUs overlap settled prices) and pass a matching ``--start/--end`` here.

Usage:
    python scripts/load_actuals.py --area NO2 --start -P7D --end P0D
    python scripts/load_actuals.py --area NO2 \
        --dam-csv data/uploads/dam_price_NO2.csv \
        --imbalance-csv data/uploads/actual_imbalance_NO2.csv
"""

from __future__ import annotations

import argparse

from iar.db.models import Portfolio
from iar.db.session import DEFAULT_DB_PATH, get_session, init_db
from iar.ingestion.flatfile_loader import (
    load_actual_imbalance_prices,
    load_dam_prices,
    store_actual_imbalance_price_records,
    store_dam_price_records,
)
from iar.ingestion.markets_client import OptimeeringMarketsClient
from iar.risk.realised_cost import compute_realised_cost, realised_period_cost


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Load realised prices + compute realised cost (3.1).")
    ap.add_argument("--area", default="NO2", help="price area (NO1/NO2/SE3)")
    ap.add_argument("--start", default="-P7D", help="window start (ISO datetime or duration)")
    ap.add_argument("--end", default="P0D", help="window end (ISO datetime or duration)")
    ap.add_argument("--dam-csv", default=None,
                    help="offline: load DAM price from this CSV instead of the SDK")
    ap.add_argument("--imbalance-csv", default=None,
                    help="offline: load actual imbalance price from this CSV instead of the SDK")
    return ap.parse_args()


def _portfolio(area: str):
    with get_session() as s:
        pf = (s.query(Portfolio).filter_by(price_area=area)
              .order_by(Portfolio.portfolio_id.desc()).first())
        return (pf.portfolio_id, pf.name) if pf else (None, None)


def load_prices(args) -> tuple[int, int]:
    """Store DAM + actual imbalance prices for the area; return (n_dam, n_imb)."""
    with get_session() as s:
        if args.dam_csv or args.imbalance_csv:
            n_dam = load_dam_prices(s, args.area, args.dam_csv) if args.dam_csv else 0
            n_imb = (load_actual_imbalance_prices(s, args.area, args.imbalance_csv)
                     if args.imbalance_csv else 0)
        else:
            client = OptimeeringMarketsClient()  # raises a clear error if the wheel is missing
            dam = client.get_dam_prices(args.area, start=args.start, end=args.end)
            imb = client.get_imbalance_prices(args.area, start=args.start, end=args.end)
            n_dam = store_dam_price_records(s, args.area, dam)
            n_imb = store_actual_imbalance_price_records(s, args.area, imb)
        s.commit()
    return n_dam, n_imb


def main() -> None:
    args = parse_args()
    init_db()

    pid, pf_name = _portfolio(args.area)
    if pid is None:
        raise SystemExit(
            f"No portfolio loaded for area {args.area!r}. "
            "Run scripts/load_windsim_data.py first (positions + actual delivery)."
        )

    try:
        n_dam, n_imb = load_prices(args)
    except ImportError as exc:
        raise SystemExit(
            f"Realised price fetch needs the vendored optipyclient wheel.\n  {exc}\n"
            "Install it (docs/README.md 'Setup') or pass --dam-csv/--imbalance-csv."
        )

    bar = "=" * 64
    print(bar)
    print(f"Realised cost (3.1) | area={args.area} portfolio='{pf_name}' (#{pid})")
    print(f"stored: DAMPrice={n_dam} rows, ActualImbalancePrice={n_imb} rows -> {DEFAULT_DB_PATH}")
    print(bar)

    with get_session() as s:
        df = compute_realised_cost(s, pid)
        period = realised_period_cost(s, pid)

    if period["n_mtus"] == 0:
        print("No settled MTUs overlap (positions ∩ actuals ∩ DAM price ∩ imbalance price).")
        print("Tip: realised imbalance prices are PAST-settled; generate windsim over a past\n"
              "window so its MTUs overlap, then re-run with a matching --start/--end.")
        return

    print(f"settled MTUs : {period['n_mtus']}  "
          f"({period['first_mtu']:%Y-%m-%d %H:%M} .. {period['last_mtu']:%Y-%m-%d %H:%M} UTC)")
    print(f"Realised GROSS  cost : {period['gross']:+12,.0f} EUR")
    print(f"Realised SPREAD cost : {period['spread']:+12,.0f} EUR")
    print(bar)
    print("Per-MTU head (positive = cost):")
    cols = ["timestamp", "imbalance_mwh", "actual_imbalance_price", "gross_cost", "spread_cost"]
    print(df[cols].head(8).to_string(index=False))
    print(bar)
    print("Sign convention: imbalance = DAM position − actual delivery; "
          "positive cost = bad. Feeds the 3.3 backtest (iar.risk.backtest.run_backtest).")


if __name__ == "__main__":
    main()
