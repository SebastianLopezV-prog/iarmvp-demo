"""Seed a clean multi-area demo: one portfolio each for NO1, NO2, SE3 (Week-4 prep).

Resets portfolios (and their positions/runs/alerts) to a tidy one-per-area set so
the dashboard's area/portfolio selector and the end-to-end test (4.3) have real
data across NO1/NO2/SE3 — and the earlier duplicate NO2 portfolios are cleared.

Area-keyed price tables (``dam_prices``, ``actual_imbalance_prices``,
``imbalance_price_forecasts``) are NOT portfolio-scoped, so they survive the reset;
windsim regenerates the same dates deterministically, so any realised prices you
loaded (e.g. NO2 via ``load_actuals.py``) stay aligned for the backtest.

Each area is filled from the same windsim portfolio (synthetic, MVP) under a
distinct user, reusing the existing windsim DuckDB if present.

Usage:
    python scripts/seed_demo.py
    python scripts/seed_demo.py --areas NO2 SE3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from iar.db.models import Portfolio, User
from iar.db.session import get_session, init_db

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Seed a clean one-portfolio-per-area demo.")
    ap.add_argument("--areas", nargs="+", default=["NO2"],
                    help="price areas to seed (MVP focuses on NO2 only)")
    ap.add_argument("--windsim-portfolio", default="north", help="windsim source portfolio")
    ap.add_argument("--keep", action="store_true",
                    help="do not wipe existing portfolios first")
    return ap.parse_args()


def reset_portfolios() -> int:
    """Delete all users via the ORM so cascades remove portfolios + their children.

    (A bulk ``query(User).delete()`` bypasses ORM cascade and trips the FK
    constraint; deleting mapped objects lets SQLAlchemy order the child deletes.)
    """
    init_db()
    with get_session() as s:
        users = s.query(User).all()
        n = len(users)
        for u in users:
            s.delete(u)
        s.commit()
    return n


def main() -> None:
    args = parse_args()

    if not args.keep:
        n = reset_portfolios()
        print(f"[reset] cleared {n} user/portfolio(s); area-keyed price tables kept.")

    for area in args.areas:
        print(f"\n=== seeding {area} ===")
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "load_windsim_data.py"),
             "--area", area, "--user", f"Wind Co {area}",
             "--windsim-portfolio", args.windsim_portfolio],
            check=True,
        )

    init_db()
    with get_session() as s:
        pfs = s.query(Portfolio).order_by(Portfolio.portfolio_id).all()
        print("\n" + "=" * 56)
        print("Seeded portfolios:")
        for p in pfs:
            print(f"  #{p.portfolio_id}  {p.name}  ({p.price_area})")
        print("=" * 56)
    print("Tip: realised prices exist only where load_actuals.py has run (NO2 here); "
          "use the dashboard 'Populate demo' button for the others.")


if __name__ == "__main__":
    main()
