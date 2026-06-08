"""Headless end-to-end smoke path: ingest -> store -> imbalance number (Task 1.5).

Proves the whole chain works before the engine/UI exist:

  1. ensure stub flat files exist (generates them if missing)
  2. create the SQLite schema and a stub NO2 wind portfolio
  3. load DAM positions, generation forecast, and actual delivery from the files
  4. fetch a REAL Optimeering imbalance price forecast and store it
  5. compute the trivial EXPECTED imbalance (DAM - forecast generation) per MTU
  6. persist everything to data/iar.db and print a readable summary

This is intentionally *not* the Monte Carlo engine (that is Week 2). It only
confirms data flows API -> DB -> a number.

Run:  python scripts/run_pipeline.py
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

from iar.db.models import (
    ActualDelivery,
    DAMPosition,
    DAMPrice,
    GenerationForecast,
    ImbalancePriceForecast,
)
from iar.db.session import DEFAULT_DB_PATH, get_session, init_db
from iar.ingestion.flatfile_loader import (
    get_or_create_portfolio,
    load_actual_delivery,
    load_dam_positions,
    load_dam_prices,
    load_generation_forecasts,
)
from iar.ingestion.optimeering_client import OptimeeringForecastClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOADS = PROJECT_ROOT / "data" / "uploads"

AREA = "NO2"
USER = "Stub User"
PORTFOLIO = "NO2 Wind"
FILES = {
    "dam": UPLOADS / "dam_positions_NO2.csv",
    "gen": UPLOADS / "generation_forecast_NO2.csv",
    "act": UPLOADS / "actual_delivery_NO2.csv",
    "dam_price": UPLOADS / "dam_price_NO2.csv",
}


def _ensure_sample_files() -> None:
    if not all(p.exists() for p in FILES.values()):
        print("Sample files missing -> generating them...")
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "make_sample_data.py")],
            check=True,
        )


def _parse(ts: str | None) -> datetime | None:
    return datetime.fromisoformat(ts) if ts else None


def _store_forecast(session, records: list[dict]) -> int:
    """Replace stored forecast for the area, then insert the new records."""
    areas = {r["price_area"] for r in records}
    session.query(ImbalancePriceForecast).filter(
        ImbalancePriceForecast.price_area.in_(areas)
    ).delete(synchronize_session=False)
    session.add_all(
        ImbalancePriceForecast(
            price_area=r["price_area"],
            timestamp=_parse(r["timestamp"]),
            vintage_ts=_parse(r.get("vintage_ts")),
            statistic_type=r["statistic_type"],
            quantile=r.get("quantile"),
            unit_type=r.get("unit_type"),
            resolution=r.get("resolution"),
            value=r["value"],
        )
        for r in records
    )
    session.flush()
    return len(records)


def _hr(title: str) -> None:
    print("\n" + "=" * 64 + f"\n{title}\n" + "=" * 64)


def run() -> None:
    _hr("IaR MVP - end-to-end smoke path (Task 1.5)")
    _ensure_sample_files()

    init_db()
    print(f"database: {DEFAULT_DB_PATH}")

    with get_session() as s:
        # --- positions / forecast / actuals from flat files -------------- #
        pf = get_or_create_portfolio(s, USER, PORTFOLIO, AREA)
        pid = pf.portfolio_id
        n_dam = load_dam_positions(s, pid, FILES["dam"])
        n_gen = load_generation_forecasts(s, pid, FILES["gen"])
        n_act = load_actual_delivery(s, pid, FILES["act"])
        n_price = load_dam_prices(s, AREA, FILES["dam_price"])
        print(
            f"portfolio #{pid} '{pf.name}' ({pf.price_area}) | loaded [STUB] "
            f"DAM_pos(MWh)={n_dam}, forecast(MWh)={n_gen}, actual(MWh)={n_act}, "
            f"DAM_price(EUR/MWh)={n_price} rows"
        )

        # --- real Optimeering price forecast ----------------------------- #
        client = OptimeeringForecastClient()
        records = client.get_imbalance_price_forecast(AREA)
        n_fc = _store_forecast(s, records)
        quantiles = sorted({r["quantile"] for r in records if r["quantile"] is not None})
        unit_types = {r["unit_type"] for r in records}
        vintage = next((r["vintage_ts"] for r in records), None)
        print(
            f"Optimeering forecast: {n_fc} rows | quantiles={quantiles} | "
            f"{unit_types} | vintage={vintage}"
        )

        # --- trivial expected imbalance = DAM - forecast gen ------------- #
        dam = {r.timestamp: r.mwh for r in s.query(DAMPosition).filter_by(portfolio_id=pid)}
        gen = {
            r.timestamp: r.forecast_mwh
            for r in s.query(GenerationForecast).filter_by(portfolio_id=pid)
        }
        common = sorted(set(dam) & set(gen))
        imbalance = {t: dam[t] - gen[t] for t in common}
        total_imb = sum(imbalance.values())

        # --- illustrative euro figures over the price/position overlap --- #
        # P50 SPREAD (EUR vs spot) straight from Optimeering.
        p50_spread = {
            r.timestamp: r.value
            for r in s.query(ImbalancePriceForecast).filter_by(
                price_area=AREA, statistic_type="Quantile", quantile=50.0
            )
        }
        # DAM (spot) price, per area, from the flat file.
        dam_price = {
            r.timestamp: r.price
            for r in s.query(DAMPrice).filter_by(price_area=AREA)
        }
        # Absolute imbalance price = DAM + spread -> enables GROSS settlement.
        gross_price = {
            t: dam_price[t] + p50_spread[t]
            for t in set(dam_price) & set(p50_spread)
        }

        overlap = sorted(set(imbalance) & set(p50_spread))
        euro_spread = sum(imbalance[t] * p50_spread[t] for t in overlap)
        gross_overlap = sorted(set(imbalance) & set(gross_price))
        euro_gross = sum(imbalance[t] * gross_price[t] for t in gross_overlap)

        s.commit()

    # --- summary --------------------------------------------------------- #
    _hr("RESULT")
    print("Legend: DAM_pos = day-ahead POSITION (MWh, quantity sold), NOT a price.")
    print("        All portfolio inputs below are [STUB] synthetic data; only the")
    print("        Optimeering imbalance spread is live. Euro figures are illustrative.")
    print()
    print(f"MTUs with positions:            {len(imbalance)}")
    print(f"Total expected imbalance:       {total_imb:+.2f} MWh   (DAM_pos - forecast gen)")
    print("first 3 MTUs:")
    for t in common[:3]:
        print(
            f"  {t.isoformat()}  DAM_pos={dam[t]:6.2f} MWh  gen={gen[t]:6.2f} MWh  "
            f"imbalance={imbalance[t]:+6.2f} MWh"
        )
    if overlap:
        print(
            f"\n[STUB] Illustrative SPREAD settlement over {len(overlap)} overlapping MTUs: "
            f"{euro_spread:+,.0f} EUR"
        )
        print("  (= [STUB] imbalance x [LIVE] P50 spread; basis for Spread IaR.)")
    if gross_overlap:
        print(
            f"[STUB] Illustrative GROSS settlement over {len(gross_overlap)} overlapping MTUs: "
            f"{euro_gross:+,.0f} EUR"
        )
        print("  (= [STUB] imbalance x ([STUB] DAM price + [LIVE] P50 spread); basis for Gross IaR.)")
    if not overlap and not gross_overlap:
        print("\n(no timestamp overlap between positions and live forecast for a euro figure)")
    print("\nDone. Data persisted to the database. [OK]")


if __name__ == "__main__":
    run()
