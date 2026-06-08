"""End-to-end verification of everything built so far (1.1 -> 2.x unblocker).

A self-contained smoke check you can run from VS Code (Run & Debug -> "Verify all")
or the terminal. It exercises each layer in order and prints a pass/fail line for
each, then an overall summary. Designed to be:

  * **deterministic & isolated** — uses an in-memory SQLite DB and temp files, so
    it never touches data/iar.db or your uploads;
  * **offline-friendly** — the only network step (live Optimeering fetch) is
    best-effort and reported as [SKIP] if there's no key/connection, so the rest
    still verifies.

What it checks
--------------
  1.1  package + submodules import (incl. the 2.x simulation skeletons)
  1.2  DB schema: all tables create, incl. the new `dam_prices`
  1.4  flat-file loaders: positions/forecast/actual + DAM price; validation rejects bad input
  1.3  Optimeering client: live forecast fetch (best-effort)
  unblocker  Gross-IaR reconstruction: imbalance_price = DAM price + spread

Run:  python scripts/verify_all.py   (exit code 0 = all non-skipped checks passed)
"""

from __future__ import annotations

import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Tiny check harness
# --------------------------------------------------------------------------- #
class Check:
    PASS, FAIL, SKIP = "OK", "FAIL", "SKIP"


_results: list[tuple[str, str, str]] = []  # (label, status, detail)


def record(label: str, status: str, detail: str = "") -> None:
    tag = {Check.PASS: "[ OK ]", Check.FAIL: "[FAIL]", Check.SKIP: "[SKIP]"}[status]
    print(f"  {tag}  {label}" + (f"  -- {detail}" if detail else ""))
    _results.append((label, status, detail))


def section(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


# --------------------------------------------------------------------------- #
# 1.1  Imports
# --------------------------------------------------------------------------- #
def check_imports() -> None:
    section("1.1  Package + submodules import")
    modules = [
        "iar",
        "iar.db.models",
        "iar.db.session",
        "iar.ingestion.optimeering_client",
        "iar.ingestion.flatfile_loader",
        "iar.simulation.imbalance_model",
        "iar.simulation.price_sampler",
        "iar.simulation.engine",
    ]
    import importlib

    for mod in modules:
        try:
            importlib.import_module(mod)
            record(f"import {mod}", Check.PASS)
        except Exception as exc:  # noqa: BLE001
            record(f"import {mod}", Check.FAIL, str(exc))


# --------------------------------------------------------------------------- #
# 1.2  DB schema
# --------------------------------------------------------------------------- #
def check_schema() -> "object":
    """Build an in-memory DB; return a sessionmaker for later checks."""
    section("1.2  DB schema (in-memory)")
    from sqlalchemy import inspect
    from sqlalchemy.orm import sessionmaker

    from iar.db.models import Base
    from iar.db.session import init_db, make_engine

    engine = make_engine(":memory:")
    init_db(engine)
    tables = set(inspect(engine).get_table_names())
    expected = set(Base.metadata.tables.keys())
    missing = expected - tables
    if missing:
        record("create all tables", Check.FAIL, f"missing {sorted(missing)}")
    else:
        record(f"create all tables ({len(tables)} tables)", Check.PASS)
    record(
        "dam_prices table present",
        Check.PASS if "dam_prices" in tables else Check.FAIL,
    )
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


# --------------------------------------------------------------------------- #
# 1.4  Flat-file loaders
# --------------------------------------------------------------------------- #
def check_loaders(SessionLocal) -> None:
    section("1.4  Flat-file loaders (+ DAM price, + validation)")
    import pandas as pd

    from iar.db.models import (
        ActualDelivery,
        DAMPosition,
        DAMPrice,
        GenerationForecast,
    )
    from iar.ingestion.flatfile_loader import (
        FileValidationError,
        get_or_create_portfolio,
        load_actual_delivery,
        load_dam_positions,
        load_dam_prices,
        load_generation_forecasts,
    )

    ts = ["2026-06-08T00:00:00+00:00", "2026-06-08T00:15:00+00:00"]

    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        def csv(name, col, vals):
            p = tmp / name
            pd.DataFrame({"timestamp": ts, col: vals}).to_csv(p, index=False)
            return p

        with SessionLocal() as s:
            pf = get_or_create_portfolio(s, "VerifyUser", "NO2 Wind", "NO2")
            pid = pf.portfolio_id
            try:
                n1 = load_dam_positions(s, pid, csv("dam.csv", "mwh", [10.0, 12.5]))
                n2 = load_generation_forecasts(
                    s, pid, csv("gen.csv", "forecast_mwh", [9.0, 11.0])
                )
                n3 = load_actual_delivery(
                    s, pid, csv("act.csv", "actual_mwh", [8.5, 11.5])
                )
                n4 = load_dam_prices(s, "NO2", csv("dam_price.csv", "eur_per_mwh", [40.0, 42.0]))
                ok = (
                    n1 == n2 == n3 == n4 == 2
                    and s.query(DAMPosition).count() == 2
                    and s.query(GenerationForecast).count() == 2
                    and s.query(ActualDelivery).count() == 2
                    and s.query(DAMPrice).count() == 2
                )
                record("load 4 series (positions/forecast/actual/dam_price)",
                       Check.PASS if ok else Check.FAIL)
                # DAM price keyed by area, not portfolio
                area_ok = {r.price_area for r in s.query(DAMPrice)} == {"NO2"}
                record("dam_price keyed by area", Check.PASS if area_ok else Check.FAIL)
            except Exception as exc:  # noqa: BLE001
                record("load 4 series", Check.FAIL, str(exc))

            # validation must reject a bad file
            bad = tmp / "bad.csv"
            pd.DataFrame({"timestamp": ts[:1], "wrong": [1.0]}).to_csv(bad, index=False)
            try:
                load_dam_positions(s, pid, bad)
                record("validation rejects missing column", Check.FAIL, "no error raised")
            except FileValidationError:
                record("validation rejects missing column", Check.PASS)


# --------------------------------------------------------------------------- #
# 1.3  Optimeering live fetch (best-effort)
# --------------------------------------------------------------------------- #
def check_optimeering() -> None:
    section("1.3  Optimeering client (live, best-effort)")
    try:
        from iar.ingestion.optimeering_client import (
            MissingApiKeyError,
            OptimeeringForecastClient,
        )
    except Exception as exc:  # noqa: BLE001
        record("import client", Check.FAIL, str(exc))
        return

    try:
        client = OptimeeringForecastClient()
    except MissingApiKeyError:
        record("live forecast fetch", Check.SKIP, "no OPTIMEERING_API_KEY in .env")
        return
    except Exception as exc:  # noqa: BLE001
        record("construct client", Check.FAIL, str(exc))
        return

    try:
        records = client.get_imbalance_price_forecast("NO2")
        quantiles = sorted({r["quantile"] for r in records if r["quantile"] is not None})
        ok = len(records) > 0 and 50.0 in quantiles
        record(
            f"live forecast fetch ({len(records)} rows, quantiles={quantiles})",
            Check.PASS if ok else Check.FAIL,
        )
    except Exception as exc:  # noqa: BLE001
        record("live forecast fetch", Check.SKIP, f"network/API issue: {exc}")


# --------------------------------------------------------------------------- #
# unblocker  Gross-IaR reconstruction
# --------------------------------------------------------------------------- #
def check_gross_reconstruction(SessionLocal) -> None:
    section("2.x unblocker  Gross price = DAM + spread")
    from iar.db.models import DAMPrice, ImbalancePriceForecast

    t = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)
    with SessionLocal() as s:
        # store a DAM (spot) price and a P50 imbalance SPREAD for the same MTU
        s.add(DAMPrice(price_area="SE3", timestamp=t, price=50.0))
        s.add(
            ImbalancePriceForecast(
                price_area="SE3", timestamp=t, vintage_ts=t,
                statistic_type="Quantile", quantile=50.0,
                unit_type="Price_Spread", resolution="PT15M", value=12.0,
            )
        )
        s.flush()

        # NB: SQLite does not persist tz info, so DB timestamps come back naive.
        # Both dicts are keyed off the same DB round-trip, so their keys still align.
        dam = {r.timestamp: r.price for r in s.query(DAMPrice).filter_by(price_area="SE3")}
        spread = {
            r.timestamp: r.value
            for r in s.query(ImbalancePriceForecast).filter_by(
                price_area="SE3", statistic_type="Quantile", quantile=50.0
            )
        }
        common = set(dam) & set(spread)
        gross_price = {ts_: dam[ts_] + spread[ts_] for ts_ in common}
        ok = len(gross_price) == 1 and abs(next(iter(gross_price.values())) - 62.0) < 1e-9
        record(
            "imbalance_price = DAM(50) + spread(12) = 62",
            Check.PASS if ok else Check.FAIL,
            f"got {list(gross_price.values())}",
        )


# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 64)
    print("IaR MVP - verify everything (1.1 -> 2.x unblocker)")
    print("=" * 64)

    try:
        check_imports()
        SessionLocal = check_schema()
        check_loaders(SessionLocal)
        check_optimeering()
        check_gross_reconstruction(SessionLocal)
    except Exception:  # noqa: BLE001 — unexpected harness error
        print("\nUNEXPECTED ERROR:")
        traceback.print_exc()
        return 2

    # summary
    n_pass = sum(1 for _, st, _ in _results if st == Check.PASS)
    n_fail = sum(1 for _, st, _ in _results if st == Check.FAIL)
    n_skip = sum(1 for _, st, _ in _results if st == Check.SKIP)
    print("\n" + "=" * 64)
    print(f"SUMMARY: {n_pass} passed, {n_fail} failed, {n_skip} skipped")
    print("=" * 64)
    if n_fail:
        print("FAILED checks:")
        for label, st, detail in _results:
            if st == Check.FAIL:
                print(f"  - {label}: {detail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
