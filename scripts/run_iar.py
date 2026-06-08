"""Run the Monte Carlo IaR engine end-to-end and print the result.

Wires together the Week-2 components with REAL market data:
  * live Optimeering spread quantiles      -> QuantilePriceSampler (2.2)
  * REAL DAM cleared (spot) price           -> internal MarketsApi (markets_client)
  * a STUB wind portfolio                   -> ImbalanceModel       (2.1)
  * independent Monte Carlo                 -> run_simulation       (2.3)

and prints Gross/Spread IaR + CIaR.

What's real vs stub now:
  * Imbalance-price SPREAD quantiles: LIVE (Optimeering public SDK).
  * DAM (spot) price: LIVE (Optimeering INTERNAL SDK, DAM cleared price) — requires the
    vendored optipyclient wheel (see docs/README.md). Falls back to a flat --dam-price
    with a warning if unavailable.
  * Portfolio (positions, generation): still a SYNTHETIC stub, so the absolute euro
    figures remain illustrative until real portfolio files are loaded.

The simulation runs over the MTUs where BOTH the live spread forecast AND a real DAM
price exist (their intersection) — so Gross IaR uses genuine spot prices throughout.

Usage:
    python scripts/run_iar.py
    python scripts/run_iar.py --area NO2 --scenarios 50000 --confidence 0.99
    python scripts/run_iar.py --dist student_t --sigma-fraction 0.15
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from iar.db.models import DAMPosition, GenerationForecast, Portfolio
from iar.db.session import DEFAULT_DB_PATH, get_session, init_db
from iar.ingestion.flatfile_loader import get_or_create_portfolio
from iar.ingestion.markets_client import OptimeeringMarketsClient
from iar.ingestion.optimeering_client import OptimeeringForecastClient
from iar.simulation.engine import EngineConfig, run_simulation
from iar.simulation.imbalance_model import ImbalanceModel, ImbalanceModelConfig
from iar.simulation.persistence import persist_report
from iar.simulation.price_sampler import QuantilePriceSampler

MTU_HOURS = 0.25  # 15-minute MTU


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the Monte Carlo IaR engine.")
    ap.add_argument("--area", default="NO2", help="price area (NO1/NO2/SE3)")
    ap.add_argument("--scenarios", type=int, default=10_000, help="number of MC scenarios")
    ap.add_argument("--confidence", type=float, default=0.95, help="confidence level (0-1)")
    ap.add_argument("--capacity-mw", type=float, default=100.0, help="[STUB] installed capacity")
    ap.add_argument("--sigma-fraction", type=float, default=0.10,
                    help="[STUB] imbalance sigma as a fraction of per-MTU capacity")
    ap.add_argument("--dam-price", type=float, default=45.0,
                    help="fallback flat spot price EUR/MWh (only if the real DAM fetch fails)")
    ap.add_argument("--dist", choices=["normal", "student_t"], default="normal",
                    help="imbalance distribution shape")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (reproducibility)")
    ap.add_argument("--store", action="store_true",
                    help="persist the run to data/iar.db (SimulationRun + IaRResult)")
    return ap.parse_args()


def fetch_spread_matrix(area: str):
    """Return (timestamps, percentile_levels, spread_matrix, vintage) from live forecast."""
    recs = OptimeeringForecastClient().get_imbalance_price_forecast(area)
    by_ts: dict[str, dict[float, float]] = defaultdict(dict)
    vintage = None
    for r in recs:
        if r["quantile"] is not None:
            by_ts[r["timestamp"]][r["quantile"]] = r["value"]
            vintage = vintage or r.get("vintage_ts")
    if not by_ts:
        raise SystemExit(f"No quantile forecast returned for area {area!r}.")
    pct = sorted({q for d in by_ts.values() for q in d})
    times = [t for t in sorted(by_ts) if all(q in by_ts[t] for q in pct)]
    spread = np.array([[by_ts[t][q] for q in pct] for t in times])
    return times, np.array(pct), spread, vintage


def fetch_dam_map(area: str):
    """Return {pd.Timestamp(UTC): price} of the real DAM cleared price, or {} on failure."""
    try:
        recs = OptimeeringMarketsClient().get_dam_prices(area, start="-P1D", end="P3D")
    except Exception as exc:  # noqa: BLE001 — optipyclient missing / auth / lookup
        print(f"[warn] real DAM price unavailable ({type(exc).__name__}: "
              f"{str(exc)[:120]}); falling back to flat --dam-price.")
        return {}
    return {pd.to_datetime(r["timestamp"], utc=True): float(r["eur_per_mwh"]) for r in recs}


def stub_portfolio(n_mtus: int, capacity_mw: float, seed: int):
    """Synthetic [STUB] wind portfolio aligned to the simulated horizon."""
    rng = np.random.default_rng(seed)
    cap_mwh = capacity_mw * MTU_HOURS
    factor = np.clip(rng.normal(0.45, 0.12, n_mtus), 0.05, 0.95)
    gen = factor * cap_mwh                                   # forecast generation
    dam_pos = np.clip(gen + rng.normal(0, 0.06 * cap_mwh, n_mtus), 0, cap_mwh)
    return dam_pos, gen, cap_mwh


def main() -> None:
    args = parse_args()

    times, pct, spread, vintage = fetch_spread_matrix(args.area)
    dam_map = fetch_dam_map(args.area)

    # Simulate over the intersection of (live spread MTUs) and (real DAM MTUs).
    ft = pd.to_datetime(times, utc=True)
    if dam_map:
        keep = [i for i, t in enumerate(ft) if t in dam_map]
        if keep:
            spread = spread[keep]
            dam_price = np.array([dam_map[ft[i]] for i in keep])
            dam_src = "LIVE (Optimeering DAM cleared price, internal SDK)"
        else:
            dam_price = np.full(len(times), args.dam_price)
            dam_src = f"flat {args.dam_price:.2f} [STUB — no overlap with live DAM]"
    else:
        dam_price = np.full(len(times), args.dam_price)
        dam_src = f"flat {args.dam_price:.2f} EUR/MWh [STUB fallback]"

    n_mtus = len(dam_price)
    price = QuantilePriceSampler.from_percentiles(pct, spread)
    dam_pos, gen, cap_mwh = stub_portfolio(n_mtus, args.capacity_mw, args.seed)
    imb = ImbalanceModel.from_inputs(
        dam_pos, gen, capacity_mwh=cap_mwh,
        config=ImbalanceModelConfig(
            dist=args.dist, sigma_fraction=args.sigma_fraction, scale_basis="capacity"
        ),
    )

    rep = run_simulation(
        price, imb, dam_price,
        EngineConfig(n_scenarios=args.scenarios, confidence=args.confidence, seed=args.seed),
    )

    bar = "=" * 64
    print(bar)
    print(f"IaR Monte Carlo  |  area={args.area}  MTUs={n_mtus}  "
          f"scenarios={rep.n_scenarios:,}  confidence={rep.confidence:.0%}")
    print(f"forecast vintage : {vintage}   (LIVE Optimeering spread)")
    print(f"DAM spot price   : {dam_src}")
    print(f"imbalance model  : {args.dist}, sigma={args.sigma_fraction:.0%} of capacity  [STUB portfolio]")
    print(bar)
    for name, m in (("GROSS ", rep.gross), ("SPREAD", rep.spread)):
        print(f"{name} | IaR = {m.iar:+12,.0f} EUR   "
              f"CIaR = {m.ciar:+12,.0f} EUR   mean = {m.mean:+12,.0f} EUR")
    print(bar)
    print("IaR = worst-case settlement cost at the confidence level (positive = cost).")
    print("CIaR = average cost in the worst (1 - confidence) tail.")
    print("NOTE: spread + DAM spot are LIVE; portfolio (positions/generation) is still a STUB.")

    if args.store:
        init_db()
        with get_session() as s:
            pf = get_or_create_portfolio(s, "MC Runner", f"{args.area} Wind", args.area)
            v = datetime.fromisoformat(vintage) if vintage else datetime.now(timezone.utc)
            run = persist_report(
                s, rep, pf.portfolio_id, vintage_ts=v, horizon=f"{n_mtus}xPT15M",
                extra_config={"area": args.area, "dist": args.dist,
                              "sigma_fraction": args.sigma_fraction,
                              "dam_source": dam_src},
            )
            s.commit()
            print(bar)
            print(f"[stored] SimulationRun #{run.run_id} (portfolio #{pf.portfolio_id}) "
                  f"+ 2 IaRResult rows -> {DEFAULT_DB_PATH}")


if __name__ == "__main__":
    main()
