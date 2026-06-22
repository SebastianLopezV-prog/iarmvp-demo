"""Country-level IaR aggregation across bidding zones (1.2).

A country's Imbalance at Risk is **not** the sum of its zones' IaRs - that would ignore
diversification across zones. Instead we:

  1. read each zone's latest stored inputs from the database (forecast spread quantiles,
     DAM cleared price, and the loaded positions / generation) - no live feeds, so this is
     deterministic and source-agnostic (works on the real-feed live build and the synthetic
     demo alike);
  2. run the Monte Carlo per zone with an **independent** seed (so zones diversify);
  3. sum the per-scenario settlement cost **across zones** and take the quantile.

So country IaR = the confidence-quantile of the summed cost (diversified), consistent with
how the per-MTU period IaR already sums across MTUs before quantiling - never a naive sum of
per-zone IaRs. The same pass yields each zone's own IaR, so the per-zone breakdown and the
country total are internally consistent.

Zones are grouped into a country by their area prefix: ``SE1``..``SE4`` -> ``SE`` (Sweden),
``NO1``..``NO5`` -> ``NO`` (Norway).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from iar.db.models import DAMPosition, GenerationForecast, Portfolio
from iar.simulation.engine import EngineConfig, run_simulation
from iar.simulation.imbalance_model import ImbalanceModel, ImbalanceModelConfig
from iar.simulation.price_sampler import QuantilePriceSampler

MTU_HOURS = 0.25  # 15-minute MTU

#: Human-readable country names for the area prefixes.
COUNTRY_NAMES = {"SE": "Sweden", "NO": "Norway"}


def country_of(area: str) -> str:
    """Map a bidding zone to its country prefix, e.g. ``SE3`` -> ``SE``."""
    return area[:2].upper()


def country_name(country: str) -> str:
    """Friendly country label, e.g. ``SE`` -> ``Sweden`` (falls back to the code)."""
    return COUNTRY_NAMES.get(country.upper(), country.upper())


@dataclass(frozen=True)
class AggregateConfig:
    """Knobs for the country aggregation. Defaults mirror ``run_iar`` so the per-zone
    figures line up with the stored live runs."""

    n_scenarios: int = 10_000
    confidence: float = 0.95
    dist: str = "normal"
    sigma_fraction: float = 0.10
    capacity_mw: float = 100.0
    base_seed: int = 42


# --------------------------------------------------------------------------- #
# Inputs: forecast from the area's forecast client (real on live, synthetic on demo);
# positions and DAM price from the database (both products persist those).
# --------------------------------------------------------------------------- #
def _forecast_client():
    """Resolve the forecast client for the current product.

    The demo ships a synthetic factory (``iar.ingestion.clients``); the live build uses the
    real Optimeering client directly. This adapter lets the rest of the module stay identical
    on both products (mirror the feature, not the data-source layer).
    """
    try:
        from iar.ingestion.clients import get_forecast_client

        return get_forecast_client()
    except ImportError:
        from iar.ingestion.optimeering_client import OptimeeringForecastClient

        return OptimeeringForecastClient()


def _forecast_matrix(area: str):
    """Build ``(times, pct, spread)`` from the live/synthetic forecast for ``area``.

    Returns ``(list[pd.Timestamp UTC], np.ndarray pct, np.ndarray spread)`` or ``None`` if the
    client returns no usable quantile forecast.
    """
    try:
        recs = _forecast_client().get_imbalance_price_forecast(area)
    except Exception:
        return None
    by_ts: dict = defaultdict(dict)
    for r in recs or []:
        if r.get("quantile") is not None:
            by_ts[r["timestamp"]][float(r["quantile"])] = float(r["value"])
    if not by_ts:
        return None
    pct = sorted({q for d in by_ts.values() for q in d})
    times = [t for t in sorted(by_ts) if all(q in by_ts[t] for q in pct)]
    if not times:
        return None
    spread = np.array([[by_ts[t][q] for q in pct] for t in times])
    times_utc = [pd.to_datetime(t, utc=True) for t in times]
    return times_utc, np.array(pct), spread


def _markets_client():
    """Resolve the markets (DAM / realised price) client for the current product.

    Synthetic factory on the demo, real internal SDK client on live - same adapter pattern
    as :func:`_forecast_client`.
    """
    try:
        from iar.ingestion.clients import get_markets_client

        return get_markets_client()
    except ImportError:
        from iar.ingestion.markets_client import OptimeeringMarketsClient

        return OptimeeringMarketsClient()


def _dam_map(area: str) -> dict:
    """``{pd.Timestamp(UTC): price}`` of the forward DAM cleared price for ``area``.

    Fetched from the markets client over now-1d..now+3d (matching ``run_iar``) so it covers
    the forward forecast window. ``{}`` if the client is unavailable (then a flat fallback
    price is used, as in ``run_iar``).
    """
    try:
        recs = _markets_client().get_dam_prices(area, start="-P1D", end="P3D")
    except Exception:
        return {}
    return {pd.to_datetime(r["timestamp"], utc=True): float(r["eur_per_mwh"]) for r in recs}


def _latest_positions(session, area: str):
    """Return ``(pos_map, portfolio)`` for the most recent portfolio in ``area`` that has
    positions, where ``pos_map`` is ``{pd.Timestamp(UTC): (dam_pos_mwh, gen_mwh)}``.

    ``(None, None)`` if no portfolio in the area has loaded positions.
    """
    pfs = (
        session.query(Portfolio)
        .filter_by(price_area=area)
        .order_by(Portfolio.portfolio_id.desc())
        .all()
    )
    for pf in pfs:
        dam = {r.timestamp: r.mwh for r in session.query(DAMPosition).filter_by(portfolio_id=pf.portfolio_id)}
        gen = {
            r.timestamp: r.forecast_mwh
            for r in session.query(GenerationForecast).filter_by(portfolio_id=pf.portfolio_id)
        }
        out = {pd.to_datetime(t, utc=True): (dam[t], gen[t]) for t in (set(dam) & set(gen))}
        if out:
            return out, pf
    return None, None


@dataclass
class ZoneResult:
    """Per-zone outcome of a country pass."""

    area: str
    portfolio_id: int
    portfolio_name: str
    n_mtus: int
    gross_iar: float
    gross_ciar: float
    spread_iar: float
    spread_ciar: float
    gross_cost: np.ndarray  # per-scenario summed cost (length n_scenarios)
    spread_cost: np.ndarray


def compute_zone(session, area: str, cfg: AggregateConfig, *, seed: int) -> ZoneResult | None:
    """Run the MC for one zone from its latest stored inputs. ``None`` if inputs are missing.

    Mirrors ``scripts/run_iar.py`` assembly but reads everything from the database, so it is
    deterministic and never contacts a live feed.
    """
    fm = _forecast_matrix(area)
    pos_map, pf = _latest_positions(session, area)
    if fm is None or pos_map is None or pf is None:
        return None
    times, pct, spread = fm
    dam_map = _dam_map(area)

    keep = [i for i, t in enumerate(times) if t in pos_map and (not dam_map or t in dam_map)]
    if not keep:
        return None
    spread = spread[keep]
    n_mtus = len(keep)
    dam_pos = np.array([pos_map[times[i]][0] for i in keep])
    gen = np.array([pos_map[times[i]][1] for i in keep])
    dam_price = (
        np.array([dam_map[times[i]] for i in keep])
        if dam_map
        else np.full(n_mtus, 45.0)
    )

    price = QuantilePriceSampler.from_percentiles(pct, spread)
    imb = ImbalanceModel.from_inputs(
        dam_pos,
        gen,
        capacity_mwh=cfg.capacity_mw * MTU_HOURS,
        config=ImbalanceModelConfig(
            dist=cfg.dist, sigma_fraction=cfg.sigma_fraction, scale_basis="capacity"
        ),
    )
    rep = run_simulation(
        price,
        imb,
        dam_price,
        EngineConfig(n_scenarios=cfg.n_scenarios, confidence=cfg.confidence, seed=seed),
    )
    return ZoneResult(
        area=area,
        portfolio_id=pf.portfolio_id,
        portfolio_name=pf.name,
        n_mtus=n_mtus,
        gross_iar=rep.gross.iar,
        gross_ciar=rep.gross.ciar,
        spread_iar=rep.spread.iar,
        spread_ciar=rep.spread.ciar,
        gross_cost=rep.gross.cost,
        spread_cost=rep.spread.cost,
    )


def combine_costs(cost_vectors, confidence: float) -> tuple[float, float]:
    """Aggregate per-scenario cost vectors into ``(iar, ciar)``.

    Sums the vectors elementwise (truncating to the shortest, so mismatched scenario counts
    still align by index) and takes the confidence-quantile of the **summed** cost; CIaR is
    the mean of the tail beyond it. This is the diversified roll-up: because the quantile of a
    sum is at most the sum of the quantiles, the result never exceeds a naive sum of the
    per-vector IaRs.
    """
    vectors = [np.asarray(v) for v in cost_vectors]
    if not vectors:
        return 0.0, 0.0
    n = min(v.shape[0] for v in vectors)
    total = np.sum([v[:n] for v in vectors], axis=0)
    iar = float(np.quantile(total, confidence))
    tail = total[total >= iar]
    ciar = float(tail.mean()) if tail.size else iar
    return iar, ciar


def compute_country(session, areas, cfg: AggregateConfig | None = None) -> dict:
    """Aggregate the given zones into a country roll-up plus a per-zone breakdown.

    Each zone is drawn with an independent seed; the per-scenario cost is summed across the
    zones that share a common scenario count, then the quantile is taken (diversified country
    IaR). Returns a plain dict (UI/JSON friendly)::

        {
          "country": "SE", "country_name": "Sweden", "confidence": 0.95,
          "gross_iar": ..., "gross_ciar": ..., "spread_iar": ..., "spread_ciar": ...,
          "n_zones": 4, "zones": [ {area, portfolio_id, portfolio_name, n_mtus,
                                    gross_iar, gross_ciar, spread_iar, spread_ciar}, ... ],
          "diversification_gross": <sum-of-zone-IaRs / country-IaR>,  # >= 1
        }
    """
    cfg = cfg or AggregateConfig()
    zones: list[ZoneResult] = []
    for i, area in enumerate(sorted(set(areas))):
        zr = compute_zone(session, area, cfg, seed=cfg.base_seed + 1009 * (i + 1))
        if zr is not None:
            zones.append(zr)

    country = country_of(areas[0]) if areas else ""
    out = {
        "country": country,
        "country_name": country_name(country),
        "confidence": cfg.confidence,
        "n_zones": len(zones),
        "zones": [
            {
                "area": z.area,
                "portfolio_id": z.portfolio_id,
                "portfolio_name": z.portfolio_name,
                "n_mtus": z.n_mtus,
                "gross_iar": z.gross_iar,
                "gross_ciar": z.gross_ciar,
                "spread_iar": z.spread_iar,
                "spread_ciar": z.spread_ciar,
            }
            for z in zones
        ],
        "gross_iar": None,
        "gross_ciar": None,
        "spread_iar": None,
        "spread_ciar": None,
        "diversification_gross": None,
        "diversification_spread": None,
    }
    if not zones:
        return out

    # Sum per-scenario cost across zones, then quantile (diversified country IaR).
    out["gross_iar"], out["gross_ciar"] = combine_costs([z.gross_cost for z in zones], cfg.confidence)
    out["spread_iar"], out["spread_ciar"] = combine_costs(
        [z.spread_cost for z in zones], cfg.confidence
    )

    # Diversification ratio: sum of standalone zone IaRs vs the diversified country IaR.
    sum_g = sum(z.gross_iar for z in zones)
    sum_s = sum(z.spread_iar for z in zones)
    if out["gross_iar"]:
        out["diversification_gross"] = float(sum_g / out["gross_iar"])
    if out["spread_iar"]:
        out["diversification_spread"] = float(sum_s / out["spread_iar"])
    return out
