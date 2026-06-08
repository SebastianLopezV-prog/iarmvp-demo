"""THROWAWAY demo UI — IaR MVP Risk Dashboard (gitignored, delete anytime).

Now runs on REAL inputs (matching scripts/run_iar.py):
  * live Optimeering spread quantiles      -> QuantilePriceSampler (2.2)
  * REAL DAM cleared (spot) price           -> internal MarketsApi (markets_client)
  * REAL portfolio positions/generation     -> loaded from iar.db (windsim via client CSVs)
  * independent Monte Carlo                 -> run_simulation (2.3)

Simulates over the MTUs where all available real sources overlap. Only the imbalance
`sigma` is still a parametric knob (Week-3 calibration). Reuses the real backend modules.

Run:  .\\venv\\Scripts\\python.exe -m streamlit run "beginner UI\\dashboard.py"
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from iar.db.models import DAMPosition, GenerationForecast, Portfolio
from iar.db.session import get_session, init_db
from iar.ingestion.markets_client import OptimeeringMarketsClient
from iar.ingestion.optimeering_client import OptimeeringForecastClient
from iar.simulation.engine import EngineConfig, run_simulation
from iar.simulation.imbalance_model import ImbalanceModel, ImbalanceModelConfig
from iar.simulation.price_sampler import QuantilePriceSampler

MTU_HOURS = 0.25
st.set_page_config(page_title="IaR MVP — Risk Dashboard", layout="wide")


@st.cache_data(show_spinner="Fetching live Optimeering spread…")
def fetch_spread(area: str):
    recs = OptimeeringForecastClient().get_imbalance_price_forecast(area)
    bt: dict[str, dict[float, float]] = defaultdict(dict)
    vintage = None
    for r in recs:
        if r["quantile"] is not None:
            bt[r["timestamp"]][r["quantile"]] = r["value"]
            vintage = vintage or r.get("vintage_ts")
    pct = sorted({q for d in bt.values() for q in d})
    times = [t for t in sorted(bt) if all(q in bt[t] for q in pct)]
    spread = np.array([[bt[t][q] for q in pct] for t in times])
    return times, pct, spread, vintage


@st.cache_data(show_spinner="Fetching real DAM (spot) price…")
def fetch_dam(area: str):
    """{ISO timestamp: price}. Empty on failure (e.g. optipyclient wheel missing)."""
    try:
        recs = OptimeeringMarketsClient().get_dam_prices(area, start="-P1D", end="P3D")
        return {r["timestamp"]: float(r["eur_per_mwh"]) for r in recs}, None
    except Exception as exc:  # noqa: BLE001
        return {}, f"{type(exc).__name__}: {exc}"


def fetch_positions(area: str):
    """{ISO(UTC) timestamp: (dam_pos, gen)} from the DB, + portfolio name. Empty if none."""
    init_db()
    with get_session() as s:
        pf = (s.query(Portfolio).filter_by(price_area=area)
              .order_by(Portfolio.portfolio_id.desc()).first())
        if pf is None:
            return {}, None
        dam = {r.timestamp: r.mwh for r in s.query(DAMPosition).filter_by(portfolio_id=pf.portfolio_id)}
        gen = {r.timestamp: r.forecast_mwh
               for r in s.query(GenerationForecast).filter_by(portfolio_id=pf.portfolio_id)}
        out = {pd.to_datetime(t, utc=True).isoformat(): (dam[t], gen[t])
               for t in (set(dam) & set(gen))}
        return out, pf.name


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("Controls")
area = st.sidebar.selectbox("Price area", ["NO2", "NO1", "SE3"], index=0)
scenarios = st.sidebar.select_slider("Scenarios", [2_000, 5_000, 10_000, 25_000, 50_000], value=10_000)
confidence = st.sidebar.slider("Confidence", 0.80, 0.99, 0.95, 0.01)
sigma = st.sidebar.slider("Imbalance sigma (% of capacity)", 0.02, 0.30, 0.10, 0.01,
                          help="The ONLY remaining stub — forecast-error size; calibrated in Week 3.")
capacity = st.sidebar.number_input("Capacity (MW) — sigma basis", 1.0, 1000.0, 19.0, 1.0)
dist = st.sidebar.selectbox("Imbalance distribution", ["normal", "student_t"], index=0)
seed = int(st.sidebar.number_input("Seed", 0, 9999, 42, 1))
dam_fallback = st.sidebar.number_input("Fallback flat DAM price (if real unavailable)", 0.0, 300.0, 45.0, 1.0)
if st.sidebar.button("↻ Refresh live data"):
    fetch_spread.clear()
    fetch_dam.clear()

# --------------------------------------------------------------------------- #
st.title("IaR MVP — Risk Dashboard")

try:
    times, pct, spread, vintage = fetch_spread(area)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not fetch the live forecast for {area}: {exc}")
    st.stop()

dam_map, dam_err = fetch_dam(area)
pos_map, pf_name = fetch_positions(area)

ft = [t for t in times]  # ISO strings; align by exact UTC ISO
keep = [i for i, t in enumerate(ft)
        if (not dam_map or t in dam_map) and (not pos_map or t in pos_map)]
if not keep:
    st.error("No overlapping MTUs across forecast / DAM price / positions.")
    st.stop()

spread = spread[keep]
n = len(keep)

if dam_map:
    dam_price = np.array([dam_map[ft[i]] for i in keep]); dam_real = True
else:
    dam_price = np.full(n, dam_fallback); dam_real = False

if pos_map:
    dam_pos = np.array([pos_map[ft[i]][0] for i in keep])
    gen = np.array([pos_map[ft[i]][1] for i in keep]); pos_real = True
else:
    rng = np.random.default_rng(seed)
    cap = capacity * MTU_HOURS
    gen = np.clip(rng.normal(0.45, 0.12, n), 0.05, 0.95) * cap
    dam_pos = np.clip(gen + rng.normal(0, 0.06 * cap, n), 0, cap); pos_real = False

price = QuantilePriceSampler.from_percentiles(np.array(pct), spread)
imb = ImbalanceModel.from_inputs(
    dam_pos, gen, capacity_mwh=capacity * MTU_HOURS,
    config=ImbalanceModelConfig(dist=dist, sigma_fraction=sigma, scale_basis="capacity"),
)
rep = run_simulation(price, imb, dam_price,
                     EngineConfig(n_scenarios=int(scenarios), confidence=confidence, seed=seed))

# --- data-source status -------------------------------------------------- #
def tag(real: bool) -> str:
    return "🟢 LIVE/REAL" if real else "🔴 STUB"

st.caption(
    f"area **{area}** · **{n}** MTUs (overlap) · {int(scenarios):,} scenarios · "
    f"{confidence:.0%} confidence · vintage `{vintage}`"
)
s1, s2, s3, s4 = st.columns(4)
s1.markdown(f"**Spread**  \n{tag(True)} (Optimeering)")
s2.markdown(f"**DAM spot**  \n{tag(dam_real)}" + ("" if dam_real else "  \n_wheel missing_"))
s3.markdown(f"**Portfolio**  \n{tag(pos_real)}" + (f"  \n_{pf_name}_" if pos_real else "  \n_synthetic_"))
s4.markdown(f"**Sigma**  \n🟡 stub ({sigma:.0%})")

# --- headline P&L metrics (negative = loss) ------------------------------- #
c1, c2, c3, c4 = st.columns(4)
c1.metric("Gross IaR", f"{-rep.gross.iar:,.0f} EUR", help="Worst-case P&L (negative = loss)")
c2.metric("Gross CIaR", f"{-rep.gross.ciar:,.0f} EUR", help="Average P&L in the worst tail")
c3.metric("Spread IaR", f"{-rep.spread.iar:,.0f} EUR")
c4.metric("Spread CIaR", f"{-rep.spread.ciar:,.0f} EUR")

# --- charts --------------------------------------------------------------- #
left, right = st.columns([3, 2])
with left:
    basis = st.radio("P&L basis", ["Gross", "Spread"], horizontal=True)
    m = rep.gross if basis == "Gross" else rep.spread
    pnl, pnl_mean, pnl_iar, pnl_ciar = -m.cost, -m.mean, -m.iar, -m.ciar
    fig = go.Figure()
    fig.add_histogram(x=pnl, nbinsx=60, marker_color="#4C78A8", name="scenarios")
    fig.add_vline(x=pnl_mean, line_color="green", line_dash="dot",
                  annotation_text=f"mean {pnl_mean:,.0f}", annotation_position="top right")
    fig.add_vline(x=pnl_iar, line_color="orange",
                  annotation_text=f"IaR {pnl_iar:,.0f}", annotation_position="top")
    fig.add_vline(x=pnl_ciar, line_color="red",
                  annotation_text=f"CIaR {pnl_ciar:,.0f}", annotation_position="top left")
    fig.update_layout(title=f"{basis} P&L distribution ({int(scenarios):,} scenarios)",
                      xaxis_title="EUR over horizon  (positive = gain, negative = loss)",
                      yaxis_title="scenarios", bargap=0.02, height=400, margin=dict(t=50, b=40))
    st.plotly_chart(fig, use_container_width=True)

with right:
    idx = {p: i for i, p in enumerate(pct)}
    x = pd.to_datetime([ft[i] for i in keep], utc=True)
    p05, p50, p95 = spread[:, idx[5.0]], spread[:, idx[50.0]], spread[:, idx[95.0]]
    fan = go.Figure()
    fan.add_scatter(x=x, y=p95, line=dict(width=0), showlegend=False, hoverinfo="skip")
    fan.add_scatter(x=x, y=p05, fill="tonexty", fillcolor="rgba(76,120,168,0.2)",
                    line=dict(width=0), name="P05–P95")
    fan.add_scatter(x=x, y=p50, line=dict(color="#4C78A8"), name="P50 (median)")
    fan.update_layout(title="Live Optimeering imbalance SPREAD forecast",
                      xaxis_title="time (UTC)", yaxis_title="EUR/MWh vs spot",
                      height=400, margin=dict(t=50, b=40), legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fan, use_container_width=True)

if dam_err:
    st.warning(f"Real DAM price unavailable → using flat fallback. ({dam_err})")
st.info(
    "P&L view: **negative = loss, positive = gain**. Spread (Optimeering), DAM spot "
    "(internal MarketsApi) and the portfolio (windsim, via the client CSV path) are all "
    "REAL when available; the imbalance **sigma** is the only remaining parametric stub "
    "(calibrated against realised actuals in Week 3)."
)
