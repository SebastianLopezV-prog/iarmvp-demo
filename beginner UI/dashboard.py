"""THROWAWAY demo UI — IaR MVP Risk Dashboard (gitignored, delete anytime).

Two tabs:
  * **Live IaR** — a real Monte Carlo run (matching scripts/run_iar.py):
      live Optimeering spread (2.2) + real DAM spot (markets_client) + real
      windsim positions (from iar.db) -> run_simulation (2.3).
  * **Backtest (3.1 + 3.2)** — surfaces the new Week-3 backend:
      realised imbalance cost (3.1, iar.risk.realised_cost), backfilled day-ahead
      IaR estimates (3.2, iar.risk.replay), and the vintage comparison join
      (iar.risk.backtest.estimate_for_period). A one-click "populate demo data"
      button fills the realised-price + estimate history the panels read, because
      the real realised-price feed needs the vendored optipyclient wheel.

Run:  .\\venv\\Scripts\\python.exe -m streamlit run "beginner UI\\dashboard.py"
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from iar.db.models import (
    ActualDelivery,
    DAMPosition,
    GenerationForecast,
    Portfolio,
    SimulationRun,
)
from iar.db.session import get_session, init_db
from iar.ingestion.flatfile_loader import (
    store_actual_imbalance_price_records,
    store_dam_price_records,
)
from iar.ingestion.markets_client import OptimeeringMarketsClient
from iar.ingestion.optimeering_client import OptimeeringForecastClient
from iar.risk.alerts import classify_severity, load_limits
from iar.risk.backtest import estimate_for_period, iar_estimate_for_period, run_backtest
from iar.risk.calibration import calibrate_sigma
from iar.risk.realised_cost import compute_realised_cost
from iar.risk.replay import backfill_iar
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
# Backtest helpers (3.1 + 3.2) — read the real backend modules off the DB
# --------------------------------------------------------------------------- #
def portfolio_id_for(area: str):
    init_db()
    with get_session() as s:
        pf = (s.query(Portfolio).filter_by(price_area=area)
              .order_by(Portfolio.portfolio_id.desc()).first())
        return (pf.portfolio_id, pf.name) if pf else (None, None)


def load_estimates(pid: int) -> pd.DataFrame:
    """Backfilled day-ahead IaR estimates (3.2) for a portfolio, oldest first."""
    with get_session() as s:
        runs = (s.query(SimulationRun).filter_by(portfolio_id=pid)
                .order_by(SimulationRun.vintage_ts).all())
        rows = []
        for run in runs:
            res = {r.iar_type: r for r in run.results}
            cfg = json.loads(run.config_json or "{}")
            day = res["gross"].horizon if "gross" in res else ""
            rows.append({
                "delivery_day": day,
                "vintage_ts": pd.to_datetime(run.vintage_ts, utc=True),
                "gross_IaR": res["gross"].iar_value if "gross" in res else None,
                "spread_IaR": res["spread"].iar_value if "spread" in res else None,
                "gross_CIaR": res["gross"].ciar_value if "gross" in res else None,
                "n_mtus": cfg.get("n_mtus"),
                "source": "replay" if cfg.get("replay") else "live --store",
            })
        return pd.DataFrame(rows)


def realised_frame(pid: int) -> pd.DataFrame:
    with get_session() as s:
        return compute_realised_cost(s, pid)


def populate_demo_backtest(area, pid, pct, curve, dam_map_iso, fallback, cfg):
    """Fill the DB with the inputs the backtest panels need (clearly DEMO).

    REAL: positions/generation/actual delivery (windsim) + DAM spot (where the
    wheel served it). DEMO: the realised imbalance *price* (synthesised as
    DAM + a spread sampled from the live quantile curve) and per-day forecast
    vintages (the live curve replayed as each day's day-ahead estimate). Lets
    3.1/3.2 run end-to-end without the vendored realised-price feed.
    """
    with get_session() as s:
        dampos = {pd.to_datetime(r.timestamp, utc=True): r.mwh
                  for r in s.query(DAMPosition).filter_by(portfolio_id=pid)}
        gen = {pd.to_datetime(r.timestamp, utc=True): r.forecast_mwh
               for r in s.query(GenerationForecast).filter_by(portfolio_id=pid)}
        act = {pd.to_datetime(r.timestamp, utc=True): r.actual_mwh
               for r in s.query(ActualDelivery).filter_by(portfolio_id=pid)}
    mtus = sorted(set(dampos) & set(gen) & set(act))
    if not mtus:
        return {"mtus": 0, "runs": 0}

    dam = {t: float(dam_map_iso.get(t.isoformat(), fallback)) for t in mtus}

    # DEMO realised imbalance price = DAM + a spread drawn from the live curve.
    sampler = QuantilePriceSampler.from_percentiles(
        np.array(pct, dtype=float), np.tile(np.asarray(curve, dtype=float), (len(mtus), 1))
    )
    rng = np.random.default_rng(cfg["seed"])
    realised_spread = sampler.ppf(rng.random((1, len(mtus))))[0]
    realised_price = np.array([dam[t] for t in mtus]) + realised_spread

    # DEMO forecast vintages: the live curve as each day's day-ahead forecast.
    pct_list = list(pct)
    forecast_records = []
    for t in mtus:
        vintage = (t.normalize() - pd.Timedelta(hours=12)).isoformat()
        for q in pct_list:
            forecast_records.append({
                "vintage_ts": vintage, "timestamp": t.isoformat(),
                "quantile": q, "value": float(curve[pct_list.index(q)]),
            })

    with get_session() as s:
        store_dam_price_records(
            s, area, [{"timestamp": t.isoformat(), "eur_per_mwh": dam[t]} for t in mtus]
        )
        store_actual_imbalance_price_records(
            s, area,
            [{"timestamp": t.isoformat(), "eur_per_mwh": float(p)}
             for t, p in zip(mtus, realised_price)],
        )
        runs = backfill_iar(
            s, pid,
            forecast_records=forecast_records,
            dam_price_map={t: dam[t] for t in mtus},
            position_map={t: (dampos[t], gen[t]) for t in mtus},
            capacity_mwh=cfg["capacity"] * MTU_HOURS,
            model_config=ImbalanceModelConfig(
                dist=cfg["dist"], sigma_fraction=cfg["sigma"], scale_basis="capacity"
            ),
            engine_config=EngineConfig(
                n_scenarios=int(cfg["scenarios"]), confidence=cfg["confidence"], seed=cfg["seed"]
            ),
        )
        s.commit()
        return {"mtus": len(mtus), "runs": len(runs)}


def calibrate_demo(pid, pct, curve, dam_map_iso, fallback, cfg, basis):
    """Sweep sigma against the backtest and recommend the best-calibrated value.

    Reuses the same DEMO forecast curve as the populator; realised cost is read
    from whatever is already in the DB (populate it first). Returns a
    ``CalibrationResult`` (or ``None`` if there are no positions to estimate).
    """
    with get_session() as s:
        dampos = {pd.to_datetime(r.timestamp, utc=True): r.mwh
                  for r in s.query(DAMPosition).filter_by(portfolio_id=pid)}
        gen = {pd.to_datetime(r.timestamp, utc=True): r.forecast_mwh
               for r in s.query(GenerationForecast).filter_by(portfolio_id=pid)}
    mtus = sorted(set(dampos) & set(gen))
    if not mtus:
        return None

    dam = {t: float(dam_map_iso.get(t.isoformat(), fallback)) for t in mtus}
    pct_list = list(pct)
    forecast_records = [
        {"vintage_ts": (t.normalize() - pd.Timedelta(hours=12)).isoformat(),
         "timestamp": t.isoformat(), "quantile": q, "value": float(curve[pct_list.index(q)])}
        for t in mtus for q in pct_list
    ]
    with get_session() as s:
        return calibrate_sigma(
            s, pid,
            forecast_records=forecast_records,
            dam_price_map={t: dam[t] for t in mtus},
            position_map={t: (dampos[t], gen[t]) for t in mtus},
            capacity_mwh=cfg["capacity"] * MTU_HOURS,
            engine_config=EngineConfig(
                n_scenarios=int(cfg["scenarios"]), confidence=cfg["confidence"], seed=cfg["seed"]
            ),
            iar_type=basis,
        )


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
# Shared data + live run
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

curve = np.median(spread, axis=0)  # representative spread curve for the demo backtest


def tag(real: bool) -> str:
    return "🟢 LIVE/REAL" if real else "🔴 STUB"


tab_live, tab_bt = st.tabs(["📉 Live IaR", "🔁 Backtest (3.1–3.3)"])

# =========================================================================== #
# TAB 1 — Live IaR (existing functionality)
# =========================================================================== #
with tab_live:
    st.caption(
        f"area **{area}** · **{n}** MTUs (overlap) · {int(scenarios):,} scenarios · "
        f"{confidence:.0%} confidence · vintage `{vintage}`"
    )
    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(f"**Spread**  \n{tag(True)} (Optimeering)")
    s2.markdown(f"**DAM spot**  \n{tag(dam_real)}" + ("" if dam_real else "  \n_wheel missing_"))
    s3.markdown(f"**Portfolio**  \n{tag(pos_real)}" + (f"  \n_{pf_name}_" if pos_real else "  \n_synthetic_"))
    s4.markdown(f"**Sigma**  \n🟡 stub ({sigma:.0%})")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gross IaR", f"{-rep.gross.iar:,.0f} EUR", help="Worst-case P&L (negative = loss)")
    c2.metric("Gross CIaR", f"{-rep.gross.ciar:,.0f} EUR", help="Average P&L in the worst tail")
    c3.metric("Spread IaR", f"{-rep.spread.iar:,.0f} EUR")
    c4.metric("Spread CIaR", f"{-rep.spread.ciar:,.0f} EUR")

    # --- limit status (3.4): live IaR vs configured remaining-day euro-limits --- #
    _badge = {"hard": "🔴 HARD breach", "soft": "🟠 soft warning", None: "🟢 within limit"}
    st.markdown("**Limit status** — remaining-day euro-limits (3.4)")
    try:
        limits = load_limits()
        lname = pf_name or "default"
        lc = st.columns(2)
        for col, (label, meas) in zip(lc, [("Gross", rep.gross), ("Spread", rep.spread)]):
            lim = limits.limit_for(lname, label.lower(), "remaining_day")
            if lim is None:
                col.markdown(f"**{label}** · _no limit configured_")
            else:
                sev = classify_severity(meas.iar, lim)
                col.markdown(
                    f"**{label}** · {_badge[sev]}  \n"
                    f"IaR {meas.iar:,.0f} / limit {lim:,.0f} EUR ({meas.iar / lim:.0%} used)"
                )
        st.caption("IaR is in cost terms (positive = cost); a breach is IaR > limit, "
                   "a soft warning is IaR > 80% of the limit. Limits from `config/limits.toml`.")
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Limit config unavailable: {exc}")

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

# =========================================================================== #
# TAB 2 — Backtest (3.1 realised cost + 3.2 vintage replay & join)
# =========================================================================== #
with tab_bt:
    st.subheader("Week-3 backtest building blocks")
    st.caption(
        "**3.1** realised imbalance cost (`iar.risk.realised_cost`) · "
        "**3.2** backfilled day-ahead estimates + vintage join "
        "(`iar.risk.replay`, `iar.risk.backtest`). Calibration (Kupiec/exceedance %) is **3.3**."
    )

    pid, bt_pf_name = portfolio_id_for(area)
    if pid is None:
        st.warning(
            f"No portfolio loaded for **{area}**. Run "
            "`scripts/load_windsim_data.py` first to ingest positions/actuals."
        )
        st.stop()

    estimates = load_estimates(pid)
    realised = realised_frame(pid)
    has_data = not estimates.empty or not realised.empty

    cfg = {"scenarios": scenarios, "confidence": confidence, "sigma": sigma,
           "capacity": capacity, "dist": dist, "seed": seed}

    top = st.columns([3, 2])
    with top[0]:
        st.markdown(f"Portfolio **{bt_pf_name}** (#{pid}) · area **{area}**")
        st.caption(
            "Realised price + forecast vintages here are **DEMO** (the live realised-price "
            "feed needs the vendored optipyclient wheel). Positions, generation, actual "
            "delivery and DAM spot are REAL where available."
        )
    with top[1]:
        if st.button("⚙️ Populate / refresh demo backtest data", type="primary"):
            with st.spinner("Storing realised prices + backfilling day-ahead estimates…"):
                out = populate_demo_backtest(area, pid, pct, curve, dam_map, dam_fallback, cfg)
            if out["mtus"] == 0:
                st.error("No MTUs with positions + generation + actual delivery in the DB.")
            else:
                st.success(f"Stored {out['mtus']} realised prices · backfilled {out['runs']} day(s).")
                st.rerun()

    if not has_data:
        st.info("No backtest data yet — click **Populate / refresh demo backtest data** above.")
        st.stop()

    # ---- 3.2 backfilled estimates ---------------------------------------- #
    st.markdown("### 3.2 — Backfilled day-ahead IaR estimates")
    if estimates.empty:
        st.info("No stored estimates yet.")
    else:
        st.dataframe(
            estimates.style.format({
                "gross_IaR": "{:,.0f}", "spread_IaR": "{:,.0f}", "gross_CIaR": "{:,.0f}",
            }),
            use_container_width=True, hide_index=True,
        )
        st.caption("Each row is the estimate stamped with the vintage that *preceded* its "
                   "delivery day (no look-ahead) — the history the backtest joins against.")

    # ---- 3.1 realised cost ----------------------------------------------- #
    st.markdown("### 3.1 — Realised imbalance cost")
    if realised.empty:
        st.info("No realised cost yet (needs actual delivery + actual imbalance price + DAM price).")
    else:
        tot_gross = float(realised["gross_cost"].sum())
        tot_spread = float(realised["spread_cost"].sum())
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Realised Gross cost", f"{tot_gross:,.0f} EUR",
                   help="Σ imbalance × imbalance price (positive = cost)")
        rc2.metric("Realised Spread cost", f"{tot_spread:,.0f} EUR")
        rc3.metric("Settled MTUs", f"{len(realised):,}")

        rfig = go.Figure()
        rfig.add_bar(x=realised["timestamp"], y=realised["gross_cost"], name="gross cost",
                     marker_color="#E45756")
        rfig.add_bar(x=realised["timestamp"], y=realised["spread_cost"], name="spread cost",
                     marker_color="#F58518")
        rfig.update_layout(title="Realised settlement cost per MTU (positive = cost)",
                           barmode="overlay", xaxis_title="time (UTC)", yaxis_title="EUR",
                           height=320, margin=dict(t=50, b=40),
                           legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(rfig, use_container_width=True)

    # ---- 3.3 calibration: exceedances + Kupiec POF ----------------------- #
    st.markdown("### 3.3 — Calibration: exceedances + Kupiec POF")
    bt_basis = st.radio("IaR basis", ["gross", "spread"], horizontal=True, key="bt_basis")
    with get_session() as s:
        res = run_backtest(s, pid, bt_basis, persist=True)
        s.commit()

    if res.n_periods == 0:
        st.info("No settled periods yet — populate demo data (settled days need realised cost).")
    else:
        k = res.kupiec
        verdict = ("🟢 calibrated" if k.well_calibrated else "🔴 mis-calibrated") \
            if k.well_calibrated is not None else "—"
        kc = st.columns(4)
        kc[0].metric("Exceedances", f"{res.n_exceedances} / {res.n_periods}")
        kc[1].metric("Observed rate", f"{k.observed_rate:.0%}",
                     help=f"Expected ≈ {k.expected_rate:.0%} at this confidence")
        kc[2].metric("Kupiec LR", f"{k.lr_statistic:.2f}" if k.lr_statistic is not None else "—",
                     help="χ²(1) likelihood-ratio statistic")
        kc[3].metric("Kupiec verdict", verdict,
                     help=f"p-value {k.p_value:.3f}" if k.p_value is not None else "")

        df = res.as_frame()
        df["exceeded"] = df["exceeded"].map(lambda v: "🔴 exceeded" if v else "🟢 within")
        st.dataframe(
            df.rename(columns={"iar_estimate": f"{bt_basis}_IaR_estimate",
                               "realised_cost": f"realised_{bt_basis}_cost"})
              .style.format({f"{bt_basis}_IaR_estimate": "{:,.0f}",
                             f"realised_{bt_basis}_cost": "{:,.0f}"}),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            f"Each settled day's realised {bt_basis} cost vs the day-ahead IaR estimate that "
            "preceded it (3.2 join). An **exceedance** is realised cost worse than the estimate. "
            "Rows are persisted as `HistoricalPerformanceRecord`. With only a few settled days the "
            "Kupiec test has low power — it's a calibration *readout*, not a verdict, until more "
            "history accrues."
        )

    # ---- the join function itself ---------------------------------------- #
    with st.expander("🔎 Vintage lookup — `backtest.estimate_for_period`"):
        st.caption("Pick a moment; see which stored estimate the backtest would use for it "
                   "(the latest vintage at or before that instant).")
        d = st.date_input("As-of date (period start, UTC)")
        if d is not None:
            ps = pd.Timestamp(d, tz="UTC")
            with get_session() as s:
                run = estimate_for_period(s, pid, ps)
                if run is None:
                    st.write("No estimate precedes that instant.")
                else:
                    g = iar_estimate_for_period(s, pid, ps, "gross")
                    sp = iar_estimate_for_period(s, pid, ps, "spread")
                    st.write(
                        f"→ estimate for delivery day **{run.results[0].horizon}**, "
                        f"vintage `{pd.to_datetime(run.vintage_ts, utc=True).isoformat()}` · "
                        f"Gross IaR **{g.iar_value:,.0f}** · Spread IaR **{sp.iar_value:,.0f}** EUR"
                    )
