"""Client-facing DEMO dashboard — safe to host as a public link.

Runs the REAL IaR engine, backtest, Kupiec test, limits and sigma calibration, but
on fully SYNTHETIC data generated in-process. It needs NO Optimeering API key, NO
internal `optipyclient` wheel, and NO `windsim` — so nothing confidential is exposed
and it can be deployed to Streamlit Community Cloud from a public repo.

Deploy (see docs/DEPLOY.md):
  - main file: app/demo_app.py
  - requirements: requirements.txt (public deps only)
  - set a password in the host's secrets:  password = "..."

Everything shown is ILLUSTRATIVE (synthetic inputs + the MVP independence
assumption), not a real risk figure. The banner says so.
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm
from sqlalchemy.orm import sessionmaker

from iar.db.models import (
    ActualDelivery,
    ActualImbalancePrice,
    DAMPosition,
    DAMPrice,
    GenerationForecast,
)
from iar.db.session import init_db, make_engine
from iar.ingestion.flatfile_loader import get_or_create_portfolio  # imports models only (no SDK)
from iar.risk.alerts import classify_severity, load_limits
from iar.risk.backtest import run_backtest
from iar.risk.calibration import calibrate_sigma
from iar.risk.replay import backfill_iar
from iar.simulation.engine import EngineConfig, run_simulation
from iar.simulation.imbalance_model import ImbalanceModel, ImbalanceModelConfig
from iar.simulation.price_sampler import QuantilePriceSampler

MTU_HOURS = 0.25
PCT = [5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0]
SEED = 7
N_MTUS_DAY = 96            # live view: one full day
BT_DAYS = 6                # backtest: settled days
BT_MTUS_PER_DAY = 12       # keep the backfill quick

st.set_page_config(page_title="Imbalance at Risk - Demo", layout="wide")


# --------------------------------------------------------------------------- #
# Password gate (open if no password configured, e.g. local dev)
# --------------------------------------------------------------------------- #
def _password_ok() -> bool:
    try:
        configured = st.secrets.get("password")
    except Exception:
        configured = None
    configured = configured or os.environ.get("DEMO_PASSWORD")
    if not configured:
        return True
    if st.session_state.get("auth_ok"):
        return True
    pw = st.text_input("Access password", type="password")
    if pw and pw == configured:
        st.session_state["auth_ok"] = True
        return True
    if pw:
        st.error("Incorrect password.")
    return False


if not _password_ok():
    st.stop()


# --------------------------------------------------------------------------- #
# Synthetic data (deterministic) — the REAL engine runs on these
# --------------------------------------------------------------------------- #
def _spread_matrix(n_mtus: int, day_offset: int = 0) -> np.ndarray:
    """Per-MTU imbalance-spread quantile curve (n_mtus x len(PCT)), monotonic."""
    hours = np.linspace(0, 24, n_mtus, endpoint=False)
    median = 2.0 + 4.0 * np.sin((hours - 7) / 24 * 2 * np.pi) + day_offset
    width = 16.0 + 7.0 * np.cos((hours - 18) / 24 * 2 * np.pi)
    z = norm.ppf(np.array(PCT) / 100.0)
    spread = median[:, None] + width[:, None] * z[None, :]
    spread[:, -1] += 10.0  # a mild heavy upper tail
    return np.maximum.accumulate(spread, axis=1)


def _dam_price(n_mtus: int) -> np.ndarray:
    hours = np.linspace(0, 24, n_mtus, endpoint=False)
    return 40.0 + 12.0 * np.sin((hours - 8) / 24 * 2 * np.pi)


def _positions(n_mtus: int, rng: np.random.Generator, cap_mwh: float):
    cf = np.clip(rng.normal(0.45, 0.12, n_mtus), 0.05, 0.95)
    gen = cf * cap_mwh
    dam_pos = np.clip(gen + rng.normal(0, 0.06 * cap_mwh, n_mtus), 0.0, cap_mwh)
    return dam_pos, gen


@st.cache_data(show_spinner=False)
def live_inputs(capacity_mw: float):
    """Synthetic inputs for the live view (cached so they're stable across reruns)."""
    rng = np.random.default_rng(SEED)
    cap = capacity_mw * MTU_HOURS
    spread = _spread_matrix(N_MTUS_DAY)
    dam = _dam_price(N_MTUS_DAY)
    dam_pos, gen = _positions(N_MTUS_DAY, rng, cap)
    t0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
    times = [t0 + timedelta(minutes=15 * i) for i in range(N_MTUS_DAY)]
    return np.array(PCT), spread, dam, dam_pos, gen, times


@st.cache_resource(show_spinner="Building synthetic backtest history...")
def seeded_backtest_db(capacity_mw: float):
    """A temp-file SQLite seeded with BT_DAYS of synthetic settled history + estimates."""
    cap = capacity_mw * MTU_HOURS
    db_path = Path(tempfile.gettempdir()) / "iar_demo_backtest.db"
    if db_path.exists():
        db_path.unlink()
    engine = make_engine(str(db_path))
    init_db(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False, future=True)

    rng = np.random.default_rng(SEED + 1)
    forecast_records, dam_map, pos_map = [], {}, {}
    start_day = datetime(2026, 5, 20, tzinfo=timezone.utc)

    with Session() as s:
        pf = get_or_create_portfolio(s, "Demo Co", "NO2 Wind", "NO2")
        pid = pf.portfolio_id
        for d in range(BT_DAYS):
            day = start_day + timedelta(days=d)
            spread = _spread_matrix(BT_MTUS_PER_DAY, day_offset=float(rng.normal(0, 1.5)))
            dam = _dam_price(BT_MTUS_PER_DAY)
            dam_pos, gen = _positions(BT_MTUS_PER_DAY, rng, cap)
            actual = np.clip(gen + rng.normal(0, 0.12 * cap, BT_MTUS_PER_DAY), 0, None)
            # realised absolute imbalance price = DAM + a realised spread draw
            realised_spread = spread[:, 3] + rng.normal(0, 8, BT_MTUS_PER_DAY)
            vintage = (day - timedelta(hours=12)).isoformat()
            for i in range(BT_MTUS_PER_DAY):
                ts = day + timedelta(minutes=15 * i)
                s.add(DAMPosition(portfolio_id=pid, timestamp=ts, mwh=float(dam_pos[i])))
                s.add(GenerationForecast(portfolio_id=pid, timestamp=ts, forecast_mwh=float(gen[i])))
                s.add(ActualDelivery(portfolio_id=pid, timestamp=ts, actual_mwh=float(actual[i])))
                s.add(DAMPrice(price_area="NO2", timestamp=ts, price=float(dam[i])))
                s.add(ActualImbalancePrice(price_area="NO2", timestamp=ts,
                                           price=float(dam[i] + realised_spread[i])))
                dam_map[ts] = float(dam[i])
                pos_map[ts] = (float(dam_pos[i]), float(gen[i]))
                for j, q in enumerate(PCT):
                    forecast_records.append({"vintage_ts": vintage, "timestamp": ts.isoformat(),
                                             "quantile": q, "value": float(spread[i, j])})
        s.flush()
        backfill_iar(
            s, pid, forecast_records=forecast_records, dam_price_map=dam_map, position_map=pos_map,
            capacity_mwh=cap,
            engine_config=EngineConfig(n_scenarios=3000, confidence=0.95, seed=SEED),
        )
        s.commit()
    return {"engine": engine, "pid": pid, "forecast_records": forecast_records,
            "dam_map": dam_map, "pos_map": pos_map, "cap": cap}


# --------------------------------------------------------------------------- #
# Sidebar + banner
# --------------------------------------------------------------------------- #
st.sidebar.title("Controls")
confidence = st.sidebar.slider("Confidence", 0.80, 0.99, 0.95, 0.01)
scenarios = st.sidebar.select_slider("Scenarios", [2_000, 5_000, 10_000, 25_000], value=10_000)
capacity = st.sidebar.number_input("Capacity (MW)", 1.0, 500.0, 100.0, 1.0)
sigma = st.sidebar.slider("Imbalance sigma (% of capacity)", 0.02, 0.40, 0.10, 0.01)
dist = st.sidebar.selectbox("Imbalance distribution", ["normal", "student_t"], index=0)

st.title("Imbalance at Risk (IaR) - Demo")
st.warning(
    "Illustrative demo on SYNTHETIC data. Numbers are not real risk figures. The MVP "
    "samples price and position independently, which biases IaR optimistically low."
)

tab_live, tab_bt = st.tabs(["Live IaR", "Backtest"])

# --------------------------------------------------------------------------- #
# Live IaR
# --------------------------------------------------------------------------- #
with tab_live:
    pct, spread, dam_price, dam_pos, gen, times = live_inputs(capacity)
    price = QuantilePriceSampler.from_percentiles(pct, spread)
    imb = ImbalanceModel.from_inputs(
        dam_pos, gen, capacity_mwh=capacity * MTU_HOURS,
        config=ImbalanceModelConfig(dist=dist, sigma_fraction=sigma, scale_basis="capacity"),
    )
    rep = run_simulation(price, imb, dam_price,
                         EngineConfig(n_scenarios=int(scenarios), confidence=confidence, seed=SEED))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gross IaR", f"{-rep.gross.iar:,.0f} EUR", help="P&L; negative = loss")
    c2.metric("Gross CIaR", f"{-rep.gross.ciar:,.0f} EUR")
    c3.metric("Spread IaR", f"{-rep.spread.iar:,.0f} EUR")
    c4.metric("Spread CIaR", f"{-rep.spread.ciar:,.0f} EUR")

    # limit status vs config/limits.toml (defaults; no override for this portfolio)
    st.markdown("**Limit status** (remaining-day euro-limits)")
    try:
        limits = load_limits()
        badge = {"hard": "HARD breach", "soft": "soft warning", None: "within limit"}
        lc = st.columns(2)
        for col, (label, meas) in zip(lc, [("Gross", rep.gross), ("Spread", rep.spread)]):
            lim = limits.limit_for("Demo NO2", label.lower(), "remaining_day")
            if lim:
                sev = classify_severity(meas.iar, lim)
                col.write(f"{label}: {badge[sev]} - IaR {meas.iar:,.0f} / limit {lim:,.0f} EUR "
                          f"({meas.iar / lim:.0%} used)")
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Limits unavailable: {exc}")

    left, right = st.columns([3, 2])
    with left:
        basis = st.radio("P&L basis", ["Gross", "Spread"], horizontal=True)
        m = rep.gross if basis == "Gross" else rep.spread
        pnl = -m.cost
        fig = go.Figure()
        fig.add_histogram(x=pnl, nbinsx=60, marker_color="#4C78A8")
        fig.add_vline(x=-m.iar, line_color="orange", annotation_text=f"IaR {-m.iar:,.0f}")
        fig.add_vline(x=-m.ciar, line_color="red", annotation_text=f"CIaR {-m.ciar:,.0f}")
        fig.update_layout(title=f"{basis} P&L distribution", xaxis_title="EUR (negative = loss)",
                          yaxis_title="scenarios", height=380, margin=dict(t=50, b=40))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        idx = {p: i for i, p in enumerate(pct)}
        x = pd.to_datetime(times, utc=True)
        fan = go.Figure()
        fan.add_scatter(x=x, y=spread[:, idx[95.0]], line=dict(width=0), showlegend=False, hoverinfo="skip")
        fan.add_scatter(x=x, y=spread[:, idx[5.0]], fill="tonexty",
                        fillcolor="rgba(76,120,168,0.2)", line=dict(width=0), name="P05-P95")
        fan.add_scatter(x=x, y=spread[:, idx[50.0]], line=dict(color="#4C78A8"), name="P50")
        fan.update_layout(title="Imbalance spread forecast (synthetic)", xaxis_title="time (UTC)",
                          yaxis_title="EUR/MWh vs spot", height=380, margin=dict(t=50, b=40),
                          legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fan, use_container_width=True)

# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
with tab_bt:
    st.caption("Backfilled day-ahead estimates vs realised cost, with the Kupiec calibration test.")
    db = seeded_backtest_db(capacity)
    Session = sessionmaker(bind=db["engine"], expire_on_commit=False, future=True)
    bt_basis = st.radio("IaR basis", ["gross", "spread"], horizontal=True, key="bt_basis")

    with Session() as s:
        res = run_backtest(s, db["pid"], bt_basis, persist=False)

    if res.n_periods == 0:
        st.info("No settled periods.")
    else:
        k = res.kupiec
        verdict = ("calibrated" if k.well_calibrated else "mis-calibrated") \
            if k.well_calibrated is not None else "n/a"
        kc = st.columns(4)
        kc[0].metric("Exceedances", f"{res.n_exceedances} / {res.n_periods}")
        kc[1].metric("Observed rate", f"{k.observed_rate:.0%}", help=f"Expected ~{k.expected_rate:.0%}")
        kc[2].metric("Kupiec LR", f"{k.lr_statistic:.2f}" if k.lr_statistic is not None else "n/a")
        kc[3].metric("Verdict", verdict)
        df = res.as_frame().copy()
        df["exceeded"] = df["exceeded"].map(lambda v: "exceeded" if v else "within")
        st.dataframe(df.style.format({"iar_estimate": "{:,.0f}", "realised_cost": "{:,.0f}"}),
                     use_container_width=True, hide_index=True)
        st.caption("Low statistical power with few settled days - a readout, not a verdict.")

    st.markdown("### Calibrate sigma")
    if st.button("Run sigma calibration"):
        with st.spinner("Sweeping sigma..."):
            with Session() as s:
                cal = calibrate_sigma(
                    s, db["pid"], forecast_records=db["forecast_records"],
                    dam_price_map=db["dam_map"], position_map=db["pos_map"],
                    capacity_mwh=db["cap"],
                    engine_config=EngineConfig(n_scenarios=3000, confidence=confidence, seed=SEED),
                    iar_type=bt_basis,
                )
        if cal.recommended_sigma_fraction is None:
            st.info(cal.note)
        else:
            st.metric("Recommended sigma", f"{cal.recommended_sigma_fraction:.0%}")
            gdf = pd.DataFrame([(s_, r) for s_, r in cal.grid if r is not None],
                               columns=["sigma_fraction", "exceedance_rate"])
            if not gdf.empty:
                gfig = go.Figure()
                gfig.add_scatter(x=gdf["sigma_fraction"], y=gdf["exceedance_rate"], mode="lines+markers")
                gfig.add_hline(y=cal.target_rate, line_color="green", line_dash="dot",
                               annotation_text=f"target {cal.target_rate:.0%}")
                gfig.update_layout(title="Exceedance rate vs sigma", xaxis_title="sigma",
                                   yaxis_title="exceedance rate", height=300, margin=dict(t=50, b=40))
                st.plotly_chart(gfig, use_container_width=True)
