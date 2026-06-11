"""Streamlit dashboard (Task 4.1) — UI only, source-agnostic.

The Imbalance-at-Risk **Command Centre**. This module contains *no* simulation,
database or API logic: it talks exclusively to a :class:`~data_source.DataSource`
(see ``app/data_source.py``), so the same UI renders either the real pipeline
output (``ServiceDataSource`` → ``iar.service`` → SQLite) or a fully synthetic
demo feed (``DemoDataSource``). Flip the source in the sidebar — nothing else
changes. **Live feeds (Optimeering / markets SDK) are never imported here.**

Run:  ``streamlit run app/dashboard.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ``streamlit run app/dashboard.py`` puts app/ on sys.path[0]; make the sibling
# import robust regardless of the working directory the app is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_source import get_data_source  # noqa: E402

# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #
VOLUE_ORANGE = "#FF5C39"
INK = "#10141C"
OK_GREEN = "#1f9d57"
WARN_AMBER = "#E69500"
BREACH_RED = "#D8453B"

#: (emoji, label, colour) per severity. ``None`` ⇒ within limit.
SEVERITY = {
    "hard": ("🔴", "Breach", BREACH_RED),
    "soft": ("🟠", "Approaching limit", WARN_AMBER),
    None: ("🟢", "Within limit", OK_GREEN),
}

_CSS = f"""
<style>
  .block-container {{ padding-top: 1.2rem; max-width: 1500px; }}
  .volue-bar {{
      background: {INK}; color: #fff; border-radius: 8px;
      padding: 12px 18px; margin-bottom: 4px;
      border-bottom: 3px solid {VOLUE_ORANGE};
      display: flex; justify-content: space-between; align-items: center;
  }}
  .volue-bar .brand {{ font-weight: 800; letter-spacing: 3px; }}
  .volue-bar .brand span {{ color: {VOLUE_ORANGE}; }}
  .volue-bar .meta {{ font-size: 0.85rem; opacity: 0.85; }}
  .kpi {{
      border: 1px solid #e7e7ea; border-top: 4px solid #ccc;
      border-radius: 8px; padding: 12px 14px; background: #fff; height: 100%;
  }}
  .kpi .lbl {{ font-size: 0.68rem; letter-spacing: 0.3px; color: #6b7280;
               text-transform: uppercase; min-height: 2.4em; line-height: 1.2; }}
  .kpi .val {{ font-size: 1.55rem; font-weight: 800; color: {INK}; line-height: 1.15;
               white-space: nowrap; margin-top: 4px; }}
  .kpi .sub {{ font-size: 0.78rem; color: #6b7280; margin-top: 4px; min-height: 2.4em; }}
  .chip {{ display: inline-block; padding: 2px 9px; border-radius: 999px;
           font-size: 0.74rem; font-weight: 600; margin-top: 8px; }}
  .feed {{ border-left: 3px solid #ccc; padding: 6px 10px; margin-bottom: 8px; background: #fafafa; }}
  .feed .ttl {{ font-weight: 600; font-size: 0.86rem; }}
  .feed .bdy {{ font-size: 0.8rem; color: #555; }}
  .feed .tms {{ font-size: 0.72rem; color: #999; }}
</style>
"""


# --------------------------------------------------------------------------- #
# Cached reads (keyed by the source *kind* string, so swapping is transparent)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=30, show_spinner=False)
def r_portfolios(kind: str) -> pd.DataFrame:
    return get_data_source(kind).list_portfolios()


@st.cache_data(ttl=30, show_spinner=False)
def r_overview(kind: str, pid: int, confidence: float) -> dict | None:
    return get_data_source(kind).overview(pid, confidence=confidence)


@st.cache_data(ttl=30, show_spinner=False)
def r_intraday(kind: str, pid: int, basis: str) -> pd.DataFrame:
    return get_data_source(kind).intraday(pid, basis=basis)


@st.cache_data(ttl=30, show_spinner=False)
def r_heatmap(kind: str, pid: int, basis: str) -> pd.DataFrame:
    return get_data_source(kind).heatmap(pid, basis=basis)


@st.cache_data(ttl=30, show_spinner=False)
def r_limits(kind: str, pid: int) -> pd.DataFrame:
    return get_data_source(kind).limit_status(pid)


@st.cache_data(ttl=30, show_spinner=False)
def r_alerts(kind: str, pid: int) -> pd.DataFrame:
    return get_data_source(kind).alerts(pid)


@st.cache_data(ttl=30, show_spinner=False)
def r_curve(kind: str, pid: int, basis: str) -> pd.DataFrame:
    return get_data_source(kind).iar_curve(pid, basis=basis)


@st.cache_data(ttl=30, show_spinner=False)
def r_backtest(kind: str, pid: int, basis: str, significance: float) -> dict:
    return get_data_source(kind).backtest(pid, basis=basis, significance=significance)


# --------------------------------------------------------------------------- #
# Small formatters
# --------------------------------------------------------------------------- #
def eur(x, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"€{x:+,.0f}" if signed else f"€{x:,.0f}"


def pct(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:.0%}"


def fmt_ts(ts) -> str:
    if ts is None or pd.isna(ts):
        return "—"
    return pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M")


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #
def render_header(pf: dict, ov: dict | None, kind: str) -> None:
    as_of = fmt_ts(ov["run_ts"]) if ov else "no run"
    warn = ""
    if ov and (ov["n_warnings"] or ov["n_breaches"]):
        n = ov["n_warnings"] + ov["n_breaches"]
        warn = f" &nbsp;·&nbsp; <span style='color:{WARN_AMBER}'>⚠ {n} active warning(s)</span>"
    tag = "LIVE" if kind == "live" else "DEMO"
    st.markdown(
        f"""
        <div class="volue-bar">
          <div class="brand">VOL<span>U</span>E &nbsp; <span style="color:#fff;font-weight:600;
               letter-spacing:0;">Imbalance at Risk</span></div>
          <div class="meta">{pf['name']} &nbsp;·&nbsp; Area {pf['price_area']}
               &nbsp;·&nbsp; <b>{tag}</b> · as of {as_of} (Norway){warn}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str, severity) -> str:
    emoji, text, colour = SEVERITY.get(severity, SEVERITY[None])
    return f"""
      <div class="kpi" style="border-top-color:{colour}">
        <div class="lbl">{label}</div>
        <div class="val">{value}</div>
        <div class="sub">{sub}</div>
        <div class="chip" style="background:{colour}20;color:{colour}">{emoji} {text}</div>
      </div>"""


def render_kpis(ov: dict) -> None:
    g, s = ov["gross"], ov["spread"]
    cards = [
        kpi_card(
            "Period Gross IaR — Remaining day", eur(g["period_iar"]),
            f"Limit {eur(g['limit'])} · {pct(g['utilisation'])} utilised", g["severity"],
        ),
        kpi_card(
            "Period Spread IaR — Remaining day", eur(s["period_iar"]),
            f"Limit {eur(s['limit'])} · {pct(s['utilisation'])} utilised", s["severity"],
        ),
        kpi_card(
            "Peak MTU Gross IaR",
            eur(g["peak_mtu_iar"]) if g["peak_mtu_iar"] is not None else "n/a",
            "Worst single 15-min MTU" if g["peak_mtu_iar"] is not None
            else "engine emits period IaR only", None if g["peak_mtu_iar"] is not None else None,
        ),
        kpi_card(
            "Overperformance ratio",
            f"{ov['overperformance_ratio']:.2f}" if ov["overperformance_ratio"] is not None else "n/a",
            "Net benefit vs naive sum of MTU IaRs" if ov["overperformance_ratio"] is not None
            else "not available from this source", None,
        ),
    ]
    for col, html in zip(st.columns(4), cards):
        col.markdown(html, unsafe_allow_html=True)


def render_intraday(df: pd.DataFrame, basis: str) -> None:
    st.markdown("##### Intraday IaR — 15-min MTUs")
    if df.empty:
        st.info(
            "No per-MTU IaR series from this source. The MVP engine emits a single "
            "**period** IaR (not a per-MTU series) — switch to the **Demo** source to "
            "preview this panel, or see `docs/assumptions.md`.",
            icon="ℹ️",
        )
        return
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colours = [VOLUE_ORANGE if not p else "#F2C9B8" for p in df["is_past"]]
    fig.add_bar(x=df["timestamp"], y=df["forecast_iar"], name="Forecast IaR",
                marker_color=colours, opacity=0.9)
    if df["realised_iar"].notna().any():
        fig.add_bar(x=df["timestamp"], y=df["realised_iar"], name="Realised IaR",
                    marker_color="#9aa0a6", opacity=0.65)
    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df["position_mwh"], name="Position (MWh)",
                   line=dict(color="#2a9d8f", width=2)),
        secondary_y=True,
    )
    mtu_limit = float(df["mtu_limit"].iloc[0])
    if pd.notna(mtu_limit):
        fig.add_hline(y=mtu_limit, line=dict(color=BREACH_RED, dash="dash"),
                      annotation_text="MTU limit", annotation_position="top left")
    fig.update_layout(
        barmode="overlay", height=340, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.12), bargap=0.15,
    )
    fig.update_yaxes(title_text=f"{basis.capitalize()} IaR (€)", secondary_y=False)
    fig.update_yaxes(title_text="Position (MWh)", secondary_y=True, showgrid=False)
    fig.update_xaxes(title_text="Time (Norway, CET/CEST)")
    st.plotly_chart(fig, use_container_width=True)


def render_heatmap(df: pd.DataFrame, basis: str) -> None:
    st.markdown(f"##### MTU Risk Heatmap — {basis.capitalize()} IaR intensity (Norway time)")
    if df.empty:
        st.info("No per-MTU series for this run.", icon="ℹ️")
        return
    # Always render a full 24h × 4-quarter clock-face; MTUs the run didn't cover
    # (already-settled morning, or beyond the forecast horizon) stay grey.
    grid = (
        df.pivot_table(index="quarter", columns="hour", values="iar", aggfunc="mean")
        .reindex(index=[0, 15, 30, 45], columns=list(range(24)))
    )
    fig = go.Figure(
        go.Heatmap(
            z=grid.values, x=[f"{h:02d}" for h in grid.columns],
            y=[f":{q:02d}" for q in grid.index],
            colorscale=[[0, "#FFF6EE"], [0.5, VOLUE_ORANGE], [1, BREACH_RED]],
            colorbar=dict(title="€"), hoverongaps=False,
            xgap=1, ygap=1,
        )
    )
    fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="Hour of day (00–23, Norway time)", yaxis_title="Quarter",
                      plot_bgcolor="#e9e9ec")  # grey shows through for un-simulated MTUs
    fig.update_xaxes(dtick=1)
    st.plotly_chart(fig, use_container_width=True)


def render_limit_table(df: pd.DataFrame) -> None:
    st.markdown("##### Limit Status")
    if df.empty:
        st.info("No limits configured / no run to evaluate.", icon="ℹ️")
        return
    head = ("<tr style='text-align:left;color:#6b7280;font-size:0.78rem'>"
            "<th>Limit</th><th>Current IaR</th><th>Limit</th><th>Utilisation</th><th>Status</th></tr>")
    rows = []
    for _, r in df.iterrows():
        emoji, text, colour = SEVERITY.get(r["severity"], SEVERITY[None])
        util = r["utilisation"]
        bar_w = min(max(util, 0.0), 1.0) * 100 if pd.notna(util) else 0
        bar = (f"<div style='background:#eee;border-radius:4px;height:8px;width:120px'>"
               f"<div style='background:{colour};height:8px;border-radius:4px;width:{bar_w:.0f}%'></div></div>")
        rows.append(
            f"<tr style='border-top:1px solid #eee'>"
            f"<td style='padding:6px 8px'>{r['label']}</td>"
            f"<td>{eur(r['current_iar'])}</td><td>{eur(r['limit'])}</td>"
            f"<td>{bar}<span style='font-size:0.74rem;color:#888'>{pct(util)}</span></td>"
            f"<td><span class='chip' style='background:{colour}20;color:{colour}'>{emoji} {text}</span></td></tr>"
        )
    st.markdown(f"<table style='width:100%'>{head}{''.join(rows)}</table>", unsafe_allow_html=True)


def render_alerts(df: pd.DataFrame) -> None:
    st.markdown("##### Alert Feed")
    if df.empty:
        st.success("No alerts — all limits respected.", icon="✅")
        return
    for _, a in df.iterrows():
        _, _, colour = SEVERITY.get(a["severity"], SEVERITY[None])
        st.markdown(
            f"<div class='feed' style='border-left-color:{colour}'>"
            f"<div class='tms'>{fmt_ts(a['ts'])}</div>"
            f"<div class='ttl'>{a['title']}</div><div class='bdy'>{a['body']}</div></div>",
            unsafe_allow_html=True,
        )


def render_curve(df: pd.DataFrame, ov: dict | None, basis: str) -> None:
    st.markdown(f"##### {basis.capitalize()} IaR over time (per vintage) vs limit")
    if df.empty:
        st.info("No stored runs to chart. Run `scripts/run_iar.py --store` (and "
                "`backfill_iar.py` for a series), or switch to the Demo source.", icon="ℹ️")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["vintage_ts"], y=df["iar_value"], name="IaR",
                             mode="lines+markers", line=dict(color=VOLUE_ORANGE, width=2)))
    if df["ciar_value"].notna().any():
        fig.add_trace(go.Scatter(x=df["vintage_ts"], y=df["ciar_value"], name="CIaR",
                                 mode="lines", line=dict(color="#888", width=1, dash="dot")))
    limit = ov[basis]["limit"] if ov and ov.get(basis) else None
    if limit:
        fig.add_hline(y=float(limit), line=dict(color=BREACH_RED, dash="dash"),
                      annotation_text="Day limit", annotation_position="top left")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.1),
                      yaxis_title="€ (positive = cost)", xaxis_title="Vintage")
    st.plotly_chart(fig, use_container_width=True)


def render_backtest(bt: dict, basis: str) -> None:
    st.markdown(f"##### Backtest — {basis.capitalize()} IaR vs realised cost (Kupiec POF)")
    periods = bt.get("periods", pd.DataFrame())
    if periods is None or periods.empty:
        st.info("No settled periods to backtest yet. Needs realised cost "
                "(`load_actuals.py`) + backfilled vintages (`backfill_iar.py`), or the "
                "Demo source.", icon="ℹ️")
        return
    c = st.columns(5)
    c[0].metric("Periods", bt["n_periods"])
    c[1].metric("Exceedances", bt["n_exceedances"])
    c[2].metric("Observed rate", pct(bt["observed_rate"]))
    c[3].metric("Expected rate", pct(bt["expected_rate"]))
    p = bt["kupiec_p_value"]
    c[4].metric("Kupiec p-value", "—" if p is None else f"{p:.3f}")
    cal = bt["well_calibrated"]
    if cal is None:
        st.caption("Calibration verdict unavailable (no observations).")
    elif cal:
        st.success("Kupiec POF: calibration **not rejected** — exceedance rate consistent "
                   "with the confidence level.", icon="✅")
    else:
        st.warning("Kupiec POF: calibration **rejected** — observed exceedances inconsistent "
                   "with the model (note: low power on short windows).", icon="⚠️")

    fig = go.Figure()
    fig.add_bar(x=periods["period"], y=periods["realised_cost"], name="Realised cost",
                marker_color=["#D8453B" if e else "#9aa0a6" for e in periods["exceeded"]])
    fig.add_trace(go.Scatter(x=periods["period"], y=periods["iar_estimate"], name="IaR estimate",
                             mode="lines+markers", line=dict(color=VOLUE_ORANGE, width=2)))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.12), yaxis_title="€ (positive = cost)")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Per-period detail"):
        st.dataframe(periods, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def render_settings(kind: str):
    """Render the controls (in the Settings tab) and return the chosen selections.

    Defined and called *before* the other tabs so their selected values are
    available when those tabs render (Streamlit executes every tab body each run).
    """
    st.markdown("#### Controls")
    try:
        pfs = r_portfolios(kind)
    except Exception as exc:  # noqa: BLE001 — surface DB/import issues gracefully
        st.error(f"Could not load portfolios from the database: {exc}")
        st.stop()
    if pfs.empty:
        st.warning("No portfolios found. Seed them with `scripts/seed_demo.py` and run "
                   "`scripts/run_iar.py --store`.")
        st.stop()

    labels = [f"{r['price_area']} · {r['name']}" for _, r in pfs.iterrows()]
    idx = st.selectbox("Portfolio", range(len(labels)), format_func=lambda i: labels[i])
    pf = pfs.iloc[idx].to_dict()

    basis = st.radio("IaR basis", ["gross", "spread"], horizontal=True, format_func=str.capitalize)
    confidence = st.select_slider("Confidence (α)", options=[0.90, 0.95, 0.99], value=0.95,
                                  format_func=lambda c: f"{c:.0%}  (α={1 - c:.2f})")
    significance = st.select_slider("Kupiec significance", options=[0.01, 0.05, 0.10], value=0.05)
    if st.button("↻ Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption(
        "Data is read live from the SQLite database via `iar.service` — the dashboard "
        "never calls Optimeering or the markets SDK directly. The pipeline (live feeds → "
        "DB) is run by the backend scripts:"
    )
    st.code("python scripts/run_iar.py --area NO2 --store\n"
            "python scripts/backfill_iar.py --area NO2 --start=-P5D --end=P0D\n"
            "python scripts/run_backtest.py --area NO2", language="bash")
    st.caption("Confidence (α) is the level the stored run was computed at; changing it here "
               "drives the backtest target — re-run the pipeline to re-estimate at a new α.")
    return pf, basis, confidence, significance


def main() -> None:
    st.set_page_config(page_title="Imbalance at Risk", page_icon="⚡", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)
    kind = "live"

    header_box = st.container()  # filled after we know the selection (renders above tabs)
    tabs = st.tabs(["⊞ Command Centre", "📈 Risk Analytics", "🗓 Historical", "⚙ Settings"])

    # Settings first (in code) so the selections drive the other tabs.
    with tabs[3]:
        pf, basis, confidence, significance = render_settings(kind)
    pid = int(pf["portfolio_id"])

    ov = r_overview(kind, pid, confidence)
    with header_box:
        render_header(pf, ov, kind)

    # --- Command Centre -------------------------------------------------- #
    with tabs[0]:
        if ov is None:
            st.info("No simulation run stored for this portfolio yet. Run "
                    "`scripts/run_iar.py --area " + pf["price_area"] + " --store`.", icon="ℹ️")
        else:
            render_kpis(ov)
            st.divider()
            render_intraday(r_intraday(kind, pid, basis), basis)
            render_heatmap(r_heatmap(kind, pid, basis), basis)
            st.divider()
            left, right = st.columns([3, 2])
            with left:
                render_limit_table(r_limits(kind, pid))
            with right:
                render_alerts(r_alerts(kind, pid))

    # --- Risk Analytics (IaR curve vs limit) ----------------------------- #
    with tabs[1]:
        render_curve(r_curve(kind, pid, basis), ov, basis)
        st.caption("Portfolio, Gross/Spread basis and confidence are in the Settings tab.")

    # --- Historical (backtest) ------------------------------------------- #
    with tabs[2]:
        render_backtest(r_backtest(kind, pid, basis, significance), basis)


if __name__ == "__main__":
    main()
