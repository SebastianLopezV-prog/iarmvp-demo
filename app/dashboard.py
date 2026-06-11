"""Streamlit dashboard (Task 4.1): UI only, source-agnostic.

The Imbalance-at-Risk Command Centre. This module contains no simulation,
database or API logic: it talks exclusively to a :class:`~data_source.DataSource`
(see ``app/data_source.py``), which reads only the SQLite database. Live feeds
(Optimeering / markets SDK) are never imported here.

Run:  ``streamlit run app/dashboard.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors as pcolors
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

#: Auto-reload the page every N seconds so the view tracks the refreshed database.
AUTO_REFRESH_SECONDS = 60

# ``streamlit run app/dashboard.py`` puts app/ on sys.path[0]; make the sibling
# import robust regardless of the working directory the app is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_source import get_data_source  # noqa: E402

# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #
VOLUE_ORANGE = "#FF5C39"
INK = "#141821"
TEAL = "#1FA8A0"
MUTED = "#6b7280"
OK_GREEN = "#1f9d57"
WARN_AMBER = "#E08A00"
BREACH_RED = "#D8453B"
FONT = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"

#: (label, colour) per severity. ``None`` => within limit.
SEVERITY = {
    "hard": ("Breach", BREACH_RED),
    "soft": ("Approaching limit", WARN_AMBER),
    None: ("Within limit", OK_GREEN),
}

_CSS = f"""
<style>
  html, body, [class*="css"] {{ font-family: {FONT}; }}
  [data-testid="stAppViewContainer"] {{ background: #f5f6f8; }}
  .block-container {{ padding-top: 1.1rem; padding-bottom: 3rem; max-width: 1500px; }}

  .volue-bar {{
      background: linear-gradient(100deg, {INK} 0%, #1d2533 100%); color: #fff;
      border-radius: 14px; padding: 16px 22px; margin-bottom: 14px;
      box-shadow: 0 6px 20px rgba(20,24,33,.18);
      border-bottom: 3px solid {VOLUE_ORANGE};
      display: flex; justify-content: space-between; align-items: center;
  }}
  .volue-bar .brand {{ font-weight: 800; letter-spacing: 4px; font-size: 1.05rem; }}
  .volue-bar .brand span {{ color: {VOLUE_ORANGE}; }}
  .volue-bar .title {{ font-weight: 600; letter-spacing: 0; opacity: .9; margin-left: 12px; }}
  .volue-bar .meta {{ font-size: 0.82rem; opacity: 0.82; text-align: right; }}
  .volue-bar .tag {{ background: {VOLUE_ORANGE}; color: #fff; border-radius: 6px;
                     padding: 1px 7px; font-weight: 700; font-size: .72rem; letter-spacing: 1px; }}

  .sec {{ font-size: 1.05rem; font-weight: 800; letter-spacing: .4px;
          color: {INK}; margin: 18px 0 4px; }}
  /* Make Streamlit captions readable (they default to a very faint grey). */
  [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{
      color: #4b5563 !important; font-size: 0.84rem !important; }}

  .kpi {{
      border: 1px solid #ebedf0; border-top: 3px solid #d7dade;
      border-radius: 14px; padding: 16px 18px; background: #fff; height: 100%;
      box-shadow: 0 2px 10px rgba(20,24,33,.05);
  }}
  .kpi .lbl {{ font-size: 0.68rem; letter-spacing: .4px; color: {MUTED};
               text-transform: uppercase; min-height: 2.4em; line-height: 1.25; }}
  .kpi .val {{ font-size: 1.7rem; font-weight: 800; color: {INK}; line-height: 1.1;
               white-space: nowrap; margin-top: 6px; }}
  .kpi .sub {{ font-size: 0.78rem; color: {MUTED}; margin-top: 6px; min-height: 2.2em; }}

  .dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%;
          margin-right: 6px; vertical-align: middle; }}
  .chip {{ display: inline-block; padding: 3px 11px; border-radius: 999px;
           font-size: 0.74rem; font-weight: 600; }}

  table.lim {{ width: 100%; border-collapse: collapse; }}
  table.lim th {{ text-align: left; color: {MUTED}; font-size: 0.72rem; font-weight: 700;
                  text-transform: uppercase; letter-spacing: .4px; padding: 0 8px 8px; }}
  table.lim td {{ padding: 9px 8px; border-top: 1px solid #eef0f2; font-size: 0.88rem;
                  color: {INK}; }}
  table.lim td:first-child {{ font-weight: 600; }}

  .feed {{ border-left: 3px solid #d7dade; padding: 8px 12px; margin-bottom: 9px;
           background: #fff; border-radius: 0 10px 10px 0; box-shadow: 0 1px 6px rgba(20,24,33,.04); }}
  .feed .ttl {{ font-weight: 600; font-size: 0.86rem; color: {INK}; }}
  .feed .bdy {{ font-size: 0.8rem; color: #555; margin-top: 1px; }}
  .feed .tms {{ font-size: 0.72rem; color: #9aa0a6; }}

  div[data-testid="stPlotlyChart"] {{ background: #fff; border: 1px solid #ebedf0;
       border-radius: 14px; padding: 8px 6px; box-shadow: 0 2px 10px rgba(20,24,33,.05); }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
  .stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
</style>
"""


# --------------------------------------------------------------------------- #
# Cached reads (keyed by the source *kind* string)
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
# Formatters / small helpers
# --------------------------------------------------------------------------- #
def eur(x, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"EUR {x:+,.0f}" if signed else f"EUR {x:,.0f}"


def pct(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.0%}"


def fmt_ts(ts) -> str:
    if ts is None or pd.isna(ts):
        return "n/a"
    return pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M")


def section(title: str) -> None:
    st.markdown(f"<div class='sec'>{title}</div>", unsafe_allow_html=True)


def _style_fig(fig: go.Figure) -> go.Figure:
    """Apply the shared chart look: clean font, transparent bg, soft gridlines."""
    fig.update_layout(
        font=dict(family=FONT, size=12, color="#374151"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=12, r=12, t=12, b=10),
        legend=dict(orientation="h", y=1.14, x=0, font=dict(size=11)),
        hoverlabel=dict(font_size=12),
    )
    tick = dict(size=12, color="#2b3038")
    title = dict(size=12.5, color="#4b5563")
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#e5e7eb",
                     tickfont=tick, title_font=title)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f1f4", zeroline=False,
                     tickfont=tick, title_font=title)
    return fig


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #
def render_header(pf: dict, ov: dict | None, kind: str) -> None:
    as_of = fmt_ts(ov["run_ts"]) if ov else "no run"
    warn = ""
    if ov and (ov["n_warnings"] or ov["n_breaches"]):
        n = ov["n_warnings"] + ov["n_breaches"]
        warn = (f" &nbsp;&middot;&nbsp; <span class='dot' style='background:{WARN_AMBER}'></span>"
                f"<span style='color:#f3c98b'>{n} active warning(s)</span>")
    tag = "LIVE" if kind == "live" else "DEMO"
    st.markdown(
        f"""
        <div class="volue-bar">
          <div><span class="brand">VOL<span>U</span>E</span>
               <span class="title">Imbalance at Risk</span></div>
          <div class="meta">{pf['name']} &nbsp;&middot;&nbsp; Area {pf['price_area']}
               &nbsp;&middot;&nbsp; <span class="tag">{tag}</span>
               &nbsp; as of {as_of} (Norway){warn}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str, severity=None, show_chip: bool = True) -> str:
    text, colour = SEVERITY.get(severity, SEVERITY[None])
    top = colour if show_chip else "#d7dade"
    chip = (f"<div class='chip' style='background:{colour}1A;color:{colour}'>"
            f"<span class='dot' style='background:{colour}'></span>{text}</div>") if show_chip else ""
    return f"""
      <div class="kpi" style="border-top-color:{top}">
        <div class="lbl">{label}</div>
        <div class="val">{value}</div>
        <div class="sub">{sub}</div>
        {chip}
      </div>"""


def render_kpis(ov: dict) -> None:
    g, s = ov["gross"], ov["spread"]
    cards = [
        kpi_card("Period Gross IaR, remaining day", eur(g["period_iar"]),
                 f"Limit {eur(g['limit'])} &middot; {pct(g['utilisation'])} utilised", g["severity"]),
        kpi_card("Period Spread IaR, remaining day", eur(s["period_iar"]),
                 f"Limit {eur(s['limit'])} &middot; {pct(s['utilisation'])} utilised", s["severity"]),
        kpi_card("Peak MTU Gross IaR",
                 eur(g["peak_mtu_iar"]) if g["peak_mtu_iar"] is not None else "n/a",
                 "Worst single 15-min MTU", show_chip=False),
        kpi_card("Overperformance ratio",
                 f"{ov['overperformance_ratio']:.2f}" if ov["overperformance_ratio"] is not None else "n/a",
                 "Period IaR vs naive sum of MTU IaRs", show_chip=False),
    ]
    for col, html in zip(st.columns(4), cards):
        col.markdown(html, unsafe_allow_html=True)


def render_intraday(df: pd.DataFrame, basis: str) -> None:
    section("Intraday IaR (15-min MTUs)")
    if df.empty:
        st.info("No per-MTU series for this run yet.")
        return
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colours = [VOLUE_ORANGE if not p else "#F2C9B8" for p in df["is_past"]]
    fig.add_bar(x=df["timestamp"], y=df["forecast_iar"], name="Forecast IaR",
                marker_color=colours, marker_line_width=0)
    if df["realised_iar"].notna().any():
        fig.add_bar(x=df["timestamp"], y=df["realised_iar"], name="Realised cost",
                    marker_color="#9aa0a6", opacity=0.7, marker_line_width=0)
    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df["position_mwh"], name="Position (MWh)",
                   line=dict(color=TEAL, width=2)),
        secondary_y=True,
    )
    mtu_limit = float(df["mtu_limit"].iloc[0])
    if pd.notna(mtu_limit):
        fig.add_hline(y=mtu_limit, line=dict(color=BREACH_RED, dash="dash"),
                      annotation_text="MTU limit", annotation_position="top left")
    _style_fig(fig)
    fig.update_layout(barmode="overlay", height=340, bargap=0.2)
    fig.update_yaxes(title_text=f"{basis.capitalize()} IaR (EUR)", secondary_y=False)
    fig.update_yaxes(title_text="Position (MWh)", secondary_y=True, showgrid=False)
    fig.update_xaxes(title_text="Time (Norway, CET/CEST)")
    st.plotly_chart(fig, use_container_width=True)


def _heat_grid(df: pd.DataFrame, value_col: str):
    """Pivot a tidy [hour, quarter, value] frame to a full 24h x 4-quarter grid."""
    return (
        df.pivot_table(index="quarter", columns="hour", values=value_col, aggfunc="mean")
        .reindex(index=[0, 15, 30, 45], columns=list(range(24)))
    )


def _rounded_heatmap(grid, *, colorscale, colorbar_title: str, diverging: bool = False):
    """A heatmap drawn as individual rounded-corner cells (one per MTU)."""
    z = grid.values  # rows = quarters, cols = hours
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        vmin, vmax = 0.0, 1.0
    elif diverging:
        m = float(np.nanmax(np.abs(finite))) or 1.0
        vmin, vmax = -m, m
    else:
        vmin, vmax = float(finite.min()), float(finite.max())
        if vmin == vmax:
            vmax = vmin + 1.0

    def color_at(v: float) -> str:
        t = 0.5 if vmax == vmin else min(max((v - vmin) / (vmax - vmin), 0.0), 1.0)
        return pcolors.sample_colorscale(colorscale, [t])[0]

    quarters, hours = list(grid.index), list(grid.columns)
    nq = len(quarters)
    pad, rx, ry = 0.07, 0.17, 0.17
    fig = go.Figure()
    hx, hy, ht = [], [], []
    for qi in range(nq):
        for hi in range(len(hours)):
            v = z[qi, hi]
            if not np.isfinite(v):
                continue
            x0, x1, y0, y1 = hi + pad, hi + 1 - pad, qi + pad, qi + 1 - pad
            path = (
                f"M {x0 + rx},{y0} L {x1 - rx},{y0} Q {x1},{y0} {x1},{y0 + ry} "
                f"L {x1},{y1 - ry} Q {x1},{y1} {x1 - rx},{y1} "
                f"L {x0 + rx},{y1} Q {x0},{y1} {x0},{y1 - ry} "
                f"L {x0},{y0 + ry} Q {x0},{y0} {x0 + rx},{y0} Z"
            )
            fig.add_shape(type="path", path=path, fillcolor=color_at(v),
                          line=dict(width=0), layer="below")
            hx.append(hi + 0.5)
            hy.append(qi + 0.5)
            ht.append(f"{hours[hi]:02d}:{quarters[qi]:02d}<br>EUR {v:,.0f}")
    fig.add_trace(go.Scatter(x=hx, y=hy, mode="markers",
                             marker=dict(size=18, color="rgba(0,0,0,0)"),
                             hoverinfo="text", text=ht, showlegend=False))
    # Off-canvas proxy point just to render the colour bar.
    fig.add_trace(go.Scatter(
        x=[-5], y=[-5], mode="markers", hoverinfo="skip", showlegend=False,
        marker=dict(size=6, color=[vmin], colorscale=colorscale, cmin=vmin, cmax=vmax,
                    showscale=True, colorbar=dict(title=colorbar_title, thickness=12, outlinewidth=0)),
    ))
    fig.update_layout(
        height=230, margin=dict(l=12, r=12, t=8, b=10),
        font=dict(family=FONT, size=12, color="#374151"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    axis_tick = dict(size=13, color="#2b3038")
    axis_title = dict(size=12.5, color="#4b5563")
    fig.update_xaxes(range=[-0.2, 24.2], tickvals=[i + 0.5 for i in range(len(hours))],
                     ticktext=[f"{h:02d}" for h in hours], showgrid=False, zeroline=False,
                     tickfont=axis_tick, title_text="Hour of day (00-23, Norway time)",
                     title_font=axis_title)
    fig.update_yaxes(range=[-0.2, nq + 0.2], tickvals=[i + 0.5 for i in range(nq)],
                     ticktext=[f":{q:02d}" for q in quarters], showgrid=False, zeroline=False,
                     tickfont=axis_tick, title_text="Quarter", title_font=axis_title)
    return fig


def render_heatmaps(df: pd.DataFrame, basis: str) -> None:
    """Two separate heatmaps: forecast worst-case IaR and realised cost.

    Different metrics (a P95 risk bound vs a single settled outcome), so each gets its
    own colour scale and they are never compared on one. Forecast fills the forward
    MTUs; realised fills the settled ones; the empty halves are expected.
    """
    if df.empty or "forecast_iar" not in df.columns:
        st.info("No per-MTU series for this run.")
        return

    section(f"Forecast IaR heatmap: {basis.capitalize()} P95 worst-case (Norway time)")
    st.caption("95th-percentile worst-case cost per MTU (a risk bound). Forward MTUs only; "
               "the past is not forecast.")
    st.plotly_chart(
        _rounded_heatmap(_heat_grid(df, "forecast_iar"),
                         colorscale=[[0, "#FFF1E8"], [0.5, VOLUE_ORANGE], [1, BREACH_RED]],
                         colorbar_title="EUR"),
        use_container_width=True,
    )

    section(f"Realised cost heatmap: {basis.capitalize()} settled outcome (Norway time)")
    rl = _heat_grid(df, "realised_iar")
    if not np.isfinite(np.nanmax(np.abs(rl.values))):
        st.info("No settled realised cost for this day yet. Imbalance prices publish with a "
                "delay, so this fills in as the day settles.")
        return
    st.caption("Actual cost per MTU once settled (one realised outcome, not a worst case). "
               "Blue is net revenue, red is net cost, on its own scale.")
    st.plotly_chart(
        _rounded_heatmap(rl, colorscale="RdBu_r", colorbar_title="EUR", diverging=True),
        use_container_width=True,
    )


def render_limit_table(df: pd.DataFrame) -> None:
    section("Limit status")
    if df.empty:
        st.info("No limits configured, or no run to evaluate.")
        return
    head = ("<tr><th>Limit</th><th>Current IaR</th><th>Limit</th>"
            "<th>Utilisation</th><th>Status</th></tr>")
    rows = []
    for _, r in df.iterrows():
        text, colour = SEVERITY.get(r["severity"], SEVERITY[None])
        util = r["utilisation"]
        bar_w = min(max(util, 0.0), 1.0) * 100 if pd.notna(util) else 0
        bar = (f"<div style='background:#eef0f2;border-radius:5px;height:7px;width:120px'>"
               f"<div style='background:{colour};height:7px;border-radius:5px;width:{bar_w:.0f}%'></div></div>")
        rows.append(
            f"<tr><td>{r['label']}</td><td>{eur(r['current_iar'])}</td><td>{eur(r['limit'])}</td>"
            f"<td>{bar}<span style='font-size:0.74rem;color:#9aa0a6'>{pct(util)}</span></td>"
            f"<td><span class='chip' style='background:{colour}1A;color:{colour}'>"
            f"<span class='dot' style='background:{colour}'></span>{text}</span></td></tr>"
        )
    st.markdown(f"<table class='lim'>{head}{''.join(rows)}</table>", unsafe_allow_html=True)


def render_alerts(df: pd.DataFrame) -> None:
    section("Alert feed")
    if df.empty:
        st.success("No alerts. All limits respected.")
        return
    for _, a in df.iterrows():
        _, colour = SEVERITY.get(a["severity"], SEVERITY[None])
        st.markdown(
            f"<div class='feed' style='border-left-color:{colour}'>"
            f"<div class='tms'>{fmt_ts(a['ts'])}</div>"
            f"<div class='ttl'>{a['title']}</div><div class='bdy'>{a['body']}</div></div>",
            unsafe_allow_html=True,
        )


def render_curve(df: pd.DataFrame, ov: dict | None, basis: str) -> None:
    section(f"{basis.capitalize()} IaR over time (per vintage) vs limit")
    if df.empty:
        st.info("No stored runs to chart yet. The scheduled pipeline populates this over time.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["vintage_ts"], y=df["iar_value"], name="IaR",
                             mode="lines+markers", line=dict(color=VOLUE_ORANGE, width=2.5),
                             marker=dict(size=6)))
    if df["ciar_value"].notna().any():
        fig.add_trace(go.Scatter(x=df["vintage_ts"], y=df["ciar_value"], name="CIaR",
                                 mode="lines", line=dict(color="#9aa0a6", width=1.5, dash="dot")))
    limit = ov[basis]["limit"] if ov and ov.get(basis) else None
    if limit:
        fig.add_hline(y=float(limit), line=dict(color=BREACH_RED, dash="dash"),
                      annotation_text="Day limit", annotation_position="top left")
    _style_fig(fig)
    fig.update_layout(height=360)
    fig.update_yaxes(title_text="EUR (positive = cost)")
    fig.update_xaxes(title_text="Vintage (Norway time)")
    st.plotly_chart(fig, use_container_width=True)


def render_backtest(bt: dict, basis: str) -> None:
    section(f"Backtest: {basis.capitalize()} IaR vs realised cost (Kupiec POF)")
    periods = bt.get("periods", pd.DataFrame())
    if periods is None or periods.empty:
        st.info("No settled periods to backtest yet. The scheduled pipeline accrues these as "
                "delivery days settle.")
        return
    c = st.columns(5)
    c[0].metric("Periods", bt["n_periods"])
    c[1].metric("Exceedances", bt["n_exceedances"])
    c[2].metric("Observed rate", pct(bt["observed_rate"]))
    c[3].metric("Expected rate", pct(bt["expected_rate"]))
    p = bt["kupiec_p_value"]
    c[4].metric("Kupiec p-value", "n/a" if p is None else f"{p:.3f}")
    cal = bt["well_calibrated"]
    if cal is None:
        st.caption("Calibration verdict unavailable (no observations).")
    elif cal:
        st.success("Kupiec POF: calibration not rejected. Exceedance rate is consistent with "
                   "the confidence level.")
    else:
        st.warning("Kupiec POF: calibration rejected. Observed exceedances are inconsistent with "
                   "the model (note: low power on short windows).")

    fig = go.Figure()
    fig.add_bar(x=periods["period"], y=periods["realised_cost"], name="Realised cost",
                marker_color=[BREACH_RED if e else "#9aa0a6" for e in periods["exceeded"]],
                marker_line_width=0)
    fig.add_trace(go.Scatter(x=periods["period"], y=periods["iar_estimate"], name="IaR estimate",
                             mode="lines+markers", line=dict(color=VOLUE_ORANGE, width=2.5)))
    _style_fig(fig)
    fig.update_layout(height=320)
    fig.update_yaxes(title_text="EUR (positive = cost)")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Per-period detail"):
        st.dataframe(periods, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def render_settings(kind: str):
    """Render the controls (in the Settings tab) and return the chosen selections.

    Defined and called before the other tabs so their selected values are available
    when those tabs render (Streamlit executes every tab body each run).
    """
    section("Controls")
    try:
        pfs = r_portfolios(kind)
    except Exception:  # noqa: BLE001 -- show a clean message, not a stack trace
        st.error("Data is currently unavailable. Please try again shortly.")
        st.stop()
    if pfs.empty:
        st.warning("No portfolios available yet.")
        st.stop()

    # Selections persist in the URL query params so the 60s auto-refresh (a full page
    # reload) keeps the chosen portfolio/basis instead of snapping back to defaults.
    qp = st.query_params

    def _qp(key, default):
        return qp.get(key, default)

    labels = [f"{r['price_area']} - {r['name']}" for _, r in pfs.iterrows()]
    try:
        idx_default = min(max(int(_qp("pf", 0)), 0), len(labels) - 1)
    except ValueError:
        idx_default = 0
    idx = st.selectbox("Portfolio", range(len(labels)), index=idx_default,
                       format_func=lambda i: labels[i])
    pf = pfs.iloc[idx].to_dict()

    basis_opts = ["gross", "spread"]
    basis_default = basis_opts.index(_qp("basis", "gross")) if _qp("basis", "gross") in basis_opts else 0
    basis = st.radio("IaR basis", basis_opts, index=basis_default, horizontal=True,
                     format_func=str.capitalize)

    conf_opts = [0.90, 0.95, 0.99]
    try:
        conf_default = float(_qp("conf", 0.95))
    except ValueError:
        conf_default = 0.95
    confidence = st.select_slider("Confidence (alpha)", options=conf_opts,
                                  value=conf_default if conf_default in conf_opts else 0.95,
                                  format_func=lambda c: f"{c:.0%}  (alpha={1 - c:.2f})")
    sig_opts = [0.01, 0.05, 0.10]
    try:
        sig_default = float(_qp("sig", 0.05))
    except ValueError:
        sig_default = 0.05
    significance = st.select_slider("Kupiec significance", options=sig_opts,
                                    value=sig_default if sig_default in sig_opts else 0.05)

    # Persist the selection to the URL (no rerun) for the next auto-reload.
    st.query_params.update(
        {"pf": str(idx), "basis": basis, "conf": str(confidence), "sig": str(significance)}
    )

    if st.button("Refresh now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("Figures update automatically every 15 minutes; use Refresh to pull the latest now.")
    return pf, basis, confidence, significance


def main() -> None:
    st.set_page_config(page_title="Imbalance at Risk", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)
    kind = "live"

    header_box = st.container()  # header rendered once, above the tab strip
    tabs = st.tabs(["Command Centre", "Risk Analytics", "Historical", "Settings"])

    # Settings first (in code) so the selections drive the other tabs.
    with tabs[3]:
        pf, basis, confidence, significance = render_settings(kind)
    pid = int(pf["portfolio_id"])

    with header_box:
        render_header(pf, r_overview(kind, pid, confidence), kind)

    # Each tab's content lives in its own fragment that re-runs in place every
    # AUTO_REFRESH_SECONDS (no full page reload). Defining the fragment INSIDE the tab
    # means it owns that tab's slot and REPLACES its output on each rerun, so content
    # never duplicates. Cached reads (ttl 30s) re-fetch the DB on each rerun.
    with tabs[0]:
        @st.fragment(run_every=AUTO_REFRESH_SECONDS)
        def _command_centre() -> None:
            ov = r_overview(kind, pid, confidence)
            if ov is None:
                st.info("No data available for this portfolio yet.")
                return
            render_kpis(ov)
            st.divider()
            render_intraday(r_intraday(kind, pid, basis), basis)
            render_heatmaps(r_heatmap(kind, pid, basis), basis)
            st.divider()
            left, right = st.columns([3, 2])
            with left:
                render_limit_table(r_limits(kind, pid))
            with right:
                render_alerts(r_alerts(kind, pid))
        _command_centre()

    with tabs[1]:
        @st.fragment(run_every=AUTO_REFRESH_SECONDS)
        def _risk_analytics() -> None:
            render_curve(r_curve(kind, pid, basis), r_overview(kind, pid, confidence), basis)
            st.caption("Portfolio, Gross/Spread basis and confidence are in the Settings tab.")
        _risk_analytics()

    with tabs[2]:
        @st.fragment(run_every=AUTO_REFRESH_SECONDS)
        def _historical() -> None:
            render_backtest(r_backtest(kind, pid, basis, significance), basis)
        _historical()


if __name__ == "__main__":
    main()
