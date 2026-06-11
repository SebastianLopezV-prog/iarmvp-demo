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

#: The live view re-reads the database (in place) every N seconds.
AUTO_REFRESH_SECONDS = 120

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
  .block-container {{ padding-top: 0.8rem; padding-bottom: 1.8rem; padding-left: 1rem;
                      padding-right: 1rem; max-width: 1820px; }}
  /* Comfortable spacing (gaps ~16px, well above the 1/16in floor) with a wide content area. */
  [data-testid="stVerticalBlock"] {{ gap: 1rem; }}
  [data-testid="stHorizontalBlock"] {{ gap: 1rem; }}
  [data-testid="stMainBlockContainer"] {{ padding-left: 1.2rem; padding-right: 1.2rem; }}
  hr {{ margin: 0.9rem 0 !important; }}

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
          color: {INK}; margin: 16px 0 6px; }}
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
  .stTabs [data-baseweb="tab"] {{ font-weight: 600; color: #2b3038; }}
  .stTabs [data-baseweb="tab"] p {{ color: #2b3038 !important; }}
  .stTabs [aria-selected="true"] p {{ color: {VOLUE_ORANGE} !important; }}

  /* ---- Usage tab ---- */
  .u-hero {{ background: linear-gradient(100deg, #ffffff, #fff6f2); border: 1px solid #ffd9cc;
             border-left: 4px solid {VOLUE_ORANGE}; border-radius: 14px; padding: 16px 20px;
             margin: 4px 0 6px; box-shadow: 0 2px 10px rgba(20,24,33,.05); }}
  .u-hero {{ padding: 22px 26px; }}
  .u-hero .lead {{ font-size: 1.2rem; color: {INK}; line-height: 1.6; }}
  .u-hero .sub {{ font-size: 1.0rem; color: {MUTED}; margin-top: 10px; }}
  /* Feature cards (top rule + outline icon + large heading + body), like the reference. */
  .feat {{ padding-right: 14px; margin-bottom: 18px; }}
  .feat .rule {{ border-top: 1px solid #cfd4da; margin: 0 0 22px; }}
  .feat .ic {{ margin-bottom: 18px; height: 32px; }}
  .feat .ic svg {{ width: 32px; height: 32px; stroke: {INK}; fill: none;
                   stroke-width: 1.7; stroke-linecap: round; stroke-linejoin: round; }}
  .feat .h {{ font-size: 1.5rem; font-weight: 700; color: {INK}; line-height: 1.22; margin: 0 0 15px; }}
  .feat .p {{ font-size: 1.08rem; color: #3a4452; line-height: 1.6; }}
  .u-card {{ border: 1px solid #ebedf0; border-radius: 12px; padding: 13px 15px; background: #fff;
             height: 100%; box-shadow: 0 2px 10px rgba(20,24,33,.05); }}
  .u-card .term {{ font-weight: 800; color: {INK}; font-size: 1.02rem; }}
  .u-card .body {{ font-size: 0.93rem; color: #4b5563; margin-top: 5px; line-height: 1.5; }}
  .u-callout {{ background: #f1f6fc; border-left: 4px solid #2a7fd4; border-radius: 0 10px 10px 0;
                padding: 15px 18px; font-size: 1.05rem; color: #33414f; line-height: 1.6; }}
  .pipe {{ display: flex; align-items: stretch; gap: 0; flex-wrap: wrap; margin: 2px 0 4px; }}
  .pipe-box {{ flex: 1; min-width: 140px; background: #fff; border: 1px solid #ebedf0;
               border-top: 3px solid {VOLUE_ORANGE}; border-radius: 12px; padding: 11px 13px;
               box-shadow: 0 2px 10px rgba(20,24,33,.05); }}
  .pipe-box {{ padding: 14px 16px; }}
  .pipe-box .t {{ font-weight: 700; color: {INK}; font-size: 1.05rem; }}
  .pipe-box .s {{ font-size: 0.9rem; color: {MUTED}; margin-top: 4px; line-height: 1.45; }}
  .pipe-arrow {{ display: flex; align-items: center; justify-content: center; padding: 0 9px; }}
  .pipe-arrow::after {{ content: ''; width: 0; height: 0; border-top: 7px solid transparent;
                        border-bottom: 7px solid transparent; border-left: 11px solid {VOLUE_ORANGE}; }}
  .u-step {{ display: flex; gap: 14px; margin-bottom: 16px; align-items: flex-start; }}
  .u-step .n {{ flex: 0 0 32px; height: 32px; width: 32px; border-radius: 50%;
                background: {VOLUE_ORANGE}; color: #fff; font-weight: 700; font-size: 0.95rem;
                display: flex; align-items: center; justify-content: center; }}
  .u-step .x {{ font-size: 1.08rem; color: #374151; line-height: 1.6; padding-top: 4px; }}
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
    fig.update_layout(barmode="overlay", height=380, bargap=0.2)
    fig.update_yaxes(title_text=f"{basis.capitalize()} IaR (EUR)", secondary_y=False)
    fig.update_yaxes(title_text="Position (MWh)", secondary_y=True, showgrid=False)
    fig.update_xaxes(title_text="Time (Norway, CET/CEST)")
    st.plotly_chart(fig, use_container_width=True)


_QUARTERS = [0, 15, 30, 45]
_QUARTER_LABELS = [":00", ":15", ":30", ":45"]


def _grid_from(df: pd.DataFrame, value_col: str, *, timeline: bool):
    """Build (z, col_labels, col_hovers, divider_index) for one metric.

    timeline=True  -> chronological hour columns across the span, so a span crossing
                      midnight keeps today's 14:00 and tomorrow's 14:00 as distinct
                      columns; returns the index of the first next-day column.
    timeline=False -> a single-day clock (hours 00-23).
    """
    sub = df[df[value_col].notna()]
    ts = pd.to_datetime(sub["timestamp"])
    minute = ts.dt.minute
    rowidx = {q: i for i, q in enumerate(_QUARTERS)}
    if timeline:
        keys = sorted(set(zip(ts.dt.date, ts.dt.hour)))
        colidx = {k: i for i, k in enumerate(keys)}
        z = np.full((4, len(keys)), np.nan)
        for d, h, m, v in zip(ts.dt.date, ts.dt.hour, minute, sub[value_col]):
            if m in rowidx:
                z[rowidx[m], colidx[(d, h)]] = v
        first_date = keys[0][0] if keys else None
        col_labels = [f"{h:02d}" for (_, h) in keys]
        col_hovers = [("Today " if d == first_date else "Next day ") + f"{h:02d}" for (d, h) in keys]
        divider_index = next((i for i, (d, _) in enumerate(keys) if d != first_date), None)
        return z, col_labels, col_hovers, divider_index
    z = np.full((4, 24), np.nan)
    for h, m, v in zip(ts.dt.hour, minute, sub[value_col]):
        if m in rowidx:
            z[rowidx[m], h] = v
    labels = [f"{h:02d}" for h in range(24)]
    hovers = [f"{h:02d}" for h in range(24)]
    return z, labels, hovers, None


def _rounded_heatmap(z, *, row_labels, col_labels, col_hovers, colorscale, colorbar_title,
                     diverging: bool = False, divider_index=None, divider_label=None):
    """Heatmap drawn as individual rounded cells; arbitrary columns + optional day divider."""
    z = np.asarray(z, dtype=float)
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

    nrows, ncols = z.shape
    pad, rx, ry = 0.07, 0.17, 0.17
    fig = go.Figure()
    hx, hy, ht = [], [], []
    for qi in range(nrows):
        for ci in range(ncols):
            v = z[qi, ci]
            if not np.isfinite(v):
                continue
            x0, x1, y0, y1 = ci + pad, ci + 1 - pad, qi + pad, qi + 1 - pad
            path = (
                f"M {x0 + rx},{y0} L {x1 - rx},{y0} Q {x1},{y0} {x1},{y0 + ry} "
                f"L {x1},{y1 - ry} Q {x1},{y1} {x1 - rx},{y1} "
                f"L {x0 + rx},{y1} Q {x0},{y1} {x0},{y1 - ry} "
                f"L {x0},{y0 + ry} Q {x0},{y0} {x0 + rx},{y0} Z"
            )
            fig.add_shape(type="path", path=path, fillcolor=color_at(v),
                          line=dict(width=0), layer="below")
            hx.append(ci + 0.5)
            hy.append(qi + 0.5)
            ht.append(f"{col_hovers[ci]}{row_labels[qi]}<br>EUR {v:,.0f}")
    fig.add_trace(go.Scatter(x=hx, y=hy, mode="markers",
                             marker=dict(size=18, color="rgba(0,0,0,0)"),
                             hoverinfo="text", text=ht, showlegend=False))
    # Off-canvas proxy point just to render the colour bar.
    fig.add_trace(go.Scatter(
        x=[-5], y=[-5], mode="markers", hoverinfo="skip", showlegend=False,
        marker=dict(size=6, color=[vmin], colorscale=colorscale, cmin=vmin, cmax=vmax,
                    showscale=True, colorbar=dict(title=colorbar_title, thickness=12, outlinewidth=0)),
    ))
    if divider_index is not None:
        fig.add_shape(type="line", x0=divider_index, x1=divider_index, y0=-0.15, y1=nrows + 0.15,
                      line=dict(color="#374151", width=2, dash="dot"), layer="above")
        if divider_label:
            fig.add_annotation(x=divider_index + 0.1, y=nrows + 0.12, text=divider_label,
                               showarrow=False, xanchor="left", yanchor="bottom",
                               font=dict(size=12, color="#374151"))
    axis_tick = dict(size=13, color="#2b3038")
    axis_title = dict(size=12.5, color="#4b5563")
    fig.update_layout(
        height=300, margin=dict(l=12, r=12, t=26, b=10),
        font=dict(family=FONT, size=13, color="#374151"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(range=[-0.2, ncols + 0.2], tickvals=[i + 0.5 for i in range(ncols)],
                     ticktext=col_labels, showgrid=False, zeroline=False, tickfont=axis_tick,
                     title_text="Hour of day (Norway time)", title_font=axis_title)
    fig.update_yaxes(range=[-0.2, nrows + 0.2], tickvals=[i + 0.5 for i in range(nrows)],
                     ticktext=row_labels, showgrid=False, zeroline=False, tickfont=axis_tick,
                     title_text="Quarter", title_font=axis_title)
    return fig


def render_heatmaps(df: pd.DataFrame, basis: str) -> None:
    """Two separate heatmaps: forecast worst-case IaR (up to 24h ahead, day-demarcated)
    and realised cost (today's settled). Different metrics, each on its own scale."""
    if df.empty or "forecast_iar" not in df.columns:
        st.info("No per-MTU series for this run.")
        return

    section(f"Forecast IaR heatmap: {basis.capitalize()} P95 worst-case (Norway time)")
    st.caption("95th-percentile worst-case cost per MTU (a risk bound), projected from now up to "
               "24 hours ahead. The dotted line marks the start of the next day.")
    fz, fcl, fch, fdiv = _grid_from(df, "forecast_iar", timeline=True)
    st.plotly_chart(
        _rounded_heatmap(fz, row_labels=_QUARTER_LABELS, col_labels=fcl, col_hovers=fch,
                         colorscale=[[0, "#FFF1E8"], [0.5, VOLUE_ORANGE], [1, BREACH_RED]],
                         colorbar_title="EUR", divider_index=fdiv, divider_label="Next day"),
        use_container_width=True,
    )

    section(f"Realised cost heatmap: {basis.capitalize()} settled outcome (Norway time)")
    if not df["realised_iar"].notna().any():
        st.info("No settled realised cost for today yet. Imbalance prices publish with a delay, "
                "so this fills in as the day settles.")
        return
    st.caption("Actual cost per MTU once settled (one realised outcome, not a worst case). "
               "Blue is net revenue, red is net cost, on its own scale.")
    rz, rcl, rch, _ = _grid_from(df, "realised_iar", timeline=False)
    st.plotly_chart(
        _rounded_heatmap(rz, row_labels=_QUARTER_LABELS, col_labels=rcl, col_hovers=rch,
                         colorscale="RdBu_r", colorbar_title="EUR", diverging=True),
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
        st.info("No history to chart yet.")
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
    section(f"Backtest: {basis.capitalize()} IaR versus realised cost")
    periods = bt.get("periods", pd.DataFrame())
    if periods is None or periods.empty:
        st.info("No settled periods to show yet. The backtest compares each delivered day's "
                "realised cost against the day-ahead IaR estimate, and fills in as days settle.")
        return

    obs, exp = bt["observed_rate"], bt["expected_rate"]
    p, lr = bt["kupiec_p_value"], bt.get("kupiec_lr")
    cal = bt["well_calibrated"]

    cards = [
        kpi_card("Settled periods", str(bt["n_periods"]),
                 "Delivery days compared", show_chip=False),
        kpi_card("Exceedances", str(bt["n_exceedances"]),
                 "Days the realised cost exceeded the estimate", show_chip=False),
        kpi_card("Observed exceedance rate", pct(obs),
                 f"Expected about {pct(exp)}", show_chip=False),
        kpi_card("Kupiec p-value", "n/a" if p is None else f"{p:.3f}",
                 "Likelihood ratio " + ("n/a" if lr is None else f"{lr:.2f}"), show_chip=False),
    ]
    for col, html in zip(st.columns(4), cards):
        col.markdown(html, unsafe_allow_html=True)

    if cal is None:
        vcol, vtext = MUTED, "Calibration verdict unavailable; there are no settled observations."
    elif cal:
        vcol, vtext = (OK_GREEN, "Calibration is not rejected. The observed exceedance rate is "
                       "consistent with the confidence level at the chosen significance.")
    else:
        vcol, vtext = (WARN_AMBER, "Calibration is rejected. The observed exceedances are "
                       "inconsistent with the model. The test has low power over short windows, "
                       "so treat this as an indicator rather than a verdict.")
    st.markdown(
        f"<div class='u-callout' style='border-left-color:{vcol};background:{vcol}12'>"
        f"<b>Kupiec proportion-of-failures test.</b> {vtext}</div>",
        unsafe_allow_html=True,
    )

    st.divider()
    section("Realised cost versus IaR estimate")
    fig = go.Figure()
    fig.add_bar(x=periods["period"], y=periods["realised_cost"], name="Realised cost",
                marker_color=[BREACH_RED if e else "#9aa0a6" for e in periods["exceeded"]],
                marker_line_width=0)
    fig.add_trace(go.Scatter(x=periods["period"], y=periods["iar_estimate"], name="IaR estimate",
                             mode="lines+markers", line=dict(color=VOLUE_ORANGE, width=2.5),
                             marker=dict(size=6)))
    _style_fig(fig)
    fig.update_layout(height=360)
    fig.update_yaxes(title_text="EUR (positive = cost)")
    fig.update_xaxes(title_text="Delivery day (Norway time)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Grey bars are within the estimate; red bars are exceedances where the realised "
               "cost was worse than the day-ahead IaR. The orange line is that IaR estimate.")

    st.divider()
    section("Cumulative exceedances versus expected")
    n = len(periods)
    cum = periods["exceeded"].astype(int).cumsum().tolist()
    expected_line = [exp * (k + 1) for k in range(n)]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=periods["period"], y=cum, name="Actual (cumulative)",
                              mode="lines+markers", marker=dict(size=6),
                              line=dict(color=VOLUE_ORANGE, width=2.5, shape="hv")))
    fig2.add_trace(go.Scatter(x=periods["period"], y=expected_line, name=f"Expected at {pct(exp)}",
                              mode="lines", line=dict(color="#9aa0a6", width=1.8, dash="dash")))
    _style_fig(fig2)
    fig2.update_layout(height=320)
    fig2.update_yaxes(title_text="Exceedances")
    fig2.update_xaxes(title_text="Delivery day (Norway time)")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("When the model is well calibrated the actual breach count tracks the expected "
               "line. Sustained divergence above the line indicates the IaR is set too low.")

    st.divider()
    section("Per-period detail")
    st.dataframe(
        periods, hide_index=True, use_container_width=True,
        column_config={
            "period": st.column_config.TextColumn("Delivery day"),
            "iar_estimate": st.column_config.NumberColumn("IaR estimate (EUR)", format="%.0f"),
            "realised_cost": st.column_config.NumberColumn("Realised cost (EUR)", format="%.0f"),
            "exceeded": st.column_config.CheckboxColumn("Exceeded"),
        },
    )
    st.caption("Each day is compared against the most recent IaR estimate whose vintage precedes "
               "the delivery day, so only information available beforehand is used. The Gross or "
               "Spread basis and the significance level are set in the Settings tab.")


_ICONS = {
    "scale": "<svg viewBox='0 0 24 24'><line x1='12' y1='3' x2='12' y2='21'/>"
             "<line x1='4' y1='7' x2='20' y2='7'/><path d='M4 7l-3 7h6z'/>"
             "<path d='M20 7l-3 7h6z'/><line x1='8' y1='21' x2='16' y2='21'/></svg>",
    "activity": "<svg viewBox='0 0 24 24'><polyline points='22 12 18 12 15 21 9 3 6 12 2 12'/></svg>",
    "trending": "<svg viewBox='0 0 24 24'><polyline points='23 6 13.5 15.5 8.5 10.5 1 18'/>"
                "<polyline points='17 6 23 6 23 12'/></svg>",
    "target": "<svg viewBox='0 0 24 24'><circle cx='12' cy='12' r='10'/>"
              "<circle cx='12' cy='12' r='6'/><circle cx='12' cy='12' r='2'/></svg>",
    "layers": "<svg viewBox='0 0 24 24'><polygon points='12 2 2 7 12 12 22 7 12 2'/>"
              "<polyline points='2 17 12 22 22 17'/><polyline points='2 12 12 17 22 12'/></svg>",
    "shield": "<svg viewBox='0 0 24 24'><path d='M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z'/></svg>",
    "peak": "<svg viewBox='0 0 24 24'><polyline points='2 20 9 9 13 14 17 7 22 20'/></svg>",
    "grid": "<svg viewBox='0 0 24 24'><rect x='3' y='3' width='7' height='7'/>"
            "<rect x='14' y='3' width='7' height='7'/><rect x='14' y='14' width='7' height='7'/>"
            "<rect x='3' y='14' width='7' height='7'/></svg>",
    "bell": "<svg viewBox='0 0 24 24'><path d='M18 8a6 6 0 0 0-12 0c0 7-3 9-3 9h18s-3-2-3-9'/>"
            "<path d='M13.73 21a2 2 0 0 1-3.46 0'/></svg>",
    "clock": "<svg viewBox='0 0 24 24'><circle cx='12' cy='12' r='10'/>"
             "<polyline points='12 6 12 12 16 14'/></svg>",
    "cube": "<svg viewBox='0 0 24 24'><path d='M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4"
            "A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z'/>"
            "<polyline points='3.27 6.96 12 12.01 20.73 6.96'/><line x1='12' y1='22' x2='12' y2='12'/></svg>",
    "database": "<svg viewBox='0 0 24 24'><ellipse cx='12' cy='5' rx='9' ry='3'/>"
                "<path d='M21 12c0 1.66-4 3-9 3s-9-1.34-9-3'/>"
                "<path d='M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5'/></svg>",
    "spread": "<svg viewBox='0 0 24 24'><line x1='3' y1='12' x2='21' y2='12'/>"
              "<polyline points='7 8 3 12 7 16'/><polyline points='17 8 21 12 17 16'/></svg>",
}


def _icon(key: str) -> str:
    return _ICONS.get(key, _ICONS["activity"])


_DEFINITIONS = [
    ("Imbalance", "scale",
     "Per 15-minute period (MTU), the day-ahead position minus actual delivery. A positive "
     "value means short (delivered less than sold); a negative value means long."),
    ("Gross IaR", "activity",
     "Worst-case total settlement cost: imbalance times the absolute imbalance price, summed "
     "across the horizon. It is the full cash exposure on the imbalance position."),
    ("Spread IaR", "spread",
     "Worst-case underperformance versus the day-ahead outcome: imbalance times the difference "
     "between the imbalance price and the day-ahead price. It isolates the settlement cost from "
     "the price level already locked in."),
    ("CIaR (Expected Shortfall)", "target",
     "The average cost across scenarios beyond the IaR threshold. IaR marks the edge of the worst "
     "tail; CIaR measures how severe that tail is on average. CIaR is always at least as large as "
     "IaR."),
    ("Period IaR", "layers",
     "The quantile of cost summed across all MTUs in the horizon. The sum is taken first, per "
     "scenario, then the quantile. It is not the sum of per-MTU IaRs, which would assume every "
     "period goes wrong at once and overstate the risk."),
    ("Overperformance ratio", "shield",
     "Period IaR divided by the naive sum of per-MTU IaRs. A value below 1 quantifies the "
     "diversification benefit: the worst day is less severe than the sum of the worst individual "
     "periods. A lower value means more diversification."),
    ("Peak MTU IaR", "peak",
     "The largest single-MTU IaR across the horizon. It identifies the most exposed 15-minute "
     "period."),
]

_INPUTS = [
    ("Imbalance price spread", "activity",
     "A live forecast from Optimeering, supplied as a quantile curve per MTU. It is sampled by "
     "inverse-CDF so the forecast tails are preserved rather than refitted."),
    ("Day-ahead price", "trending",
     "The cleared spot price, used to convert the spread into an absolute imbalance price for "
     "Gross IaR."),
    ("Positions and generation", "database",
     "The portfolio's day-ahead position and forecast generation, which set the expected "
     "imbalance for each period."),
    ("Imbalance uncertainty (sigma)", "cube",
     "The single modelled quantity: the spread of the imbalance volume, set as a fraction of "
     "capacity because no historical forecast-error record exists yet."),
]

_MC_STEPS = [
    "Draw 10,000 scenarios. For each scenario and each MTU, sample an imbalance price "
    "spread and an imbalance volume.",
    "Sample the price spread by inverse-CDF from the Optimeering quantile curve, which "
    "preserves the forecast tails rather than refitting to a normal distribution. Sample "
    "the volume from a parametric distribution centred on the expected imbalance "
    "(day-ahead position minus forecast generation).",
    "Sample price and volume independently. This is a deliberate simplification for this "
    "version; no dependence between price and volume is modelled.",
    "For each scenario, compute the settlement cost per MTU and sum it across the horizon.",
    "Read the measures off the resulting distribution of summed cost: IaR is the "
    "upper-tail quantile at the chosen confidence, and CIaR is the mean beyond it.",
]

_PANELS = [
    ("KPI cards", "target",
     "Period Gross and Spread IaR for the remaining day against their limits and utilisation, "
     "plus Peak MTU IaR and the overperformance ratio."),
    ("Intraday IaR", "activity",
     "A per-MTU view across the day in Norway time. Bars are forecast IaR for forward periods; "
     "the line is the portfolio position; the dashed line is the per-MTU limit."),
    ("Forecast IaR heatmap", "grid",
     "The 95th-percentile worst-case cost per period, projected up to 24 hours ahead. The dotted "
     "divider marks the start of the next day. It is a risk bound, so values exceed a typical "
     "outcome."),
    ("Realised cost heatmap", "grid",
     "Actual settled cost per cleared period, on its own diverging scale (blue revenue, red "
     "cost). A single realised outcome, not a worst case, so it is kept on a separate scale."),
    ("Limit status", "shield",
     "Current IaR against configured euro limits, by limit type (period, rolling window, "
     "per-MTU) and basis. Limits are a risk-appetite setting in configuration, not derived by "
     "the tool."),
    ("Alert feed", "bell",
     "Limit breaches and warnings raised against the most recent run."),
    ("IaR over time", "trending",
     "The IaR estimate for each stored run against the limit line, showing how the figure moves "
     "across successive runs."),
    ("Backtest (Kupiec)", "clock",
     "Each settled period's realised cost versus the estimate that applied to it. The Kupiec "
     "test checks whether the exceedance rate matches the confidence level (about 5% at 95%). It "
     "has low power over short windows, so it is read as an indicator."),
]


def _iar_illustration() -> go.Figure:
    """Illustrative fat-tailed (Student-t) distribution; IaR and CIaR on the bottom-5% tail."""
    rng = np.random.default_rng(7)
    x = rng.standard_t(df=3, size=60000) * 800.0  # Student-t, fat tails (illustrative)
    q05 = float(np.quantile(x, 0.05))             # bottom 5%
    es = float(x[x <= q05].mean())                # average of the bottom 5%
    lo, hi = (float(v) for v in np.quantile(x, [0.003, 0.997]))
    fig = go.Figure()
    fig.add_histogram(x=x, xbins=dict(start=lo, end=hi, size=(hi - lo) / 150),
                      marker_color="#cdd3da", marker_line_width=0)
    fig.add_vrect(x0=lo, x1=q05, fillcolor=BREACH_RED, opacity=0.12, line_width=0)
    fig.add_vline(x=q05, line=dict(color=VOLUE_ORANGE, width=2.5),
                  annotation_text="IaR (bottom 5%)", annotation_position="top")
    fig.add_vline(x=es, line=dict(color=BREACH_RED, width=2, dash="dot"),
                  annotation_text="CIaR (mean of bottom 5%)", annotation_position="bottom left")
    _style_fig(fig)
    fig.update_layout(height=260, showlegend=False, bargap=0.02)
    fig.update_xaxes(range=[lo, hi], title_text="Simulated outcome (illustrative; lower is worse)")
    fig.update_yaxes(title_text="Scenarios", showticklabels=False)
    return fig


def _feat_card(term: str, icon_key: str, body: str) -> str:
    return (f"<div class='feat'><div class='rule'></div>"
            f"<div class='ic'>{_icon(icon_key)}</div>"
            f"<div class='h'>{term}</div><div class='p'>{body}</div></div>")


def _feat_grid(items, cols: int = 3) -> None:
    """Render feature cards in aligned rows (a fresh column set per row keeps tops level)."""
    for start in range(0, len(items), cols):
        columns = st.columns(cols, gap="large")
        for col, (term, icon_key, body) in zip(columns, items[start:start + cols]):
            col.markdown(_feat_card(term, icon_key, body), unsafe_allow_html=True)


def render_usage() -> None:
    section("Usage and methodology")
    st.markdown(
        "<div class='u-hero'><div class='lead'>Imbalance at Risk (IaR) quantifies the "
        "worst-case cost of a wind portfolio's imbalance over a forward horizon, at a stated "
        "confidence level. A portfolio sells volume on the day-ahead market; actual generation "
        "differs from that sold position, and the difference (the imbalance) settles at the "
        "system operator's imbalance price. IaR expresses that exposure as a single figure: at "
        "95% confidence, the imbalance cost over the horizon is not expected to exceed this "
        "amount. It is the electricity-balancing analogue of Value at Risk in finance.</div>"
        "<div class='sub'>Sign convention: a positive figure is a cost (unfavourable); a "
        "negative figure is revenue (favourable).</div></div>",
        unsafe_allow_html=True,
    )

    st.divider()
    section("How it works")
    st.markdown(
        "<div class='pipe'>"
        "<div class='pipe-box'><div class='t'>Live inputs</div><div class='s'>Optimeering price "
        "spread, day-ahead price, positions and generation</div></div>"
        "<div class='pipe-arrow'></div>"
        "<div class='pipe-box'><div class='t'>Monte Carlo</div><div class='s'>10,000 scenarios; "
        "price and volume sampled independently</div></div>"
        "<div class='pipe-arrow'></div>"
        "<div class='pipe-box'><div class='t'>Risk read-off</div><div class='s'>IaR, CIaR, "
        "per-MTU and period figures from the cost distribution</div></div>"
        "<div class='pipe-arrow'></div>"
        "<div class='pipe-box'><div class='t'>Dashboard</div><div class='s'>Stored in the "
        "database and read by this view</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(_iar_illustration(), use_container_width=True)
    st.caption("Illustrative fat-tailed (Student-t) distribution shown in the conventional "
               "Value-at-Risk orientation (lower is worse). IaR marks the bottom 5%; CIaR is the "
               "average of the shaded bottom-5% tail. In the dashboard the same tail is reported "
               "as cost, where positive is worse.")

    st.divider()
    section("Key measures")
    _feat_grid(_DEFINITIONS, cols=3)

    st.divider()
    section("Inputs and data")
    _feat_grid(_INPUTS, cols=2)

    st.divider()
    section("The Monte Carlo method")
    steps = "".join(
        f"<div class='u-step'><div class='n'>{i+1}</div><div class='x'>{s}</div></div>"
        for i, s in enumerate(_MC_STEPS)
    )
    st.markdown(steps, unsafe_allow_html=True)
    st.markdown(
        "<div class='u-callout'>Each scheduled refresh runs this simulation again, with the "
        "latest inputs, for every portfolio. The only modelled quantity is the imbalance-volume "
        "uncertainty (sigma), set as a fraction of capacity because no historical forecast-error "
        "record exists yet. Every other input is observed data.</div>",
        unsafe_allow_html=True,
    )

    st.divider()
    section("Reading each panel")
    _feat_grid(_PANELS, cols=3)

    st.divider()
    section("Confidence and refresh")
    c1, c2 = st.columns(2, gap="large")
    c1.markdown(
        "<div class='u-callout'><b>Confidence.</b> The confidence level (for example 95%) sets "
        "how far into the tail the IaR is read. Higher confidence gives a larger, more "
        "conservative figure. Stored runs use a fixed confidence; the selector drives the "
        "backtest target and the display.</div>",
        unsafe_allow_html=True,
    )
    c2.markdown(
        "<div class='u-callout'><b>Refresh.</b> The view re-reads the database every two minutes "
        "and shows a brief notice when it does. The underlying figures are regenerated by the "
        "scheduled pipeline, which runs the full simulation for each portfolio at a fixed "
        "interval.</div>",
        unsafe_allow_html=True,
    )


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
    tabs = st.tabs(["Command Centre", "Risk Analytics", "Historical", "Usage", "Settings"])

    # Settings first (in code) so the selections drive the other tabs.
    with tabs[4]:
        pf, basis, confidence, significance = render_settings(kind)
    pid = int(pf["portfolio_id"])

    with header_box:
        @st.fragment(run_every=AUTO_REFRESH_SECONDS)
        def _header() -> None:
            # Brief on-screen notice each time the live view re-reads the database.
            st.toast("Refreshing live data...")
            render_header(pf, r_overview(kind, pid, confidence), kind)
        _header()

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

    with tabs[3]:
        render_usage()


if __name__ == "__main__":
    main()
