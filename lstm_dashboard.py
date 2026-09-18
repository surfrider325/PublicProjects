"""
lstm_dashboard.py
=================
Streamlit UI for the LSTM Breakout Predictor — prediction-only.

Models are trained offline (occasionally, by hand) via:

    python lstm_train_historical.py

This app loads the most recently saved shared weights
(models/bullish_lstm.pt, models/bearish_lstm.pt) and, conditioning on the
first 15 minutes of the latest trading day, forecasts the REST of that day's
15-minute high/low path for any ticker you enter. Bars that have already
traded are shown as actuals alongside the prediction.

Run:
    streamlit run lstm_dashboard.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# Local Massive Parquet store; yfinance-compatible download() that falls back
# to yfinance for any ticker the store doesn't hold.
import massive_feed
import joblib
import torch

warnings.filterwarnings("ignore")

# ── Shared-model pipeline (training script handles its own breakout_detector path)
from lstm_train_historical import (
    predict_next_day_shared,
    score_daily_slice,
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & THEME
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="LSTM Breakout Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d0f13;
    color: #c9d1d9;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #090b0e;
    border-right: 1px solid #1e2530;
}
section[data-testid="stSidebar"] * { color: #8b9ab0 !important; }
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stSelectbox label { color: #5a6a80 !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.08em; }

/* Main heading */
.main-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 22px;
    font-weight: 600;
    color: #e6edf3;
    letter-spacing: 0.04em;
    border-bottom: 1px solid #1e2530;
    padding-bottom: 12px;
    margin-bottom: 24px;
}
.main-title span { color: #39d353; }

/* Metric cards */
.metric-row { display: flex; gap: 12px; margin-bottom: 20px; }
.metric-card {
    flex: 1;
    background: #111419;
    border: 1px solid #1e2530;
    border-radius: 6px;
    padding: 14px 18px;
}
.metric-card .label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #4a5568;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}
.metric-card .value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 22px;
    font-weight: 600;
    color: #e6edf3;
}
.metric-card .value.up   { color: #39d353; }
.metric-card .value.down { color: #f85149; }
.metric-card .value.neutral { color: #58a6ff; }
.metric-card .sub {
    font-size: 11px;
    color: #586069;
    margin-top: 4px;
    font-family: 'IBM Plex Mono', monospace;
}

/* Section headers */
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #4a5568;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 24px 0 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1e2530;
}

/* Ticker badge */
.ticker-badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 2px 8px;
    color: #58a6ff;
    margin: 2px;
    cursor: default;
}

/* Status dot */
.dot-green  { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #39d353; margin-right: 6px; }
.dot-yellow { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #d29922; margin-right: 6px; }
.dot-red    { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #f85149; margin-right: 6px; }

/* Buttons */
div.stButton > button {
    background: #161b22;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.04em;
    padding: 8px 18px;
    transition: all 0.15s ease;
}
div.stButton > button:hover {
    background: #1c2128;
    border-color: #58a6ff;
    color: #58a6ff;
}
div.stButton > button[kind="primary"] {
    background: #0d4a2e;
    border-color: #39d353;
    color: #39d353;
}
div.stButton > button[kind="primary"]:hover {
    background: #145a38;
}

/* Dataframe */
.stDataFrame { border: 1px solid #1e2530 !important; border-radius: 6px; }

/* Divider */
hr { border-color: #1e2530; margin: 16px 0; }

/* Expander */
.streamlit-expanderHeader {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #4a5568 !important;
}
</style>
""", unsafe_allow_html=True)

SAVE_DIR = "models"
Path(SAVE_DIR).mkdir(exist_ok=True)

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE  — only the most recent forecast is retained
# ══════════════════════════════════════════════════════════════════════════════

if "forecast" not in st.session_state:
    st.session_state.forecast = None   # dict: {ticker, direction, score, note, df}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def saved_models() -> list[str]:
    """Directions with shared weights currently on disk."""
    return sorted([
        p.stem.replace("_lstm", "")
        for p in Path(SAVE_DIR).glob("*_lstm.pt")
        if p.stem.replace("_lstm", "") in ("bullish", "bearish")
    ])


def holdout_coverage(direction: str):
    """Read the saved validation coverage for a direction, if present."""
    import json
    p = Path(SAVE_DIR) / f"{direction}_eval.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())["model"]["coverage"]
    except Exception:
        return None


def detect_direction(ticker: str):
    """Score recent daily action to choose the bullish/bearish model.
    Returns (direction, score) or (None, None) if no data."""
    daily = massive_feed.download(ticker, period="2y", interval="1d",
                                  auto_adjust=True, progress=False)
    if daily is None or daily.empty:
        return None, None
    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns = daily.columns.get_level_values(0)
    daily.index = pd.to_datetime(daily.index).tz_localize(None)
    score = score_daily_slice(daily)
    return ("bullish" if score >= 0 else "bearish"), score


def pct_class(pct: float) -> str:
    return "up" if pct >= 0 else "down"

# ══════════════════════════════════════════════════════════════════════════════
# CHART BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_prediction_chart(ticker: str, df: pd.DataFrame) -> go.Figure:
    """15-min high/low chart — predicted range bars + realized-actual overlay."""
    times    = df.index
    highs    = df["predicted_high"].values
    lows     = df["predicted_low"].values
    mids     = df["predicted_mid"].values
    is_act   = df["is_actual"].values if "is_actual" in df else np.zeros(len(df), bool)
    act_high = df["actual_high"].values if "actual_high" in df else np.full(len(df), np.nan)
    act_low  = df["actual_low"].values  if "actual_low"  in df else np.full(len(df), np.nan)
    day_high = highs.max()
    day_low  = lows.min()
    day_move = (mids[-1] / mids[0] - 1) * 100

    bar_colors = [
        "#39d353" if h > l and i > 0 and mids[i] >= mids[i-1] else "#f85149"
        for i, (h, l) in enumerate(zip(highs, lows))
    ]

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.70, 0.30],
        shared_xaxes=True,
        vertical_spacing=0.03,
    )

    # ── Predicted High/Low range bars (faded where the bar has already traded)
    for t, h, l, m, col, a in zip(times, highs, lows, mids, bar_colors, is_act):
        fig.add_trace(go.Scatter(
            x=[t, t], y=[l, h],
            mode="lines",
            line=dict(color=col, width=10),
            opacity=0.35 if a else 1.0,
            showlegend=False,
            hovertemplate=f"<b>{t.strftime('%H:%M')}</b> (pred)<br>H: ${h:.2f}<br>L: ${l:.2f}<br>Mid: ${m:.2f}<extra></extra>",
        ), row=1, col=1)

    # ── Realized actual High/Low bars (overlay, thin & bright) where traded
    if is_act.any():
        for t, ah, al in zip(times[is_act], act_high[is_act], act_low[is_act]):
            fig.add_trace(go.Scatter(
                x=[t, t], y=[al, ah],
                mode="lines",
                line=dict(color="#e6edf3", width=3),
                showlegend=False,
                hovertemplate=f"<b>{t.strftime('%H:%M')}</b> (actual)<br>H: ${ah:.2f}<br>L: ${al:.2f}<extra></extra>",
            ), row=1, col=1)
        # legend proxies
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
            line=dict(color="#58a6ff", width=8), name="Predicted"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
            line=dict(color="#e6edf3", width=3), name="Actual (realized)"), row=1, col=1)

    # ── Mid price line
    fig.add_trace(go.Scatter(
        x=times, y=mids,
        mode="lines+markers",
        name="Predicted Mid",
        line=dict(color="#58a6ff", width=1.5),
        marker=dict(size=5, color="#58a6ff"),
        hovertemplate="<b>%{x|%H:%M}</b>  Mid: $%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    # ── Day high/low annotations
    fig.add_hline(y=day_high, line=dict(color="#39d353", width=0.7, dash="dot"),
                  annotation_text=f"Day H ${day_high:.2f}",
                  annotation_font=dict(color="#39d353", size=10, family="IBM Plex Mono"),
                  annotation_position="right", row=1, col=1)
    fig.add_hline(y=day_low,  line=dict(color="#f85149", width=0.7, dash="dot"),
                  annotation_text=f"Day L ${day_low:.2f}",
                  annotation_font=dict(color="#f85149", size=10, family="IBM Plex Mono"),
                  annotation_position="right", row=1, col=1)

    # ── H/L spread per 15-min bar (row 2)
    spreads       = (highs - lows) / lows * 100
    spread_colors = ["#39d353" if s > spreads.mean() else "#4a5568" for s in spreads]
    fig.add_trace(go.Bar(
        x=times, y=spreads,
        name="H/L Spread %",
        marker_color=spread_colors,
        marker_line_width=0,
        showlegend=False,
        hovertemplate="<b>%{x|%H:%M}</b><br>Spread: %{y:.3f}%<extra></extra>",
    ), row=2, col=1)

    title_color = "#39d353" if day_move >= 0 else "#f85149"
    sign        = "▲" if day_move >= 0 else "▼"

    fig.update_layout(
        title=dict(
            text=f"<span style='font-family:IBM Plex Mono'>{ticker} — 15-min High/Low Forecast (rest of day)  "
                 f"<span style='color:{title_color}'>{sign} {abs(day_move):.2f}%</span></span>",
            font=dict(size=14, color="#e6edf3"), x=0,
        ),
        paper_bgcolor="#0d0f13",
        plot_bgcolor="#0d0f13",
        font=dict(family="IBM Plex Mono", color="#8b949e", size=11),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right",
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        margin=dict(l=10, r=90, t=50, b=10),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#161b22", bordercolor="#30363d",
                        font=dict(family="IBM Plex Mono", size=11)),
    )
    for row in [1, 2]:
        fig.update_xaxes(showgrid=False, tickformat="%H:%M",
                         tickfont=dict(size=10, family="IBM Plex Mono"),
                         color="#4a5568", row=row, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#1a1f26",
                         tickfont=dict(size=10, family="IBM Plex Mono"),
                         color="#4a5568", row=row, col=1)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR  — status only (training happens offline)
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='font-family:IBM Plex Mono;font-size:13px;color:#58a6ff;
                letter-spacing:0.08em;padding:8px 0 16px'>
        📡 LSTM BREAKOUT<br>
        <span style='color:#4a5568;font-size:10px'>INTRADAY 15-MIN FORECAST</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**── SAVED WEIGHTS**")
    trained = saved_models()
    for d in ["bullish", "bearish"]:
        icon = "🟢" if d in trained else "⚪"
        meta_str = ""
        if d in trained:
            mpath = Path(f"{SAVE_DIR}/{d}_lstm.pt")
            mtime = pd.Timestamp(mpath.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")
            meta_str = f"<br><span style='color:#3a4452;font-size:9px;padding-left:18px'>{mtime}</span>"
        st.markdown(
            f"<span style='font-family:IBM Plex Mono;font-size:11px'>{icon} {d}_lstm.pt</span>{meta_str}",
            unsafe_allow_html=True
        )

    if not trained:
        st.markdown(
            "<div style='font-family:IBM Plex Mono;font-size:10px;color:#d29922;padding:8px 0'>"
            "No weights found.<br>Run: python lstm_train_historical.py</div>",
            unsafe_allow_html=True
        )

    st.markdown("**── DEVICE**")
    st.markdown(f"""
    <div style='font-family:IBM Plex Mono;font-size:11px;color:#4a5568;padding:4px 0'>
        <span class='dot-{"green" if DEVICE!="cpu" else "yellow"}'></span>
        {DEVICE.upper()}
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN — single-ticker next-day forecast
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class='main-title'>
    LSTM BREAKOUT PREDICTOR
    <span>●</span>
    <span style='font-size:13px;color:#4a5568'>15-min intraday forecast · conditioned on the opening bar</span>
</div>
""", unsafe_allow_html=True)

avail = saved_models()

# ── Input row
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    ticker_in = st.text_input("Ticker symbol", placeholder="NVDA").strip().upper()
with c2:
    override = st.selectbox("Model", ["Auto", "Bullish", "Bearish"])
with c3:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    run = st.button("▶ Predict", type="primary", use_container_width=True,
                    disabled=not avail)

if not avail:
    st.error(
        "No trained models found in ./models. "
        "Train first with:  python lstm_train_historical.py"
    )

# ── Run prediction
if run:
    if not ticker_in:
        st.warning("Enter a ticker symbol first.")
    else:
        with st.spinner(f"Forecasting {ticker_in} …"):
            try:
                if override == "Auto":
                    direction, score = detect_direction(ticker_in)
                else:
                    direction = override.lower()
                    _, score = detect_direction(ticker_in)

                if direction is None:
                    st.error(f"No market data returned for {ticker_in}.")
                else:
                    used, note = direction, None
                    if used not in avail:
                        used = avail[0]
                        note = (f"{direction.capitalize()} weights not on disk — "
                                f"used {used} model instead.")
                    df = predict_next_day_shared(ticker_in, used, DEVICE)
                    if df is None or df.empty:
                        st.error(f"Not enough recent 15-min data to forecast {ticker_in}.")
                        st.session_state.forecast = None
                    else:
                        st.session_state.forecast = {
                            "ticker": ticker_in, "direction": used,
                            "score": score, "note": note, "df": df,
                        }
            except Exception as e:
                st.error(f"Forecast failed: {e}")
                st.session_state.forecast = None

# ── Render most recent forecast (persists across reruns, e.g. CSV download)
fc = st.session_state.forecast
if fc is not None:
    df       = fc["df"]
    ticker   = fc["ticker"]
    dirn     = fc["direction"]
    score    = fc["score"]

    if fc["note"]:
        st.warning(fc["note"])

    # ── Target-day status banner (which day, opening bar, realized vs predicted)
    tgt       = df.attrs.get("target_day")
    open_px   = df.attrs.get("open_close")
    n_real    = df.attrs.get("n_realized", 0)
    n_total   = len(df)
    complete  = df.attrs.get("complete", False)
    tgt_str   = pd.Timestamp(tgt).strftime("%a %b %d") if tgt is not None else "—"
    if complete:
        status = (f"<span class='dot-yellow'></span>{tgt_str} is complete — "
                  f"showing all {n_total} bars as actuals vs the model's forecast. "
                  f"The next day hasn't opened yet, so no forward forecast is shown.")
    elif n_real > 0:
        status = (f"<span class='dot-green'></span>Forecasting the rest of {tgt_str} — "
                  f"{n_real} bar(s) already traded (actual), {n_total - n_real} predicted.")
    else:
        status = (f"<span class='dot-green'></span>Forecasting {tgt_str} from the opening "
                  f"15-min bar — full session predicted ({n_total} bars).")
    st.markdown(
        f"<div style='font-family:IBM Plex Mono;font-size:12px;color:#8b949e;"
        f"padding:4px 0 10px'>{status}</div>", unsafe_allow_html=True)

    p0     = df["predicted_mid"].iloc[0]
    p1     = df["predicted_mid"].iloc[-1]
    ph     = df["predicted_high"].max()
    pl     = df["predicted_low"].min()
    move   = (p1 / p0 - 1) * 100
    spread = ((df["predicted_high"] - df["predicted_low"]) / df["predicted_low"] * 100).mean()
    open_str = f"${open_px:.2f}" if open_px is not None else "—"

    dir_color = "#39d353" if dirn == "bullish" else "#f85149"
    score_str = f"{score:+d}" if score is not None else "—"
    cov       = holdout_coverage(dirn)
    cov_str   = f"{cov:.0%}" if cov is not None else "—"

    st.markdown(f"""
    <div class='metric-row'>
        <div class='metric-card'>
            <div class='label'>Model Used</div>
            <div class='value' style='color:{dir_color};text-transform:capitalize'>{dirn}</div>
            <div class='sub'>daily score {score_str}</div>
        </div>
        <div class='metric-card'>
            <div class='label'>Opening Bar</div>
            <div class='value neutral'>{open_str}</div>
            <div class='sub'>first 15-min close</div>
        </div>
        <div class='metric-card'>
            <div class='label'>Pred Close</div>
            <div class='value {pct_class(move)}'>${p1:.2f}</div>
            <div class='sub'>{move:+.2f}% rest-of-day</div>
        </div>
        <div class='metric-card'>
            <div class='label'>Pred High / Low</div>
            <div class='value'><span style='color:#39d353'>${ph:.2f}</span>
            &nbsp;/&nbsp;<span style='color:#f85149'>${pl:.2f}</span></div>
            <div class='sub'>range ${ph-pl:.2f}</div>
        </div>
        <div class='metric-card'>
            <div class='label'>Avg H/L Spread</div>
            <div class='value neutral'>{spread:.3f}%</div>
            <div class='sub'>avg 15-min range</div>
        </div>
        <div class='metric-card'>
            <div class='label'>Holdout Coverage</div>
            <div class='value neutral'>{cov_str}</div>
            <div class='sub'>val high/low hit rate</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.plotly_chart(build_prediction_chart(ticker, df), use_container_width=True)

    # ── Raw data + CSV download
    with st.expander("Raw forecast data"):
        display_df = df[["predicted_high", "predicted_low", "predicted_mid",
                         "actual_high", "actual_low",
                         "high_return", "low_return"]].copy()
        display_df["high_return_%"] = display_df["high_return"] * 100
        display_df["low_return_%"]  = display_df["low_return"] * 100
        display_df = display_df.drop(columns=["high_return", "low_return"])
        st.dataframe(display_df.style.format({
            "predicted_high": "${:.2f}",
            "predicted_low":  "${:.2f}",
            "predicted_mid":  "${:.2f}",
            "actual_high":    "${:.2f}",
            "actual_low":     "${:.2f}",
            "high_return_%":  "{:+.4f}%",
            "low_return_%":   "{:+.4f}%",
        }), use_container_width=True)

        st.download_button(
            "⬇ Download CSV",
            df.to_csv(),
            file_name=f"{ticker}_{dirn}_forecast.csv",
            mime="text/csv",
        )
