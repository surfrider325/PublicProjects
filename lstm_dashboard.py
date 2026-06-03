"""
lstm_dashboard.py
=================
Streamlit UI for the LSTM Breakout Predictor.
Place this file alongside lstm_breakout_predictor.py and run:

    streamlit run lstm_dashboard.py

Features:
  • Run breakout screener to find candidate stocks
  • Train per-ticker LSTM models with live progress
  • Predict next trading day minute-by-minute
  • Interactive price forecast chart with volume & return overlays
  • Side-by-side comparison of multiple tickers
  • Model metadata & training loss curves
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
import joblib
import torch

warnings.filterwarnings("ignore")

# ── Import core pipeline functions
from lstm_breakout_predictor import (
    BREAKOUT_APP_AVAILABLE,
    FEATURE_COLS,
    StockLSTM,
    compute_features,
    get_breakout_tickers,
    predict_next_day,
    train_model,
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

/* Log output */
.log-box {
    background: #090b0e;
    border: 1px solid #1e2530;
    border-radius: 6px;
    padding: 12px 16px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #39d353;
    max-height: 200px;
    overflow-y: auto;
    white-space: pre-wrap;
}

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

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: transparent; border-bottom: 1px solid #1e2530; gap: 0; }
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #4a5568;
    background: transparent;
    border: none;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] { color: #58a6ff !important; border-bottom: 2px solid #58a6ff !important; }

/* Dataframe */
.stDataFrame { border: 1px solid #1e2530 !important; border-radius: 6px; }

/* Progress bar */
.stProgress > div > div { background-color: #39d353; }

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
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

for key, default in [
    ("breakout_tickers", []),
    ("trained_tickers",  []),
    ("predictions",      {}),
    ("train_logs",       {}),
    ("train_history",    {}),
    ("screener_ran",     False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def saved_models() -> list[str]:
    return sorted([p.stem.replace("_lstm", "") for p in Path(SAVE_DIR).glob("*_lstm.pt")])


def saved_predictions() -> list[str]:
    return sorted([p.stem.replace("_predictions", "") for p in Path(SAVE_DIR).glob("*_predictions.csv")])


def load_prediction(ticker: str) -> pd.DataFrame | None:
    path = f"{SAVE_DIR}/{ticker}_predictions.csv"
    if not Path(path).exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def move_color(pct: float) -> str:
    if pct > 1.5:  return "#39d353"
    if pct > 0.3:  return "#56d364"
    if pct > 0:    return "#3fb950"
    if pct > -0.3: return "#f85149"
    return "#da3633"


def pct_class(pct: float) -> str:
    return "up" if pct >= 0 else "down"


# ══════════════════════════════════════════════════════════════════════════════
# CHART BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_prediction_chart(ticker: str, df: pd.DataFrame) -> go.Figure:
    """Full-featured prediction chart: price curve + return heatband + volume proxy."""
    times  = df.index
    prices = df["predicted_price"].values
    rets   = df["predicted_return"].values * 100  # in %

    open_p   = prices[0]
    close_p  = prices[-1]
    high_p   = prices.max()
    low_p    = prices.min()
    move_pct = (close_p / open_p - 1) * 100

    # Color-coded return bars
    bar_colors = [move_color(r) for r in rets]

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.72, 0.28],
        shared_xaxes=True,
        vertical_spacing=0.03,
    )

    # ── Price line with gradient fill
    fig.add_trace(go.Scatter(
        x=times, y=prices,
        mode="lines",
        name="Predicted Price",
        line=dict(color="#58a6ff", width=1.8),
        fill="tozeroy",
        fillcolor="rgba(88,166,255,0.04)",
        hovertemplate="<b>%{x|%H:%M}</b><br>$%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    # ── High / Low bands
    fig.add_hline(y=high_p, line=dict(color="#39d353", width=0.6, dash="dot"),
                  annotation_text=f"H ${high_p:.2f}", annotation_position="right",
                  annotation_font=dict(color="#39d353", size=10, family="IBM Plex Mono"),
                  row=1, col=1)
    fig.add_hline(y=low_p,  line=dict(color="#f85149", width=0.6, dash="dot"),
                  annotation_text=f"L ${low_p:.2f}",  annotation_position="right",
                  annotation_font=dict(color="#f85149", size=10, family="IBM Plex Mono"),
                  row=1, col=1)
    fig.add_hline(y=open_p, line=dict(color="#8b949e", width=0.8, dash="dash"),
                  row=1, col=1)

    # ── Return bars (row 2)
    fig.add_trace(go.Bar(
        x=times, y=rets,
        name="1-min Return %",
        marker_color=bar_colors,
        marker_line_width=0,
        opacity=0.85,
        hovertemplate="<b>%{x|%H:%M}</b><br>%{y:.4f}%<extra></extra>",
    ), row=2, col=1)

    # ── Zero line for returns
    fig.add_hline(y=0, line=dict(color="#30363d", width=1), row=2, col=1)

    title_color = "#39d353" if move_pct >= 0 else "#f85149"
    sign = "▲" if move_pct >= 0 else "▼"

    fig.update_layout(
        title=dict(
            text=f"<span style='font-family:IBM Plex Mono'>{ticker} — Next Day Forecast  "
                 f"<span style='color:{title_color}'>{sign} {abs(move_pct):.2f}%</span></span>",
            font=dict(size=14, color="#e6edf3"),
            x=0,
        ),
        paper_bgcolor="#0d0f13",
        plot_bgcolor="#0d0f13",
        font=dict(family="IBM Plex Mono", color="#8b949e", size=11),
        legend=dict(
            orientation="h", y=1.02, x=1, xanchor="right",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10),
        ),
        margin=dict(l=10, r=80, t=50, b=10),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#161b22",
            bordercolor="#30363d",
            font=dict(family="IBM Plex Mono", size=11),
        ),
        xaxis2=dict(
            showgrid=False, zeroline=False,
            tickformat="%H:%M",
            tickfont=dict(size=10),
            color="#4a5568",
        ),
    )

    for row in [1, 2]:
        fig.update_xaxes(
            showgrid=False, zeroline=False,
            tickformat="%H:%M",
            tickfont=dict(size=10, family="IBM Plex Mono"),
            color="#4a5568",
            row=row, col=1,
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor="#1a1f26",
            gridwidth=1,
            zeroline=False,
            tickfont=dict(size=10, family="IBM Plex Mono"),
            color="#4a5568",
            row=row, col=1,
        )

    return fig


def build_comparison_chart(predictions: dict[str, pd.DataFrame]) -> go.Figure:
    """Normalised price comparison: all tickers indexed to 100 at open."""
    PALETTE = ["#58a6ff", "#39d353", "#d29922", "#f85149", "#bc8cff", "#56d364", "#ff7b72"]

    fig = go.Figure()
    for i, (ticker, df) in enumerate(predictions.items()):
        prices  = df["predicted_price"].values
        norm    = (prices / prices[0]) * 100
        color   = PALETTE[i % len(PALETTE)]
        move    = (prices[-1] / prices[0] - 1) * 100
        sign    = "▲" if move >= 0 else "▼"
        fig.add_trace(go.Scatter(
            x=df.index,
            y=norm,
            mode="lines",
            name=f"{ticker} {sign}{abs(move):.2f}%",
            line=dict(color=color, width=1.6),
            hovertemplate=f"<b>{ticker}</b> %{{x|%H:%M}}<br>Norm: %{{y:.2f}}<extra></extra>",
        ))

    fig.add_hline(y=100, line=dict(color="#30363d", width=1, dash="dash"))

    fig.update_layout(
        title=dict(
            text="<span style='font-family:IBM Plex Mono'>Normalised Forecast Comparison (Open = 100)</span>",
            font=dict(size=13, color="#e6edf3"), x=0,
        ),
        paper_bgcolor="#0d0f13",
        plot_bgcolor="#0d0f13",
        font=dict(family="IBM Plex Mono", color="#8b949e", size=11),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="#1e2530",
            borderwidth=1,
            font=dict(size=10),
        ),
        margin=dict(l=10, r=20, t=50, b=10),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#161b22", bordercolor="#30363d",
                        font=dict(family="IBM Plex Mono", size=11)),
        xaxis=dict(showgrid=False, tickformat="%H:%M", color="#4a5568",
                   tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="#1a1f26", color="#4a5568",
                   tickfont=dict(size=10), ticksuffix=""),
    )
    return fig


def build_loss_chart(history: dict) -> go.Figure:
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"],
                             mode="lines", name="Train",
                             line=dict(color="#58a6ff", width=1.5)))
    fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"],
                             mode="lines", name="Val",
                             line=dict(color="#f0883e", width=1.5)))
    fig.update_layout(
        paper_bgcolor="#090b0e", plot_bgcolor="#090b0e",
        font=dict(family="IBM Plex Mono", color="#586069", size=10),
        margin=dict(l=0, r=0, t=8, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        xaxis=dict(showgrid=False, color="#4a5568", title="Epoch"),
        yaxis=dict(showgrid=True, gridcolor="#1a1f26", color="#4a5568", title="Huber Loss"),
        height=180,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='font-family:IBM Plex Mono;font-size:13px;color:#58a6ff;
                letter-spacing:0.08em;padding:8px 0 16px'>
        📡 LSTM BREAKOUT<br>
        <span style='color:#4a5568;font-size:10px'>INTRADAY PREDICTOR</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**── SCREENER**")
    min_score  = st.slider("Min Breakout Score", 40, 90, 55, 5)
    max_scan   = st.slider("Tickers to Scan",    10, 200, 50, 10)

    st.markdown("**── MODEL**")
    epochs    = st.slider("Training Epochs",  5, 50, 20, 5)
    seq_len   = st.slider("Sequence Length",  30, 120, 60, 10)

    st.markdown("**── DEVICE**")
    st.markdown(f"""
    <div style='font-family:IBM Plex Mono;font-size:11px;color:#4a5568;padding:4px 0'>
        <span class='dot-{"green" if DEVICE!="cpu" else "yellow"}'></span>
        {DEVICE.upper()}
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Quick-add manual tickers
    st.markdown("**── MANUAL TICKERS**")
    manual_input = st.text_input("Add tickers (comma-separated)", placeholder="NVDA, AMD, TSLA")

    if st.button("＋ Add to Queue"):
        if manual_input.strip():
            new = [t.strip().upper() for t in manual_input.split(",") if t.strip()]
            existing = set(st.session_state.breakout_tickers)
            added = [t for t in new if t not in existing]
            st.session_state.breakout_tickers.extend(added)
            st.success(f"Added: {', '.join(added)}")

    st.divider()

    # Model inventory
    models = saved_models()
    if models:
        st.markdown(f"**── SAVED MODELS** ({len(models)})")
        for m in models[:10]:
            pct = "—"
            pred = load_prediction(m)
            if pred is not None:
                p0, p1 = pred["predicted_price"].iloc[0], pred["predicted_price"].iloc[-1]
                v = (p1/p0-1)*100
                col = "#39d353" if v >= 0 else "#f85149"
                sign = "▲" if v >= 0 else "▼"
                pct = f"<span style='color:{col}'>{sign}{abs(v):.1f}%</span>"
            st.markdown(
                f"<span style='font-family:IBM Plex Mono;font-size:11px;"
                f"color:#58a6ff'>{m}</span> "
                f"<span style='font-size:10px;color:#4a5568'>{pct}</span>",
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class='main-title'>
    LSTM BREAKOUT PREDICTOR
    <span>●</span>
    <span style='font-size:13px;color:#4a5568'>minute-by-minute next-day forecast</span>
</div>
""", unsafe_allow_html=True)

tab_run, tab_charts, tab_compare, tab_models = st.tabs([
    "⚡  RUN PIPELINE",
    "📈  FORECASTS",
    "⚖   COMPARE",
    "🧠  MODELS",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RUN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

with tab_run:
    col_screen, col_train, col_predict = st.columns(3)

    # ── STEP 1: Screen
    with col_screen:
        st.markdown("<div class='section-header'>01 · SCREEN</div>", unsafe_allow_html=True)
        st.caption("Find breakout candidates using your detector")

        if not BREAKOUT_APP_AVAILABLE:
            st.warning("breakout_detector.py not found. Use Manual Tickers in sidebar.")
        else:
            if st.button("▶ Run Screener", type="primary", use_container_width=True):
                with st.spinner("Scanning…"):
                    prog = st.progress(0)
                    tickers_found = []

                    try:
                        from breakout_detector import analyze_ticker, SP500_TICKERS
                        scan_list = SP500_TICKERS[:max_scan]
                        for i, ticker in enumerate(scan_list, 1):
                            try:
                                result = analyze_ticker(ticker, lookback_weeks=20)
                                if result and result["composite"]["score"] >= min_score:
                                    tickers_found.append(ticker)
                            except Exception:
                                pass
                            prog.progress(i / len(scan_list))
                            time.sleep(0.03)
                        prog.empty()
                        st.session_state.breakout_tickers = tickers_found
                        st.session_state.screener_ran = True
                    except Exception as e:
                        st.error(str(e))

        if st.session_state.breakout_tickers:
            st.markdown(f"<div class='section-header'>{len(st.session_state.breakout_tickers)} candidates</div>",
                        unsafe_allow_html=True)
            badges = " ".join(
                f"<span class='ticker-badge'>{t}</span>"
                for t in st.session_state.breakout_tickers
            )
            st.markdown(badges, unsafe_allow_html=True)

    # ── STEP 2: Train
    with col_train:
        st.markdown("<div class='section-header'>02 · TRAIN</div>", unsafe_allow_html=True)
        st.caption("Train an LSTM model per breakout ticker")

        if st.button("▶ Train Models", type="primary", use_container_width=True,
                     disabled=len(st.session_state.breakout_tickers) == 0):
            tickers = st.session_state.breakout_tickers
            prog_bar    = st.progress(0)
            status_text = st.empty()
            log_area    = st.empty()
            log_lines   = []

            for i, ticker in enumerate(tickers):
                status_text.markdown(
                    f"<span style='font-family:IBM Plex Mono;font-size:12px;color:#58a6ff'>"
                    f"Training {ticker} ({i+1}/{len(tickers)})…</span>",
                    unsafe_allow_html=True
                )
                log_lines.append(f"[{ticker}] starting …")
                log_area.markdown(
                    f"<div class='log-box'>{'<br>'.join(log_lines[-12:])}</div>",
                    unsafe_allow_html=True
                )

                result = train_model(
                    ticker,
                    seq_len=seq_len,
                    epochs=epochs,
                    save_dir=SAVE_DIR,
                    device=DEVICE,
                )

                if result:
                    st.session_state.trained_tickers.append(ticker)
                    st.session_state.train_history[ticker] = result["history"]
                    log_lines.append(
                        f"[{ticker}] ✓  val_loss={result['best_val_loss']:.6f}"
                    )
                else:
                    log_lines.append(f"[{ticker}] ✗  skipped (insufficient data)")

                prog_bar.progress((i + 1) / len(tickers))
                log_area.markdown(
                    f"<div class='log-box'>{'<br>'.join(log_lines[-12:])}</div>",
                    unsafe_allow_html=True
                )

            status_text.markdown(
                "<span style='font-family:IBM Plex Mono;font-size:12px;color:#39d353'>"
                "✓ Training complete</span>",
                unsafe_allow_html=True
            )

    # ── STEP 3: Predict
    with col_predict:
        st.markdown("<div class='section-header'>03 · PREDICT</div>", unsafe_allow_html=True)
        st.caption("Generate next-day minute-by-minute forecasts")

        trained = saved_models()
        if st.button("▶ Run Predictions", type="primary", use_container_width=True,
                     disabled=len(trained) == 0):
            prog_bar    = st.progress(0)
            status_text = st.empty()
            results     = {}

            for i, ticker in enumerate(trained):
                status_text.markdown(
                    f"<span style='font-family:IBM Plex Mono;font-size:12px;color:#58a6ff'>"
                    f"Predicting {ticker}…</span>",
                    unsafe_allow_html=True
                )
                preds = predict_next_day(
                    ticker, seq_len=seq_len, save_dir=SAVE_DIR, device=DEVICE
                )
                if preds is not None:
                    results[ticker] = preds
                    preds.to_csv(f"{SAVE_DIR}/{ticker}_predictions.csv")

                prog_bar.progress((i + 1) / len(trained))

            st.session_state.predictions = results
            status_text.markdown(
                f"<span style='font-family:IBM Plex Mono;font-size:12px;color:#39d353'>"
                f"✓ {len(results)} forecasts ready</span>",
                unsafe_allow_html=True
            )

    # ── Summary table
    preds_available = saved_predictions()
    if preds_available:
        st.markdown("<div class='section-header'>FORECAST SUMMARY</div>", unsafe_allow_html=True)

        rows = []
        for ticker in preds_available:
            df = load_prediction(ticker)
            if df is None:
                continue
            p_open  = df["predicted_price"].iloc[0]
            p_close = df["predicted_price"].iloc[-1]
            p_high  = df["predicted_price"].max()
            p_low   = df["predicted_price"].min()
            move    = (p_close / p_open - 1) * 100
            vol     = df["predicted_return"].std() * 100
            rows.append({
                "Ticker": ticker,
                "Pred Open":  f"${p_open:.2f}",
                "Pred Close": f"${p_close:.2f}",
                "Pred High":  f"${p_high:.2f}",
                "Pred Low":   f"${p_low:.2f}",
                "Move %":     f"{move:+.2f}%",
                "Vol (σ ret)":f"{vol:.4f}%",
            })

        if rows:
            summary_df = pd.DataFrame(rows)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — INDIVIDUAL FORECASTS
# ══════════════════════════════════════════════════════════════════════════════

with tab_charts:
    preds_available = saved_predictions()

    if not preds_available:
        st.info("No predictions yet. Run the pipeline in the **Run** tab.")
    else:
        selected = st.selectbox(
            "Select ticker",
            preds_available,
            key="chart_ticker",
        )

        df = load_prediction(selected)
        if df is not None:
            p0, p1 = df["predicted_price"].iloc[0], df["predicted_price"].iloc[-1]
            ph, pl = df["predicted_price"].max(),   df["predicted_price"].min()
            move   = (p1 / p0 - 1) * 100
            vol    = df["predicted_return"].std() * 100
            max_up  = df["predicted_return"].max() * 100
            max_dn  = df["predicted_return"].min() * 100

            # ── Metric cards
            st.markdown(f"""
            <div class='metric-row'>
                <div class='metric-card'>
                    <div class='label'>Pred Open</div>
                    <div class='value neutral'>${p0:.2f}</div>
                </div>
                <div class='metric-card'>
                    <div class='label'>Pred Close</div>
                    <div class='value {pct_class(move)}'>${p1:.2f}</div>
                    <div class='sub'>{move:+.2f}% day move</div>
                </div>
                <div class='metric-card'>
                    <div class='label'>Pred High / Low</div>
                    <div class='value'><span style='color:#39d353'>${ph:.2f}</span>
                    &nbsp;/&nbsp;<span style='color:#f85149'>${pl:.2f}</span></div>
                    <div class='sub'>range ${ph-pl:.2f}</div>
                </div>
                <div class='metric-card'>
                    <div class='label'>Return Volatility</div>
                    <div class='value neutral'>{vol:.4f}%</div>
                    <div class='sub'>σ of 1-min returns</div>
                </div>
                <div class='metric-card'>
                    <div class='label'>Best / Worst Min</div>
                    <div class='value'><span style='color:#39d353'>{max_up:+.4f}%</span>
                    &nbsp;/&nbsp;<span style='color:#f85149'>{max_dn:+.4f}%</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.plotly_chart(build_prediction_chart(selected, df), use_container_width=True)

            # ── Raw data expander
            with st.expander("Raw forecast data"):
                display_df = df.copy()
                display_df["predicted_return_%"] = display_df["predicted_return"] * 100
                display_df = display_df.drop(columns=["predicted_return"])
                st.dataframe(display_df.style.format({
                    "predicted_price":    "${:.4f}",
                    "predicted_return_%": "{:+.5f}%",
                }), use_container_width=True)

                csv = display_df.to_csv()
                st.download_button(
                    "⬇ Download CSV",
                    csv,
                    file_name=f"{selected}_forecast.csv",
                    mime="text/csv",
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — COMPARE
# ══════════════════════════════════════════════════════════════════════════════

with tab_compare:
    preds_available = saved_predictions()

    if len(preds_available) < 2:
        st.info("Need at least 2 tickers with predictions to compare.")
    else:
        selected_multi = st.multiselect(
            "Select tickers to compare",
            preds_available,
            default=preds_available[:min(5, len(preds_available))],
        )

        if len(selected_multi) >= 2:
            compare_preds = {t: load_prediction(t) for t in selected_multi
                            if load_prediction(t) is not None}

            st.plotly_chart(build_comparison_chart(compare_preds), use_container_width=True)

            # ── Ranked summary
            st.markdown("<div class='section-header'>RANKED BY PREDICTED MOVE</div>",
                        unsafe_allow_html=True)

            rank_rows = []
            for ticker, df in compare_preds.items():
                p0, p1 = df["predicted_price"].iloc[0], df["predicted_price"].iloc[-1]
                move   = (p1 / p0 - 1) * 100
                vol    = df["predicted_return"].std() * 100
                sharpe = move / (vol * np.sqrt(390)) if vol > 0 else 0
                rank_rows.append({
                    "Ticker":      ticker,
                    "Move %":      move,
                    "Volatility":  vol,
                    "Intraday Sharpe": sharpe,
                })

            rank_df = (pd.DataFrame(rank_rows)
                        .sort_values("Move %", ascending=False)
                        .reset_index(drop=True))
            rank_df.index += 1

            def style_move(v):
                c = "#39d353" if v >= 0 else "#f85149"
                return f"color: {c}; font-family: IBM Plex Mono"

            st.dataframe(
                rank_df.style
                    .format({"Move %": "{:+.2f}%", "Volatility": "{:.4f}%", "Intraday Sharpe": "{:.3f}"})
                    .applymap(style_move, subset=["Move %"]),
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL DETAILS
# ══════════════════════════════════════════════════════════════════════════════

with tab_models:
    models_list = saved_models()

    if not models_list:
        st.info("No trained models found. Run Step 2 (Train) first.")
    else:
        selected_model = st.selectbox("Select model", models_list, key="model_select")

        col_info, col_loss = st.columns([1, 2])

        with col_info:
            st.markdown("<div class='section-header'>MODEL INFO</div>", unsafe_allow_html=True)

            # File metadata
            model_path  = Path(f"{SAVE_DIR}/{selected_model}_lstm.pt")
            scaler_path = Path(f"{SAVE_DIR}/{selected_model}_scaler.pkl")
            pred_path   = Path(f"{SAVE_DIR}/{selected_model}_predictions.csv")

            size_kb = model_path.stat().st_size / 1024 if model_path.exists() else 0
            mtime   = pd.Timestamp(model_path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M") \
                      if model_path.exists() else "—"

            st.markdown(f"""
            <div style='font-family:IBM Plex Mono;font-size:11px;color:#586069;line-height:2'>
                <b style='color:#8b949e'>Ticker</b>     {selected_model}<br>
                <b style='color:#8b949e'>Architecture</b> 2-layer LSTM<br>
                <b style='color:#8b949e'>Hidden Size</b>  128<br>
                <b style='color:#8b949e'>Features</b>     {len(FEATURE_COLS)}<br>
                <b style='color:#8b949e'>Model File</b>   {size_kb:.1f} KB<br>
                <b style='color:#8b949e'>Trained</b>      {mtime}<br>
                <b style='color:#8b949e'>Scaler</b>       {"✓" if scaler_path.exists() else "✗"}<br>
                <b style='color:#8b949e'>Predictions</b>  {"✓" if pred_path.exists() else "✗"}<br>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div class='section-header' style='margin-top:20px'>FEATURES</div>",
                        unsafe_allow_html=True)
            for f in FEATURE_COLS:
                st.markdown(
                    f"<span style='font-family:IBM Plex Mono;font-size:10px;"
                    f"color:#4a5568'>▸ {f}</span>",
                    unsafe_allow_html=True
                )

        with col_loss:
            st.markdown("<div class='section-header'>TRAINING LOSS</div>", unsafe_allow_html=True)
            if selected_model in st.session_state.train_history:
                hist = st.session_state.train_history[selected_model]
                st.plotly_chart(build_loss_chart(hist), use_container_width=True)

                best_val = min(hist["val_loss"])
                final_train = hist["train_loss"][-1]
                st.markdown(f"""
                <div style='font-family:IBM Plex Mono;font-size:11px;color:#586069'>
                    Best val loss: <span style='color:#58a6ff'>{best_val:.6f}</span>
                    &nbsp;|&nbsp;
                    Final train loss: <span style='color:#f0883e'>{final_train:.6f}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Loss history available after training in this session. "
                        "Re-train the model to see the curve.")

            # Scaler stats
            if scaler_path.exists():
                st.markdown("<div class='section-header' style='margin-top:20px'>FEATURE SCALER RANGES</div>",
                            unsafe_allow_html=True)
                try:
                    scaler = joblib.load(scaler_path)
                    scaler_df = pd.DataFrame({
                        "Feature":  FEATURE_COLS,
                        "Min":      scaler.data_min_,
                        "Max":      scaler.data_max_,
                        "Range":    scaler.data_max_ - scaler.data_min_,
                    })
                    st.dataframe(
                        scaler_df.style.format({"Min": "{:.4f}", "Max": "{:.4f}", "Range": "{:.4f}"}),
                        use_container_width=True,
                        height=280,
                    )
                except Exception:
                    st.caption("Could not load scaler details.")
