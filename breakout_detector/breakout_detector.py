"""
S&P 500 Breakout Detector — Streamlit Dashboard
================================================
Run with:  streamlit run breakout_detector.py

Dependencies:
    pip install streamlit yfinance pandas numpy plotly pandas-ta
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta
import time

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="S&P 500 Breakout Detector",
    page_icon="📈",
    layout="wide",
)

# ─────────────────────────────────────────────
# S&P 500 TICKERS (subset – extend as needed)
# ─────────────────────────────────────────────
SP500_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","BRK-B","JPM","V",
    "UNH","XOM","LLY","MA","HD","JNJ","AVGO","PG","MRK","COST",
    "ABBV","CVX","KO","ADBE","CRM","MCD","NFLX","AMD","PEP","ACN",
    "WMT","BAC","TMO","CSCO","ABT","DHR","ORCL","LIN","NKE","NEE",
    "DIS","TXN","PM","QCOM","WFC","RTX","MS","GE","AMGN","SPGI",
    "CAT","HON","INTU","UPS","BKNG","BLK","SBUX","NOW","GS","PLD",
    "ISRG","SYK","ELV","MDT","GILD","REGN","ZTS","VRTX","MO","CI",
    "T","AXP","LOW","DE","SCHW","TJX","ADP","IBM","ETN","EOG",
    "SLB","CME","BSX","ADI","LRCX","KLAC","SNPS","CDNS","ITW","MMC",
    "PH","HUM","COP","MDLZ","NXPI","PANW","SO","DUK","ICE","BDX",
]

# ─────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        return df
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.index = pd.to_datetime(df.index)
    return df.dropna()


# ─────────────────────────────────────────────
# SIGNAL 1 — PRICE BREAKOUT / BREAKDOWN
# ─────────────────────────────────────────────
def price_signal(df: pd.DataFrame, lookback: int = 52) -> dict:
    """
    Bullish : close crosses above N-period resistance high.
    Bearish : close crosses below N-period support low.
    """
    if len(df) < lookback + 1:
        return {"bullish": False, "bearish": False, "bull_score": 0, "bear_score": 0,
                "resistance": np.nan, "support": np.nan, "pct_vs_resist": 0, "pct_vs_support": 0}

    window      = df["Close"].iloc[-(lookback + 1):-1]
    prior_high  = float(window.max())
    prior_low   = float(window.min())
    current     = float(df["Close"].iloc[-1])
    prev        = float(df["Close"].iloc[-2])

    bullish = (current > prior_high) and (prev <= prior_high)
    bearish = (current < prior_low)  and (prev >= prior_low)

    rng        = max(prior_high - prior_low, 1e-9)
    bull_score = max(0, min(100, round((current - prior_low) / rng * 100)))
    bear_score = max(0, min(100, round((prior_high - current) / rng * 100)))

    return {
        "bullish":        bullish,
        "bearish":        bearish,
        "bull_score":     bull_score,
        "bear_score":     bear_score,
        "resistance":     round(prior_high, 2),
        "support":        round(prior_low, 2),
        "pct_vs_resist":  round((current / prior_high - 1) * 100, 2),
        "pct_vs_support": round((current / prior_low  - 1) * 100, 2),
    }


# ─────────────────────────────────────────────
# SIGNAL 2 — VOLUME SPIKE (directional)
# ─────────────────────────────────────────────
def volume_signal(df: pd.DataFrame, vol_window: int = 20, threshold: float = 2.0) -> dict:
    """
    Bullish : volume spike on an up day.
    Bearish : volume spike on a down day (distribution / selling pressure).
    """
    if len(df) < vol_window + 1:
        return {"bullish": False, "bearish": False, "bull_score": 0, "bear_score": 0, "vol_ratio": 0}

    avg_vol   = float(df["Volume"].iloc[-(vol_window + 1):-1].mean())
    today_vol = float(df["Volume"].iloc[-1])
    vol_ratio = today_vol / max(avg_vol, 1)

    price_up  = float(df["Close"].iloc[-1]) > float(df["Close"].iloc[-2])

    bullish = (vol_ratio >= threshold) and price_up
    bearish = (vol_ratio >= threshold) and (not price_up)

    score = min(100, round((vol_ratio / 4) * 100))

    return {
        "bullish":    bullish,
        "bearish":    bearish,
        "bull_score": max(0, min(100, score if bullish else score - 20)),
        "bear_score": max(0, min(100, score if bearish else score - 20)),
        "vol_ratio":  round(vol_ratio, 2),
    }


# ─────────────────────────────────────────────
# SIGNAL 3 — BOLLINGER BAND SQUEEZE
# ─────────────────────────────────────────────
def bb_signal(df: pd.DataFrame, bb_window: int = 20) -> dict:
    """
    Bullish : price breaks ABOVE upper band after a squeeze.
    Bearish : price breaks BELOW lower band after a squeeze.
    """
    if len(df) < 60:
        return {"bullish": False, "bearish": False, "bull_score": 0, "bear_score": 0,
                "bb_width": np.nan, "squeeze": False, "upper_band": np.nan, "lower_band": np.nan}

    closes = df["Close"]
    mid    = closes.rolling(bb_window).mean()
    std    = closes.rolling(bb_window).std()
    upper  = mid + 2 * std
    lower  = mid - 2 * std
    bw     = (upper - lower) / mid * 100

    bw_now      = float(bw.iloc[-1])
    bw_60_max   = float(bw.iloc[-60:].max())
    was_squeeze = float(bw.iloc[-5:-1].min()) < float(bw.iloc[-60:-5].mean()) * 0.75
    squeeze     = bw_now == float(bw.iloc[-60:].min())

    above_upper = float(closes.iloc[-1]) > float(upper.iloc[-1])
    below_lower = float(closes.iloc[-1]) < float(lower.iloc[-1])

    bullish = above_upper and was_squeeze
    bearish = below_lower and was_squeeze

    base_score = max(0, min(100, round((1 - bw_now / max(bw_60_max, 1e-9)) * 100)))
    bull_score = max(base_score, 85) if bullish else base_score
    bear_score = max(base_score, 85) if bearish else base_score

    return {
        "bullish":    bullish,
        "bearish":    bearish,
        "bull_score": bull_score,
        "bear_score": bear_score,
        "bb_width":   round(bw_now, 2),
        "squeeze":    squeeze,
        "upper_band": round(float(upper.iloc[-1]), 2),
        "lower_band": round(float(lower.iloc[-1]), 2),
    }


# ─────────────────────────────────────────────
# SIGNAL 4 — PATTERN (Bull / Bear Flag)
# ─────────────────────────────────────────────
def pattern_signal(df: pd.DataFrame, atr_window: int = 14, consol_window: int = 15) -> dict:
    """
    Bullish : consolidation + ATR expansion upward (bull flag).
    Bearish : consolidation + ATR expansion downward (bear flag).
    """
    if len(df) < atr_window + consol_window + 5:
        return {"bullish": False, "bearish": False, "bull_score": 0, "bear_score": 0,
                "atr": np.nan, "pattern": "none"}

    atr = ta.atr(df["High"], df["Low"], df["Close"], length=atr_window)
    if atr is None or atr.isna().all():
        return {"bullish": False, "bearish": False, "bull_score": 0, "bear_score": 0,
                "atr": np.nan, "pattern": "none"}

    consol_atr = float(atr.iloc[-(consol_window + 5):-5].mean())
    recent_atr = float(atr.iloc[-1])
    expansion  = recent_atr / max(consol_atr, 1e-9)

    consol_close = df["Close"].iloc[-(consol_window + 5):-5]
    price_range  = (consol_close.max() - consol_close.min()) / consol_close.mean() * 100
    tight        = price_range < 4.0

    direction_up = float(df["Close"].iloc[-1]) > float(df["Close"].iloc[-(consol_window + 5)])

    bullish = tight and (expansion >= 1.5) and direction_up
    bearish = tight and (expansion >= 1.5) and (not direction_up)

    if bullish:
        pattern = "bull flag"
    elif bearish:
        pattern = "bear flag"
    elif tight:
        pattern = "consolidating"
    else:
        pattern = "trending"

    base_score = min(100, round(expansion * 40)) if tight else max(0, min(50, round(50 - price_range * 5)))

    return {
        "bullish":    bullish,
        "bearish":    bearish,
        "bull_score": base_score,
        "bear_score": base_score,
        "atr":        round(recent_atr, 3),
        "expansion":  round(expansion, 2),
        "pattern":    pattern,
    }


# ─────────────────────────────────────────────
# COMPOSITE SCORE  (+ve = bullish, -ve = bearish)
# ─────────────────────────────────────────────
WEIGHTS = {"price": 0.35, "volume": 0.25, "bb": 0.20, "pattern": 0.20}

def composite_score(p, v, bb, pt) -> dict:
    bull = (p["bull_score"]  * WEIGHTS["price"]   +
            v["bull_score"]  * WEIGHTS["volume"]  +
            bb["bull_score"] * WEIGHTS["bb"]      +
            pt["bull_score"] * WEIGHTS["pattern"])

    bear = (p["bear_score"]  * WEIGHTS["price"]   +
            v["bear_score"]  * WEIGHTS["volume"]  +
            bb["bear_score"] * WEIGHTS["bb"]      +
            pt["bear_score"] * WEIGHTS["pattern"])

    bull_sigs = sum([p["bullish"], v["bullish"], bb["bullish"], pt["bullish"]])
    bear_sigs = sum([p["bearish"], v["bearish"], bb["bearish"], pt["bearish"]])
    net       = round(bull - bear)

    if net >= 75:
        label, color, direction = "🔥 Strong Breakout",    "#00cc88", "bullish"
    elif net >= 40:
        label, color, direction = "⚡ Emerging Breakout",  "#44bb88", "bullish"
    elif net >= 10:
        label, color, direction = "👀 Watch (Bullish)",    "#ffcc00", "bullish"
    elif net <= -75:
        label, color, direction = "🔻 Strong Breakdown",   "#ff4466", "bearish"
    elif net <= -40:
        label, color, direction = "⚠️ Emerging Breakdown", "#ff7744", "bearish"
    elif net <= -10:
        label, color, direction = "👀 Watch (Bearish)",    "#ffcc00", "bearish"
    else:
        label, color, direction = "😴 No Signal",          "#888888", "neutral"

    return {
        "net":        net,
        "bull_score": round(bull),
        "bear_score": round(bear),
        "label":      label,
        "color":      color,
        "direction":  direction,
        "bull_sigs":  bull_sigs,
        "bear_sigs":  bear_sigs,
    }


# ─────────────────────────────────────────────
# FULL ANALYSIS
# ─────────────────────────────────────────────
def analyze_ticker(ticker: str, lookback_weeks: int = 20) -> dict | None:
    df = fetch_data(ticker)
    if df is None or len(df) < 80:
        return None

    p   = price_signal(df, lookback=lookback_weeks * 5)
    v   = volume_signal(df)
    bb  = bb_signal(df)
    pt  = pattern_signal(df)
    cs  = composite_score(p, v, bb, pt)

    return {
        "ticker":     ticker,
        "price":      float(df["Close"].iloc[-1]),
        "change_pct": round((float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-2]) - 1) * 100, 2),
        "composite":  cs,
        "signals":    {"price": p, "volume": v, "bb": bb, "pattern": pt},
        "df":         df,
    }


# ─────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────
def build_chart(result: dict) -> go.Figure:
    df     = result["df"]
    ticker = result["ticker"]
    sigs   = result["signals"]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.03,
        subplot_titles=(f"{ticker} — Price + Bollinger Bands", "Volume", "ATR"),
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price",
        increasing_line_color="#00cc88", decreasing_line_color="#ff4466",
    ), row=1, col=1)

    mid   = df["Close"].rolling(20).mean()
    std   = df["Close"].rolling(20).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    for band, name, color in [
        (upper, "Upper BB", "rgba(100,180,255,0.3)"),
        (lower, "Lower BB", "rgba(100,180,255,0.3)"),
        (mid,   "Mid BB",   "rgba(100,180,255,0.8)"),
    ]:
        fig.add_trace(go.Scatter(x=df.index, y=band, name=name,
                                 line=dict(color=color, width=1), showlegend=False), row=1, col=1)

    resist  = sigs["price"].get("resistance")
    support = sigs["price"].get("support")
    if resist:
        fig.add_hline(y=resist, line=dict(color="#ffcc00", width=1.5, dash="dash"),
                      annotation_text=f"Resistance {resist}", row=1, col=1)
    if support:
        fig.add_hline(y=support, line=dict(color="#ff7744", width=1.5, dash="dash"),
                      annotation_text=f"Support {support}", row=1, col=1)

    last = df.index[-1]
    if sigs["price"]["bullish"]:
        fig.add_annotation(x=last, y=float(df["High"].iloc[-1]) * 1.01,
                           text="🚀 BREAKOUT", showarrow=True, arrowhead=2,
                           arrowcolor="#00cc88", font=dict(color="#00cc88", size=12), row=1, col=1)
    elif sigs["price"]["bearish"]:
        fig.add_annotation(x=last, y=float(df["Low"].iloc[-1]) * 0.99,
                           text="🔻 BREAKDOWN", showarrow=True, arrowhead=2,
                           arrowcolor="#ff4466", font=dict(color="#ff4466", size=12), row=1, col=1)

    colors  = ["#00cc88" if c >= o else "#ff4466" for c, o in zip(df["Close"], df["Open"])]
    avg_vol = df["Volume"].rolling(20).mean()
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=colors,
                         name="Volume", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=avg_vol, name="Avg Vol",
                             line=dict(color="#ffcc00", width=1.5), showlegend=False), row=2, col=1)

    atr_series = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    if atr_series is not None:
        fig.add_trace(go.Scatter(x=df.index, y=atr_series, name="ATR(14)",
                                 line=dict(color="#bb88ff", width=1.5), showlegend=False,
                                 fill="tozeroy", fillcolor="rgba(187,136,255,0.1)"), row=3, col=1)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        xaxis_rangeslider_visible=False, height=700,
        margin=dict(l=40, r=20, t=40, b=20),
        font=dict(family="monospace", size=12),
    )
    return fig


# ─────────────────────────────────────────────
# SCREENER
# ─────────────────────────────────────────────
def run_screener(tickers: list, direction_filter: str, min_abs_score: int, progress_bar) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(tickers):
        progress_bar.progress((i + 1) / len(tickers), text=f"Scanning {t}…")
        try:
            r = analyze_ticker(t)
            if not r:
                continue
            cs  = r["composite"]
            net = cs["net"]

            if direction_filter == "Bullish only" and net < min_abs_score:
                continue
            if direction_filter == "Bearish only" and net > -min_abs_score:
                continue
            if direction_filter == "Both" and abs(net) < min_abs_score:
                continue

            sigs = r["signals"]
            rows.append({
                "Ticker":    r["ticker"],
                "Price":     f"${r['price']:.2f}",
                "Change %":  r["change_pct"],
                "Net Score": net,
                "Direction": cs["direction"].title(),
                "Label":     cs["label"],
                "Bull Sigs": cs["bull_sigs"],
                "Bear Sigs": cs["bear_sigs"],
                "Price":     "✅" if sigs["price"]["bullish"]   else ("🔻" if sigs["price"]["bearish"]   else "—"),
                "Volume":    "✅" if sigs["volume"]["bullish"]  else ("🔻" if sigs["volume"]["bearish"]  else "—"),
                "BB":        "✅" if sigs["bb"]["bullish"]      else ("🔻" if sigs["bb"]["bearish"]      else "—"),
                "Pattern":   sigs["pattern"]["pattern"],
            })
        except Exception:
            continue
        time.sleep(0.05)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Net Score", ascending=False).reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════
st.markdown("""
<style>
    h1 { font-family: monospace; }
</style>
""", unsafe_allow_html=True)

st.title("📈 S&P 500 Breakout Detector")
st.caption("Real-time signal engine: price, volume, volatility & pattern — bullish and bearish")

st.sidebar.header("⚙️ Settings")
mode             = st.sidebar.radio("Mode", ["Single Ticker Analysis", "Screener"])
direction_filter = st.sidebar.radio("Direction", ["Bullish only", "Bearish only", "Both"], index=2)
lookback_weeks   = st.sidebar.slider("Price Lookback (weeks)", 4, 52, 20)
_vol_threshold   = st.sidebar.slider("Volume Spike Threshold (×avg)", 1.5, 4.0, 2.0, 0.1)
min_abs_score    = st.sidebar.slider("Min |Net Score|", 0, 100, 30)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Net Score = Bull score − Bear score**
- Positive → bullish momentum
- Negative → bearish momentum

| Signal | Weight |
|--------|--------|
| Price  | 35%    |
| Volume | 25%    |
| BB Squeeze | 20% |
| Pattern | 20%  |
""")


# ═══════════════════════
# MODE 1: SINGLE TICKER
# ═══════════════════════
if mode == "Single Ticker Analysis":
    ticker_input = st.selectbox("Select Ticker", SP500_TICKERS, index=SP500_TICKERS.index("AAPL"))

    if st.button("🔍 Analyze", type="primary"):
        with st.spinner(f"Fetching data for {ticker_input}…"):
            result = analyze_ticker(ticker_input, lookback_weeks=lookback_weeks)

        if result is None:
            st.error("Could not fetch data. Check the ticker and try again.")
        else:
            cs   = result["composite"]
            sigs = result["signals"]

            col1, col2, col3, col4, col5 = st.columns([1.5, 1, 1, 1, 2])
            col1.metric("Price",        f"${result['price']:.2f}", f"{result['change_pct']}%")
            col2.metric("Net Score",    cs["net"])
            col3.metric("Bull Signals", f"{cs['bull_sigs']} / 4")
            col4.metric("Bear Signals", f"{cs['bear_sigs']} / 4")
            col5.markdown(
                f"<br><span style='color:{cs['color']};font-size:20px;font-weight:700'>{cs['label']}</span>",
                unsafe_allow_html=True,
            )

            st.divider()
            st.subheader("🟢 Bullish Signals")
            b1, b2, b3, b4 = st.columns(4)
            for col, key, title, detail_key, detail_label in [
                (b1, "price",   "Price Breakout",  "pct_vs_resist", "% vs resistance"),
                (b2, "volume",  "Volume (Up Day)",  "vol_ratio",     "Vol ratio"),
                (b3, "bb",      "BB Above Upper",   "upper_band",    "Upper band"),
                (b4, "pattern", "Bull Flag",        "pattern",       "Pattern"),
            ]:
                sig   = sigs[key]
                score = sig["bull_score"]
                col.markdown(f"**{'✅' if sig['bullish'] else '⬜'} {title}**")
                col.progress(max(0, min(100, score)) / 100)
                col.caption(f"Score: {score}/100")
                if detail_key in sig:
                    col.caption(f"{detail_label}: {sig[detail_key]}")

            st.divider()
            st.subheader("🔴 Bearish Signals")
            r1, r2, r3, r4 = st.columns(4)
            for col, key, title, detail_key, detail_label in [
                (r1, "price",   "Price Breakdown",  "pct_vs_support", "% vs support"),
                (r2, "volume",  "Volume (Down Day)", "vol_ratio",      "Vol ratio"),
                (r3, "bb",      "BB Below Lower",    "lower_band",     "Lower band"),
                (r4, "pattern", "Bear Flag",         "pattern",        "Pattern"),
            ]:
                sig   = sigs[key]
                score = sig["bear_score"]
                col.markdown(f"**{'🔻' if sig['bearish'] else '⬜'} {title}**")
                col.progress(max(0, min(100, score)) / 100)
                col.caption(f"Score: {score}/100")
                if detail_key in sig:
                    col.caption(f"{detail_label}: {sig[detail_key]}")

            st.divider()
            st.subheader("Chart")
            st.plotly_chart(build_chart(result), use_container_width=True)

            with st.expander("🔬 Detailed Signal Data"):
                for name, sig in sigs.items():
                    st.markdown(f"**{name.title()} Signal**")
                    clean = {k: v for k, v in sig.items() if not isinstance(v, pd.Series)}
                    st.json(clean)


# ═══════════════════════
# MODE 2: SCREENER
# ═══════════════════════
else:
    st.subheader("S&P 500 Breakout / Breakdown Screener")

    num_tickers     = st.slider("Number of tickers to scan", 10, len(SP500_TICKERS), 50)
    tickers_to_scan = SP500_TICKERS[:num_tickers]

    if st.button("🚀 Run Screener", type="primary"):
        prog       = st.progress(0, text="Starting scan…")
        df_results = run_screener(tickers_to_scan, direction_filter, min_abs_score, prog)
        prog.empty()

        if df_results.empty:
            st.info("No results matching your filters. Try lowering the min score or changing direction.")
        else:
            bull_count = len(df_results[df_results["Net Score"] > 0])
            bear_count = len(df_results[df_results["Net Score"] < 0])
            st.success(f"Found **{len(df_results)}** candidates — 🟢 {bull_count} bullish, 🔴 {bear_count} bearish")

            def color_net(val):
                if val >= 75:  return "color: #00cc88; font-weight: bold"
                if val >= 40:  return "color: #44bb88; font-weight: bold"
                if val >= 10:  return "color: #ffcc00"
                if val <= -75: return "color: #ff4466; font-weight: bold"
                if val <= -40: return "color: #ff7744; font-weight: bold"
                if val <= -10: return "color: #ffcc00"
                return "color: #888888"

            styled = df_results.style.map(color_net, subset=["Net Score"])
            st.dataframe(styled, use_container_width=True, height=500)

            st.divider()
            st.subheader("Quick Drill-Down")
            selected = st.selectbox("Pick a ticker to analyze", df_results["Ticker"].tolist())
            if selected:
                with st.spinner(f"Analyzing {selected}…"):
                    r = analyze_ticker(selected)
                if r:
                    st.plotly_chart(build_chart(r), use_container_width=True)
