import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import warnings, os
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="S&P 500 Pattern Scanner",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main .block-container { padding-top: 1.5rem; }
.signal-card { background: #fff; border: 1px solid #e5e7eb; border-radius: 12px;
               padding: 14px 16px; margin-bottom: 10px; border-left: 4px solid #ccc; }
.signal-card.bull { border-left-color: #16a34a; }
.signal-card.bear { border-left-color: #dc2626; }
.card-ticker { font-size: 18px; font-weight: 700; color: #111; }
.card-pattern { font-size: 12px; color: #6b7280; margin-top: 2px; }
.badge { display:inline-block; padding:2px 8px; border-radius:8px; font-size:11px; font-weight:500; }
.badge-bull { background:#dcfce7; color:#15803d; }
.badge-bear { background:#fee2e2; color:#b91c1c; }
.stat-box { background:#f9fafb; border:1px solid #e5e7eb; border-radius:8px; padding:12px; text-align:center; }
.stat-val { font-size:24px; font-weight:700; color:#111; }
.stat-lbl { font-size:11px; color:#6b7280; margin-top:2px; }
.conf-bar-bg { background:#f3f4f6; border-radius:4px; height:4px; margin-top:6px; }
.conf-bar-fill { height:4px; border-radius:4px; }
</style>
""", unsafe_allow_html=True)

PATTERN_NAMES = [
    "Head & Shoulders", "Inv Head & Shoulders", "Double Top", "Double Bottom",
    "Rising Wedge", "Falling Wedge", "Bull Flag", "Bear Flag"
]
N_PAT = 8
WINDOW = 30
BULLISH = {1, 3, 5, 6}
BEARISH = {0, 2, 4, 7}
DEFAULT_CKPT = r"C:\Users\rcsch\Documents\python\sp500_pattern_lstm.pt"

# -- MODEL --
class PatternLSTM(nn.Module):
    def __init__(self, in_=5, hid=192, layers=3, drop=0.35, n_cls=8):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_,48), nn.LayerNorm(48), nn.GELU(), nn.Dropout(drop*.4))
        self.lstm = nn.LSTM(48, hid, layers, batch_first=True, dropout=drop, bidirectional=True)
        self.attn = nn.Sequential(nn.Linear(hid*2,96), nn.Tanh(), nn.Linear(96,1))
        self.head = nn.Sequential(
            nn.LayerNorm(hid*2), nn.Dropout(drop),
            nn.Linear(hid*2,256), nn.GELU(),
            nn.Dropout(drop*.5), nn.Linear(256,128), nn.GELU(), nn.Linear(128,n_cls)
        )
    def forward(self, x):
        x = self.proj(x)
        o, _ = self.lstm(x)
        return self.head((o * torch.softmax(self.attn(o), dim=1)).sum(dim=1))

def make_feat(df, s, e):
    c=df["Close"].values[s:e]; o=df["Open"].values[s:e]
    h=df["High"].values[s:e];  l=df["Low"].values[s:e]
    v=df["Volume"].values[s:e].astype(float)
    base=c[0] if c[0]!=0 else 1.0; vm=v.mean() if v.mean()!=0 else 1.0
    return np.stack([o/base-1, h/base-1, l/base-1, c/base-1, v/vm-1], axis=1).astype(np.float32)

# -- DATA HELPERS --
@st.cache_data(ttl=3600)
def get_sp500_tickers():
    resp = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "Mozilla/5.0"}, timeout=15
    )
    tbl = BeautifulSoup(resp.text, "html.parser").find("table", {"id": "constituents"})
    rows = [(r.find_all("td")[0].text.strip().replace(".","-"),
             r.find_all("td")[2].text.strip())
            for r in tbl.find_all("tr")[1:] if r.find_all("td")]
    return [(t, s) for t, s in rows]

@st.cache_data(ttl=900)
def fetch_prices(ticker):
    df = yf.download(ticker, period="3mo", auto_adjust=True, progress=False)
    if hasattr(df.columns, "levels"):
        df.columns = [c[0] for c in df.columns]
    return df if len(df) >= WINDOW else None

@st.cache_data(ttl=900)
def fetch_prices_1y(ticker):
    df = yf.download(ticker, period="1y", auto_adjust=True, progress=False)
    if hasattr(df.columns, "levels"):
        df.columns = [c[0] for c in df.columns]
    return df if len(df) >= WINDOW else None

@st.cache_resource
def load_model(ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)
    m      = PatternLSTM().to(device)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    thresh = [float(t) for t in ckpt["thresholds"]]
    return m, device, thresh

def run_inference(model, device, thresholds, df):
    n = len(df)
    if n < WINDOW: return []
    feat = make_feat(df, n - WINDOW, n)
    if np.isnan(feat).any() or np.isinf(feat).any(): return []
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.FloatTensor(feat[None]).to(device))).cpu().numpy()[0]
    return [(i, float(prob[i])) for i in range(N_PAT) if prob[i] > thresholds[i]]

def price_stats(df):
    c = df["Close"].values
    last = round(float(c[-1]), 2)
    chg1d = round((c[-1]/c[-2]-1)*100, 1) if len(c) > 1 else 0
    chg1w = round((c[-1]/c[-6]-1)*100, 1) if len(c) > 6 else 0
    chg1m = round((c[-1]/c[-22]-1)*100, 1) if len(c) > 22 else 0
    return last, chg1d, chg1w, chg1m

def sparkline(closes_arr, color):
    n = len(closes_arr)
    mn, mx = min(closes_arr), max(closes_arr)
    W, H = 140, 30
    if mx == mn: pts = " ".join(f"{i*(W/max(1,n-1)):.1f},{H/2:.1f}" for i in range(n))
    else: pts = " ".join(f"{i*(W/max(1,n-1)):.1f},{H - (v-mn)/(mx-mn)*H:.1f}" for i,v in enumerate(closes_arr))
    return f'<svg width="{W}" height="{H}"><polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5" stroke-linejoin="round"/></svg>'

# -- SIDEBAR --
with st.sidebar:
    st.markdown("## Settings")
    ckpt_path = st.text_input("Model checkpoint path", value=DEFAULT_CKPT)
    st.markdown("---")
    st.markdown("**Filters**")
    sentiment_filter = st.radio("Sentiment", ["All", "Bullish only", "Bearish only"], index=0)
    min_conf = st.slider("Min confidence %", 60, 90, 70, 1)
    pattern_filter = st.multiselect("Pattern types", PATTERN_NAMES, default=PATTERN_NAMES)
    st.markdown("---")
    st.markdown("**Scan scope**")
    scan_mode = st.radio("Stocks to scan", ["Full S&P 500", "Custom list"], index=0)
    custom_tickers = ""
    if scan_mode == "Custom list":
        custom_tickers = st.text_area("Tickers (comma-separated)", "AAPL,MSFT,NVDA,GOOGL,META,AMZN,NFLX")
    st.markdown("---")
    run_btn = st.button("Run scan", type="primary", width='stretch')
    st.caption(f"BiLSTM 2.3M params | Window: {WINDOW}d | GPU: {'YES' if torch.cuda.is_available() else 'NO'}")

# -- MAIN --
st.markdown("# S&P 500 Pattern Scanner")
st.markdown("Real-time LSTM detection of technical formations on the latest 30-day window.")

if not os.path.exists(ckpt_path):
    st.error(f"Checkpoint not found: {ckpt_path}  \n\nUpdate the path in the sidebar.")
    st.stop()

with st.spinner("Loading model weights..."):
    model, device, thresholds = load_model(ckpt_path)

if not run_btn and "scan_results" not in st.session_state:
    st.info("Configure filters in the sidebar and click **Run scan** to detect formations.")
    st.stop()

# -- SCAN --
if run_btn:
    if scan_mode == "Full S&P 500":
        ticker_list = [t for t, _ in get_sp500_tickers()]
    else:
        ticker_list = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]

    signals = []
    prog_bar = st.progress(0, text="Starting scan...")
    status   = st.empty()
    total    = len(ticker_list)

    for i, ticker in enumerate(ticker_list):
        prog_bar.progress((i+1)/total, text=f"Scanning {ticker} ({i+1}/{total})")
        df = fetch_prices(ticker)
        if df is None: continue
        detections = run_inference(model, device, thresholds, df)
        if not detections: continue
        last, chg1d, chg1w, chg1m = price_stats(df)
        spark_closes = [float(v) for v in df["Close"].values[-30:]]
        dates_disp   = [str(d.date()) for d in df.index[-50:]]
        closes_disp  = [round(float(v),2) for v in df["Close"].values[-50:]]
        for pat_idx, conf in detections:
            signals.append({
                "ticker": ticker, "pattern": PATTERN_NAMES[pat_idx], "pat_idx": pat_idx,
                "sentiment": "Bullish" if pat_idx in BULLISH else "Bearish",
                "conf": round(conf*100,1), "last": last,
                "chg1d": chg1d, "chg1w": chg1w, "chg1m": chg1m,
                "spark": spark_closes, "dates": dates_disp, "closes": closes_disp,
                "as_of": str(df.index[-1].date()),
            })

    prog_bar.empty(); status.empty()
    signals.sort(key=lambda x: -x["conf"])
    st.session_state["scan_results"] = signals
    st.session_state["scan_time"]    = datetime.now().strftime("%H:%M:%S")
    st.session_state["scan_n"]       = len(ticker_list)

# -- DISPLAY --
signals     = st.session_state.get("scan_results", [])
scan_time   = st.session_state.get("scan_time", "")
scan_n      = st.session_state.get("scan_n", 0)

def apply_filters(sigs):
    out = sigs
    if sentiment_filter == "Bullish only":
        out = [s for s in out if s["sentiment"] == "Bullish"]
    elif sentiment_filter == "Bearish only":
        out = [s for s in out if s["sentiment"] == "Bearish"]
    out = [s for s in out if s["conf"] >= min_conf]
    out = [s for s in out if s["pattern"] in pattern_filter]
    return out

filtered = apply_filters(signals)
bull_cnt = sum(1 for s in filtered if s["sentiment"] == "Bullish")
bear_cnt = sum(1 for s in filtered if s["sentiment"] == "Bearish")

c1,c2,c3,c4,c5 = st.columns(5)
with c1: st.markdown(f'<div class="stat-box"><div class="stat-val">{scan_n}</div><div class="stat-lbl">stocks scanned</div></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="stat-box"><div class="stat-val">{len(signals)}</div><div class="stat-lbl">total signals</div></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="stat-box"><div class="stat-val" style="color:#16a34a">{bull_cnt}</div><div class="stat-lbl">bullish</div></div>', unsafe_allow_html=True)
with c4: st.markdown(f'<div class="stat-box"><div class="stat-val" style="color:#dc2626">{bear_cnt}</div><div class="stat-lbl">bearish</div></div>', unsafe_allow_html=True)
with c5: st.markdown(f'<div class="stat-box"><div class="stat-val" style="font-size:16px">{scan_time}</div><div class="stat-lbl">last scan</div></div>', unsafe_allow_html=True)

st.markdown("")
if not filtered:
    st.warning("No signals match the current filters.")
    st.stop()

col_sort, col_view, _ = st.columns([2,2,6])
with col_sort:
    sort_by = st.selectbox("Sort by", ["Confidence","1d change","1m change","Ticker"], label_visibility="collapsed")
with col_view:
    view = st.radio("View", ["Cards","Table"], horizontal=True, label_visibility="collapsed")

sort_map = {"Confidence": lambda x: -x["conf"], "1d change": lambda x: -x["chg1d"],
            "1m change": lambda x: -x["chg1m"], "Ticker": lambda x: x["ticker"]}
filtered.sort(key=sort_map[sort_by])

# TABLE VIEW
if view == "Table":
    df_disp = pd.DataFrame([{
        "Ticker": s["ticker"], "Pattern": s["pattern"], "Sentiment": s["sentiment"],
        "Conf %": s["conf"], "Price": f'${s["last"]:.2f}',
        "1d %": f'{s["chg1d"]:+.1f}%', "1w %": f'{s["chg1w"]:+.1f}%', "1m %": f'{s["chg1m"]:+.1f}%',
        "As of": s["as_of"],
    } for s in filtered])
    st.dataframe(df_disp, width='stretch', hide_index=True)

# CARD VIEW
else:
    seen, deduped = set(), []
    for s in filtered:
        if s["ticker"] not in seen:
            seen.add(s["ticker"]); deduped.append(s)

    COLS = 3
    for row_sigs in [deduped[i:i+COLS] for i in range(0, len(deduped), COLS)]:
        cols = st.columns(COLS)
        for col, sig in zip(cols, row_sigs):
            with col:
                is_bull = sig["sentiment"] == "Bullish"
                col_accent  = "#16a34a" if is_bull else "#dc2626"
                badge_cls   = "badge-bull" if is_bull else "badge-bear"
                chg1d_col   = "#16a34a" if sig["chg1d"] >= 0 else "#dc2626"
                chg1m_col   = "#16a34a" if sig["chg1m"] >= 0 else "#dc2626"
                spark_svg   = sparkline(sig["spark"], col_accent)
                st.markdown(f"""
<div class="signal-card {'bull' if is_bull else 'bear'}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start">
    <div><div class="card-ticker">{sig["ticker"]}</div>
         <div class="card-pattern">{sig["pattern"]}</div></div>
    <div style="text-align:right">
      <div style="font-size:17px;font-weight:600">${sig["last"]:.2f}</div>
      <div style="font-size:11px;color:{chg1d_col}">{sig["chg1d"]:+.1f}% today</div>
    </div>
  </div>
  <div style="margin:7px 0 4px;display:flex;align-items:center;justify-content:space-between">
    <span class="badge {badge_cls}">{sig["sentiment"]}</span>
    <span style="font-size:11px;color:#6b7280">{sig["conf"]:.1f}% conf</span>
    <span style="font-size:11px;color:{chg1m_col}">{sig["chg1m"]:+.1f}% 1m</span>
  </div>
  <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{sig["conf"]}%;background:{col_accent}"></div></div>
  <div style="margin-top:8px">{spark_svg}</div>
  <div style="font-size:10px;color:#9ca3af;margin-top:4px">as of {sig["as_of"]}</div>
</div>""", unsafe_allow_html=True)

                if st.button(f"Detail", key=f"d_{sig['ticker']}_{sig['pat_idx']}"):
                    st.session_state["detail_sig"] = sig
                    st.rerun()

# DETAIL PANEL
if "detail_sig" in st.session_state:
    sig = st.session_state["detail_sig"]
    st.markdown("---")

    is_bull_det = sig["sentiment"] == "Bullish"
    badge_col   = "#15803d" if is_bull_det else "#b91c1c"
    badge_bg    = "#dcfce7" if is_bull_det else "#fee2e2"
    st.markdown(
        f"### {sig['ticker']} "
        f'<span style="font-size:13px;background:{badge_bg};color:{badge_col};'
        f'padding:2px 10px;border-radius:8px;font-weight:500">{sig["sentiment"]}</span>'
        f' <span style="font-size:14px;color:#6b7280;font-weight:400">{sig["pattern"]} | {sig["conf"]:.1f}% confidence</span>',
        unsafe_allow_html=True
    )

    with st.spinner(f"Loading 1-year history for {sig['ticker']}..."):
        df_1y = fetch_prices_1y(sig["ticker"])

    col_a, col_b = st.columns([3, 1])
    with col_a:
        if df_1y is not None and len(df_1y) > 0:
            closes_1y = [round(float(v), 2) for v in df_1y["Close"].values]
            dates_1y  = [str(d.date()) for d in df_1y.index]
            n1y       = len(closes_1y)
            sma20_1y  = [round(float(np.mean(closes_1y[max(0,i-19):i+1])), 2) for i in range(n1y)]
            sma50_1y  = [round(float(np.mean(closes_1y[max(0,i-49):i+1])), 2) for i in range(n1y)]
            import plotly.graph_objects as go
            dates_dt = pd.to_datetime(dates_1y)
            # Tight y-axis: pad 2% above/below the actual data range
            y_min = min(closes_1y) * 0.98
            y_max = max(closes_1y) * 1.02
            # Shade the 30-day pattern window
            pat_start_dt = dates_dt[-WINDOW]

            fig = go.Figure()
            # Pattern window shading
            fig.add_vrect(
                x0=pat_start_dt, x1=dates_dt[-1],
                fillcolor="rgba(99,153,34,0.08)" if is_bull_det else "rgba(220,38,38,0.08)",
                line_width=0,
                annotation_text="Pattern window",
                annotation_position="top left",
                annotation_font_size=10,
                annotation_font_color="#6b7280",
            )
            # SMA 50
            fig.add_trace(go.Scatter(
                x=dates_dt, y=sma50_1y, name="SMA 50",
                line=dict(color="#f59e0b", width=1, dash="dot"),
                hovertemplate="%{y:.2f}",
            ))
            # SMA 20
            fig.add_trace(go.Scatter(
                x=dates_dt, y=sma20_1y, name="SMA 20",
                line=dict(color="#3b82f6", width=1, dash="dot"),
                hovertemplate="%{y:.2f}",
            ))
            # Invisible baseline at y_min so fill doesn't go to zero
            fig.add_trace(go.Scatter(
                x=dates_dt, y=[y_min] * n1y,
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))
            # Price line filled down to y_min (not zero)
            fig.add_trace(go.Scatter(
                x=dates_dt, y=closes_1y, name="Close",
                line=dict(color="#dc2626" if not is_bull_det else "#16a34a", width=1.8),
                fill="tonexty",
                fillcolor="rgba(220,38,38,0.05)" if not is_bull_det else "rgba(22,163,74,0.05)",
                hovertemplate="%{x|%b %d %Y}  $%{y:.2f}<extra></extra>",
            ))
            fig.update_layout(
                height=320,
                margin=dict(l=8, r=8, t=8, b=8),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(
                    range=[y_min, y_max],
                    tickprefix="$",
                    tickformat=".0f",
                    gridcolor="rgba(0,0,0,0.05)",
                    showline=False,
                    zeroline=False,
                ),
                xaxis=dict(
                    gridcolor="rgba(0,0,0,0.05)",
                    showline=False,
                    zeroline=False,
                ),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(size=11),
                ),
                hovermode="x unified",
            )
            st.plotly_chart(fig, width='stretch')
            st.caption(
                f"1-year daily close with 20d and 50d moving averages. "
                f"Y-axis scaled to price range. "
                f"Shaded region = 30-day pattern window (ending {sig['as_of']})."
            )
        else:
            st.warning("Could not load 1-year data for this ticker.")

    with col_b:
        st.markdown("&nbsp;")
        st.metric("Last price",  f'${sig["last"]:.2f}')
        st.metric("Today",       f'{sig["chg1d"]:+.1f}%')
        st.metric("1 week",      f'{sig["chg1w"]:+.1f}%')
        st.metric("1 month",     f'{sig["chg1m"]:+.1f}%')
        st.metric("Confidence",  f'{sig["conf"]:.1f}%')
        if df_1y is not None and len(df_1y) > 5:
            hi_52  = round(float(max(df_1y["High"].values)), 2)
            lo_52  = round(float(min(df_1y["Low"].values)), 2)
            ret_1y = round((float(df_1y["Close"].values[-1]) / float(df_1y["Close"].values[0]) - 1) * 100, 1)
            st.metric("52w High",  f'${hi_52:.2f}')
            st.metric("52w Low",   f'${lo_52:.2f}')
            st.metric("1y Return", f'{ret_1y:+.1f}%')
        st.markdown("&nbsp;")
        if st.button("Close detail"):
            del st.session_state["detail_sig"]
            st.rerun()

st.markdown("---")
st.caption("BiLSTM pattern scanner | 2.3M params | Trained on 501 S&P 500 stocks | Not financial advice")