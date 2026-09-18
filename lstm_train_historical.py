"""
lstm_train_historical.py
========================
Builds a training dataset by replaying the breakout detector scoring logic
across historical 15-minute data, then trains two shared LSTM models:

    models/bullish_lstm.pt   — trained on bullish breakout events
    models/bearish_lstm.pt   — trained on bearish breakout events

Each training sample conditions on the FIRST 15 minutes of the target day:
    X       : last SEQ_LEN 15-min bars ending at the target day's *opening* bar
              →  (SEQ_LEN, n_features)
    context : a per-sample static vector (overnight gap, opening-bar
              microstructure, daily breakout-score components) + a sector id
    y       : the REMAINING 15-min bars of the target day, parameterised as
              (mid_return, log_range) and ATR-normalised  →  (N_OUT, 2)
                y[:, 0] = log(sqrt(H*L) / open_close) / sigma     (mid return)
                y[:, 1] = log(H / L)                  / sigma     (>= 0, range)
              where sigma = ATR14 / Close on the signal day. Predicting in
              volatility units makes tickers/days comparable and lets the cone
              widen on volatile days and tighten on calm ones.

The model reconstructs high/low from (mid, range) so high >= low is guaranteed
structurally — no post-hoc clamp needed.

DATA SOURCE: intraday and daily bars come from the local Massive Parquet store
(C:/market_data) via massive_feed, which derives any interval from 5 years of
1-minute bars. The old yfinance ~60-day cap on 15-minute history no longer
applies, so INTRADAY_PERIOD is a real choice rather than a limit. Tickers
absent from the store fall back to yfinance automatically.

Usage:
    python lstm_train_historical.py
    python lstm_train_historical.py --bullish-threshold 35 --bearish-threshold 25
    python lstm_train_historical.py --epochs 40 --bullish-only
"""

import argparse
import json
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import yfinance as yf
import joblib

# Local Massive Parquet store; yfinance-compatible download(). Falls back to
# yfinance for any ticker not in the store.
import massive_feed

warnings.filterwarnings("ignore")

import sys

# Windows consoles default to cp1252, which cannot encode the ✓ and — glyphs in
# this script's progress output. Without this, a *successful* training run dies
# with UnicodeEncodeError on its own "saved" log line, after the model is
# already written to disk. Harmless to re-run on a console that is already UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).parent / "breakout_detector"))

from breakout_detector import SP500_TICKERS  # type: ignore

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

SAVE_DIR   = Path("models")
SAVE_DIR.mkdir(exist_ok=True)

META_VERSION = 2          # bump when the model I/O contract changes

# ── Data resolution
INTERVAL        = "15m"   # intraday bar interval (derived from 1-min store)
INTRADAY_PERIOD = "2y"    # was "60d" under yfinance; store holds 5y
DAILY_PERIOD    = "2y"    # daily bars (scoring / vol scale)

# ── Per-bar features (fed as the encoder input sequence)
FEATURE_COLS = [
    "ret_1h", "ret_3h", "ret_6h", "ret_12h",
    "vwap_delta", "vol_ratio_10",
    "rsi_14", "macd_hist", "bb_pos", "atr_norm",
    "bar_body", "bar_range",
    "tod_sin", "tod_cos", "dow_sin", "dow_cos",
]

# ── Static per-sample context (day-specific conditioning, NOT a time series)
CONTEXT_COLS = [
    "gap",              # log(open_bar_open / prev_day_close)  — overnight gap
    "open_ret",         # log(open_bar_close / open_bar_open)  — opening body
    "open_range",       # log(open_bar_high / open_bar_low)    — opening range
    "open_vol_ratio",   # opening-bar volume / recent avg
    "d_price_pos",      # daily 52w price position (0..1)
    "d_rsi",            # daily RSI-14 / 100
    "d_bb_pos",         # daily Bollinger position
]

# A regular-session trading day (09:30–16:00 = 6.5h) is 26 fifteen-minute bars.
BARS_PER_DAY = 26
SEQ_LEN      = 26          # 15-min bars of context (input ends at opening bar)
N_OUT        = BARS_PER_DAY - 1   # 25 bars predicted = the rest of the day
N_TARGETS    = 2          # [mid_return, log_range] per bar (both ATR-normalised)

VOL_FLOOR    = 1e-3       # floor on the ATR vol scale to avoid divide-by-zero

# ── Default loss weights (overridable via CLI)
LAM_RANGE    = 1.0        # weight on the range head
LAM_COV      = 0.5        # weight on the coverage (pinball) term
TAU_HI       = 0.9        # upper-quantile target for the predicted high
TAU_LO       = 0.1        # lower-quantile target for the predicted low

SECTOR_MAP_PATH = SAVE_DIR / "sector_map.json"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — SECTOR MAP (cached one-time yfinance pull)
# ══════════════════════════════════════════════════════════════════════════════

def get_sector_map(tickers: list, fetch_missing: bool = True) -> dict:
    """
    ticker -> GICS sector string, cached to models/sector_map.json.

    The first training run fetches any missing sectors via yfinance (slow,
    ~1 call/ticker) and caches them. Subsequent runs and inference just read
    the cache. Anything unknown maps to 'Unknown'.
    """
    cache = {}
    if SECTOR_MAP_PATH.exists():
        try:
            cache = json.loads(SECTOR_MAP_PATH.read_text())
        except Exception:
            cache = {}

    missing = [t for t in tickers if t not in cache]
    if fetch_missing and missing:
        print(f"\n  Resolving sectors for {len(missing)} ticker(s) "
              f"(one-time, cached to {SECTOR_MAP_PATH.name}) ...")
        for i, t in enumerate(missing):
            sector = "Unknown"
            try:
                info = yf.Ticker(t).info or {}
                sector = info.get("sector") or "Unknown"
            except Exception:
                sector = "Unknown"
            cache[t] = sector
            if (i + 1) % 25 == 0:
                print(f"    {i+1}/{len(missing)} ...")
                SECTOR_MAP_PATH.write_text(json.dumps(cache, indent=2))
        SECTOR_MAP_PATH.write_text(json.dumps(cache, indent=2))
        print("  Sector map cached.\n")

    return cache


def build_sector_vocab(sector_map: dict) -> list:
    """Sorted list of sectors; index 0 is always 'Unknown'."""
    sectors = sorted({s for s in sector_map.values() if s and s != "Unknown"})
    return ["Unknown"] + sectors


def sector_to_idx(sector: str, vocab: list) -> int:
    try:
        return vocab.index(sector)
    except ValueError:
        return 0   # Unknown


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — FEATURE ENGINEERING  (15-min bars)
# ══════════════════════════════════════════════════════════════════════════════

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c  = df["Close"]
    o  = df["Open"]
    v  = df["Volume"]

    df["ret_1h"]  = c.pct_change(1)
    df["ret_3h"]  = c.pct_change(3)
    df["ret_6h"]  = c.pct_change(6)
    df["ret_12h"] = c.pct_change(12)

    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    df["vwap_10"]    = (tp * v).rolling(10).sum() / v.rolling(10).sum()
    df["vwap_delta"] = (c - df["vwap_10"]) / df["vwap_10"].replace(0, np.nan)

    df["vol_ratio_10"] = v / v.rolling(10).mean().clip(lower=1)

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    df["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()

    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    df["bb_pos"] = (c - mid) / (2 * std + 1e-9)

    hl  = df["High"] - df["Low"]
    hc  = (df["High"] - c.shift()).abs()
    lc  = (df["Low"]  - c.shift()).abs()
    atr = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df["atr_norm"] = atr / c.replace(0, np.nan)

    # ── Per-bar microstructure
    df["bar_body"]  = np.log(c / o.replace(0, np.nan))            # close vs open
    df["bar_range"] = np.log(df["High"] / df["Low"].replace(0, np.nan))

    if hasattr(df.index, "hour"):
        # minute-of-day fraction — finer than hourly so 15-min bars are distinct
        minute_frac   = (df.index.hour * 60 + df.index.minute) / (24 * 60)
        df["tod_sin"] = np.sin(2 * np.pi * minute_frac)
        df["tod_cos"] = np.cos(2 * np.pi * minute_frac)
    else:
        df["tod_sin"] = 0.0
        df["tod_cos"] = 1.0

    if hasattr(df.index, "dayofweek"):
        dow           = df.index.dayofweek / 4
        df["dow_sin"] = np.sin(2 * np.pi * dow)
        df["dow_cos"] = np.cos(2 * np.pi * dow)
    else:
        df["dow_sin"] = 0.0
        df["dow_cos"] = 1.0

    df.dropna(inplace=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SCORE A DAILY SLICE (historical replay)
# ══════════════════════════════════════════════════════════════════════════════

def score_daily_slice(daily_df: pd.DataFrame, return_components: bool = False):
    """
    Breakout score (-100..+100) on a daily OHLCV slice. Positive = bullish.

    With return_components=True also returns a dict of the raw building blocks
    (price_pos, rsi, bb_pos, vol_ratio) so they can be fed to the model as
    conditioning features rather than only used to pick bullish/bearish.
    """
    if len(daily_df) < 40:
        return (0, {}) if return_components else 0

    c = daily_df["Close"]
    v = daily_df["Volume"]

    window     = min(252, len(c) - 1)
    roll_high  = float(c.iloc[-(window+1):-1].max())
    roll_low   = float(c.iloc[-(window+1):-1].min())
    curr       = float(c.iloc[-1])
    rng        = max(roll_high - roll_low, 1e-9)
    price_pos  = (curr - roll_low) / rng           # 0=at low, 1=at high
    price_score = (price_pos - 0.5) * 200          # -100 to +100

    avg_vol   = float(v.iloc[-21:-1].mean()) if len(v) > 21 else float(v.mean())
    vol_ratio = float(v.iloc[-1]) / max(avg_vol, 1)
    day_up    = curr > float(c.iloc[-2])
    vol_score = min(50.0, (vol_ratio - 1) * 25) * (1 if day_up else -1)

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = float(gain.iloc[-1]) / max(float(loss.iloc[-1]), 1e-9)
    rsi   = 100 - (100 / (1 + rs))
    rsi_score = (rsi - 50) * 1.0                   # -50 to +50

    mid      = c.rolling(20).mean()
    std      = c.rolling(20).std()
    bb_pos   = float((c.iloc[-1] - mid.iloc[-1]) / (2 * std.iloc[-1] + 1e-9))
    bb_score = float(np.clip(bb_pos * 100, -50, 50))

    net = (
        price_score * 0.35 +
        vol_score   * 0.25 +
        rsi_score   * 0.20 +
        bb_score    * 0.20
    )
    score = int(np.clip(round(net), -100, 100))

    if return_components:
        comps = {
            "price_pos": float(price_pos),
            "rsi":       float(rsi),
            "bb_pos":    float(bb_pos),
            "vol_ratio": float(vol_ratio),
        }
        return score, comps
    return score


def vol_scale(daily_slice: pd.DataFrame) -> float:
    """ATR14 / Close on the most recent daily bar — the per-sample vol unit."""
    if len(daily_slice) < 16:
        return VOL_FLOOR
    h = daily_slice["High"]; l = daily_slice["Low"]; c = daily_slice["Close"]
    hl = h - l
    hc = (h - c.shift()).abs()
    lc = (l - c.shift()).abs()
    atr = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean().iloc[-1]
    last_c = float(c.iloc[-1])
    sigma = float(atr) / last_c if last_c > 0 else VOL_FLOOR
    return max(sigma, VOL_FLOOR)


def build_context(prev_close: float, open_bar, open_vol_ratio: float,
                  comps: dict) -> np.ndarray:
    """Assemble the static context vector (order must match CONTEXT_COLS)."""
    o = float(open_bar["Open"]); h = float(open_bar["High"])
    l = float(open_bar["Low"]);  cl = float(open_bar["Close"])
    gap        = np.log(max(o, 1e-9)  / max(prev_close, 1e-9))
    open_ret   = np.log(max(cl, 1e-9) / max(o, 1e-9))
    open_range = np.log(max(h, 1e-9)  / max(l, 1e-9))
    return np.array([
        gap, open_ret, open_range, open_vol_ratio,
        comps.get("price_pos", 0.5),
        comps.get("rsi", 50.0) / 100.0,
        comps.get("bb_pos", 0.0),
    ], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BUILD TRAINING DATASET
# ══════════════════════════════════════════════════════════════════════════════

def _targets_mid_range(highs, lows, open_close, sigma):
    """(H, L) arrays -> ATR-normalised (mid_return, log_range) targets."""
    log_h = np.log(highs / open_close)
    log_l = np.log(lows  / open_close)
    m = 0.5 * (log_h + log_l)        # geometric-mid log return
    r = log_h - log_l                # log range  (>= 0)
    return (np.stack([m / sigma, r / sigma], axis=1)).astype(np.float32)


def build_training_dataset(
    tickers:           list,
    bullish_threshold: int  = 35,
    bearish_threshold: int  = 25,
    sector_map:        dict = None,
    sector_vocab:      list = None,
    seq_len:           int  = SEQ_LEN,
    n_out:             int  = N_OUT,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  Building training dataset ({INTERVAL}, opening-bar conditioned)")
    print(f"  Tickers: {len(tickers)}  |  Thresholds: bull +{bullish_threshold} / bear -{bearish_threshold}")
    print(f"  Intraday window: {INTRADAY_PERIOD} (local Massive store, 5y available)")
    print(f"{'='*60}\n")

    sector_map   = sector_map or {}
    sector_vocab = sector_vocab or ["Unknown"]

    data = {
        "bullish": {"X": [], "y": [], "ctx": [], "sec": [], "days": []},
        "bearish": {"X": [], "y": [], "ctx": [], "sec": [], "days": []},
    }

    for t_idx, ticker in enumerate(tickers):
        print(f"  [{t_idx+1:3d}/{len(tickers)}] {ticker} ...", end=" ", flush=True)
        sec_id = sector_to_idx(sector_map.get(ticker, "Unknown"), sector_vocab)

        try:
            daily = massive_feed.download(ticker, period=DAILY_PERIOD, interval="1d",
                                          auto_adjust=True, progress=False)
            if daily.empty or len(daily) < 100:
                print("skip (no daily data)")
                continue
            if isinstance(daily.columns, pd.MultiIndex):
                daily.columns = daily.columns.get_level_values(0)
            daily.index = pd.to_datetime(daily.index).tz_localize(None)

            intraday = massive_feed.download(ticker, period=INTRADAY_PERIOD, interval=INTERVAL,
                                             auto_adjust=True, progress=False)
            if intraday.empty or len(intraday) < seq_len + n_out:
                print("skip (no/short intraday data)")
                continue
            if isinstance(intraday.columns, pd.MultiIndex):
                intraday.columns = intraday.columns.get_level_values(0)
            intraday.index = pd.to_datetime(intraday.index).tz_localize(None)

            intra_feat = compute_features(intraday)
            if intra_feat.empty:
                print("skip (features failed)")
                continue

            avg_vol_15 = float(intraday["Volume"].rolling(20).mean().mean())
            avg_vol_15 = max(avg_vol_15, 1.0)

            intra_days = sorted(intra_feat.index.normalize().unique())

            samples = 0
            for i in range(len(intra_days) - 1):
                signal_day = intra_days[i]
                target_day = intra_days[i + 1]

                daily_slice = daily[daily.index.normalize() <= signal_day]
                if len(daily_slice) < 80:
                    continue

                score, comps = score_daily_slice(daily_slice, return_components=True)
                if   score >=  bullish_threshold: direction = "bullish"
                elif score <= -bearish_threshold: direction = "bearish"
                else:                              continue

                tgt_bars = intraday[intraday.index.normalize() == target_day]
                if len(tgt_bars) < 3:
                    continue
                open_ts = tgt_bars.index[0]

                hist_feat = intra_feat[intra_feat.index <= open_ts]
                if len(hist_feat) < seq_len or hist_feat.index[-1] != open_ts:
                    continue
                X = hist_feat[FEATURE_COLS].values[-seq_len:].astype(np.float32)

                open_close = float(tgt_bars["Close"].iloc[0])
                if open_close <= 0:
                    continue

                prev_close     = float(daily_slice["Close"].iloc[-1])
                open_vol_ratio = float(tgt_bars["Volume"].iloc[0]) / avg_vol_15
                ctx   = build_context(prev_close, tgt_bars.iloc[0], open_vol_ratio, comps)
                sigma = vol_scale(daily_slice)

                rest = tgt_bars.iloc[1:1 + n_out]
                if len(rest) < 3:
                    continue
                if len(rest) < n_out:
                    pad  = pd.concat([rest.iloc[[-1]]] * (n_out - len(rest)))
                    rest = pd.concat([rest, pad])

                y = _targets_mid_range(rest["High"].values, rest["Low"].values,
                                       open_close, sigma)

                d = data[direction]
                d["X"].append(X); d["y"].append(y); d["ctx"].append(ctx)
                d["sec"].append(sec_id); d["days"].append(target_day)
                samples += 1

            print(f"{samples} samples")

        except Exception as e:
            print(f"error: {e}")

        time.sleep(0.1)

    print(f"\n  Dataset complete:")
    print(f"    Bullish samples : {len(data['bullish']['X'])}")
    print(f"    Bearish samples : {len(data['bearish']['X'])}")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DATASET
# ══════════════════════════════════════════════════════════════════════════════

class BreakoutDataset(Dataset):
    """
    X   : (N, seq_len, n_features)   scaled with MinMaxScaler
    ctx : (N, n_context)             scaled with StandardScaler
    sec : (N,)                       sector index (long)
    y   : (N, n_out, 2)             [mid_return, log_range] (ATR-normalised)
    """
    def __init__(self, X, ctx, sec, y, x_scaler=None, ctx_scaler=None):
        X_arr   = np.stack(X)     # (N, T, F)
        ctx_arr = np.stack(ctx)   # (N, C)
        y_arr   = np.stack(y)     # (N, n_out, 2)
        N, T, Fdim = X_arr.shape

        if x_scaler is None:
            self.x_scaler = MinMaxScaler().fit(X_arr.reshape(-1, Fdim))
        else:
            self.x_scaler = x_scaler
        if ctx_scaler is None:
            self.ctx_scaler = StandardScaler().fit(ctx_arr)
        else:
            self.ctx_scaler = ctx_scaler

        X_scaled   = self.x_scaler.transform(X_arr.reshape(-1, Fdim)).reshape(N, T, Fdim)
        ctx_scaled = self.ctx_scaler.transform(ctx_arr)

        self.X   = torch.tensor(X_scaled,   dtype=torch.float32)
        self.ctx = torch.tensor(ctx_scaled, dtype=torch.float32)
        self.sec = torch.tensor(np.asarray(sec), dtype=torch.long)
        self.y   = torch.tensor(y_arr,      dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.ctx[idx], self.sec[idx], self.y[idx]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MODEL
# ══════════════════════════════════════════════════════════════════════════════

class BreakoutLSTM(nn.Module):
    """
    Encoder-decoder LSTM with a static-context fusion.

    Encoder : seq_len 15-min input bars (ending at the opening bar).
    Fusion  : context vector + sector embedding -> MLP -> added to the decoder's
              initial hidden state (injects day-specific conditioning).
    Decoder : generates n_out steps of [mid_return, log_range]; the range head
              is softplus so range >= 0 and reconstructed high >= low.
    """

    def __init__(
        self,
        n_features:     int = len(FEATURE_COLS),
        n_context:      int = len(CONTEXT_COLS),
        n_sectors:      int = 1,
        sector_emb_dim: int = 8,
        hidden_size:    int = 256,
        num_layers:     int = 3,
        dropout:        float = 0.3,
        n_out:          int = N_OUT,
    ):
        super().__init__()
        self.n_out       = n_out
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.encoder = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = nn.LSTM(
            input_size=2, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.sector_emb = nn.Embedding(max(n_sectors, 1), sector_emb_dim)
        self.ctx_mlp = nn.Sequential(
            nn.Linear(n_context + sector_emb_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_size, 2)   # -> [mid_return, range_raw]

    def forward(self, x, context, sector, target_seq=None, teacher_forcing_ratio=0.5):
        B = x.size(0)
        _, (h, c) = self.encoder(x)

        # ── Fuse static context into the initial decoder hidden state
        emb  = self.sector_emb(sector)                  # (B, e)
        cvec = torch.cat([context, emb], dim=1)         # (B, C+e)
        fuse = self.ctx_mlp(cvec).unsqueeze(0)          # (1, B, hidden)
        h    = h + fuse                                  # broadcast over layers

        outputs   = []
        dec_input = torch.zeros(B, 1, 2, device=x.device)   # SOS

        for t in range(self.n_out):
            out, (h, c) = self.decoder(dec_input, (h, c))
            raw = self.fc(self.drop(self.norm(out)))    # (B, 1, 2)
            m   = raw[..., :1]
            r   = F.softplus(raw[..., 1:2])             # range >= 0
            pred = torch.cat([m, r], dim=-1)            # (B, 1, 2)
            outputs.append(pred)

            if target_seq is not None and torch.rand(1).item() < teacher_forcing_ratio:
                dec_input = target_seq[:, t:t+1, :]
            else:
                dec_input = pred

        return torch.cat(outputs, dim=1)   # (B, n_out, 2) = [mid, range]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — LOSS & METRICS
# ══════════════════════════════════════════════════════════════════════════════

def hl_loss(pred, true, huber, lam_range=LAM_RANGE, lam_cov=LAM_COV,
            tau_hi=TAU_HI, tau_lo=TAU_LO):
    """Huber on mid & range + pinball coverage on reconstructed high/low."""
    m_p, r_p = pred[..., 0], pred[..., 1]
    m_t, r_t = true[..., 0], true[..., 1]

    l_m = huber(m_p, m_t)
    l_r = huber(r_p, r_t)

    hi_p = m_p + 0.5 * r_p; lo_p = m_p - 0.5 * r_p
    hi_t = m_t + 0.5 * r_t; lo_t = m_t - 0.5 * r_t
    e_hi = hi_t - hi_p
    e_lo = lo_t - lo_p
    pin_hi = torch.maximum(tau_hi * e_hi, (tau_hi - 1) * e_hi).mean()
    pin_lo = torch.maximum(tau_lo * e_lo, (tau_lo - 1) * e_lo).mean()

    return l_m + lam_range * l_r + lam_cov * (pin_hi + pin_lo)


@torch.no_grad()
def evaluate_model(model, ds, device, baseline_path):
    """Metrics on a dataset vs a flat mean-cone baseline (in vol units)."""
    model.eval()
    Xs   = ds.X.to(device); cs = ds.ctx.to(device); ss = ds.sec.to(device)
    y    = ds.y.numpy()                                   # (N, n_out, 2)
    pred = model(Xs, cs, ss, teacher_forcing_ratio=0.0).cpu().numpy()

    def metrics(P):
        m_p, r_p = P[..., 0], P[..., 1]
        m_t, r_t = y[..., 0], y[..., 1]
        hi_p = m_p + 0.5 * r_p; lo_p = m_p - 0.5 * r_p
        hi_t = m_t + 0.5 * r_t; lo_t = m_t - 0.5 * r_t
        coverage   = float(np.mean((hi_t <= hi_p) & (lo_t >= lo_p)))
        mae_hi     = float(np.mean(np.abs(hi_t - hi_p)))
        mae_lo     = float(np.mean(np.abs(lo_t - lo_p)))
        dir_acc    = float(np.mean(np.sign(m_p[:, -1]) == np.sign(m_t[:, -1])))
        eod_spread = float(np.std(m_p[:, -1]))            # differentiation proxy
        return {"coverage": coverage, "mae_high": mae_hi, "mae_low": mae_lo,
                "dir_acc": dir_acc, "eod_spread": eod_spread}

    model_m = metrics(pred)
    base_path = np.broadcast_to(baseline_path, pred.shape).copy()
    base_m  = metrics(base_path)

    out = {"model": model_m, "baseline": base_m, "n_val": int(len(y))}
    return out


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def _time_split(d: dict, val_split: float):
    """Chronological holdout: most recent `val_split` of samples -> validation."""
    order = np.argsort(pd.to_datetime(pd.Series(d["days"])).values)
    n_val = max(1, int(len(order) * val_split))
    val_idx   = order[-n_val:]
    train_idx = order[:-n_val]

    def take(idx):
        return ([d["X"][i]   for i in idx],
                [d["ctx"][i] for i in idx],
                [d["sec"][i] for i in idx],
                [d["y"][i]   for i in idx])
    return take(train_idx), take(val_idx)


def train_shared_model(
    d:              dict,
    direction:      str,
    n_sectors:      int,
    sector_vocab:   list,
    epochs:         int   = 30,
    batch_size:     int   = 64,
    lr:             float = 5e-4,
    val_split:      float = 0.15,
    hidden_size:    int   = 256,
    num_layers:     int   = 3,
    dropout:        float = 0.3,
    sector_emb_dim: int   = 8,
    lam_range:      float = LAM_RANGE,
    lam_cov:        float = LAM_COV,
    device:         str   = "cpu",
) -> dict:
    print(f"\n{'='*60}")
    print(f"  Training {direction.upper()} model  |  {len(d['X'])} samples  |  device={device}")
    print(f"{'='*60}")

    if len(d["X"]) < 20:
        print(f"  [WARN] Too few samples ({len(d['X'])}) — skipping {direction}. "
              f"Consider lowering the {direction} threshold.")
        return {}

    (trX, trC, trS, trY), (vaX, vaC, vaS, vaY) = _time_split(d, val_split)

    train_ds = BreakoutDataset(trX, trC, trS, trY)
    val_ds   = BreakoutDataset(vaX, vaC, vaS, vaY,
                               x_scaler=train_ds.x_scaler,
                               ctx_scaler=train_ds.ctx_scaler)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    n_out      = train_ds.y.shape[1]
    n_features = train_ds.X.shape[2]
    n_context  = train_ds.ctx.shape[1]

    model = BreakoutLSTM(
        n_features=n_features, n_context=n_context, n_sectors=n_sectors,
        sector_emb_dim=sector_emb_dim, hidden_size=hidden_size,
        num_layers=num_layers, dropout=dropout, n_out=n_out,
    ).to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=lr, epochs=epochs, steps_per_epoch=max(1, len(train_dl)))
    huber = nn.HuberLoss(delta=0.5)   # targets are in vol units (~O(1))

    best_val = float("inf")
    tf_start, tf_end = 0.8, 0.1

    vaX_t = val_ds.X.to(device); vaC_t = val_ds.ctx.to(device)
    vaS_t = val_ds.sec.to(device); vaY_t = val_ds.y.to(device)

    for epoch in range(1, epochs + 1):
        tf_ratio = tf_start - (tf_start - tf_end) * (epoch / epochs)

        model.train()
        tr_losses = []
        for Xb, Cb, Sb, Yb in train_dl:
            Xb, Cb, Sb, Yb = Xb.to(device), Cb.to(device), Sb.to(device), Yb.to(device)
            optimiser.zero_grad()
            preds = model(Xb, Cb, Sb, target_seq=Yb, teacher_forcing_ratio=tf_ratio)
            loss  = hl_loss(preds, Yb, huber, lam_range, lam_cov)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step(); scheduler.step()
            tr_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            vp = model(vaX_t, vaC_t, vaS_t, teacher_forcing_ratio=0.0)
            vl = hl_loss(vp, vaY_t, huber, lam_range, lam_cov).item()

        tl = float(np.mean(tr_losses)) if tr_losses else float("nan")
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  train={tl:.5f}  val={vl:.5f}  tf={tf_ratio:.2f}")

        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), SAVE_DIR / f"{direction}_lstm.pt")
            joblib.dump(train_ds.x_scaler,   SAVE_DIR / f"{direction}_scaler.pkl")
            joblib.dump(train_ds.ctx_scaler, SAVE_DIR / f"{direction}_ctx_scaler.pkl")
            joblib.dump({
                "version":        META_VERSION,
                "n_out":          n_out,
                "bars_per_day":   BARS_PER_DAY,
                "interval":       INTERVAL,
                "feature_cols":   FEATURE_COLS,
                "context_cols":   CONTEXT_COLS,
                "n_features":     n_features,
                "n_context":      n_context,
                "n_sectors":      n_sectors,
                "sector_vocab":   sector_vocab,
                "sector_emb_dim": sector_emb_dim,
                "hidden_size":    hidden_size,
                "num_layers":     num_layers,
                "direction":      direction,
                "seq_len":        SEQ_LEN,
                "vol_floor":      VOL_FLOOR,
                "target_norm":    "atr",
                "target_param":   "mid_logrange",
            }, SAVE_DIR / f"{direction}_meta.pkl")

    # ── Reload best, evaluate vs baseline, persist metrics
    model.load_state_dict(torch.load(SAVE_DIR / f"{direction}_lstm.pt", map_location=device))
    baseline_path = train_ds.y.numpy().mean(axis=0)        # (n_out, 2) mean cone
    evald = evaluate_model(model, val_ds, device, baseline_path)
    (SAVE_DIR / f"{direction}_eval.json").write_text(json.dumps(evald, indent=2))

    m, b = evald["model"], evald["baseline"]
    print(f"\n  ✓ {direction} saved — best val loss: {best_val:.5f}")
    print(f"    coverage  model {m['coverage']:.2%}  vs baseline {b['coverage']:.2%}")
    print(f"    dir-acc   model {m['dir_acc']:.2%}  vs baseline {b['dir_acc']:.2%}")
    print(f"    MAE(hi)   model {m['mae_high']:.3f}  vs baseline {b['mae_high']:.3f} (vol units)")
    print(f"    eod spread model {m['eod_spread']:.3f}  vs baseline {b['eod_spread']:.3f} (differentiation)")
    return {"direction": direction, "best_val_loss": best_val, "eval": evald}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def load_shared_model(direction: str, device: str = "cpu"):
    meta_path   = SAVE_DIR / f"{direction}_meta.pkl"
    model_path  = SAVE_DIR / f"{direction}_lstm.pt"
    scaler_path = SAVE_DIR / f"{direction}_scaler.pkl"
    ctx_path    = SAVE_DIR / f"{direction}_ctx_scaler.pkl"

    if not all(p.exists() for p in [meta_path, model_path, scaler_path, ctx_path]):
        raise FileNotFoundError(f"Model files missing for '{direction}'. Run training first.")

    meta = joblib.load(meta_path)
    if meta.get("version", 1) < META_VERSION or "context_cols" not in meta:
        raise ValueError(
            f"'{direction}' model is an old-format checkpoint. "
            f"Retrain with:  python lstm_train_historical.py"
        )

    x_scaler   = joblib.load(scaler_path)
    ctx_scaler = joblib.load(ctx_path)

    model = BreakoutLSTM(
        n_features=meta["n_features"], n_context=meta["n_context"],
        n_sectors=meta["n_sectors"], sector_emb_dim=meta["sector_emb_dim"],
        hidden_size=meta["hidden_size"], num_layers=meta["num_layers"],
        n_out=meta["n_out"],
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, x_scaler, ctx_scaler, meta


def predict_next_day_shared(
    ticker:    str,
    direction: str,
    device:    str = "cpu",
) -> pd.DataFrame | None:
    """
    Forecast the remaining 15-min high/low path of the most recent trading day
    that already has an *opening* 15-min bar, conditioning on that opening bar,
    the static context vector, and the ticker's sector embedding.

    Behaviour (unchanged from the prior version):
      • Target day = latest day with a printed opening bar; its prior day is the
        signal day. Mid-session runs forecast the rest of today (realised bars
        returned as actuals); after-close runs show the completed day and do NOT
        roll forward to a day whose opening bar doesn't exist yet.

    Returns a DataFrame indexed by 15-min timestamps with columns:
        predicted_high, predicted_low, predicted_mid,
        actual_high, actual_low, high_return, low_return, is_actual
    df.attrs carries: target_day, open_close, open_time, n_realized, complete.
    """
    model, x_scaler, ctx_scaler, meta = load_shared_model(direction, device)
    seq_len  = meta["seq_len"]
    n_out    = meta["n_out"]
    interval = meta.get("interval", INTERVAL)
    vocab    = meta.get("sector_vocab", ["Unknown"])

    # refresh=True tops up the current year for this ticker before reading, so
    # inference sees the same source the model trained on rather than yfinance.
    raw = massive_feed.download(ticker, period="1mo", interval=interval,
                                auto_adjust=True, progress=False, refresh=True)
    daily = massive_feed.download(ticker, period="1y", interval="1d",
                                  auto_adjust=True, progress=False)
    if raw.empty or daily.empty:
        return None
    for d in (raw, daily):
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
    raw.index   = pd.to_datetime(raw.index).tz_localize(None)
    daily.index = pd.to_datetime(daily.index).tz_localize(None)

    feat = compute_features(raw)
    if feat.empty:
        return None

    feat_days = sorted(feat.index.normalize().unique())
    if not feat_days:
        return None
    target_day = feat_days[-1]

    tgt_raw = raw[raw.index.normalize() == target_day]
    if tgt_raw.empty:
        return None
    open_ts    = tgt_raw.index[0]
    open_close = float(tgt_raw["Close"].iloc[0])
    if open_close <= 0:
        return None

    hist_feat = feat[feat.index <= open_ts]
    if len(hist_feat) < seq_len or hist_feat.index[-1] != open_ts:
        return None

    # ── Static context (must mirror training)
    daily_slice = daily[daily.index.normalize() < target_day]
    if len(daily_slice) < 80:
        return None
    _, comps  = score_daily_slice(daily_slice, return_components=True)
    sigma     = vol_scale(daily_slice)
    prev_close = float(daily_slice["Close"].iloc[-1])
    avg_vol_15 = max(float(raw["Volume"].rolling(20).mean().mean()), 1.0)
    open_vol_ratio = float(tgt_raw["Volume"].iloc[0]) / avg_vol_15
    ctx = build_context(prev_close, tgt_raw.iloc[0], open_vol_ratio, comps)

    sec_id = sector_to_idx(
        get_sector_map([ticker], fetch_missing=False).get(ticker, "Unknown"), vocab)

    X_scaled   = x_scaler.transform(
        hist_feat[FEATURE_COLS].values[-seq_len:].astype(np.float32))
    ctx_scaled = ctx_scaler.transform(ctx.reshape(1, -1))

    X_t   = torch.tensor(X_scaled,   dtype=torch.float32).unsqueeze(0).to(device)
    C_t   = torch.tensor(ctx_scaled, dtype=torch.float32).to(device)
    S_t   = torch.tensor([sec_id],   dtype=torch.long).to(device)

    with torch.no_grad():
        preds = model(X_t, C_t, S_t, teacher_forcing_ratio=0.0).squeeze(0).cpu().numpy()

    # ── Reconstruct prices from ATR-normalised (mid, range)
    m = preds[:, 0] * sigma
    r = np.clip(preds[:, 1], 0, None) * sigma     # range >= 0
    log_high = m + 0.5 * r
    log_low  = m - 0.5 * r
    pred_highs = open_close * np.exp(log_high)
    pred_lows  = open_close * np.exp(log_low)
    pred_mids  = open_close * np.exp(m)

    pd_freq    = interval.replace("m", "min") if interval.endswith("m") else interval
    pred_times = pd.date_range(open_ts, periods=n_out + 1, freq=pd_freq)[1:]

    realized    = tgt_raw.reindex(pred_times)
    actual_high = realized["High"].values
    actual_low  = realized["Low"].values
    is_actual   = ~np.isnan(actual_high)
    n_realized  = int(is_actual.sum())

    df = pd.DataFrame({
        "timestamp":      pred_times,
        "predicted_high": pred_highs,
        "predicted_low":  pred_lows,
        "predicted_mid":  pred_mids,
        "actual_high":    actual_high,
        "actual_low":     actual_low,
        "high_return":    log_high,
        "low_return":     log_low,
        "is_actual":      is_actual,
    }).set_index("timestamp")

    df.attrs["target_day"] = pd.Timestamp(target_day)
    df.attrs["open_close"] = open_close
    df.attrs["open_time"]  = open_ts
    df.attrs["sigma"]      = float(sigma)
    df.attrs["n_realized"] = n_realized
    df.attrs["complete"]   = n_realized >= n_out
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train shared bullish/bearish LSTM (15-min, opening-bar conditioned, ATR-normalised)")
    parser.add_argument("--bullish-threshold", type=int,   default=35)
    parser.add_argument("--bearish-threshold", type=int,   default=25)
    parser.add_argument("--epochs",            type=int,   default=30)
    parser.add_argument("--batch-size",        type=int,   default=64)
    parser.add_argument("--hidden-size",       type=int,   default=256)
    parser.add_argument("--num-layers",        type=int,   default=3)
    parser.add_argument("--sector-emb-dim",    type=int,   default=8)
    parser.add_argument("--lam-range",         type=float, default=LAM_RANGE)
    parser.add_argument("--lam-cov",           type=float, default=LAM_COV)
    parser.add_argument("--tickers",           nargs="+",  default=None)
    parser.add_argument("--intraday-period",   type=str,   default=INTRADAY_PERIOD,
                        help="intraday history window, e.g. 60d / 1y / 2y / 5y "
                             "(store holds 5y; longer = more samples, slower build)")
    parser.add_argument("--bullish-only",      action="store_true")
    parser.add_argument("--bearish-only",      action="store_true")
    args = parser.parse_args()

    INTRADAY_PERIOD = args.intraday_period

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\n  Device: {device}")

    tickers = args.tickers or SP500_TICKERS

    sector_map   = get_sector_map(tickers, fetch_missing=True)
    sector_vocab = build_sector_vocab(sector_map)
    n_sectors    = len(sector_vocab)
    print(f"  Sectors: {n_sectors} ({', '.join(sector_vocab[:8])}{' ...' if n_sectors>8 else ''})")

    data = build_training_dataset(
        tickers           = tickers,
        bullish_threshold = args.bullish_threshold,
        bearish_threshold = args.bearish_threshold,
        sector_map        = sector_map,
        sector_vocab      = sector_vocab,
    )

    results = []
    common  = dict(
        n_sectors=n_sectors, sector_vocab=sector_vocab,
        epochs=args.epochs, batch_size=args.batch_size,
        hidden_size=args.hidden_size, num_layers=args.num_layers,
        sector_emb_dim=args.sector_emb_dim,
        lam_range=args.lam_range, lam_cov=args.lam_cov, device=device,
    )

    if not args.bearish_only and data["bullish"]["X"]:
        results.append(train_shared_model(data["bullish"], "bullish", **common))
    if not args.bullish_only and data["bearish"]["X"]:
        results.append(train_shared_model(data["bearish"], "bearish", **common))

    print(f"\n{'='*60}")
    print("  TRAINING COMPLETE")
    for r in results:
        if r:
            m = r["eval"]["model"]
            print(f"  {r['direction']:10s}  val_loss={r['best_val_loss']:.5f}  "
                  f"coverage={m['coverage']:.2%}  dir_acc={m['dir_acc']:.2%}")
    print(f"  Models saved to: {SAVE_DIR.resolve()}")
    print(f"{'='*60}\n")
