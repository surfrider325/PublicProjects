"""
lstm_breakout_predictor.py
==========================
Minute-by-minute next-day price prediction using a PyTorch LSTM,
restricted to stocks currently identified as breakouts by the S&P 500
Breakout Detector (breakout_detector.py).

Usage:
    python lstm_breakout_predictor.py                        # full run
    python lstm_breakout_predictor.py --min-score 60        # only strong breakouts
    python lstm_breakout_predictor.py --tickers AAPL NVDA   # override with manual list
    python lstm_breakout_predictor.py --train-only           # train without predicting
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# Local Massive Parquet store; yfinance-compatible download() with automatic
# fallback to yfinance for tickers the store doesn't hold.
import massive_feed
import joblib

warnings.filterwarnings("ignore")

# 1-min history window for training. yfinance capped this at 60d; the store has 5y.
MINUTE_PERIOD = "1y"

# ──────────────────────────────────────────────────────────────────────────────
# OPTIONAL: Import your breakout detector's core analysis function.
# If breakout_detector.py is in the same directory, the import below works.
# The screener_runner() function below is the bridge between the two apps.
# ──────────────────────────────────────────────────────────────────────────────
try:
    from breakout_detector import analyze_ticker, SP500_TICKERS  # type: ignore
    BREAKOUT_APP_AVAILABLE = True
except ImportError:
    BREAKOUT_APP_AVAILABLE = False
    print("[WARN] breakout_detector.py not found. Use --tickers to supply tickers manually.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — BREAKOUT SCREENER BRIDGE
# ══════════════════════════════════════════════════════════════════════════════

def get_breakout_tickers(min_score: int = 55, max_tickers: int = 50) -> list[str]:
    """
    Runs the breakout detector screener and returns tickers whose composite
    score meets the minimum threshold. This is the integration point between
    the two applications.

    Args:
        min_score:   Minimum composite breakout score (0–100).
        max_tickers: Cap on how many tickers to scan (for speed).

    Returns:
        List of ticker strings that are currently in a breakout formation.
    """
    if not BREAKOUT_APP_AVAILABLE:
        raise RuntimeError(
            "breakout_detector.py not importable. "
            "Pass --tickers explicitly or ensure breakout_detector.py is in the same folder."
        )

    print(f"\n{'='*60}")
    print(f"  STEP 1 — Running Breakout Screener (min_score={min_score})")
    print(f"{'='*60}")

    tickers_to_scan = SP500_TICKERS[:max_tickers]
    breakout_tickers = []

    for i, ticker in enumerate(tickers_to_scan, 1):
        try:
            result = analyze_ticker(ticker, lookback_weeks=20)
            if result and result["composite"]["score"] >= min_score:
                score = result["composite"]["score"]
                label = result["composite"]["label"]
                print(f"  ✓ {ticker:6s}  score={score:3d}  [{label}]")
                breakout_tickers.append(ticker)
        except Exception as e:
            pass
        time.sleep(0.05)

        if (i % 10) == 0:
            print(f"  ... scanned {i}/{len(tickers_to_scan)}")

    print(f"\n  → {len(breakout_tickers)} breakout candidates found.")
    return breakout_tickers


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers features from raw 1-minute OHLCV data.
    All features are computed from past data only (no look-ahead).
    """
    df = df.copy()
    c = df["Close"]
    v = df["Volume"]

    # ── Returns
    df["ret_1"]  = c.pct_change(1)
    df["ret_5"]  = c.pct_change(5)
    df["ret_15"] = c.pct_change(15)
    df["ret_30"] = c.pct_change(30)

    # ── VWAP (rolling 30-bar proxy)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    df["vwap_30"] = (tp * v).rolling(30).sum() / v.rolling(30).sum()
    df["vwap_delta"] = (c - df["vwap_30"]) / df["vwap_30"]

    # ── Volume features
    df["vol_ratio_20"] = v / v.rolling(20).mean().clip(lower=1)

    # ── RSI (14-bar)
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # ── MACD (12/26/9)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    df["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()

    # ── Bollinger Band position
    mid  = c.rolling(20).mean()
    std  = c.rolling(20).std()
    df["bb_pos"] = (c - mid) / (2 * std + 1e-9)   # -1 to +1 roughly

    # ── ATR (14-bar normalised)
    hl   = df["High"] - df["Low"]
    hc   = (df["High"] - c.shift()).abs()
    lc   = (df["Low"]  - c.shift()).abs()
    atr  = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df["atr_norm"] = atr / c

    # ── Time-of-day features (cyclical encoding)
    # Works for any timezone — uses minute index within the trading day
    if hasattr(df.index, "hour"):
        mins = df.index.hour * 60 + df.index.minute
        day_min  = (mins - 9*60 - 30)          # minutes since open
        day_frac = day_min / 390               # 0 = open, 1 = close
        df["tod_sin"] = np.sin(2 * np.pi * day_frac)
        df["tod_cos"] = np.cos(2 * np.pi * day_frac)
    else:
        df["tod_sin"] = 0.0
        df["tod_cos"] = 1.0

    # ── Day of week (cyclical)
    if hasattr(df.index, "dayofweek"):
        dow = df.index.dayofweek / 4
        df["dow_sin"] = np.sin(2 * np.pi * dow)
        df["dow_cos"] = np.cos(2 * np.pi * dow)
    else:
        df["dow_sin"] = 0.0
        df["dow_cos"] = 1.0

    # ── Target: 1-minute forward log return
    df["target"] = np.log(c.shift(-1) / c)

    # Drop rows with NaNs (startup period for rolling features)
    df.dropna(inplace=True)
    return df


FEATURE_COLS = [
    "ret_1", "ret_5", "ret_15", "ret_30",
    "vwap_delta", "vol_ratio_20",
    "rsi_14", "macd_hist", "bb_pos", "atr_norm",
    "tod_sin", "tod_cos", "dow_sin", "dow_cos",
]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DATASET
# ══════════════════════════════════════════════════════════════════════════════

class MinuteDataset(Dataset):
    """
    Sliding-window dataset for sequence-to-one prediction.
    Each sample is a window of `seq_len` feature vectors → 1 target return.
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int = 60):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets,  dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x = self.X[idx : idx + self.seq_len]          # (seq_len, n_features)
        y = self.y[idx + self.seq_len]                 # scalar
        return x, y


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MODEL
# ══════════════════════════════════════════════════════════════════════════════

class StockLSTM(nn.Module):
    """
    Stacked LSTM with:
    - Variational (locked) dropout between layers
    - LayerNorm on the final hidden state
    - Linear head → predicted 1-min log return
    """

    def __init__(
        self,
        n_features:  int   = len(FEATURE_COLS),
        hidden_size: int   = 128,
        num_layers:  int   = 2,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm   = nn.LayerNorm(hidden_size)
        self.drop   = nn.Dropout(dropout)
        self.fc     = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)                  # (batch, seq_len, hidden)
        h      = out[:, -1, :]                 # last time-step
        h      = self.drop(self.norm(h))
        return self.fc(h).squeeze(-1)          # (batch,)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
    ticker:       str,
    seq_len:      int   = 60,
    hidden_size:  int   = 128,
    num_layers:   int   = 2,
    dropout:      float = 0.3,
    lr:           float = 1e-3,
    epochs:       int   = 20,
    batch_size:   int   = 256,
    val_days:     int   = 5,
    save_dir:     str   = "models",
    device:       str   = "cpu",
) -> dict:
    """
    Downloads 60 days of 1-min data, engineers features, trains the LSTM with
    a strict walk-forward train/val split, and saves the model + scaler.

    Returns a dict with training history and best validation loss.
    """
    print(f"\n  [{ticker}] Loading 1-min data …")
    # Was yfinance period="60d" (its hard cap on 1-min bars). The local store
    # holds 5 years at this resolution, so MINUTE_PERIOD is a real choice now.
    raw = massive_feed.download(ticker, interval="1m", period=MINUTE_PERIOD, progress=False)
    if raw.empty or len(raw) < 500:
        print(f"  [{ticker}] Insufficient data — skipping.")
        return {}

    # Flatten multi-index columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = compute_features(raw)

    # ── Walk-forward split: last `val_days` trading days → validation
    trading_days = df.index.normalize().unique()
    if len(trading_days) < val_days + 5:
        print(f"  [{ticker}] Not enough trading days — skipping.")
        return {}

    cutoff = trading_days[-val_days]
    train_df = df[df.index.normalize() < cutoff]
    val_df   = df[df.index.normalize() >= cutoff]

    # ── Feature scaling (fit on train only)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_COLS].values)
    X_val   = scaler.transform(val_df[FEATURE_COLS].values)
    y_train = train_df["target"].values.astype(np.float32)
    y_val   = val_df["target"].values.astype(np.float32)

    train_ds = MinuteDataset(X_train, y_train, seq_len)
    val_ds   = MinuteDataset(X_val,   y_val,   seq_len)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # ── Model, optimiser, scheduler
    model = StockLSTM(
        n_features=len(FEATURE_COLS),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    criterion = nn.HuberLoss(delta=0.01)   # robust to outlier spikes

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    print(f"  [{ticker}] Training: {len(train_df)} train bars | {len(val_df)} val bars")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_losses = []
        for X_batch, y_batch in train_dl:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimiser.zero_grad()
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_dl:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                val_losses.append(criterion(preds, y_batch).item())

        tl = np.mean(train_losses)
        vl = np.mean(val_losses)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{epochs}  train={tl:.6f}  val={vl:.6f}")

        # Save best model
        if vl < best_val_loss:
            best_val_loss = vl
            Path(save_dir).mkdir(exist_ok=True)
            torch.save(model.state_dict(), f"{save_dir}/{ticker}_lstm.pt")
            joblib.dump(scaler, f"{save_dir}/{ticker}_scaler.pkl")

    print(f"  [{ticker}] Best val loss: {best_val_loss:.6f}")
    return {"ticker": ticker, "best_val_loss": best_val_loss, "history": history}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PREDICTION (next trading day, minute by minute)
# ══════════════════════════════════════════════════════════════════════════════

def predict_next_day(
    ticker:      str,
    seq_len:     int  = 60,
    hidden_size: int  = 128,
    num_layers:  int  = 2,
    dropout:     float = 0.3,
    save_dir:    str  = "models",
    device:      str  = "cpu",
) -> pd.DataFrame | None:
    """
    Loads the saved model + scaler for `ticker`, takes the most recent
    `seq_len` minutes of live data, and autoregressively predicts the full
    next trading day (390 minutes).

    Returns a DataFrame with columns: [minute, predicted_price, predicted_return]
    Indices are the predicted minute timestamps.
    """
    model_path  = f"{save_dir}/{ticker}_lstm.pt"
    scaler_path = f"{save_dir}/{ticker}_scaler.pkl"

    if not Path(model_path).exists():
        print(f"  [{ticker}] No saved model found. Run training first.")
        return None

    scaler = joblib.load(scaler_path)

    model = StockLSTM(
        n_features=len(FEATURE_COLS),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ── Fetch the most recent days of 1-min data for the seed window.
    # refresh=True tops up today's bars so inference matches the training source.
    raw = massive_feed.download(ticker, interval="1m", period="5d",
                                progress=False, refresh=True)
    if raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = compute_features(raw)
    if len(df) < seq_len:
        return None

    # ── Use last seq_len bars as the initial context
    seed_features = scaler.transform(df[FEATURE_COLS].values[-seq_len:])  # (seq_len, F)
    last_price    = float(df["Close"].iloc[-1])
    last_time     = df.index[-1]

    # ── Build next-day minute timestamps (390 minutes, 09:30–16:00)
    next_open = pd.Timestamp(last_time.date()) + pd.Timedelta(days=1)
    while next_open.weekday() >= 5:          # skip weekends
        next_open += pd.Timedelta(days=1)
    next_open = next_open.replace(hour=9, minute=30, second=0)
    pred_times = pd.date_range(next_open, periods=390, freq="1min")

    # ── Autoregressive rollout
    window = seed_features.tolist()   # list of feature vectors, length seq_len
    predicted_returns = []
    predicted_prices  = [last_price]
    current_price     = last_price

    with torch.no_grad():
        for i, ts in enumerate(pred_times):
            x  = torch.tensor([window[-seq_len:]], dtype=torch.float32).to(device)
            log_ret = model(x).item()                     # predicted log return
            predicted_returns.append(log_ret)

            # Step forward price
            current_price = current_price * np.exp(log_ret)
            predicted_prices.append(current_price)

            # ── Construct approximate next feature vector
            # We propagate returns and use neutral/zero values for volume-based
            # features since we can't observe them in the future.
            # Time-of-day features are deterministic and computed exactly.
            ret_1  = log_ret
            # Use expanding average of predicted returns for longer-horizon lags
            hist   = predicted_returns
            ret_5  = float(np.sum(hist[-5:]))  if len(hist) >= 5  else ret_1
            ret_15 = float(np.sum(hist[-15:])) if len(hist) >= 15 else ret_1
            ret_30 = float(np.sum(hist[-30:])) if len(hist) >= 30 else ret_1

            mins    = ts.hour * 60 + ts.minute
            day_min = (mins - 9*60 - 30) / 390
            tod_sin = np.sin(2 * np.pi * day_min)
            tod_cos = np.cos(2 * np.pi * day_min)
            dow_sin = np.sin(2 * np.pi * ts.dayofweek / 4)
            dow_cos = np.cos(2 * np.pi * ts.dayofweek / 4)

            # Neutral values for unknown future features
            new_feat_raw = np.array([[
                ret_1, ret_5, ret_15, ret_30,
                0.0,   1.0,           # vwap_delta, vol_ratio (neutral)
                50.0,  0.0,   0.0, 0.005,  # rsi, macd_hist, bb_pos, atr_norm
                tod_sin, tod_cos, dow_sin, dow_cos,
            ]], dtype=np.float32)

            new_feat_scaled = scaler.transform(new_feat_raw)[0].tolist()
            window.append(new_feat_scaled)

    results = pd.DataFrame({
        "timestamp":        pred_times,
        "predicted_price":  predicted_prices[1:],   # drop the seed price
        "predicted_return": predicted_returns,
    }).set_index("timestamp")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    tickers:    list[str] | None = None,
    min_score:  int  = 55,
    max_scan:   int  = 50,
    epochs:     int  = 20,
    seq_len:    int  = 60,
    save_dir:   str  = "models",
    train_only: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n  Device: {device}")

    # ── Step 1: Get candidate tickers
    if tickers:
        candidates = tickers
        print(f"\n  Using manual ticker list: {candidates}")
    else:
        candidates = get_breakout_tickers(min_score=min_score, max_tickers=max_scan)

    if not candidates:
        print("\n  No breakout candidates found. Try lowering --min-score.")
        return

    # ── Step 2: Train a model per ticker
    print(f"\n{'='*60}")
    print(f"  STEP 2 — Training LSTM models ({len(candidates)} tickers)")
    print(f"{'='*60}")

    train_results = []
    for ticker in candidates:
        result = train_model(
            ticker, seq_len=seq_len, epochs=epochs,
            save_dir=save_dir, device=device
        )
        if result:
            train_results.append(result)

    # ── Step 3: Predict next day
    if train_only:
        print("\n  --train-only flag set. Skipping prediction step.")
        return

    print(f"\n{'='*60}")
    print(f"  STEP 3 — Predicting Next Trading Day")
    print(f"{'='*60}")

    all_predictions = {}
    for ticker in candidates:
        preds = predict_next_day(ticker, seq_len=seq_len, save_dir=save_dir, device=device)
        if preds is not None:
            all_predictions[ticker] = preds
            open_p  = preds["predicted_price"].iloc[0]
            close_p = preds["predicted_price"].iloc[-1]
            chg     = (close_p / open_p - 1) * 100
            print(f"  {ticker:6s}  open={open_p:.2f}  close={close_p:.2f}  predicted_move={chg:+.2f}%")

    # ── Save predictions to CSV
    Path(save_dir).mkdir(exist_ok=True)
    for ticker, df in all_predictions.items():
        df.to_csv(f"{save_dir}/{ticker}_predictions.csv")
    print(f"\n  Predictions saved to {save_dir}/<ticker>_predictions.csv")

    return all_predictions


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM Next-Day Minute Predictor (Breakout Stocks)")
    parser.add_argument("--tickers",    nargs="+", default=None,  help="Override with specific tickers")
    parser.add_argument("--min-score",  type=int,  default=55,    help="Min composite breakout score")
    parser.add_argument("--max-scan",   type=int,  default=50,    help="Max tickers to scan in screener")
    parser.add_argument("--epochs",     type=int,  default=20,    help="Training epochs per ticker")
    parser.add_argument("--seq-len",    type=int,  default=60,    help="LSTM input sequence length (minutes)")
    parser.add_argument("--save-dir",   type=str,  default="models", help="Directory for models/predictions")
    parser.add_argument("--train-only", action="store_true",      help="Train models without predicting")
    args = parser.parse_args()

    run_pipeline(
        tickers=args.tickers,
        min_score=args.min_score,
        max_scan=args.max_scan,
        epochs=args.epochs,
        seq_len=args.seq_len,
        save_dir=args.save_dir,
        train_only=args.train_only,
    )
