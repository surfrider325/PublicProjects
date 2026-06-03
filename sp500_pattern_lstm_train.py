#!/usr/bin/env python3
"""
sp500_pattern_lstm_train.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
S&P 500 Technical Pattern LSTM â€” Training Script
Detects 8 chart formations via a Bidirectional LSTM + temporal attention model.

Patterns: Head & Shoulders, Inv Head & Shoulders, Double Top, Double Bottom,
          Rising Wedge, Falling Wedge, Bull Flag, Bear Flag

Final results (60 epochs, 501 stocks, 80k windows):
  Macro F1: 35.7%  |  Weighted F1: 38.7%
  Best classes: Double Bottom 46.7%, Double Top 42.0%, Bull Flag 39.0%

Usage:
  python sp500_pattern_lstm_train.py              # full train
  python sp500_pattern_lstm_train.py --resume     # resume from checkpoint
  python sp500_pattern_lstm_train.py --eval       # eval only (loads best weights)
  python sp500_pattern_lstm_train.py --epochs 3   # run N epochs then stop
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
"""

import argparse, os, json, time, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, precision_score, recall_score

warnings.filterwarnings("ignore")

# â”€â”€ CONFIG â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
PATTERN_NAMES = [
    "Head & Shoulders", "Inv Head & Shoulders",
    "Double Top",       "Double Bottom",
    "Rising Wedge",     "Falling Wedge",
    "Bull Flag",        "Bear Flag",
]
N_PAT        = len(PATTERN_NAMES)
WINDOW       = 30          # lookback days
TOTAL_EPOCHS = 60
BATCH_SIZE   = 1024
LR           = 3e-4
WEIGHT_DECAY = 1e-4
FOCAL_GAMMA  = 2.5
POS_OVERSAMPLE = 5.0       # weight for positive windows in sampler
TARGET_WINDOWS = 80_000    # subsample target (keeps all positives)
CKPT_PATH    = "lstm_checkpoint.pt"
MODEL_PATH   = "sp500_pattern_lstm.pt"


# â”€â”€ DATA â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def fetch_sp500_tickers():
    """Scrape S&P 500 constituent list from Wikipedia."""
    import requests
    from bs4 import BeautifulSoup
    resp = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "Mozilla/5.0"}, timeout=20,
    )
    soup  = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"id": "constituents"})
    tickers = []
    for row in table.find_all("tr")[1:]:
        cols = row.find_all("td")
        if cols:
            tickers.append(cols[0].text.strip().replace(".", "-"))
    print(f"Scraped {len(tickers)} S&P 500 tickers")
    return tickers


def download_prices(tickers, years=2):
    """Batch-download OHLCV data via yfinance."""
    import yfinance as yf
    from datetime import datetime, timedelta
    end   = datetime.today()
    start = end - timedelta(days=int(years * 365))
    raw, failed = {}, []
    for i in range(0, len(tickers), 50):
        batch = tickers[i : i + 50]
        try:
            df_b = yf.download(
                batch, start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True, progress=False, group_by="ticker",
            )
            for t in batch:
                try:
                    df_t = df_b[t].dropna(how="all") if len(batch) > 1 else df_b.copy()
                    if len(df_t) >= 300 and "Close" in df_t.columns:
                        raw[t] = df_t
                    else:
                        failed.append(t)
                except Exception:
                    failed.append(t)
        except Exception:
            failed.extend(batch)
        print(f"  Batch {i//50+1}/{(len(tickers)-1)//50+1} | Loaded: {len(raw)}", end="\r")
    print(f"\nLoaded: {len(raw)} stocks | Failed: {len(failed)}")
    return raw


# â”€â”€ PATTERN DETECTION â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def find_extrema(arr, w=5):
    peaks, troughs = [], []
    for i in range(w, len(arr) - w):
        if arr[i] == max(arr[i - w : i + w + 1]):
            peaks.append(i)
        if arr[i] == min(arr[i - w : i + w + 1]):
            troughs.append(i)
    return peaks, troughs


def detect_patterns(closes, highs, lows, n):
    """Returns dict {pattern_idx: [(start_idx, end_idx), ...]}"""
    pk, tr = find_extrema(closes)
    found  = {i: [] for i in range(N_PAT)}

    # Head & Shoulders (bearish)
    for i in range(len(pk) - 2):
        ls, h, rs = pk[i], pk[i + 1], pk[i + 2]
        if (closes[h] > closes[ls] and closes[h] > closes[rs]
                and abs(closes[ls] - closes[rs]) / closes[h] < 0.05):
            t1 = [t for t in tr if ls < t < h]
            t2 = [t for t in tr if h  < t < rs]
            if t1 and t2:
                found[0].append((ls, rs))

    # Inverse Head & Shoulders (bullish)
    for i in range(len(tr) - 2):
        ls, h, rs = tr[i], tr[i + 1], tr[i + 2]
        if (closes[h] < closes[ls] and closes[h] < closes[rs]
                and abs(closes[ls] - closes[rs]) / closes[h] < 0.05):
            p1 = [p for p in pk if ls < p < h]
            p2 = [p for p in pk if h  < p < rs]
            if p1 and p2:
                found[1].append((ls, rs))

    # Double Top / Double Bottom
    for i in range(len(pk) - 1):
        p1, p2 = pk[i], pk[i + 1]
        if abs(closes[p1] - closes[p2]) / closes[p1] < 0.03:
            found[2].append((p1, p2))
    for i in range(len(tr) - 1):
        t1, t2 = tr[i], tr[i + 1]
        if abs(closes[t1] - closes[t2]) / closes[t1] < 0.03:
            found[3].append((t1, t2))

    # Rising / Falling Wedge
    W = 20
    for s in range(0, n - W, 5):
        e  = s + W
        x  = np.arange(W)
        hs = np.polyfit(x, highs[s:e], 1)[0]
        ls = np.polyfit(x, lows[s:e],  1)[0]
        if hs > 0.05 and ls > 0.05 and ls > hs * 1.1:
            found[4].append((s, e))          # rising wedge (bearish)
        if hs < -0.05 and ls < -0.05 and hs < ls * 1.1:
            found[5].append((s, e))          # falling wedge (bullish)

    # Bull / Bear Flag
    PW, FW = 8, 12
    for s in range(0, n - PW - FW, 5):
        pe  = s + PW
        fe  = pe + FW
        mv  = (closes[pe - 1] - closes[s]) / closes[s]
        fl  = closes[pe:fe]
        fr  = (fl.max() - fl.min()) / fl.mean()
        if  mv >  0.03 and fr < 0.05:
            found[6].append((s, fe))         # bull flag
        elif mv < -0.03 and fr < 0.05:
            found[7].append((s, fe))         # bear flag

    return found


def make_features(df, s, e):
    """Normalised OHLCV window â†’ shape (window, 5)."""
    c = df["Close"].values[s:e]
    o = df["Open"].values[s:e]
    h = df["High"].values[s:e]
    l = df["Low"].values[s:e]
    v = df["Volume"].values[s:e].astype(float)
    base = c[0] if c[0] != 0 else 1.0
    vm   = v.mean() if v.mean() != 0 else 1.0
    return np.stack(
        [o / base - 1, h / base - 1, l / base - 1, c / base - 1, v / vm - 1],
        axis=1,
    ).astype(np.float32)


def build_dataset(raw, window=WINDOW, target=TARGET_WINDOWS, seed=42):
    """
    Slide a 30-day window over every stock, label end-of-window day,
    then subsample: keep ALL positives + random negatives up to `target`.
    """
    np.random.seed(seed)
    X_list, y_list = [], []
    for df in raw.values():
        c = df["Close"].values
        h = df["High"].values
        l = df["Low"].values
        n = len(df)
        pats     = detect_patterns(c, h, l, n)
        day_lbl  = np.zeros((n, N_PAT), dtype=np.float32)
        for pi, inst in pats.items():
            for s, e in inst:
                if e < n:
                    day_lbl[e, pi] = 1.0
        for ws in range(0, n - window, 1):
            feat = make_features(df, ws, ws + window)
            if not (np.isnan(feat).any() or np.isinf(feat).any()):
                X_list.append(feat)
                y_list.append(day_lbl[ws + window - 1])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    pos_idx = np.where(y.sum(axis=1) > 0)[0]
    neg_idx = np.where(y.sum(axis=1) == 0)[0]
    n_neg   = min(len(neg_idx), target - len(pos_idx))
    keep    = np.sort(np.concatenate([pos_idx,
                                      np.random.choice(neg_idx, n_neg, replace=False)]))
    print(f"Dataset: {len(keep):,} windows  ({len(pos_idx):,} pos + {n_neg:,} neg)")
    return X[keep], y[keep]


# â”€â”€ MODEL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class PatternLSTM(nn.Module):
    def __init__(self, in_=5, hid=192, layers=3, drop=0.35, n_cls=8):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_, 48), nn.LayerNorm(48), nn.GELU(), nn.Dropout(drop * 0.4),
        )
        self.lstm = nn.LSTM(
            48, hid, layers, batch_first=True, dropout=drop, bidirectional=True,
        )
        self.attn = nn.Sequential(
            nn.Linear(hid * 2, 96), nn.Tanh(), nn.Linear(96, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hid * 2), nn.Dropout(drop),
            nn.Linear(hid * 2, 256), nn.GELU(),
            nn.Dropout(drop * 0.5), nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, n_cls),
        )

    def forward(self, x):
        x       = self.proj(x)
        out, _  = self.lstm(x)
        attn_w  = torch.softmax(self.attn(out), dim=1)
        context = (out * attn_w).sum(dim=1)
        return self.head(context)


# â”€â”€ LOSS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class FocalLoss(nn.Module):
    def __init__(self, gamma=FOCAL_GAMMA, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pw    = pos_weight

    def forward(self, logits, targets):
        bce  = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pw, reduction="none"
        )
        p    = torch.sigmoid(logits)
        p_t  = p * targets + (1 - p) * (1 - targets)
        return ((1 - p_t) ** self.gamma * bce).mean()


# â”€â”€ EVAL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def evaluate(loader, model, criterion, device):
    model.eval()
    lgs, lbs = [], []
    with torch.no_grad():
        for xb, yb in loader:
            lgs.append(model(xb.to(device)).cpu())
            lbs.append(yb)
    lg = torch.cat(lgs); lb = torch.cat(lbs)
    loss  = criterion(lg.to(device), lb.to(device)).item()
    probs = torch.sigmoid(lg).numpy()
    y_true = lb.numpy()
    preds  = (probs > 0.5).astype(float)
    f1   = f1_score(y_true, preds, average="macro",    zero_division=0)
    prec = precision_score(y_true, preds, average="macro", zero_division=0)
    rec  = recall_score(y_true, preds, average="macro",   zero_division=0)
    return loss, f1, prec, rec, lg, lb


def tune_thresholds(val_probs, val_true):
    """Find per-class threshold that maximises F1 on the validation set."""
    thresholds = []
    for i in range(N_PAT):
        bt, bf = 0.5, 0.0
        for t in np.linspace(0.05, 0.95, 100):
            f = f1_score(val_true[:, i],
                         (val_probs[:, i] > t).astype(float), zero_division=0)
            if f >= bf:
                bf, bt = f, t
        thresholds.append(bt)
    return thresholds


def print_results(test_probs, test_true, thresholds):
    print(f"\n{'Pattern':<25} | {'Thresh':>6} | {'F1%':>6} | {'Prec%':>6} | {'Rec%':>6} | {'TP':>5} | {'FP':>5} | {'FN':>5} | {'Sup':>5}")
    print("â”€" * 88)
    f1s = []
    for i, nm in enumerate(PATTERN_NAMES):
        t   = thresholds[i]
        pd_ = (test_probs[:, i] > t).astype(float)
        f1  = f1_score(test_true[:, i],  pd_, zero_division=0)
        pr  = precision_score(test_true[:, i], pd_, zero_division=0)
        re  = recall_score(test_true[:, i],  pd_, zero_division=0)
        tp  = int(((pd_ == 1) & (test_true[:, i] == 1)).sum())
        fp  = int(((pd_ == 1) & (test_true[:, i] == 0)).sum())
        fn  = int(((pd_ == 0) & (test_true[:, i] == 1)).sum())
        sup = int(test_true[:, i].sum())
        f1s.append(f1)
        print(f"{nm:<25} | {t:>6.2f} | {f1*100:>6.1f} | {pr*100:>6.1f} | {re*100:>6.1f} | {tp:>5} | {fp:>5} | {fn:>5} | {sup:>5}")
    print("â”€" * 88)
    print(f"{'Macro avg':<25} | {'':>6} | {np.mean(f1s)*100:>6.1f}")


# â”€â”€ MAIN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--eval",   action="store_true", help="Evaluate saved model only")
    parser.add_argument("--epochs", type=int, default=TOTAL_EPOCHS, help="Epochs to run")
    parser.add_argument("--ckpt",   type=str, default=CKPT_PATH)
    parser.add_argument("--out",    type=str, default=MODEL_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # â”€â”€ Data â”€â”€
    tickers = fetch_sp500_tickers()
    raw     = download_prices(tickers)
    X, y    = build_dataset(raw)

    n      = len(X)
    n_tr   = int(n * 0.70)
    n_vl   = int(n * 0.85)
    X_tr, y_tr = X[:n_tr],      y[:n_tr]
    X_vl, y_vl = X[n_tr:n_vl],  y[n_tr:n_vl]
    X_te, y_te = X[n_vl:],      y[n_vl:]

    sw      = np.where(y_tr.sum(axis=1) > 0, POS_OVERSAMPLE, 1.0)
    sampler = WeightedRandomSampler(torch.FloatTensor(sw), len(sw), replacement=True)
    train_ldr = DataLoader(TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)),
                           batch_size=BATCH_SIZE, sampler=sampler)
    val_ldr   = DataLoader(TensorDataset(torch.FloatTensor(X_vl), torch.FloatTensor(y_vl)),
                           batch_size=BATCH_SIZE, shuffle=False)
    test_ldr  = DataLoader(TensorDataset(torch.FloatTensor(X_te), torch.FloatTensor(y_te)),
                           batch_size=BATCH_SIZE, shuffle=False)

    pos_c = y_tr.sum(axis=0)
    neg_c = len(y_tr) - pos_c
    pos_w = torch.FloatTensor(neg_c / (pos_c + 1e-6)).to(device)

    # â”€â”€ Model â”€â”€
    model     = PatternLSTM().to(device)
    criterion = FocalLoss(pos_weight=pos_w)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-5)

    hist        = {"tl": [], "vl": [], "vf1": []}
    epoch_start = 1
    best_f1     = 0.0
    best_state  = None

    if (args.resume or args.eval) and os.path.exists(args.ckpt):
        ckpt        = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        hist        = ckpt.get("hist", hist)
        epoch_start = ckpt.get("epoch", 0) + 1
        best_f1     = ckpt.get("best_f1", 0.0)
        best_state  = ckpt.get("best_state", None)
        for _ in range(epoch_start - 1):
            scheduler.step()
        print(f"Resumed from epoch {epoch_start - 1}  best_f1={best_f1:.4f}")

    if args.eval:
        print("Eval-only mode.")
    else:
        # â”€â”€ Training loop â”€â”€
        target_ep = min(epoch_start + args.epochs - 1, TOTAL_EPOCHS)
        print(f"\nTraining epochs {epoch_start} â†’ {target_ep}")
        print(f"{'Ep':>4} | {'T-Loss':>8} | {'V-Loss':>8} | {'Val F1':>7} | {'Time':>6}")
        print("â”€" * 44)
        for ep in range(epoch_start, target_ep + 1):
            t0 = time.time()
            model.train(); tl = 0
            for xb, yb in train_ldr:
                optimizer.zero_grad()
                loss = criterion(model(xb.to(device)), yb.to(device))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tl += loss.item()
            tl /= len(train_ldr)
            scheduler.step()

            vl, vf1, _, _, _, _ = evaluate(val_ldr, model, criterion, device)
            hist["tl"].append(tl); hist["vl"].append(vl); hist["vf1"].append(vf1)

            if vf1 > best_f1:
                best_f1    = vf1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            print(f"{ep:>4} | {tl:>8.4f} | {vl:>8.4f} | {vf1:>7.4f} | {time.time()-t0:.1f}s")

        torch.save({"model": model.state_dict(), "hist": hist, "epoch": target_ep,
                    "best_f1": best_f1, "best_state": best_state}, args.ckpt)
        print(f"Checkpoint â†’ {args.ckpt}")

    # â”€â”€ Evaluate best model on test set â”€â”€
    if best_state:
        model.load_state_dict(best_state)

    _, _, _, _, val_lgs, val_lbs = evaluate(val_ldr, model, criterion, device)
    val_probs  = torch.sigmoid(val_lgs).numpy()
    val_true   = val_lbs.numpy()
    thresholds = tune_thresholds(val_probs, val_true)

    _, _, _, _, tst_lgs, tst_lbs = evaluate(test_ldr, model, criterion, device)
    test_probs = torch.sigmoid(tst_lgs).numpy()
    test_true  = tst_lbs.numpy()
    print_results(test_probs, test_true, thresholds)

    # â”€â”€ Save final model â”€â”€
    torch.save({
        "model_state_dict": best_state or model.state_dict(),
        "model_config": {"in_": 5, "hid": 192, "layers": 3, "drop": 0.35, "n_cls": 8},
        "pattern_names": PATTERN_NAMES,
        "thresholds":    thresholds,
        "hist":          hist,
        "best_f1":       best_f1,
        "epochs_trained": len(hist["tl"]),
        "window_size":   WINDOW,
    }, args.out)
    print(f"Model saved â†’ {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
