import yfinance as yf
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta

end = datetime.today()
start = end - timedelta(days=182)

print("Downloading S&P 500 data...")
df = yf.download("^GSPC", start=start, end=end, progress=False)
df = df[["Close"]].copy()
df.columns = ["Close"]
df.dropna(inplace=True)
close = df["Close"].values
dates = df.index

# --- helpers ---
def find_peaks(data, order=5):
    idx = argrelextrema(data, np.greater, order=order)[0]
    return idx

def find_troughs(data, order=5):
    idx = argrelextrema(data, np.less, order=order)[0]
    return idx

# --- 1. Head and Shoulders ---
def detect_head_and_shoulders(close, dates):
    peaks = find_peaks(close, order=7)
    troughs = find_troughs(close, order=7)
    results = []
    for i in range(len(peaks) - 2):
        lsh, head, rsh = peaks[i], peaks[i+1], peaks[i+2]
        if close[head] > close[lsh] and close[head] > close[rsh]:
            if abs(close[lsh] - close[rsh]) / close[head] < 0.05:
                between_troughs = [t for t in troughs if lsh < t < rsh]
                if len(between_troughs) >= 2:
                    results.append({
                        "type": "Head & Shoulders",
                        "left_shoulder": (dates[lsh], close[lsh]),
                        "head": (dates[head], close[head]),
                        "right_shoulder": (dates[rsh], close[rsh]),
                        "neckline": np.mean([close[t] for t in between_troughs]),
                        "indices": (lsh, head, rsh),
                    })
    return results

# --- 2. Cup and Handle ---
def detect_cup_and_handle(close, dates):
    results = []
    n = len(close)
    window = 40
    for i in range(0, n - window, 5):
        segment = close[i:i+window]
        left = segment[:5].mean()
        bottom = segment[10:20].mean()
        right = segment[25:35].mean()
        if left > bottom and right > bottom:
            if abs(left - right) / left < 0.05:
                depth = (left - bottom) / left
                if 0.05 < depth < 0.25:
                    handle_start = i + 30
                    handle_end = min(i + window, n - 1)
                    handle = close[handle_start:handle_end]
                    if len(handle) > 3 and handle.mean() < right and handle[-1] > handle[0] * 0.97:
                        results.append({
                            "type": "Cup & Handle",
                            "cup_start": (dates[i], close[i]),
                            "cup_bottom": (dates[i+15], close[i+15]),
                            "cup_end": (dates[i+30], close[i+30]),
                            "handle_end": (dates[handle_end], close[handle_end]),
                            "indices": (i, i+15, i+30, handle_end),
                        })
    return results[:2]

# --- 3. Rising Wedge ---
def detect_rising_wedge(close, dates):
    results = []
    n = len(close)
    window = 30
    for i in range(0, n - window, 5):
        segment = close[i:i+window]
        x = np.arange(len(segment))
        peaks_idx = find_peaks(segment, order=3)
        troughs_idx = find_troughs(segment, order=3)
        if len(peaks_idx) >= 2 and len(troughs_idx) >= 2:
            peak_slope = np.polyfit(peaks_idx, segment[peaks_idx], 1)[0]
            trough_slope = np.polyfit(troughs_idx, segment[troughs_idx], 1)[0]
            if peak_slope > 0 and trough_slope > 0 and trough_slope > peak_slope:
                results.append({
                    "type": "Rising Wedge",
                    "start": (dates[i], close[i]),
                    "end": (dates[i+window-1], close[i+window-1]),
                    "peak_slope": peak_slope,
                    "trough_slope": trough_slope,
                    "indices": (i, i+window-1),
                    "peaks": [i + p for p in peaks_idx],
                    "troughs": [i + t for t in troughs_idx],
                })
    return results[:2]

hs = detect_head_and_shoulders(close, dates)
ch = detect_cup_and_handle(close, dates)
rw = detect_rising_wedge(close, dates)

all_patterns = hs + ch + rw
print(f"\nPatterns found: {len(hs)} Head & Shoulders, {len(ch)} Cup & Handle, {len(rw)} Rising Wedge\n")

# --- Plot ---
fig, axes = plt.subplots(len(all_patterns) + 1, 1,
                         figsize=(14, 4 * (len(all_patterns) + 1)),
                         constrained_layout=True)
if len(all_patterns) == 0:
    axes = [axes]

ax0 = axes[0]
ax0.plot(dates, close, color="#378ADD", linewidth=1.2)
ax0.set_title("S&P 500 — last 6 months", fontsize=13)
ax0.set_ylabel("Price")
ax0.grid(True, alpha=0.3)

colors = {"Head & Shoulders": "#E24B4A", "Cup & Handle": "#1D9E75", "Rising Wedge": "#BA7517"}

for idx, pattern in enumerate(all_patterns):
    ax = axes[idx + 1]
    ax.plot(dates, close, color="#888780", linewidth=0.8, alpha=0.5)
    color = colors.get(pattern["type"], "#378ADD")

    if pattern["type"] == "Head & Shoulders":
        li, hi, ri = pattern["indices"]
        seg_dates = dates[li:ri+1]
        seg_close = close[li:ri+1]
        ax.plot(seg_dates, seg_close, color=color, linewidth=1.8)
        ax.axhline(pattern["neckline"], color=color, linestyle="--", alpha=0.6, label="Neckline")
        ax.scatter([pattern["left_shoulder"][0], pattern["head"][0], pattern["right_shoulder"][0]],
                   [pattern["left_shoulder"][1], pattern["head"][1], pattern["right_shoulder"][1]],
                   color=color, zorder=5, s=60)
        for label, point in [("L", pattern["left_shoulder"]), ("H", pattern["head"]), ("R", pattern["right_shoulder"])]:
            ax.annotate(label, xy=point, xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=9, color=color, fontweight="bold")

    elif pattern["type"] == "Cup & Handle":
        i0, i_bot, i_end, i_handle = pattern["indices"]
        seg = slice(i0, i_handle+1)
        ax.plot(dates[seg], close[seg], color=color, linewidth=1.8)
        ax.axhline(close[i0], color=color, linestyle="--", alpha=0.5, label="Rim level")
        ax.annotate("Cup", xy=(dates[i_bot], close[i_bot]),
                    xytext=(0, -18), textcoords="offset points",
                    ha="center", fontsize=9, color=color, fontweight="bold")
        ax.annotate("Handle", xy=(dates[i_handle], close[i_handle]),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=9, color=color, fontweight="bold")

    elif pattern["type"] == "Rising Wedge":
        i0, i1 = pattern["indices"]
        ax.plot(dates[i0:i1+1], close[i0:i1+1], color=color, linewidth=1.8)
        pk_idx = pattern["peaks"]
        tr_idx = pattern["troughs"]
        if len(pk_idx) >= 2:
            x_pk = np.array([i0 + (pk - i0) for pk in pk_idx])
            y_pk = close[pk_idx]
            m, b = np.polyfit(range(len(x_pk)), y_pk, 1)
            upper_y = [b + m * j for j in range(len(x_pk))]
            ax.plot(dates[pk_idx], upper_y, color=color, linestyle="--", alpha=0.7, label="Upper trendline")
        if len(tr_idx) >= 2:
            x_tr = np.array([i0 + (tr - i0) for tr in tr_idx])
            y_tr = close[tr_idx]
            m2, b2 = np.polyfit(range(len(x_tr)), y_tr, 1)
            lower_y = [b2 + m2 * j for j in range(len(x_tr))]
            ax.plot(dates[tr_idx], lower_y, color=color, linestyle="--", alpha=0.7, label="Lower trendline")

    ax.set_title(f"{pattern['type']} — detected", fontsize=12, color=color)
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

if not all_patterns:
    print("No patterns detected in this 6-month window. Markets don't always cooperate!")
    for ax in axes[1:]:
        ax.set_visible(False)

plt.suptitle("S&P 500 Chart Pattern Detection", fontsize=15, fontweight="500", y=1.01)
plt.savefig("sp500_patterns.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nChart saved as sp500_patterns.png")
