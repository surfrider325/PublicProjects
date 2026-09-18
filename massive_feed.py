"""
massive_feed.py
===============
Drop-in replacement for ``yfinance.download`` backed by the local Massive
Parquet store at C:\\market_data.

Why this exists: yfinance caps 15-minute history at ~60 days, which is what
forced lstm_train_historical.py to train on a 60-day window. The local store
holds 5 years of 1-minute bars, so any intraday interval can be derived from it
without a network call.

    import massive_feed
    df = massive_feed.download("AAPL", period="2y", interval="15m")

Returns the same shape yfinance does -- capitalized OHLCV columns, tz-naive
index in Eastern wall-clock time -- so calling code needs no other changes.
Tickers not present in the store fall back to yfinance automatically.

Differences from yfinance worth knowing
---------------------------------------
* Massive adjusts for SPLITS only. yfinance's auto_adjust=True also adjusts for
  dividends. Price levels therefore differ slightly on dividend payers; returns,
  ranges and oscillators are effectively unaffected, but don't mix the two
  sources inside one training set.
* Daily bars here are built from regular-session minute bars, so the daily
  close is the 16:00 print rather than the official consolidated close. These
  agree to a cent or two on liquid names.
* The store is only as fresh as your last backfill. Pass refresh=True (or set
  MASSIVE_AUTO_REFRESH=1) to top up the current year before reading.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

DATA_ROOT = Path(os.environ.get("MASSIVE_DATA_ROOT", r"C:\market_data"))
LOADER_DIR = Path(r"C:\Users\rcsch\OneDrive\Documents\massive")

if str(LOADER_DIR) not in sys.path:
    sys.path.insert(0, str(LOADER_DIR))

from massive_loader import load_bars, resample_ohlcv, _filter_rth  # noqa: E402

_PERIOD_RE = re.compile(r"^(\d+)(d|wk|mo|y)$")
_PERIOD_DAYS = {"d": 1, "wk": 7, "mo": 31, "y": 366}

# yfinance interval string -> pandas resample rule
_RULE = {
    "1m": None,       # native resolution, no resampling
    "2m": "2min", "5m": "5min", "15m": "15min", "30m": "30min",
    "60m": "60min", "1h": "60min", "90m": "90min",
    "1d": "1D", "1wk": "1W", "1mo": "1ME",
}

_CACHE: dict[tuple[str, str], pd.DataFrame] = {}

# Symbols that changed since older ticker lists were written. Without these the
# lookup misses, falls back to yfinance, and quietly reimposes the 60-day cap on
# that ticker -- the exact failure this module exists to remove.
RENAMES = {
    "MMC": "MRSH",    # Marsh & McLennan
    "FLT": "CPAY",    # FLEETCOR -> Corpay
    "ABC": "COR",     # AmerisourceBergen -> Cencora
    "FB":  "META",
    "FISV": "FI",     # Fiserv
    "RE":  "EG",      # Everest Re
    "PKI": "RVTY",    # PerkinElmer -> Revvity
    "ANTM": "ELV",    # Anthem -> Elevance
}


def resolve(ticker: str) -> str:
    """
    Map a caller's symbol onto what the store actually holds.

    Handles the dash/dot convention for class shares (BRK-B -> BRK.B, which is
    what Massive serves) and a short table of renames.
    """
    t = str(ticker).upper().strip()
    root = DATA_ROOT / "1min"
    if (root / t).is_dir():
        return t
    dotted = t.replace("-", ".")
    if dotted != t and (root / dotted).is_dir():
        return dotted
    renamed = RENAMES.get(t)
    if renamed and (root / renamed).is_dir():
        return renamed
    return t


def _period_to_start(period: str | None, end: date) -> date | None:
    """'60d' / '2y' / '1mo' -> a start date. None or 'max' means everything."""
    if not period or period == "max":
        return None
    m = _PERIOD_RE.match(period.strip().lower())
    if not m:
        raise ValueError(f"Unrecognized period {period!r}")
    n, unit = int(m.group(1)), m.group(2)
    return end - timedelta(days=n * _PERIOD_DAYS[unit])


def available(ticker: str) -> bool:
    return (DATA_ROOT / "1min" / resolve(ticker)).is_dir()


def latest_bar(ticker: str) -> pd.Timestamp | None:
    try:
        df = load_bars(resolve(ticker), data_root=DATA_ROOT)
    except FileNotFoundError:
        return None
    return df.index.max() if len(df) else None


def refresh_ticker(ticker: str, max_age_hours: float = 20.0) -> bool:
    """
    Top up the current year for one ticker if its newest bar is stale.

    Returns True if a download was attempted. Silently no-ops when no API key
    is set, so training runs on a plane still work.
    """
    if not os.environ.get("MASSIVE_API_KEY"):
        return False

    newest = latest_bar(ticker)
    if newest is not None:
        age = pd.Timestamp.now(tz=newest.tz) - newest
        if age < pd.Timedelta(hours=max_age_hours):
            return False

    cmd = [
        sys.executable,
        str(LOADER_DIR / "massive_loader.py"),
        "backfill",
        "--tickers", ticker,
        "--years", "1",
        "--data-root", str(DATA_ROOT),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
    except Exception as exc:  # noqa: BLE001 - a stale cache beats a crash
        print(f"  [massive_feed] refresh of {ticker} failed: {exc}")
        return False

    _CACHE.pop((ticker.upper(), "raw"), None)
    return True


def _to_yf_shape(df: pd.DataFrame) -> pd.DataFrame:
    """Capitalized OHLCV columns, tz-naive Eastern index -- yfinance's layout."""
    out = df.rename(
        columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume", "vwap": "VWAP",
        }
    )
    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in out.columns]
    out = out[keep].copy()
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.index.name = "Datetime"
    return out


def download(
    ticker: str,
    period: str | None = None,
    interval: str = "1d",
    start=None,
    end=None,
    auto_adjust: bool = True,      # accepted for signature compatibility
    progress: bool = False,        # ditto
    prepost: bool = False,         # yfinance's default: regular session only
    refresh: bool | None = None,
    **kwargs,
) -> pd.DataFrame:
    """yfinance.download-compatible reader over the local Massive store."""
    ticker = resolve(ticker)

    if refresh is None:
        refresh = os.environ.get("MASSIVE_AUTO_REFRESH", "").strip() in ("1", "true", "yes")
    if refresh:
        refresh_ticker(ticker)

    if not available(ticker):
        import yfinance as yf
        return yf.download(ticker, period=period, interval=interval,
                           auto_adjust=auto_adjust, progress=progress, **kwargs)

    if interval not in _RULE:
        raise ValueError(f"Unsupported interval {interval!r}; known: {sorted(_RULE)}")

    key = (ticker, "raw")
    if key not in _CACHE:
        _CACHE[key] = load_bars(ticker, data_root=DATA_ROOT)
    minute = _CACHE[key]

    # Regular session only by default -- this matches yfinance (prepost=False),
    # and extended-hours bars would otherwise corrupt daily aggregates and put
    # ragged bar counts in each intraday day. The 1-minute path needs the same
    # filter as the resampled ones, or callers asking for "1m" silently get
    # 04:00-20:00 bars where yfinance gave them 09:30-16:00.
    rule = _RULE[interval]
    if rule is None:
        bars = minute if prepost else _filter_rth(minute)
    else:
        bars = resample_ohlcv(minute, rule, rth_only=not prepost)

    if end is None:
        end_d = date.today()
    else:
        end_d = pd.Timestamp(end).date()
        bars = bars[bars.index.date <= end_d]

    if start is not None:
        bars = bars[bars.index.date >= pd.Timestamp(start).date()]
    elif period:
        m = _PERIOD_RE.match(period.strip().lower())
        if m and m.group(2) == "d":
            # yfinance counts "5d" as 5 TRADING days, not 5 calendar days. Using
            # a calendar cutoff here silently returns 4 sessions over a weekend,
            # which shortens seed windows that callers sized by bar count.
            n = int(m.group(1))
            sessions = bars.index.normalize().unique()
            if len(sessions) > n:
                bars = bars[bars.index.normalize() >= sessions[-n]]
        else:
            start_d = _period_to_start(period, end_d)
            if start_d is not None:
                bars = bars[bars.index.date >= start_d]

    return _to_yf_shape(bars)


def clear_cache() -> None:
    _CACHE.clear()


if __name__ == "__main__":
    for iv, per in (("1d", "2y"), ("15m", "2y"), ("1h", "1y")):
        d = download("AAPL", period=per, interval=iv)
        print(f"{iv:>4} {per:>3}: {len(d):>7,} bars  {d.index.min()} -> {d.index.max()}")
        print(f"          cols={list(d.columns)}  bars/day={len(d) / max(d.index.normalize().nunique(), 1):.1f}")
