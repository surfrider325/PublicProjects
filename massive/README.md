# Massive intraday bar loader

Backfills historical OHLCV bars from [Massive](https://massive.com) (formerly
Polygon.io) into a local Parquet store, and reads them back into pandas.

Built against the **Stocks Starter** plan: 5 years of history, minute and second
aggregates, 15-minute delayed, flat-file access included.

## Setup

```powershell
& "C:\Users\rcsch\AppData\Local\Programs\Python\Python312\python.exe" -m pip install -r requirements.txt

# API key from https://massive.com/dashboard/keys
$env:MASSIVE_API_KEY = "your_key_here"
```

To make the key stick across sessions:

```powershell
[Environment]::SetEnvironmentVariable("MASSIVE_API_KEY", "your_key_here", "User")
```

## Backfill

```powershell
# refresh the universe (writes sp500.txt)
python fetch_sp500.py

# smoke test - one ticker, one year
python massive_loader.py backfill --tickers AAPL --years 1 --data-root C:\market_data

# the real run
python massive_loader.py backfill --tickers-file sp500.txt --years 5 --data-root C:\market_data
```

The store lives at `C:\market_data`, deliberately **outside OneDrive** so several
gigabytes of Parquet don't sync to the cloud. Always pass `--data-root
C:\market_data`, or you'll start a second store in the project folder.

Writes `C:\market_data\1min\AAPL\2025.parquet`, one file per ticker-year, plus a
`_backfill_summary.csv` listing rows written and any failures.

Re-running is cheap: completed past-year partitions are skipped, so an
interrupted run resumes where it stopped. The current year is always refreshed.
Add `--force` to re-download everything.

`sp500.txt` is just one ticker per line; blank lines and `#` comments ignored.

## Reading it back

```python
from massive_loader import load_bars, resample_ohlcv

df = load_bars("AAPL", data_root=r"C:\market_data")              # tz-aware ET
rth = load_bars("AAPL", data_root=r"C:\market_data", rth_only=True)  # 09:30-16:00
hourly = resample_ohlcv(df, "1h")                  # derive hourly from minute
```

Keeping one 1-minute store and resampling means the hourly LSTM pipelines keep
working unchanged, with no second download to keep in sync.

## Flat files (whole market)

Faster than REST past a couple thousand symbols, and it doesn't touch the API
rate budget. Uses separate S3 credentials from the same dashboard page.

```powershell
$env:MASSIVE_S3_ACCESS_KEY = "..."
$env:MASSIVE_S3_SECRET_KEY = "..."
python massive_loader.py flatfiles --start 2026-01-01 --end 2026-09-17 --out .\flat_files
```

## Gotchas worth remembering

**Split adjustment is relative to today.** `adjusted=true` (the default) means
re-downloading a ticker after a split rewrites its history underneath you. For
reproducible training sets, either pull `--raw` and apply your own factors, or
re-pull the whole universe at once and version the directory. This is the
failure mode that makes a backtest look great and live trading not.

**Missing bars are missing, not zero.** Minutes with no trades simply have no
row. The loader does not forward-fill; do that downstream where you know what
your model assumes.

**Starter is 15-minute delayed.** Fine for training and for a scanner running on
closed bars. Same-minute signals need the Advanced tier.

**Class shares use dots, not dashes.** Massive wants `BRK.B` and `BF.B`. Asking
for `BRK-B` returns zero bars with no error -- the ticker simply looks empty.

**Ticker renames and spinoffs leave holes.** Roughly 50 partitions are missing
across the universe because the company didn't trade under that symbol yet:
GEHC (spun off Jan 2023), GEV (Apr 2024), KVUE (May 2023), CPAY (renamed from
FLEETCOR), COR (renamed from ABC), and similar. Not an error, but your training
code needs to tolerate tickers with short histories.

**The universe is current membership, so it carries survivorship bias.**
`sp500.txt` is today's index, not point-in-time membership. Anything that got
dropped from the index isn't in it.

## Data layout

```
market_data/
  1min/
    AAPL/
      2022.parquet
      ...
  _backfill_summary.csv
```

Columns: `timestamp` (UTC, tz-aware), `open`, `high`, `low`, `close`, `volume`,
`vwap`, `transactions`.
