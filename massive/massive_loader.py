"""
massive_loader.py
=================
Bulk historical intraday bar loader for Massive (formerly Polygon.io).

Pulls OHLCV aggregate bars for a ticker universe and writes them to a local
Parquet store, partitioned by interval / ticker / year. Designed to be run
repeatedly: completed partitions are skipped, so an interrupted backfill just
picks up where it left off.

Two ingestion paths:

  REST   -- per-ticker aggregate bars. Best for a few hundred tickers.
            Paid plans have unlimited API calls; this module still backs off
            politely on 429/5xx.
  FLAT   -- whole-market daily CSV files from S3. Best when you want every
            ticker for a date range. Included on all paid plans.

Quick start
-----------
    set MASSIVE_API_KEY=your_key_here

    # backfill 5 years of 1-minute bars for a universe
    python massive_loader.py backfill --tickers-file sp500.txt --years 5

    # same thing, one ticker, for a smoke test
    python massive_loader.py backfill --tickers AAPL --years 1

    # load it back into pandas
    >>> from massive_loader import load_bars, resample_ohlcv
    >>> df = load_bars("AAPL", interval="1min")
    >>> hourly = resample_ohlcv(df, "1h")

Notes on data hygiene (these matter for model training)
------------------------------------------------------
* ``adjusted=true`` (the default) applies split adjustment *relative to today*.
  Re-downloading a ticker after a split silently rewrites its history. If you
  need reproducible training sets, pull with ``--raw`` and apply your own
  adjustment factors, or re-pull the whole universe together and version it.
* Bars are absent, not zero, for minutes with no trades. This loader does NOT
  forward-fill -- do that downstream where you know your model's assumptions.
* Timestamps are stored as tz-aware UTC. ``load_bars`` converts to
  America/New_York on the way out, which is what you want for session logic.
* Extended-hours bars are included by default. ``--rth-only`` restricts to
  09:30-16:00 ET.
"""

from __future__ import annotations

import argparse
import gzip
import io
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pandas as pd
import requests

LOG = logging.getLogger("massive_loader")

API_BASE = "https://api.massive.com"
S3_ENDPOINT = "https://files.massive.com"
S3_BUCKET = "flatfiles"

MARKET_TZ = "America/New_York"
BAR_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "vwap", "transactions"]

# Massive's aggregate response uses single-letter keys.
_FIELD_MAP = {
    "t": "timestamp",
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "vw": "vwap",
    "n": "transactions",
}


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


@dataclass
class Config:
    api_key: str
    data_root: Path = Path("./market_data")
    multiplier: int = 1
    timespan: str = "minute"
    adjusted: bool = True
    rth_only: bool = False
    max_workers: int = 8
    max_retries: int = 6
    timeout: int = 60

    @property
    def interval(self) -> str:
        """Short label used in the on-disk path, e.g. '1min', '5min', '1hour'."""
        suffix = {"minute": "min", "hour": "hour", "day": "day", "second": "sec"}[self.timespan]
        return f"{self.multiplier}{suffix}"

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        key = os.environ.get("MASSIVE_API_KEY") or os.environ.get("POLYGON_API_KEY")
        if not key:
            raise SystemExit(
                "No API key found. Set MASSIVE_API_KEY in your environment "
                "(get one at https://massive.com/dashboard/keys)."
            )
        return cls(api_key=key, **overrides)


# --------------------------------------------------------------------------
# REST client
# --------------------------------------------------------------------------


class MassiveClient:
    """Thin wrapper over the aggregates endpoint with retries and pagination."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {cfg.api_key}",
                "Accept": "application/json",
                "User-Agent": "massive_loader/1.0",
            }
        )

    def _get(self, url: str, params: dict | None = None) -> dict:
        """GET with exponential backoff on 429 and 5xx."""
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=self.cfg.timeout)
            except requests.RequestException as exc:
                if attempt == self.cfg.max_retries - 1:
                    raise
                LOG.warning("network error (%s), retrying: %s", attempt + 1, exc)
                self._sleep(attempt, resp=None)
                continue

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code in (401, 403):
                raise RuntimeError(
                    f"Auth failed ({resp.status_code}). Check MASSIVE_API_KEY and that your "
                    f"plan covers this history depth. Body: {resp.text[:300]}"
                )

            if resp.status_code == 404:
                return {"results": [], "status": "NOT_FOUND"}

            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == self.cfg.max_retries - 1:
                    resp.raise_for_status()
                LOG.warning("HTTP %s, backing off (attempt %s)", resp.status_code, attempt + 1)
                self._sleep(attempt, resp=resp)
                continue

            resp.raise_for_status()

        raise RuntimeError("exhausted retries")

    @staticmethod
    def _sleep(attempt: int, resp: requests.Response | None) -> None:
        if resp is not None and resp.headers.get("Retry-After"):
            try:
                time.sleep(float(resp.headers["Retry-After"]))
                return
            except ValueError:
                pass
        time.sleep(min(60.0, (2**attempt) + random.random()))

    def aggregates(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Fetch all bars for ``ticker`` between ``start`` and ``end`` inclusive."""
        cfg = self.cfg
        url = (
            f"{API_BASE}/v2/aggs/ticker/{ticker}/range/{cfg.multiplier}/{cfg.timespan}"
            f"/{start.isoformat()}/{end.isoformat()}"
        )
        params = {
            "adjusted": "true" if cfg.adjusted else "false",
            "sort": "asc",
            "limit": 50_000,
        }

        rows: list[dict] = []
        page = 0
        while url:
            payload = self._get(url, params=params)
            rows.extend(payload.get("results") or [])
            url = payload.get("next_url")
            params = None  # next_url already carries the query string
            page += 1
            if page > 500:  # guard against a pagination loop
                LOG.error("%s: aborting after 500 pages", ticker)
                break

        return _frame_from_results(rows)


def _frame_from_results(rows: Sequence[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=BAR_COLUMNS)

    df = pd.DataFrame(rows).rename(columns=_FIELD_MAP)
    for col in BAR_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[BAR_COLUMNS]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "vwap"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["transactions"] = pd.to_numeric(df["transactions"], errors="coerce").astype("Int64")

    return df.dropna(subset=["open", "high", "low", "close"]).sort_values("timestamp").reset_index(drop=True)


# --------------------------------------------------------------------------
# on-disk store
# --------------------------------------------------------------------------


def _partition_path(cfg: Config, ticker: str, year: int) -> Path:
    return cfg.data_root / cfg.interval / ticker.upper() / f"{year}.parquet"


def backfill_ticker(client: MassiveClient, ticker: str, years: Sequence[int], force: bool = False) -> dict:
    """Download and persist one ticker, one Parquet file per calendar year."""
    cfg = client.cfg
    ticker = ticker.upper()
    stats = {"ticker": ticker, "years_written": 0, "rows": 0, "skipped": 0, "error": None}
    today = date.today()

    try:
        for year in years:
            path = _partition_path(cfg, ticker, year)
            # A past year's partition is immutable; the current year is not.
            if path.exists() and not force and year < today.year:
                stats["skipped"] += 1
                continue

            start = date(year, 1, 1)
            end = min(date(year, 12, 31), today)
            if start > today:
                continue

            df = client.aggregates(ticker, start, end)
            if df.empty:
                LOG.info("%s %s: no data", ticker, year)
                continue

            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".parquet.tmp")
            df.to_parquet(tmp, index=False, compression="zstd")
            tmp.replace(path)  # atomic-ish: never leave a half-written partition

            stats["years_written"] += 1
            stats["rows"] += len(df)
            LOG.info("%s %s: %d bars -> %s", ticker, year, len(df), path.name)
    except Exception as exc:  # noqa: BLE001 - one bad ticker shouldn't kill the run
        stats["error"] = f"{type(exc).__name__}: {exc}"
        LOG.error("%s failed: %s", ticker, stats["error"])

    return stats


def backfill(cfg: Config, tickers: Sequence[str], years: Sequence[int], force: bool = False) -> pd.DataFrame:
    """Backfill a universe in parallel. Returns a per-ticker summary frame."""
    client = MassiveClient(cfg)
    results: list[dict] = []

    LOG.info(
        "backfilling %d tickers x %d years (%s bars) -> %s",
        len(tickers), len(years), cfg.interval, cfg.data_root.resolve(),
    )

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        futures = {pool.submit(backfill_ticker, client, t, years, force): t for t in tickers}
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % 25 == 0 or i == len(futures):
                LOG.info("progress: %d/%d tickers", i, len(futures))

    summary = pd.DataFrame(results)
    failed = summary[summary["error"].notna()]
    if not failed.empty:
        LOG.warning("%d tickers failed: %s", len(failed), ", ".join(failed["ticker"].head(20)))
    return summary


def load_bars(
    ticker: str,
    interval: str = "1min",
    data_root: Path | str = "./market_data",
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    tz: str | None = MARKET_TZ,
    rth_only: bool = False,
) -> pd.DataFrame:
    """Read a ticker back out of the Parquet store, indexed by timestamp."""
    root = Path(data_root) / interval / ticker.upper()
    files = sorted(root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No data for {ticker} at {root}")

    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")

    if tz:
        df["timestamp"] = df["timestamp"].dt.tz_convert(tz)
    df = df.set_index("timestamp")

    if start is not None:
        df = df.loc[df.index >= pd.Timestamp(start, tz=df.index.tz)]
    if end is not None:
        df = df.loc[df.index <= pd.Timestamp(end, tz=df.index.tz)]
    if rth_only:
        df = _filter_rth(df)

    return df


def _filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only regular-session bars (09:30-16:00 ET), weekdays."""
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    local = idx.tz_convert(MARKET_TZ)
    mask = (
        (local.dayofweek < 5)
        & (local.time >= pd.Timestamp("09:30").time())
        & (local.time < pd.Timestamp("16:00").time())
    )
    return df[mask]


def resample_ohlcv(df: pd.DataFrame, rule: str = "1h", rth_only: bool = True) -> pd.DataFrame:
    """
    Aggregate finer bars up to a coarser interval.

    Lets you keep one 1-minute store on disk and derive the hourly bars your
    existing pipelines expect, instead of maintaining two downloads.
    """
    if rth_only:
        df = _filter_rth(df)

    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    if "transactions" in df.columns:
        agg["transactions"] = "sum"

    out = df.resample(rule, label="left", closed="left", origin="start_day").agg(agg)

    # Volume-weight the VWAP rather than averaging it.
    if "vwap" in df.columns:
        notional = (df["vwap"] * df["volume"]).resample(rule, label="left", closed="left", origin="start_day").sum()
        vol = df["volume"].resample(rule, label="left", closed="left", origin="start_day").sum()
        out["vwap"] = notional / vol.replace(0, pd.NA)

    return out.dropna(subset=["open"])


# --------------------------------------------------------------------------
# flat files (S3) -- whole-market daily minute bars
# --------------------------------------------------------------------------


def download_flat_files(
    access_key: str,
    secret_key: str,
    start: date,
    end: date,
    out_dir: Path | str,
    tickers: Iterable[str] | None = None,
    prefix: str = "us_stocks_sip/minute_aggs_v1",
) -> list[Path]:
    """
    Pull whole-market minute-aggregate files from Massive's S3 flat-file store.

    Faster than the REST API when you want the entire market, and it does not
    touch your API rate budget at all. S3 keys are your *flat-file* credentials
    from https://massive.com/dashboard/keys, not your REST API key.

    If ``tickers`` is given, each day is filtered down to those symbols before
    being written, which keeps the local footprint manageable.
    """
    import boto3  # imported lazily: only the flat-file path needs it
    from botocore.config import Config as BotoConfig

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = {t.upper() for t in tickers} if tickers else None

    s3 = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    ).client("s3", endpoint_url=S3_ENDPOINT, config=BotoConfig(signature_version="s3v4"))

    written: list[Path] = []
    day = start
    while day <= end:
        if day.weekday() < 5:  # skip weekends; holidays just 404
            key = f"{prefix}/{day.year}/{day.month:02d}/{day.isoformat()}.csv.gz"
            dest = out_dir / f"{day.isoformat()}.parquet"
            if not dest.exists():
                try:
                    blob = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
                except Exception as exc:  # noqa: BLE001 - market holiday or missing day
                    LOG.debug("%s unavailable: %s", key, exc)
                    day += timedelta(days=1)
                    continue

                df = pd.read_csv(io.BytesIO(gzip.decompress(blob)))
                if keep is not None:
                    df = df[df["ticker"].str.upper().isin(keep)]
                # window_start is nanoseconds since epoch, UTC
                df["timestamp"] = pd.to_datetime(df["window_start"], unit="ns", utc=True)
                df = df.drop(columns=["window_start"])
                df.to_parquet(dest, index=False, compression="zstd")
                written.append(dest)
                LOG.info("%s: %d rows", day, len(df))
        day += timedelta(days=1)

    return written


# --------------------------------------------------------------------------
# helpers + CLI
# --------------------------------------------------------------------------


def read_universe(path: Path | str) -> list[str]:
    """One ticker per line; blank lines and '#' comments ignored."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip().upper() for ln in lines if ln.strip() and not ln.startswith("#")]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Backfill historical intraday bars from Massive.")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("backfill", help="download bars into the local Parquet store")
    b.add_argument("--tickers", nargs="+", help="explicit ticker list")
    b.add_argument("--tickers-file", help="file with one ticker per line")
    b.add_argument("--years", type=int, default=5, help="how many calendar years back (default 5)")
    b.add_argument("--data-root", default="./market_data")
    b.add_argument("--multiplier", type=int, default=1)
    b.add_argument("--timespan", default="minute", choices=["second", "minute", "hour", "day"])
    b.add_argument("--raw", action="store_true", help="unadjusted prices (adjusted is the default)")
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--force", action="store_true", help="re-download partitions that already exist")

    f = sub.add_parser("flatfiles", help="download whole-market daily files from S3")
    f.add_argument("--start", required=True, help="YYYY-MM-DD")
    f.add_argument("--end", required=True, help="YYYY-MM-DD")
    f.add_argument("--out", default="./flat_files")
    f.add_argument("--tickers-file", help="optional filter")

    return p


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_parser().parse_args(argv)

    if args.command == "backfill":
        tickers = list(args.tickers or [])
        if args.tickers_file:
            tickers += read_universe(args.tickers_file)
        if not tickers:
            raise SystemExit("Give --tickers and/or --tickers-file")

        cfg = Config.from_env(
            data_root=Path(args.data_root),
            multiplier=args.multiplier,
            timespan=args.timespan,
            adjusted=not args.raw,
            max_workers=args.workers,
        )

        this_year = date.today().year
        years = list(range(this_year - args.years + 1, this_year + 1))

        t0 = time.time()
        summary = backfill(cfg, sorted(set(tickers)), years, force=args.force)
        LOG.info(
            "done in %.1f min | %d partitions, %d bars, %d failures",
            (time.time() - t0) / 60,
            int(summary["years_written"].sum()),
            int(summary["rows"].sum()),
            int(summary["error"].notna().sum()),
        )
        summary.to_csv(Path(args.data_root) / "_backfill_summary.csv", index=False)

    elif args.command == "flatfiles":
        access = os.environ.get("MASSIVE_S3_ACCESS_KEY")
        secret = os.environ.get("MASSIVE_S3_SECRET_KEY")
        if not (access and secret):
            raise SystemExit("Set MASSIVE_S3_ACCESS_KEY and MASSIVE_S3_SECRET_KEY")

        files = download_flat_files(
            access,
            secret,
            date.fromisoformat(args.start),
            date.fromisoformat(args.end),
            args.out,
            tickers=read_universe(args.tickers_file) if args.tickers_file else None,
        )
        LOG.info("wrote %d daily files to %s", len(files), args.out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
