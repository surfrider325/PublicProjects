"""HTTP extraction: drives the paginator, applies auth, retries, and rate limits."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

import httpx

from .auth import Authenticator
from .config import SourceConfig
from .paginate import make_paginator
from .transform import extract_records

log = logging.getLogger(__name__)


class ExtractError(Exception):
    """Raised when an endpoint cannot be read after all retries."""


@dataclass
class Page:
    """One successful HTTP response worth of records."""

    number: int
    url: str
    records: list[Any]
    status_code: int


class RateLimiter:
    """Enforces a minimum wall-clock interval between requests."""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        try:
            from email.utils import parsedate_to_datetime

            when = parsedate_to_datetime(value)
            return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
        except Exception:
            return None


def format_watermark(value: Any, source: SourceConfig) -> str:
    """Render an incremental cursor value for use as a query parameter."""
    inc = source.incremental
    if isinstance(value, datetime):
        moment = value
        if inc.lookback_seconds:
            moment = moment - timedelta(seconds=inc.lookback_seconds)
        if inc.format:
            return moment.strftime(inc.format)
        return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return str(value)


class Extractor:
    """Reads every page of one source, yielding records as they arrive."""

    def __init__(self, source: SourceConfig, client: httpx.Client | None = None) -> None:
        self.source = source
        self.client = client or httpx.Client(follow_redirects=True)
        self._owns_client = client is None
        self.auth = Authenticator(source.auth)
        self.limiter = RateLimiter(source.min_request_interval)
        self.requests_made = 0

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> Extractor:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def iter_pages(self, watermark: Any = None) -> Iterator[Page]:
        src = self.source
        base_params: dict[str, Any] = dict(src.request.params)
        if watermark is not None and src.incremental.mode == "timestamp":
            assert src.incremental.param
            base_params[src.incremental.param] = format_watermark(watermark, src)
            log.info("[%s] incremental pull from %s", src.name, base_params[src.incremental.param])

        paginator = make_paginator(src.pagination)
        page_number = 0

        while True:
            request = paginator.next_request(src.request.url, base_params)
            if request is None:
                return
            url, params = request
            page_number += 1

            response = self._request_with_retries(url, params)
            body = self._decode(response, url)
            records = extract_records(body, src.records_path)

            log.info(
                "[%s] page %d -> %d records (%s)",
                src.name,
                page_number,
                len(records),
                response.status_code,
            )
            yield Page(
                number=page_number, url=str(response.url), records=records,
                status_code=response.status_code,
            )
            paginator.observe(body, dict(response.headers), len(records))

    def iter_records(self, watermark: Any = None) -> Iterator[Any]:
        for page in self.iter_pages(watermark):
            yield from page.records

    # ------------------------------------------------------------------ internals

    def _decode(self, response: httpx.Response, url: str) -> Any:
        try:
            return response.json()
        except ValueError as exc:
            raise ExtractError(
                f"[{self.source.name}] non-JSON response from {url}: "
                f"{response.text[:200]!r}"
            ) from exc

    def _request_with_retries(self, url: str, params: dict[str, Any]) -> httpx.Response:
        src = self.source
        retry = src.retry
        delay = retry.backoff_seconds
        last_error: str = "unknown error"

        for attempt in range(1, retry.max_attempts + 1):
            headers = dict(src.request.headers)
            call_params = dict(params)
            self.auth.apply(self.client, headers, call_params)
            self.limiter.wait()

            try:
                response = self.client.request(
                    src.request.method,
                    url,
                    params=call_params or None,
                    headers=headers,
                    json=src.request.json_body,
                    timeout=src.request.timeout,
                )
                self.requests_made += 1
            except httpx.HTTPError as exc:
                last_error = f"transport error: {exc}"
            else:
                if response.status_code < 400:
                    return response

                last_error = f"HTTP {response.status_code}: {response.text[:300]}"

                # A 401 mid-run usually means an expired token, so retry once fresh.
                if response.status_code == 401 and attempt == 1:
                    self.auth.invalidate()
                elif response.status_code not in retry.retry_on_status:
                    raise ExtractError(f"[{src.name}] {url} -> {last_error}")

                if retry.respect_retry_after:
                    suggested = _retry_after_seconds(response)
                    if suggested is not None:
                        delay = min(suggested, retry.max_backoff_seconds)

            if attempt == retry.max_attempts:
                break

            sleep_for = min(delay, retry.max_backoff_seconds) * (0.5 + random.random())
            log.warning(
                "[%s] attempt %d/%d failed (%s); retrying in %.1fs",
                src.name, attempt, retry.max_attempts, last_error, sleep_for,
            )
            time.sleep(sleep_for)
            delay = min(delay * retry.backoff_multiplier, retry.max_backoff_seconds)

        raise ExtractError(
            f"[{src.name}] {url} failed after {retry.max_attempts} attempts: {last_error}"
        )
