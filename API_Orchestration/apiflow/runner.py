"""Run one source end to end: extract -> transform -> load, with bookkeeping."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from . import db
from .config import SourceConfig
from .extract import Extractor
from .load import Loader, deduplicate
from .transform import coerce, transform_batch

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    source: str
    status: str  # success | failed | skipped
    pages: int = 0
    requests: int = 0
    rows_read: int = 0
    rows_written: int = 0
    rows_rejected: int = 0
    duration_s: float = 0.0
    watermark_in: Any = None
    watermark_out: Any = None
    error: str | None = None
    sample: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        if self.status == "failed":
            return f"{self.source}: FAILED after {self.duration_s:.1f}s — {self.error}"
        if self.status == "skipped":
            return f"{self.source}: skipped ({self.error})"
        extra = f", {self.rows_rejected} rejected" if self.rows_rejected else ""
        return (
            f"{self.source}: {self.rows_written} rows written from {self.rows_read} read "
            f"across {self.pages} page(s) in {self.duration_s:.1f}s{extra}"
        )


def _parse_stored_watermark(value: str | None, source: SourceConfig) -> Any:
    """Cursors are stored as text; convert back to the field's declared type."""
    if value is None:
        return None
    spec = source.fields.get(source.incremental.field or "")
    if spec is None:
        return value
    try:
        return coerce(value, spec.type)
    except Exception:
        return value


def run_source(
    source: SourceConfig,
    *,
    dsn: str | None = None,
    full_refresh: bool = False,
    dry_run: bool = False,
    on_record_error: str = "skip",
    limit_pages: int | None = None,
) -> RunResult:
    """Execute one source. `dry_run=True` performs no database work at all."""
    started = time.monotonic()
    result = RunResult(source=source.name, status="running")

    if not source.enabled:
        result.status = "skipped"
        result.error = "source is disabled"
        return result

    if dry_run:
        return _run_dry(source, result, started, on_record_error, limit_pages)

    with db.connect(dsn) as conn:
        lock_key = f"apiflow:{source.name}"
        if not db.try_advisory_lock(conn, lock_key):
            result.status = "skipped"
            result.error = "another run of this source is already in progress"
            return result
        try:
            db.init_meta(conn)
            db.ensure_target(conn, source)

            watermark_in = None
            if source.incremental.mode == "timestamp" and not full_refresh:
                watermark_in = _parse_stored_watermark(
                    db.get_watermark(conn, source.name), source
                )
            result.watermark_in = watermark_in

            run_id = db.start_run(conn, source.name, watermark_in)
            loader = Loader(conn, source, run_id=run_id)
            all_rejects: list[tuple[Any, str]] = []
            highest = watermark_in

            try:
                if source.target.mode == "replace":
                    loader.truncate()

                with Extractor(source) as extractor:
                    for page in extractor.iter_pages(watermark_in):
                        result.pages += 1
                        result.rows_read += len(page.records)

                        rows, rejects = transform_batch(
                            page.records, source, on_error=on_record_error
                        )
                        all_rejects.extend(rejects)
                        rows, raws = deduplicate(
                            rows, page.records, source.target.primary_key
                        )
                        loader.add_many(rows, raws)
                        highest = _advance_watermark(highest, rows, source)

                        if limit_pages and result.pages >= limit_pages:
                            log.info("[%s] stopping early at page limit", source.name)
                            break
                    result.requests = extractor.requests_made

                result.rows_written = loader.commit()
                result.rows_rejected = len(all_rejects)
                db.record_rejects(conn, run_id, source.name, all_rejects)

                if source.incremental.mode == "timestamp" and highest is not None:
                    db.set_watermark(conn, source.name, _serialize(highest))
                result.watermark_out = highest
                result.status = "success"

            except Exception as exc:
                loader.rollback()
                result.status = "failed"
                result.error = f"{type(exc).__name__}: {exc}"
                log.exception("[%s] run failed", source.name)

            result.duration_s = time.monotonic() - started
            db.finish_run(
                conn, run_id,
                status=result.status,
                pages=result.pages,
                requests=result.requests,
                rows_read=result.rows_read,
                rows_written=result.rows_written,
                rows_rejected=result.rows_rejected,
                watermark_out=None if result.watermark_out is None else _serialize(result.watermark_out),
                error=result.error,
            )
            return result
        finally:
            db.advisory_unlock(conn, lock_key)


def _run_dry(
    source: SourceConfig,
    result: RunResult,
    started: float,
    on_record_error: str,
    limit_pages: int | None,
) -> RunResult:
    """Fetch and transform without touching Postgres — for validating a new config."""
    try:
        with Extractor(source) as extractor:
            for page in extractor.iter_pages(None):
                result.pages += 1
                result.rows_read += len(page.records)
                rows, rejects = transform_batch(
                    page.records, source, on_error=on_record_error
                )
                result.rows_rejected += len(rejects)
                for reason in rejects[:5]:
                    log.warning("[%s] rejected record: %s", source.name, reason[1])
                if len(result.sample) < 3:
                    result.sample.extend(rows[: 3 - len(result.sample)])
                if limit_pages and result.pages >= limit_pages:
                    break
            result.requests = extractor.requests_made
        result.rows_written = 0
        result.status = "success"
    except Exception as exc:
        result.status = "failed"
        result.error = f"{type(exc).__name__}: {exc}"
    result.duration_s = time.monotonic() - started
    return result


def _advance_watermark(current: Any, rows: list[dict[str, Any]], source: SourceConfig) -> Any:
    if source.incremental.mode != "timestamp":
        return current
    column = source.incremental.field
    assert column
    for row in rows:
        value = row.get(column)
        if value is None:
            continue
        if current is None:
            current = value
            continue
        try:
            if value > current:
                current = value
        except TypeError:
            # Mixed types (e.g. naive vs aware datetimes) — keep what we have.
            continue
    return current


def _serialize(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)
