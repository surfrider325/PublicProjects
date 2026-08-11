"""Batched writes into the destination table."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from . import sqlgen
from .config import SourceConfig

if TYPE_CHECKING:  # psycopg is only needed at runtime by db.py
    import psycopg

log = logging.getLogger(__name__)


def deduplicate(rows: list[dict[str, Any]], raws: list[Any], primary_key: list[str]):
    """Collapse duplicate primary keys within a batch.

    Postgres refuses an ON CONFLICT upsert that touches the same row twice in one
    statement, and paginated APIs regularly emit the same record on two pages.
    Last occurrence wins.
    """
    if not primary_key:
        return rows, raws
    index: dict[tuple, int] = {}
    out_rows: list[dict[str, Any]] = []
    out_raws: list[Any] = []
    for row, raw in zip(rows, raws):
        key = tuple(row.get(c) for c in primary_key)
        position = index.get(key)
        if position is None:
            index[key] = len(out_rows)
            out_rows.append(row)
            out_raws.append(raw)
        else:
            out_rows[position] = row
            out_raws[position] = raw
    return out_rows, out_raws


class Loader:
    """Accumulates rows and flushes them in batches within a single transaction.

    The whole run is one transaction: if page 7 of 9 fails, nothing is committed
    and the next run starts from the same watermark. That keeps the target table
    free of half-loaded state.
    """

    def __init__(
        self,
        conn: "psycopg.Connection",
        source: SourceConfig,
        run_id: int | None = None,
        extracted_at: datetime | None = None,
    ) -> None:
        self.conn = conn
        self.source = source
        self.run_id = run_id
        self.extracted_at = extracted_at or datetime.now(timezone.utc)
        self.columns = sqlgen.target_columns(source)
        self.statement = sqlgen.insert_statement(source)
        self.buffer: list[tuple] = []
        self.rows_written = 0

    def _to_tuple(self, row: dict[str, Any], raw: Any) -> tuple:
        values: list[Any] = [row.get(c) for c in self.source.fields]
        values.append(self.source.name)       # _source
        values.append(self.extracted_at)      # _extracted_at
        values.append(self.run_id)            # _run_id
        if self.source.target.store_raw:
            values.append(json.dumps(raw, default=str))
        return tuple(values)

    def add(self, row: dict[str, Any], raw: Any) -> None:
        self.buffer.append(self._to_tuple(row, raw))
        if len(self.buffer) >= self.source.target.batch_size:
            self.flush()

    def add_many(self, rows: list[dict[str, Any]], raws: list[Any]) -> None:
        for row, raw in zip(rows, raws):
            self.add(row, raw)

    def flush(self) -> int:
        if not self.buffer:
            return 0
        batch, self.buffer = self.buffer, []
        with self.conn.cursor() as cur:
            cur.executemany(self.statement, batch)
        self.rows_written += len(batch)
        log.debug("[%s] flushed %d rows", self.source.name, len(batch))
        return len(batch)

    def truncate(self) -> None:
        """Used by target.mode=replace, inside the same transaction as the insert."""
        with self.conn.cursor() as cur:
            cur.execute(sqlgen.truncate(self.source))
        log.info("[%s] truncated target table", self.source.name)

    def commit(self) -> int:
        self.flush()
        self.conn.commit()
        return self.rows_written

    def rollback(self) -> None:
        self.buffer.clear()
        self.conn.rollback()
