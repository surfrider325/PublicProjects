"""Postgres connectivity, DDL management, and run bookkeeping."""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row

from . import sqlgen
from .config import SourceConfig

log = logging.getLogger(__name__)

DEFAULT_DSN_ENV = "APIFLOW_DATABASE_URL"


class DatabaseError(Exception):
    """Raised for connection or DDL problems that are not psycopg's own."""


def get_dsn(dsn: str | None = None) -> str:
    resolved = dsn or os.environ.get(DEFAULT_DSN_ENV)
    if not resolved:
        raise DatabaseError(
            f"no database URL: set {DEFAULT_DSN_ENV} or pass --dsn "
            "(e.g. postgresql://apiflow:apiflow@localhost:5432/apiflow)"
        )
    return resolved


@contextmanager
def connect(dsn: str | None = None) -> Iterator[psycopg.Connection]:
    """Open a connection with dict rows and autocommit off."""
    with psycopg.connect(get_dsn(dsn), row_factory=dict_row) as conn:
        yield conn


def init_meta(conn: psycopg.Connection) -> None:
    """Create the apiflow bookkeeping schema. Idempotent."""
    with conn.cursor() as cur:
        cur.execute(sqlgen.META_DDL)
    conn.commit()
    log.info("bookkeeping schema %r ready", sqlgen.META_SCHEMA)


def existing_columns(conn: psycopg.Connection, schema: str, table: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            "select column_name from information_schema.columns "
            "where table_schema = %s and table_name = %s",
            (schema, table),
        )
        return {row["column_name"] for row in cur.fetchall()}


def ensure_target(conn: psycopg.Connection, source: SourceConfig) -> None:
    """Create or migrate the destination table so it matches the config."""
    target = source.target
    with conn.cursor() as cur:
        cur.execute(sqlgen.create_schema(target.schema_))
        cur.execute(sqlgen.create_table(source))
    conn.commit()

    present = existing_columns(conn, target.schema_, target.table)
    statements = sqlgen.add_missing_columns(source, present)
    if statements:
        with conn.cursor() as cur:
            for stmt in statements:
                log.info("[%s] migrating: %s", source.name, stmt)
                cur.execute(stmt)
        conn.commit()


# ------------------------------------------------------------------ concurrency


def try_advisory_lock(conn: psycopg.Connection, key: str) -> bool:
    """Session-level lock so two schedulers never run the same source at once."""
    with conn.cursor() as cur:
        cur.execute("select pg_try_advisory_lock(hashtext(%s)) as ok", (key,))
        row = cur.fetchone()
    return bool(row and row["ok"])


def advisory_unlock(conn: psycopg.Connection, key: str) -> None:
    with conn.cursor() as cur:
        cur.execute("select pg_advisory_unlock(hashtext(%s))", (key,))
    conn.commit()


# ------------------------------------------------------------------ run tracking


def start_run(conn: psycopg.Connection, source: str, watermark_in: Any) -> int:
    with conn.cursor() as cur:
        cur.execute(
            f'insert into "{sqlgen.META_SCHEMA}"."runs" (source, status, watermark_in) '
            "values (%s, 'running', %s) returning run_id",
            (source, None if watermark_in is None else str(watermark_in)),
        )
        row = cur.fetchone()
    conn.commit()
    assert row is not None
    return int(row["run_id"])


def finish_run(
    conn: psycopg.Connection,
    run_id: int,
    *,
    status: str,
    pages: int = 0,
    requests: int = 0,
    rows_read: int = 0,
    rows_written: int = 0,
    rows_rejected: int = 0,
    watermark_out: Any = None,
    error: str | None = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f'update "{sqlgen.META_SCHEMA}"."runs" set '
            "status = %s, finished_at = now(), "
            "duration_ms = (extract(epoch from (now() - started_at)) * 1000)::int, "
            "pages = %s, requests = %s, rows_read = %s, rows_written = %s, "
            "rows_rejected = %s, watermark_out = %s, error = %s "
            "where run_id = %s",
            (
                status, pages, requests, rows_read, rows_written, rows_rejected,
                None if watermark_out is None else str(watermark_out),
                error, run_id,
            ),
        )
    conn.commit()


def record_rejects(
    conn: psycopg.Connection, run_id: int, source: str, rejects: list[tuple[Any, str]]
) -> None:
    if not rejects:
        return
    with conn.cursor() as cur:
        cur.executemany(
            f'insert into "{sqlgen.META_SCHEMA}"."rejects" (run_id, source, reason, record) '
            "values (%s, %s, %s, %s)",
            [
                (run_id, source, reason, json.dumps(record, default=str))
                for record, reason in rejects
            ],
        )
    conn.commit()


def recent_runs(conn: psycopg.Connection, limit: int = 20, source: str | None = None):
    clause = "where source = %s " if source else ""
    params: tuple = (source, limit) if source else (limit,)
    with conn.cursor() as cur:
        cur.execute(
            f'select * from "{sqlgen.META_SCHEMA}"."runs" {clause}'
            "order by started_at desc limit %s",
            params,
        )
        return cur.fetchall()


# ------------------------------------------------------------------ watermarks


def get_watermark(conn: psycopg.Connection, source: str) -> str | None:
    with conn.cursor() as cur:
        cur.execute(
            f'select watermark from "{sqlgen.META_SCHEMA}"."cursors" where source = %s',
            (source,),
        )
        row = cur.fetchone()
    return row["watermark"] if row else None


def set_watermark(conn: psycopg.Connection, source: str, value: Any) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f'insert into "{sqlgen.META_SCHEMA}"."cursors" (source, watermark) '
            "values (%s, %s) on conflict (source) do update set "
            "watermark = excluded.watermark, updated_at = now()",
            (source, str(value)),
        )
    conn.commit()


def clear_watermark(conn: psycopg.Connection, source: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f'delete from "{sqlgen.META_SCHEMA}"."cursors" where source = %s', (source,)
        )
    conn.commit()
