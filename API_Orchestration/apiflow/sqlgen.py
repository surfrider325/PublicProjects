"""Pure SQL string generation — no database driver required, so it's unit-testable.

Every identifier that reaches this module is validated against a strict pattern
and then double-quoted. Values are never interpolated; they are always bound as
query parameters by the caller.
"""

from __future__ import annotations

import re

from .config import PG_TYPES, SourceConfig

_IDENT_RE = re.compile(r"[a-z_][a-z0-9_]*", re.IGNORECASE)

# Bookkeeping columns added to every target table.
META_COLUMNS: dict[str, str] = {
    "_source": "text not null",
    "_extracted_at": "timestamptz not null default now()",
    "_run_id": "bigint",
}
RAW_COLUMN = "_raw"


class SqlError(Exception):
    """Raised when an identifier fails validation."""


def ident(name: str) -> str:
    """Validate and quote a SQL identifier."""
    if not _IDENT_RE.fullmatch(name or ""):
        raise SqlError(f"unsafe SQL identifier: {name!r}")
    return f'"{name}"'


def qualified(schema: str, table: str) -> str:
    return f"{ident(schema)}.{ident(table)}"


def create_schema(schema: str) -> str:
    return f"create schema if not exists {ident(schema)}"


def target_columns(source: SourceConfig) -> list[str]:
    """Data columns in a stable order, then meta columns."""
    cols = list(source.fields.keys())
    cols.extend(META_COLUMNS.keys())
    if source.target.store_raw:
        cols.append(RAW_COLUMN)
    return cols


def create_table(source: SourceConfig) -> str:
    target = source.target
    lines: list[str] = []
    for column, spec in source.fields.items():
        pg_type = PG_TYPES[spec.type]
        null = " not null" if spec.required else ""
        lines.append(f"  {ident(column)} {pg_type}{null}")
    for column, ddl in META_COLUMNS.items():
        lines.append(f"  {ident(column)} {ddl}")
    if target.store_raw:
        lines.append(f"  {ident(RAW_COLUMN)} jsonb")
    if target.primary_key:
        pk = ", ".join(ident(c) for c in target.primary_key)
        lines.append(f"  primary key ({pk})")
    body = ",\n".join(lines)
    return f"create table if not exists {qualified(target.schema_, target.table)} (\n{body}\n)"


def add_missing_columns(source: SourceConfig, existing: set[str]) -> list[str]:
    """ALTER statements to bring an already-created table up to date with the config."""
    statements: list[str] = []
    table = qualified(source.target.schema_, source.target.table)
    for column, spec in source.fields.items():
        if column not in existing:
            statements.append(
                f"alter table {table} add column if not exists "
                f"{ident(column)} {PG_TYPES[spec.type]}"
            )
    for column, ddl in META_COLUMNS.items():
        if column not in existing:
            # Drop NOT NULL when back-filling an existing table.
            statements.append(
                f"alter table {table} add column if not exists {ident(column)} "
                f"{ddl.replace(' not null', '')}"
            )
    if source.target.store_raw and RAW_COLUMN not in existing:
        statements.append(
            f"alter table {table} add column if not exists {ident(RAW_COLUMN)} jsonb"
        )
    return statements


def insert_statement(source: SourceConfig) -> str:
    """INSERT ... ON CONFLICT DO UPDATE (upsert) or a plain INSERT (append/replace)."""
    target = source.target
    columns = target_columns(source)
    col_sql = ", ".join(ident(c) for c in columns)
    placeholders = ", ".join(["%s"] * len(columns))
    stmt = (
        f"insert into {qualified(target.schema_, target.table)} ({col_sql})\n"
        f"values ({placeholders})"
    )
    if target.mode != "upsert":
        return stmt

    pk = set(target.primary_key)
    updatable = [c for c in columns if c not in pk and c != "_extracted_at"]
    assignments = ", ".join(f"{ident(c)} = excluded.{ident(c)}" for c in updatable)
    assignments += ", " + f"{ident('_extracted_at')} = now()"
    conflict = ", ".join(ident(c) for c in target.primary_key)
    return f"{stmt}\non conflict ({conflict}) do update set {assignments}"


def truncate(source: SourceConfig) -> str:
    return f"truncate table {qualified(source.target.schema_, source.target.table)}"


def max_watermark(source: SourceConfig) -> str:
    column = source.incremental.field
    if not column:
        raise SqlError("source has no incremental field")
    return (
        f"select max({ident(column)}) from "
        f"{qualified(source.target.schema_, source.target.table)}"
    )


# ------------------------------------------------------------------ meta schema

META_SCHEMA = "apiflow"

META_DDL = f"""
create schema if not exists {ident(META_SCHEMA)};

create table if not exists {ident(META_SCHEMA)}."runs" (
  run_id        bigserial primary key,
  source        text        not null,
  status        text        not null,
  started_at    timestamptz not null default now(),
  finished_at   timestamptz,
  duration_ms   integer,
  pages         integer     not null default 0,
  requests      integer     not null default 0,
  rows_read     integer     not null default 0,
  rows_written  integer     not null default 0,
  rows_rejected integer     not null default 0,
  watermark_in  text,
  watermark_out text,
  error         text
);

create index if not exists runs_source_started_idx
  on {ident(META_SCHEMA)}."runs" (source, started_at desc);

create table if not exists {ident(META_SCHEMA)}."cursors" (
  source      text primary key,
  watermark   text        not null,
  updated_at  timestamptz not null default now()
);

create table if not exists {ident(META_SCHEMA)}."rejects" (
  reject_id  bigserial primary key,
  run_id     bigint      not null,
  source     text        not null,
  reason     text        not null,
  record     jsonb,
  created_at timestamptz not null default now()
);

create index if not exists rejects_run_idx
  on {ident(META_SCHEMA)}."rejects" (run_id);
"""
