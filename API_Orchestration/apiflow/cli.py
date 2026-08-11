"""Command line interface.

    apiflow validate                     # parse every source config, print the plan
    apiflow list                         # what's configured and on what schedule
    apiflow init-db                      # create bookkeeping + target tables
    apiflow run <source> [--dry-run]     # one source, now
    apiflow run-all                      # every enabled source, sequentially
    apiflow schedule                     # foreground cron loop
    apiflow status [--source X]          # recent run history
    apiflow reset <source>               # clear the incremental watermark
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import db, logging_conf, sqlgen
from .config import ConfigError, SourceConfig, load_all_sources

DEFAULT_SOURCES_DIR = os.environ.get("APIFLOW_SOURCES", "sources")


def _load(args: argparse.Namespace, *, strict_env: bool = True) -> list[SourceConfig]:
    try:
        return load_all_sources(args.sources, strict_env=strict_env)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        raise SystemExit(2)


def _pick(sources: list[SourceConfig], name: str) -> SourceConfig:
    for source in sources:
        if source.name == name:
            return source
    known = ", ".join(s.name for s in sources) or "(none)"
    print(f"unknown source {name!r}. available: {known}", file=sys.stderr)
    raise SystemExit(2)


# ------------------------------------------------------------------- commands


def cmd_validate(args: argparse.Namespace) -> int:
    sources = _load(args, strict_env=False)
    for source in sources:
        state = "enabled" if source.enabled else "disabled"
        print(f"OK  {source.name}  [{state}]")
        print(f"    {source.request.method} {source.request.url}")
        print(f"    pagination: {source.pagination.type}   auth: {source.auth.type}")
        print(
            f"    -> {source.target.schema_}.{source.target.table} "
            f"({source.target.mode}, pk={source.target.primary_key or '-'})"
        )
        if args.show_ddl:
            print("\n" + sqlgen.create_table(source) + "\n")
    print(f"\n{len(sources)} source(s) validated.")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    sources = _load(args, strict_env=False)
    width = max((len(s.name) for s in sources), default=4)
    print(f"{'SOURCE'.ljust(width)}  {'SCHEDULE':<16} {'ENABLED':<8} TARGET")
    for source in sources:
        print(
            f"{source.name.ljust(width)}  {source.schedule or '-':<16} "
            f"{str(source.enabled).lower():<8} {source.target.schema_}.{source.target.table}"
        )
    return 0


def cmd_init_db(args: argparse.Namespace) -> int:
    sources = _load(args)
    with db.connect(args.dsn) as conn:
        db.init_meta(conn)
        for source in sources:
            db.ensure_target(conn, source)
            print(f"ready: {source.target.schema_}.{source.target.table}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .runner import run_source

    sources = _load(args)
    source = _pick(sources, args.source)
    result = run_source(
        source,
        dsn=args.dsn,
        full_refresh=args.full_refresh,
        dry_run=args.dry_run,
        on_record_error="fail" if args.strict else "skip",
        limit_pages=args.max_pages,
    )
    print(result.summary())
    if args.dry_run and result.sample:
        print("\nsample rows:")
        for row in result.sample:
            print("  " + json.dumps(row, default=str))
    return 0 if result.status in ("success", "skipped") else 1


def cmd_run_all(args: argparse.Namespace) -> int:
    from .runner import run_source

    sources = [s for s in _load(args) if s.enabled]
    failures = 0
    for source in sources:
        result = run_source(source, dsn=args.dsn, full_refresh=args.full_refresh)
        print(result.summary())
        if result.status == "failed":
            failures += 1
    print(f"\n{len(sources) - failures}/{len(sources)} source(s) succeeded.")
    return 1 if failures else 0


def cmd_schedule(args: argparse.Namespace) -> int:
    from .scheduler import Scheduler

    sources = _load(args)
    with db.connect(args.dsn) as conn:
        db.init_meta(conn)
        for source in sources:
            if source.enabled:
                db.ensure_target(conn, source)
    Scheduler(sources, dsn=args.dsn, max_workers=args.workers).run_forever()
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    with db.connect(args.dsn) as conn:
        rows = db.recent_runs(conn, limit=args.limit, source=args.source)
    if not rows:
        print("no runs recorded yet")
        return 0
    header = f"{'RUN':>6}  {'SOURCE':<24} {'STATUS':<9} {'ROWS':>7} {'SECS':>6}  STARTED"
    print(header)
    for row in rows:
        secs = (row["duration_ms"] or 0) / 1000
        print(
            f"{row['run_id']:>6}  {row['source']:<24} {row['status']:<9} "
            f"{row['rows_written']:>7} {secs:>6.1f}  "
            f"{row['started_at'].strftime('%Y-%m-%d %H:%M:%S')}"
        )
        if row["error"]:
            print(f"        error: {row['error'][:160]}")
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    sources = _load(args)
    source = _pick(sources, args.source)
    with db.connect(args.dsn) as conn:
        db.clear_watermark(conn, source.name)
    print(f"watermark cleared for {source.name}; the next run is a full pull")
    return 0


# ---------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="apiflow", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", default=DEFAULT_SOURCES_DIR,
                        help="directory of source YAML files (default: sources)")
    parser.add_argument("--dsn", default=None,
                        help="Postgres URL; defaults to $APIFLOW_DATABASE_URL")
    parser.add_argument("--log-level", default=os.environ.get("APIFLOW_LOG_LEVEL", "INFO"))
    parser.add_argument("--log-json", action="store_true", help="emit JSON lines logs")

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="parse and check every source config")
    p.add_argument("--show-ddl", action="store_true", help="print the CREATE TABLE statement")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("list", help="list configured sources")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("init-db", help="create bookkeeping and target tables")
    p.set_defaults(func=cmd_init_db)

    p = sub.add_parser("run", help="run one source now")
    p.add_argument("source")
    p.add_argument("--dry-run", action="store_true", help="fetch and transform, write nothing")
    p.add_argument("--full-refresh", action="store_true", help="ignore the stored watermark")
    p.add_argument("--strict", action="store_true", help="fail the run on any bad record")
    p.add_argument("--max-pages", type=int, default=None, help="stop after N pages")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("run-all", help="run every enabled source once")
    p.add_argument("--full-refresh", action="store_true")
    p.set_defaults(func=cmd_run_all)

    p = sub.add_parser("schedule", help="run the cron scheduler in the foreground")
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(func=cmd_schedule)

    p = sub.add_parser("status", help="show recent run history")
    p.add_argument("--source", default=None)
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("reset", help="clear a source's incremental watermark")
    p.add_argument("source")
    p.set_defaults(func=cmd_reset)

    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv(Path.cwd() / ".env")
    except ImportError:
        pass

    args = build_parser().parse_args(argv)
    logging_conf.configure(args.log_level, args.log_json)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
